import os
import copy
import torch
import torch.nn as nn
from torch.nn import functional as F

def f_divergence_uniform_relative_distance_loss(
    delta: torch.Tensor,
    label: torch.Tensor,
    *,
    eps: float = 1e-8,
    min_per_class: int = 2,
) -> torch.Tensor:
    """
    "CLIP for All Things" f-divergence regularizer (Eq. 4) for multi-category FG-ZS-SBIR.

    Relative distance for a triplet:
      δ(s, p+, p-) = d(s, p+) - d(s, p-)

    For each category c present in the batch, form a discrete distribution:
      D_c = softmax({δ_i}_{i=1..N_s})

    Then minimize the average KL divergence over all ordered category pairs:
      L_δ = avg_{c!=c'} KL(D_c || D_c')

    Notes:
    - This is computed within a mini-batch; it is most stable with PK sampling (same N_s per class).
    - If per-class counts differ (e.g., due to filtering None samples), we truncate each class to the
      minimum count among the classes used.
    """
    if delta.numel() == 0:
        return delta.new_tensor(0.0)

    delta = delta.view(-1)
    label = label.view(-1).to(device=delta.device, dtype=torch.long)
    if delta.numel() != label.numel():
        raise ValueError("delta and label must have the same number of elements")

    uniq, counts = torch.unique(label, return_counts=True)
    keep = counts.ge(int(min_per_class))
    uniq = uniq[keep]
    counts = counts[keep]

    if uniq.numel() < 2:
        return delta.new_tensor(0.0)

    m = int(counts.min().item())
    if m < int(min_per_class):
        return delta.new_tensor(0.0)

    # Gather per-category delta vectors of equal length m.
    per_cat = []
    for lab in uniq:
        idx = torch.nonzero(label.eq(lab), as_tuple=False).squeeze(1)
        if idx.numel() < m:
            continue
        per_cat.append(delta[idx[:m]])

    if len(per_cat) < 2:
        return delta.new_tensor(0.0)

    delta_mat = torch.stack(per_cat, dim=0)  # [C, m]
    D = F.softmax(delta_mat, dim=1)
    logD = torch.log(D.clamp_min(eps))

    # KL matrix over ordered pairs: KL(i||j) = sum_k D_i,k (logD_i,k - logD_j,k)
    kl = (D.unsqueeze(1) * (logD.unsqueeze(1) - logD.unsqueeze(0))).sum(dim=2)  # [C, C]
    C = kl.size(0)
    kl = kl - torch.diag_embed(torch.diagonal(kl))  # zero diagonal
    return kl.sum() / (C * (C - 1))

def cross_loss(feature_1, feature_2, args):
    feat_device = feature_1.device
    labels = torch.cat([torch.arange(len(feature_1), device=feat_device) for _ in range(2)], dim=0)
    labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()

    feature_1 = F.normalize(feature_1, dim=1)
    feature_2 = F.normalize(feature_2, dim=1)
    features = torch.cat((feature_1, feature_2), dim=0)  # (2*B, Feat_dim)

    similarity_matrix = torch.matmul(features, features.T)  # (2*B, 2*B)

    # discard the main diagonal from both: labels and similarities matrix
    mask = torch.eye(labels.shape[0], dtype=torch.bool, device=feat_device)
    labels = labels[~mask].view(labels.shape[0], -1)
    similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)  # (2*B, 2*B - 1)

    # select and combine multiple positives
    positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)  # (2*B, 1)
    negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)  # (2*B, 2*(B - 1))

    logits = torch.cat([positives, negatives], dim=1)
    labels = torch.zeros(logits.shape[0], dtype=torch.long, device=feat_device)

    logits = logits / args.temperature

    return nn.CrossEntropyLoss()(logits, labels)

def conditional_cross_modal_jigsaw_loss(
    jig_logits_r: torch.Tensor,
    jig_logits_pos: torch.Tensor,
    jig_logits_neg: torch.Tensor,
    perm_idx: torch.Tensor,
) -> torch.Tensor:
    """
    SpLIP (ECCV 2024) conditional cross-modal jigsaw loss (Eq. 9-10):
      Lcjs = Lce(Fjs(r), yperm) + [ Lce(Fjs(r+), yperm) - Lce(Fjs(r-), yperm) ]_+
    """
    perm_idx = perm_idx.to(device=jig_logits_r.device, dtype=torch.long)

    ce_r = F.cross_entropy(jig_logits_r, perm_idx)
    ce_pos = F.cross_entropy(jig_logits_pos, perm_idx, reduction="none")
    ce_neg = F.cross_entropy(jig_logits_neg, perm_idx, reduction="none")
    margin = F.relu(ce_pos - ce_neg).mean()

    return ce_r + margin

def xmodal_infonce(sk, im, temperature=0.07):
    """
    sk_feat: (B, D) normalized
    img_feat: (B, D) normalized
    positives are diagonal pairs (i,i)
    negatives are off-diagonal (i,j), including same-class instances
    """
    logits = (sk @ im.t()) / temperature          # (B,B)
    labels = torch.arange(logits.size(0), device=logits.device)

    loss_s2i = F.cross_entropy(logits, labels)
    loss_i2s = F.cross_entropy(logits.t(), labels)
    return 0.5*(loss_s2i + loss_i2s)

def _select_batch_hard_negative(
    *,
    anchor_feat: torch.Tensor,
    pos_feat: torch.Tensor,
    rand_neg_feat: torch.Tensor,
    label: torch.Tensor,
    pos_id: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Hard-negative / hard-triplet mining (batch-hard within the same category).

    We treat other positive photos in the batch as candidate negatives, but only
    within the same category and (optionally) excluding the same photo instance.

    Returns:
      neg_feat: (B, D)
      neg_label: (B,)
    """
    B = anchor_feat.size(0)
    device = anchor_feat.device

    label = label.to(device=device, dtype=torch.long)
    if pos_id is not None:
        pos_id = pos_id.to(device=device, dtype=torch.long)

    # cosine similarity since features are normalized
    sim = anchor_feat @ pos_feat.t()  # (B,B)
    eye = torch.eye(B, dtype=torch.bool, device=device)

    same_cat = label.view(-1, 1).eq(label.view(1, -1))
    mask = same_cat & ~eye
    if pos_id is not None:
        mask = mask & pos_id.view(-1, 1).ne(pos_id.view(1, -1))

    masked_sim = sim.masked_fill(~mask, float("-inf"))
    hard_j = masked_sim.argmax(dim=1)
    has_candidate = mask.any(dim=1)

    hard_neg_feat = pos_feat[hard_j]
    hard_neg_label = label[hard_j]

    # Compare against the randomly sampled within-category negative from dataset
    sim_rand = (anchor_feat * rand_neg_feat).sum(dim=1)
    sim_hard = (anchor_feat * hard_neg_feat).sum(dim=1)

    use_hard = has_candidate & (sim_hard > sim_rand)
    neg_feat = torch.where(use_hard.view(-1, 1), hard_neg_feat, rand_neg_feat)
    neg_label = torch.where(use_hard, hard_neg_label, label)

    return neg_feat, neg_label

def adaptive_margin_triplet_loss(
    anchor: torch.Tensor,
    positive: torch.Tensor,
    negative: torch.Tensor,
    margin: torch.Tensor,
) -> torch.Tensor:
    """
    SpLIP adaptive-margin triplet (Eq. in Sec 3.5):
      [ ||a - p||^2 - ||a - n||^2 + margin ]_+
    """
    d_pos = (anchor - positive).pow(2).sum(dim=1)
    d_neg = (anchor - negative).pow(2).sum(dim=1)
    return F.relu(d_pos - d_neg + margin).mean()

def loss_fn(args, model, features, mode='train'):
    photo_features_norm, sk_feature_norm, neg_feature_norm, photo_aug_features, sk_aug_features, \
            label, pos_logits, sk_logits, photo_feature, sk_feature, jig_logits_r, jig_logits_pos, jig_logits_neg, perm_idx, text_features, pos_id = features

    label = label.to(pos_logits.device)
    if torch.is_tensor(pos_id):
        pos_id = pos_id.to(pos_logits.device)
    else:
        pos_id = None
    
    loss_ce_photo = F.cross_entropy(pos_logits, label)
    loss_ce_sk = F.cross_entropy(sk_logits, label)
    loss_cls = loss_ce_photo + loss_ce_sk
    
    loss_distill_photo = cross_loss(photo_feature, photo_aug_features, args)
    loss_distill_sk = cross_loss(sk_feature, sk_aug_features, args)
    loss_distill = loss_distill_photo + loss_distill_sk 
    
    mined_neg_feat, mined_neg_label = _select_batch_hard_negative(
        anchor_feat=sk_feature_norm,
        pos_feat=photo_features_norm,
        rand_neg_feat=neg_feature_norm,
        label=label,
        pos_id=pos_id,
    )

    # f-divergence loss (CLIP for All Things): stabilize relative distances across categories.
    # Use squared L2 on normalized features, consistent with our triplet implementation.
    d_pos = (sk_feature_norm.float() - photo_features_norm.float()).pow(2).sum(dim=1)
    d_neg = (sk_feature_norm.float() - mined_neg_feat.float()).pow(2).sum(dim=1)
    delta = d_pos - d_neg
    loss_fdiv = f_divergence_uniform_relative_distance_loss(delta, label)

    # Adaptive margin (SpLIP): mu(c+, c-) = cos(Ft(c+), Ft(c-))
    text_features = text_features.to(device=pos_logits.device)
    text_features = F.normalize(text_features.float(), dim=1)
    mu = (text_features[label] * text_features[mined_neg_label]).sum(dim=1)

    loss_triplet = adaptive_margin_triplet_loss(
        sk_feature_norm.float(),
        photo_features_norm.float(),
        mined_neg_feat.float(),
        mu,
    )
    
    loss_cjs = conditional_cross_modal_jigsaw_loss(
        jig_logits_r=jig_logits_r,
        jig_logits_pos=jig_logits_pos,
        jig_logits_neg=jig_logits_neg,
        perm_idx=perm_idx,
    )
    
    alpha = float(getattr(args, "alpha", 1.0))
    beta = float(getattr(args, "beta", 0.1))
    lambd = float(getattr(args, "lambd", 0.0))

    return 10 * loss_triplet + alpha * loss_cls + 2 * loss_distill + beta * loss_cjs + lambd * loss_fdiv
