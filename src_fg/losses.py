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

def _multi_positive_nce_from_logits(
    logits: torch.Tensor,
    pos_mask: torch.Tensor,
    *,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Multi-positive InfoNCE / supervised contrastive over a precomputed logits matrix.

    logits: [B, B]
    pos_mask: [B, B] boolean, True where j is a positive for anchor i.
    """
    if logits.dim() != 2 or logits.size(0) != logits.size(1):
        raise ValueError("logits must be square [B,B]")
    if pos_mask.shape != logits.shape:
        raise ValueError("pos_mask must have the same shape as logits")

    log_probs = F.log_softmax(logits, dim=1)
    pos = pos_mask.to(dtype=log_probs.dtype)
    pos_counts = pos.sum(dim=1).clamp_min(1.0)

    loss = -(log_probs * pos).sum(dim=1) / pos_counts
    loss = torch.nan_to_num(loss, nan=0.0, posinf=0.0, neginf=0.0)
    return loss.mean()

def instance_xmodal_infonce_posid(
    sk_feat: torch.Tensor,
    ph_feat: torch.Tensor,
    pos_id: torch.Tensor,
    *,
    temperature: float = 0.07,
) -> torch.Tensor:
    """
    Instance-level cross-modal contrastive loss using photo instance ids (pos_id).

    Positives for sketch i are all photos j in the batch that share the same pos_id.
    This supports multi-positive cases (multiple sketches per photo instance).

    sk_feat: [B, D] normalized
    ph_feat: [B, D] normalized (positive photo per sketch)
    pos_id: [B] long, photo instance id for each pair
    """
    if temperature <= 0:
        raise ValueError("temperature must be > 0")
    if sk_feat.dim() != 2 or ph_feat.dim() != 2:
        raise ValueError("sk_feat and ph_feat must be [B,D]")
    if sk_feat.size(0) != ph_feat.size(0):
        raise ValueError("sk_feat and ph_feat must have the same batch size")

    B = sk_feat.size(0)
    pos_id = pos_id.view(-1).to(device=sk_feat.device, dtype=torch.long)
    if pos_id.numel() != B:
        raise ValueError("pos_id must have shape [B]")

    logits = (sk_feat @ ph_feat.t()) / float(temperature)  # [B,B]

    valid = pos_id.ne(-1)
    pos_mask = pos_id.view(-1, 1).eq(pos_id.view(1, -1)) & valid.view(-1, 1) & valid.view(1, -1)
    pos_mask.fill_diagonal_(True)

    loss_s2p = _multi_positive_nce_from_logits(logits, pos_mask)
    loss_p2s = _multi_positive_nce_from_logits(logits.t(), pos_mask.t())
    return 0.5 * (loss_s2p + loss_p2s)

def patch_maxsim_alignment_score(
    a_patch: torch.Tensor,
    b_patch: torch.Tensor,
) -> torch.Tensor:
    """
    Patch-level alignment score via max-sim (directed chamfer-style).

    a_patch: [B, N, D] normalized
    b_patch: [B, N, D] normalized

    Returns:
      score: [B] where higher means more aligned.
    """
    if a_patch.dim() != 3 or b_patch.dim() != 3:
        raise ValueError("a_patch and b_patch must be [B,N,D]")
    if a_patch.shape != b_patch.shape:
        raise ValueError("a_patch and b_patch must have the same shape")

    sim = torch.bmm(a_patch, b_patch.transpose(1, 2))  # [B,N,N]
    return sim.max(dim=2).values.mean(dim=1)  # [B]

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
    queue_feat: torch.Tensor | None = None,
    queue_label: torch.Tensor | None = None,
    queue_pos_id: torch.Tensor | None = None,
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

    sim_batch = torch.where(has_candidate, sim_hard, torch.full_like(sim_hard, float("-inf")))

    # Optional: also mine from a cross-batch queue (within the same category).
    if queue_feat is not None and queue_label is not None and queue_pos_id is not None and queue_feat.numel() > 0:
        q_feat = queue_feat.to(device=device)
        q_lab = queue_label.to(device=device, dtype=torch.long).view(-1)
        q_pid = queue_pos_id.to(device=device, dtype=torch.long).view(-1)

        sim_q = anchor_feat.float() @ q_feat.float().t()  # [B,Q]
        mask_q = label.view(-1, 1).eq(q_lab.view(1, -1)) & q_pid.view(1, -1).ne(-1)
        if pos_id is not None:
            mask_q = mask_q & pos_id.view(-1, 1).ne(q_pid.view(1, -1))

        masked_sim_q = sim_q.masked_fill(~mask_q, float("-inf"))
        sim_queue, hard_q_idx = masked_sim_q.max(dim=1)
        has_q = mask_q.any(dim=1)
        sim_queue = torch.where(has_q, sim_queue, torch.full_like(sim_queue, float("-inf")))

        hard_q_feat = q_feat[hard_q_idx]
        hard_q_label = q_lab[hard_q_idx]
    else:
        sim_queue = torch.full_like(sim_rand, float("-inf"))
        hard_q_feat = rand_neg_feat
        hard_q_label = label

    sims = torch.stack([sim_rand, sim_batch, sim_queue], dim=1)  # [B,3]
    best = sims.argmax(dim=1)  # 0=rand, 1=batch, 2=queue

    neg_feat = rand_neg_feat.clone()
    neg_label = label.clone()

    use_batch = best.eq(1)
    neg_feat[use_batch] = hard_neg_feat[use_batch]
    neg_label[use_batch] = hard_neg_label[use_batch]

    use_queue = best.eq(2)
    neg_feat[use_queue] = hard_q_feat[use_queue]
    neg_label[use_queue] = hard_q_label[use_queue]

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
            label, pos_logits, sk_logits, photo_feature, sk_feature, jig_logits_r, jig_logits_pos, jig_logits_neg, perm_idx, text_features, pos_id, \
            photo_patch_features_norm, sk_patch_features_norm, neg_patch_features_norm = features

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

    queue_feat = queue_label = queue_pos_id = None
    if mode == "train" and model is not None and getattr(model, "queue_size", 0) > 0 and hasattr(model, "queue_len"):
        q_len = int(model.queue_len.detach().cpu().item())
        if q_len > 0:
            if q_len < int(model.queue_size):
                queue_feat = model.queue_feat[:q_len]
                queue_label = model.queue_label[:q_len]
                queue_pos_id = model.queue_pos_id[:q_len]
            else:
                queue_feat = model.queue_feat
                queue_label = model.queue_label
                queue_pos_id = model.queue_pos_id

    mined_neg_feat, mined_neg_label = _select_batch_hard_negative(
        anchor_feat=sk_feature_norm,
        pos_feat=photo_features_norm,
        rand_neg_feat=neg_feature_norm,
        label=label,
        pos_id=pos_id,
        queue_feat=queue_feat,
        queue_label=queue_label,
        queue_pos_id=queue_pos_id,
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
    inst_w = float(getattr(args, "gamma", 0.0))
    lambd = float(getattr(args, "lambd", 0.0))

    loss_inst = pos_logits.new_tensor(0.0)
    if inst_w > 0 and pos_id is not None:
        loss_inst = instance_xmodal_infonce_posid(
            sk_feat=sk_feature_norm.float(),
            ph_feat=photo_features_norm.float(),
            pos_id=pos_id,
            temperature=float(getattr(args, "temperature", 0.07)),
        )

    patch_w = float(getattr(args, "patch_weight", 0.0))
    patch_temp = float(getattr(args, "patch_temperature", getattr(args, "temperature", 0.07)))
    loss_patch = pos_logits.new_tensor(0.0)
    if patch_w > 0 and sk_patch_features_norm is not None and photo_patch_features_norm is not None and neg_patch_features_norm is not None:
        score_pos = patch_maxsim_alignment_score(
            sk_patch_features_norm.float(),
            photo_patch_features_norm.float(),
        )
        score_neg = patch_maxsim_alignment_score(
            sk_patch_features_norm.float(),
            neg_patch_features_norm.float(),
        )
        logits_patch = torch.stack([score_pos, score_neg], dim=1) / max(patch_temp, 1e-6)
        labels_patch = torch.zeros(logits_patch.size(0), device=logits_patch.device, dtype=torch.long)
        loss_patch = F.cross_entropy(logits_patch, labels_patch)

    if mode == "train" and model is not None and hasattr(model, "enqueue_photo_features"):
        model.enqueue_photo_features(photo_features_norm, label, pos_id)

    return (
        10 * loss_triplet
        + alpha * loss_cls
        + 2 * loss_distill
        + beta * loss_cjs
        + lambd * loss_fdiv
        + inst_w * loss_inst
        + patch_w * loss_patch
    )
