import os
import copy
import torch
import torch.nn as nn
from torch.nn import functional as F

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

def loss_fn(args, model, features, mode='train'):
    photo_features_norm, sk_feature_norm, neg_feature_norm, photo_aug_features, sk_aug_features, \
            label, pos_logits, sk_logits, photo_feature, sk_feature, jig_logits_r, jig_logits_pos, jig_logits_neg, perm_idx = features

    label = label.to(pos_logits.device)
    
    loss_ce_photo = F.cross_entropy(pos_logits, label)
    loss_ce_sk = F.cross_entropy(sk_logits, label)
    loss_cls = loss_ce_photo + loss_ce_sk
    
    loss_distill_photo = cross_loss(photo_feature, photo_aug_features, args)
    loss_distill_sk = cross_loss(sk_feature, sk_aug_features, args)
    loss_distill = loss_distill_photo + loss_distill_sk 
    
    distance_fn = lambda x, y: 1.0 - F.cosine_similarity(x, y)
    # triplet = nn.TripletMarginWithDistanceLoss(distance_function=distance_fn, margin=0.3)
    triplet = nn.TripletMarginLoss(margin=0.3)
    loss_triplet = triplet(sk_feature_norm, photo_features_norm, neg_feature_norm)
    
    loss_cjs = conditional_cross_modal_jigsaw_loss(
        jig_logits_r=jig_logits_r,
        jig_logits_pos=jig_logits_pos,
        jig_logits_neg=jig_logits_neg,
        perm_idx=perm_idx,
    )
    
    alpha = float(getattr(args, "alpha", 1.0))
    beta = float(getattr(args, "beta", 0.1))

    return 10 * loss_triplet + alpha * loss_cls + 2 * loss_distill + beta * loss_cjs
