import os
import copy
import torch
import torch.nn as nn
from torch.nn import functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def cross_loss(feature_1, feature_2, args):
    labels = torch.cat([torch.arange(len(feature_1)) for _ in range(2)], dim=0)
    labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
    labels = labels.to(device)

    feature_1 = F.normalize(feature_1, dim=1)
    feature_2 = F.normalize(feature_2, dim=1)
    features = torch.cat((feature_1, feature_2), dim=0)  # (2*B, Feat_dim)

    similarity_matrix = torch.matmul(features, features.T)  # (2*B, 2*B)

    # discard the main diagonal from both: labels and similarities matrix
    mask = torch.eye(labels.shape[0], dtype=torch.bool).to(device)
    labels = labels[~mask].view(labels.shape[0], -1)
    similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)  # (2*B, 2*B - 1)

    # select and combine multiple positives
    positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)  # (2*B, 1)
    negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)  # (2*B, 2*(B - 1))

    logits = torch.cat([positives, negatives], dim=1)
    labels = torch.zeros(logits.shape[0], dtype=torch.long).to(device)

    logits = logits / args.temperature

    return nn.CrossEntropyLoss()(logits, labels)

def conditional_cross_modal_jigsaw_loss(logits_r, logits_pos, logits_neg, perm_idx, margin=0.0):
    perm_idx = perm_idx.to(logits_r.device)
    loss_ce = F.cross_entropy(logits_r, perm_idx)
    loss_pos = F.cross_entropy(logits_pos, perm_idx)
    loss_neg = F.cross_entropy(logits_neg, perm_idx)
    loss_margin = F.relu(margin + loss_pos - loss_neg)
    return loss_ce + loss_margin

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
            label, pos_logits, sk_logits, photo_feature, sk_feature, cjs_logits_r, cjs_logits_pos, cjs_logits_neg, perm_idx = features

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

    cjs_margin = getattr(args, "cjs_margin", 0.0)
    cjs_loss = conditional_cross_modal_jigsaw_loss(
        logits_r=cjs_logits_r,
        logits_pos=cjs_logits_pos,
        logits_neg=cjs_logits_neg,
        perm_idx=perm_idx,
        margin=cjs_margin,
    )

    return 10*loss_triplet + loss_cls + 2*loss_distill + 1*cjs_loss