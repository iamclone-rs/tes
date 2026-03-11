import numpy as np
import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.nn import functional as F

from src.coprompt import MultiModalPromptLearner, Adapter, TextEncoder
from src.utils import load_clip_to_cpu, get_all_categories, is_fg_dataset
from src.losses import loss_fn

def freeze_model(m):
    m.requires_grad_(False)
    
def freeze_all_but_bn(m):
    if not isinstance(m, torch.nn.LayerNorm):
        if hasattr(m, 'weight') and m.weight is not None:
            m.weight.requires_grad_(False)
        if hasattr(m, 'bias') and m.bias is not None:
            m.bias.requires_grad_(False)


class JigsawSolver(nn.Module):
    def __init__(
        self,
        *,
        embed_dim,
        num_classes,
        num_layers=2,
        nhead=8,
        dropout=0.1,
    ):
        super().__init__()
        if embed_dim % nhead != 0:
            raise ValueError("embed_dim must be divisible by nhead")

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=nhead,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.pos_embed = nn.Parameter(torch.zeros(1, 2, embed_dim))
        self.classifier = nn.Linear(embed_dim, num_classes)

        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.classifier.weight, std=0.02)
        nn.init.zeros_(self.classifier.bias)

    def forward(self, tokens):
        if tokens.dim() != 3 or tokens.size(1) != 2:
            raise ValueError("Expected tokens with shape [B, 2, D]")
        encoded = self.encoder(tokens + self.pos_embed.to(dtype=tokens.dtype, device=tokens.device))
        pooled = encoded.mean(dim=1)
        return self.classifier(pooled)
            
class CustomCLIP(nn.Module):
    def __init__(
        self, cfg, clip_model, clip_model_distill
    ):
        super().__init__()
        self.cfg = cfg
        clip_model.apply(freeze_all_but_bn)
        clip_model_distill.apply(freeze_all_but_bn)
        self.dtype = clip_model.dtype
        self.prompt_learner_photo = MultiModalPromptLearner(cfg, clip_model, type='photo')
        self.prompt_learner_sketch = MultiModalPromptLearner(cfg, clip_model, type='sketch')
        
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        
        self.adapter_photo = Adapter(512, 4).to(clip_model.dtype)
        self.adapter_text = Adapter(512, 4).to(clip_model.dtype)
        
        self.model_distill = clip_model_distill
        self.image_adapter_m = 0.1
        self.text_adapter_m = 0.1
        self.use_cjs = bool(getattr(cfg, "use_cjs", True))
        if self.use_cjs:
            self.jigsaw_solver = JigsawSolver(
                embed_dim=512,
                num_classes=int(getattr(cfg, "jigsaw_num_perm", 100)),
                num_layers=int(getattr(cfg, "jigsaw_layers", 2)),
                nhead=int(getattr(cfg, "jigsaw_nhead", 8)),
                dropout=float(getattr(cfg, "jigsaw_dropout", 0.1)),
            )
    
    def get_logits(self, img_tensor, classnames, type='photo'):
        if type=='photo':
            prompt_learner = self.prompt_learner_photo
        else:
            prompt_learner = self.prompt_learner_sketch
        # tokenized_prompts = prompt_learner.tokenized_prompts
        logit_scale = self.logit_scale.exp()
        (
            tokenized_prompts,
            prompts,
            shared_ctx,
            deep_compound_prompts_text,
            deep_compound_prompts_vision,
        ) = prompt_learner(classnames)
        
        text_features = self.text_encoder(
            prompts, tokenized_prompts, deep_compound_prompts_text
        ) # (n_classes, 512)
        
        image_features = self.image_encoder(
                img_tensor.type(self.dtype), shared_ctx, deep_compound_prompts_vision
            ) # (batch_size, 768)
        
        if type=='photo' and self.cfg.use_adapt_ph:
            x_a = self.adapter_photo(image_features)
            image_features = (
                self.image_adapter_m * x_a + (1 - self.image_adapter_m) * image_features
            )
        
        if type=='sketch' and self.cfg.use_adapt_sk:
            x_a = self.adapter_photo(image_features)
            image_features = (
                self.image_adapter_m * x_a + (1 - self.image_adapter_m) * image_features
            ) 

        if self.cfg.use_adapt_txt:
            x_b = self.adapter_text(text_features)
            text_features = (
                self.text_adapter_m * x_b + (1 - self.text_adapter_m) * text_features
            )

        image_features_normalize = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        # image_features = F.normalize(image_features, dim=-1)
        # text_features = F.normalize(text_features, dim=-1)

        logits = logit_scale * image_features_normalize @ text_features.t()
        
        return logits, image_features_normalize, image_features
        
    def forward(self, x, classnames):
        if self.use_cjs:
            photo_tensor, sk_tensor, photo_aug_tensor, sk_aug_tensor, neg_tensor, \
                sk_perm_tensor, perm_idx, label, pos_id = x
        else:
            photo_tensor, sk_tensor, photo_aug_tensor, sk_aug_tensor, neg_tensor, label, pos_id = x
            sk_perm_tensor = None
            perm_idx = None

        pos_logits, photo_features_norm, photo_feature = self.get_logits(photo_tensor, classnames)
        sk_logits, sk_feature_norm, sk_feature = self.get_logits(sk_tensor, classnames, type='sketch')
        _, neg_feature_norm, _ = self.get_logits(neg_tensor, classnames)
        
        if self.cfg.use_co_ph:
            photo_aug_features = self.model_distill.encode_image(photo_aug_tensor)
        else:
            photo_aug_features = None
            
        if self.cfg.use_co_sk:
            sk_aug_features = self.model_distill.encode_image(sk_aug_tensor)
        else:
            sk_aug_features = None

        jig_logits_r, jig_logits_pos, jig_logits_neg = None, None, None
        if self.use_cjs:
            _, sk_perm_feature_norm, _ = self.get_logits(sk_perm_tensor, classnames, type='sketch')
            r = torch.stack([sk_feature_norm, sk_perm_feature_norm], dim=1)
            r_pos = torch.stack([photo_features_norm, sk_perm_feature_norm], dim=1)
            r_neg = torch.stack([neg_feature_norm, sk_perm_feature_norm], dim=1)
            jigsaw_logits = self.jigsaw_solver(torch.cat([r, r_pos, r_neg], dim=0).float())
            jig_logits_r, jig_logits_pos, jig_logits_neg = jigsaw_logits.chunk(3, dim=0)

        return photo_features_norm, sk_feature_norm, photo_aug_features, sk_aug_features, \
            neg_feature_norm, label, pos_logits, sk_logits, photo_feature, sk_feature, pos_id, \
            jig_logits_r, jig_logits_pos, jig_logits_neg, perm_idx
        
            
    def extract_feature(self, image, classname, type='photo'):
        _, feature, _ = self.get_logits(image, classnames=classname, type=type)
        return feature
            
class ZS_SBIR(pl.LightningModule):
    def __init__(self, args, classname):
        super(ZS_SBIR, self).__init__()
        self.args = args
        self.classname = classname
        self.is_fg = is_fg_dataset(args)
        clip_model = load_clip_to_cpu(args)
        
        design_details = {
            "trainer": "CoOp",
            "vision_depth": 0,
            "language_depth": 0,
            "vision_ctx": 0,
            "language_ctx": 0,
        }
        clip_model_distill = load_clip_to_cpu(args, design_details=design_details)
        
        self.best_metric = 1e-3
        
        self.model = CustomCLIP(cfg=args, clip_model=clip_model, clip_model_distill=clip_model_distill)
    
        self.val_step_outputs_sk = []
        self.val_step_outputs_ph = []

    def _epoch_prefix(self):
        return f"Epoch {self.current_epoch + 1}"

    def _log_epoch_metrics(self, acc1, acc5):
        self.log("acc1", acc1, on_step=False, on_epoch=True)
        self.log("acc5", acc5, on_step=False, on_epoch=True)
        if self.global_step > 0:
            self.best_metric = self.best_metric if (self.best_metric > acc1) else acc1

        train_loss = self.trainer.callback_metrics.get("train_loss", None)
        train_loss_str = f"{train_loss.item():.6f}" if train_loss is not None else "n/a"
        print(
            f"{self._epoch_prefix()} | train_loss={train_loss_str} | "
            f"acc@1={acc1:.6f} | acc@5={acc5:.6f} | best_acc@1={self.best_metric:.6f}"
        )
        
        
    def configure_optimizers(self):
        optimizer = torch.optim.SGD(params=self.model.parameters(), lr=self.args.lr, weight_decay=1e-3, momentum=0.9)
        
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer=optimizer,
            step_size=5,
            gamma=0.1
        )
        
        return [optimizer] , [scheduler]
    
    def forward(self, data, classname):
        return self.model(data, classname)
    
    def training_step(self, batch, batch_idx):
        classname = get_all_categories(self.args)
        features = self.forward(batch, classname)
        
        loss = loss_fn(self.args, self.model, features=features, mode='train')
        self.log('train_loss', loss, on_step=False, on_epoch=True)
        return loss
    
    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        classnames = get_all_categories(self.args, mode="test")
        if self.is_fg:
            image_tensor, label, instance_id = batch
        else:
            image_tensor, label = batch

        if dataloader_idx == 0:
            feat = self.model.extract_feature(image_tensor, classname=classnames, type='sketch')
            if self.is_fg:
                self.val_step_outputs_sk.append((feat.detach().cpu(), label.detach().cpu(), list(instance_id)))
            else:
                self.val_step_outputs_sk.append((feat, label))
        else:
            feat = self.model.extract_feature(image_tensor, classname=classnames, type='photo')
            if self.is_fg:
                self.val_step_outputs_ph.append((feat.detach().cpu(), label.detach().cpu(), list(instance_id)))
            else:
                self.val_step_outputs_ph.append((feat, label))
    
    def on_validation_epoch_end(self):
        if self.is_fg:
            self._on_validation_epoch_end_fg()
            return

        query_len = len(self.val_step_outputs_sk)
        gallery_len = len(self.val_step_outputs_ph)
        
        query_feat_all = torch.cat([self.val_step_outputs_sk[i][0] for i in range(query_len)])
        gallery_feat_all = torch.cat([self.val_step_outputs_ph[i][0] for i in range(gallery_len)])
        
        all_sketch_category = np.array(sum([list(self.val_step_outputs_sk[i][1].detach().cpu().numpy()) for i in range(query_len)], []))
        all_photo_category = np.array(sum([list(self.val_step_outputs_ph[i][1].detach().cpu().numpy()) for i in range(gallery_len)], []))

        acc1_total, acc5_total, total_sk = 0, 0, 0
        for idx, sk_feat in enumerate(query_feat_all):
            category = all_sketch_category[idx]
            similarity = F.cosine_similarity(sk_feat.unsqueeze(0), gallery_feat_all)
            ranking = torch.argsort(similarity, descending=True)
            target = torch.from_numpy(all_photo_category == category)

            top1_idx = ranking[:1].cpu()
            top5_idx = ranking[: min(5, ranking.numel())].cpu()
            acc1_total += int(target[top1_idx].any().item())
            acc5_total += int(target[top5_idx].any().item())
            total_sk += 1

        acc1 = acc1_total / total_sk if total_sk else 0.0
        acc5 = acc5_total / total_sk if total_sk else 0.0
        self._log_epoch_metrics(acc1, acc5)
        self.val_step_outputs_sk.clear()
        self.val_step_outputs_ph.clear()

    def _on_validation_epoch_end_fg(self):
        query_len = len(self.val_step_outputs_sk)
        gallery_len = len(self.val_step_outputs_ph)
        if query_len == 0 or gallery_len == 0:
            self.val_step_outputs_sk.clear()
            self.val_step_outputs_ph.clear()
            return

        query_feat_all = torch.cat([self.val_step_outputs_sk[i][0] for i in range(query_len)])
        gallery_feat_all = torch.cat([self.val_step_outputs_ph[i][0] for i in range(gallery_len)])

        all_sketch_category = np.concatenate([
            self.val_step_outputs_sk[i][1].numpy() for i in range(query_len)
        ])
        all_photo_category = np.concatenate([
            self.val_step_outputs_ph[i][1].numpy() for i in range(gallery_len)
        ])
        all_sketch_instance = sum([self.val_step_outputs_sk[i][2] for i in range(query_len)], [])
        all_photo_instance = sum([self.val_step_outputs_ph[i][2] for i in range(gallery_len)], [])

        acc1_total, acc5_total, total_sk = 0, 0, 0
        for idx, sk_feat in enumerate(query_feat_all):
            category = all_sketch_category[idx]
            gallery_indices = np.where(all_photo_category == category)[0]
            if gallery_indices.size == 0:
                continue

            target_instance = all_sketch_instance[idx]
            target_positions = [
                local_idx for local_idx, gallery_idx in enumerate(gallery_indices)
                if all_photo_instance[gallery_idx] == target_instance
            ]
            if not target_positions:
                continue

            category_gallery = gallery_feat_all[gallery_indices]
            distance = 1.0 - F.cosine_similarity(sk_feat.unsqueeze(0), category_gallery)
            ranking = torch.argsort(distance)
            best_rank = min(
                int((ranking == target_position).nonzero(as_tuple=False)[0].item()) + 1
                for target_position in target_positions
            )

            acc1_total += int(best_rank <= 1)
            acc5_total += int(best_rank <= 5)
            total_sk += 1

        acc1 = acc1_total / total_sk if total_sk else 0.0
        acc5 = acc5_total / total_sk if total_sk else 0.0
        self._log_epoch_metrics(acc1, acc5)

        self.val_step_outputs_sk.clear()
        self.val_step_outputs_ph.clear()
