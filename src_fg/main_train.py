import os
import torch
import numpy as np
import random
import argparse
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer 
from pytorch_lightning.loggers import TensorBoardLogger 
from pytorch_lightning.callbacks import ModelCheckpoint 

from src_fg.dataset import SketchyDataset
from src_fg.model import ZS_SBIR
from src.utils import get_all_categories

class PKBatchSampler(torch.utils.data.Sampler):
    def __init__(
        self,
        dataset,
        *,
        batch_size: int,
        samples_per_class: int,
        seed: int = 42,
        drop_last: bool = True,
    ):
        if samples_per_class <= 0:
            raise ValueError("samples_per_class must be > 0")
        if batch_size <= 0:
            raise ValueError("batch_size must be > 0")
        if batch_size % samples_per_class != 0:
            raise ValueError("batch_size must be divisible by samples_per_class")

        self.dataset = dataset
        self.batch_size = batch_size
        self.samples_per_class = samples_per_class
        self.classes_per_batch = batch_size // samples_per_class
        self.seed = seed
        self.drop_last = drop_last

        cat_to_label = {c: i for i, c in enumerate(self.dataset.all_categories)}
        label_to_indices = {}
        for idx, sk_path in enumerate(self.dataset.all_sketches_path):
            category = sk_path.split(os.path.sep)[-2]
            lab = cat_to_label.get(category, None)
            if lab is None:
                continue
            label_to_indices.setdefault(lab, []).append(idx)

        self.label_to_indices = {k: v for k, v in label_to_indices.items() if len(v) > 0}
        self.labels = list(self.label_to_indices.keys())
        if len(self.labels) == 0:
            raise ValueError("No labels found for PK sampling")

        self._epoch = 0

    def __len__(self):
        if self.drop_last:
            return len(self.dataset) // self.batch_size
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        rng = np.random.RandomState(self.seed + self._epoch)
        self._epoch += 1
        for _ in range(len(self)):
            chosen_labels = rng.choice(
                self.labels,
                size=self.classes_per_batch,
                replace=len(self.labels) < self.classes_per_batch,
            )
            batch = []
            for lab in chosen_labels:
                pool = self.label_to_indices[lab]
                replace = len(pool) < self.samples_per_class
                picked = rng.choice(pool, size=self.samples_per_class, replace=replace)
                batch.extend(int(i) for i in picked.tolist())
            yield batch

     
def collate_fn(batch):
    batch = [b for b in batch if b is not None]
    return torch.utils.data.default_collate(batch)

def get_datasets(opts):
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # torch.cuda.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)
    train_dataset = SketchyDataset(opts, mode="train")
    val_dataset = SketchyDataset(opts, mode="test")
    
    use_pk = bool(getattr(opts, "pk_sampling", True))
    if use_pk:
        samples_per_class = int(getattr(opts, "samples_per_class", 2))
        sampler_seed = int(getattr(opts, "sampler_seed", seed))
        batch_sampler = PKBatchSampler(
            train_dataset,
            batch_size=opts.batch_size,
            samples_per_class=samples_per_class,
            seed=sampler_seed,
        )
        train_loader = DataLoader(dataset=train_dataset, batch_sampler=batch_sampler, num_workers=opts.workers, collate_fn=collate_fn)
    else:
        train_loader = DataLoader(dataset=train_dataset, batch_size=opts.batch_size, num_workers=opts.workers, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(dataset=val_dataset, batch_size=opts.test_batch_size, num_workers=opts.workers, shuffle=False, collate_fn=collate_fn)
    
    return train_loader, val_loader
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="/kaggle/input/sketchy", help="path to dataset")
    parser.add_argument("--ckpt_path", type=str, default="", help="path to dataset")
    parser.add_argument("--dataset", type=str, default="sketchy_2", help="type of dataset")
    parser.add_argument("--output_dir", type=str, default="", help="output directory")
    parser.add_argument("--backbone", type=str, default="ViT-B/32")
    parser.add_argument("--n_ctx", type=int, default=2)
    parser.add_argument("--img_ctx", type=int, default=2)
    parser.add_argument("--max_size", type=int, default=224)
    parser.add_argument("--prompt_depth", type=int, default=12)
    parser.add_argument("--data_split", type=int, default=-1)
    parser.add_argument("--prec", type=str, default="fp16")
    parser.add_argument("--distill", type=str, default="cosine")
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--alpha", type=float, default=0.8)
    parser.add_argument("--gamma", type=float, default=0.1)
    parser.add_argument("--beta", type=float, default=1)
    parser.add_argument("--lambd", type=float, default=0.1)

    # Conditional cross-modal jigsaw (SpLIP-style)
    parser.add_argument("--jigsaw_grid", type=int, default=3, help="grid size for patch permutation (e.g., 3 -> 3x3)")
    parser.add_argument("--jigsaw_num_perm", type=int, default=1000, help="number of permutation classes |Yperm|")
    parser.add_argument("--jigsaw_seed", type=int, default=0, help="seed for permutation set generation")
    parser.add_argument("--jigsaw_layers", type=int, default=2, help="number of transformer encoder layers in Fjs")
    parser.add_argument("--jigsaw_nhead", type=int, default=8, help="number of attention heads in Fjs")
    parser.add_argument("--jigsaw_dropout", type=float, default=0.1, help="dropout for Fjs")
    parser.add_argument("--xmodal_weight", type=float, default=1.0, help="weight for sketch-photo cross-modal InfoNCE")
     
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--test_batch_size', type=int, default=1024)
    parser.add_argument('--epochs', type=int, default=25)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--progress', type=bool, default=False)
    parser.add_argument('--use_subset', type=bool, default=False)
    parser.add_argument('--pk_sampling', type=bool, default=True, help='use PK sampler for hard triplet mining')
    parser.add_argument('--samples_per_class', type=int, default=2, help='K for PK sampling (batch has P classes, K samples each)')
    parser.add_argument('--sampler_seed', type=int, default=42, help='seed for PK sampler')
    
    parser.add_argument('--exp_name', type=str, default='Co_prompt')
    
    args = parser.parse_args()
    logger = TensorBoardLogger('tb_logs', name=args.exp_name)
    
    checkpoint_callback = ModelCheckpoint(
        monitor='top1',
        dirpath='saved_models/%s'%args.exp_name,
        filename="epoch{epoch:02d}-top1{mAP:.4f}",
        save_top_k=1,
        mode='max',
        save_last=True)
    
    ckpt_path = args.ckpt_path
    if not os.path.exists(ckpt_path):
        ckpt_path = None
    else:
        print ('resuming training from %s'%ckpt_path)

    train_loader, val_loader = get_datasets(args)
    trainer = Trainer(accelerator='gpu', devices=1, 
        min_epochs=1, max_epochs=args.epochs,
        benchmark=True,
        logger=logger,
        check_val_every_n_epoch=1,
        enable_progress_bar=args.progress,
        callbacks=[checkpoint_callback]
    )

    classnames = get_all_categories(args)
 
    if ckpt_path is None:
        model = ZS_SBIR(args=args, classname=classnames)
    else:
        ckpt = torch.load(ckpt_path, map_location="cpu")
        sd = ckpt["state_dict"]

        skip = [
            "model.prompt_learner_photo.token_prefix",
            "model.prompt_learner_photo.token_suffix",
            "model.prompt_learner_sketch.token_prefix",
            "model.prompt_learner_sketch.token_suffix",
        ]
        for k in skip:
            sd.pop(k, None)

        model = ZS_SBIR(args=args, classname=classnames)  # classnames = 220
        missing, unexpected = model.load_state_dict(sd, strict=False)

    trainer.fit(model, train_loader, val_loader)
