import os
import sys
import torch
import numpy as np
import random
import argparse
from torch.utils.data import DataLoader, Subset
from pytorch_lightning import Trainer 
from pytorch_lightning.loggers import TensorBoardLogger 
from pytorch_lightning.callbacks import ModelCheckpoint 

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.sketchy_dataset import TrainDataset, ValidDataset
from src.model import ZS_SBIR
from src.utils import get_all_categories, is_fg_dataset


class PKBatchSampler(torch.utils.data.Sampler):
    def __init__(
        self,
        dataset,
        *,
        batch_size,
        samples_per_class,
        seed=42,
        drop_last=True,
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
        self._epoch = 0

        label_to_indices = {}
        for idx, sketch_path in enumerate(self.dataset.all_sketches_path):
            category = sketch_path.split(os.path.sep)[-2]
            label = self.dataset.category_to_idx.get(category)
            if label is None:
                continue
            label_to_indices.setdefault(label, []).append(idx)

        self.label_to_indices = {
            label: indices for label, indices in label_to_indices.items() if indices
        }
        self.labels = list(self.label_to_indices.keys())
        if not self.labels:
            raise ValueError("No labels found for PK sampling")

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
            batch_indices = []
            for label in chosen_labels:
                pool = self.label_to_indices[label]
                replace = len(pool) < self.samples_per_class
                selected = rng.choice(pool, size=self.samples_per_class, replace=replace)
                batch_indices.extend(int(index) for index in selected.tolist())
            yield batch_indices

def get_datasets(opts, subset_ratio=0.2):
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    
    train_dataset = TrainDataset(opts)
    val_sketch = ValidDataset(opts, mode='sketch')
    val_photo = ValidDataset(opts)

    if opts.use_subset:
        train_size = int(len(train_dataset) * subset_ratio)
        train_indices = random.sample(range(len(train_dataset)), train_size)
        train_dataset = Subset(train_dataset, train_indices)
        
        val_size = int(len(val_sketch) * subset_ratio)
        val_indices = random.sample(range(len(val_sketch)), val_size)
        val_sketch = Subset(val_sketch, val_indices)
        
        val_size = int(len(val_photo) * subset_ratio)
        val_indices = random.sample(range(len(val_photo)), val_size)
        val_photo = Subset(val_photo, val_indices)

    use_pk_sampling = (
        bool(getattr(opts, "pk_sampling", True))
        and is_fg_dataset(opts)
        and not isinstance(train_dataset, Subset)
    )
    if use_pk_sampling:
        batch_sampler = PKBatchSampler(
            train_dataset,
            batch_size=opts.batch_size,
            samples_per_class=opts.samples_per_class,
            seed=opts.sampler_seed,
        )
        train_loader = DataLoader(dataset=train_dataset, batch_sampler=batch_sampler, num_workers=opts.workers)
    else:
        train_loader = DataLoader(dataset=train_dataset, batch_size=opts.batch_size, num_workers=opts.workers, shuffle=True)
    val_sketch_loader = DataLoader(dataset=val_sketch, batch_size=opts.test_batch_size, num_workers=opts.workers, shuffle=False)
    val_photo_loader = DataLoader(dataset=val_photo, batch_size=opts.test_batch_size, num_workers=opts.workers, shuffle=False)

    return train_loader, val_sketch_loader, val_photo_loader

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="../datasets/tuberlin", help="path to dataset")
    parser.add_argument("--ckpt_path", type=str, default="", help="path to dataset")
    parser.add_argument("--dataset", type=str, default="sketchy_2", help="type of dataset")
    parser.add_argument("--output_dir", type=str, default="", help="output directory")
    parser.add_argument("--backbone", type=str, default="ViT-B/32")
    parser.add_argument("--n_ctx", type=int, default=2)
    parser.add_argument("--img_ctx", type=int, default=2)
    parser.add_argument("--max_size", type=int, default=224)
    parser.add_argument("--prompt_depth", type=int, default=12)
    parser.add_argument("--data_split", type=int, default=-1, help="zero-shot split id for Sketchy/FG-SBIR: 1 or 2; -1 defaults to split 2")
    parser.add_argument("--prec", type=str, default="fp16")
    parser.add_argument("--distill", type=str, default="cosine")
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--w_triplet", type=float, default=1.0, help="weight for triplet loss")
    parser.add_argument("--w_cross", type=float, default=1.0, help="weight for cross-modal contrastive loss")
    parser.add_argument("--w_distill", type=float, default=1.0, help="weight for distillation loss")
    parser.add_argument("--w_cls", type=float, default=1.0, help="weight for classification loss")
    parser.add_argument("--w_mcc", type=float, default=1.0, help="weight for MCC loss")
    parser.add_argument("--w_cjs", type=float, default=0.1, help="weight for conditional cross-modal jigsaw loss")
    parser.add_argument("--triplet_margin", type=float, default=0.3, help="margin used for triplet loss")
    parser.add_argument("--use_cjs", type=bool, default=True, help="enable conditional cross-modal jigsaw training")
    parser.add_argument("--jigsaw_grid", type=int, default=3, help="grid size for patch permutation")
    parser.add_argument("--jigsaw_num_perm", type=int, default=100, help="number of selected patch permutations")
    parser.add_argument("--jigsaw_seed", type=int, default=0, help="seed used for jigsaw permutation bank")
    parser.add_argument("--jigsaw_layers", type=int, default=2, help="number of layers in the jigsaw solver")
    parser.add_argument("--jigsaw_nhead", type=int, default=8, help="number of attention heads in the jigsaw solver")
    parser.add_argument("--jigsaw_dropout", type=float, default=0.1, help="dropout in the jigsaw solver")
    
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--test_batch_size', type=int, default=1024)
    parser.add_argument('--epochs', type=int, default=60)
    parser.add_argument('--workers', type=int, default=12)
    parser.add_argument('--use_adapt_sk', type=bool, default=True)
    parser.add_argument('--use_adapt_ph', type=bool, default=True)
    parser.add_argument('--use_adapt_txt', type=bool, default=True)
    parser.add_argument('--use_co_sk', type=bool, default=True)
    parser.add_argument('--use_co_ph', type=bool, default=True)
    parser.add_argument('--progress', type=bool, default=False, help=argparse.SUPPRESS)
    parser.add_argument('--use_subset', type=bool, default=False)
    parser.add_argument('--pk_sampling', type=bool, default=True, help='use PK sampler for FG hard triplet training')
    parser.add_argument('--samples_per_class', type=int, default=4, help='number of samples per class in one PK batch')
    parser.add_argument('--sampler_seed', type=int, default=42, help='seed for PK sampler')
    
    parser.add_argument('--exp_name', type=str, default='Co_prompt')
    
    args = parser.parse_args()
    logger = TensorBoardLogger('tb_logs', name=args.exp_name)
    monitor_metric = 'acc1'
    checkpoint_name = "{epoch:02d}-{acc1:.4f}"
    
    checkpoint_callback = ModelCheckpoint(
        monitor=monitor_metric,
        dirpath='saved_models/%s'%args.exp_name,
        filename=checkpoint_name,
        save_top_k=1,
        mode='max',
        save_last=True)
    
    ckpt_path = args.ckpt_path
    if not os.path.exists(ckpt_path):
        ckpt_path = None
    else:
        print ('resuming training from %s'%ckpt_path)

    train_loader, val_sketch_loader, val_photo_loader = get_datasets(args)
    trainer = Trainer(accelerator='gpu', devices=1, 
        min_epochs=1, max_epochs=args.epochs,
        benchmark=True,
        logger=logger,
        check_val_every_n_epoch=1,
        enable_progress_bar=False,
        enable_model_summary=False,
        num_sanity_val_steps=0,
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

    trainer.fit(model, train_loader, [val_sketch_loader, val_photo_loader])
