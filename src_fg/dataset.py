import os
import glob
import numpy as np
import torch
import math 
import random
from torch.nn import functional as F

from torchvision import transforms
from PIL import Image, ImageOps
from src.data_config import UNSEEN_CLASSES
        
def aumented_transform():
    transform_list = [
        transforms.RandomResizedCrop(224, scale=(0.85, 1.0)),
        transforms.RandomHorizontalFlip(0.8),
        transforms.ColorJitter(
            brightness=0.15, contrast=0.15, saturation=0.15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                             0.229, 0.224, 0.225])
    ]
    return transforms.Compose(transform_list)


def normal_transform():
    dataset_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                             0.229, 0.224, 0.225])
    ])
    return dataset_transforms

def split_img(img, grid=3):
    splitimg = []
    width, height = img.size
    tile_w = int(width // grid)
    tile_h = int(height // grid)
    for row in range(grid):
        for col in range(grid):
            box = (tile_w * col, tile_h * row, tile_w * (col + 1), tile_h * (row + 1))
            region = img.crop(box)
            splitimg.append(region)
    return splitimg

def rebuild_from_perm(img, perm, grid=3):
    tiles = split_img(img, grid=grid)
    if torch.is_tensor(perm):
        perm = perm.tolist()

    width, height = img.size
    tile_w = width // grid
    tile_h = height // grid

    new_img = Image.new(img.mode, (tile_w * grid, tile_h * grid))
    for idx, src_idx in enumerate(perm):
        row = idx // grid
        col = idx % grid
        new_img.paste(tiles[src_idx], (col * tile_w, row * tile_h))
    return new_img

def make_jigsaw(img, grid=3, perm=None):
    if perm is None:
        perm = torch.randperm(grid ** 2)
    jig_img = rebuild_from_perm(img, perm=perm, grid=grid)
    return jig_img, perm

def extract_instance_id(file_path_or_name):
    file_name = os.path.basename(file_path_or_name)
    stem, _ = os.path.splitext(file_name)
    if "-" in stem:
        return stem.rsplit("-", 1)[0]
    return stem

def generate_permutation_bank(grid=3, num_perms=30, seed=42):
    rng = random.Random(seed)
    total_tiles = grid ** 2
    base = list(range(total_tiles))
    permutations = [tuple(base)]
    seen = {tuple(base)}

    while len(permutations) < num_perms:
        candidate = base.copy()
        rng.shuffle(candidate)
        candidate_t = tuple(candidate)
        if candidate_t not in seen:
            seen.add(candidate_t)
            permutations.append(candidate_t)

    return [torch.tensor(p, dtype=torch.long) for p in permutations]

def remove_patches_white_normalized(
    img: torch.Tensor,
    grid: int = 3,
    remove_ids = [1],
    mean = (0.485, 0.456, 0.406),
    std  = (0.229, 0.224, 0.225),
):
    """
    img: Tensor [C, H, W] was normalize by (mean, std)
    return: Tensor [C, H, W]
    """
    img = img.clone()
    C, H, W = img.shape

    patch_w = W // grid
    patch_h = H // grid

    mean_t = torch.tensor(mean, device=img.device, dtype=img.dtype).view(-1, 1, 1)
    std_t  = torch.tensor(std,  device=img.device, dtype=img.dtype).view(-1, 1, 1)
    white_norm = (1.0 - mean_t) / std_t  # shape [3,1,1]

    if C == 3:
        fill_val = white_norm
    else:
        fill_val = img.max().detach()  # scalar
        fill_val = torch.full((C, 1, 1), float(fill_val), device=img.device, dtype=img.dtype)

    for patch_id in remove_ids:
        pid = patch_id - 1
        row = pid // grid
        col = pid % grid

        y1, y2 = row * patch_h, (row + 1) * patch_h
        x1, x2 = col * patch_w, (col + 1) * patch_w

        img[:, y1:y2, x1:x2] = fill_val.expand(C, patch_h, patch_w)

    return img

def remove_patches(img, grid=2, remove_ids=[5, 9]): 
    img = img.copy() 
    w, h = img.size 
    patch_w = w // grid 
    patch_h = h // grid 
    img_np = np.array(img) 
    
    for patch_id in remove_ids: 
        patch_id -= 1 
        row = patch_id // grid 
        col = patch_id % grid 
        
        x1 = col * patch_w 
        y1 = row * patch_h 
        x2 = x1 + patch_w 
        y2 = y1 + patch_h # tạo noise ngẫu nhiên (0-255) 
        
        noise = np.random.randint(255, 256, (patch_h, patch_w, 3), dtype=np.uint8 ) 
        img_np[y1:y2, x1:x2] = noise 
        
    return Image.fromarray(img_np)

class SketchyDataset(torch.utils.data.Dataset):
    def __init__(self, args, mode):
        self.args = args
        self.mode = mode
        unseen_classes = UNSEEN_CLASSES[self.args.dataset]

        self.all_categories = os.listdir(os.path.join(self.args.root, 'sketch'))
        self.transform = normal_transform()
        self.aumentation = aumented_transform()

        if self.mode == "train":
            self.all_categories = sorted(list(set(self.all_categories) - set(unseen_classes)))
        else:
            self.all_categories = sorted(list(set(unseen_classes)))

        self.all_sketches_path = []
        self.all_photos_path = {}
        self.cjs_grid = getattr(self.args, "cjs_grid", 3)
        self.cjs_num_perms = getattr(self.args, "cjs_num_perms", 30)
        self.permutation_bank = generate_permutation_bank(
            grid=self.cjs_grid,
            num_perms=self.cjs_num_perms,
            seed=42,
        )

        for category in self.all_categories:
            self.all_sketches_path.extend(glob.glob(os.path.join(self.args.root, 'sketch', category, '*')))
            self.all_photos_path[category] = glob.glob(os.path.join(self.args.root, 'photo', category, '*'))

    def __len__(self):
        return len(self.all_sketches_path)

    def __getitem__(self, index):
        sk_path = self.all_sketches_path[index]
        category = sk_path.split(os.path.sep)[-2]

        pos_sample = extract_instance_id(sk_path)
        pos_path = glob.glob(os.path.join(self.args.root, 'photo', category, pos_sample + '.*'))
        if len(pos_path) == 0:
            print(sk_path)
            return None

        pos_path = pos_path[0]
        photo_category = self.all_photos_path[category].copy()
        photo_category = [p for p in photo_category if p != pos_path]

        neg_path = np.random.choice(photo_category)

        sk_data  = ImageOps.pad(Image.open(sk_path).convert('RGB'),  size=(self.args.max_size, self.args.max_size))
        img_data = ImageOps.pad(Image.open(pos_path).convert('RGB'), size=(self.args.max_size, self.args.max_size))
        neg_data = ImageOps.pad(Image.open(neg_path).convert('RGB'), size=(self.args.max_size, self.args.max_size))

        sk_tensor = self.transform(sk_data)
        img_tensor = self.transform(img_data)
        neg_tensor = self.transform(neg_data)

        if self.mode == "train":
            sk_aug_tensor = self.aumentation(sk_data)
            img_aug_tensor = self.aumentation(img_data)

            perm_idx = np.random.randint(0, len(self.permutation_bank))
            perm = self.permutation_bank[perm_idx]
            sk_jigsaw_img, _ = make_jigsaw(sk_data, grid=self.cjs_grid, perm=perm)

            sk_jigsaw = self.transform(sk_jigsaw_img)
            
            # sk_jigsaw = remove_patches(img=sk_data, grid=3, remove_ids=[1, 9])
            # pos_jigsaw = remove_patches(img=img_data, grid=3, remove_ids=[1, 9])
            # neg_jigsaw = remove_patches(img=img_data, grid=3, remove_ids=[2, 8])
            # sk_jigsaw = self.transform(sk_jigsaw)
            # pos_jigsaw = self.transform(pos_jigsaw)
            # neg_jigsaw = self.transform(neg_jigsaw)
            
            return img_tensor, sk_tensor, img_aug_tensor, sk_aug_tensor, neg_tensor, \
                sk_jigsaw, torch.tensor(perm_idx, dtype=torch.long), self.all_categories.index(category)

        else:
            return sk_tensor, sk_path, img_tensor, pos_sample, self.all_categories.index(category)