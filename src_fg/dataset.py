import os
import glob
import numpy as np
import torch
import math 
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

def _hamming_distance(a: np.ndarray, b: np.ndarray) -> int:
    return int(np.sum(a != b))

def _random_unique_permutations(
    rng: np.random.RandomState,
    identity: np.ndarray,
    n: int,
    *,
    exclude_identity: bool,
    prefilled: list[np.ndarray] | None = None,
):
    """
    Generate `n` unique permutations by repeated shuffling.

    NOTE: This is efficient for small n relative to num_patches! (e.g., 9 patches and n<=10k).
    """
    if n <= 0:
        return np.empty((0, identity.size), dtype=np.int64)

    perms: list[np.ndarray] = []
    seen: set[tuple[int, ...]] = set()

    if prefilled:
        for p in prefilled:
            key = tuple(int(x) for x in p.tolist())
            if key in seen:
                continue
            perms.append(p.copy())
            seen.add(key)

    while len(perms) < n:
        cand = identity.copy()
        rng.shuffle(cand)
        if exclude_identity and np.array_equal(cand, identity):
            continue
        key = tuple(int(x) for x in cand.tolist())
        if key in seen:
            continue
        perms.append(cand.copy())
        seen.add(key)

    return np.stack(perms, axis=0)

def _farthest_first_select(
    pool: np.ndarray,
    *,
    k: int,
    init: np.ndarray | None = None,
) -> np.ndarray:
    """
    Farthest-first (maximin) selection under Hamming distance.

    pool: [N, P] unique permutations
    k: number of items to select from pool
    init: optional initial permutation used to seed distances (not necessarily in pool)
    """
    if k <= 0:
        return np.empty((0, pool.shape[1]), dtype=pool.dtype)
    if k > pool.shape[0]:
        raise ValueError("k cannot be larger than pool size")

    P = pool.shape[1]
    if init is None:
        first_idx = 0
        selected_idx = np.empty(k, dtype=np.int64)
        selected_idx[0] = first_idx
        min_dist = np.sum(pool != pool[first_idx], axis=1, dtype=np.int16)
        min_dist[first_idx] = -1
        start = 1
    else:
        if init.shape != (P,):
            raise ValueError("init must have shape [num_patches]")
        selected_idx = np.empty(k, dtype=np.int64)
        min_dist = np.sum(pool != init, axis=1, dtype=np.int16)
        start = 0

    for i in range(start, k):
        next_idx = int(min_dist.argmax())
        selected_idx[i] = next_idx

        dist = np.sum(pool != pool[next_idx], axis=1, dtype=np.int16)
        min_dist = np.minimum(min_dist, dist)
        min_dist[next_idx] = -1

    return pool[selected_idx]

def generate_jigsaw_permutations(
    *,
    num_patches: int,
    num_permutations: int,
    seed: int = 0,
    exclude_identity: bool = True,
    candidates_per_round: int = 2048,
    strategy: str = "farthest",
    pool_multiplier: int = 10,
):
    """
    Select a set of patch permutations for jigsaw classification.

    Default strategy ("farthest") uses farthest-first selection on a random candidate pool.
    This approximates Noroozi & Favaro-style "max mutual Hamming distance" sets, while
    avoiding the O(K^2) Python-loop implementation that can appear to "hang" for K=1000.

    Returns: torch.LongTensor of shape [num_permutations, num_patches]
    """
    if num_patches <= 1:
        raise ValueError("num_patches must be > 1")
    if num_permutations <= 0:
        raise ValueError("num_permutations must be > 0")
    if pool_multiplier <= 0:
        raise ValueError("pool_multiplier must be > 0")

    max_unique = math.factorial(num_patches)
    if num_permutations > max_unique - (1 if exclude_identity else 0):
        raise ValueError(
            f"num_permutations={num_permutations} exceeds unique permutations for num_patches={num_patches}"
        )

    rng = np.random.RandomState(seed)
    identity = np.arange(num_patches, dtype=np.int64)

    strategy = str(strategy).strip().lower()
    if strategy not in {"farthest", "random", "legacy"}:
        raise ValueError("strategy must be one of: farthest | random | legacy")

    # Fast path: random unique permutations (good enough for most training runs).
    if strategy == "random":
        if exclude_identity:
            arr = _random_unique_permutations(rng, identity, num_permutations, exclude_identity=True)
            return torch.as_tensor(arr, dtype=torch.long)

        prefilled = [identity]
        arr_rest = _random_unique_permutations(
            rng,
            identity,
            num_permutations - 1,
            exclude_identity=True,
        )
        arr = np.concatenate([np.stack(prefilled, axis=0), arr_rest], axis=0)
        return torch.as_tensor(arr, dtype=torch.long)

    # Legacy Python-loop greedy (kept for reference; can be very slow for K=1000).
    if strategy == "legacy":
        selected = []
        selected_set = set()

        def _add_perm(p: np.ndarray) -> None:
            key = tuple(int(x) for x in p.tolist())
            selected.append(p.copy())
            selected_set.add(key)

        # seed selection
        if not exclude_identity:
            _add_perm(identity)
        else:
            # ensure first perm is not identity
            p = identity.copy()
            while True:
                rng.shuffle(p)
                if not np.array_equal(p, identity):
                    break
            _add_perm(p)

        while len(selected) < num_permutations:
            best = None
            best_score = -1
            # sample candidates and pick the one maximizing min Hamming distance to selected set
            for _ in range(candidates_per_round):
                cand = identity.copy()
                rng.shuffle(cand)
                if exclude_identity and np.array_equal(cand, identity):
                    continue
                key = tuple(int(x) for x in cand.tolist())
                if key in selected_set:
                    continue

                score = min(_hamming_distance(cand, s) for s in selected)
                if score > best_score:
                    best_score = score
                    best = cand

            if best is None:
                # fallback: sampling until a new permutation appears
                while True:
                    cand = identity.copy()
                    rng.shuffle(cand)
                    if exclude_identity and np.array_equal(cand, identity):
                        continue
                    key = tuple(int(x) for x in cand.tolist())
                    if key not in selected_set:
                        best = cand
                        break

            _add_perm(best)

        return torch.as_tensor(np.stack(selected, axis=0), dtype=torch.long)

    # Default: farthest-first selection on a candidate pool.
    available = max_unique - (1 if exclude_identity else 0)
    pool_size = max(int(candidates_per_round), int(num_permutations) * int(pool_multiplier))
    pool_size = max(pool_size, int(num_permutations))
    pool_size = min(pool_size, int(available))

    if exclude_identity:
        pool = _random_unique_permutations(rng, identity, pool_size, exclude_identity=True)
        selected = _farthest_first_select(pool, k=num_permutations)
        return torch.as_tensor(selected, dtype=torch.long)

    # include identity as the first permutation if identity is allowed
    pool = _random_unique_permutations(rng, identity, pool_size, exclude_identity=True)
    selected_rest = _farthest_first_select(pool, k=num_permutations - 1, init=identity)
    out = np.concatenate([identity.reshape(1, -1), selected_rest], axis=0)
    return torch.as_tensor(out, dtype=torch.long)

def permute_patches_tensor(img: torch.Tensor, *, grid: int, perm: torch.Tensor) -> torch.Tensor:
    """
    Apply a patch permutation on a normalized tensor image.

    img: Tensor [C, H, W]
    perm: Tensor [grid*grid] where perm[tgt_id] = src_id (0-indexed, row-major)
    """
    if img.dim() != 3:
        raise ValueError("img must be [C,H,W]")
    if perm.numel() != grid * grid:
        raise ValueError("perm length must be grid*grid")

    img = img.clone()
    C, H, W = img.shape
    patch_w = W // grid
    patch_h = H // grid

    region_w = patch_w * grid
    region_h = patch_h * grid

    # Only permute the top-left divisible region; leave remainder borders intact.
    region = img[:, :region_h, :region_w]
    out_region = torch.empty_like(region)

    perm = perm.to(device=img.device)
    for tgt_id in range(grid * grid):
        src_id = int(perm[tgt_id].item())
        tgt_r, tgt_c = divmod(tgt_id, grid)
        src_r, src_c = divmod(src_id, grid)

        ys, ye = src_r * patch_h, (src_r + 1) * patch_h
        xs, xe = src_c * patch_w, (src_c + 1) * patch_w
        yt, yte = tgt_r * patch_h, (tgt_r + 1) * patch_h
        xt, xte = tgt_c * patch_w, (tgt_c + 1) * patch_w

        out_region[:, yt:yte, xt:xte] = region[:, ys:ye, xs:xe]

    img[:, :region_h, :region_w] = out_region
    return img

class SketchyDataset(torch.utils.data.Dataset):
    def __init__(self, args, mode):
        self.args = args
        self.mode = mode
        unseen_classes = UNSEEN_CLASSES[getattr(self.args, "dataset", "sketchy_2")]

        self.all_categories = os.listdir(os.path.join(self.args.root, 'sketch'))
        self.transform = normal_transform()
        self.aumentation = aumented_transform()

        if self.mode == "train":
            self.all_categories = sorted(list(set(self.all_categories) - set(unseen_classes)))
        else:
            self.all_categories = sorted(list(set(unseen_classes)))

        self.all_sketches_path = []
        self.all_photos_path = {}
        self.photo_id_map = {}

        # Conditional cross-modal jigsaw settings (SpLIP-style)
        self.jigsaw_grid = int(getattr(self.args, "jigsaw_grid", 3))
        self.jigsaw_num_perm = int(getattr(self.args, "jigsaw_num_perm", 100))
        self.jigsaw_seed = int(getattr(self.args, "jigsaw_seed", 0))
        self.jigsaw_perm_strategy = str(getattr(self.args, "jigsaw_perm_strategy", "farthest"))
        self.jigsaw_perm_pool_mult = int(getattr(self.args, "jigsaw_perm_pool_mult", 10))
        self.jigsaw_perm_pool_min = int(getattr(self.args, "jigsaw_perm_pool_min", 2048))
        self.jigsaw_perms = None
        if self.mode == "train":
            self.jigsaw_perms = generate_jigsaw_permutations(
                num_patches=self.jigsaw_grid * self.jigsaw_grid,
                num_permutations=self.jigsaw_num_perm,
                seed=self.jigsaw_seed,
                exclude_identity=True,
                strategy=self.jigsaw_perm_strategy,
                pool_multiplier=self.jigsaw_perm_pool_mult,
                candidates_per_round=self.jigsaw_perm_pool_min,
            )

        for category in self.all_categories:
            self.all_sketches_path.extend(glob.glob(os.path.join(self.args.root, 'sketch', category, '*')))
            photo_paths = glob.glob(os.path.join(self.args.root, 'photo', category, '*'))
            self.all_photos_path[category] = photo_paths
            for p in photo_paths:
                key = os.path.normcase(os.path.normpath(p))
                if key not in self.photo_id_map:
                    self.photo_id_map[key] = len(self.photo_id_map)

    def __len__(self):
        return len(self.all_sketches_path)

    def __getitem__(self, index):
        sk_path = self.all_sketches_path[index]
        category = sk_path.split(os.path.sep)[-2]

        sk_base = os.path.basename(sk_path)
        sk_parts = sk_base.split("-")
        pos_sample = "-".join(sk_parts[:-1]) if len(sk_parts) > 1 else os.path.splitext(sk_base)[0]
        pos_path = glob.glob(os.path.join(self.args.root, 'photo', category, pos_sample + '.*'))
        if len(pos_path) == 0:
            print(sk_path)
            return None

        pos_path = pos_path[0]
        pos_key = os.path.normcase(os.path.normpath(pos_path))
        pos_id = self.photo_id_map.get(pos_key, -1)
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
            
            # SpLIP-style: permute sketch patches with a sampled permutation y_perm
            perm_idx = int(np.random.randint(0, self.jigsaw_perms.shape[0]))
            perm = self.jigsaw_perms[perm_idx]
            sk_perm_tensor = permute_patches_tensor(sk_tensor, grid=self.jigsaw_grid, perm=perm)
            
            return img_tensor, sk_tensor, img_aug_tensor, sk_aug_tensor, neg_tensor, \
                sk_perm_tensor, perm_idx, self.all_categories.index(category), pos_id

        else:
            return sk_tensor, sk_path, img_tensor, pos_sample, self.all_categories.index(category)
