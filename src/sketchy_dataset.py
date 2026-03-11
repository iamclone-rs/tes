import os
import glob
import math
import itertools
import numpy as np
import torch
from torchvision import transforms
from PIL import Image, ImageOps
from src.data_config import UNSEEN_CLASSES
from src.utils import get_zero_shot_split_key, is_fg_dataset

def aumented_transform():
    transform_list = [
        transforms.RandomResizedCrop(224, scale=(0.85, 1.0)),
        transforms.RandomHorizontalFlip(0.5),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]
    return transforms.Compose(transform_list)

def aumented_transform_1():
    transform_list = [
        transforms.RandomResizedCrop(224, scale=(0.85, 1.0)),
        transforms.RandomHorizontalFlip(0.5),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]
    return transforms.Compose(transform_list)

def aumented_transform_2():
    strong_color_jitter = transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1)
    transform_list = [
        transforms.Resize(224),
        transforms.RandomRotation(5),
        transforms.RandomApply([strong_color_jitter], p=0.8),
        transforms.RandomGrayscale(p=0.5),
        transforms.ToTensor(),
        transforms.RandomErasing(p=0.3, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]
    return transforms.Compose(transform_list)

def normal_transform():
    dataset_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return dataset_transforms


_ALL_PERMUTATION_CACHE = {}


def _get_all_permutations(num_patches, exclude_identity=True):
    cache_key = (num_patches, exclude_identity)
    cached = _ALL_PERMUTATION_CACHE.get(cache_key)
    if cached is not None:
        return cached

    identity = tuple(range(num_patches))
    permutations = []
    for permutation in itertools.permutations(range(num_patches)):
        if exclude_identity and permutation == identity:
            continue
        permutations.append(permutation)

    all_permutations = np.asarray(permutations, dtype=np.int16)
    _ALL_PERMUTATION_CACHE[cache_key] = all_permutations
    return all_permutations


def _farthest_first_select(pool, num_select, init=None, rng=None):
    if num_select <= 0:
        return np.empty((0, pool.shape[1]), dtype=pool.dtype)
    if num_select > pool.shape[0]:
        raise ValueError("num_select cannot be larger than pool size")

    patch_count = pool.shape[1]
    if init is None:
        selected_indices = np.empty(num_select, dtype=np.int64)
        first_index = int(rng.randint(pool.shape[0])) if rng is not None else 0
        selected_indices[0] = first_index
        min_distance = np.sum(pool != pool[first_index], axis=1, dtype=np.int16)
        min_distance[first_index] = -1
        start_index = 1
    else:
        if init.shape != (patch_count,):
            raise ValueError("init must match permutation length")
        selected_indices = np.empty(num_select, dtype=np.int64)
        min_distance = np.sum(pool != init, axis=1, dtype=np.int16)
        start_index = 0

    for idx in range(start_index, num_select):
        candidate_indices = np.flatnonzero(min_distance == min_distance.max())
        if candidate_indices.size == 0:
            raise RuntimeError("No candidate found during farthest-first selection")
        if rng is not None:
            next_index = int(candidate_indices[rng.randint(candidate_indices.size)])
        else:
            next_index = int(candidate_indices[0])
        selected_indices[idx] = next_index
        distance = np.sum(pool != pool[next_index], axis=1, dtype=np.int16)
        min_distance = np.minimum(min_distance, distance)
        min_distance[next_index] = -1

    return pool[selected_indices]


def generate_jigsaw_permutations(
    *,
    num_patches,
    num_permutations,
    seed=0,
    exclude_identity=True,
):
    if num_patches <= 1:
        raise ValueError("num_patches must be > 1")
    if num_permutations <= 0:
        raise ValueError("num_permutations must be > 0")

    max_unique = math.factorial(num_patches)
    available = max_unique - (1 if exclude_identity else 0)
    if num_permutations > available:
        raise ValueError("num_permutations exceeds the number of unique permutations")

    rng = np.random.RandomState(seed)

    if available > 1_000_000:
        raise ValueError("exact_farthest is too expensive for this grid size")

    full_pool = _get_all_permutations(num_patches, exclude_identity=exclude_identity)
    selected = _farthest_first_select(full_pool, num_permutations, rng=rng)
    return torch.as_tensor(selected, dtype=torch.long)


def permute_patches_tensor(image_tensor, *, grid, permutation):
    if image_tensor.dim() != 3:
        raise ValueError("image_tensor must be [C, H, W]")
    if permutation.numel() != grid * grid:
        raise ValueError("permutation length must equal grid * grid")

    output = image_tensor.clone()
    channels, height, width = output.shape
    patch_width = width // grid
    patch_height = height // grid
    region_width = patch_width * grid
    region_height = patch_height * grid

    region = output[:, :region_height, :region_width]
    permuted_region = torch.empty_like(region)
    permutation = permutation.to(device=output.device)

    for target_idx in range(grid * grid):
        source_idx = int(permutation[target_idx].item())
        target_row, target_col = divmod(target_idx, grid)
        source_row, source_col = divmod(source_idx, grid)

        source_y1, source_y2 = source_row * patch_height, (source_row + 1) * patch_height
        source_x1, source_x2 = source_col * patch_width, (source_col + 1) * patch_width
        target_y1, target_y2 = target_row * patch_height, (target_row + 1) * patch_height
        target_x1, target_x2 = target_col * patch_width, (target_col + 1) * patch_width

        permuted_region[:, target_y1:target_y2, target_x1:target_x2] = region[:, source_y1:source_y2, source_x1:source_x2]

    output[:, :region_height, :region_width] = permuted_region
    return output


def _clean_categories(root_dir):
    categories = os.listdir(root_dir)
    if ".ipynb_checkpoints" in categories:
        categories.remove(".ipynb_checkpoints")
    return sorted(categories)


def _photo_instance_id(path):
    return os.path.splitext(os.path.basename(path))[0]


def _sketch_instance_id(path):
    stem = os.path.splitext(os.path.basename(path))[0]
    if "-" in stem:
        return stem.rsplit("-", 1)[0]
    return stem

class TrainDataset(torch.utils.data.Dataset):
    def __init__(self, opts):
        self.opts = opts
        self.is_fg = is_fg_dataset(opts)
        self.use_cjs = bool(getattr(opts, "use_cjs", True))
        split_key = get_zero_shot_split_key(opts)
        if split_key == "sketchy_1":
            self.transform1 = normal_transform()
            self.transform2 = aumented_transform()
        else:
            self.transform1 = aumented_transform_1()
            self.transform2 = aumented_transform_2()
        
        all_categories = _clean_categories(os.path.join(self.opts.root, 'sketch'))
        if split_key not in UNSEEN_CLASSES:
            self.all_categories = all_categories
        else:
            unseen_classes = UNSEEN_CLASSES[split_key]
            self.all_categories = sorted(list(set(all_categories) - set(unseen_classes)))
        self.category_to_idx = {category: idx for idx, category in enumerate(self.all_categories)}

        self.all_sketches_path = []
        self.all_photos_path = {}
        self.fg_pos_photo = {}
        self.all_photo_paths = []
        self.photo_id_map = {}
        self.jigsaw_grid = int(getattr(self.opts, "jigsaw_grid", 3))
        self.jigsaw_num_perm = int(getattr(self.opts, "jigsaw_num_perm", 100))
        self.jigsaw_seed = int(getattr(self.opts, "jigsaw_seed", 0))
        self.jigsaw_permutations = None
        if self.use_cjs:
            self.jigsaw_permutations = generate_jigsaw_permutations(
                num_patches=self.jigsaw_grid * self.jigsaw_grid,
                num_permutations=self.jigsaw_num_perm,
                seed=self.jigsaw_seed,
                exclude_identity=True,
            )

        for category in self.all_categories:
            sketch_paths = sorted(glob.glob(os.path.join(self.opts.root, 'sketch', category, '*')))
            photo_paths = sorted(glob.glob(os.path.join(self.opts.root, 'photo', category, '*')))
            self.all_photos_path[category] = photo_paths
            self.all_photo_paths.extend(photo_paths)
            for photo_path in photo_paths:
                normalized_path = os.path.normcase(os.path.normpath(photo_path))
                if normalized_path not in self.photo_id_map:
                    self.photo_id_map[normalized_path] = len(self.photo_id_map)

            if self.is_fg:
                photo_by_instance = {_photo_instance_id(path): path for path in photo_paths}
                for sketch_path in sketch_paths:
                    instance_id = _sketch_instance_id(sketch_path)
                    pos_path = photo_by_instance.get(instance_id)
                    if pos_path is None:
                        continue
                    self.all_sketches_path.append(sketch_path)
                    self.fg_pos_photo[sketch_path] = pos_path
            else:
                self.all_sketches_path.extend(sketch_paths)

    def __len__(self):
        return len(self.all_sketches_path)
        
    def __getitem__(self, index):
        filepath = self.all_sketches_path[index]                
        category = filepath.split(os.path.sep)[-2]

        sk_path  = filepath
        if self.is_fg:
            img_path = self.fg_pos_photo[sk_path]
            neg_candidates = [path for path in self.all_photos_path[category] if path != img_path]
            if not neg_candidates:
                neg_candidates = [path for path in self.all_photo_paths if path != img_path]
        else:
            neg_classes = self.all_categories.copy()
            neg_classes.remove(category)
            img_path = np.random.choice(self.all_photos_path[category])
            neg_candidates = self.all_photos_path[np.random.choice(neg_classes)]

        if not neg_candidates:
            raise RuntimeError(f"No negative candidate found for category '{category}'")
        neg_path = np.random.choice(neg_candidates)

        sk_data  = ImageOps.pad(Image.open(sk_path).convert('RGB'),  size=(self.opts.max_size, self.opts.max_size))
        img_data = ImageOps.pad(Image.open(img_path).convert('RGB'), size=(self.opts.max_size, self.opts.max_size))
        neg_data = ImageOps.pad(Image.open(neg_path).convert('RGB'), size=(self.opts.max_size, self.opts.max_size))

        sk_tensor  = self.transform1(sk_data)
        img_tensor = self.transform1(img_data)
        neg_tensor = self.transform1(neg_data)
        
        sk_aug_tensor = self.transform2(sk_data)
        img_aug_tensor = self.transform2(img_data)

        normalized_img_path = os.path.normcase(os.path.normpath(img_path))
        pos_id = self.photo_id_map.get(normalized_img_path, -1)

        if self.use_cjs:
            permutation_index = int(np.random.randint(0, self.jigsaw_permutations.shape[0]))
            permutation = self.jigsaw_permutations[permutation_index]
            sk_perm_tensor = permute_patches_tensor(
                sk_tensor,
                grid=self.jigsaw_grid,
                permutation=permutation,
            )
            return (
                img_tensor,
                sk_tensor,
                img_aug_tensor,
                sk_aug_tensor,
                neg_tensor,
                sk_perm_tensor,
                permutation_index,
                self.category_to_idx[category],
                pos_id,
            )

        return (
            img_tensor,
            sk_tensor,
            img_aug_tensor,
            sk_aug_tensor,
            neg_tensor,
            self.category_to_idx[category],
            pos_id,
        )


class ValidDataset(torch.utils.data.Dataset):
    def __init__(self, args, mode='photo'):
        super(ValidDataset, self).__init__()
        self.args = args
        self.mode = mode
        self.is_fg = is_fg_dataset(args)
        self.transform = normal_transform()
        split_key = get_zero_shot_split_key(args)
        all_categories = _clean_categories(os.path.join(self.args.root, 'sketch'))
        if split_key not in UNSEEN_CLASSES:
            self.all_categories = all_categories
        else:
            unseen_classes = UNSEEN_CLASSES[split_key]
            self.all_categories = sorted([category for category in unseen_classes if category in all_categories])
        self.category_to_idx = {category: idx for idx, category in enumerate(self.all_categories)}

        self.paths = []
        self.instance_ids = []
        for category in self.all_categories:
            if self.mode == "photo":
                photo_paths = sorted(glob.glob(os.path.join(self.args.root, 'photo', category, '*')))
                self.paths.extend(photo_paths)
                if self.is_fg:
                    self.instance_ids.extend([_photo_instance_id(path) for path in photo_paths])
            else:
                sketch_paths = sorted(glob.glob(os.path.join(self.args.root, 'sketch', category, '*')))
                if self.is_fg:
                    photo_ids = {
                        _photo_instance_id(path)
                        for path in glob.glob(os.path.join(self.args.root, 'photo', category, '*'))
                    }
                    for path in sketch_paths:
                        instance_id = _sketch_instance_id(path)
                        if instance_id not in photo_ids:
                            continue
                        self.paths.append(path)
                        self.instance_ids.append(instance_id)
                else:
                    self.paths.extend(sketch_paths)

    def __getitem__(self, index):
        filepath = self.paths[index]                
        category = filepath.split(os.path.sep)[-2]
        
        image = ImageOps.pad(Image.open(filepath).convert('RGB'),  size=(self.args.max_size, self.args.max_size))
        image_tensor = self.transform(image)

        label = self.category_to_idx[category]
        if self.is_fg:
            return image_tensor, label, self.instance_ids[index]
        return image_tensor, label
    
    def __len__(self):
        return len(self.paths)
