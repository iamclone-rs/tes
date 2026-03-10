import os
import glob
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

        for category in self.all_categories:
            sketch_paths = sorted(glob.glob(os.path.join(self.opts.root, 'sketch', category, '*')))
            photo_paths = sorted(glob.glob(os.path.join(self.opts.root, 'photo', category, '*')))
            self.all_photos_path[category] = photo_paths
            self.all_photo_paths.extend(photo_paths)

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
        
        return img_tensor, sk_tensor, img_aug_tensor, sk_aug_tensor, neg_tensor, self.category_to_idx[category]


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
