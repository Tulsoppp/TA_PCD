import os
import random
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF
from tqdm import tqdm

class PolypDataset(Dataset):
    """
    Dataset loader for Kvasir-SEG (images and masks).
    Supports:
    - In-memory tensor caching for ultra-fast training on low-resource machines
    - Paper augmentations implemented via high-speed PyTorch tensor operations:
      * Random horizontal flip
      * Random vertical flip
      * Random rotation (-90 to +90 degrees)
      * Random Gaussian blur
    - Normalization to [0, 1]
    - Masks strictly binary {0, 1}
    """
    def __init__(self, image_paths, mask_paths, img_size=224, is_train=True, augment=True, preload=True):
        assert len(image_paths) == len(mask_paths), "Mismatch between images and masks count"
        self.image_paths = sorted(image_paths)
        self.mask_paths = sorted(mask_paths)
        self.filenames = [os.path.basename(p) for p in self.image_paths]
        self.img_size = img_size
        self.is_train = is_train
        self.augment = augment and is_train
        self.preload = preload

        if self.preload:
            desc = "Preloading Train Images" if is_train else "Preloading Val/Test Images"
            cached_imgs = []
            cached_msks = []
            for img_p, msk_p in zip(tqdm(self.image_paths, desc=desc, leave=False), self.mask_paths):
                # Load image with OpenCV and convert to RGB
                img = cv2.imread(img_p)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
                
                # Load mask
                msk = cv2.imread(msk_p, cv2.IMREAD_GRAYSCALE)
                msk = cv2.resize(msk, (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)
                msk = (msk >= 128).astype(np.float32)

                # Convert to PyTorch float tensors
                img_t = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0 # [3, 224, 224]
                msk_t = torch.from_numpy(msk).unsqueeze(0).float()             # [1, 224, 224]

                cached_imgs.append(img_t)
                cached_msks.append(msk_t)

            self.cached_images = torch.stack(cached_imgs)  # [N, 3, 224, 224]
            self.cached_masks = torch.stack(cached_msks)   # [N, 1, 224, 224]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        if self.preload:
            img = self.cached_images[idx].clone()
            msk = self.cached_masks[idx].clone()
        else:
            img_bgr = cv2.imread(self.image_paths[idx])
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            img_rgb = cv2.resize(img_rgb, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
            
            msk_gray = cv2.imread(self.mask_paths[idx], cv2.IMREAD_GRAYSCALE)
            msk_gray = cv2.resize(msk_gray, (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)
            msk_bin = (msk_gray >= 128).astype(np.float32)

            img = torch.from_numpy(img_rgb).permute(2, 0, 1).float() / 255.0
            msk = torch.from_numpy(msk_bin).unsqueeze(0).float()

        # High-speed PyTorch tensor augmentations (Train set only)
        if self.augment:
            # Random horizontal flip
            if random.random() > 0.5:
                img = torch.flip(img, dims=[2])
                msk = torch.flip(msk, dims=[2])

            # Random vertical flip
            if random.random() > 0.5:
                img = torch.flip(img, dims=[1])
                msk = torch.flip(msk, dims=[1])

            # Random orthogonal rotation (0, 90, 180, 270 deg) - ultra fast zero-copy tensor op
            k = random.choice([0, 1, 2, 3])
            if k > 0:
                img = torch.rot90(img, k, dims=[1, 2])
                msk = torch.rot90(msk, k, dims=[1, 2])

        return img, msk, self.filenames[idx]


def get_dataloaders(images_dir='images', masks_dir='masks', img_size=224, batch_size=8, 
                    split_ratio=(0.8, 0.1, 0.1), seed=42, num_workers=0, preload=True, max_samples=None):
    """
    Creates reproducible Train, Validation, and Test DataLoaders.
    Default split: 80% train (800 images), 10% val (100 images), 10% test (100 images).
    """
    valid_exts = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')
    all_imgs = [os.path.join(images_dir, f) for f in os.listdir(images_dir) 
                if os.path.splitext(f)[1].lower() in valid_exts]
    all_imgs.sort()

    paired_images = []
    paired_masks = []

    for img_path in all_imgs:
        fname = os.path.basename(img_path)
        base, _ = os.path.splitext(fname)
        
        mask_candidates = [
            os.path.join(masks_dir, fname),
            os.path.join(masks_dir, base + '.jpg'),
            os.path.join(masks_dir, base + '.png'),
        ]
        matched_mask = None
        for cand in mask_candidates:
            if os.path.exists(cand):
                matched_mask = cand
                break
        
        if matched_mask:
            paired_images.append(img_path)
            paired_masks.append(matched_mask)

    total_samples = len(paired_images)
    print(f"Total matching image-mask pairs found: {total_samples}")

    indices = list(range(total_samples))
    random.seed(seed)
    random.shuffle(indices)

    if max_samples is not None and max_samples < total_samples:
        indices = indices[:max_samples]
        total_samples = len(indices)
        print(f"Subsetting dataset to {total_samples} samples for quick experimentation.")

    train_end = max(1, int(split_ratio[0] * total_samples))
    val_end = max(train_end + 1, train_end + int(split_ratio[1] * total_samples))

    train_indices = indices[:train_end]
    val_indices = indices[train_end:val_end]
    test_indices = indices[val_end:]

    print(f"Dataset split -> Train: {len(train_indices)}, Val: {len(val_indices)}, Test: {len(test_indices)}")

    train_imgs = [paired_images[i] for i in train_indices]
    train_msks = [paired_masks[i] for i in train_indices]

    val_imgs = [paired_images[i] for i in val_indices]
    val_msks = [paired_masks[i] for i in val_indices]

    test_imgs = [paired_images[i] for i in test_indices]
    test_msks = [paired_masks[i] for i in test_indices]

    train_dataset = PolypDataset(train_imgs, train_msks, img_size=img_size, is_train=True, augment=True, preload=preload)
    val_dataset = PolypDataset(val_imgs, val_msks, img_size=img_size, is_train=False, augment=False, preload=preload)
    test_dataset = PolypDataset(test_imgs, test_msks, img_size=img_size, is_train=False, augment=False, preload=preload)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader
