import os
import sys
import random
from pathlib import Path

# 限制线程数，防止共享服务器上 fork/内存分配失败
for _k in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_k, '1')

import cv2
import numpy as np
import pandas as pd
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader

from DataInfo import DataInfo


# ---------------------------------------------------------------------------
# Per-dataset configs — all augmentation and path parameters baked in here
# ---------------------------------------------------------------------------

DATASET_CONFIGS = {
    "phantom_1": {
        "source_type":     "csv",
        "data_dir":        Path("data/phantom_1"),
        "npz_prefix":      "augmented_phantom_1_1",
        "aug_times_train": 20,
        "mask_class_map":  {0: 0, 255: 1},
        "target_size":     (576, 544),      # (H, W)
        "color_mode":      "gray",
        "crop":            (10, 15, 50, 50),  # top, bottom, left, right pixels to remove
        "seed":            42,
    },
    "dataset1": {
        "source_type":     "folder",
        "data_dir":        Path("data/dataset1/Common Carotid Artery Ultrasound Images"),
        "images_dir":      Path("data/dataset1/Common Carotid Artery Ultrasound Images/US images"),
        "masks_dir":       Path("data/dataset1/Common Carotid Artery Ultrasound Images/Expert mask images"),
        "npz_out_dir":     Path("data/dataset1"),
        "npz_prefix":      "augmented_dataset1_1",
        "aug_times_train": 5,
        "mask_class_map":  {0: 0, 255: 1},
        "target_size":     (576, 544),
        "color_mode":      "rgb_to_gray",   # load RGB then convert
        "crop":            None,
        "seed":            42,
    },
    "phantom_2": {
        "source_type":     "csv",
        "data_dir":        Path("data/phantom_2"),
        "meta_file":       Path("data/phantom_2/meta_phantom_2_1.csv"),
        "npz_prefix":      "augmented_phantom_2_1",
        "aug_times_train": 20,
        "mask_class_map":  {0: 0, 255: 1},
        "target_size":     (576, 544),
        "color_mode":      "gray",
        "crop":            None,
        "seed":            42,
        # --- augmentation overrides ---
        "aug_rotation_limit": 10,
        "aug_hflip":          True,
        "aug_vflip":          True,
        "aug_brightness":     (-0.1, 0.1),
        "aug_contrast":       (-0.1, 0.1),
        "aug_bc_prob":        0.5,
        "aug_noise_prob":     0.3,
    },
}


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def read_npz_file(npz_file: Path) -> tuple:
    """Read NPZ file and return (images, masks, image_type)."""
    if not Path(npz_file).exists():
        raise FileNotFoundError(f"NPZ file not found: {npz_file}")
    data = np.load(npz_file, allow_pickle=True)
    return data['images'], data['masks'], data['image_type']


# ---------------------------------------------------------------------------
# DataPipeline  —  correct order: split first, then augment
# ---------------------------------------------------------------------------

class DataPipeline(DataInfo):
    """
    Correct pipeline to avoid data leakage:

      1. Load rows (from CSV or folder scan depending on dataset).
      2. Split ORIGINAL images into train / val / test (no augmentation yet).
      3. Train split  → full random augmentation × aug_times_train per image.
      4. Val / Test   → deterministic resize + normalize only (each image used once).
      5. Save PNG files and NPZ files for each split.

    Usage:
        DataPipeline("phantom_1").run()
        DataPipeline("dataset1").run(save_png=False)
    """

    def __init__(self, dataset_name: str = "phantom_1",
                 train_ratio: float = 0.7,
                 val_ratio: float = 0.15,
                 test_ratio: float = 0.15):

        if dataset_name not in DATASET_CONFIGS:
            raise ValueError(f"Unknown dataset '{dataset_name}'. "
                             f"Available: {list(DATASET_CONFIGS)}")
        self.dataset_name = dataset_name
        self.cfg = DATASET_CONFIGS[dataset_name]

        super().__init__()  # builds DataInfo (phantom paths); key attrs overwritten below

        # Override meta_file for datasets that specify their own CSV path
        if "meta_file" in self.cfg:
            self.meta_file = self.cfg["meta_file"]

        self.train_ratio = train_ratio
        self.val_ratio   = val_ratio
        self.test_ratio  = test_ratio

        # Override DataInfo attrs with dataset-specific values
        self.target_size      = self.cfg["target_size"]          # (H, W)
        self.mask_class_map   = self.cfg["mask_class_map"]
        self.seed             = self.cfg["seed"]
        self.target_aug_times = {"true": self.cfg["aug_times_train"]}  # keep _process_split compat

        # NPZ output paths
        npz_out = self.cfg.get("npz_out_dir", self.cfg["data_dir"])
        prefix  = self.cfg["npz_prefix"]
        self.train_npz = npz_out / f"{prefix}_train.npz"
        self.val_npz   = npz_out / f"{prefix}_val.npz"
        self.test_npz  = npz_out / f"{prefix}_test.npz"

        # PNG output dirs (overwrite DataInfo's phantom-specific dirs)
        base = self.cfg["data_dir"]
        self.images_aug_train_dir = base / "images_aug" / "train"
        self.images_aug_val_dir   = base / "images_aug" / "val"
        self.images_aug_test_dir  = base / "images_aug" / "test"
        self.masks_aug_train_dir  = base / "masks_aug"  / "train"
        self.masks_aug_val_dir    = base / "masks_aug"  / "val"
        self.masks_aug_test_dir   = base / "masks_aug"  / "test"
        for d in [self.images_aug_train_dir, self.images_aug_val_dir, self.images_aug_test_dir,
                  self.masks_aug_train_dir,  self.masks_aug_val_dir,  self.masks_aug_test_dir]:
            d.mkdir(parents=True, exist_ok=True)

        np.random.seed(self.seed)
        random.seed(self.seed)

        # ------------------------------------------------------------------
        # Train: full random augmentation
        # ------------------------------------------------------------------
        cfg = self.cfg
        self.train_transform = A.Compose([
            # Upscale 1.5× before rotation to avoid black corners
            A.Resize(int(self.target_size[0] * 1.5), int(self.target_size[1] * 1.5),
                     interpolation=cv2.INTER_CUBIC, p=1.0),
            A.Rotate(limit=cfg.get("aug_rotation_limit", 10), p=0.5,
                     interpolation=cv2.INTER_CUBIC,
                     border_mode=cv2.BORDER_REPLICATE, fill=0),
            A.HorizontalFlip(p=0.5 if cfg.get("aug_hflip", True) else 0.0),
            A.VerticalFlip(p=0.5 if cfg.get("aug_vflip", True) else 0.0),
            A.Resize(self.target_size[0], self.target_size[1],
                     interpolation=cv2.INTER_AREA, p=1.0),
            A.RandomBrightnessContrast(
                brightness_limit=cfg.get("aug_brightness", (-0.1, 0.1)),
                contrast_limit=cfg.get("aug_contrast", (-0.1, 0.1)),
                p=cfg.get("aug_bc_prob", 0.6)),
            A.GaussNoise(std_range=(0.05, 0.1), p=cfg.get("aug_noise_prob", 0.3)),
        ], additional_targets={'mask': 'mask'})

        # ------------------------------------------------------------------
        # Val / Test: deterministic resize only (no randomness)
        # ------------------------------------------------------------------
        self.eval_transform = A.Compose([
            A.Resize(self.target_size[0], self.target_size[1],
                     interpolation=cv2.INTER_AREA, p=1.0),
        ], additional_targets={'mask': 'mask'})

        # Normalize + to tensor (shared by all splits)
        self.final_transform = A.Compose([
            A.Normalize(mean=[0.5], std=[0.5], p=1.0),
            ToTensorV2(),
        ], additional_targets={'mask': 'mask'})

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _remap_mask(self, mask: np.ndarray) -> np.ndarray:
        result = np.zeros_like(mask, dtype=np.uint8)
        for k, v in self.mask_class_map.items():
            result[mask == k] = v
        return result

    def _load_and_crop(self, image_path: Path, mask_path: Path) -> tuple:
        color_mode = self.cfg["color_mode"]
        if color_mode == "gray":
            image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
            if image is None:
                raise ValueError(f"Failed to read image: {image_path}")
        elif color_mode == "rgb_to_gray":
            bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if bgr is None:
                raise ValueError(f"Failed to read image: {image_path}")
            image = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        else:
            raise ValueError(f"Unknown color_mode: {color_mode}")

        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise ValueError(f"Failed to read mask: {mask_path}")
        mask = self._remap_mask(mask)

        crop = self.cfg["crop"]
        if crop is not None:
            top, bottom, left, right = crop
            image = image[top: image.shape[0] - bottom, left: image.shape[1] - right]
            mask  = mask [top: mask.shape[0]  - bottom, left: mask.shape[1]  - right]

        return image, mask

    def _load_rows_from_folder(self) -> list:
        """Scan image/mask folder pair, return rows compatible with _process_split."""
        images_dir = self.cfg["images_dir"]
        masks_dir  = self.cfg["masks_dir"]
        rows = []
        for img_path in sorted(images_dir.glob("*.png")):
            mask_path = masks_dir / img_path.name
            if not mask_path.exists():
                continue
            rows.append({
                "filename":      img_path.name,
                "relative_path": str(img_path),
                "mask_path":     str(mask_path),
                "mask_status":   "true",
            })
        if not rows:
            raise FileNotFoundError(f"No paired PNG files found in {images_dir}")
        return rows

    def _apply_and_tensorize(self, image: np.ndarray, mask: np.ndarray,
                              is_train: bool) -> tuple:
        """Return (img_tensor [1,H,W], msk_tensor [1,H,W]) as numpy arrays."""
        transform = self.train_transform if is_train else self.eval_transform
        out = transform(image=image, mask=mask)
        final = self.final_transform(image=out['image'], mask=out['mask'])

        img_t = final['image'].numpy()   # (C, H, W)  float32
        msk_t = final['mask'].numpy()    # (H, W)     uint8

        if img_t.ndim == 2:
            img_t = img_t[np.newaxis]
        if msk_t.ndim == 2:
            msk_t = msk_t[np.newaxis]

        assert img_t.shape == (1, *self.target_size), f"img shape error: {img_t.shape}"
        assert msk_t.shape == (1, *self.target_size), f"msk shape error: {msk_t.shape}"
        return img_t, msk_t

    def _resolve_paths(self, row: dict) -> tuple:
        if sys.platform.startswith('linux'):
            img_path  = Path(row['relative_path'].replace('\\', '/'))
            mask_path = Path(row['mask_path'].replace('\\', '/'))
        else:
            img_path  = Path(row['relative_path'])
            mask_path = Path(row['mask_path'])
        return img_path, mask_path

    def _process_split(self, rows: list, split_name: str) -> tuple:
        """
        Process all original images for one split.
        - train: each image repeated aug_times with random augmentation
        - val / test: each image processed once, deterministic
        """
        is_train  = (split_name == 'train')
        aug_times = self.target_aug_times.get('true', 20) if is_train else 1

        all_images, all_masks, all_types, all_filenames = [], [], [], []

        for row in tqdm(rows, desc=f"  [{split_name}] processing"):
            img_path, mask_path = self._resolve_paths(row)
            image, mask = self._load_and_crop(img_path, mask_path)
            stem = row['filename'].rsplit('.', 1)[0]

            for i in range(aug_times):
                img_t, msk_t = self._apply_and_tensorize(image, mask, is_train)
                all_images.append(img_t)
                all_masks.append(msk_t)
                all_types.append(row['mask_status'])
                fname = f"{stem}_{i:04d}" if is_train else stem
                all_filenames.append(fname)

        return all_images, all_masks, all_types, all_filenames

    def _save_split(self, images: list, masks: list,
                    image_type: list, filenames: list,
                    npz_path: Path, img_dir: Path, mask_dir: Path,
                    split_name: str, save_png: bool) -> None:

        images_arr = np.array(images,     dtype=np.float32)
        masks_arr  = np.array(masks,      dtype=np.uint8)
        type_arr   = np.array(image_type, dtype='<U10')

        # Validate
        assert images_arr.min() >= -1.01 and images_arr.max() <= 1.01
        assert masks_arr.min()  >= 0     and masks_arr.max()  <= 1

        np.savez(
            npz_path,
            images=images_arr,
            masks=masks_arr,
            image_type=type_arr,
            metadata=np.array([{
                'split': split_name,
                'dataset': self.dataset_name,
                'target_size': self.target_size,
                'num_samples': len(images_arr),
                'creation_time': pd.Timestamp.now().isoformat(),
                'aug_config': self.target_aug_times,
            }], dtype=object)
        )
        size_mb = npz_path.stat().st_size / (1024 ** 2)
        print(f"    NPZ saved: {size_mb:.1f} MB  ({len(images_arr)} samples)")

        if save_png:
            for i, fname in enumerate(tqdm(filenames, desc=f"    saving PNGs", leave=False)):
                img_hw = images_arr[i, 0]
                msk_hw = masks_arr[i, 0]
                # Denormalize [-1,1] → [0,255]
                if img_hw.min() < -0.1:
                    img_hw = (img_hw + 1.0) / 2.0
                img_hw = (img_hw * 255).clip(0, 255).astype(np.uint8)
                msk_hw = (msk_hw * 255).astype(np.uint8)
                cv2.imwrite(str(img_dir  / f"{fname}.png"),      img_hw)
                cv2.imwrite(str(mask_dir / f"{fname}_mask.png"), msk_hw)

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(self, save_png: bool = True) -> None:
        """Main entry: load rows → split originals → augment → save NPZ."""
        source_type = self.cfg["source_type"]

        if source_type == "csv":
            if not self.meta_file.exists():
                raise FileNotFoundError(f"Metadata file not found: {self.meta_file}")
            df = pd.read_csv(self.meta_file)
            true_df = df[df['mask_status'].astype(str).str.lower() == 'true'].copy()
            true_df = true_df.dropna(subset=['mask_path'])
            true_df = true_df[true_df['mask_path'].apply(lambda x: isinstance(x, str))]
            true_df = true_df.sample(frac=1, random_state=self.seed).reset_index(drop=True)
            all_rows = true_df.to_dict('records')

        elif source_type == "folder":
            all_rows = self._load_rows_from_folder()
            random.shuffle(all_rows)  # seed already set in __init__

        else:
            raise ValueError(f"Unknown source_type: {source_type}")

        n       = len(all_rows)
        n_train = int(n * self.train_ratio)
        n_val   = int(n * self.val_ratio)
        train_rows = all_rows[:n_train]
        val_rows   = all_rows[n_train: n_train + n_val]
        test_rows  = all_rows[n_train + n_val:]

        aug_times = self.target_aug_times.get('true', 20)
        print(f"Dataset: {self.dataset_name}  |  Original images: {n}")
        print(f"Split → train: {len(train_rows)} (×{aug_times} aug)  "
              f"val: {len(val_rows)} (×1)  "
              f"test: {len(test_rows)} (×1)")

        splits = [
            ('train', train_rows, self.train_npz,
             self.images_aug_train_dir, self.masks_aug_train_dir),
            ('val',   val_rows,   self.val_npz,
             self.images_aug_val_dir,   self.masks_aug_val_dir),
            ('test',  test_rows,  self.test_npz,
             self.images_aug_test_dir,  self.masks_aug_test_dir),
        ]

        for split_name, rows, npz_path, img_dir, mask_dir in splits:
            print(f"\n[{split_name}]")
            images, masks, types, filenames = self._process_split(rows, split_name)
            self._save_split(images, masks, types, filenames,
                             npz_path, img_dir, mask_dir, split_name, save_png)

        print(f"\nDone.")
        print(f"  train : {len(train_rows)} original → {len(train_rows) * aug_times} samples")
        print(f"  val   : {len(val_rows)}   original → {len(val_rows)} samples")
        print(f"  test  : {len(test_rows)}  original → {len(test_rows)} samples")


# ---------------------------------------------------------------------------
# ReadDataset  —  reads from pre-built NPZ, used by train.py / test.py
# ---------------------------------------------------------------------------

class ReadDataset(Dataset):
    """Load a pre-built NPZ split for training / evaluation."""

    def __init__(self, dataset_type: str = 'train',
                 batch_size: int = 8,
                 dataset_name: str = "phantom_1"):
        super().__init__()
        self.dataset_type = dataset_type
        self.batch_size   = batch_size
        self.dataset_name = dataset_name

        if dataset_name not in DATASET_CONFIGS:
            raise ValueError(f"Unknown dataset '{dataset_name}'. "
                             f"Available: {list(DATASET_CONFIGS)}")

        cfg     = DATASET_CONFIGS[dataset_name]
        npz_out = cfg.get("npz_out_dir", cfg["data_dir"])
        prefix  = cfg["npz_prefix"]
        path_map = {
            'train':      npz_out / f"{prefix}_train.npz",
            'validation': npz_out / f"{prefix}_val.npz",
            'test':       npz_out / f"{prefix}_test.npz",
        }
        if dataset_type not in path_map:
            raise ValueError(f"Invalid dataset_type '{dataset_type}'. "
                             f"Choose from {list(path_map)}")
        self.data_path = path_map[dataset_type]

        self.images, self.masks, self.image_type = read_npz_file(self.data_path)
        print(f"Loaded '{self.dataset_type}' [{dataset_name}]: {len(self.images)} samples")
        unique_types, counts = np.unique(self.image_type, return_counts=True)
        for t, c in zip(unique_types, counts):
            print(f"  {t}: {c} ({c / len(self.image_type) * 100:.1f}%)")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if idx >= len(self.images):
            raise IndexError(f"Index {idx} out of range ({len(self.images)})")
        image_tensor = torch.from_numpy(self.images[idx]).float()   # (1, H, W)
        mask_tensor  = torch.from_numpy(self.masks[idx]).float()    # (1, H, W)
        assert image_tensor.ndim == 3 and image_tensor.shape[0] == 1
        assert mask_tensor.ndim  == 3 and mask_tensor.shape[0]  == 1
        return image_tensor, mask_tensor, self.image_type[idx]

    def get_dataloader(self, shuffle: bool = None,
                       num_workers: int = None,
                       drop_last: bool = False) -> DataLoader:
        if shuffle is None:
            shuffle = (self.dataset_type == 'train')
        if num_workers is None:
            num_workers = 0 if sys.platform.startswith('win') else 4
        return DataLoader(
            self,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            drop_last=drop_last,
            pin_memory=torch.cuda.is_available(),
        )

    def get_class_weights(self) -> dict:
        unique_types, counts = np.unique(self.image_type, return_counts=True)
        total = len(self.image_type)
        return {t: total / (len(unique_types) * c)
                for t, c in zip(unique_types, counts)}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="phantom_1",
                        choices=list(DATASET_CONFIGS),
                        help="Which dataset to process")
    parser.add_argument("--no-png", action="store_true",
                        help="Skip saving PNG files (faster)")
    parser.add_argument("--test-loader", action="store_true",
                        help="Test the DataLoader after processing")
    args = parser.parse_args()

    pipeline = DataPipeline(args.dataset)
    pipeline.run(save_png=not args.no_png)

    if args.test_loader:
        print("=" * 50)
        print("Testing DataLoader...")
        ds = ReadDataset(dataset_type='train', batch_size=32, dataset_name=args.dataset)
        loader = ds.get_dataloader()
        for i, (imgs, msks, labels) in enumerate(loader):
            print(f"Batch {i+1}: images={imgs.shape}, masks={msks.shape}")
            if i >= 2:
                break
        print("DataLoader test OK.")
