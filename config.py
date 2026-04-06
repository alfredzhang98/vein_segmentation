"""
Shared project configuration.
All paths are relative to the project root directory (where this file lives).
"""

import csv
from pathlib import Path

# Project root = directory containing this file
PROJECT_ROOT = Path(__file__).resolve().parent


class DataInfo:
    def __init__(self, test_name="phantom_taobao", test_id="1"):
        self.test_id = test_id
        self.test_name = test_name

        self.data_dir = PROJECT_ROOT / "data"
        self.checkpoints_dir = PROJECT_ROOT / "checkpoints"

        self.base_dir       = self.data_dir / self.test_name
        self.images_dir     = self.base_dir / "images"
        self.images_aug_dir = self.base_dir / "images_aug"
        self.masks_dir      = self.base_dir / "masks"
        self.masks_aug_dir  = self.base_dir / "masks_aug"
        self.meta_file      = self.base_dir / f"meta_{self.test_name}_{self.test_id}.csv"

        self.target_aug_times = {
            "true": 20,
            # "test": 1
        }
        self.mask_class = {"background": 0, "vein": 255, "nerve": 128}
        self.mask_class_map = {0: 0, 255: 1, 128: 2}

        # UNet input size
        self.max_height = 576
        self.max_width = 544

        # Data augmentation
        self.seed = 42

        # NPZ files
        self.all_npz   = self.base_dir / f"augmented_{self.test_name}_{self.test_id}.npz"
        self.train_npz = self.base_dir / f"augmented_{self.test_name}_{self.test_id}_train.npz"
        self.val_npz   = self.base_dir / f"augmented_{self.test_name}_{self.test_id}_val.npz"
        self.test_npz  = self.base_dir / f"augmented_{self.test_name}_{self.test_id}_test.npz"

        self._ensure_dirs()

    def _ensure_dirs(self):
        for d in (self.base_dir, self.images_dir, self.masks_dir,
                  self.images_aug_dir, self.masks_aug_dir, self.checkpoints_dir):
            d.mkdir(parents=True, exist_ok=True)

        if not self.meta_file.exists():
            with open(self.meta_file, 'w', newline='') as f:
                csv.writer(f).writerow([
                    "id", "filename", "relative_path",
                    "raw_timestamp", "timestamp",
                    "depth", "gain", "frequency",
                    "bpp", "image_width", "image_height", "micropixel",
                    "test_name", "test_id", "mask_status"])
