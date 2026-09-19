"""
Shared project configuration.
All paths are resolved against the repository root, two levels above this file.
"""

import csv
from pathlib import Path

# Project root = two levels up from data/collection/
PROJECT_ROOT = Path(__file__).resolve().parents[2]


class DataInfo:
    def __init__(self, test_name="phantom_taobao", test_id="1"):
        self.test_id = test_id
        self.test_name = test_name

        self.data_dir = PROJECT_ROOT / "data" / "datasets"

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
        # What label.py paints. Training reads these PNGs through
        # data/pipeline/dataPrepare.py, whose per-dataset `mask_class_map` maps 255 to the
        # shared id space (0 bg / 1 vein / 2 artery / 3 vessel-untyped). Keep the two
        # in step: a value painted here that no mask_class_map mentions is dropped.
        self.mask_class = {"background": 0, "vessel": 255}
        self.mask_class_map = {0: 0, 255: 1}

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

    def ensure_dirs(self):
        """Create required directories and meta CSV. Call before writing data."""
        for d in (self.base_dir, self.images_dir, self.masks_dir,
                  self.images_aug_dir, self.masks_aug_dir):
            d.mkdir(parents=True, exist_ok=True)

        if not self.meta_file.exists():
            with open(self.meta_file, 'w', newline='') as f:
                csv.writer(f).writerow([
                    "id", "filename", "relative_path",
                    "raw_timestamp", "timestamp",
                    "depth", "gain", "frequency",
                    "bpp", "image_width", "image_height", "micropixel",
                    "test_name", "test_id", "mask_status"])
