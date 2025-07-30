import csv
import sys
from pathlib import Path

class DataInfo:
    def __init__(self):
        self.test_id = "1"
        self.test_name = "phantom_taobao"
        self.data_dir = Path("data")
        self.output_dir = Path("outputs")
        self.checkpoints_dir = self.output_dir / "checkpoints"
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
        self.mask_class_map= {0: 0, 255: 1, 128: 2}

        # This is the force resize size for the images and masks
        self.max_height = 576
        self.max_width = 544

        # data augmentation parameters
        self.seed = 42  

        # all data npz files
        self.all_npz = self.base_dir / f"augmented_{self.test_name}_{self.test_id}.npz"
        self.train_npz = self.base_dir / f"augmented_{self.test_name}_{self.test_id}_train.npz"
        self.val_npz = self.base_dir / f"augmented_{self.test_name}_{self.test_id}_val.npz"
        self.test_npz = self.base_dir / f"augmented_{self.test_name}_{self.test_id}_test.npz"

        # check configuration paths
        self.__check_config_paths()

    def __check_config_paths(self):
        if not self.base_dir.exists():
            # Create base directory if it does not exist
            self.base_dir.mkdir(parents=True, exist_ok=True)
        if not self.images_dir.exists():
            # Create images directory if it does not exist
            self.images_dir.mkdir(parents=True, exist_ok=True)
        if not self.masks_dir.exists():
            # Create masks directory if it does not exist
            self.masks_dir.mkdir(parents=True, exist_ok=True)
        if not self.images_aug_dir.exists():
            # Create augmented images directory if it does not exist
            self.images_aug_dir.mkdir(parents=True, exist_ok=True)
        if not self.checkpoints_dir.exists():
            # Create checkpoints directory if it does not exist
            self.checkpoints_dir.mkdir(parents=True, exist_ok=True)
        if not self.masks_aug_dir.exists():
            # Create augmented masks directory if it does not exist
            self.masks_aug_dir.mkdir(parents=True, exist_ok=True)
        if not self.meta_file.exists():
            with open(self.meta_file, 'w', newline='') as f:
                csv.writer(f).writerow(["id","filename","relative_path",
                                        "raw_timestamp","timestamp",
                                        "depth","gain","frequency",
                                        "bpp", "image_width","image_height", "micropixel",
                                        "test_name", "test_id", "mask_status"])