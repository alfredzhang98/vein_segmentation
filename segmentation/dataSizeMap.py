
from pathlib import Path
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF
from PIL import Image
import sys
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
import torch
import torch.nn.functional as F

test_id = "1"
test_name = "phantom_taobao"
BASE_DIR = Path(f"data/{test_name}")
CSV_PATH = BASE_DIR / f"meta_{test_name}_{test_id}_augmented.csv"
IMAGES_DIR = BASE_DIR / "images"
IMAGES_AUG_DIR = BASE_DIR / "images_augmented"
MASKS_DIR = BASE_DIR / "masks"
MASK_AUG_DIR = BASE_DIR / "masks_augmented"
OUTPUT_DIR = Path("outputs")
TRAIN_DIR = OUTPUT_DIR / "train.csv"
VALIDATION_DIR = OUTPUT_DIR / "validation.csv"
TEST_DIR = OUTPUT_DIR / "test.csv"
MAX_HEIGHT = 576
MAX_WIDTH = 544

class DatasetSizeMap(Dataset):
    def __init__(self, df, transform = A.Compose([
        A.Normalize(mean=(0.5,), std=(0.5,)),
        ToTensorV2()
    ]), max_height = MAX_HEIGHT, max_width = MAX_WIDTH):
        self.df = df
        self.transform = transform
        # This should be 576*544 for the dataset, becasue it could be devided by 32 and also it is larger then max image size
        self.max_height = max_height
        self.max_width = max_width

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_file_path = row['relative_path']
        mask_file_path = row['mask_path']

        # if the system is linux, convert \\ to / 
        if sys.platform.startswith("linux"):
            image_file_path = image_file_path.replace("\\", "/")
            mask_file_path = mask_file_path.replace("\\", "/")

        # Bmode image is gray picture so we just use the one channel to keep the gray channel
        image = Image.open(image_file_path).convert("L")
        mask  = Image.open(mask_file_path).convert("L")

        # To tensor we use the transform pipeline convert the image and mask to tensor
        augmented = self.transform(image=np.array(image), mask=np.array(mask))
        image = augmented['image']
        # normalize the mask to [0, 1] range
        mask  = augmented['mask'].float().div_(255.0)

        # Resize the image to the max height and width
        image = self.resize_and_pad(image, self.max_height, self.max_width, is_mask=False)
        mask  = self.resize_and_pad(mask,  self.max_height, self.max_width, is_mask=True)

        return image, mask

    def resize_and_pad(self, image, target_height, target_width, is_mask=False):
        # Handle different tensor shapes
        if len(image.shape) == 2:  # (H, W) - typical for masks
            h, w = image.shape
            image = image.unsqueeze(0)  # Add channel dimension: (H, W) -> (1, H, W)
        elif len(image.shape) == 3:  # (C, H, W) - typical for images
            _, h, w = image.shape
        else:
            raise ValueError(f"Unexpected tensor shape: {image.shape}")
        
        scale = min(target_height / h, target_width / w)
        new_h, new_w = int(h * scale), int(w * scale)

        # Resize
        if is_mask:
            # For masks, use nearest interpolation without align_corners
            image = F.interpolate(
                image.unsqueeze(0),
                size=(new_h, new_w),
                mode='nearest'
            ).squeeze(0)
        else:
            # For images, use bilinear interpolation with align_corners
            image = F.interpolate(
                image.unsqueeze(0),
                size=(new_h, new_w),
                mode='bilinear',
                align_corners=False
            ).squeeze(0)

        # Calculate centered padding
        pad_h = target_height - new_h
        pad_w = target_width - new_w
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left

        # Pad the image with 0 means black
        pad_value = 0 
        image = F.pad(image, (pad_left, pad_right, pad_top, pad_bottom), value=pad_value)
        
        # For masks, we want to keep the channel dimension for consistency
        # For images, they already have the channel dimension
        return image


    @staticmethod
    def denorm_image(img_tensor, mean=(0.5,), std=(0.5,)):
        # img_tensor: [C,H,W]
        mean = torch.tensor(mean).view(1,1,1)
        std  = torch.tensor(std).view(1,1,1)
        img = img_tensor * std + mean
        img = (img * 255).clamp(0, 255)
        return img.byte()
    
    @staticmethod
    def denorm_mask(mask_tensor):
        # mask_tensor: [C,H,W]
        mask = (mask_tensor * 255).clamp(0, 255)
        return mask.byte()

    @staticmethod
    def unpack(batch):
        if isinstance(batch, dict):
            return batch['image'], batch['mask']
        return batch[0], batch[1]

    @staticmethod
    def debug_data_range(loader, name="data"):
        print(f"\n=== {name} Debug Info ===")
        for i, batch in enumerate(loader):
            imgs, masks = DatasetSizeMap.unpack(batch)
            print(f"Batch {i}:")
            print(f"  Images - Shape: {imgs.shape}, Min: {imgs.min():.4f}, Max: {imgs.max():.4f}")
            print(f"  Masks  - Shape: {masks.shape}, Min: {masks.min():.4f}, Max: {masks.max():.4f}")
            print(f"  Mask unique values: {torch.unique(masks)}")
            break  # Only check the first batch for debugging

# main
if __name__ == "__main__":

    train_df = pd.read_csv(TRAIN_DIR)
    train_ds = DatasetSizeMap(train_df)
    train_data = DataLoader(train_ds, batch_size=8, shuffle=False, num_workers=0)
    DatasetSizeMap.debug_data_range(train_data, "Train")

    test, test2 =  train_ds[10]

    # print the data info
    print(f"Test Image Shape: {test.shape}, Min: {test.min():.4f}, Max: {test.max():.4f}")
    print(f"Test Mask Shape: {test2.shape}, Min: {test2.min():.4f}, Max: {test2.max():.4f}")
    print(f"Test Image Unique Values: {torch.unique(test)}")
    print(f"Test Mask Unique Values: {torch.unique(test2)}")

    # 数据加载
    # train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=4)
    # val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=4)
    # test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=4)