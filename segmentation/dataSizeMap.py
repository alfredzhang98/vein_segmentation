
from pathlib import Path
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import sys
import albumentations as A
from albumentations.pytorch import ToTensorV2
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
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

class DatasetSizeMap(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform
        # 统计 image_width list种类
        print(f"image_width list种类: {self.df['image_width'].unique()}")
        # 统计 image_height list种类
        print(f"image_height list种类: {self.df['image_height'].unique()}")
        # 统计 micropixel list种类
        print(f"micropixel list种类: {self.df['micropixel'].unique()}")
        self.max_height = self.df['image_height'].max()
        self.max_width = self.df['image_width'].max()

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

        # Bmode image is gray picture
        image = Image.open(image_file_path).convert("L")
        mask  = Image.open(mask_file_path).convert("L")

        # To tensor
        if self.transform is not None:
            augmented = self.transform(image=np.array(image), mask=np.array(mask))
            image = augmented['image']
            mask  = augmented['mask'].float().div_(255.0)
        else:
            image = ToTensorV2()(image=np.array(image))['image']
            mask = ToTensorV2()(image=np.array(mask))['image']
            mask = mask.float().div_(255.0)

        dh = self.max_height - image.shape[1]
        dw = self.max_width - image.shape[2]
        if dh or dw:
            image = F.pad(image, (0, dw, 0, dh), value=0)
            mask  = F.pad(mask,  (0, dw, 0, dh), value=0)

        H2, W2 = image.shape[1], image.shape[2]
        dh2 = (-H2) % 32
        dw2 = (-W2) % 32
        if dh2 or dw2:
            image = F.pad(image, (0, dw2, 0, dh2), value=0)
            mask  = F.pad(mask,  (0, dw2, 0, dh2), value=0)

        if mask.ndim == 2:
            mask = mask.unsqueeze(0)

        return image, mask
    
def denorm(img, mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5)):
    # img: [C,H,W]
    mean = torch.tensor(mean).view(3,1,1)
    std  = torch.tensor(std).view(3,1,1)
    return img * std + mean

def unpack(batch):
    if isinstance(batch, dict):
        return batch['image'], batch['mask']
    return batch[0], batch[1]

def debug_data_range(loader, name="data"):
    print(f"\n=== {name} Debug Info ===")
    for i, batch in enumerate(loader):
        imgs, masks = unpack(batch)
        
        print(f"Batch {i}:")
        print(f"  Images - Shape: {imgs.shape}, Min: {imgs.min():.4f}, Max: {imgs.max():.4f}")
        print(f"  Masks  - Shape: {masks.shape}, Min: {masks.min():.4f}, Max: {masks.max():.4f}")
        print(f"  Mask unique values: {torch.unique(masks)}")
        
        if i >= 2:  # 只检查前几个batch
            break
    print("=" * 30)

# main
if __name__ == "__main__":

    transform_train = A.Compose([
        A.Normalize(mean=(0.5,), std=(0.5,)),
        ToTensorV2(),
    ])

    train_df = pd.read_csv(TRAIN_DIR)
    train_ds = DatasetSizeMap(train_df, transform_train)

    image, mask = train_ds[111]
    print(f"Image type: {type(image)}, Mask type: {type(mask)}")
    print(f"Image shape: {image.shape}, Mask shape: {mask.shape}")

    train_data = DataLoader(train_ds, batch_size=8, shuffle=False, num_workers=0)
    val_loader = DataLoader(train_ds, batch_size=8, shuffle=False, num_workers=0)

    debug_data_range(train_data, "Train")
    debug_data_range(val_loader, "Validation")

    # image_vis = denorm(image).permute(1,2,0).cpu().numpy()
    # mask_vis  = mask.squeeze().cpu().numpy()

    # plt.figure(figsize=(10,4))
    # plt.subplot(1,2,1); plt.imshow(image_vis); plt.title("Image"); plt.axis('off')
    # plt.subplot(1,2,2); plt.imshow(mask_vis, cmap='gray'); plt.title("Mask"); plt.axis('off')
    # plt.tight_layout()
    # plt.show()

    # 数据加载
    # train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=4)
    # val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=4)
    # test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=4)