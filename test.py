# test_only.py
import torch
from pathlib import Path
from unet import UNet
from train import validate, CONFIG, DEVICE, unpack 
import albumentations as A
from albumentations.pytorch import ToTensorV2
from dataSizeMap import DatasetSizeMap


import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
from PIL import Image
import time

# 

def save_sample_predictions(model, test_loader, output_dir, num_samples=10, cfg=CONFIG):
    """
    随机抽取样本并保存原始图像、标签和预测结果
    """
    model.eval()
    output_dir = Path(output_dir)
    samples_dir = output_dir / "sample_predictions"
    samples_dir.mkdir(parents=True, exist_ok=True)
    
    # 收集所有测试数据
    all_samples = []
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            imgs, masks = unpack(batch)
            imgs = imgs.to(DEVICE, dtype=torch.float32, memory_format=torch.channels_last)
            masks = masks.to(DEVICE, dtype=torch.float32)
            
            # 获取预测
            with torch.autocast(DEVICE.type if DEVICE.type != 'mps' else 'cpu', enabled=cfg["amp"]):
                logits = model(imgs)
                if cfg["num_classes"] == 1:
                    preds = torch.sigmoid(logits)
                else:
                    preds = F.softmax(logits, dim=1)
            
            # 将每个样本添加到列表中
            for i in range(imgs.size(0)):
                all_samples.append({
                    'image': imgs[i].cpu(),
                    'mask': masks[i].cpu(), 
                    'pred': preds[i].cpu(),
                    'batch_idx': batch_idx,
                    'sample_idx': i
                })
    
    # 随机选择样本
    if len(all_samples) > num_samples:
        selected_samples = random.sample(all_samples, num_samples)
    else:
        selected_samples = all_samples
    
    print(f"保存 {len(selected_samples)} 个样本到 {samples_dir}")
    
    # 保存每个选中的样本
    for idx, sample in enumerate(selected_samples):
        img = sample['image'].squeeze().numpy()  # [H, W]
        mask = sample['mask'].squeeze().numpy()  # [H, W] 
        pred = sample['pred'].squeeze().numpy()  # [H, W]
        
        # 正确的反归一化: x_original = x_normalized * std + mean
        # 因为归一化时: x_normalized = (x_original - mean) / std
        img_denorm = img * 0.5 + 0.5  # 从 [-1, 1] 恢复到 [0, 1]
        img_denorm = img_denorm * 255  # 转换到 [0, 255]
        img_denorm = np.clip(img_denorm, 0, 255).astype(np.uint8)
        
        # 将mask和prediction转换为0-255范围
        mask_vis = (mask * 255).astype(np.uint8)
        pred_vis = (pred * 255).astype(np.uint8)
        
        # 创建三联图
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        axes[0].imshow(img_denorm, cmap='gray')
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        axes[1].imshow(mask_vis, cmap='gray')
        axes[1].set_title('Ground Truth Mask')
        axes[1].axis('off')
        
        axes[2].imshow(pred_vis, cmap='gray')
        axes[2].set_title('Prediction')
        axes[2].axis('off')
        
        plt.tight_layout()
        
        # 保存组合图
        sample_name = f"sample_{idx:02d}_batch{sample['batch_idx']}_idx{sample['sample_idx']}"
        plt.savefig(samples_dir / f"{sample_name}_combined.png", dpi=150, bbox_inches='tight')
        plt.close()
        
        # 单独保存每个图像
        Image.fromarray(img_denorm).save(samples_dir / f"{sample_name}_image.png")
        Image.fromarray(mask_vis).save(samples_dir / f"{sample_name}_mask.png") 
        Image.fromarray(pred_vis).save(samples_dir / f"{sample_name}_pred.png")
    
    print(f"样本保存完成！文件保存在: {samples_dir}")

def run_test(ckpt_path: str, test_loader, cfg=CONFIG, save_samples=True):
    # 1) 读取权重
    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    # 用保存下来的 config 覆盖当前的（避免手滑不一致）
    if "config" in ckpt:
        cfg = {**cfg, **ckpt["config"]}

    # 2) 构建同结构模型并加载参数
    model = UNet(n_channels=1, n_classes=cfg["num_classes"], bilinear=cfg["bilinear"])
    model = model.to(DEVICE, memory_format=torch.channels_last)
    model.load_state_dict(ckpt["model"])
    model.eval()

    # 3) 计算推理帧率
    print("开始计算推理帧率...")
    model.eval()
    total_inference_time = 0.0
    total_samples = 0
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(test_loader, desc="推理帧率测试")):
            imgs, masks = unpack(batch)
            imgs = imgs.to(DEVICE, dtype=torch.float32, memory_format=torch.channels_last)
            batch_size = imgs.size(0)
            
            # 记录推理时间
            start_time = time.time()
            with torch.autocast(DEVICE.type if DEVICE.type != 'mps' else 'cpu', enabled=cfg["amp"]):
                logits = model(imgs)
            torch.cuda.synchronize() if DEVICE.type == 'cuda' else None  # 确保GPU计算完成
            end_time = time.time()
            
            total_inference_time += (end_time - start_time)
            total_samples += batch_size
    
    avg_inference_time_per_image = total_inference_time / total_samples
    fps = 1.0 / avg_inference_time_per_image
    
    print(f"[推理性能] 总样本数: {total_samples}")
    print(f"[推理性能] 总推理时间: {total_inference_time:.4f}s")
    print(f"[推理性能] 平均每张图片推理时间: {avg_inference_time_per_image:.4f}s")
    print(f"[推理性能] 推理帧率 (FPS): {fps:.2f}")

    # 4) 直接复用 validate()
    test_loss, test_dice = validate(model, test_loader, cfg)
    print(f"[TEST] loss: {test_loss:.4f} | dice: {test_dice:.4f}")
    
    # 5) 保存样本预测结果
    if save_samples:
        save_sample_predictions(model, test_loader, "outputs", num_samples=10, cfg=cfg)
    
    return model, (test_loss, test_dice)

if __name__ == "__main__":

    
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
    VAL = OUTPUT_DIR / "val.csv"
    TEST_DIR = OUTPUT_DIR / "test.csv"

    transform_test = A.Compose([
        A.Normalize(mean=(0.5,), std=(0.5,)),
        ToTensorV2(),
    ])

    test_df = pd.read_csv(TEST_DIR)
    test_ds = DatasetSizeMap(test_df, transform_test)
    test_loader = DataLoader(test_ds, batch_size=CONFIG["batch_size"], shuffle=False, num_workers=4)


    ckpt_path = "outputs/checkpoints/best_model.pth"
    run_test(ckpt_path, test_loader)
