# test_only.py
import torch
from pathlib import Path
from unet import UNet
from train import validate, CONFIG, DEVICE
from DataInfo import DataInfo
from dataPrepare import ReadDataset

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
import sys

def save_sample_predictions(model, test_loader, output_dir, num_samples=10, cfg=CONFIG):
    """
    随机抽取样本并保存原始图像、标签和预测结果
    
    Args:
        model: 训练好的模型
        test_loader: 测试数据加载器
        output_dir: 输出目录
        num_samples: 保存的样本数量
        cfg: 配置字典
    """
    model.eval()
    output_dir = Path(output_dir)
    samples_dir = output_dir / "sample_predictions"
    samples_dir.mkdir(parents=True, exist_ok=True)
    
    # 收集所有测试数据
    all_samples = []
    print("正在收集测试样本...")
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(test_loader, desc="收集样本", leave=False)):
            try:
                imgs, masks, labels = batch  # ReadDataset 返回 (imgs, masks, labels)
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
                        'image': imgs[i].float().cpu(),  # 转换为float32避免BFloat16问题
                        'mask': masks[i].float().cpu(), 
                        'pred': preds[i].float().cpu(),
                        'batch_idx': batch_idx,
                        'sample_idx': i
                    })
                    
            except Exception as e:
                print(f"处理批次 {batch_idx} 时出错: {e}")
                continue
    
    if len(all_samples) == 0:
        print("未能收集到任何样本！")
        return
    
    # 随机选择样本
    if len(all_samples) > num_samples:
        selected_samples = random.sample(all_samples, num_samples)
    else:
        selected_samples = all_samples
    
    print(f"保存 {len(selected_samples)} 个样本到 {samples_dir}")
    
    # 保存每个选中的样本
    for idx, sample in enumerate(tqdm(selected_samples, desc="保存样本")):
        try:
            # 转换为float32避免BFloat16类型错误
            img = sample['image'].float().squeeze().numpy()  # [H, W]
            mask = sample['mask'].float().squeeze().numpy()  # [H, W] 
            pred = sample['pred'].float().squeeze().numpy()  # [H, W]
            
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
            
        except Exception as e:
            print(f"保存样本 {idx} 时出错: {e}")
            continue
    
    print(f"样本保存完成！文件保存在: {samples_dir}")

def run_test(ckpt_path: str, test_loader, datainfo, cfg=CONFIG, save_samples=True):
    # 添加datainfo到配置中以便在保存样本时使用
    cfg = {**cfg, "datainfo": datainfo}
    
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
            imgs, masks, labels = batch  # ReadDataset 返回 (imgs, masks, labels)
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
        save_sample_predictions(model, test_loader, cfg["datainfo"].output_dir, num_samples=10, cfg=cfg)
    
    return model, (test_loss, test_dice)

def create_test_loader():
    """
    创建测试数据加载器的辅助函数
    
    Returns:
        tuple: (datainfo, test_loader)
    """
    datainfo = DataInfo()
    
    # 使用ReadDataset直接加载测试数据
    print("加载测试数据集...")
    test_ds = ReadDataset(
        dataset_type='test', 
        batch_size=CONFIG["batch_size"], 
        auto_load=True, 
        balanced_sampling=False  # 测试时不需要平衡采样
    )
    
    # 获取DataLoader
    test_loader = test_ds.get_dataloader(shuffle=False)
    
    return datainfo, test_loader

if __name__ == "__main__":
    # 使用DataInfo获取配置
    datainfo = DataInfo()
    
    print(f"使用配置:")
    print(f"  测试名称: {datainfo.test_name}")
    print(f"  测试ID: {datainfo.test_id}")
    print(f"  数据目录: {datainfo.base_dir}")
    print(f"  输出目录: {datainfo.output_dir}")
    
    # 使用ReadDataset直接加载测试数据
    print("\n加载测试数据集...")
    test_ds = ReadDataset(
        dataset_type='test', 
        batch_size=CONFIG["batch_size"], 
        auto_load=True, 
        balanced_sampling=False  # 测试时不需要平衡采样
    )
    
    # 获取DataLoader
    test_loader = test_ds.get_dataloader(shuffle=False)
    
    # 模型路径 - 检查可用的模型文件
    possible_models = [
        datainfo.checkpoints_dir / "best_model_v1.pth"
    ]
    
    ckpt_path = None
    for model_path in possible_models:
        if model_path.exists():
            ckpt_path = model_path
            break
    
    if ckpt_path is None:
        print("错误：未找到任何模型文件！")
        print("可用文件:")
        for file in datainfo.checkpoints_dir.glob("*.pth"):
            print(f"  {file}")
        sys.exit(1)
    
    print(f"  模型路径: {ckpt_path}")
    print(f"  测试数据: {len(test_ds)} 个样本")
    
    # 运行测试
    run_test(str(ckpt_path), test_loader, datainfo)
