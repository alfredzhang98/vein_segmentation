"""
Pure training script built on top of your existing code (train/val loaders already created).
- 所有参数都在代码里的 CONFIG 里写死，无需 argparse。
- 仅包含训练 + 验证（用于监控 loss/dice），没有推理/测试部分。
- 支持 binary 或 multi-class（设定 CONFIG['num_classes']）。
- 默认使用 AdamW、BCE/CrossEntropy + DiceLoss、AMP、梯度裁剪与简单的 ReduceLROnPlateau 调度器。
- 可选 wandb 记录；不想用就把 wandb_enable=False。

把此文件放在你已有的数据准备代码后面或单独保存再 import 那段代码，直接运行即可。
"""
import os
import re
import math
import sys
import numpy as np
import pandas as pd
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataPrepare import ReadDataset

import logging
from typing import Dict, Any, Tuple
import wandb
from unet import UNet

# ----------------------------------------------------------------------------------
#                                      CONFIG
# ----------------------------------------------------------------------------------
CONFIG: Dict[str, Any] = {
    "epochs": 200,                                      # total epochs
    "early_stop_patience": 15,                          # e.g. 15 or None to disable
    "min_delta": 1e-3,                                  # minimum change to qualify as an improvement
    ## learning rate & scheduler
    "learning_rate": 1e-4,
    "scheduler": "plateau",                             # none | plateau | cosine
    "plateau_mode": "max",                              # max for dice
    "plateau_patience": 5,
    "cosine_Tmax": 50,
    "amp": True,
    ## model
    "batch_size": 32,                                   # batch size per GPU
    "n_channels": 1,                                    # input channels, 1 for grayscale ultrasound images
    "num_classes": 1,                                   # 1 -> binary, >1 -> multiclass
    ## training and optimizer
    "optimizer": "adamw",                               # adam | adamw | rmsprop | sgd
    "weight_decay": 1e-5,                               # weight decay for optimizer
    "momentum": 0.9,                                    # only for rmsprop/sgd
    "gradient_clip": 1.0,
    "bilinear": False,                                  # use bilinear upsampling in UNet
    ## wandb config
    "wandb_entity": "worldangle",                       # wandb wandb_entity
    "wandb_project_name": "us-segmentation",            # wandb project
    "wandb_enable": True,
    "log_histograms": False,
    ## checkpoint config
    "checkpoint_dir": "outputs/checkpoints",
    "resume_path": None,                                # path to .pth to resume, or None
}

DEVICE = torch.device("cuda:8" if torch.cuda.is_available() else "cpu")

# ----------------------------------------------------------------------------------
#                               LOSS & METRICS
# ----------------------------------------------------------------------------------


@torch.no_grad()
def dice_coeff(prob: torch.Tensor, target: torch.Tensor, eps: float = 1e-6, multiclass: bool = False, reduce_batch_first: bool = False) -> torch.Tensor:
    """
    Calculate Dice coefficient for semantic segmentation
    
    Args:
        prob: Predicted probabilities tensor
        target: Ground truth tensor  
        eps: Small epsilon to avoid division by zero
        multiclass: Whether this is multiclass segmentation
        reduce_batch_first: Whether to reduce batch dimension first
    
    Returns:
        Dice coefficient as tensor
    """
    # Ensure input and target have same size
    assert prob.size() == target.size(), f"Size mismatch: prob {prob.size()} vs target {target.size()}"
    assert prob.dim() == 3 or not reduce_batch_first, "reduce_batch_first requires at least 3D tensors"
    
    if not multiclass:
        # Binary segmentation: prob/target shape (N,1,H,W) or (N,H,W)
        if prob.dim() == 4:  # (N,1,H,W)
            sum_dim = (-1, -2) if not reduce_batch_first else (-1, -2, -3)
        else:  # (N,H,W) 
            sum_dim = (-1, -2) if prob.dim() == 2 or not reduce_batch_first else (-1, -2, -3)
    else:
        # Multiclass segmentation: one-hot encoded (N,C,H,W)
        sum_dim = (-1, -2) if not reduce_batch_first else (-1, -2, -3)
    
    # Calculate intersection and union
    inter = 2 * (prob * target).sum(dim=sum_dim)
    sets_sum = prob.sum(dim=sum_dim) + target.sum(dim=sum_dim)
    
    # Handle edge case where both prob and target are zero
    sets_sum = torch.where(sets_sum == 0, inter, sets_sum)
    
    # Calculate Dice coefficient
    dice = (inter + eps) / (sets_sum + eps)
    return dice.mean()

def dice_loss_from_logits(logits: torch.Tensor, target: torch.Tensor, multiclass: bool) -> torch.Tensor:
    if multiclass:
        probs = F.softmax(logits, dim=1)
        # check the one_hot whether already has the right channel order (N,C,H,W)
        tgt_1h = F.one_hot(target, num_classes=logits.shape[1]).permute(0, 3, 1, 2).float()
        return 1 - dice_coeff(probs, tgt_1h, multiclass=True)
    else:
        probs = torch.sigmoid(logits)
        return 1 - dice_coeff(probs, target.float(), multiclass=False)

# for the validation stage we do not need to store gradients so we use torch.no_grad()
@torch.no_grad()
def validate(model: nn.Module, loader: DataLoader, cfg: Dict[str, Any]) -> Tuple[float, float]:
    model.eval()  # 切换模型到评估模式，关闭 Dropout、BatchNorm 等训练专用行为
    # 累积指标：总 dice、总 loss、样本总数
    total_dice, total_loss, n = 0.0, 0.0, 0
    # 根据任务类型选用二分类或多分类交叉熵损失
    criterion = nn.BCEWithLogitsLoss() if cfg["num_classes"] == 1 else nn.CrossEntropyLoss()
    for batch in loader:
        # ReadDataset 返回 (image_tensor, mask_tensor, image_type)
        imgs, masks, image_type = batch
        # 将 C 通道 存储在最后一个维度 torch.channels_last
        # 由于模型还在cuda 上，所以需要将数据也转移到cuda上 以便于validation
        # 必须先做 imgs = imgs.to(DEVICE)，把图像张量搬到 GPU，才能进行后面的前向/反向计算。
        imgs = imgs.to(DEVICE, dtype=torch.float32, memory_format=torch.channels_last)
        if cfg["num_classes"] == 1:
            masks = masks.to(DEVICE, dtype=torch.float32)
        else:
            masks = masks.to(DEVICE, dtype=torch.long)
        # 默认cuda 如果 apple mps 就有 cpu
        with torch.autocast(DEVICE.type if DEVICE.type != 'mps' else 'cpu', enabled=cfg["amp"]):
            logits = model(imgs) # 前向计算，得到 raw logits

            # 计算 dice loss 
            if cfg["num_classes"] == 1:
                # squeeze 维度以匹配 BCEWithLogitsLoss 需要的 (N, H, W)
                # logits.shape == (N, 1, H, W) -> (N, H, W) by squeeze(1) squeeze the channel dimension
                # masks.shape == (N, 1, H, W) -> (N, H, W) by squeeze(1) squeeze the channel dimension
                ce = criterion(logits.squeeze(1), masks.squeeze(1))
                dsc = dice_loss_from_logits(logits, masks, multiclass=False)
                loss = ce + dsc
            else:
                ce = criterion(logits, masks)
                dsc = dice_loss_from_logits(logits, masks, multiclass=True)
                # 计算总损失
                loss = ce + dsc

            # 计算 dice 系数 计算指标，不参与梯度 越接近 1 表示重叠越好
            if cfg["num_classes"] == 1:
                probs = torch.sigmoid(logits)
                dice = dice_coeff(probs, masks.float(), multiclass=False)
            else:
                probs = F.softmax(logits, dim=1)
                masks_1h = F.one_hot(masks, num_classes=cfg["num_classes"]).permute(0,3,1,2).float()
                dice = dice_coeff(probs, masks_1h, multiclass=True)

        # 当前 batch 大小
        bs = imgs.size(0)
        # 累加当前 batch 的损失和 dice
        total_loss += loss.item() * bs
        total_dice += dice.item() * bs
        n += bs
    # 计算平均损失和 loss dice 返回
    return total_loss / n, total_dice / n

# ----------------------------------------------------------------------------------
#                                   TRAIN LOOP
# ----------------------------------------------------------------------------------


def train(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, cfg: Dict[str, Any]):
    # --- 优化器（Optimizer）设置 ---
    opt_name = cfg["optimizer"].lower()
    if opt_name == "adam":
        optimizer = optim.Adam(
            model.parameters(),
            lr=cfg["learning_rate"],
            weight_decay=cfg["weight_decay"],
            foreach=True
        )
    elif opt_name == "adamw":
        optimizer = optim.AdamW(
            model.parameters(),
            lr=cfg["learning_rate"],
            weight_decay=cfg["weight_decay"],
            foreach=True
        )
    elif opt_name == "rmsprop":
        optimizer = optim.RMSprop(
            model.parameters(),
            lr=cfg["learning_rate"],
            weight_decay=cfg["weight_decay"],
            momentum=cfg["momentum"],
            foreach=True
        )
    elif opt_name == "sgd":
        optimizer = optim.SGD(
            model.parameters(),
            lr=cfg["learning_rate"],
            momentum=cfg["momentum"],
            weight_decay=cfg["weight_decay"],
            nesterov=True
        )
    else:
        raise ValueError(f"Unknown optimizer: {cfg['optimizer']}")
    # Scheduler
    scheduler = None
    if cfg["scheduler"].lower() == "plateau":
        # ReduceLROnPlateau 根据验证指标不再提升时降低学习率
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=cfg["plateau_mode"],
            patience=cfg["plateau_patience"],
            factor=0.5
        )
    elif cfg["scheduler"].lower() == "cosine":
        # CosineAnnealingLR 做余弦衰减
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=cfg["cosine_Tmax"]
        )

    # 选择损失函数
    criterion = nn.BCEWithLogitsLoss() if cfg["num_classes"] == 1 else nn.CrossEntropyLoss()
    # AMP 梯度缩放器
    scaler = torch.amp.GradScaler('cuda', enabled=cfg["amp"] and DEVICE.type == 'cuda')

    # --- 恢复训练（Resume）相关 ---
    start_epoch = 1
    best_dice = -math.inf
    no_improve = 0

    # if you storage the model checkpoint you can fill the resume_path in the config to resume training
    if cfg["resume_path"] and Path(cfg["resume_path"]).is_file():
        ckpt = torch.load(cfg["resume_path"], map_location=DEVICE)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        if scheduler and ckpt.get("scheduler") is not None:
            scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt.get("epoch", 1) + 1
        best_dice = ckpt.get("best_score", -math.inf)
        logging.info(f"Resumed from {cfg['resume_path']} @ epoch {start_epoch}")

    # wandb 记录初始化
    run = None
    if cfg.get("wandb_enable", True):
        mode = os.environ.get("WANDB_MODE", "online")
        run = wandb.init(entity=cfg["wandb_entity"], project=cfg["wandb_project_name"], config = {
            "learning_rate": cfg["learning_rate"],
            "batch_size": cfg["batch_size"],
            "architecture": "UNet",
            "epochs": cfg["epochs"],
            "val_percent": 0.15,  # validation split
            "amp": cfg["amp"],
        })


    # --- 训练主循环 ---
    for epoch in range(start_epoch, cfg["epochs"] + 1):
        # 训练模式 启用 Dropout、更新 BatchNorm 统计量等。
        model.train()
        # 记录每个 epoch 的损失
        epoch_loss = 0.0
        # tqdm 进度条
        pbar = tqdm(
            total=len(train_loader.dataset),
            desc=f"Epoch {epoch}/{cfg['epochs']}",
            unit="img"
        )

        # 遍历训练集 DataLoader，每次取出一个 batch。
        for batch in train_loader:
            # ReadDataset 返回 (image_tensor, mask_tensor, image_type)
            imgs, masks, image_type = batch
            # 将图像张量搬到 GPU，并设置为 channels_last 格式
            imgs = imgs.to(DEVICE, dtype=torch.float32, memory_format=torch.channels_last)
            if cfg["num_classes"] == 1:
                masks = masks.to(DEVICE, dtype=torch.float32)
            else:
                masks = masks.to(DEVICE, dtype=torch.long)

            # 梯度清零：将上一轮累积的梯度全部清除。
            optimizer.zero_grad(set_to_none=True)

            # 前向计算
            with torch.autocast(DEVICE.type if DEVICE.type != 'mps' else 'cpu', enabled=cfg["amp"]):
                logits = model(imgs)
                if cfg["num_classes"] == 1:
                    ce  = criterion(logits.squeeze(1), masks.squeeze(1))
                    dsc = dice_loss_from_logits(logits, masks, multiclass=False)
                    loss = ce + dsc
                else:
                    ce  = criterion(logits, masks)
                    dsc = dice_loss_from_logits(logits, masks, multiclass=True)
                    loss = ce + dsc

            # 反向传播
            # scale → unscale 这对操作本身并不会改变最终用于更新模型的梯度值；
            # AMP（自动混合精度）后，所有被 autocast 包裹的前向和反向计算都会隐式地使用 float16：
            # AMP 的目的是 autocast 上下文会自动把支持的算子（卷积、矩阵乘法、点积等）切换到 float16 进行计算，以加速并减少显存占用；不支持的算子仍保留在 float32。
            # 它的作用只是为了在半精度（float16）下保护非常小的梯度不被“下溢”成 0，从而保证训练的稳定性。
            scaler.scale(loss).backward()
            if cfg["gradient_clip"]:
                # 取消对梯度的缩放，以便准确裁剪。
                scaler.unscale_(optimizer)
                # 梯度裁剪：防止梯度爆炸。
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["gradient_clip"])
            # 根据缩放梯度后结果更新参数。
            scaler.step(optimizer)
            # 更新缩放器
            scaler.update()
            
            # 当前 batch 的实际样本数
            bs = imgs.size(0)
            # 累加当前 batch 的损失
            epoch_loss += loss.item() * bs
            # 更新进度条
            pbar.update(bs)
            pbar.set_postfix(loss=f"{loss.item():.4f}")

            if run:
                wandb.log({"train/loss": loss.item(), "epoch": epoch})
        pbar.close()
        epoch_loss /= len(train_loader.dataset)

        # validate at end of epoch
        # 算本轮平均训练 loss：把累加的总 loss 除以训练集总样本数
        val_loss, val_dice = validate(model, val_loader, cfg)

        # print log local
        logging.info(f"Epoch {epoch}/{cfg['epochs']}: "
                     f"Train Loss: {epoch_loss:.4f}, "
                     f"Val Loss: {val_loss:.4f}, "
                     f"Val Dice: {val_dice:.4f}")

        if run:
            log_dict = {
                "val/loss": val_loss,
                "val/dice": val_dice,
                "train/epoch_loss": epoch_loss,
                "lr": optimizer.param_groups[0]['lr'],
                "epoch": epoch
            }

            # Save sample images to wandb 
            # try:
            #     # Log last batch samples
            #     sample_img = imgs[0].detach().cpu()
            #     if cfg["num_classes"] == 1:
            #         sample_true = masks[0].detach().cpu()
            #         sample_pred = torch.sigmoid(logits[0]).detach().cpu()
            #     else:
            #         sample_true = masks[0].detach().cpu().unsqueeze(0)
            #         sample_pred = logits.argmax(dim=1)[0].detach().cpu().unsqueeze(0)
            #     log_dict.update({
            #         "sample/image": wandb.Image(sample_img),
            #         "sample/true": wandb.Image(sample_true),
            #         "sample/pred": wandb.Image(sample_pred)
            #     })
            # except Exception:
            #     pass

            # Log histograms of model parameters
            # 卷积层权重 encoder.block1.conv1.weight、decoder.upconv.weigh
            # 卷积偏置等 *.bias
            # BatchNorm 全称是 Batch Normalization（批量归一化） bn.weight、bn.bias
            # 最后分类头的权重和偏置 final_conv.weight、final_conv.bias
            if cfg["log_histograms"]:
                for name, param in model.named_parameters():
                    if param.requires_grad and not torch.isnan(param).any():
                        log_dict[f"hist/{name}"] = wandb.Histogram(param.detach().cpu())

            wandb.log(log_dict)

        # --- Scheduler 更新 ---
        # 在每个 epoch 结束后更新学习率
        # 当该指标在连续若干个 epoch 内 没有改善（即“plateau”期）时，就自动把学习率乘以一个因子（如 0.5）降低
        if scheduler is not None:
            if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_dice if cfg["plateau_mode"] == "max" else val_loss)
            else:
                scheduler.step()

        # save
        is_best = val_dice > best_dice + cfg["min_delta"]
        if is_best:
            best_dice = val_dice
            no_improve = 0
        else:
            no_improve += 1

        if is_best:
            # save first
            save_ckpt(model, optimizer, scheduler, epoch, best_dice, cfg)
            # then remove the old best checkpoint, keep only the latest best
            remove_ckpt(cfg, keep_epoch=epoch)

        # early stop
        if cfg["early_stop_patience"] is not None and no_improve >= cfg["early_stop_patience"]:
            logging.info(f"Early stop at epoch {epoch}")
            break

    if run:
        run.finish()

    logging.info(f"Training complete. Best Dice: {best_dice:.4f}")

def save_ckpt(model, optimizer, scheduler, epoch, best_score, cfg):
    ckpt_dir = Path(cfg["checkpoint_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    model_path = ckpt_dir / f'checkpoint_{epoch}_dice_{best_score:.4f}.pth'
    torch.save({
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "best_score": best_score,
        "config": cfg
    }, model_path)
    logging.info(f"New best model saved: {model_path} (Dice: {best_score:.4f})")

def remove_ckpt(cfg, keep_epoch: int):
    """
    遍历 checkpoint_dir 下所有符合 checkpoint_{epoch}_dice_*.pth
    格式的文件，删除其中 epoch != keep_epoch 的文件。
    """
    ckpt_dir = Path(cfg["checkpoint_dir"])
    # 正则匹配：从文件名提取出 epoch 数字
    pattern = re.compile(r'^checkpoint_(\d+)_dice_.*\.pth$')

    for f in ckpt_dir.glob('checkpoint_*_dice_*.pth'):
        m = pattern.match(f.name)
        if not m:
            continue
        ep = int(m.group(1))
        # 如果不是当前要保留的 epoch，就删掉
        if ep != keep_epoch:
            try:
                f.unlink()
                logging.info(f"Removed old checkpoint: {f.name}")
            except Exception as e:
                logging.warning(f"Failed to remove {f.name}: {e}")

# ----------------------------------------------------------------------------------
#                                        MAIN
# ----------------------------------------------------------------------------------
if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logging.info(f"Device: {DEVICE}")

    # 使用 ReadDataset 直接加载处理好的数据
    print("=" * 60)
    print("初始化数据加载器")
    print("=" * 60)
    
    # 创建训练集数据加载器
    print("加载训练数据集...")
    train_ds = ReadDataset(
        dataset_type='train', 
        batch_size=CONFIG["batch_size"]
    )
    train_loader = train_ds.get_dataloader(shuffle=True)  # 训练集需要打乱
    
    # 创建验证集数据加载器
    print("加载验证数据集...")
    validation_ds = ReadDataset(
        dataset_type='validation', 
        batch_size=CONFIG["batch_size"]
    )
    validation_loader = validation_ds.get_dataloader(shuffle=False)  # 验证集不需要打乱
    
    # 打印数据集信息
    print(f"\n训练集信息: {len(train_ds)} 样本")
    train_class_weights = train_ds.get_class_weights()
    print("训练集类别权重:", train_class_weights)
    
    print(f"\n验证集信息: {len(validation_ds)} 样本")
    val_class_weights = validation_ds.get_class_weights()
    print("验证集类别权重:", val_class_weights)

    try:
        train_loader
        validation_loader
        print(f"\n数据加载器创建成功!")
        print(f"训练批次数: {len(train_loader)}")
        print(f"验证批次数: {len(validation_loader)}")
    except NameError:
        raise RuntimeError("train_loader / validation_loader 未定义，请检查 ReadDataset 加载是否成功。")

    # 创建模型
    print(f"\n创建 U-Net 模型 (输入通道: {CONFIG['n_channels']}, 输出类别: {CONFIG['num_classes']})")
    model = UNet(n_channels=CONFIG["n_channels"], n_classes=CONFIG["num_classes"], bilinear=CONFIG["bilinear"])
    model = model.to(DEVICE, memory_format=torch.channels_last)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型参数总数: {total_params:,}")
    print(f"可训练参数: {trainable_params:,}")

    print("\n" + "=" * 60)
    print("开始训练")
    print("=" * 60)
    
    # 开始训练
    train(model, train_loader, validation_loader, CONFIG)
    
    print("训练完成。")
