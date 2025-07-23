"""
Pure training script built on top of your existing code (train/val loaders already created).
- 所有参数都在代码里的 CONFIG 里写死，无需 argparse。
- 仅包含训练 + 验证（用于监控 loss/dice），没有推理/测试部分。
- 支持 binary 或 multi-class（设定 CONFIG['num_classes']）。
- 默认使用 AdamW、BCE/CrossEntropy + DiceLoss、AMP、梯度裁剪与简单的 ReduceLROnPlateau 调度器。
- 可选 wandb 记录；不想用就把 USE_WANDB=False。

把此文件放在你已有的数据准备代码后面或单独保存再 import 那段代码，直接运行即可。
"""
import os
import math
import random
import numpy as np
import pandas as pd
from PIL import Image
from pathlib import Path
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm

import albumentations as A
from albumentations.pytorch import ToTensorV2
from dataSizeMap import DatasetSizeMap

import logging
from typing import Dict, Any, Tuple
import wandb
from unet import UNet

# ----------------------------------------------------------------------------------
#                                      CONFIG
# ----------------------------------------------------------------------------------
CONFIG: Dict[str, Any] = {
    "epochs": 100,                                      # total epochs
    "learning_rate": 1e-4,
    "batch_size": 8,                                   # batch size per GPU
    "weight_decay": 1e-5,                               # weight decay for optimizer
    "optimizer": "adamw",                               # adam | adamw | rmsprop | sgd
    "momentum": 0.9,                                    # only for rmsprop/sgd
    "gradient_clip": 1.0,
    "amp": True,
    "num_classes": 1,                                   # 1 -> binary, >1 -> multiclass
    "bilinear": False,
    "threshold": 0.5,                                   # for binary dice/iou
    "checkpoint_dir": "outputs/checkpoints",
    "resume_path": None,                                # path to .pth to resume, or None
    "scheduler": "plateau",                             # none | plateau | cosine
    "plateau_mode": "max",                              # max for dice
    "plateau_patience": 5,
    "cosine_Tmax": 50,
    "early_stop_patience": None,                        # e.g. 15 or None to disable
    "log_histograms": False,
    "entity": "worldangle",                             # wandb entity
    "project_name": "us-segmentation",                  # wandb project
    "use_wandb": True,
    "best_ckpt_name": "best_model.pth",
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------------------------------------------------------------------------
#                               LOSS & METRICS
# ----------------------------------------------------------------------------------

@torch.no_grad()
def dice_coeff(prob: torch.Tensor, target: torch.Tensor, eps: float = 1e-6, multiclass: bool = False) -> torch.Tensor:
    if not multiclass:  # prob/target: (N,1,H,W)
        intersection = (prob * target).sum(dim=(2, 3))
        union = prob.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
        dice = (2. * intersection + eps) / (union + eps)
        return dice.mean()
    else:               # one-hot both: (N,C,H,W)
        intersection = (prob * target).sum(dim=(2, 3))
        union = prob.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
        dice = (2. * intersection + eps) / (union + eps)
        return dice.mean()


def dice_loss_from_logits(logits: torch.Tensor, target: torch.Tensor, multiclass: bool) -> torch.Tensor:
    if multiclass:
        probs = F.softmax(logits, dim=1)
        tgt_1h = F.one_hot(target, num_classes=logits.shape[1]).permute(0, 3, 1, 2).float()
        return 1 - dice_coeff(probs, tgt_1h, multiclass=True)
    else:
        probs = torch.sigmoid(logits)
        return 1 - dice_coeff(probs, target.float(), multiclass=False)


@torch.no_grad()
def validate(model: nn.Module, loader: DataLoader, cfg: Dict[str, Any]) -> Tuple[float, float]:
    model.eval()
    total_dice, total_loss, n = 0.0, 0.0, 0
    criterion = nn.BCEWithLogitsLoss() if cfg["num_classes"] == 1 else nn.CrossEntropyLoss()
    for batch in loader:
        imgs, masks = unpack(batch)
        imgs = imgs.to(DEVICE, dtype=torch.float32, memory_format=torch.channels_last)
        if cfg["num_classes"] == 1:
            masks = masks.to(DEVICE, dtype=torch.float32)
        else:
            masks = masks.to(DEVICE, dtype=torch.long)
        with torch.autocast(DEVICE.type if DEVICE.type != 'mps' else 'cpu', enabled=cfg["amp"]):
            logits = model(imgs)
            if cfg["num_classes"] == 1:
                ce = criterion(logits.squeeze(1), masks.squeeze(1))
                dsc = dice_loss_from_logits(logits, masks, multiclass=False)
            else:
                ce = criterion(logits, masks)
                dsc = dice_loss_from_logits(logits, masks, multiclass=True)
            loss = ce + dsc
            if cfg["num_classes"] == 1:
                probs = torch.sigmoid(logits)
                dice = dice_coeff(probs, masks.float(), multiclass=False)
            else:
                probs = F.softmax(logits, dim=1)
                masks_1h = F.one_hot(masks, num_classes=cfg["num_classes"]).permute(0,3,1,2).float()
                dice = dice_coeff(probs, masks_1h, multiclass=True)
        bs = imgs.size(0)
        total_loss += loss.item() * bs
        total_dice += dice.item() * bs
        n += bs
    return total_loss / n, total_dice / n


def unpack(batch):
    if isinstance(batch, dict):
        return batch['image'], batch['mask']
    return batch[0], batch[1]

# ----------------------------------------------------------------------------------
#                                   TRAIN LOOP
# ----------------------------------------------------------------------------------


def train(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, cfg: Dict[str, Any]):
    # Optimizer
    opt_name = cfg["optimizer"].lower()
    if opt_name == "adam":
        optimizer = optim.Adam(model.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"], foreach=True)
    elif opt_name == "adamw":
        optimizer = optim.AdamW(model.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"], foreach=True)
    elif opt_name == "rmsprop":
        optimizer = optim.RMSprop(model.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"], momentum=cfg["momentum"], foreach=True)
    elif opt_name == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=cfg["learning_rate"], momentum=cfg["momentum"], weight_decay=cfg["weight_decay"], nesterov=True)
    else:
        raise ValueError(f"Unknown optimizer: {cfg['optimizer']}")

    # Scheduler
    scheduler = None
    if cfg["scheduler"].lower() == "plateau":
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode=cfg["plateau_mode"], patience=cfg["plateau_patience"], factor=0.5)
    elif cfg["scheduler"].lower() == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg["cosine_Tmax"])

    criterion = nn.BCEWithLogitsLoss() if cfg["num_classes"] == 1 else nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler('cuda', enabled=cfg["amp"] and DEVICE.type == 'cuda')

    start_epoch = 1
    best_dice = -math.inf
    no_improve = 0

    # Resume
    if cfg["resume_path"] and Path(cfg["resume_path"]).is_file():
        ckpt = torch.load(cfg["resume_path"], map_location=DEVICE)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        if scheduler and ckpt.get("scheduler") is not None:
            scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt.get("epoch", 1) + 1
        best_dice = ckpt.get("best_score", -math.inf)
        logging.info(f"Resumed from {cfg['resume_path']} @ epoch {start_epoch}")

    run = None
    if cfg.get("use_wandb", True):
        mode = os.environ.get("WANDB_MODE", "online")
        run = wandb.init(entity=cfg["entity"], project=cfg["project_name"], config = {
            "learning_rate": cfg["learning_rate"],
            "batch_size": cfg["batch_size"],
            "architecture": "UNet",
            "epochs": cfg["epochs"],
            "val_percent": 0.15,  # validation split
            "amp": cfg["amp"],
        })

    for epoch in range(start_epoch, cfg["epochs"] + 1):
        model.train()
        epoch_loss = 0.0
        pbar = tqdm(total=len(train_loader.dataset), desc=f"Epoch {epoch}/{cfg['epochs']}", unit="img")

        for batch in train_loader:
            imgs, masks = unpack(batch)
            imgs = imgs.to(DEVICE, dtype=torch.float32, memory_format=torch.channels_last)
            if cfg["num_classes"] == 1:
                masks = masks.to(DEVICE, dtype=torch.float32)
            else:
                masks = masks.to(DEVICE, dtype=torch.long)

            optimizer.zero_grad(set_to_none=True)
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

            scaler.scale(loss).backward()
            if cfg["gradient_clip"]:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["gradient_clip"])
            scaler.step(optimizer)
            scaler.update()

            bs = imgs.size(0)
            epoch_loss += loss.item() * bs
            pbar.update(bs)
            pbar.set_postfix(loss=f"{loss.item():.4f}")

            if run:
                wandb.log({"train/loss": loss.item(), "epoch": epoch})

        pbar.close()
        epoch_loss /= len(train_loader.dataset)

        # validate at end of epoch
        val_loss, val_dice = validate(model, val_loader, cfg)

        #pint log local
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

            if cfg["log_histograms"]:
                for name, param in model.named_parameters():
                    if param.requires_grad and not torch.isnan(param).any():
                        log_dict[f"hist/{name}"] = wandb.Histogram(param.detach().cpu())

            wandb.log(log_dict)

        # scheduler step
        if scheduler is not None:
            if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_dice if cfg["plateau_mode"] == "max" else val_loss)
            else:
                scheduler.step()

        # save
        is_best = val_dice > best_dice
        if is_best:
            best_dice = val_dice
            no_improve = 0
        else:
            no_improve += 1

        if is_best:
            save_ckpt(model, optimizer, scheduler, epoch, best_dice, cfg, is_best)

        elif epoch == cfg["epochs"]:
            save_ckpt(model, optimizer, scheduler, epoch, best_dice, cfg, False)

        # early stop
        if cfg["early_stop_patience"] is not None and no_improve >= cfg["early_stop_patience"]:
            logging.info(f"Early stop at epoch {epoch}")
            break

    if run:
        run.finish()
    logging.info(f"Training complete. Best Dice: {best_dice:.4f}")


def save_ckpt(model, optimizer, scheduler, epoch, best_score, cfg, is_best):
    ckpt_dir = Path(cfg["checkpoint_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    
    # 只保存最佳模型
    best_path = ckpt_dir / cfg.get("best_ckpt_name", "best_model.pth")
    torch.save({
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "best_score": best_score,
        "config": cfg
    }, best_path)
    logging.info(f"New best model saved: {best_path} (Dice: {best_score:.4f})")


# ----------------------------------------------------------------------------------
#                                        MAIN
# ----------------------------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logging.info(f"Device: {DEVICE}")


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

    # Get the dataloaders
    transform_train = A.Compose([
        A.Normalize(mean=(0.5,), std=(0.5,)),
        ToTensorV2(),
    ])

    transform_val = A.Compose([
        A.Normalize(mean=(0.5,), std=(0.5,)),
        ToTensorV2(),
    ])

    transform_test = A.Compose([
        A.Normalize(mean=(0.5,), std=(0.5,)),
        ToTensorV2(),
    ])

    train_df = pd.read_csv(TRAIN_DIR)
    train_ds = DatasetSizeMap(train_df, transform_train)
    validation_df = pd.read_csv(VAL)
    validation_ds = DatasetSizeMap(validation_df, transform_val)
    test_df = pd.read_csv(TEST_DIR)
    test_ds = DatasetSizeMap(test_df, transform_test)

    train_loader = DataLoader(train_ds, batch_size=CONFIG["batch_size"], shuffle=True, num_workers=4)
    validation_loader = DataLoader(validation_ds, batch_size=CONFIG["batch_size"], shuffle=False, num_workers=4)
    test_loader = DataLoader(test_ds, batch_size=CONFIG["batch_size"], shuffle=False, num_workers=4)

    try:
        train_loader
        validation_loader
    except NameError:
        raise RuntimeError("train_loader / validation_loader 未定义，请在此文件上方粘贴你创建 DataLoader 的代码或在 import 前构建好它们。")

    model = UNet(n_channels=1, n_classes=CONFIG["num_classes"], bilinear=CONFIG["bilinear"])
    model = model.to(DEVICE, memory_format=torch.channels_last)

    # 可选：小显存时用 checkpointing（如果你的 UNet 实现里有这个方法）
    # if hasattr(model, 'use_checkpointing') and torch.cuda.is_available():
    #     total_mem = torch.cuda.get_device_properties(0).total_memory
    #     if total_mem < 6 * 1024**3:
    #         logging.info("VRAM < 6GB, enabling gradient checkpointing")
    #         model.use_checkpointing()

    train(model, train_loader, validation_loader, CONFIG)
