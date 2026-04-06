"""
U-Net Training Script
- 所有参数通过 .env 文件配置
- 自动检测空闲 GPU，多 GPU 并行训练（DataParallel）
- 支持 binary / multi-class，AMP，梯度裁剪，ReduceLROnPlateau / Cosine，wandb
"""
import os

# 限制线程数，防止共享服务器上 fork/内存分配失败
for _k in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_k, '1')
import re
import math
from datetime import datetime
import sys
import subprocess
import logging
import numpy as np
from pathlib import Path
from typing import Dict, Any, Tuple, List

from dotenv import load_dotenv
load_dotenv()   # 读取 .env 文件，必须在所有 os.environ.get 之前

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb

from dataPrepare import ReadDataset
from unet import UNet

# ----------------------------------------------------------------------------------
#                              从 .env 读取配置
# ----------------------------------------------------------------------------------

def _env(key: str, default, cast=str):
    """读取环境变量，自动类型转换。"""
    val = os.environ.get(key, '').strip()
    if not val:
        return default
    if cast is bool:
        return val.lower() in ('1', 'true', 'yes')
    try:
        return cast(val)
    except (ValueError, TypeError):
        return default


CONFIG: Dict[str, Any] = {
    # epochs & early stop
    "epochs":               _env('EPOCHS',               200,    int),
    "early_stop_patience":  _env('EARLY_STOP_PATIENCE',  15,     int),
    "min_delta":            _env('MIN_DELTA',             1e-3,   float),
    # learning rate & scheduler
    "learning_rate":        _env('LEARNING_RATE',         1e-4,   float),
    "scheduler":            _env('SCHEDULER',             'plateau'),
    "plateau_mode":         _env('PLATEAU_MODE',          'max'),
    "plateau_patience":     _env('PLATEAU_PATIENCE',      5,      int),
    "cosine_Tmax":          _env('COSINE_TMAX',           50,     int),
    "amp":                  _env('AMP',                   True,   bool),
    # model
    "batch_size_per_gpu":   _env('BATCH_SIZE_PER_GPU',   20,     int),
    "n_channels":           _env('N_CHANNELS',            1,      int),
    "num_classes":          _env('NUM_CLASSES',           1,      int),
    "bilinear":             _env('BILINEAR',              False,  bool),
    # optimizer
    "optimizer":            _env('OPTIMIZER',             'adamw'),
    "weight_decay":         _env('WEIGHT_DECAY',          1e-5,   float),
    "momentum":             _env('MOMENTUM',              0.9,    float),
    "gradient_clip":        _env('GRADIENT_CLIP',         1.0,    float),
    # dataset & transfer learning
    "dataset":              _env('DATASET',               'phantom_1'),
    "val_datasets":         [s.strip() for s in _env('VAL_DATASET', 'phantom_1').split(',') if s.strip()],
    "freeze_encoder":       _env('FREEZE_ENCODER',        False,  bool),
    "unfreeze_epoch":       _env('UNFREEZE_EPOCH',        10,     int),
    "unfreeze_layers":      _env('UNFREEZE_LAYERS',       'down3,down4'),
    # checkpoint
    "checkpoint_dir":       _env('CHECKPOINT_DIR',        'outputs/checkpoints'),
    "resume_path":          _env('RESUME_PATH',           '') or None,
    "reset_best_dice":      _env('RESET_BEST_DICE',       False,  bool),
    "run_name":             _env('RUN_NAME',              'run'),
    # wandb
    "wandb_entity":         _env('WANDB_ENTITY',          ''),
    "wandb_project_name":   _env('WANDB_PROJECT_NAME',    'us-segmentation'),
    "wandb_enable":         _env('WANDB_ENABLE',          False,  bool),
    "log_histograms":       _env('LOG_HISTOGRAMS',        False,  bool),
}

# ----------------------------------------------------------------------------------
#                              GPU 自动检测
# ----------------------------------------------------------------------------------

def _query_gpu_free() -> List[Tuple[int, float]]:
    """查询所有 GPU 的 (物理index, free_gb)，按空闲显存从大到小排序。"""
    try:
        out = subprocess.run(
            ['nvidia-smi',
             '--query-gpu=index,memory.free,memory.total',
             '--format=csv,noheader,nounits'],
            capture_output=True, text=True, check=True
        ).stdout.strip()
        gpus = []
        for line in out.split('\n'):
            if not line.strip():
                continue
            parts = line.split(',')
            idx = int(parts[0].strip())
            free_gb = int(parts[1].strip()) / 1024
            total_gb = int(parts[2].strip()) / 1024
            gpus.append((idx, free_gb, total_gb))
        gpus.sort(key=lambda x: x[1], reverse=True)
        return gpus
    except Exception as e:
        logging.warning(f"nvidia-smi query failed: {e}")
        return []


def _get_sys_free_mem_gb() -> float:
    """获取系统可用内存 (GB)。"""
    try:
        with open('/proc/meminfo') as f:
            for line in f:
                if line.startswith('MemFree:'):
                    return int(line.split()[1]) / (1024 * 1024)
    except Exception:
        pass
    return float('inf')


def select_devices() -> Tuple[torch.device, List[int]]:
    """
    自动选择 GPU，按实时空闲显存排序。
    返回 (primary_device, logical_gpu_ids)
    """
    if not torch.cuda.is_available():
        logging.info("CUDA not available, using CPU")
        return torch.device('cpu'), []

    if os.environ.get('CUDA_VISIBLE_DEVICES', '') != '':
        n = torch.cuda.device_count()
        ids = list(range(n))
        logging.info(f"CUDA_VISIBLE_DEVICES set, using logical GPUs: {ids}")
        return torch.device('cuda:0'), ids

    free_mem_gb = _env('GPU_MEM_FREE_GB', 10.0, float)
    gpus = _query_gpu_free()

    if not gpus:
        return torch.device('cuda:0'), [0]

    selected = [(idx, free, total) for idx, free, total in gpus if free >= free_mem_gb]
    if not selected:
        best = gpus[0]
        logging.warning(
            f"No GPU with >= {free_mem_gb}GB free, "
            f"selecting GPU {best[0]} ({best[1]:.1f}GB free / {best[2]:.0f}GB total)"
        )
        selected = [best]

    sys_free = _get_sys_free_mem_gb()
    max_gpus = max(1, int(sys_free / 6))
    if len(selected) > max_gpus:
        logging.warning(
            f"System memory {sys_free:.1f}GB, limiting GPUs {len(selected)} → {max_gpus}"
        )
        selected = selected[:max_gpus]

    physical_ids = [str(idx) for idx, _, _ in selected]
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(physical_ids)
    torch.cuda.init()

    logical_ids = list(range(len(selected)))
    for i, (pid, free, total) in enumerate(selected):
        tag = " ← primary" if i == 0 else ""
        logging.info(
            f"  GPU {pid}(logical {i}) | {free:.1f}GB free / {total:.0f}GB total{tag}"
        )
    logging.info(f"Selected {len(logical_ids)} GPUs: physical {physical_ids} → logical {logical_ids}")

    return torch.device('cuda:0'), logical_ids


# ----------------------------------------------------------------------------------
#                               LOSS & METRICS
# ----------------------------------------------------------------------------------

def _unwrap(model: nn.Module) -> nn.Module:
    """DataParallel unwrap."""
    return model.module if isinstance(model, nn.DataParallel) else model


@torch.no_grad()
def dice_coeff(prob: torch.Tensor, target: torch.Tensor,
               eps: float = 1e-6,
               multiclass: bool = False,
               reduce_batch_first: bool = False) -> torch.Tensor:
    assert prob.size() == target.size()
    assert prob.dim() == 3 or not reduce_batch_first

    if not multiclass:
        sum_dim = (-1, -2) if not reduce_batch_first else (-1, -2, -3)
    else:
        sum_dim = (-1, -2) if not reduce_batch_first else (-1, -2, -3)

    inter    = 2 * (prob * target).sum(dim=sum_dim)
    sets_sum = prob.sum(dim=sum_dim) + target.sum(dim=sum_dim)
    sets_sum = torch.where(sets_sum == 0, inter, sets_sum)
    return ((inter + eps) / (sets_sum + eps)).mean()


def dice_loss_from_logits(logits: torch.Tensor, target: torch.Tensor,
                          multiclass: bool) -> torch.Tensor:
    logits_f = logits.float()
    if multiclass:
        probs  = F.softmax(logits_f, dim=1)
        tgt_1h = F.one_hot(target, num_classes=logits.shape[1]).permute(0, 3, 1, 2).float()
        return 1 - dice_coeff(probs, tgt_1h, multiclass=True)
    else:
        probs = torch.sigmoid(logits_f)
        return 1 - dice_coeff(probs, target.float(), multiclass=False)


@torch.no_grad()
def validate(model: nn.Module, loader: DataLoader,
             cfg: Dict[str, Any], device: torch.device) -> Tuple[float, float]:
    model.eval()
    total_dice, total_loss, n = 0.0, 0.0, 0
    criterion = nn.BCEWithLogitsLoss() if cfg["num_classes"] == 1 else nn.CrossEntropyLoss()

    for imgs, masks, _ in loader:
        imgs  = imgs.to(device, dtype=torch.float32, memory_format=torch.channels_last)
        masks = masks.to(device, dtype=torch.float32 if cfg["num_classes"] == 1 else torch.long)

        autocast_dtype = device.type if device.type != 'mps' else 'cpu'
        with torch.autocast(autocast_dtype, enabled=cfg["amp"]):
            logits = model(imgs)
            if cfg["num_classes"] == 1:
                ce  = criterion(logits.squeeze(1), masks.squeeze(1))
                dsc = dice_loss_from_logits(logits, masks, multiclass=False)
            else:
                ce  = criterion(logits, masks)
                dsc = dice_loss_from_logits(logits, masks, multiclass=True)
            loss = ce + dsc

        if cfg["num_classes"] == 1:
            dice = dice_coeff(torch.sigmoid(logits.float()), masks.float(), multiclass=False)
        else:
            masks_1h = F.one_hot(masks, num_classes=cfg["num_classes"]).permute(0, 3, 1, 2).float()
            dice = dice_coeff(F.softmax(logits.float(), dim=1), masks_1h, multiclass=True)

        bs = imgs.size(0)
        total_loss += loss.item() * bs
        total_dice += dice.item() * bs
        n += bs

    return total_loss / n, total_dice / n


# ----------------------------------------------------------------------------------
#                                   TRAIN LOOP
# ----------------------------------------------------------------------------------

def train(model: nn.Module, train_loader: DataLoader,
          val_loaders: Dict[str, DataLoader],
          cfg: Dict[str, Any], device: torch.device):

    # Freeze encoder if requested (for finetune stage)
    if cfg.get("freeze_encoder"):
        for name, param in _unwrap(model).named_parameters():
            if any(s in name for s in ['inc', 'down1', 'down2', 'down3', 'down4']):
                param.requires_grad = False
        frozen = sum(1 for p in _unwrap(model).parameters() if not p.requires_grad)
        total  = sum(1 for _ in _unwrap(model).parameters())
        logging.info(f"Encoder frozen: {frozen}/{total} parameters")

    # Optimizer — only optimize trainable parameters
    trainable_params = [p for p in _unwrap(model).parameters() if p.requires_grad]
    opt_name = cfg["optimizer"].lower()
    opt_kwargs = dict(lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"], foreach=True)
    if opt_name == "adam":
        optimizer = optim.Adam(trainable_params, **opt_kwargs)
    elif opt_name == "adamw":
        optimizer = optim.AdamW(trainable_params, **opt_kwargs)
    elif opt_name == "rmsprop":
        optimizer = optim.RMSprop(trainable_params,
                                  momentum=cfg["momentum"], **opt_kwargs)
    elif opt_name == "sgd":
        optimizer = optim.SGD(trainable_params,
                              momentum=cfg["momentum"], nesterov=True, **opt_kwargs)
    else:
        raise ValueError(f"Unknown optimizer: {cfg['optimizer']}")

    # Scheduler
    scheduler = None
    if cfg["scheduler"].lower() == "plateau":
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode=cfg["plateau_mode"],
            patience=cfg["plateau_patience"], factor=0.5)
    elif cfg["scheduler"].lower() == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cfg["cosine_Tmax"])

    criterion = nn.BCEWithLogitsLoss() if cfg["num_classes"] == 1 else nn.CrossEntropyLoss()
    scaler    = torch.amp.GradScaler('cuda', enabled=cfg["amp"] and device.type == 'cuda')

    start_epoch    = 1
    best_dice      = -math.inf
    no_improve     = 0
    prev_ckpt_path = None

    # Resume
    if cfg["resume_path"] and Path(cfg["resume_path"]).is_file():
        ckpt = torch.load(cfg["resume_path"], map_location=device)
        _unwrap(model).load_state_dict(ckpt["model"])
        try:
            optimizer.load_state_dict(ckpt["optimizer"])
            if scheduler and ckpt.get("scheduler") is not None:
                scheduler.load_state_dict(ckpt["scheduler"])
        except ValueError:
            logging.warning("Optimizer state skipped (parameter groups mismatch)")
        start_epoch = ckpt.get("epoch", 1) + 1
        best_dice   = -math.inf if cfg.get("reset_best_dice") else ckpt.get("best_score", -math.inf)
        logging.info(f"Resumed from {cfg['resume_path']} @ epoch {start_epoch}")

    # WandB
    run = None
    if cfg.get("wandb_enable"):
        os.environ.setdefault('WANDB_DISABLE_STATS', 'true')
        os.environ.setdefault('WANDB_INIT_TIMEOUT', '10')
        try:
            run = wandb.init(
                entity=cfg["wandb_entity"] or None,
                project=cfg["wandb_project_name"],
                config={k: v for k, v in cfg.items() if not k.startswith('wandb')},
                settings=wandb.Settings(disable_code=True, init_timeout=10),
            )
        except Exception as e:
            logging.warning(f"WandB init failed, skipping: {e}")
            run = None

    autocast_dtype = device.type if device.type != 'mps' else 'cpu'

    unfrozen = False

    for epoch in range(start_epoch, cfg["epochs"] + 1):

        # Gradual unfreeze
        if (cfg.get("freeze_encoder") and not unfrozen
                and cfg.get("unfreeze_epoch")
                and (epoch - start_epoch) >= cfg["unfreeze_epoch"]):
            unfreeze_names = [s.strip() for s in cfg["unfreeze_layers"].split(',')]
            for name, param in _unwrap(model).named_parameters():
                if any(s in name for s in unfreeze_names):
                    param.requires_grad = True
            trainable_params = [p for p in _unwrap(model).parameters() if p.requires_grad]
            unfreeze_lr = cfg["learning_rate"] * 0.5
            opt_kwargs = dict(lr=unfreeze_lr, weight_decay=cfg["weight_decay"], foreach=True)
            optimizer = optim.AdamW(trainable_params, **opt_kwargs)
            if scheduler is not None:
                scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer, mode=cfg["plateau_mode"],
                    patience=cfg["plateau_patience"], factor=0.5)
            unfrozen = True
            frozen = sum(1 for p in _unwrap(model).parameters() if not p.requires_grad)
            total  = sum(1 for _ in _unwrap(model).parameters())
            logging.info(
                f"Epoch {epoch}: unfroze {unfreeze_names}, "
                f"still frozen {frozen}/{total}, LR → {unfreeze_lr:.1e}"
            )

        model.train()
        epoch_loss = 0.0
        pbar = tqdm(total=len(train_loader.dataset),
                    desc=f"Epoch {epoch}/{cfg['epochs']}", unit="img")

        for imgs, masks, _ in train_loader:
            imgs  = imgs.to(device, dtype=torch.float32, memory_format=torch.channels_last)
            masks = masks.to(device, dtype=torch.float32 if cfg["num_classes"] == 1 else torch.long)

            optimizer.zero_grad(set_to_none=True)

            with torch.autocast(autocast_dtype, enabled=cfg["amp"]):
                logits = model(imgs)
                if cfg["num_classes"] == 1:
                    ce   = criterion(logits.squeeze(1), masks.squeeze(1))
                    dsc  = dice_loss_from_logits(logits, masks, multiclass=False)
                else:
                    ce   = criterion(logits, masks)
                    dsc  = dice_loss_from_logits(logits, masks, multiclass=True)
                loss = ce + dsc

            scaler.scale(loss).backward()
            if cfg["gradient_clip"]:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(_unwrap(model).parameters(), cfg["gradient_clip"])
            scaler.step(optimizer)
            scaler.update()

            bs          = imgs.size(0)
            epoch_loss += loss.item() * bs
            pbar.update(bs)
            pbar.set_postfix(loss=f"{loss.item():.4f}")
            if run:
                wandb.log({"train/loss": loss.item(), "epoch": epoch})

        pbar.close()
        epoch_loss /= len(train_loader.dataset)

        # Validate on all datasets; first one is primary (for checkpoint / scheduler)
        primary_name = cfg["val_datasets"][0]
        val_results = {}
        for vname, vloader in val_loaders.items():
            vl, vd = validate(model, vloader, cfg, device)
            val_results[vname] = {"loss": vl, "dice": vd}

        val_loss = val_results[primary_name]["loss"]
        val_dice = val_results[primary_name]["dice"]

        parts = [f"train_loss={epoch_loss:.4f}"]
        for vname, vr in val_results.items():
            tag = "(primary)" if vname == primary_name else ""
            parts.append(f"{vname}{tag}: loss={vr['loss']:.4f} dice={vr['dice']:.4f}")
        logging.info(f"Epoch {epoch}/{cfg['epochs']}: " + "  |  ".join(parts))

        if run:
            log_dict = {
                "train/epoch_loss": epoch_loss,
                "lr": optimizer.param_groups[0]['lr'],
                "epoch": epoch,
            }
            for vname, vr in val_results.items():
                log_dict[f"val/{vname}/loss"] = vr["loss"]
                log_dict[f"val/{vname}/dice"] = vr["dice"]
            log_dict["val/loss"] = val_loss
            log_dict["val/dice"] = val_dice
            if cfg["log_histograms"]:
                for name, param in _unwrap(model).named_parameters():
                    if param.requires_grad and not torch.isnan(param).any():
                        log_dict[f"hist/{name}"] = wandb.Histogram(param.detach().cpu())
            wandb.log(log_dict)

        # Scheduler step (based on primary val dataset)
        if scheduler is not None:
            if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_dice if cfg["plateau_mode"] == "max" else val_loss)
            else:
                scheduler.step()

        # Checkpoint (based on primary val dataset)
        is_best = val_dice > best_dice + cfg["min_delta"]
        if is_best:
            best_dice  = val_dice
            no_improve = 0
            saved_path = save_ckpt(model, optimizer, scheduler, epoch, best_dice, cfg)
            remove_ckpt(prev_ckpt_path)
            prev_ckpt_path = saved_path
        else:
            no_improve += 1

        # Early stop
        if cfg["early_stop_patience"] and no_improve >= cfg["early_stop_patience"]:
            logging.info(f"Early stop at epoch {epoch}")
            break

    if run:
        run.finish()
    logging.info(f"Training complete. Best Val Dice: {best_dice:.4f}")


# ----------------------------------------------------------------------------------
#                              Checkpoint helpers
# ----------------------------------------------------------------------------------

def save_ckpt(model, optimizer, scheduler, epoch, best_score, cfg):
    ckpt_dir = Path(cfg["checkpoint_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
    name = cfg.get("run_name", "run")
    path = ckpt_dir / f"checkpoint_{name}_{epoch}_dice_{best_score:.4f}_{ts}.pth"
    torch.save({
        "epoch":      epoch,
        "model":      _unwrap(model).state_dict(),
        "optimizer":  optimizer.state_dict(),
        "scheduler":  scheduler.state_dict() if scheduler else None,
        "best_score": best_score,
        "config":     cfg,
    }, path)
    logging.info(f"Checkpoint saved: {path}  (Dice: {best_score:.4f})")
    return path


def remove_ckpt(prev_path: Path):
    """Remove previous checkpoint from this training run."""
    if prev_path is not None and prev_path.exists():
        try:
            prev_path.unlink()
            logging.info(f"Removed old checkpoint: {prev_path.name}")
        except Exception as e:
            logging.warning(f"Failed to remove {prev_path.name}: {e}")


# ----------------------------------------------------------------------------------
#                                        MAIN
# ----------------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')

    # 1. Auto-detect GPUs
    DEVICE, gpu_ids = select_devices()

    print("=" * 60)
    print("Environment")
    print("=" * 60)
    logging.info(f"PyTorch  : {torch.__version__}")
    logging.info(f"CUDA     : {torch.version.cuda}")
    if gpu_ids:
        for gid in gpu_ids:
            prop = torch.cuda.get_device_properties(gid)
            logging.info(f"  Logical GPU {gid} | {prop.name} | SM×{prop.multi_processor_count}")
    else:
        logging.info("  Using CPU")

    # 2. Compute actual batch size
    n_gpus      = max(len(gpu_ids), 1)
    total_batch = CONFIG["batch_size_per_gpu"] * n_gpus
    logging.info(f"GPUs: {n_gpus}  |  batch/GPU: {CONFIG['batch_size_per_gpu']}  |  total batch: {total_batch}")

    # 3. Data loading
    print("=" * 60)
    print("Loading datasets")
    print("=" * 60)
    dataset_mode = CONFIG["dataset"]
    logging.info(f"Training dataset: {dataset_mode}")

    if dataset_mode == "mixed":
        from torch.utils.data import ConcatDataset
        ph_train = ReadDataset('train', batch_size=1, dataset_name='phantom_1')
        mn_train = ReadDataset('train', batch_size=1, dataset_name='dataset1')
        combined = ConcatDataset([ph_train, mn_train])
        train_loader = DataLoader(combined, batch_size=total_batch, shuffle=True,
                                  num_workers=4, drop_last=False,
                                  pin_memory=torch.cuda.is_available())
    else:
        train_ds     = ReadDataset('train', batch_size=total_batch, dataset_name=dataset_mode)
        train_loader = train_ds.get_dataloader(shuffle=True, num_workers=0)

    # val — supports multiple val datasets, first is primary
    val_loaders = {}
    for vname in CONFIG["val_datasets"]:
        vds = ReadDataset('validation', batch_size=total_batch, dataset_name=vname)
        val_loaders[vname] = vds.get_dataloader(shuffle=False, num_workers=0)
    print(f"Train batches: {len(train_loader)}  Val datasets: {list(val_loaders.keys())}")

    # 4. Create model
    print("\nCreating U-Net model...")
    model = UNet(n_channels=CONFIG["n_channels"],
                 n_classes=CONFIG["num_classes"],
                 bilinear=CONFIG["bilinear"])
    model = model.to(DEVICE, memory_format=torch.channels_last)

    # 5. Multi-GPU (DataParallel)
    if len(gpu_ids) > 1:
        model = nn.DataParallel(model, device_ids=gpu_ids)
        logging.info(f"DataParallel enabled, using GPUs: {gpu_ids}")

    total_params = sum(p.numel() for p in _unwrap(model).parameters())
    logging.info(f"Model parameters: {total_params:,}")

    # 6. Train
    print("\n" + "=" * 60)
    print("Training")
    print("=" * 60)
    train(model, train_loader, val_loaders, CONFIG, DEVICE)
    print("Training complete.")
