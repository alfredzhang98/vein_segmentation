"""
U-Net Training Script
- 参数通过 .env 或 --config 指定的实验配置文件读取
- 自动检测空闲 GPU，多 GPU 并行训练（DataParallel）
- 支持 binary / multi-class，AMP，梯度裁剪，ReduceLROnPlateau / Cosine，wandb
"""
import os
import argparse
import json
import random

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

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_training_environment(config_path=None):
    """Use one dotenv file; explicit shell variables take precedence over it.

    An experiment replaces the root .env rather than inheriting mutable settings
    from it. Relative config paths are resolved from the caller's working directory.
    """
    path = Path(config_path).resolve() if config_path else PROJECT_ROOT / ".env"
    if config_path and not path.is_file():
        raise FileNotFoundError(f"Training config not found: {path}")
    load_dotenv(path, override=False)
    return path


def apply_training_overrides(overrides):
    """Apply explicit KEY=VALUE arguments after file/shell configuration."""
    parsed = {}
    for item in overrides:
        key, sep, value = item.partition("=")
        if not sep or not re.fullmatch(r"[A-Z][A-Z0-9_]*", key):
            raise ValueError(f"Expected KEY=VALUE, got {item!r}")
        parsed[key] = value
    os.environ.update(parsed)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train U-Net with the partial-label mixed loss")
    parser.add_argument("--config", type=Path, help="Experiment dotenv file; default: repository .env")
    parser.add_argument("--set", nargs="+", action="extend", default=[], metavar="KEY=VALUE",
                        help="Override config values for this run; may be repeated")
    parser.add_argument("--show-config", action="store_true",
                        help="Print effective settings and exit without loading data or selecting a GPU")
    args = parser.parse_args()
    try:
        load_training_environment(args.config)
        apply_training_overrides(args.set)
    except (FileNotFoundError, ValueError) as e:
        parser.error(str(e))
    # Dataset/checkpoint paths retain the documented repository-relative meaning.
    os.chdir(PROJECT_ROOT)
else:
    load_training_environment()

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb

# Repo root on sys.path, so `common` and `models` import the same way however this
# file is launched (python models/unet/train.py, or -m models.unet.train).
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from data.pipeline.dataPrepare import (ReadDataset, MODE_IDS, check_stale,
                                       CLASS_BG, CLASS_VEIN, CLASS_ARTERY, CLASS_VESSEL)
from models.unet.model import UNet

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


def _parse_weights(spec: str) -> Dict[str, float]:
    """'a:0.5, b:0.35' -> {'a': 0.5, 'b': 0.35}. Keys must match VAL_DATASET entries."""
    out: Dict[str, float] = {}
    for item in spec.split(','):
        item = item.strip()
        if not item:
            continue
        name, _, w = item.rpartition(':')
        if not name:
            raise ValueError(f"VAL_WEIGHTS 格式错误: '{item}'，应为 '数据集名:权重'")
        out[name.strip()] = float(w)
    return out


def _parse_train_repeats(spec: str) -> Dict[str, int]:
    """Per-source online draws per original; independent of validation weights."""
    out = {}
    for item in spec.split(','):
        if not item.strip():
            continue
        name, sep, value = item.strip().partition(':')
        name, value = name.strip(), value.strip()
        if not sep or not name or not re.fullmatch(r'[0-9]+', value) or int(value) < 1:
            raise ValueError(f"TRAIN_REPEATS expects dataset:positive_integer, got {item!r}")
        if name in out:
            raise ValueError(f"Duplicate dataset in TRAIN_REPEATS: {name}")
        out[name] = int(value)
    return out


def validate_train_repeats(cfg):
    spec = 'phantom_taobao+mendeley' if cfg['dataset'] == 'mixed' else cfg['dataset']
    names = {name.strip() for name in spec.split('+') if name.strip()}
    unknown = set(cfg.get('train_repeats', {})) - names
    if unknown:
        raise ValueError(f"TRAIN_REPEATS keys must be selected in DATASET: {sorted(unknown)}")


CONFIG: Dict[str, Any] = {
    "seed": _env('SEED', 42, int),
    "history_path": _env('HISTORY_PATH', ''),
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
    "amp_dtype":            _env('AMP_DTYPE',             'float16'),
    # model
    "batch_size_per_gpu":   _env('BATCH_SIZE_PER_GPU',   20,     int),
    "n_channels":           _env('N_CHANNELS',            1,      int),
    # 1 = legacy binary sigmoid model. 3 = bg / vein / artery with the partial-label
    # marginal loss (see partial_label_loss) — the one that spans all four datasets.
    "num_classes":          _env('NUM_CLASSES',           1,      int),
    # Capacity. 64 = the original 31M-param U-Net; 32 -> 7.8M; 16 -> 1.9M.
    # The vein memorises 80 subjects (train Dice 0.931 / val 0.726). See unet_model.py.
    "base_ch":              _env('BASE_CH',               64,     int),
    "dropout":              _env('DROPOUT',               0.0,    float),
    "ce_class_weights":     [float(x) for x in _env('CE_CLASS_WEIGHTS', '').split(',') if x.strip()],
    # Tversky: beta 惩罚漏报(FN)，alpha 惩罚误报(FP)。留空 = 用 Dice (等价于 0.5/0.5)。
    # 实测静脉有 37% 的像素被判成背景（漏检），beta>alpha 正是针对这个。
    "tversky_beta":         _env('TVERSKY_BETA',          0.0,    float),
    "tversky_alpha":        _env('TVERSKY_ALPHA',         0.0,    float),
    "bilinear":             _env('BILINEAR',              False,  bool),
    # optimizer
    "optimizer":            _env('OPTIMIZER',             'adamw'),
    "weight_decay":         _env('WEIGHT_DECAY',          1e-5,   float),
    "momentum":             _env('MOMENTUM',              0.9,    float),
    "gradient_clip":        _env('GRADIENT_CLIP',         1.0,    float),
    # dataset & transfer learning
    "dataset":              _env('DATASET',               'phantom_taobao'),
    "train_repeats":        _parse_train_repeats(_env('TRAIN_REPEATS', '')),
    "val_datasets":         [s.strip() for s in _env('VAL_DATASET', 'phantom_taobao').split(',') if s.strip()],
    # 决定 checkpoint / early-stop 的准则：
    #   'weighted'  = 各验证集按 VAL_WEIGHTS 加权求和（推荐）
    #   'mean'      = 各验证集等权
    #   '<数据集名>' = 只用那一个
    "primary_val":          _env('PRIMARY_VAL',           'weighted'),
    "val_weights":          _parse_weights(_env('VAL_WEIGHTS', '')),
    "freeze_encoder":       _env('FREEZE_ENCODER',        False,  bool),
    "unfreeze_epoch":       _env('UNFREEZE_EPOCH',        10,     int),   # 训练多少 epoch 后解冻深层 encoder
    "unfreeze_layers":      _env('UNFREEZE_LAYERS',       'down3,down4'),  # 解冻哪些层
    # checkpoint
    "checkpoint_dir":       _env('CHECKPOINT_DIR',        'models/unet/checkpoints'),
    "resume_path":          _env('RESUME_PATH',           '') or None,
    "init_path":            _env('INIT_PATH',             '') or None,
    "reset_best_dice":      _env('RESET_BEST_DICE',       False,  bool),
    "run_name":             _env('RUN_NAME',              'run'),
    # wandb
    "wandb_entity":         _env('WANDB_ENTITY',          'worldangle'),
    "wandb_project_name":   _env('WANDB_PROJECT_NAME',    'us-segmentation'),
    "wandb_enable":         _env('WANDB_ENABLE',          True,   bool),
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
        logging.warning(f"nvidia-smi 查询失败: {e}")
        return []


def _get_sys_free_mem_gb() -> float:
    """获取系统可用内存 (GB)，overcommit=2 下用 MemFree 更准确。"""
    try:
        with open('/proc/meminfo') as f:
            for line in f:
                if line.startswith('MemFree:'):
                    return int(line.split()[1]) / (1024 * 1024)
    except Exception:
        pass
    return float('inf')  # 读取失败不限制


def select_devices() -> Tuple[torch.device, List[int]]:
    """
    自动选择 GPU，按实时空闲显存排序。
    返回 (primary_device, logical_gpu_ids)

    ⚠ 顺序是这个函数唯一重要的东西：CUDA_VISIBLE_DEVICES 必须在**任何** torch.cuda.*
    调用之前设好。

    这里原本是坏的，而且坏得很安静。原版第一行就是 `torch.cuda.is_available()` ——
    那一下就会初始化 PyTorch 的设备枚举，而 CUDA runtime 只在那一刻读一次
    CUDA_VISIBLE_DEVICES。等到函数末尾再去 os.environ[...] = '2'，已经晚了，会被静默
    忽略；`torch.cuda.init()` 也不会重读。

    后果：日志信誓旦旦地写着"选择 1 张 GPU: 物理['2'] → 逻辑[0]"，进程却老老实实跑在
    物理 GPU 0 上。实测抓到过一次：日志说选了空闲 139.8GB 的 GPU 2，nvidia-smi 显示
    进程其实在 GPU 0，和另一个用户 78GB 的任务挤在一起，吞吐从 201 img/s 掉到 74。
    这个项目此前每一次训练，无论日志怎么说，都跑在物理 GPU 0 上。

    所以：先 nvidia-smi（纯 subprocess，不碰 torch），再设环境变量，最后才允许 torch
    看见 CUDA。
    """
    # 用户手动指定了 GPU，直接使用（这条路径一直是对的 —— 环境变量在进程启动前就设好了）
    if os.environ.get('CUDA_VISIBLE_DEVICES', '') != '':
        if not torch.cuda.is_available():
            logging.info("CUDA 不可用，使用 CPU")
            return torch.device('cpu'), []
        ids = list(range(torch.cuda.device_count()))
        logging.info(f"CUDA_VISIBLE_DEVICES 已设置，使用逻辑 GPU: {ids}")
        return torch.device('cuda:0'), ids

    free_mem_gb = _env('GPU_MEM_FREE_GB', 10.0, float)
    gpus = _query_gpu_free()  # [(物理idx, free_gb, total_gb)] 按空闲显存降序 —— 不碰 torch

    if not gpus:
        # nvidia-smi 没结果：要么没有 GPU，要么查询失败。这时才第一次问 torch。
        if not torch.cuda.is_available():
            logging.info("CUDA 不可用，使用 CPU")
            return torch.device('cpu'), []
        logging.warning("nvidia-smi 查询失败，回退到 cuda:0")
        return torch.device('cuda:0'), [0]

    # 选空闲显存 >= 阈值的 GPU（已按空闲显存降序）
    selected = [(idx, free, total) for idx, free, total in gpus if free >= free_mem_gb]
    if not selected:
        # 都不够，选空闲显存最大的那一个
        best = gpus[0]
        logging.warning(
            f"没有 GPU 空闲显存 >= {free_mem_gb}GB，"
            f"选择 GPU {best[0]}（{best[1]:.1f}GB 空闲 / {best[2]:.0f}GB 总）"
        )
        selected = [best]

    # 受系统内存约束：每张 GPU 约需 6GB 系统内存
    sys_free = _get_sys_free_mem_gb()
    max_gpus = max(1, int(sys_free / 6))
    if len(selected) > max_gpus:
        logging.warning(
            f"系统可用内存 {sys_free:.1f}GB，限制 GPU 数量 {len(selected)} → {max_gpus}"
        )
        selected = selected[:max_gpus]

    # MAX_GPUS 上限（默认 1）。这里的 H200 一次只允许占用一张卡，
    # 多卡会走 DataParallel，既违反该限制、也是唯一一次观察到 val loss 变 NaN 的配置。
    cap = _env('MAX_GPUS', 1, int)
    if cap > 0 and len(selected) > cap:
        selected = selected[:cap]
        logging.info(f"MAX_GPUS={cap}，只使用 GPU {[i for i, _, _ in selected]}")

    # 设置 CUDA_VISIBLE_DEVICES，把物理 id 映射为逻辑 id 0,1,2...
    # 到这一行为止，torch.cuda 一次都没有被碰过 —— 所以这次真的会生效。
    physical_ids = [str(idx) for idx, _, _ in selected]
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(physical_ids)

    if not torch.cuda.is_available():          # 第一次接触 CUDA，就在这里
        logging.info("CUDA 不可用，使用 CPU")
        return torch.device('cpu'), []

    logical_ids = list(range(len(selected)))
    for i, (pid, free, total) in enumerate(selected):
        tag = "← 主 GPU" if i == 0 else ""
        logging.info(
            f"  GPU {pid}(逻辑{i}) | {free:.1f}GB 空闲 / {total:.0f}GB 总 {tag}"
        )
    logging.info(f"选择 {len(logical_ids)} 张 GPU: 物理{physical_ids} → 逻辑{logical_ids}")

    # 别只相信日志 —— 这是上一版栽的跟头。回读 torch 实际看到的卡，和我们要的对上号。
    seen = torch.cuda.device_count()
    if seen != len(logical_ids):
        raise RuntimeError(
            f"CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']} 没有生效："
            f"torch 看到 {seen} 张卡，应该是 {len(logical_ids)} 张。"
            f"说明在 select_devices() 之前已经有别的代码碰过 torch.cuda。"
        )
    logging.info(f"  已核实: torch 看到 {seen} 张卡, "
                 f"{torch.cuda.get_device_name(0)} (物理 {physical_ids[0]})")

    return torch.device('cuda:0'), logical_ids


# ----------------------------------------------------------------------------------
#                               LOSS & METRICS
# ----------------------------------------------------------------------------------

def _unwrap(model: nn.Module) -> nn.Module:
    """DataParallel 解包，安全获取原始模型。"""
    return model.module if isinstance(model, nn.DataParallel) else model


def dice_coeff(prob: torch.Tensor, target: torch.Tensor,
               eps: float = 1e-6,
               multiclass: bool = False,
               reduce_batch_first: bool = False) -> torch.Tensor:
    """
    Soft Dice. MUST stay differentiable.

    This used to carry an @torch.no_grad() decorator, which silently turned the Dice
    term of the compound loss into a constant: dice_loss_from_logits returned a
    detached tensor, so every model up to now was trained on BCE alone. Both call
    sites that only need a metric (validate / evaluate_metrics) already run inside
    their own no_grad context, so the decorator bought nothing and cost the Dice
    gradient — which is exactly the term that counteracts the 96% background
    imbalance.
    """
    assert prob.size() == target.size()
    assert prob.dim() == 3 or not reduce_batch_first

    sum_dim  = (-1, -2) if not reduce_batch_first else (-1, -2, -3)
    inter    = 2 * (prob * target).sum(dim=sum_dim)
    sets_sum = prob.sum(dim=sum_dim) + target.sum(dim=sum_dim)
    sets_sum = torch.where(sets_sum == 0, inter, sets_sum)
    return ((inter + eps) / (sets_sum + eps)).mean()


def dice_loss_from_logits(logits: torch.Tensor, target: torch.Tensor,
                          multiclass: bool) -> torch.Tensor:
    # Cast to float32 first: fp16 sum over 576×544 pixels overflows (max ~65504)
    logits_f = logits.float()
    if multiclass:
        probs  = F.softmax(logits_f, dim=1)
        tgt_1h = F.one_hot(target, num_classes=logits.shape[1]).permute(0, 3, 1, 2).float()
        return 1 - dice_coeff(probs, tgt_1h, multiclass=True)
    else:
        probs = torch.sigmoid(logits_f)
        return 1 - dice_coeff(probs, target.float(), multiclass=False)


# ----------------------------------------------------------------------------------
#              PARTIAL-LABEL (MARGINAL) LOSS  —  3-channel model, 3 label modes
# ----------------------------------------------------------------------------------
#
# The model emits 3 channels: 0 = background, 1 = vein, 2 = artery. Masks arrive in the
# shared id space defined in dataPrepare (0 bg / 1 vein / 2 artery / 3 vessel-untyped),
# and each sample carries a `label_mode` saying which marginal its mask actually pins
# down. We supervise that marginal and nothing more.
#
#   full3   (Mus-V)     mask ∈ {0,1,2}. Everything is known. Plain 3-way CE + Dice.
#
#   vessel  (phantoms)  mask ∈ {0,3}. "There is a vessel here", type unknown. Supervise
#                       the merged probability p1+p2 against it. Because the loss
#                       depends on the SUM only, it is invariant to any reallocation of
#                       mass between vein and artery — the split is left free, and is
#                       decided by whatever the model learned from Mus-V. That is what
#                       lets us ask a phantom "does this tube look arterial or venous?"
#
#   artery  (mendeley)  mask ∈ {0,2}. The expert masks label the common carotid ARTERY.
#                       The jugular vein is usually in frame but UNLABELLED, sitting
#                       inside the "background" region. So supervise the marginal p2
#                       only. That loss depends on p2 alone, hence is invariant to how
#                       the remaining mass splits between background and vein — an
#                       unlabelled vein can be predicted as a vein with ZERO penalty.
#                       Supervising those pixels as background instead would teach the
#                       model that veins look like background and destroy the class.
#
#                       This invariance is real and was verified numerically: the loss
#                       is a function of p2 alone, and the softmax gradient therefore
#                       scales logit_0 and logit_1 in proportion to p0 and p1, which
#                       leaves the ratio p0:p1 untouched. mendeley genuinely does not
#                       train the vein — in either direction.
#
#                       The METRIC, however, did punish it. See evaluate_metrics.
#
# Everything is computed from log_softmax + logsumexp, so log(p1+p2) is exact and
# numerically stable; no epsilon fudging, and it is safe under autocast.

from models.losses import (LABEL_MODES, _soft_dice, _soft_tversky, _region_loss,
                           partial_label_loss)


@torch.no_grad()
def evaluate_metrics(logits: torch.Tensor, target: torch.Tensor,
                     modes: List[str]) -> Dict[str, Tuple[float, int]]:
    """
    Per-mode HARD Dice, keyed by metric name -> (sum_over_batch, n_samples).

    Hard, not soft: the prediction is thresholded (argmax) and the Dice is computed on
    the resulting binary mask. This is what the deployed system consumes and what the
    literature reports.

    THE BUG THIS FIXES
    ------------------
    An `artery`-mode dataset used to also emit `dice_vessel`, computed as

        dice(predicted_vein ∪ predicted_artery,  artery_only_label)

    with a comment claiming the unlabelled jugular vein was something "this metric will
    not punish". It punished it directly: every correctly-found vein pixel is a false
    positive against a label that only marks the artery. And `primary_score` fell
    through to exactly that number for mendeley, because an artery-mode dataset has no
    `dice_vein` key — so 0.1 of the checkpoint criterion was paying the model to
    suppress veins.

    It worked. Over the mixav_v1 run, mendeley dice_artery sat flat at ~0.95 from epoch
    3 while its dice_vessel climbed 0.751 -> 0.894 — and the only mechanism that lifts
    the merged score toward the artery-only score is the vein predictions dying off.
    Mus-V vein Dice fell 0.614 -> 0.574 over the same epochs. Two views of one event.
    In the PRIMARY decomposition from epoch 7 to 27, that term contributed +0.0030
    while Mus-V contributed -0.0055: it masked most of the damage it was causing.

    So: an artery-mode dataset now reports `dice_artery` and NOTHING else. What the
    model does with the unlabelled vein is a question for test time, not for the
    metric that picks the checkpoint. See test.py --vein-probe.
    """
    pred = F.softmax(logits.float(), dim=1).argmax(1)     # (B, H, W) hard class ids
    tgt  = target.squeeze(1).long()
    out: Dict[str, Tuple[float, int]] = {}

    def add(key, val, n):
        s, c = out.get(key, (0.0, 0))
        out[key] = (s + val * n, c + n)

    for mode in LABEL_MODES:
        sel = torch.tensor([m == mode for m in modes], device=logits.device)
        if not sel.any():
            continue
        p, y, n = pred[sel], tgt[sel], int(sel.sum())
        p_vessel = ((p == CLASS_VEIN) | (p == CLASS_ARTERY)).float()

        if mode == "full3":
            add("dice_vein",   _soft_dice((p == CLASS_VEIN).float(),
                                          (y == CLASS_VEIN).float()).item(), n)
            add("dice_artery", _soft_dice((p == CLASS_ARTERY).float(),
                                          (y == CLASS_ARTERY).float()).item(), n)
            add("dice_vessel", _soft_dice(p_vessel, (y > 0).float()).item(), n)

        elif mode == "vessel":
            add("dice_vessel", _soft_dice(p_vessel,
                                          (y == CLASS_VESSEL).float()).item(), n)

        elif mode == "artery":
            add("dice_artery", _soft_dice((p == CLASS_ARTERY).float(),
                                          (y == CLASS_ARTERY).float()).item(), n)
            # NO dice_vessel here. The vein in this frame is real and unlabelled;
            # scoring it as a false positive is what killed the vein class.

    return out


def primary_score(per_class: Dict[str, float]) -> float:
    """
    The scalar that drives checkpointing / early stopping / the LR scheduler, for ONE
    validation set. Each set contributes what its labels can actually support:

      full3  (Mus-V)     (vein + artery) / 2 — telling the two apart is the whole job
                         there, and merged vessel Dice would sit high while the model
                         confused them.
      artery (mendeley)  dice_artery. NOT dice_vessel — see evaluate_metrics.
      vessel (phantoms)  dice_vessel, which is all their labels can say.
    """
    if "dice_vein" in per_class and "dice_artery" in per_class:
        return (per_class["dice_vein"] + per_class["dice_artery"]) / 2
    if "dice_artery" in per_class:
        return per_class["dice_artery"]
    return per_class.get("dice_vessel", 0.0)


@torch.no_grad()
def validate(model: nn.Module, loader: DataLoader,
             cfg: Dict[str, Any], device: torch.device) -> Tuple[float, float, Dict[str, float]]:
    """Returns (loss, primary_dice, per_class_dice)."""
    model.eval()
    total_loss, n = 0.0, 0
    acc: Dict[str, Tuple[float, int]] = {}
    binary = cfg["num_classes"] == 1
    criterion = nn.BCEWithLogitsLoss() if binary else None
    autocast_dtype = device.type if device.type != 'mps' else 'cpu'

    for imgs, masks, modes in loader:
        imgs  = imgs.to(device, dtype=torch.float32, memory_format=torch.channels_last)
        masks = masks.to(device).long()
        # Legacy binary model: foreground is "any vessel", whatever its id. Masks now
        # carry real ids (1 vein / 2 artery / 3 vessel-untyped), so BCE must be given
        # {0,1} — feeding it a raw id of 3.0 would be silently meaningless.
        if binary:
            masks = (masks > 0).float()

        with torch.autocast(autocast_dtype, enabled=cfg["amp"], dtype=getattr(torch,cfg.get('amp_dtype','float16'))):
            logits = model(imgs)
            if binary:
                loss = (criterion(logits.squeeze(1), masks.squeeze(1))
                        + dice_loss_from_logits(logits, masks, multiclass=False))
            else:
                loss = partial_label_loss(logits, masks, list(modes), cfg=cfg)

        bs = imgs.size(0)
        if binary:
            # Hard Dice at 0.5, matching the 3-class path (which argmaxes) and the
            # deployed pipeline (which thresholds before the moment analysis).
            hard = (torch.sigmoid(logits.float()) > 0.5).float()
            d = dice_coeff(hard, masks.float(), multiclass=False).item()
            s, c = acc.get("dice_vessel", (0.0, 0))
            acc["dice_vessel"] = (s + d * bs, c + bs)
        else:
            for k, (s, c) in evaluate_metrics(logits, masks, list(modes)).items():
                ps, pc = acc.get(k, (0.0, 0))
                acc[k] = (ps + s, pc + c)

        total_loss += loss.item() * bs
        n += bs

    per_class = {k: s / c for k, (s, c) in acc.items() if c}
    return total_loss / n, primary_score(per_class), per_class


# ----------------------------------------------------------------------------------
#                                   TRAIN LOOP
# ----------------------------------------------------------------------------------

def build_train_gap_loader(batch_size: int, n: int = 441):
    """
    A slice of the musv TRAIN images, evaluated exactly like val: no augmentation, one
    pass, same metric. Its only job is to make the overfit gap readable in the log.

    It must be un-augmented, or it would not be comparable to val — you would be
    measuring "train images, distorted" against "val images, clean" and the gap would
    mix generalisation with augmentation difficulty. ReadDataset(augment=False) gives
    the stored originals, which is exactly what val serves too.
    """
    from torch.utils.data import Subset
    ds = ReadDataset('train', batch_size=batch_size, dataset_name='musv', augment=False)
    idx = list(range(0, len(ds), max(1, len(ds) // n)))[:n]   # spread across sequences
    return DataLoader(Subset(ds, idx), batch_size=batch_size, shuffle=False,
                      num_workers=0, pin_memory=torch.cuda.is_available())


def train(model: nn.Module, train_loader: DataLoader,
          val_loaders: Dict[str, DataLoader],
          cfg: Dict[str, Any], device: torch.device,
          train_gap_loader: DataLoader = None):

    # Freeze encoder if requested (for finetune stage)
    if cfg.get("freeze_encoder"):
        for name, param in _unwrap(model).named_parameters():
            if any(s in name for s in ['inc', 'down1', 'down2', 'down3', 'down4']):
                param.requires_grad = False
        frozen = sum(1 for p in _unwrap(model).parameters() if not p.requires_grad)
        total  = sum(1 for _ in _unwrap(model).parameters())
        logging.info(f"Encoder 已冻结: {frozen}/{total} 参数不参与训练 (inc, down1-down4)")

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

    binary    = cfg["num_classes"] == 1
    criterion = nn.BCEWithLogitsLoss() if binary else None
    scaler    = torch.amp.GradScaler('cuda', enabled=cfg["amp"] and device.type == 'cuda' and cfg.get('amp_dtype','float16')=='float16')

    # Optional CE re-weighting for the 3-class model. Background is ~96% of pixels,
    # vein ~1.9%, artery ~2.2%; the Dice term already counteracts this, so leave
    # unset unless the minority classes are visibly under-segmented.
    class_weights = None
    if not binary and cfg.get("ce_class_weights"):
        class_weights = torch.tensor(cfg["ce_class_weights"], dtype=torch.float32, device=device)
        logging.info(f"CE class weights: {cfg['ce_class_weights']}")

    start_epoch    = 1
    best_dice      = -math.inf
    no_improve     = 0
    prev_ckpt_path = None

    # Initialisation is deliberately separate from resume: new LR, scheduler and epoch.
    if cfg.get("init_path"):
        if cfg.get("resume_path"):
            raise ValueError("INIT_PATH and RESUME_PATH are mutually exclusive")
        initial = torch.load(cfg["init_path"], map_location=device, weights_only=False)
        _unwrap(model).load_state_dict(initial["model"])
        logging.info(f"Initialised model only from {cfg['init_path']}; fresh optimizer/scheduler")

    # Resume
    if cfg["resume_path"] and Path(cfg["resume_path"]).is_file():
        ckpt = torch.load(cfg["resume_path"], map_location=device)
        _unwrap(model).load_state_dict(ckpt["model"])
        try:
            optimizer.load_state_dict(ckpt["optimizer"])
            if scheduler and ckpt.get("scheduler") is not None:
                scheduler.load_state_dict(ckpt["scheduler"])
        except ValueError:
            logging.warning("Optimizer state skipped (parameter groups mismatch — expected when freeze_encoder changed)")
        start_epoch = ckpt.get("epoch", 1) + 1
        best_dice   = -math.inf if cfg.get("reset_best_dice") else ckpt.get("best_score", -math.inf)
        logging.info(f"Resumed from {cfg['resume_path']} @ epoch {start_epoch}")

    # WandB（登录失败则自动关闭，不阻塞训练）
    run = None
    if cfg.get("wandb_enable"):
        os.environ.setdefault('WANDB_DISABLE_STATS', 'true')
        os.environ.setdefault('WANDB_INIT_TIMEOUT', '10')
        try:
            run = wandb.init(
                entity=cfg["wandb_entity"],
                project=cfg["wandb_project_name"],
                config={k: v for k, v in cfg.items() if not k.startswith('wandb')},
                settings=wandb.Settings(disable_code=True, init_timeout=10),
            )
        except Exception as e:
            logging.warning(f"WandB 初始化失败（超时或认证错误），已跳过: {e}")
            run = None

    autocast_dtype = device.type if device.type != 'mps' else 'cpu'

    unfrozen = False  # 是否已解冻深层 encoder

    for epoch in range(start_epoch, cfg["epochs"] + 1):

        # 分阶段解冻：到达指定 epoch 后解冻 down3, down4
        if (cfg.get("freeze_encoder") and not unfrozen
                and cfg.get("unfreeze_epoch")
                and (epoch - start_epoch) >= cfg["unfreeze_epoch"]):
            unfreeze_names = [s.strip() for s in cfg["unfreeze_layers"].split(',')]
            for name, param in _unwrap(model).named_parameters():
                if any(s in name for s in unfreeze_names):
                    param.requires_grad = True
            # 重建 optimizer 以包含新解冻的参数，学习率减半
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
                f"Epoch {epoch}: 解冻 {unfreeze_names}，"
                f"仍冻结 {frozen}/{total} 参数，LR → {cfg['learning_rate'] * 0.5:.1e}"
            )

        model.train()
        epoch_loss = 0.0
        pbar = tqdm(total=len(train_loader.dataset),
                    desc=f"Epoch {epoch}/{cfg['epochs']}", unit="img")

        for imgs, masks, modes in train_loader:
            imgs  = imgs.to(device, dtype=torch.float32, memory_format=torch.channels_last)
            masks = masks.to(device).long()
            if binary:
                masks = (masks > 0).float()      # see validate()

            optimizer.zero_grad(set_to_none=True)

            with torch.autocast(autocast_dtype, enabled=cfg["amp"], dtype=getattr(torch,cfg.get('amp_dtype','float16'))):
                logits = model(imgs)
                if binary:
                    loss = (criterion(logits.squeeze(1), masks.squeeze(1))
                            + dice_loss_from_logits(logits, masks, multiclass=False))
                else:
                    loss = partial_label_loss(logits, masks, list(modes), class_weights, cfg)

            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite loss; refusing to continue an invalid training run')
            scaler.scale(loss).backward()
            if cfg["gradient_clip"]:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(_unwrap(model).parameters(), cfg["gradient_clip"], error_if_nonfinite=cfg.get('amp_dtype')=='bfloat16')
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
        val_results = {}
        for vname, vloader in val_loaders.items():
            vl, vd, per_class = validate(model, vloader, cfg, device)
            val_results[vname] = {"loss": vl, "dice": vd, "per_class": per_class}

        # ---- Which epoch's weights do we actually keep? -----------------------
        #
        # One model serves two very different jobs, and they are not equally
        # important. The phantom is what the rig actually runs on, so it leads. The
        # human sets are what make vein/artery separation possible at all, so they
        # follow. VAL_WEIGHTS states that priority as a number instead of leaving it
        # implicit in whichever dataset happens to be listed first.
        #
        # Each dataset contributes its OWN score (see primary_score): Mus-V gives
        # (vein+artery)/2 because separating them is the whole point there; the
        # phantoms give merged vessel Dice because that is all their labels can say.
        pv = cfg.get("primary_val", "weighted")

        if pv == "weighted":
            w = cfg["val_weights"] or {k: 1.0 for k in val_results}
            unknown = set(w) - set(val_results)
            if unknown:
                raise ValueError(f"VAL_WEIGHTS 里的 {unknown} 不在 VAL_DATASET "
                                 f"({list(val_results)}) 里。名字必须完全一致。")
            tot = sum(w.values())
            val_dice = sum(w[k] * val_results[k]["dice"] for k in w) / tot
            val_loss = sum(w[k] * val_results[k]["loss"] for k in w) / tot
            primary_name = "weighted"
        elif pv == "mean":
            val_dice = sum(v["dice"] for v in val_results.values()) / len(val_results)
            val_loss = sum(v["loss"] for v in val_results.values()) / len(val_results)
            primary_name = "mean"
        else:
            val_dice = val_results[pv]["dice"]
            val_loss = val_results[pv]["loss"]
            primary_name = pv

        # The overfit gap, logged every epoch instead of being reconstructed afterwards.
        # This experiment has two possible outcomes and they mean opposite things:
        #   gap shrinks AND val rises  -> it WAS capacity. Keep cutting.
        #   gap shrinks BUT val flat   -> capacity was not the binding constraint; the
        #                                 80 subjects simply do not span the vein, and
        #                                 no amount of regularisation invents an 81st.
        # Without this line you cannot tell those two apart from the log.
        gap_str = ""
        if train_gap_loader is not None:
            _, _, tg = validate(model, train_gap_loader, cfg, device)
            bits = [f"{k.replace('dice_','')}={tg[k]:.4f}" for k in sorted(tg)]
            gap = {k: tg[k] - val_results["musv"]["per_class"].get(k, 0.0)
                   for k in tg if k in val_results.get("musv", {}).get("per_class", {})}
            gstr = " ".join(f"{k.replace('dice_','')}+{v:.3f}" for k, v in sorted(gap.items()))
            gap_str = f"  |  TRAIN(musv): {' '.join(bits)}  [过拟合缺口 {gstr}]"

        parts = [f"train_loss={epoch_loss:.4f}", f"PRIMARY({primary_name})={val_dice:.4f}"]
        for vname, vr in val_results.items():
            detail = " ".join(f"{k.replace('dice_','')}={v:.4f}"
                              for k, v in sorted(vr["per_class"].items()))
            wtag = f"[w={cfg['val_weights'][vname]:g}]" if vname in cfg["val_weights"] else ""
            parts.append(f"{vname}{wtag}: {detail}")
        logging.info(f"Epoch {epoch}/{cfg['epochs']}: " + "  |  ".join(parts) + gap_str)
        if cfg.get("history_path"):
            history_path = Path(cfg["history_path"])
            history_path.parent.mkdir(parents=True, exist_ok=True)
            with history_path.open("a") as history_file:
                history_file.write(json.dumps(dict(epoch=epoch, loss=epoch_loss,
                    primary=val_dice, validation=val_results,
                    learning_rate=optimizer.param_groups[0]['lr'])) + "\n")

        if run:
            log_dict = {
                "train/epoch_loss": epoch_loss,
                "lr": optimizer.param_groups[0]['lr'],
                "epoch": epoch,
            }
            for vname, vr in val_results.items():
                log_dict[f"val/{vname}/loss"] = vr["loss"]
                log_dict[f"val/{vname}/dice"] = vr["dice"]
                for k, v in vr["per_class"].items():
                    log_dict[f"val/{vname}/{k}"] = v
            # Keep top-level val/loss and val/dice pointing to primary
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
    logging.info(f"Training complete. Best PRIMARY score: {best_dice:.4f}")


# ----------------------------------------------------------------------------------
#                              Val loader construction
# ----------------------------------------------------------------------------------

def build_val_loaders(specs: List[str], batch_size: int,
                      split: str = 'validation') -> Dict[str, DataLoader]:
    """
    Each spec is either one dataset name, or several joined with '+' to be POOLED
    into a single metric.

    Pooling matters for the phantoms: phantom_taobao and customer_3d_phantom are both
    binary vessel data and both are what the rig actually runs on, so they should be
    judged as one number. Scored apart, customer_3d_phantom's 4-image val split is
    pure noise. Pooled, the Dice is a per-sample mean over all 17 images.
    """
    from torch.utils.data import ConcatDataset

    loaders: Dict[str, DataLoader] = {}
    for spec in specs:
        names = [s.strip() for s in spec.split('+') if s.strip()]
        if len(names) == 1:
            ds = ReadDataset(split, batch_size=batch_size, dataset_name=names[0])
            loaders[spec] = ds.get_dataloader(shuffle=False, num_workers=0)
            continue

        parts = [ReadDataset(split, batch_size=1, dataset_name=n) for n in names]
        modes = {p.label_mode for p in parts}
        if len(modes) > 1:
            # Pooling e.g. an 'artery' set with a 'vessel' set would average two
            # different quantities into one meaningless number.
            raise ValueError(
                f"不能把 label_mode 不同的数据集池化成一个指标: '{spec}' -> {modes}。"
                f"只有语义相同的数据集才能合并（例如两个 vessel 仿体）。")
        logging.info(f"  验证集池化 '{spec}': "
                     + " + ".join(f"{n}({len(p)})" for n, p in zip(names, parts))
                     + f" = {sum(len(p) for p in parts)} 张")
        loaders[spec] = DataLoader(ConcatDataset(parts), batch_size=batch_size,
                                   shuffle=False, num_workers=0,
                                   pin_memory=torch.cuda.is_available())
    return loaders


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
    """只删除本次训练的上一个 checkpoint，不影响其他训练 run 的文件。"""
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
    try:
        validate_train_repeats(CONFIG)
    except ValueError as exc:
        parser.error(str(exc))
    if CONFIG["init_path"] and CONFIG["resume_path"]:
        parser.error("INIT_PATH and RESUME_PATH are mutually exclusive")
    if args.show_config:
        mode = "resume" if CONFIG["resume_path"] else "finetune" if CONFIG["init_path"] else "from_scratch"
        print(json.dumps({"training_mode": mode, "config": CONFIG,
                          "runtime": {"num_workers": _env("NUM_WORKERS", 8, int),
                                      "max_gpus": _env("MAX_GPUS", 1, int)}},
                         indent=2, ensure_ascii=False))
        sys.exit(0)
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')

    # A web review/exclusion must be reflected in rebuilt arrays before training.
    stale = check_stale([name.strip() for name in CONFIG['dataset'].split('+') if name.strip()])
    if stale:
        parser.error('数据缓存已过期，请先重新运行 data/pipeline/dataPrepare.py：' + ', '.join(stale))

    # 1. 自动检测空闲 GPU
    DEVICE, gpu_ids = select_devices()
    random.seed(CONFIG['seed']); np.random.seed(CONFIG['seed']); torch.manual_seed(CONFIG['seed'])
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(CONFIG['seed'])

    # 打印设备详情
    print("=" * 60)
    print("环境信息")
    print("=" * 60)
    logging.info(f"PyTorch  : {torch.__version__}")
    logging.info(f"CUDA     : {torch.version.cuda}")
    if gpu_ids:
        for gid in gpu_ids:
            prop = torch.cuda.get_device_properties(gid)
            logging.info(
                f"  逻辑 GPU {gid} | {prop.name} | "
                f"SM×{prop.multi_processor_count}"
            )
    else:
        logging.info("  使用 CPU")

    # 2. 计算实际 batch size（每 GPU × GPU 数量）
    n_gpus      = max(len(gpu_ids), 1)
    total_batch = CONFIG["batch_size_per_gpu"] * n_gpus
    logging.info(f"GPU 数量: {n_gpus}  |  batch/GPU: {CONFIG['batch_size_per_gpu']}  |  total batch: {total_batch}")

    # 3. 数据加载
    print("=" * 60)
    print("加载数据集")
    print("=" * 60)
    dataset_mode = CONFIG["dataset"]
    logging.info(f"训练数据集模式: {dataset_mode}")

    # `DATASET` is either one dataset name, or several joined with '+' to train on
    # their union — e.g. musv+phantom_taobao+mendeley. Each sub-dataset carries its
    # own label_mode, so the partial-label loss routes every sample correctly even
    # though they disagree about what their masks mean. "mixed" is the old alias.
    if dataset_mode == "mixed":
        dataset_mode = "phantom_taobao+mendeley"

    train_names = [s.strip() for s in dataset_mode.split('+') if s.strip()]

    # Augmentation now runs on the CPU, per sample, inside __getitem__ — so the loader
    # workers do real work and there must be enough of them to keep the GPU fed.
    n_workers = _env('NUM_WORKERS', 8, int)

    if len(train_names) > 1:
        from torch.utils.data import ConcatDataset
        parts = [ReadDataset('train', batch_size=1, dataset_name=n,
                             train_repeat=CONFIG['train_repeats'].get(n)) for n in train_names]
        CONFIG['effective_train_repeats'] = {n: p.repeat for n, p in zip(train_names, parts)}
        total = sum(len(p) for p in parts)
        for n, p in zip(train_names, parts):
            logging.info(f"  + {n:22s} {len(p):6d} 样本/epoch  ({len(p)/total*100:4.1f}%)  "
                         f"label_mode={p.label_mode}")
        train_loader = DataLoader(ConcatDataset(parts), batch_size=total_batch,
                                  shuffle=True, num_workers=n_workers, drop_last=False,
                                  pin_memory=torch.cuda.is_available(),
                                  persistent_workers=n_workers > 0,
                                  prefetch_factor=4 if n_workers > 0 else None)
    else:
        train_ds     = ReadDataset('train', batch_size=total_batch, dataset_name=train_names[0],
                                   train_repeat=CONFIG['train_repeats'].get(train_names[0]))
        CONFIG['effective_train_repeats'] = {train_names[0]: train_ds.repeat}
        train_loader = train_ds.get_dataloader(shuffle=True, num_workers=n_workers)

    # val — 逗号分隔多个验证集；一项里用 '+' 连接则把它们**池化成一个指标**。
    # 例如 phantom_taobao+customer_3d_phantom：两个都是二分类仿体、都是实机实验用的
    # 东西，分开算没意义（customer_3d 的 val 只有 4 张图），池化后按样本数加权成一个
    # 血管 Dice。数据集本身仍然是分开的（增强倍数不同），只有评估时合并。
    val_loaders = build_val_loaders(CONFIG["val_datasets"], total_batch)
    print(f"训练批次: {len(train_loader)}  验证集: {list(val_loaders.keys())}")

    # 4. 创建模型
    print("\n创建 U-Net 模型...")
    model = UNet(n_channels=CONFIG["n_channels"],
                 n_classes=CONFIG["num_classes"],
                 bilinear=CONFIG["bilinear"],
                 base_ch=CONFIG["base_ch"],
                 dropout=CONFIG["dropout"])
    model = model.to(DEVICE, memory_format=torch.channels_last)
    logging.info(f"UNet base_ch={CONFIG['base_ch']}  dropout={CONFIG['dropout']}")

    # 5. 多 GPU 并行（DataParallel）
    if len(gpu_ids) > 1:
        model = nn.DataParallel(model, device_ids=gpu_ids)
        logging.info(f"DataParallel 已启用，使用 GPU: {gpu_ids}")

    total_params = sum(p.numel() for p in _unwrap(model).parameters())
    logging.info(f"模型参数: {total_params:,}")

    # 6. 开始训练
    print("\n" + "=" * 60)
    print("开始训练")
    print("=" * 60)
    # Log the train/val gap on musv every epoch — the whole point of the v3 experiment.
    gap_loader = None
    if _env('LOG_TRAIN_GAP', True, bool) and 'musv' in val_loaders:
        gap_loader = build_train_gap_loader(total_batch)
        logging.info(f"过拟合缺口监控已开启: musv train 的 {len(gap_loader.dataset)} 张"
                     f"（不增强，和 val 同一套评估）")

    train(model, train_loader, val_loaders, CONFIG, DEVICE, gap_loader)
    print("训练完成。")
