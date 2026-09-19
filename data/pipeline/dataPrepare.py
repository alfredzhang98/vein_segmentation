import os
import sys
import random
import hashlib
from pathlib import Path

# 限制线程数，防止共享服务器上 fork/内存分配失败
for _k in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_k, '1')

import cv2
import numpy as np
import pandas as pd
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader

# Repo root on sys.path, so this works both as `common.dataPrepare` and as a script.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from data.pipeline.DataInfo import DataInfo


# ---------------------------------------------------------------------------
# ONE label space, shared by every dataset
# ---------------------------------------------------------------------------
#
# Every mask, from every source, is remapped at NPZ-build time into THIS id space:
#
#     0 = background
#     1 = vein
#     2 = artery
#     3 = vessel, type unknown          <- phantoms only; never a model output
#
# This used to be per-dataset. mask_class_map sent mendeley's 255 to id 1 and the
# phantoms' 255 to id 1 as well, so the SAME id meant "vein" in Mus-V, "artery" in
# mendeley and "untyped vessel" in the phantoms. Nothing crashed, because every
# binary-mode code path tested `y > 0` and never looked at which id it was — but the
# ambiguity was already producing silent errors: test.py colourises id 1 as RED =
# vein, so every mendeley ground-truth figure ever saved drew the common carotid
# ARTERY as a vein. Any future `mask == 2` would have come back empty on mendeley.
#
# Now the id IS the meaning, everywhere, and a mis-routed sample raises instead of
# quietly scoring the wrong thing.
#
# The model still emits exactly 3 channels (bg / vein / artery). Id 3 is a LABEL, not
# a class: it says "there is a vessel here and I don't know which kind", and the loss
# (train.py: partial_label_loss) supervises only the marginal p1+p2 against it, which
# leaves the vein/artery split free. Which marginal a sample gets is decided by its
# `label_mode`:
#
#   "full3"   Mus-V     mask ∈ {0,1,2}  Everything known. Plain 3-way CE + Dice.
#   "vessel"  phantoms  mask ∈ {0,3}    Supervises p1+p2 only — the split stays free.
#   "artery"  mendeley  mask ∈ {0,2}    Supervises p2 only. The jugular vein is in
#                                       frame but UNLABELLED, so the negative region
#                                       is never pushed toward background; a vein
#                                       predicted there costs nothing. Supervising it
#                                       as background would teach the model that veins
#                                       look like background and destroy the class.

CLASS_BG, CLASS_VEIN, CLASS_ARTERY, CLASS_VESSEL = 0, 1, 2, 3
MODEL_N_CLASSES = 3          # the network's output width — id 3 is never predicted

# Which ids may legally appear in a mask, per label_mode. Enforced at NPZ build time
# and asserted again in the loss, so a bad mask_class_map fails loudly and early.
MODE_IDS = {
    "full3":  {CLASS_BG, CLASS_VEIN, CLASS_ARTERY},
    "vessel": {CLASS_BG, CLASS_VESSEL},
    "artery": {CLASS_BG, CLASS_ARTERY},
}


# ---------------------------------------------------------------------------
# Per-dataset configs — all augmentation and path parameters baked in here
# ---------------------------------------------------------------------------

# NOTE ON `aug_times_train`
# -------------------------
# This used to be "how many augmented copies to BAKE INTO the NPZ". It is now "how
# many times each original is drawn per epoch", and the augmentation is applied fresh
# on every draw (ReadDataset.__getitem__). The epoch size and the mix between datasets
# are therefore unchanged — but the model no longer sees the same 24,874 fixed images
# every single epoch. See the header of ReadDataset for why that mattered.

DATASET_CONFIGS = {
    "phantom_taobao": {
        "source_type":     "csv",
        "data_dir":        Path("data/datasets/phantom_taobao"),
        "npz_prefix":      "augmented_phantom_taobao_1",
        "aug_times_train": 40,   # 20 -> 40: musv 提到 8x 后要对冲，否则仿体在混合训练集里被稀释到 7%
        "mask_class_map":  {0: 0, 255: CLASS_VESSEL},   # gel tube: a vessel of unknown type
        "label_mode":      "vessel",
        "target_size":     (576, 544),      # (H, W)
        "color_mode":      "gray",
        "crop":            (10, 15, 50, 50),  # top, bottom, left, right pixels to remove
        "seed":            42,
        "fit_mode":        "croppad",   # croppad：不缩放，1 网络像素 = 1 crop 像素，逆映射零重采样
        "aug_scale":       (0.8, 1.2),   # ±20% 各向同性缩放（4cm/5cm 档位 mm/px 差 25%）
        "aug_gamma":       (70, 140),    # ±30% gamma —— 超声增益变化
        "aug_speckle":     (0.85, 1.15), # 乘性散斑噪声（超声的散斑是乘性的，不是加性的）
        "aug_blur":        0.3,          # 模糊/锐化二选一，模拟不同探头与聚焦
        "aug_hflip":       True,
        "aug_vflip":       True,            # a gel phantom has no fixed anatomical
                                            # up/down, and with only 64 original
                                            # images the extra diversity is worth
                                            # more than the physical implausibility
    },
    "mendeley": {
        "source_type":     "folder",
        "data_dir":        Path("data/datasets/mendeley_data/Common Carotid Artery Ultrasound Images"),
        "images_dir":      Path("data/datasets/mendeley_data/Common Carotid Artery Ultrasound Images/US images"),
        "masks_dir":       Path("data/datasets/mendeley_data/Common Carotid Artery Ultrasound Images/Expert mask images"),
        "npz_out_dir":     Path("data/datasets/mendeley_data"),
        "npz_prefix":      "augmented_mendeley_1",
        "aug_times_train": 5,
        # 255 is the COMMON CAROTID ARTERY -> id 2. It used to be mapped to id 1, which
        # is the vein. Nothing read the id (every path tested `y > 0`), so it never
        # crashed — it just drew every mendeley ground truth as a red vein.
        "mask_class_map":  {0: 0, 255: CLASS_ARTERY},
        "label_mode":      "artery",        # expert masks label the CCA only
        "target_size":     (576, 544),
        "color_mode":      "rgb_to_gray",   # load RGB then convert
        "crop":            None,
        "seed":            42,
        "fit_mode":        "letterbox",   # letterbox：749x709 塞不进 576x544，硬裁会毁掉 57 张动脉。纯辅助域，缩放无代价
        "aug_scale":       (0.8, 1.2),   # ±20% 各向同性缩放（4cm/5cm 档位 mm/px 差 25%）
        "aug_gamma":       (70, 140),    # ±30% gamma —— 超声增益变化
        "aug_speckle":     (0.85, 1.15), # 乘性散斑噪声（超声的散斑是乘性的，不是加性的）
        "aug_blur":        0.3,          # 模糊/锐化二选一，模拟不同探头与聚焦
        "aug_hflip":       True,
        "aug_vflip":       False,           # real neck anatomy: the probe face is
                                            # always the top of a B-mode frame, so a
                                            # vertically flipped carotid is an image
                                            # that cannot physically occur. (This
                                            # silently defaulted to True before.)
    },
    "customer_3d_phantom": {
        "source_type":     "csv",
        "data_dir":        Path("data/datasets/customer_3d_phantom"),
        "meta_file":       Path("data/datasets/customer_3d_phantom/meta_customer_3d_phantom_1.csv"),
        "npz_prefix":      "augmented_customer_3d_phantom_1",
        "aug_times_train": 40,              # 同上，与 phantom_taobao 保持一致
        "mask_class_map":  {0: 0, 255: CLASS_VESSEL},
        "label_mode":      "vessel",
        "target_size":     (576, 544),       # match pre-trained model input
        "color_mode":      "gray",
        "crop":            None,             # no crop, directly resize 461x481 → 576x544
        "seed":            42,
        "fit_mode":        "croppad",   # croppad：不缩放，1 网络像素 = 1 crop 像素，逆映射零重采样
        "aug_scale":       (0.8, 1.2),   # ±20% 各向同性缩放（4cm/5cm 档位 mm/px 差 25%）
        "aug_gamma":       (70, 140),    # ±30% gamma —— 超声增益变化
        "aug_speckle":     (0.85, 1.15), # 乘性散斑噪声（超声的散斑是乘性的，不是加性的）
        "aug_blur":        0.3,          # 模糊/锐化二选一，模拟不同探头与聚焦
        # --- augmentation overrides ---
        "aug_rotation_limit": 10,            # ±10° (was ±5°)
        "aug_hflip":          True,
        "aug_vflip":          True,           # enable vertical flip
        "aug_brightness":     (-0.1, 0.1),   # ±10% (was ±5%)
        "aug_contrast":       (-0.1, 0.1),
        "aug_bc_prob":        0.5,           # 50% (was 30%)
        "aug_noise_prob":     0.3,           # 30% (was 15%)
    },
    "musv": {
        # Mus-V — Multimodal Ultrasound Vascular Image Segmentation.
        # 105 probe sweeps, 3114 frames. Masks are already {0,1,2} label maps
        # (verified: every one of the 105 sequences contains both 1 and 2), so
        # mask_class_map is the identity. Class 1 collapses to zero area in 272
        # frames and has 3.6x the area variance of class 2 — compressible under
        # probe force — which is what identifies it as the vein.
        "source_type":     "musv",
        "data_dir":        Path("data/datasets/Mus-V/Multimodal Ultrasound Vascular Image Segmentation"),
        "npz_out_dir":     Path("data/datasets/Mus-V"),
        "aug_out_dir":     Path("data/datasets/Mus-V"),
        "npz_prefix":      "augmented_musv_1",
        "aug_times_train": 8,               # 3 -> 8: vein 严重过拟合 (train 0.855 / val 0.609)，
                                    # 80 个序列盖不住静脉的外观分布，靠增强扩
        # Already in model space. That class 1 is the VEIN and class 2 the ARTERY was
        # originally INFERRED from area statistics (class 1 collapses to zero area in
        # 272 frames and has 3.6x the area variance — compressible under probe force).
        # It is now independently CONFIRMED by a different dataset: mendeley's expert
        # masks label the common carotid ARTERY, and the channel trained as class 2
        # from Mus-V predicts them at Dice 0.954. Were the two swapped, that number
        # would be near zero.
        "mask_class_map":  {0: 0, 1: CLASS_VEIN, 2: CLASS_ARTERY},
        "label_mode":      "full3",
        "target_size":     (576, 544),      # keep identical to the other datasets
        "color_mode":      "rgb_to_gray",   # 3-channel PNGs, but saturation ~0
        "crop":            None,
        "seed":            42,
        "fit_mode":        "croppad",   # croppad：不缩放，1 网络像素 = 1 crop 像素，逆映射零重采样
        "aug_scale":       (0.8, 1.2),   # ±20% 各向同性缩放（4cm/5cm 档位 mm/px 差 25%）
        "aug_gamma":       (70, 140),    # ±30% gamma —— 超声增益变化
        "aug_speckle":     (0.85, 1.15), # 乘性散斑噪声（超声的散斑是乘性的，不是加性的）
        "aug_blur":        0.3,          # 模糊/锐化二选一，模拟不同探头与聚焦
        # --- augmentation overrides ---
        "aug_rotation_limit": 10,
        "aug_hflip":          True,
        "aug_vflip":          False,        # probe face is always at the top of a
                                            # B-mode frame — flipping it vertically
                                            # produces anatomy that cannot occur
        "aug_brightness":     (-0.25, 0.25),
        "aug_contrast":       (-0.25, 0.25),
        "aug_bc_prob":        0.7,
        "aug_noise_prob":     0.4,
    },
}


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def read_npz_file(npz_file: Path) -> tuple:
    """Read NPZ file and return (images, masks, image_type)."""
    if not Path(npz_file).exists():
        raise FileNotFoundError(f"NPZ file not found: {npz_file}")
    data = np.load(npz_file, allow_pickle=True)
    return data['images'], data['masks'], data['image_type']


# ---------------------------------------------------------------------------
# fit / unfit — get a cropped frame to the model's input size, invertibly.
# Shared verbatim by training AND inference, so the two cannot drift apart.
# ---------------------------------------------------------------------------
#
# The model is a pure pixel->pixel function. It knows nothing about millimetres, and
# nothing here needs to: geometry is measured downstream, in the CROP coordinate
# system, after the predicted mask has been mapped back through unfit_mask(). So the
# only two things fit() must guarantee are:
#
#   1. It is exactly invertible.       -> the mask can go back to the crop frame.
#   2. It never changes aspect ratio.  -> artery-vs-vein is a SHAPE judgement (an
#      artery is round and non-collapsible, a vein is flattened and compressible).
#      A plain resize to 576x544 scales H and W by different factors — Mus-V's
#      600x400 sweeps got stretched 1.42:1, turning round arteries into vein-shaped
#      ellipses — which attacks the exact cue the model has to learn.
#
# Two modes satisfy both:
#
#   "croppad"    No scaling at all. Pad if the frame is smaller than the target, crop
#                if larger. One network pixel IS one crop-frame pixel, so unfit is a
#                pure offset with zero resampling loss. Used wherever the frame fits:
#                phantom_taobao and customer_3d_phantom lose 0% of their vessel pixels
#                this way, Mus-V loses 0.13%.
#
#   "letterbox"  Uniform downscale, then pad. Needed for mendeley, whose 749x709
#                frames cannot be cropped to 576x544 without destroying the carotid in
#                57 of 1100 images (worst case keeps 11% of it). mendeley is a pure
#                auxiliary domain — no geometry is ever measured on it and its
#                calibration is unknown — so scaling it costs nothing.
#
# Vertical alignment is TOP, not centre: row 0 of a B-mode frame is the probe face,
# so anchoring the top keeps the skin line at row 0 across every dataset and every
# depth setting. Horizontal alignment is CENTRE, because the probe axis is the middle
# of the frame (offset_x is measured from crop_w / 2).

def fit(img: np.ndarray, mask: np.ndarray, target_hw: tuple, mode: str = "croppad") -> tuple:
    """Returns (img_out, mask_out, params). params is what unfit_mask() inverts."""
    TH, TW = target_hw
    H, W = img.shape[:2]

    if mode == "letterbox":
        scale = min(TH / H, TW / W)
    elif mode == "croppad":
        scale = 1.0
    else:
        raise ValueError(f"Unknown fit mode: {mode!r} (croppad | letterbox)")

    if scale != 1.0:
        nh, nw = int(round(H * scale)), int(round(W * scale))
        img_s = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
        # NEAREST on the mask: class ids must not be interpolated into ids that do
        # not exist (blending 0 and 2 would invent a 1 = "vein" out of nothing).
        mask_s = None if mask is None else cv2.resize(mask, (nw, nh),
                                                      interpolation=cv2.INTER_NEAREST)
    else:
        nh, nw, img_s, mask_s = H, W, img, mask

    # Vertical: anchored to the top. sy=dy=0 means "keep row 0" — if the frame is
    # taller than the target we drop rows off the BOTTOM (deep far field), and if it
    # is shorter we pad the bottom. Either way the probe face stays at row 0.
    sy = dy = 0
    # Horizontal: centred, so the probe axis stays at the middle of the frame.
    sx = max(0, (nw - TW) // 2)
    dx = max(0, (TW - nw) // 2)
    ch, cw = min(nh, TH), min(nw, TW)

    img_o = np.zeros((TH, TW), dtype=img.dtype)
    img_o[dy:dy + ch, dx:dx + cw] = img_s[sy:sy + ch, sx:sx + cw]

    mask_o = None
    if mask_s is not None:
        mask_o = np.zeros((TH, TW), dtype=mask.dtype)
        mask_o[dy:dy + ch, dx:dx + cw] = mask_s[sy:sy + ch, sx:sx + cw]

    return img_o, mask_o, dict(mode=mode, scale=scale, H=H, W=W, nh=nh, nw=nw,
                               sy=sy, sx=sx, dy=dy, dx=dx, ch=ch, cw=cw)


def unfit_mask(mask_fit: np.ndarray, p: dict) -> np.ndarray:
    """Take a mask in model-input space back to the pre-fit (crop-frame) grid."""
    scaled = np.zeros((p["nh"], p["nw"]), dtype=mask_fit.dtype)
    scaled[p["sy"]:p["sy"] + p["ch"], p["sx"]:p["sx"] + p["cw"]] = \
        mask_fit[p["dy"]:p["dy"] + p["ch"], p["dx"]:p["dx"] + p["cw"]]
    if p["scale"] == 1.0:
        return scaled                                    # exact: pure offset, no resample
    return cv2.resize(scaled, (p["W"], p["H"]), interpolation=cv2.INTER_NEAREST)


def isotropic_zoom(img: np.ndarray, mask: np.ndarray, s: float) -> tuple:
    """
    Zoom the content by ONE factor s, keeping the canvas size. s>1 zooms in (centre
    crop), s<1 zooms out (centre pad).

    Hand-rolled rather than using A.Affine(scale=(0.8, 1.2)): albumentations samples
    the x and y scale INDEPENDENTLY there, which turns a circle into anything from a
    0.70:1 to a 1.47:1 ellipse — measured, not assumed. That would silently undo the
    entire point of fit(), since artery-vs-vein is a shape judgement.

    Why zoom at all: the phantom is acquired at two depth settings (4 cm and 5 cm),
    whose mm/px differ by 25%, and the online 1280x720 stream is coarser again than
    the stored PNGs. The model has to be robust to apparent-size changes it will meet
    at deployment. This is train-time only — inference never scales, so the exact
    pixel correspondence fit() gives us is untouched.
    """
    H, W = img.shape[:2]
    nh, nw = max(1, int(round(H * s))), max(1, int(round(W * s)))
    img_r = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_CUBIC if s > 1 else cv2.INTER_AREA)
    msk_r = None if mask is None else cv2.resize(mask, (nw, nh),
                                                 interpolation=cv2.INTER_NEAREST)

    def place(src, dtype):
        out = np.zeros((H, W), dtype=dtype)
        if s >= 1.0:                                     # zoom in -> centre crop
            sy, sx = (nh - H) // 2, (nw - W) // 2
            out[:] = src[sy:sy + H, sx:sx + W]
        else:                                            # zoom out -> centre pad
            dy, dx = (H - nh) // 2, (W - nw) // 2
            out[dy:dy + nh, dx:dx + nw] = src
        return out

    return place(img_r, img.dtype), (None if msk_r is None else place(msk_r, mask.dtype))


# What actually lives in the NPZ now is the ORIGINAL frame: loaded, cropped, remapped
# into the shared id space, and fit() to target_size. Nothing else. Augmentation moved
# to ReadDataset.__getitem__ and happens at training time, so an augmentation setting
# no longer invalidates the cached arrays — only these five keys do. (The old list
# hashed every aug_* key, because those pixels really were baked in; it also hashed a
# `_FINGERPRINT_DEFAULTS` dict that had `aug_scale` written twice and four keys
# indented into it by accident, which silently did nothing.)
_FINGERPRINT_KEYS = (
    "mask_class_map", "target_size", "color_mode", "crop", "fit_mode",
)
_FINGERPRINT_DEFAULTS = {"fit_mode": "croppad"}


def aug_fingerprint(dataset_name: str) -> str:
    cfg = DATASET_CONFIGS[dataset_name]
    items = []
    for k in _FINGERPRINT_KEYS:
        v = cfg.get(k, _FINGERPRINT_DEFAULTS.get(k))
        if isinstance(v, dict):
            v = tuple(sorted(v.items()))
        items.append(f"{k}={tuple(v) if isinstance(v, list) else v!r}")
    return hashlib.sha1("|".join(items).encode()).hexdigest()[:12]


# ---------------------------------------------------------------------------
# Augmentation — built from a config, used ONLINE (ReadDataset.__getitem__)
# ---------------------------------------------------------------------------

def build_train_transform(cfg: dict) -> A.Compose:
    """
    Random augmentation for one training sample. Applied fresh on every __getitem__,
    so each epoch shows the model a different view of the same source frame.

    GEOMETRY — every op here scales both axes together or is rigid. Aspect ratio is
    load-bearing: an artery is round and a compressed vein is flat, and that is the
    ONLY cue a single grayscale frame offers for telling them apart. Nothing in this
    block may stretch one axis against the other. (Isotropic zoom is separate — see
    isotropic_zoom, which is hand-rolled precisely because A.Affine(scale=…) samples
    the x and y scale INDEPENDENTLY and would turn a circle into a 0.70:1..1.47:1
    ellipse.)

    APPEARANCE — intensity only, never geometry. Gamma and multiplicative (speckle-
    like) noise are the two that actually match how B-mode appearance varies across
    gain settings and probes; plain additive Gaussian noise does not.
    """
    TH, TW = cfg["target_size"]
    geom = [
        # Upscale 1.5x before rotation to avoid black corners, then back down.
        A.Resize(int(TH * 1.5), int(TW * 1.5), interpolation=cv2.INTER_CUBIC, p=1.0),
        A.Rotate(limit=cfg.get("aug_rotation_limit", 10), p=0.5,
                 interpolation=cv2.INTER_CUBIC,
                 border_mode=cv2.BORDER_REPLICATE, fill=0),
        A.HorizontalFlip(p=0.5 if cfg.get("aug_hflip", True) else 0.0),
        A.VerticalFlip(p=0.5 if cfg.get("aug_vflip", True) else 0.0),
        A.Resize(TH, TW, interpolation=cv2.INTER_AREA, p=1.0),
    ]
    appear = [
        A.RandomBrightnessContrast(
            brightness_limit=cfg.get("aug_brightness", (-0.1, 0.1)),
            contrast_limit=cfg.get("aug_contrast", (-0.1, 0.1)),
            p=cfg.get("aug_bc_prob", 0.6)),
        A.GaussNoise(std_range=(0.05, 0.1), p=cfg.get("aug_noise_prob", 0.3)),
    ]
    if cfg.get("aug_gamma"):
        appear.append(A.RandomGamma(gamma_limit=cfg["aug_gamma"], p=0.5))
    if cfg.get("aug_speckle"):
        appear.append(A.MultiplicativeNoise(multiplier=cfg["aug_speckle"],
                                            per_channel=False, elementwise=True, p=0.4))
    if cfg.get("aug_blur"):
        appear.append(A.OneOf([
            A.GaussianBlur(blur_limit=(3, 5), p=1.0),
            A.Sharpen(alpha=(0.1, 0.3), lightness=(0.9, 1.1), p=1.0),
        ], p=cfg["aug_blur"]))

    return A.Compose(geom + appear, additional_targets={'mask': 'mask'})


def normalize_to_tensor(img_u8: np.ndarray, mask_u8: np.ndarray) -> tuple:
    """uint8 (H,W) -> float32 (1,H,W) in [-1,1], and mask -> int64 (1,H,W) class ids."""
    img = (img_u8.astype(np.float32) / 255.0 - 0.5) / 0.5
    return img[np.newaxis], mask_u8.astype(np.int64)[np.newaxis]


def _npz_paths(name: str) -> tuple:
    cfg = DATASET_CONFIGS[name]
    out = cfg.get("npz_out_dir", cfg["data_dir"])
    return (out / f"{cfg['npz_prefix']}_train.npz",
            out / f"{cfg['npz_prefix']}.fingerprint")


def write_fingerprint(name: str) -> None:
    """Sidecar, not NPZ metadata — rewriting a 5.7 GB array to change one string is absurd."""
    _, stamp = _npz_paths(name)
    stamp.write_text(aug_fingerprint(name))


def check_stale(names=None) -> list:
    """
    Which datasets' NPZ no longer match their current augmentation config?

    An NPZ with no fingerprint is treated as STALE, not as OK. Guessing "probably
    fine" is exactly the quiet failure this is meant to prevent: an augmentation
    setting changes, nothing complains, and training silently keeps using arrays
    baked before it. If you have verified by hand that an un-fingerprinted NPZ still
    matches its config, say so explicitly with `--stamp <dataset>`.

    Report goes to stderr; the bare names go to stdout for run_train.sh to consume.
    """
    stale = []
    for name in (names or DATASET_CONFIGS):
        npz, stamp = _npz_paths(name)
        want = aug_fingerprint(name)
        if not npz.exists():
            print(f"  {name:22s} STALE  (NPZ 不存在)", file=sys.stderr)
            stale.append(name)
        elif not stamp.exists():
            print(f"  {name:22s} STALE  (无指纹，无法验证 → 保守当作过期。"
                  f"若确认没变，用 --stamp {name})", file=sys.stderr)
            stale.append(name)
        elif stamp.read_text().strip() != want:
            print(f"  {name:22s} STALE  (配置已改: {stamp.read_text().strip()} -> {want})",
                  file=sys.stderr)
            stale.append(name)
        else:
            print(f"  {name:22s} OK     ({want})", file=sys.stderr)
    return stale


# ---------------------------------------------------------------------------
# DataPipeline  —  build the NPZ cache of ORIGINALS. No augmentation here.
# ---------------------------------------------------------------------------

class DataPipeline(DataInfo):
    """
      1. Load rows (from CSV or folder scan depending on dataset).
      2. Split ORIGINAL images into train / val / test — by SEQUENCE for Mus-V, since
         consecutive frames of a probe sweep are near-duplicates and a frame-level
         split would put the same vessel in both train and test.
      3. Load → crop → remap ids into the shared space → fit() to target_size.
      4. Save as uint8. That is the whole job.

    WHAT CHANGED, AND WHY IT MATTERED
    ---------------------------------
    This used to run the random augmentation HERE, aug_times_train times per image,
    and bake the results into the NPZ as normalised float32. ReadDataset then just
    indexed that fixed array. So every epoch showed the model the IDENTICAL 24,874
    images — 2,203 Mus-V frames (from only 80 probe sweeps) frozen into 8 fixed
    variants each, forever.

    Measured consequence: train loss fell monotonically 0.575 -> 0.074 while val vein
    Dice peaked at epoch 7 (0.614) and decayed to 0.574 by epoch 27. The artery, a
    round dark blob that generalises easily, was unaffected. A 31M-parameter U-Net
    memorises 80 veins in seven epochs.

    It also cost 39 GB of NPZ on disk — read fully into RAM, not mmapped — to buy LESS
    appearance diversity than online augmentation gives for free. Storing the uint8
    originals instead is ~20x smaller, and with augmentation moved to
    ReadDataset.__getitem__ every epoch draws a fresh view: 200 epochs x 2,203 frames
    is 440k distinct views rather than 17,624 fixed ones.

    Usage:
        DataPipeline("phantom_taobao").run()
        DataPipeline("mendeley").run(save_png=False)
    """

    def __init__(self, dataset_name: str = "phantom_taobao",
                 train_ratio: float = 0.7,
                 val_ratio: float = 0.15,
                 test_ratio: float = 0.15):

        if dataset_name not in DATASET_CONFIGS:
            raise ValueError(f"Unknown dataset '{dataset_name}'. "
                             f"Available: {list(DATASET_CONFIGS)}")
        self.dataset_name = dataset_name
        self.cfg = DATASET_CONFIGS[dataset_name]

        super().__init__()  # builds DataInfo (phantom paths); key attrs overwritten below

        # Override meta_file for datasets that specify their own CSV path
        if "meta_file" in self.cfg:
            self.meta_file = self.cfg["meta_file"]

        self.train_ratio = train_ratio
        self.val_ratio   = val_ratio
        self.test_ratio  = test_ratio

        # Override DataInfo attrs with dataset-specific values
        self.target_size      = self.cfg["target_size"]          # (H, W)
        self.mask_class_map   = self.cfg["mask_class_map"]
        self.seed             = self.cfg["seed"]
        self.label_mode       = self.cfg["label_mode"]
        self.valid_ids        = MODE_IDS[self.label_mode]        # ids this mask may contain

        # NPZ output paths
        npz_out = self.cfg.get("npz_out_dir", self.cfg["data_dir"])
        prefix  = self.cfg["npz_prefix"]
        self.train_npz = npz_out / f"{prefix}_train.npz"
        self.val_npz   = npz_out / f"{prefix}_val.npz"
        self.test_npz  = npz_out / f"{prefix}_test.npz"

        # PNG output dirs (overwrite DataInfo's phantom-specific dirs)
        base = self.cfg.get("aug_out_dir", self.cfg["data_dir"])
        self.images_aug_train_dir = base / "images_aug" / "train"
        self.images_aug_val_dir   = base / "images_aug" / "val"
        self.images_aug_test_dir  = base / "images_aug" / "test"
        self.masks_aug_train_dir  = base / "masks_aug"  / "train"
        self.masks_aug_val_dir    = base / "masks_aug"  / "val"
        self.masks_aug_test_dir   = base / "masks_aug"  / "test"
        for d in [self.images_aug_train_dir, self.images_aug_val_dir, self.images_aug_test_dir,
                  self.masks_aug_train_dir,  self.masks_aug_val_dir,  self.masks_aug_test_dir]:
            d.mkdir(parents=True, exist_ok=True)

        np.random.seed(self.seed)
        random.seed(self.seed)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _remap_mask(self, mask: np.ndarray) -> np.ndarray:
        result = np.zeros_like(mask, dtype=np.uint8)
        for k, v in self.mask_class_map.items():
            result[mask == k] = v
        return result

    def _load_and_crop(self, image_path: Path, mask_path: Path) -> tuple:
        color_mode = self.cfg["color_mode"]
        if color_mode == "gray":
            image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
            if image is None:
                raise ValueError(f"Failed to read image: {image_path}")
        elif color_mode == "rgb_to_gray":
            bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if bgr is None:
                raise ValueError(f"Failed to read image: {image_path}")
            image = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        else:
            raise ValueError(f"Unknown color_mode: {color_mode}")

        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise ValueError(f"Failed to read mask: {mask_path}")
        mask = self._remap_mask(mask)

        crop = self.cfg["crop"]
        if crop is not None:
            top, bottom, left, right = crop
            image = image[top: image.shape[0] - bottom, left: image.shape[1] - right]
            mask  = mask [top: mask.shape[0]  - bottom, left: mask.shape[1]  - right]

        # Bring every dataset to target_size here, invertibly and without touching
        # aspect. Downstream augmentation only ever scales both axes together, so
        # aspect survives that too.
        image, mask, _ = fit(image, mask, self.target_size,
                             self.cfg.get("fit_mode", "croppad"))

        return image, mask

    def _load_rows_from_folder(self) -> list:
        """Scan image/mask folder pair, return rows compatible with _process_split."""
        images_dir = self.cfg["images_dir"]
        masks_dir  = self.cfg["masks_dir"]
        rows = []
        for img_path in sorted(images_dir.glob("*.png")):
            mask_path = masks_dir / img_path.name
            if not mask_path.exists():
                continue
            rows.append({
                "filename":      img_path.name,
                "relative_path": str(img_path),
                "mask_path":     str(mask_path),
                "mask_status":   "true",
            })
        if not rows:
            raise FileNotFoundError(f"No paired PNG files found in {images_dir}")
        return rows

    def _load_rows_from_musv(self) -> dict:
        """
        Mus-V ships its own train/valid split (Videos/train, Videos/valid) and one
        flat Annotations/<seq>/ tree covering both.

        Split by SEQUENCE, never by frame. Consecutive frames of a probe sweep are
        near-duplicates, so a frame-level random split puts the same vessel in both
        train and test and inflates Dice. The official train sweeps become train;
        the official valid sweeps are halved (again by sequence) into val / test.
        """
        root = self.cfg["data_dir"]
        vids, anns = root / "Videos", root / "Annotations"

        seq_split = {}
        for split in ("train", "valid"):
            for d in sorted((vids / split).iterdir()):
                if d.is_dir():
                    seq_split[d.name] = split

        valid_seqs = sorted(s for s, v in seq_split.items() if v == "valid")
        random.Random(self.seed).shuffle(valid_seqs)
        n_val      = len(valid_seqs) // 2
        split_seqs = {
            "train": sorted(s for s, v in seq_split.items() if v == "train"),
            "val":   sorted(valid_seqs[:n_val]),
            "test":  sorted(valid_seqs[n_val:]),
        }

        out = {}
        for split, seqs in split_seqs.items():
            rows = []
            for seq in seqs:
                for img in sorted((vids / seq_split[seq] / seq).glob("*.png")):
                    mask = anns / seq / img.name
                    if mask.exists():
                        rows.append({
                            "filename":      img.name,
                            "relative_path": str(img),
                            "mask_path":     str(mask),
                            "mask_status":   "true",
                            "seq":           seq,
                        })
            if not rows:
                raise FileNotFoundError(f"No Mus-V frames found for split '{split}'")
            out[split] = rows
            print(f"  {split:5s}: {len(seqs):3d} sequences  {len(rows):5d} frames")
        return out

    def _resolve_paths(self, row: dict) -> tuple:
        if sys.platform.startswith('linux'):
            img_path  = Path(row['relative_path'].replace('\\', '/'))
            mask_path = Path(row['mask_path'].replace('\\', '/'))
        else:
            img_path  = Path(row['relative_path'])
            mask_path = Path(row['mask_path'])
        return img_path, mask_path

    def _process_split(self, rows: list, split_name: str) -> tuple:
        """
        Every split is handled identically now: each original is loaded, cropped,
        id-remapped and fit() ONCE, and stored as uint8. Augmentation is not this
        function's job any more — ReadDataset applies it per draw, at training time.
        """
        all_images, all_masks, all_types, all_filenames = [], [], [], []

        for row in tqdm(rows, desc=f"  [{split_name}] processing"):
            img_path, mask_path = self._resolve_paths(row)
            image, mask = self._load_and_crop(img_path, mask_path)
            assert image.shape == tuple(self.target_size), \
                f"fit() returned {image.shape}, expected {self.target_size}"
            all_images.append(image)                      # (H, W) uint8
            all_masks.append(mask)                        # (H, W) uint8, shared id space
            all_types.append(row['mask_status'])
            all_filenames.append(row['filename'].rsplit('.', 1)[0])

        return all_images, all_masks, all_types, all_filenames

    def _save_split(self, images: list, masks: list,
                    image_type: list, filenames: list,
                    npz_path: Path, img_dir: Path, mask_dir: Path,
                    split_name: str, save_png: bool) -> None:

        images_arr = np.array(images,     dtype=np.uint8)     # (N, H, W) — raw, not normalised
        masks_arr  = np.array(masks,      dtype=np.uint8)     # (N, H, W) — shared id space
        type_arr   = np.array(image_type, dtype='<U10')

        # The mask must contain ONLY the ids this label_mode is allowed to produce. If
        # mask_class_map is ever edited into an inconsistent state this fires here, at
        # build time, instead of silently training the wrong marginal for weeks.
        found = set(np.unique(masks_arr).tolist())
        if not found <= self.valid_ids:
            raise ValueError(
                f"[{self.dataset_name}] label_mode='{self.label_mode}' only permits ids "
                f"{sorted(self.valid_ids)} (0=bg 1=vein 2=artery 3=vessel-untyped), "
                f"but the remapped masks contain {sorted(found)}. Check mask_class_map."
            )

        np.savez(
            npz_path,
            images=images_arr,
            masks=masks_arr,
            image_type=type_arr,
            metadata=np.array([{
                'split': split_name,
                'dataset': self.dataset_name,
                'target_size': self.target_size,
                'label_mode': self.label_mode,
                'mask_ids': sorted(found),
                'storage': 'uint8_original',     # NOT pre-augmented, NOT normalised
                'aug_fingerprint': aug_fingerprint(self.dataset_name),
                'num_samples': len(images_arr),
                'creation_time': pd.Timestamp.now().isoformat(),
            }], dtype=object)
        )
        size_mb = npz_path.stat().st_size / (1024 ** 2)
        counts = {int(k): int(v) for k, v in zip(*np.unique(masks_arr, return_counts=True))}
        print(f"    NPZ saved: {size_mb:.1f} MB  ({len(images_arr)} originals, uint8)  "
              f"class pixel counts: {counts}")

        if save_png:
            # Spread the ids over 0-255 so a 3-class mask stays viewable:
            # {0,1,2,3} -> {0,85,170,255}. Multiplying by 255 would overflow.
            scale = 255 // max(1, max(found) if found else 1)
            for i, fname in enumerate(tqdm(filenames, desc=f"    saving PNGs", leave=False)):
                img_hw = images_arr[i]
                msk_hw = (masks_arr[i].astype(np.uint16) * scale).clip(0, 255).astype(np.uint8)
                cv2.imwrite(str(img_dir  / f"{fname}.png"),      img_hw)
                cv2.imwrite(str(mask_dir / f"{fname}_mask.png"), msk_hw)

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(self, save_png: bool = True) -> None:
        """load rows → split ORIGINALS → fit → save uint8 NPZ. No augmentation."""
        source_type = self.cfg["source_type"]
        repeat      = self.cfg["aug_times_train"]

        print(f"Dataset: {self.dataset_name}  |  label_mode: {self.label_mode}  "
              f"|  允许的 mask id: {sorted(self.valid_ids)}  "
              f"(0=bg 1=vein 2=artery 3=vessel-untyped)")

        if source_type == "musv":
            # Split is predefined by the dataset and honoured at sequence level.
            split_rows = self._load_rows_from_musv()
            train_rows = split_rows["train"]
            val_rows   = split_rows["val"]
            test_rows  = split_rows["test"]

        else:
            if source_type == "csv":
                if not self.meta_file.exists():
                    raise FileNotFoundError(f"Metadata file not found: {self.meta_file}")
                df = pd.read_csv(self.meta_file)
                true_df = df[df['mask_status'].astype(str).str.lower() == 'true'].copy()
                true_df = true_df.dropna(subset=['mask_path'])
                true_df = true_df[true_df['mask_path'].apply(lambda x: isinstance(x, str))]
                true_df = true_df.sample(frac=1, random_state=self.seed).reset_index(drop=True)
                all_rows = true_df.to_dict('records')

            elif source_type == "folder":
                all_rows = self._load_rows_from_folder()
                random.shuffle(all_rows)  # seed already set in __init__

            else:
                raise ValueError(f"Unknown source_type: {source_type}")

            n       = len(all_rows)
            n_train = int(n * self.train_ratio)
            n_val   = int(n * self.val_ratio)
            train_rows = all_rows[:n_train]
            val_rows   = all_rows[n_train: n_train + n_val]
            test_rows  = all_rows[n_train + n_val:]
            print(f"  Original images: {n}")

        print(f"Split → train: {len(train_rows)} 张原图  "
              f"val: {len(val_rows)}  test: {len(test_rows)}  "
              f"(NPZ 只存原图；训练时每 epoch 每张抽 ×{repeat} 次，每次现场随机增强)")

        splits = [
            ('train', train_rows, self.train_npz,
             self.images_aug_train_dir, self.masks_aug_train_dir),
            ('val',   val_rows,   self.val_npz,
             self.images_aug_val_dir,   self.masks_aug_val_dir),
            ('test',  test_rows,  self.test_npz,
             self.images_aug_test_dir,  self.masks_aug_test_dir),
        ]

        for split_name, rows, npz_path, img_dir, mask_dir in splits:
            print(f"\n[{split_name}]")
            images, masks, types, filenames = self._process_split(rows, split_name)
            self._save_split(images, masks, types, filenames,
                             npz_path, img_dir, mask_dir, split_name, save_png)

        print(f"\nDone.")
        print(f"  train : {len(train_rows)} 张原图 → 每 epoch {len(train_rows) * repeat} 个样本"
              f"（每个都是新鲜增强，不再是固定的 {repeat} 份拷贝）")
        print(f"  val   : {len(val_rows)} 张（确定性，不增强）")
        print(f"  test  : {len(test_rows)} 张（确定性，不增强）")


# ---------------------------------------------------------------------------
# ReadDataset  —  reads from pre-built NPZ, used by train.py / test.py
# ---------------------------------------------------------------------------

class ReadDataset(Dataset):
    """
    A split of one dataset, read from its NPZ of uint8 ORIGINALS.

    TRAIN: augmentation happens HERE, fresh on every __getitem__. Each original is
    drawn `aug_times_train` times per epoch (that is what __len__ multiplies by), and
    every draw gets an independent random transform. The epoch size and the mix
    between datasets are byte-for-byte what they were when the augmentation was baked
    into the NPZ — musv 2203x8, mendeley 770x5, phantom 44x40, customer_3d 41x40 —
    so nothing about the training balance changed. What changed is that epoch 2 no
    longer shows the model the exact same pixels as epoch 1.

    Why that was the single biggest defect: the vein overfits. Train loss fell
    monotonically 0.575 -> 0.074 across 27 epochs while val vein Dice peaked at epoch
    7 and decayed. Mus-V's 2203 training frames come from only 80 probe sweeps — 80
    distinct veins — and a vein's appearance is exactly what varies most (it is
    compressible, low-contrast, and flattens under probe pressure). Eight frozen
    variants of 80 veins is not a distribution; it is a lookup table.

    VAL / TEST: no augmentation, no randomness. The stored original is already fit()
    to target_size, so there is nothing to do but normalise.
    """

    def __init__(self, dataset_type: str = 'train',
                 batch_size: int = 8,
                 dataset_name: str = "phantom_taobao",
                 augment: bool = None):
        super().__init__()
        self.dataset_type = dataset_type
        self.batch_size   = batch_size
        self.dataset_name = dataset_name

        if dataset_name not in DATASET_CONFIGS:
            raise ValueError(f"Unknown dataset '{dataset_name}'. "
                             f"Available: {list(DATASET_CONFIGS)}")

        cfg     = DATASET_CONFIGS[dataset_name]
        self.cfg = cfg
        npz_out = cfg.get("npz_out_dir", cfg["data_dir"])
        prefix  = cfg["npz_prefix"]
        path_map = {
            'train':      npz_out / f"{prefix}_train.npz",
            'validation': npz_out / f"{prefix}_val.npz",
            'test':       npz_out / f"{prefix}_test.npz",
        }
        if dataset_type not in path_map:
            raise ValueError(f"Invalid dataset_type '{dataset_type}'. "
                             f"Choose from {list(path_map)}")
        self.data_path = path_map[dataset_type]

        # What this dataset's mask ids mean — travels with every sample so that a
        # ConcatDataset of mixed sources still routes each one to the right loss.
        self.label_mode = cfg["label_mode"]
        self.valid_ids  = MODE_IDS[self.label_mode]

        self.augment = (dataset_type == 'train') if augment is None else augment
        self.repeat  = cfg["aug_times_train"] if self.augment else 1

        self.images, self.masks, self.image_type = read_npz_file(self.data_path)

        if self.images.dtype != np.uint8:
            raise ValueError(
                f"{self.data_path} 存的是 {self.images.dtype}，不是 uint8。这是旧格式的 NPZ"
                f"（把增强烤进去了的那种）。请重新生成：\n"
                f"    python dataPrepare.py --dataset {dataset_name} --no-png"
            )
        # A mask id outside this mode's alphabet means the loss would silently
        # supervise the wrong marginal. Fail now, loudly.
        found = set(np.unique(self.masks).tolist())
        if not found <= self.valid_ids:
            raise ValueError(f"[{dataset_name}/{dataset_type}] mask ids {sorted(found)} "
                             f"越界，label_mode='{self.label_mode}' 只允许 "
                             f"{sorted(self.valid_ids)}")

        self.transform = build_train_transform(cfg) if self.augment else None

        n = len(self.images)
        print(f"Loaded '{dataset_type}' [{dataset_name}]: {n} 张原图"
              f"{f' ×{self.repeat} = {n * self.repeat} 样本/epoch (在线增强)' if self.augment else ' (不增强)'}"
              f"  label_mode={self.label_mode}  mask_ids={sorted(found)}")

    def __len__(self):
        return len(self.images) * self.repeat

    def __getitem__(self, idx):
        # idx runs over n*repeat; every repeat of the same original gets an independent
        # random augmentation, and a different one again next epoch.
        base = idx % len(self.images)
        image = self.images[base]          # (H, W) uint8
        mask  = self.masks[base]           # (H, W) uint8

        if self.augment:
            # Isotropic zoom first, and hand-rolled: A.Affine(scale=(0.8,1.2)) samples
            # the x and y scale INDEPENDENTLY, which turns a circle into anything from
            # a 0.70:1 to a 1.47:1 ellipse. Artery-vs-vein is a SHAPE judgement, so
            # that would corrupt the one cue the model has.
            if self.cfg.get("aug_scale"):
                lo, hi = self.cfg["aug_scale"]
                image, mask = isotropic_zoom(image, mask, random.uniform(lo, hi))
            out   = self.transform(image=image, mask=mask)
            image, mask = out["image"], out["mask"]

        img_t, msk_t = normalize_to_tensor(image, mask)
        return (torch.from_numpy(img_t),            # (1, H, W) float32 in [-1, 1]
                torch.from_numpy(msk_t),            # (1, H, W) int64 class ids
                self.label_mode)

    def get_dataloader(self, shuffle: bool = None,
                       num_workers: int = None,
                       drop_last: bool = False) -> DataLoader:
        if shuffle is None:
            shuffle = (self.dataset_type == 'train')
        if num_workers is None:
            num_workers = 0 if sys.platform.startswith('win') else 4
        return DataLoader(
            self,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            drop_last=drop_last,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=(num_workers > 0),
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="phantom_taobao",
                        choices=list(DATASET_CONFIGS),
                        help="Which dataset to process")
    parser.add_argument("--no-png", action="store_true",
                        help="Skip saving PNG files (faster)")
    parser.add_argument("--test-loader", action="store_true",
                        help="Test the DataLoader after processing")
    parser.add_argument("--check-stale", action="store_true",
                        help="检查哪些数据集的 NPZ 与当前增强配置不符。只读，不改任何东西")
    parser.add_argument("--stamp", metavar="DATASET",
                        help="给已存在的 NPZ 打上当前配置的指纹，但不重新生成。"
                             "只有当你已经确认该数据集的增强配置没变过时才用")
    args = parser.parse_args()

    if args.check_stale:
        print("检查 NPZ 是否与当前增强配置一致:", file=sys.stderr)
        stale = check_stale()
        print("\n".join(stale))   # stdout 只输出名字，方便 shell 直接消费
        sys.exit(0)

    if args.stamp:
        if args.stamp not in DATASET_CONFIGS:
            raise SystemExit(f"未知数据集 '{args.stamp}'。可选: {list(DATASET_CONFIGS)}")
        npz, _ = _npz_paths(args.stamp)
        if not npz.exists():
            raise SystemExit(f"NPZ 不存在，没什么可以打指纹的: {npz}")
        write_fingerprint(args.stamp)
        print(f"已给 {args.stamp} 打上指纹 {aug_fingerprint(args.stamp)}。"
              f"（这是在声明「现有 NPZ 确实是用当前配置生成的」——如果这个声明是错的，"
              f"训练会静默使用过期数据。）")
        sys.exit(0)

    pipeline = DataPipeline(args.dataset)
    pipeline.run(save_png=not args.no_png)
    write_fingerprint(args.dataset)

    if args.test_loader:
        print("=" * 50)
        print("Testing DataLoader...")
        ds = ReadDataset(dataset_type='train', batch_size=32, dataset_name=args.dataset)
        loader = ds.get_dataloader()
        for i, (imgs, msks, labels) in enumerate(loader):
            print(f"Batch {i+1}: images={imgs.shape}, masks={msks.shape}")
            if i >= 2:
                break
        print("DataLoader test OK.")
