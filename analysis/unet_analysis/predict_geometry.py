"""
Single-image inference + vessel geometry, for the guidance pipeline.

Replaces predict.ipynb, which had two defects that both biased the numbers the
robot acts on:

  1. TRAIN/INFERENCE PREPROCESSING MISMATCH.  dataPrepare.py crops the phantom frame
     (10,15,50,50) before resizing; predict.ipynb resized the *uncropped* frame while
     its docstring claimed "预处理与训练时完全一致". A vessel therefore reached the
     network 18.9% narrower and 4.5% shorter than in training, with 50 px of unseen
     border on each side. Here the crop, colour mode and target size are read from
     DATASET_CONFIGS — the same dict dataPrepare.py trains from — so the two cannot
     drift apart again.

  2. GEOMETRY MEASURED IN THE RESIZED FRAME.  Moments, area and radius were computed
     on the 576x544 network output, then converted with mm_per_px calibrated for the
     ORIGINAL frame. The resize scales H and W by different factors, so no single
     scalar can convert both axes: lateral offset, depth and radius were each off by
     a different amount. Here the predicted mask is resampled back to native
     resolution (undoing resize AND crop) before any measurement, so the machine's
     own mm_per_px applies exactly.

Works with both model families: n_classes=3 (background/vein/artery) and the legacy
n_classes=1 binary checkpoints.

Usage:
    python predict_geometry.py --ckpt <x>.pth --image 0016.png
    python predict_geometry.py --ckpt <x>.pth --image f.png --dataset customer_3d_phantom
    python predict_geometry.py --ckpt <x>.pth --image f.png --mm-per-px 0.0463
    python predict_geometry.py --ckpt <x>.pth --image f.png --compare-naive
"""
import os
for _k in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_k, '1')

import argparse
import csv
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import binary_fill_holes, label

import sys
from pathlib import Path
# Repo root on sys.path: data pipeline in data/pipeline/, models in models/<arch>/.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from data.pipeline.dataPrepare import DATASET_CONFIGS, fit, unfit_mask
from data.pipeline.DataInfo import DataInfo
from models.unet.model import UNet

BG, VEIN, ARTERY = 0, 1, 2
CLASS_NAME = {VEIN: "vein", ARTERY: "artery"}
CLASS_BGR  = {VEIN: (60, 40, 200), ARTERY: (210, 95, 45)}
UNTYPED    = "vessel (untyped)"     # matplotlib's default font has no CJK glyphs,
                                    # so anything drawn onto the figure stays ASCII


# ---------------------------------------------------------------------------
# Preprocessing — mirrors dataPrepare.DataPipeline exactly, from the same config
# ---------------------------------------------------------------------------

def preprocess(image_path: str, cfg: dict):
    """
    Returns (original_gray, network_input_float, geom) where geom carries everything
    needed to put a network-resolution mask back onto the original canvas.
    """
    colour = cfg["color_mode"]
    if colour == "gray":
        img0 = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    elif colour == "rgb_to_gray":
        bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        img0 = None if bgr is None else cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    else:
        raise ValueError(f"Unknown color_mode: {colour}")
    if img0 is None:
        raise FileNotFoundError(f"Cannot read: {image_path}")

    H0, W0 = img0.shape
    top, bottom, left, right = cfg["crop"] or (0, 0, 0, 0)
    img_c = img0[top:H0 - bottom, left:W0 - right]           # same crop as training
    Hc, Wc = img_c.shape

    # fit() is imported from dataPrepare — literally the same function the training
    # arrays were baked with, so training and inference cannot drift apart.
    mode = cfg.get("fit_mode", "croppad")
    img_r, _, fp = fit(img_c, None, cfg["target_size"], mode)
    x = (img_r.astype(np.float32) / 255.0 - 0.5) / 0.5       # same normalisation

    geom = dict(H0=H0, W0=W0, Hc=Hc, Wc=Wc, top=top, bottom=bottom,
                left=left, right=right, fit=fp, mode=mode)
    return img0, x, geom


def mask_to_crop(mask_net: np.ndarray, geom: dict) -> np.ndarray:
    """
    Take the network's mask back to the CROP frame — the coordinate system the
    calibration is defined in (mm_per_px is mm per crop-frame pixel).

    This is the step predict.ipynb never did: it measured moments on the 576x544
    network output and converted them with mm/px calibrated for the crop frame. Under
    fit_mode='croppad' this inverse is a pure offset, so one network pixel IS one crop
    pixel and the conversion is exact with zero resampling.
    """
    return unfit_mask(mask_net, geom["fit"])


def mask_to_original(mask_net: np.ndarray, geom: dict) -> np.ndarray:
    """Crop frame -> full original frame (adds the crop offset back)."""
    back = mask_to_crop(mask_net, geom)
    full = np.zeros((geom["H0"], geom["W0"]), dtype=mask_net.dtype)
    full[geom["top"]:geom["H0"] - geom["bottom"],
         geom["left"]:geom["W0"] - geom["right"]] = back
    return full


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def load_model(ckpt_path, device):
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    saved = ck.get("config", {}) or {}
    n_classes = saved.get("num_classes", 1)
    model = UNet(1, n_classes, saved.get("bilinear", False),
                 base_ch=int(saved.get("base_ch", 64)))
    model.load_state_dict(ck["model"])
    model = model.to(device, memory_format=torch.channels_last).eval()
    return model, n_classes


@torch.no_grad()
def infer(model, n_classes, x, device, threshold=0.5) -> np.ndarray:
    t = torch.from_numpy(x)[None, None].to(device, dtype=torch.float32,
                                           memory_format=torch.channels_last)
    with torch.autocast(device.type, enabled=device.type == "cuda"):
        logits = model(t)
    if n_classes == 1:
        # Legacy binary model: foreground is a vessel of unknown type. Map it to
        # ARTERY purely so a single downstream code path can handle both models;
        # `typed` below is what decides whether that id means anything.
        fg = (torch.sigmoid(logits.float()) > threshold)[0, 0]
        return (fg.long() * ARTERY).cpu().numpy().astype(np.uint8)
    return F.softmax(logits.float(), 1).argmax(1)[0].cpu().numpy().astype(np.uint8)


def clean(binary: np.ndarray, min_area: int = 300) -> np.ndarray:
    """
    Open → close → fill holes → keep the largest connected component.

    The hole fill matters most HERE, of all three places this morphology lives, because
    this file is the one that reports radius_mm = sqrt(area_px / pi) * mm_per_px. A hole
    left inside a predicted lumen subtracts from area_px and shrinks the reported radius:
    measured on Mus-V, 7.8% of predicted veins contain one, under-reporting the radius by
    up to 9.1%. A lumen is anechoic — an enclosed hole is a segmentation artefact, not
    anatomy.

    (This function stayed on the old open→close→largest recipe after test.py and
    models/unet/infer.py gained the fill, which is exactly the drift three separate copies
    of the same morphology invite.)
    """
    if not binary.any():
        return binary
    k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    k5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    b = cv2.morphologyEx(binary.astype(np.uint8), cv2.MORPH_OPEN, k3)
    b = cv2.morphologyEx(b, cv2.MORPH_CLOSE, k5)
    b = binary_fill_holes(b).astype(np.uint8)
    lab, n = label(b.astype(bool))
    if n == 0:
        return np.zeros_like(binary, dtype=bool)
    sizes = [(lab == i).sum() for i in range(1, n + 1)]
    best = int(np.argmax(sizes)) + 1
    if sizes[best - 1] < min_area:
        return np.zeros_like(binary, dtype=bool)
    return lab == best


# ---------------------------------------------------------------------------
# Geometry — measured on the NATIVE-resolution mask
# ---------------------------------------------------------------------------

def measure(binary: np.ndarray, mm_per_px: float,
            axis_x: int = None, skin_row: int = 0) -> dict:
    if not binary.any():
        return {}
    H, W = binary.shape
    axis_x = W // 2 if axis_x is None else axis_x           # probe axis = image centre

    M = cv2.moments(binary.astype(np.uint8), binaryImage=True)
    cx, cy = M["m10"] / M["m00"], M["m01"] / M["m00"]

    ys, xs = np.where(binary)
    top_y, bottom_y = int(ys.min()), int(ys.max())
    area_px  = float(binary.sum())
    radius_px = float(np.sqrt(area_px / np.pi))             # equivalent-area radius

    return {
        "centroid_px":      (cx, cy),
        "area_px":          area_px,
        "lateral_mm":       (cx - axis_x) * mm_per_px,      # δ, signed
        "depth_centre_mm":  (cy - skin_row) * mm_per_px,    # d to vessel centre
        "depth_top_mm":     (top_y - skin_row) * mm_per_px, # d to near wall
        "radius_mm":        radius_px * mm_per_px,          # r
        "diameter_bbox_mm": (bottom_y - top_y + 1) * mm_per_px,
    }


def get_mm_per_px(image_path: str) -> float:
    """μm/px from the acquisition metadata CSV — calibrated for the ORIGINAL frame."""
    info = DataInfo()
    stem = Path(image_path).stem
    with open(info.meta_file) as f:
        r = csv.DictReader(f)
        r.fieldnames = [h.strip() for h in r.fieldnames]
        for row in r:
            row = {k.strip(): (v or "").strip() for k, v in row.items()}
            if row.get("filename") == Path(image_path).name or row.get("id") == stem:
                return float(row["micropixel"]) / 1000.0
    raise ValueError(f"'{image_path}' 不在 {info.meta_file} 里。用 --mm-per-px 手动指定。")


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--image", required=True)
    ap.add_argument("--dataset", default="phantom_taobao", choices=list(DATASET_CONFIGS),
                    help="决定预处理（crop / color_mode / target_size）。"
                         "必须和该图像所属的训练数据集一致")
    ap.add_argument("--mm-per-px", type=float, default=None,
                    help="覆盖标定值。默认从 meta CSV 的 micropixel 读")
    ap.add_argument("--skin-row", type=int, default=0)
    ap.add_argument("--threshold", type=float, default=0.5, help="仅二分类模型使用")
    ap.add_argument("--compare-naive", action="store_true",
                    help="同时按 predict.ipynb 的旧算法算一遍，打印误差")
    ap.add_argument("--out", default="results/unet/figures/geometry.png")
    args = ap.parse_args()

    cfg = DATASET_CONFIGS[args.dataset]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model, n_classes = load_model(args.ckpt, device)
    typed = n_classes > 1

    img0, x, geom = preprocess(args.image, cfg)
    pred_net = infer(model, n_classes, x, device, args.threshold)
    pred_full = mask_to_original(pred_net, geom)

    mm = args.mm_per_px if args.mm_per_px is not None else get_mm_per_px(args.image)

    TH, TW = cfg["target_size"]
    print(f"checkpoint : {Path(args.ckpt).name}  (n_classes={n_classes})")
    print(f"image      : {args.image}   原图 {geom['H0']}x{geom['W0']}")
    print(f"预处理     : crop{cfg['crop']} -> {geom['Hc']}x{geom['Wc']} "
          f"-> fit[{geom['mode']}] {TH}x{TW}   [与 dataPrepare.py 同源]")
    print(f"mm_per_px  : {mm:.6f}  ({mm*1000:.2f} μm/px)   "
          f"—— 只作用在 unfit 之后的原图坐标系，模型从头到尾不知道毫米的存在")
    print()

    classes = (VEIN, ARTERY) if typed else (ARTERY,)
    results = {}
    for c in classes:
        m = clean(pred_full == c)
        if not m.any():
            continue
        name = CLASS_NAME[c] if typed else UNTYPED
        g = measure(m, mm, skin_row=args.skin_row)
        results[c] = (name, m, g)

        # A vessel touching the crop boundary extends into a region the network was
        # never shown, so its extent — and therefore its radius — is truncated.
        t, b, l, r = cfg["crop"] or (0, 0, 0, 0)
        ys, xs = np.where(m)
        if (t and ys.min() <= t) or (b and ys.max() >= geom["H0"] - b - 1) \
           or (l and xs.min() <= l) or (r and xs.max() >= geom["W0"] - r - 1):
            print(f"  ⚠ {name}: 血管触到了 crop 边界，半径会被截断（模型看不到裁掉的区域）")

        print(f"[{name}]")
        print(f"  质心      : ({g['centroid_px'][0]:.1f}, {g['centroid_px'][1]:.1f}) px  (原图坐标)")
        print(f"  横向偏移 δ : {g['lateral_mm']:+.2f} mm")
        print(f"  深度 d    : {g['depth_centre_mm']:.2f} mm (到中心) | "
              f"{g['depth_top_mm']:.2f} mm (到近壁)")
        print(f"  半径 r    : {g['radius_mm']:.2f} mm  (等效面积半径)")
        print(f"  直径(bbox): {g['diameter_bbox_mm']:.2f} mm")
        print()

    if not results:
        print("没有检测到血管。")
        return

    if args.compare_naive:
        # predict.ipynb: measure on the 576x544 mask, convert with the ORIGINAL mm/px.
        print("=" * 64)
        print("对比 predict.ipynb 的旧算法（在 576x544 空间量，却用原图 mm_per_px）")
        print("=" * 64)
        for c, (name, _, g) in results.items():
            m_net = clean(pred_net == c)
            if not m_net.any():
                continue
            gn = measure(m_net, mm, skin_row=args.skin_row)
            print(f"[{name}]  {'量':>10s} {'旧(错)':>10s} {'新(对)':>10s} {'误差':>9s}")
            for k, lbl in (("lateral_mm", "δ 横向"), ("depth_centre_mm", "d 深度"),
                           ("radius_mm", "r 半径")):
                o, n = gn[k], g[k]
                err = (o - n) / n * 100 if abs(n) > 1e-9 else float("nan")
                print(f"{'':14s} {lbl:>10s} {o:10.2f} {n:10.2f} {err:+8.1f}%")
            print()

    # ---- figure ----
    vis = cv2.cvtColor(img0, cv2.COLOR_GRAY2RGB)
    for c, (_, m, _) in results.items():
        col = np.array(CLASS_BGR[c][::-1], np.float32)      # BGR -> RGB
        vis[m] = (0.45 * vis[m] + 0.55 * col).astype(np.uint8)

    fig, ax = plt.subplots(figsize=(7, 8))
    ax.imshow(vis)
    axis_x = geom["W0"] // 2
    ax.axvline(axis_x, color="w", ls=":", lw=1, alpha=0.6)
    for c, (name, _, g) in results.items():
        cx, cy = g["centroid_px"]
        ax.plot(cx, cy, "o", ms=8, color="yellow", mec="k")
        ax.annotate("", xy=(axis_x, cy), xytext=(cx, cy),
                    arrowprops=dict(arrowstyle="<->", color="yellow", lw=1.5))
        ax.plot([cx, cx], [args.skin_row, cy], color="yellow", ls="--", lw=1.5)
        ax.text(cx + 8, cy - 8,
                f"{name}\nδ={g['lateral_mm']:+.1f} d={g['depth_centre_mm']:.1f} "
                f"r={g['radius_mm']:.1f} mm",
                color="yellow", fontsize=9,
                bbox=dict(boxstyle="round,pad=0.3", fc="black", alpha=0.6, ec="none"))
    if cfg["crop"]:
        t, b, l, r = cfg["crop"]
        ax.add_patch(plt.Rectangle((l, t), geom["W0"] - l - r, geom["H0"] - t - b,
                                   fill=False, ec="cyan", ls="--", lw=1))
        ax.text(l + 4, t + 14, "network field of view (crop)", color="cyan", fontsize=8)
    ax.set_title("Vessel geometry (native resolution)")
    ax.axis("off")

    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"图已保存 -> {out}")


if __name__ == "__main__":
    main()
