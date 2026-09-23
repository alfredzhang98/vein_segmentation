"""Five-domain qualitative figure using the shared final postprocessing.

Run from vein_segmentation/ with downloaded data and prepared test caches:
  python results/unet/plot_segmentation_samples.py --ckpt models/unet/checkpoints/unet_v11.pth

Defaults to two visible test frames per dataset, chosen with a fixed seed without
looking at model predictions. This is an illustration, not a dataset-level metric.
Generated previews stay local except the explicitly retained final showcase PNG/JSON.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
# Claim an idle GPU BEFORE anything can touch torch.cuda — the CUDA runtime reads
# CUDA_VISIBLE_DEVICES once, on first contact, and ignores it silently thereafter.
# Hence the import sitting up here, above torch, instead of tidily with the others.
from gpu_utils import pick_idle_gpu
pick_idle_gpu()

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
import numpy as np
import torch
import torch.nn.functional as F

from data.pipeline.dataPrepare import (ReadDataset, CLASS_BG, CLASS_VEIN,
                                       CLASS_ARTERY, CLASS_VESSEL, DATASET_CONFIGS, check_stale)
from models.unet.postprocess import clean_labels, complete_region
from models.unet.model import UNet
from figure_style import FigureConfig

# ── palette: the id space IS the meaning (dataPrepare) ───────────────────────
# Anatomical convention: ARTERY = RED, VEIN = BLUE. (An earlier version had these
# swapped, inherited from the Mus-V overlays' own colouring — but every reader of a
# vascular paper arrives expecting red arteries, and a figure that inverts that costs
# more confusion than any internal consistency is worth.)
CLASS_RGB = {
    CLASS_VEIN:   np.array([0.13, 0.40, 0.72]),   # blue  — vein
    CLASS_ARTERY: np.array([0.86, 0.20, 0.18]),   # red   — artery
    CLASS_VESSEL: np.array([0.15, 0.65, 0.38]),   # green — vessel, type not asked for
}
FILL_A = 0.45
GT_EDGE = "#FFFFFF"        # ground-truth outline in the comparison column
# A bare white contour is invisible against a bright speckle field, and invisible in
# the legend (whose background is also white). The dark halo makes it read on both.
GT_HALO = [pe.withStroke(linewidth=3.4, foreground="#1A1A1A")]

# One neutral grey for all three domain bands. Was a blue-slate (#455A64), but that
# tint read as a colour with meaning next to the red/blue vessels; a plain mid-grey is
# unambiguously just a label strip. Each domain used to get its own shade, dropped for
# the same reason — the label text and inter-block gaps already group the rows.
DOMAIN_BAND_COLOR = "#595959"


def load_model(ckpt_path, device):
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ck.get("config", {}) or {}
    n_classes = int(cfg.get("num_classes", 3))
    # base_ch travels in the checkpoint; anything older than that knob is 64-wide.
    model = UNet(n_channels=1, n_classes=n_classes,
                 bilinear=bool(cfg.get("bilinear", False)),
                 base_ch=int(cfg.get("base_ch", 64)))
    model.load_state_dict(ck["model"])
    model.to(device, memory_format=torch.channels_last).eval()
    n_par = sum(p.numel() for p in model.parameters())
    print(f"Loaded {Path(ckpt_path).name}: n_classes={n_classes} "
          f"base_ch={cfg.get('base_ch', 64)} ({n_par/1e6:.1f}M params) "
          f"epoch={ck.get('epoch','?')} best={ck.get('best_score', float('nan')):.4f}")
    return model, n_classes


@torch.no_grad()
def predict(model, img_t, device, mode, min_area):
    """
    Returns a mask in the SHARED id space, post-processed as deployment does — which
    means keep_largest: at most one region per class for this task
    (see clean_binary). A figure showing a class split across two blobs would be
    advertising a mask whose centroid the geometry stage cannot use.
    """
    with torch.autocast(device.type, enabled=device.type == "cuda"):
        logits = model(img_t.unsqueeze(0).to(device, memory_format=torch.channels_last))
    pred = F.softmax(logits.float(), 1).argmax(1)[0].cpu().numpy().astype(np.uint8)

    if mode == "vessel":
        # Merge first, then clean as ONE region — a phantom has exactly one tube, and
        # its vein/artery split carries no information. This mirrors the deployed
        # binary path: (p1 + p2) > thr -> clean.
        fg = complete_region((pred == CLASS_VEIN) | (pred == CLASS_ARTERY),
                             min_area=min_area)
        return np.where(fg, CLASS_VESSEL, CLASS_BG).astype(np.uint8)

    return clean_labels(pred, min_area=min_area)


def dice(pred_bin, gt_bin):
    s = pred_bin.sum() + gt_bin.sum()
    return 1.0 if s == 0 else 2.0 * float((pred_bin & gt_bin).sum()) / float(s)


def panel_scores(pred, gt, mode):
    """Per-panel Dice, reporting only what this dataset's labels can support."""
    if mode == "full3":
        return [("vein",   dice(pred == CLASS_VEIN,   gt == CLASS_VEIN)),
                ("artery", dice(pred == CLASS_ARTERY, gt == CLASS_ARTERY))]
    if mode == "vessel":
        return [("vessel", dice(pred == CLASS_VESSEL, gt == CLASS_VESSEL))]
    # artery mode: the vein is real but unlabelled — scoring it would punish a
    # correct prediction, which is exactly the bug that killed the vein class.
    return [("artery", dice(pred == CLASS_ARTERY, gt == CLASS_ARTERY))]


def rgba_of(mask, alpha=FILL_A):
    """Shared-id mask -> RGBA, transparent on background."""
    H, W = mask.shape
    out = np.zeros((H, W, 4), float)
    for cid, rgb in CLASS_RGB.items():
        sel = mask == cid
        if sel.any():
            out[sel] = [*rgb, alpha]
    return out


def draw_gt_contours(ax, gt):
    for cid in (CLASS_VEIN, CLASS_ARTERY, CLASS_VESSEL):
        m = (gt == cid).astype(np.uint8)
        if not m.any():
            continue
        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in cnts:
            if len(c) < 3:
                continue
            c = c.squeeze(1)
            ax.plot(np.append(c[:, 0], c[0, 0]), np.append(c[:, 1], c[0, 1]),
                    color=GT_EDGE, lw=1.6, ls="-", alpha=0.95,
                    path_effects=GT_HALO)


def draw_pred_contours(ax, pred):
    """
    Outline each predicted region in its own (darkened) class colour with a dark halo.
    The translucent fill alone is easy to miss against a dark neck — most visibly the
    unsupervised jugular vein on CCA, which is the whole point of that row. A crisp edge
    makes every prediction legible without hiding the underlying anatomy the fill lets
    through. Applied to all panels, not just CCA, so the Prediction column reads the
    same way everywhere.
    """
    halo = [pe.withStroke(linewidth=3.0, foreground="#1A1A1A")]
    for cid in (CLASS_VEIN, CLASS_ARTERY, CLASS_VESSEL):
        m = (pred == cid).astype(np.uint8)
        if not m.any():
            continue
        edge = tuple(np.clip(CLASS_RGB[cid] * 1.15, 0, 1))     # a touch brighter than fill
        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in cnts:
            if len(c) < 3:
                continue
            c = c.squeeze(1)
            ax.plot(np.append(c[:, 0], c[0, 0]), np.append(c[:, 1], c[0, 1]),
                    color=edge, lw=1.8, alpha=0.95, path_effects=halo)


def normalize_display(img):
    mn, mx = img.min(), img.max()
    return (img - mn) / (mx - mn) if mx > mn else img


def pick_samples(ds, n, seed, mode):
    """
    Prefer frames with labelled foreground (both A/V for full3). Sample selection
    uses only ground truth and a fixed seed, never prediction scores. Empty-frame
    performance belongs in the complete evaluation, not this illustrative panel.
    """
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(ds))
    picked, fallback = [], []
    for i in order:
        _, m, _ = ds[int(i)]
        m = m.squeeze(0).numpy()
        if mode == "full3":
            ok = (m == CLASS_VEIN).any() and (m == CLASS_ARTERY).any()
        else:
            ok = (m > 0).any()
        (picked if ok else fallback).append(int(i))
        if len(picked) == n:
            break
    idxs = (picked + fallback)[:n]
    out = []
    for i in idxs:
        img, msk, lm = ds[i]
        out.append((img, msk.squeeze(0).numpy().astype(np.uint8), i))
    return out



def showcase_distance(left, right):
    """Compare labelled geometry and low-resolution appearance in image coordinates."""
    geometry = np.mean([1.0 - dice(a, b) for a, b in zip(left["masks"], right["masks"])])
    appearance = float(np.sqrt(np.mean((left["thumbnail"] - right["thumbnail"]) ** 2)))
    # Area separates expanded/compressed vessels even if positional displacement
    # makes two otherwise similar frames have little mask overlap.
    area = np.mean([abs(int(a.sum()) - int(b.sum())) / max(int(a.sum()), int(b.sum()), 1)
                    for a, b in zip(left["masks"], right["masks"])])
    return 0.45 * float(geometry) + 0.45 * float(area) + 0.10 * appearance


def pick_showcase_samples(ds, n, model, device, min_area, min_dice):
    """Select diverse success examples above an explicit weakest-class Dice floor.

    First choose the best weakest-class Dice; subsequent choices maximize minimum
    distance from selected frames. This is curated illustration, not test evaluation.
    """
    ranked = []
    required = (CLASS_VEIN, CLASS_ARTERY) if ds.label_mode == "full3" else (
        CLASS_ARTERY if ds.label_mode == "artery" else CLASS_VESSEL,)
    for i in range(len(ds)):
        img, mask, mode = ds[i]
        gt = mask.squeeze(0).numpy().astype(np.uint8)
        if not all((gt == c).any() for c in required):
            continue
        pred = predict(model, img, device, mode, min_area)
        scores = dict(panel_scores(pred, gt, mode))
        values = list(scores.values())
        if min(values) < min_dice:
            continue
        ranked.append(dict(worst=min(values), mean=float(np.mean(values)), index=i,
                           scores=scores, masks=[gt == c for c in required],
                           thumbnail=cv2.resize(normalize_display(img.squeeze(0).numpy()),
                                                (32, 32), interpolation=cv2.INTER_AREA)))
    ranked.sort(key=lambda r: (-r["worst"], -r["mean"], r["index"]))
    if len(ranked) < n:
        raise ValueError(f"{ds.dataset_name}: only {len(ranked)} visible frames meet "
                         f"all-class Dice >= {min_dice}; need {n}")
    print(f"  Showcase pool: {ds.dataset_name} {len(ranked)} frames, Dice >= {min_dice}")
    selected = [ranked.pop(0)]
    while len(selected) < n:
        best = max(range(len(ranked)), key=lambda k: (
            min(showcase_distance(ranked[k], other) for other in selected),
            ranked[k]["worst"], ranked[k]["mean"], -ranked[k]["index"]))
        selected.append(ranked.pop(best))
    samples = []
    for candidate in selected:
        i, scores = candidate["index"], candidate["scores"]
        print(f"  Selected showcase: {ds.dataset_name} index={i} Dice={scores}")
        img, mask, _ = ds[i]
        samples.append((img, mask.squeeze(0).numpy().astype(np.uint8), i))
    return samples


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--n", type=int, default=2, help="每个数据集取几张")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--datasets", nargs="+", choices=list(DATASET_CONFIGS),
                    default=["musv", "mendeley", "pmc9883282", "phantom_taobao", "customer_3d_phantom"])
    ap.add_argument("--showcase-datasets", nargs="+", default=[], choices=list(DATASET_CONFIGS),
                    help="仅这些域按单帧Dice择优展示；其余域保持seed选帧，JSON记录择优规则")
    ap.add_argument("--showcase-min-dice", type=float, default=0.90,
                    help="展示候选每个已标注类别的最低Dice；在达标候选中选择差异大的样本")
    ap.add_argument("--min-area", type=int, default=4)
    ap.add_argument("--out", help="输出前缀；默认按checkpoint名称命名，不覆盖已有文件")
    ap.add_argument("--formats", nargs="+", choices=["png", "pdf", "svg"], default=["png"])
    ap.add_argument("--dpi", type=int, default=200)
    ap.add_argument("--preview", action="store_true", help="仅PNG、150dpi快速预览")
    args = ap.parse_args()
    args.out = args.out or f"results/unet/figures/segmentation_samples_{Path(args.ckpt).stem}"
    formats = ["png"] if args.preview else args.formats
    existing = [Path(str(args.out) + "." + ext) for ext in [*formats, "json"]
                if Path(str(args.out) + "." + ext).exists()]
    if existing:
        ap.error("Output already exists; use a different --out prefix: " + str(existing[0]))

    if not 0 <= args.showcase_min_dice <= 1:
        ap.error("--showcase-min-dice must be between 0 and 1")
    if args.n < 1 or args.dpi < 1 or args.min_area < 1:
        ap.error("--n, --dpi and --min-area must be positive")
    if len(set(args.datasets)) != len(args.datasets):
        ap.error("Dataset names must be unique")
    if not set(args.showcase_datasets) <= set(args.datasets):
        ap.error("--showcase-datasets must be included in --datasets")
    if check_stale(args.datasets):
        ap.error("Data/cache missing or stale; follow data/README_CN.md before plotting")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, n_classes = load_model(args.ckpt, device)
    if n_classes < 3:
        raise SystemExit("这个脚本只支持三分类模型（背景/静脉/动脉）。"
                         f"这个 checkpoint 是 {n_classes} 类的旧二分类模型，"
                         "对应的绘图脚本已随二分类模型一并删除。")

    labels = {"musv": "Mus-V", "mendeley": "CCA", "pmc9883282": "PMC",
              "phantom_taobao": "Taobao phantom", "customer_3d_phantom": "Phantom"}
    offsets = {"musv": 0, "mendeley": 37, "pmc9883282": 111,
               "phantom_taobao": 148, "customer_3d_phantom": 74}
    domains = [(name, labels.get(name, name), "") for name in args.datasets]
    rows = []
    for dom, _, _ in domains:
        ds = ReadDataset("test", 1, dom, augment=False)
        if args.n > len(ds):
            ap.error(f"{dom} only has {len(ds)} test frames; reduce --n")
        seed = args.seed + offsets.get(dom, 0)
        print(f"  {dom:20s} seed={seed}")
        samples = (pick_showcase_samples(ds, args.n, model, device, args.min_area, args.showcase_min_dice)
                   if dom in args.showcase_datasets else pick_samples(ds, args.n, seed, ds.label_mode))
        for sample in samples:
            rows.append((dom, ds.label_mode, sample))

    NC, NR = 4, len(rows)

    # The 12.0" width and the cell/margin/legend arithmetic below are deliberate, not
    # free parameters. Every figure in the paper is authored at this width and scaled
    # to the paper column by the same factor (column_width / 12.0), which is what makes
    # the text in all of them come out at the same physical size on the page. Changing
    # the width here rescales this figure's text relative to the others.
    FigureConfig(grid=False, title_pad=6.0).apply()

    cell = (12.0 - 0.6) / NC                      # 2.85" per cell at NC=4
    fig = plt.figure(figsize=(12.0, NR * cell + 1.0))
    gs_root = fig.add_gridspec(1, 2, width_ratios=[0.045, 1], wspace=0.012,
                               left=0.01, right=0.99, top=0.93, bottom=0.10)

    gs_lbl = gs_root[0].subgridspec(len(domains), 1, hspace=0.02)
    for i, (dom, label, _) in enumerate(domains):
        ax = fig.add_subplot(gs_lbl[i])
        ax.set_facecolor(DOMAIN_BAND_COLOR)
        ax.text(0.5, 0.5, label, ha="center", va="center", rotation=90,
                fontweight="bold", color="white", transform=ax.transAxes)
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)

    gs_main = gs_root[1].subgridspec(NR, NC, hspace=0.04, wspace=0.03)
    col_titles = ["Input", "Ground Truth", "Prediction", "GT vs. Pred"]

    manifest = []
    for r, (dom, mode, (img_t, gt, index)) in enumerate(rows):
        pred = predict(model, img_t, device, mode, args.min_area)
        manifest.append(dict(dataset=dom, cache_index=index, label_mode=mode,
                             selection="diverse_high_dice_showcase" if dom in args.showcase_datasets else "fixed_seed",
                             dice=dict(panel_scores(pred, gt, mode))))
        img = normalize_display(img_t.squeeze(0).numpy())
        border = DOMAIN_BAND_COLOR

        for c in range(NC):
            ax = fig.add_subplot(gs_main[r, c])
            ax.imshow(img, cmap="gray", vmin=0, vmax=1, interpolation="lanczos")

            if c == 1:
                ax.imshow(rgba_of(gt), interpolation="nearest")
            elif c == 2:
                ax.imshow(rgba_of(pred), interpolation="nearest")
                draw_pred_contours(ax, pred)
                # TOP-left, and stacked.
                #   top:     row 0 of a B-mode frame is the probe face — skin line and
                #            superficial tissue. The vessels sit deeper, in the middle
                #            and lower thirds, so this is the one corner that reliably
                #            has nothing to hide. At the bottom the box landed straight
                #            on top of the predicted vein in the Mus-V rows.
                #   stacked: "Vein 0.84  Artery 0.90" on one line is wider than the
                #            2.85" cell, and a text bbox is NOT clipped to its axes —
                #            it spilled past the panel border into the gutter.
                sc = panel_scores(pred, gt, mode)
                ax.text(0.035, 0.965,
                        "\n".join(f"{k.capitalize():<6s} {v:.2f}" for k, v in sc),
                        transform=ax.transAxes, ha="left", va="top",
                        fontsize=15, fontweight="bold", color="white",
                        linespacing=1.25, family="monospace",
                        bbox=dict(boxstyle="round,pad=0.25", fc="black",
                                  alpha=0.6, ec="none"))
            elif c == 3:
                ax.imshow(rgba_of(pred, FILL_A * 0.9), interpolation="nearest")
                draw_gt_contours(ax, gt)

            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_edgecolor(border); sp.set_linewidth(2.0)
            if r == 0:
                ax.set_title(col_titles[c], fontweight="bold")

    handles = [
        mpatches.Patch(color=CLASS_RGB[CLASS_ARTERY], alpha=0.85, label="Artery"),
        mpatches.Patch(color=CLASS_RGB[CLASS_VEIN],   alpha=0.85, label="Vein"),
        mpatches.Patch(color=CLASS_RGB[CLASS_VESSEL], alpha=0.85,
                       label="Vessel (type not labelled)"),
        plt.Line2D([0], [0], color=GT_EDGE, lw=2.2, path_effects=GT_HALO,
                   label="Ground-truth outline"),
    ]
    # fontsize 18 + bbox 0.07: same as the old figure's legend, so the two render at
    # the same size on the page. ncol=2 (not 4) because this legend carries four
    # entries against the old one's two — four across 12" at 18 pt overflows.
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.07),
               ncol=2, fontsize=18, frameon=True, edgecolor="#cccccc",
               fancybox=True, framealpha=0.6)

    # Match the historical figure layout; selection provenance stays in JSON
    # and the accompanying README caption, not an extra title/footer.
    fmts = ("png",) if args.preview else tuple(args.formats)
    saved = FigureConfig.save(fig, args.out, formats=fmts, dpi=150 if args.preview else args.dpi)
    manifest_path = Path(str(args.out) + ".json")
    manifest_path.write_text(json.dumps(dict(checkpoint=str(args.ckpt), split="test",
                             seed=args.seed, samples_per_dataset=args.n, postprocessing=dict(policy="joint", min_area=args.min_area,
                             closing_radius=2, fill_holes=True, dominance=.6),
                             selection=("explicit diverse high-Dice showcase for named datasets; remaining rows fixed seed"
                                        if args.showcase_datasets else
                                        "fixed seed, prefer labelled foreground; not selected by prediction quality"),
                             showcase_datasets=args.showcase_datasets,
                             showcase_policy=dict(min_class_dice=args.showcase_min_dice,
                                 first="maximum weakest-class Dice, then mean Dice, then index",
                                 subsequent="maximize minimum distance to selected examples",
                                 distance="0.45 * mean class-mask Dice distance + 0.45 * mean relative class-area difference + 0.10 * normalized 32x32 image RMSE",
                                 note="Curated success examples, not representative test-set performance"),
                             samples=manifest), indent=2))
    print("Saved → " + "  +  ".join(str(p) for p in saved))
    print("Sample indices and scores →", manifest_path)



if __name__ == "__main__":
    main()
