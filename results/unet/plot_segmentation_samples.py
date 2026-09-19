"""
Qualitative segmentation figure for the 3-class partial-label model (RA-L).

This replaces an earlier binary-only script, deleted along with the binary model.
That one called torch.sigmoid() on a 1-channel logit and thresholded at 0.5, so it
could not run the 3-class model at all, and its palette (green = GT, coral =
prediction) encoded "truth vs prediction" — the wrong axis. The interesting axis is
WHICH VESSEL, and what each dataset's labels are even able to claim.

WHAT THIS FIGURE HAS TO SAY
---------------------------
One model, three label semantics. The columns are the same everywhere, but the
meaning of a colour depends on what the dataset knows:

  Mus-V     (full3)   Both vessels labelled  -> red artery + blue vein.
  Phantom   (vessel)  "There is a tube here", type unknown -> ONE green region.
                      The vein/artery split is not merely unlabelled here, it is
                      MEANINGLESS: the loss constrains only p1+p2, so the model is
                      free to scatter red/blue across the tube — and it does (median
                      4 vein pieces + 2 artery pieces on phantom_taobao, while the
                      merged foreground is a single blob). Drawing that confetti
                      would advertise an opinion the model was never asked for.
  Mendeley  (artery)  Only the common carotid is labelled. The jugular vein is in
                      frame and UNLABELLED, so any red in that panel is an
                      unsupervised bonus, not an error. This is the generalisation
                      story, and the figure should show it rather than hide it.

Green never shares a panel with red/blue (green is phantom-only, red/blue are
human-only), so the only two colours ever seen together are red and blue — which
stays legible under deuteranopia. Domain bands are neutral slate on purpose: the
old figure used blue/coral for DOMAIN, and reusing them here would collide with
red/blue meaning ARTERY/VEIN.

Usage:
  python results/unet/plot_segmentation_samples.py --ckpt models/unet/checkpoints/unet_v10.pth
"""
import argparse
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
                                       CLASS_ARTERY, CLASS_VESSEL)
from models.unet.test import clean_binary                      # deployment post-processing, verbatim
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
    model.to(device).eval()
    n_par = sum(p.numel() for p in model.parameters())
    print(f"Loaded {Path(ckpt_path).name}: n_classes={n_classes} "
          f"base_ch={cfg.get('base_ch', 64)} ({n_par/1e6:.1f}M params) "
          f"epoch={ck.get('epoch','?')} best={ck.get('best_score', float('nan')):.4f}")
    return model, n_classes


@torch.no_grad()
def predict(model, img_t, device, mode, min_area, max_dist):
    """
    Returns a mask in the SHARED id space, post-processed as deployment does — which
    means keep_largest: one closed region per class, since each class IS one vessel
    (see clean_binary). A figure showing a class split across two blobs would be
    advertising a mask whose centroid the geometry stage cannot use.
    """
    logits = model(img_t.unsqueeze(0).to(device))
    pred = F.softmax(logits.float(), 1).argmax(1)[0].cpu().numpy().astype(np.uint8)

    if mode == "vessel":
        # Merge first, then clean as ONE region — a phantom has exactly one tube, and
        # its vein/artery split carries no information. This mirrors the deployed
        # binary path: (p1 + p2) > thr -> clean.
        fg = clean_binary((pred == CLASS_VEIN) | (pred == CLASS_ARTERY),
                          min_area=min_area, max_dist=max_dist)
        return np.where(fg, CLASS_VESSEL, CLASS_BG).astype(np.uint8)

    # Human frames hold TWO distinct vessels, so clean per class and reassemble by
    # assignment. `v * 1 + a * 2` would sum an overlap into id 3 and invent a phantom
    # class in a Mus-V panel; clean_binary's morphological CLOSE makes that overlap real.
    v = clean_binary(pred == CLASS_VEIN,   min_area=min_area, max_dist=max_dist)
    a = clean_binary(pred == CLASS_ARTERY, min_area=min_area, max_dist=max_dist)
    out = np.zeros_like(pred)
    out[v] = CLASS_VEIN
    out[a] = CLASS_ARTERY          # artery wins ties: it is the better-segmented class
    return out


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
    Prefer frames that actually show what the panel is meant to demonstrate: for
    Mus-V that means BOTH vessels present, otherwise a random draw can hand back a
    frame whose vein is collapsed to zero area (7.5% of val frames are) and the
    figure would silently argue the model missed it.
    """
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(ds))
    picked, fallback = [], []
    for i in order:
        _, m, _ = ds[int(i)]
        m = m.squeeze(0).numpy()
        if mode == "full3":
            ok = (m == CLASS_VEIN).sum() > 1500 and (m == CLASS_ARTERY).sum() > 1500
        else:
            ok = (m > 0).sum() > 800
        (picked if ok else fallback).append(int(i))
        if len(picked) == n:
            break
    idxs = (picked + fallback)[:n]
    out = []
    for i in idxs:
        img, msk, lm = ds[i]
        out.append((img, msk.squeeze(0).numpy().astype(np.uint8), lm))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--n", type=int, default=2, help="每个域取几张")
    ap.add_argument("--seed", type=int, default=42, help="全局 seed（三个域一起换）")
    # Per-domain overrides: re-rolling one row should not disturb rows you already
    # like. Leave unset and the domain keeps deriving from --seed as before.
    ap.add_argument("--seed-musv",    type=int, default=None)
    ap.add_argument("--seed-phantom", type=int, default=None)
    ap.add_argument("--seed-cca",     type=int, default=None)
    ap.add_argument("--min-area", type=int, default=150)
    ap.add_argument("--max-dist", type=float, default=40.0)
    ap.add_argument("--phantom", default="customer_3d_phantom",
                    choices=["customer_3d_phantom", "phantom_taobao"])
    ap.add_argument("--out", default="results/unet/figures/segmentation_samples")
    ap.add_argument("--preview", action="store_true",
                    help="挑 seed 用：只出 PNG、150 dpi，快 ~5 倍。定稿时不要加，"
                         "默认会出 svg/pdf/png 三种、600 dpi")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, n_classes = load_model(args.ckpt, device)
    if n_classes < 3:
        raise SystemExit("这个脚本只支持三分类模型（背景/静脉/动脉）。"
                         f"这个 checkpoint 是 {n_classes} 类的旧二分类模型，"
                         "对应的绘图脚本已随二分类模型一并删除。")

    # Phantom last, at the bottom: it is the deployment domain, so it reads as the
    # figure's conclusion. Anatomical datasets (the auxiliary training domains) come
    # first, phantom is where the guidance actually runs.
    domains = [
        ("musv",     "Mus-V",   "vein + artery"),
        ("mendeley", "CCA",     "artery only"),
        ("phantom",  "Phantom", "vessel only"),
    ]
    ds_name = {"musv": "musv", "phantom": args.phantom, "mendeley": "mendeley"}

    seed_override = {"musv": args.seed_musv, "phantom": args.seed_phantom,
                     "mendeley": args.seed_cca}

    # Per-domain seed offset is keyed by domain NAME, not row position, so reordering
    # the rows (e.g. moving phantom to the bottom) does not change which frames each
    # domain samples. Row order and sample choice are independent knobs.
    seed_offset = {"musv": 0, "mendeley": 37, "phantom": 74}

    rows = []
    for dom, _, _ in domains:
        ds = ReadDataset("test", 1, ds_name[dom])
        seed = seed_override[dom] if seed_override[dom] is not None else args.seed + seed_offset[dom]
        print(f"  {dom:9s} seed={seed}")
        for s in pick_samples(ds, args.n, seed, ds.label_mode):
            rows.append((dom, ds.label_mode, s))

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

    for r, (dom, mode, (img_t, gt, _)) in enumerate(rows):
        pred = predict(model, img_t, device, mode, args.min_area, args.max_dist)
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

    # Default = SVG only (vector, the paper format). Rendering PDF+PNG on top was the
    # bulk of the wall-clock — data prep is only ~2 s. --preview drops to a quick 150-dpi
    # PNG for eyeballing seeds; the SVG at 600 dpi is the deliverable.
    fmts = ("png",) if args.preview else ("svg",)
    dpi = 150 if args.preview else None
    saved = FigureConfig.save(fig, args.out, formats=fmts, dpi=dpi)
    print("Saved → " + "  +  ".join(str(p) for p in saved))


if __name__ == "__main__":
    main()
