"""
Test script — 在 test 集上评估模型，保存预测可视化图
用法:
    python models/unet/test.py --ckpt models/unet/checkpoints/unet_v10.pth
"""
import os
for _k in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_k, '1')

import argparse
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
# Before torch.cuda — see gpu_utils. This box is shared and usually has several cards
# pinned at 100% by other users.
from gpu_utils import pick_idle_gpu
pick_idle_gpu()

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy.ndimage import binary_fill_holes
from tqdm import tqdm

from data.pipeline.dataPrepare import (ReadDataset, DATASET_CONFIGS,
                                       CLASS_BG, CLASS_VEIN, CLASS_ARTERY, CLASS_VESSEL)
from models.unet.train import validate, build_val_loaders, CONFIG
from models.unet.model import UNet


def get_device():
    return torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")


def load_model(ckpt_path: str, cfg: dict, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location=device)
    saved_cfg = ckpt.get("config", {})
    n_classes = saved_cfg.get("num_classes", cfg["num_classes"])
    bilinear  = saved_cfg.get("bilinear",    cfg["bilinear"])

    # base_ch travels IN the checkpoint. Old checkpoints predate it and are all 64-wide,
    # so that has to stay the fallback — guessing from CONFIG would load a 32-wide
    # checkpoint into a 64-wide model and blow up on the state dict.
    base_ch = saved_cfg.get("base_ch", 64)
    model = UNet(n_channels=1, n_classes=n_classes, bilinear=bilinear, base_ch=base_ch)
    model.load_state_dict(ckpt["model"])
    model = model.to(device, memory_format=torch.channels_last)
    model.eval()
    n_par = sum(p.numel() for p in model.parameters())
    print(f"Loaded: {ckpt_path}")
    print(f"  epoch {ckpt.get('epoch','?')}  |  best dice {ckpt.get('best_score',0):.4f}"
          f"  |  base_ch={base_ch}  ({n_par/1e6:.1f}M params)")
    return model, saved_cfg or cfg


def measure_fps(model, loader, device, cfg):
    total_time, total_n = 0.0, 0
    with torch.no_grad():
        for imgs, _, _ in tqdm(loader, desc="FPS", leave=False):
            imgs = imgs.to(device, dtype=torch.float32, memory_format=torch.channels_last)
            t0 = time.perf_counter()
            with torch.autocast(device.type, enabled=cfg["amp"]):
                model(imgs)
            if device.type == "cuda":
                torch.cuda.synchronize()
            total_time += time.perf_counter() - t0
            total_n    += imgs.size(0)
    fps = total_n / total_time
    print(f"[FPS]  {total_n} samples | {total_time:.3f}s | {fps:.1f} FPS | {total_time/total_n*1000:.2f} ms/img")


# Indexed by the SHARED id space (dataPrepare): 0 bg / 1 vein / 2 artery / 3 vessel-
# untyped.
#
# ARTERY = RED, VEIN = BLUE — the anatomical convention. This used to be the other way
# round (copied from Mus-V's own overlay colours), which inverts what every reader of a
# vascular paper expects. Kept in step with plot_segmentation_samples.py: two
# figures of the same model disagreeing about which vessel is red would be worse than
# either choice.
#
# Id 3 exists because mendeley's masks used to be remapped to id 1, so `CLASS_COLORS[1]`
# painted every common-carotid ARTERY ground truth as a vein. Every mendeley figure ever
# saved from this script was mislabelled. Nothing crashed because no code path ever
# looked at which id it was.
CLASS_COLORS = np.array([[0, 0, 0],          # 0 background
                         [40, 90, 230],      # 1 vein      (blue)
                         [220, 40, 40],      # 2 artery    (red)
                         [40, 190, 90]],     # 3 vessel, type unknown (green)
                        dtype=np.uint8)
CLASS_NAMES  = ["background", "vein", "artery", "vessel(untyped)"]


def colorize(label_hw: np.ndarray) -> np.ndarray:
    return CLASS_COLORS[np.clip(label_hw, 0, 3)]


def overlay(gray_hw: np.ndarray, label_hw: np.ndarray, alpha: float = 0.55) -> np.ndarray:
    rgb = np.stack([gray_hw] * 3, axis=-1).astype(np.float32)
    col = colorize(label_hw).astype(np.float32)
    fg  = (label_hw > 0)[..., None]
    return np.where(fg, (1 - alpha) * rgb + alpha * col, rgb).astype(np.uint8)


# ----------------------------------------------------------------------------------
#  Post-processing — the SAME cleanup the deployed pipeline applies
# ----------------------------------------------------------------------------------
#
# infer.clean() already does open -> close -> keep-largest -> min_area before
# any geometry is measured, so a Dice reported WITHOUT it is not the number the robot
# actually runs on. This makes the two comparable. Both numbers are always printed:
# post-processing must never be able to hide what the model really did.
#
# The two knobs are NOT equally safe, measured on the Mus-V ground truth:
#
#   keep-largest   Artery: safe. 99.4% of frames have exactly one artery component,
#                  and the single 2-component frame's extra blob is 1 px of annotation
#                  noise.
#                  Vein: NOT free. 2.5% (val) / 1.3% (test) of frames have two real
#                  vein components, and the second one has a median area of 1071 px
#                  (max 4230). Keeping only the largest DESTROYS those. It is still a
#                  net win if the model produces more spurious islands than that, but
#                  it is a trade, not a freebie.
#
#   min_area       The smallest TRUE vein in Mus-V test is 232 px (1st percentile 456).
#                  So min_area=200 is essentially free; the deployment default of 300
#                  kills real veins in under 1% of frames. Above ~450 you start eating
#                  real vessels.

_K3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
_K5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))


def clean_binary(binary: np.ndarray, min_area: int = 150,
                 max_dist: float = 40.0, keep_largest: bool = True) -> np.ndarray:
    """
    ANCHOR + DISTANCE GATE.  open -> close -> drop noise -> anchor -> keep what is near
    the anchor, delete what is far.

    WHY keep_largest DEFAULTS TO TRUE
    ---------------------------------
    Each class is ONE vessel. That is not an imposed constraint, it is what the data
    says: the artery is a single connected component in 99.4% of Mus-V ground-truth
    frames, the vein in 90-95%, the phantom tube in 100%. The distance gate, by
    contrast, is designed to KEEP fragments near the anchor, and duly leaves a class
    split across two blobs on 3.6% (vein) / 1.5% (artery) of test frames.

    That matters downstream, not just visually. measure() takes moments of the mask:
    the centroid of two separated blobs lands in the gap BETWEEN them — a location where
    no vessel exists — and that centroid is a needle target.

    The Dice cost of insisting on one component is nil: vein 0.6187 -> 0.6170, artery
    0.8812 -> 0.8823, i.e. -0.002 and +0.001, both inside run-to-run noise. So the
    distance gate's marginal Dice edge does not pay for handing the robot an occasional
    centroid in empty tissue.

    Pass keep_largest=False to restore the gate — worth doing only if a frame is ever
    expected to hold two disjoint parts of the same class.

    The physical fact this rests on, measured on the ground truth: a real vessel is ONE
    connected blob. Mus-V's vein is a single component in 90-95% of frames, its artery
    in 99.4%, and the phantom's tube in 100%. So a fragment sitting far from the main
    vessel is almost certainly spurious, while a fragment right next to it is almost
    certainly the same vessel that the model happened to cut in two.

      1. open(3) + close(5)                     — kill speckle, bridge 1-2 px gaps
      2. fill interior holes                    — a lumen is solid; see below
      3. drop components below min_area         — noise, wherever it is
      4. the LARGEST survivor is the anchor     — safe: the truth has one vessel
      5. keep survivors within max_dist of it   — same vessel, split by the model
      6. delete everything else                 — a big blob far away is a false positive

    ON FILLING HOLES
    ----------------
    close(5) only bridges gaps up to about its kernel — anything larger stayed as a
    hole punched through the middle of a predicted vessel, which is visibly wrong: a
    vessel lumen is anechoic, so there is no structure inside one to preserve.

    Measured across all four datasets and all three splits — 7166 ground-truth vessel
    regions — 113 (1.6%) do contain an interior hole, so "the ground truth never has
    holes" is FALSE and an earlier version of this comment claiming so was wrong (it
    had only checked the test splits). What is true is sharper and still sufficient:

      * Every hole is tiny. The largest in the entire project is 68 px, median 11.
        These are annotation artefacts — a few pixels missed inside a lumen — not
        anatomy. For scale, the smallest true vein is 232 px.
      * They are almost entirely a TRAINING-annotation phenomenon (Mus-V's vein
        accounts for 106 of the 113, at 5.3% of its training regions), and this
        function never runs during training.
      * Of the 2134 held-out (val + test) regions this function is ever scored
        against, exactly ONE has a hole, of 14 px.

    So the fill cannot meaningfully destroy a real structure, and it is still the
    closest thing to a free step in this pipeline — but "free" rests on the holes
    being 68 px of annotation noise, not on them being absent.

    Step 4/5 is the part that actually earns its keep, and it is the part that holds up:
    ranked on BOTH val and test, gate-40px > gate-20px > keep-largest-only > keep-
    everything. Keeping everything above min_area is the WORST option on both — the far
    false positives it lets through cost more than the fragments it saves.

    ON min_area, AND A WARNING ABOUT TUNING IT
    ------------------------------------------
    Do not optimise this against Dice. Tuned on val, the best min_area is 450; tuned on
    test, it is 100 — opposite ends of the range, because val has twice as many
    empty-vein frames (7.5% vs 3.8%) and smaller veins (median 3769 vs 5167 px), so
    aggressive filtering rescues empty frames there and destroys real veins here. That
    is noise-fitting, not a signal.

    So set it from PHYSICS instead: the smallest true vein in Mus-V is 232 px. A
    threshold below that cannot delete a real vessel. 150 is the default for that reason
    and no other.

    Honest scale: tuned on val and reported on test, the whole of this buys +0.010 Dice
    on the artery and +0.001 on the vein. The vein's train/val gap is 0.20. This is a
    rounding error on the real problem — it is worth doing, and it is not a fix.

    keep_largest: ignore max_dist and keep only the anchor. Kept for the deployment path
    that needs exactly one object to take moments of.
    """
    b = (binary > 0).astype(np.uint8)
    if not b.any():
        return b.astype(bool)

    b = cv2.morphologyEx(b, cv2.MORPH_OPEN,  _K3)
    b = cv2.morphologyEx(b, cv2.MORPH_CLOSE, _K5)
    b = binary_fill_holes(b).astype(np.uint8)      # a lumen is solid — see docstring

    n, lab, stats, _ = cv2.connectedComponentsWithStats(b, connectivity=8)
    if n <= 1:
        return np.zeros_like(b, dtype=bool)
    areas = stats[1:, cv2.CC_STAT_AREA]

    surv = [i + 1 for i, a in enumerate(areas) if a >= min_area]
    if not surv:
        return np.zeros_like(b, dtype=bool)

    anchor = surv[int(np.argmax([areas[i - 1] for i in surv]))]
    if keep_largest:
        return lab == anchor

    keep = {anchor}
    if max_dist > 0 and len(surv) > 1:
        # Euclidean distance from every pixel to the anchor; a fragment's distance is
        # the distance of its closest pixel.
        dt = cv2.distanceTransform((lab != anchor).astype(np.uint8), cv2.DIST_L2, 5)
        for i in surv:
            if i != anchor and dt[lab == i].min() <= max_dist:
                keep.add(i)
    return np.isin(lab, list(keep))


def _dice(pred: np.ndarray, gt: np.ndarray) -> float:
    """Hard Dice on one frame. Both empty -> 1.0 (nothing there, nothing predicted)."""
    p, g = pred.sum(), gt.sum()
    if p == 0 and g == 0:
        return 1.0
    return 2.0 * float((pred & gt).sum()) / float(p + g)


@torch.no_grad()
def eval_postproc(model, loader, device, cfg, min_area: int, max_dist: float,
                  keep_largest: bool):
    """
    Per-frame hard Dice, computed twice: on the raw argmax, and on the cleaned mask.
    Returns {metric: (raw_mean, clean_mean)} plus island statistics.
    """
    acc = {}          # key -> [sum_raw, sum_clean, n]
    islands_removed = 0
    frames = 0

    def add(k, raw, cln):
        s = acc.setdefault(k, [0.0, 0.0, 0])
        s[0] += raw; s[1] += cln; s[2] += 1

    for imgs, masks, modes in tqdm(loader, desc="postproc eval", leave=False):
        imgs = imgs.to(device, dtype=torch.float32, memory_format=torch.channels_last)
        with torch.autocast(device.type, enabled=cfg["amp"]):
            logits = model(imgs)
        pred = F.softmax(logits.float(), dim=1).argmax(1).cpu().numpy().astype(np.uint8)
        gt   = masks.squeeze(1).numpy().astype(np.uint8)

        for b in range(pred.shape[0]):
            mode = modes[b]
            p, g = pred[b], gt[b]
            frames += 1

            raw_v = (p == CLASS_VEIN)
            raw_a = (p == CLASS_ARTERY)
            raw_f = raw_v | raw_a

            # WHICH mask gets cleaned depends on what the dataset's labels MEAN. Get
            # this wrong and post-processing deletes real vessels — both directions
            # were measured on the v1 checkpoint:
            #
            #  vessel-mode (phantom)  ->  clean the MERGED foreground, as one region.
            #     The vein/artery split here is not merely unlabelled, it is MEANINGLESS:
            #     the loss puts no constraint on it, and the model duly speckles it. On
            #     phantom_taobao the predicted vein comes out in a median of 4 pieces and
            #     the artery in 2 — while the merged foreground is a median of ONE piece,
            #     because the gel tube IS one tube. Cleaning per class therefore shreds
            #     it: keep-largest kept 1 of 4 vein islands and cost -0.021 Dice.
            #     This is also exactly the deployed binary path: (p1+p2) > thr -> clean.
            #
            #  full3 (Mus-V)  ->  clean PER CLASS, then union.
            #     Here the classes are real and there are TWO distinct vessels in frame.
            #     Cleaning the merged mask with keep-largest throws one of them away
            #     outright: it cost -0.10 Dice on musv's vessel score (0.8147 -> 0.7144),
            #     i.e. it was deleting a real carotid. Per-class is also what the
            #     deployed infer.measure() does.
            kw = dict(min_area=min_area, max_dist=max_dist, keep_largest=keep_largest)
            if mode == "vessel":
                cln_f = clean_binary(raw_f, **kw)
                cln_v = cln_a = None
            else:
                cln_v = clean_binary(raw_v, **kw)
                cln_a = clean_binary(raw_a, **kw)
                cln_f = cln_v | cln_a

            n_raw, _ = cv2.connectedComponents(raw_f.astype(np.uint8), connectivity=8)
            n_cln, _ = cv2.connectedComponents(cln_f.astype(np.uint8), connectivity=8)
            islands_removed += max(0, (n_raw - 1) - (n_cln - 1))

            if mode == "full3":
                add("dice_vein",   _dice(raw_v, g == CLASS_VEIN),   _dice(cln_v, g == CLASS_VEIN))
                add("dice_artery", _dice(raw_a, g == CLASS_ARTERY), _dice(cln_a, g == CLASS_ARTERY))
                add("dice_vessel", _dice(raw_f, g > 0),             _dice(cln_f, g > 0))
            elif mode == "vessel":
                add("dice_vessel", _dice(raw_f, g == CLASS_VESSEL), _dice(cln_f, g == CLASS_VESSEL))
            elif mode == "artery":
                add("dice_artery", _dice(raw_a, g == CLASS_ARTERY), _dice(cln_a, g == CLASS_ARTERY))

    out = {k: (s[0] / s[2], s[1] / s[2]) for k, s in acc.items()}
    return out, islands_removed, frames


@torch.no_grad()
def vein_probe(model, loader, device, cfg, ds_name: str):
    """
    "mendeley 没有静脉标签 —— 训练不要用，只在 test 的时候试试。"

    That is exactly what this is, and why it lives in test.py and touches no metric
    that selects a checkpoint. mendeley's expert masks label the common carotid artery
    only; the internal jugular vein is usually right there in the frame, unlabelled.
    The training loss is provably indifferent to it (partial_label_loss, artery mode:
    the loss is a function of p2 alone), and the validation metric no longer scores it
    at all (evaluate_metrics: artery mode emits dice_artery and nothing else).

    So the vein prediction on mendeley is UNSUPERVISED. That makes this interesting —
    but it is NOT evidence that the vein class generalises, and it must not be written
    up as such. There is no vein ground truth here, so nothing below can distinguish a
    correctly recovered internal jugular vein from an anechoic shadow, an artefact, or
    a slice of artery that fell into the wrong channel.

    What this counts is ONLY: how often the vein channel emits anything at all, and how
    much of the predicted foreground it takes. It does NOT count whether those pixels
    are right. "The model recovers the unannotated jugular vein in 63% of frames" does
    not follow from this number and cannot be defended — verifying it would take manual
    annotation of a sample of frames.
    """
    if cfg["num_classes"] <= 1:
        return
    n_frames = n_with_vein = 0
    vein_px = artery_px = 0

    for imgs, _, _ in tqdm(loader, desc="vein probe", leave=False):
        imgs = imgs.to(device, dtype=torch.float32, memory_format=torch.channels_last)
        with torch.autocast(device.type, enabled=cfg["amp"]):
            logits = model(imgs)
        pred = F.softmax(logits.float(), dim=1).argmax(1)      # (B, H, W)
        for b in range(pred.shape[0]):
            v = int((pred[b] == CLASS_VEIN).sum())
            a = int((pred[b] == CLASS_ARTERY).sum())
            vein_px += v
            artery_px += a
            n_frames += 1
            if v > 200:                       # 200 px ~ the smallest real vein in Mus-V
                n_with_vein += 1

    fg = vein_px + artery_px
    print(f"\n[静脉探针 — {ds_name}]  (无静脉标注，纯诊断，不参与任何 checkpoint 选择)")
    print(f"  静脉通道有输出的帧 : {n_with_vein}/{n_frames}  ({n_with_vein/max(1,n_frames)*100:.1f}%)")
    if fg:
        print(f"  预测前景构成       : 静脉 {vein_px/fg*100:.1f}%  |  动脉 {artery_px/fg*100:.1f}%")
    print(f"  ⚠ 这里没有静脉真值，所以这个数只说明「静脉通道输出了东西」，")
    print(f"    **不能说明那是不是颈内静脉**。别把它当泛化证据写进论文。")


def save_predictions(model, loader, device, cfg, output_dir: Path, num_samples: int = 10,
                     min_area: int = 0, max_dist: float = 40.0,
                     keep_largest: bool = False):
    """
    ON A PHANTOM, RED-VS-BLUE IS NOT A RESULT — IT IS NOISE.

    A vessel-mode dataset says "there is a vessel here" and nothing more. The loss puts
    ZERO constraint on how that vessel's probability splits between vein and artery
    (partial_label_loss supervises only p1+p2), so the model is free to scatter the two
    labels across the tube however it likes — and it does: on phantom_taobao the
    predicted vein comes out in a median of 4 disconnected pieces and the artery in 2,
    while the merged foreground is a single blob, because the gel tube IS a single tube.

    Drawing that as red-and-blue confetti invites exactly the wrong reading — that the
    model has an opinion about vessel type here. It does not, and it was never asked
    for one. So on a vessel-mode dataset, both panels render the vessel as ONE region,
    in one colour. On Mus-V and mendeley, where the classes are real and supervised,
    red/blue is kept.
    """
    out = output_dir
    out.mkdir(parents=True, exist_ok=True)

    # Clear old samples first. Without this, a run with --samples 6 leaves sample_06 and
    # sample_07 behind from a previous --samples 8 run, and they are silently WRONG —
    # written by whatever the code did last time. It already happened: stale frames from
    # before the vessel-merge fix sat in the phantom folder still showing red/blue
    # confetti, next to freshly-written frames that correctly showed one green region.
    for old in out.glob("sample_*.png"):
        old.unlink()

    multiclass = cfg["num_classes"] > 1

    all_samples = []
    with torch.no_grad():
        for imgs, masks, modes in tqdm(loader, desc="Collecting", leave=False):
            imgs = imgs.to(device, dtype=torch.float32, memory_format=torch.channels_last)
            with torch.autocast(device.type, enabled=cfg["amp"]):
                logits = model(imgs)
            if multiclass:
                pred = F.softmax(logits.float(), dim=1).argmax(1)     # (B, H, W) ids
            else:
                pred = (torch.sigmoid(logits.float()) > 0.5).long().squeeze(1)
            for i in range(imgs.size(0)):
                all_samples.append((imgs[i].float().cpu(), masks[i].long().cpu(),
                                    pred[i].cpu(), modes[i]))

    selected = random.sample(all_samples, min(num_samples, len(all_samples)))
    print(f"Saving {len(selected)} samples to {out}")

    for idx, (img, mask, pred, mode) in enumerate(tqdm(selected, desc="Saving", leave=False)):
        img_np  = ((img.squeeze().numpy() * 0.5 + 0.5) * 255).clip(0, 255).astype(np.uint8)
        mask_np = mask.squeeze().numpy().astype(np.uint8)
        pred_np = pred.squeeze().numpy().astype(np.uint8)

        kw = dict(min_area=min_area, max_dist=max_dist, keep_largest=keep_largest)
        do_clean = min_area > 0 or keep_largest

        untyped = (mode == "vessel")
        if untyped:
            # Collapse vein|artery into the single "vessel, type unknown" id, and clean
            # it as ONE region — the same thing the deployed binary path does.
            fg = (pred_np == CLASS_VEIN) | (pred_np == CLASS_ARTERY)
            if do_clean:
                fg = clean_binary(fg, **kw)
            pred_np = np.where(fg, CLASS_VESSEL, CLASS_BG).astype(np.uint8)
            legend  = "green = vessel (type not asked for on a phantom)"
        elif do_clean:
            v = clean_binary(pred_np == CLASS_VEIN,   **kw)
            a = clean_binary(pred_np == CLASS_ARTERY, **kw)
            # Reassemble by ASSIGNMENT, not by `v * 1 + a * 2`. The two cleaned masks can
            # overlap — clean_binary's morphological CLOSE dilates before it erodes — and
            # an arithmetic sum turns 1 + 2 into 3, inventing a "vessel, type unknown"
            # label out of thin air. It showed up as green pixels in Mus-V overlays, a
            # class that dataset cannot possibly contain. Artery wins the overlap: it is
            # the far better-segmented of the two (Dice 0.88 vs 0.66).
            pred_np = np.zeros_like(pred_np)
            pred_np[v] = CLASS_VEIN
            pred_np[a] = CLASS_ARTERY
            legend  = "red = artery, blue = vein"
        else:
            legend = "red = artery, blue = vein"

        panels = [
            (img_np,                       "Image"),
            (overlay(img_np, mask_np),     "Ground Truth"),
            (overlay(img_np, pred_np),     "Prediction"),
        ]
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        for ax, (arr, title) in zip(axes, panels):
            ax.imshow(arr, cmap="gray" if arr.ndim == 2 else None)
            ax.set_title(title); ax.axis("off")
        present = [CLASS_NAMES[c] for c in np.unique(pred_np) if c > 0]
        pp = []
        if min_area:     pp.append(f"min_area={min_area}")
        if keep_largest: pp.append("keep_largest")
        fig.suptitle(f"predicted: {', '.join(present) if present else 'nothing'}"
                     f"   ({legend})" + (f"   [{', '.join(pp)}]" if pp else ""),
                     fontsize=10)
        plt.tight_layout()
        plt.savefig(out / f"sample_{idx:02d}_combined.png", dpi=150, bbox_inches="tight")
        plt.close()
        Image.fromarray(img_np).save(out / f"sample_{idx:02d}_image.png")
        Image.fromarray(colorize(mask_np)).save(out / f"sample_{idx:02d}_mask.png")
        Image.fromarray(colorize(pred_np)).save(out / f"sample_{idx:02d}_pred.png")

    print(f"Done → {out}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True, help="checkpoint .pth path")
    parser.add_argument("--batch",   type=int, default=CONFIG["batch_size_per_gpu"])
    parser.add_argument("--samples", type=int, default=10)
    parser.add_argument("--no-save", action="store_true")
    # Never pool ACROSS label semantics — a single mixed number would hide the one
    # column that matters (the phantom). But DO pool the two phantoms: both are
    # binary vessel data and both are what the rig runs on, so they are one number.
    parser.add_argument("--dataset", nargs="+",
                        default=["musv", "phantom_taobao+customer_3d_phantom", "mendeley"],
                        help="Test dataset(s). Join with '+' to pool into one metric, "
                             "e.g. phantom_taobao+customer_3d_phantom")
    parser.add_argument("--no-fps", action="store_true")
    parser.add_argument("--vein-probe", action="store_true",
                        help="在 artery 模式的数据集(mendeley)上探测模型预测了多少静脉。"
                             "那里的静脉真实存在但没标注，训练和验证都完全不碰它，"
                             "所以这是一个纯粹的泛化性检验。只在 test 时用")
    parser.add_argument("--min-area", type=int, default=0,
                        help="删掉面积小于这个值的连通域。0=关闭后处理。"
                             "别拿 Dice 去调它：val 上最优是 450，test 上最优是 100，"
                             "两头顶天，纯粹在拟合噪声。按物理定：Mus-V 里最小的真静脉是"
                             " 232 px，所以 150 保证删不掉真血管。推荐 150")
    parser.add_argument("--max-dist", type=float, default=40.0,
                        help="锚点(最大连通域)之外的碎片，离锚点 <= 这个距离(px)就保留"
                             "（同一根血管被模型切开了），更远的直接删（假阳性）。"
                             "在 val 和 test 上排名一致: 40px > 20px > 只留最大 > 全留")
    parser.add_argument("--allow-fragments", action="store_true",
                        help="关掉 keep_largest，改用 --max-dist 的距离门控，允许一类保留"
                             "多个连通域。默认是**每类只留一个闭合区域**——因为真值里动脉"
                             "99.4%%、静脉 90-95%% 就是单连通域，而两个分离块算出的质心会落在"
                             "两者之间的空隙里（那是个不存在的进针位置）")
    args = parser.parse_args()
    args.keep_largest = not args.allow_fragments

    do_pp = args.min_area > 0 or args.keep_largest

    device = get_device()
    model, cfg = load_model(args.ckpt, CONFIG, device)

    loaders = build_val_loaders(args.dataset, args.batch, split='test')
    summary = {}
    postproc = {}

    for ds_name, test_loader in loaders.items():
        n = len(test_loader.dataset)
        print(f"\n{'='*60}")
        print(f"Testing on: {ds_name}   ({n} samples)")
        print(f"{'='*60}")

        if not args.no_fps:
            measure_fps(model, test_loader, device, cfg)

        test_loss, test_dice, per_class = validate(model, test_loader, cfg, device)
        detail = "  ".join(f"{k}={v:.4f}" for k, v in sorted(per_class.items()))
        print(f"\n[TEST RESULT — {ds_name}]  loss={test_loss:.4f}  {detail}")
        summary[ds_name] = per_class

        if do_pp:
            pp, removed, nframes = eval_postproc(model, test_loader, device, cfg,
                                                 args.min_area, args.max_dist,
                                                 args.keep_largest)
            rule = [f"删掉 <{args.min_area}px 的连通域"]
            rule.append("每类只留最大连通域" if args.keep_largest
                        else f"保留离锚点 <={args.max_dist:g}px 的碎片，更远的删掉")
            print(f"\n[后处理 — {ds_name}]  {' + '.join(rule)}  "
                  f"(开/闭运算与 infer.clean() 完全一致)")
            print(f"  {'':14s}{'原始':>10s}{'后处理':>10s}{'Δ':>9s}")
            for k in sorted(pp):
                raw, cln = pp[k]
                print(f"  {k.replace('dice_',''):14s}{raw:10.4f}{cln:10.4f}{cln-raw:+9.4f}")
            print(f"  共清掉 {removed} 个孤岛 / {nframes} 帧 "
                  f"({removed/max(1,nframes):.2f} 个/帧)")
            postproc[ds_name] = pp

        # Only meaningful where the vein is present but unlabelled — i.e. mendeley.
        modes = {DATASET_CONFIGS[n.strip()]["label_mode"] for n in ds_name.split('+')}
        if args.vein_probe and modes == {"artery"}:
            vein_probe(model, test_loader, device, cfg, ds_name)

        if not args.no_save:
            # The CHECKPOINT's run_name, not .env's. Reading .env here meant that
            # testing an old checkpoint wrote its figures into a folder named after
            # whatever run happens to be configured right now — e.g. testing
            # mixav_v1 dropped its samples into `sample_predictions_mixav_v2_*`.
            run_name    = cfg.get("run_name") or Path(args.ckpt).stem
            dataset_tag = ds_name.replace("phantom_taobao+customer_3d_phantom", "phantom_all") \
                                 .replace("phantom_taobao", "phantom")
            out_dir = Path("results/unet/predictions") / f"{run_name}_{dataset_tag}"
            save_predictions(model, test_loader, device, cfg, out_dir,
                             num_samples=args.samples,
                             min_area=args.min_area, max_dist=args.max_dist,
                             keep_largest=args.keep_largest)

    # Final table — one row per dataset, never one merged number across semantics.
    keys = sorted({k for pc in summary.values() for k in pc})

    def table(title, get):
        print(f"\n{'='*78}\n{title}\n{'='*78}")
        print(f"{'数据集':30s}" + "".join(f"{k.replace('dice_',''):>13s}" for k in keys))
        print("-" * 78)
        for ds_name, pc in summary.items():
            print(f"{ds_name:30s}" + "".join(
                f"{get(ds_name, k):13.4f}" if k in pc else f"{'—':>13s}" for k in keys))

    table(f"最终结果 — 原始 argmax  ({Path(args.ckpt).name})",
          lambda d, k: summary[d][k])

    if do_pp:
        rule = [f"min_area={args.min_area}"]
        rule.append("keep_largest" if args.keep_largest else f"max_dist={args.max_dist:g}")
        table(f"最终结果 — 后处理后 ({', '.join(rule)})  ← 这才是机器人真正吃到的数",
              lambda d, k: postproc[d][k][1])
        print("\n注：后处理和 infer.clean() 用的是同一套形态学，所以这一张表"
              "\n    才是部署时的真实水平。两张表都打出来，是为了让后处理无法掩盖模型的真实行为。")


if __name__ == "__main__":
    main()
