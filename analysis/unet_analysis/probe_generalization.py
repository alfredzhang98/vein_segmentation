"""
Cross-dataset generalisation probes.

The question: when a model is trained where only ONE vessel type is labelled, does it
still *find* the other type — or has it learned to call it background?

Mus-V is the ruler. It labels vein AND artery in every one of its 3114 frames, so for
any model — including one that never saw a vein label in its life — we can measure
exactly how much of its predicted foreground lands on true veins vs true arteries.
That turns "肉眼看看" into a number.

Works on both model families:
  * n_classes=3  bg / vein / artery  (the new partial-label model)
  * n_classes=1  binary vessel       (the existing Exp-A..E checkpoints) — for these
                 the sigmoid foreground is the whole prediction, and we ask which of
                 Mus-V's two labelled vessels it landed on.

Usage:
    python analysis/unet_analysis/probe_generalization.py --ckpt models/unet/checkpoints/unet_v10.pth
    python probe_generalization.py --ckpt <x>.pth --datasets musv mendeley phantom_taobao
    python probe_generalization.py --ckpt <x>.pth --no-overlays
"""
import os
for _k in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_k, '1')

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

import sys
from pathlib import Path
# Repo root on sys.path: data pipeline in data/pipeline/, models in models/<arch>/.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from data.pipeline.dataPrepare import ReadDataset, DATASET_CONFIGS
from models.unet.model import UNet

BG, VEIN, ARTERY = 0, 1, 2
NAMES = {BG: "background", VEIN: "vein", ARTERY: "artery"}
# red = vein, blue = artery (BGR for cv2)
COLORS_BGR = np.array([[0, 0, 0], [40, 40, 220], [230, 90, 40]], dtype=np.uint8)


def load_model(ckpt_path: str, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    saved = ckpt.get("config", {}) or {}
    n_classes = saved.get("num_classes", 1)
    model = UNet(n_channels=1, n_classes=n_classes,
                 bilinear=saved.get("bilinear", False),
                 base_ch=int(saved.get("base_ch", 64)))
    model.load_state_dict(ckpt["model"])
    model = model.to(device, memory_format=torch.channels_last).eval()
    print(f"Loaded {Path(ckpt_path).name}")
    print(f"  trained on : {saved.get('dataset', '?')}")
    print(f"  n_classes  : {n_classes}   epoch {ckpt.get('epoch','?')}  "
          f"best {ckpt.get('best_score', float('nan')):.4f}")
    return model, n_classes, saved


@torch.no_grad()
def predict(model, loader, n_classes, device):
    """Yield (image_hw uint8, gt_hw uint8, pred_hw uint8) per sample.

    A binary model emits foreground as id 1. That id means "vessel, type unknown" —
    NOT vein — so callers must branch on `is_binary` before reading any vein/artery
    meaning into it. The probes below do; the printed tables relabel accordingly.
    """
    for imgs, masks, _ in tqdm(loader, desc="  inference", leave=False):
        x = imgs.to(device, dtype=torch.float32, memory_format=torch.channels_last)
        with torch.autocast(device.type, enabled=device.type == "cuda"):
            logits = model(x)
        if n_classes == 1:
            pred = (torch.sigmoid(logits.float()) > 0.5).long().squeeze(1)
        else:
            pred = F.softmax(logits.float(), dim=1).argmax(1)
        img = ((imgs.squeeze(1).numpy() * 0.5 + 0.5) * 255).clip(0, 255).astype(np.uint8)
        yield img, masks.squeeze(1).numpy().astype(np.uint8), pred.cpu().numpy().astype(np.uint8)


def probe_musv(model, n_classes, device, batch: int):
    """
    The measurable probe. Mus-V labels both vessels, so for ANY model we can ask:
    of the pixels that are truly vein, how many did the model put in *some* vessel
    class? A model trained only on arteries that still covers Mus-V's veins has
    learned "anechoic lumen", not "artery".
    """
    is_binary = n_classes == 1
    ds = ReadDataset('test', batch_size=batch, dataset_name='musv')
    loader = ds.get_dataloader(shuffle=False, num_workers=0)

    conf = np.zeros((3, 3), dtype=np.int64)   # GT class x predicted class
    for _, gt, pr in predict(model, loader, n_classes, device):
        for g in (BG, VEIN, ARTERY):
            m = (gt == g)
            if not m.any():
                continue
            for p in (BG, VEIN, ARTERY):
                conf[g, p] += int((pr[m] == p).sum())

    gt_tot = conf.sum(axis=1)
    # A binary model has no vein/artery axis: its id-1 output just means "vessel".
    # Collapse the two vessel columns so the table cannot be misread.
    pred_cols = [(BG, "background"), (VEIN, "vessel")] if is_binary \
        else [(BG, "background"), (VEIN, "vein"), (ARTERY, "artery")]

    print("\n  Pixel confusion on Mus-V test (row = ground truth, % of that class):")
    print("      GT \\ pred |" + "".join(f"{nm:>12s}" for _, nm in pred_cols))
    for g in (BG, VEIN, ARTERY):
        if gt_tot[g] == 0:
            continue
        cells = []
        for p, _ in pred_cols:
            v = conf[g, VEIN] + conf[g, ARTERY] if (is_binary and p == VEIN) else conf[g, p]
            cells.append(f"{100 * v / gt_tot[g]:11.2f}%")
        print(f"      {NAMES[g]:>11s} |{''.join(cells)}")

    # THE number: recall of each true vessel into ANY vessel class
    out = {}
    for g, label in ((VEIN, "vein"), (ARTERY, "artery")):
        if gt_tot[g] == 0:
            continue
        any_vessel = (conf[g, VEIN] + conf[g, ARTERY]) / gt_tot[g]
        out[f"vessel_recall_on_true_{label}"] = any_vessel
        print(f"\n  Recall of true {label:6s} into ANY vessel class: {100 * any_vessel:.2f}%")
        if not is_binary:
            correct = conf[g, VEIN if g == VEIN else ARTERY] / gt_tot[g]
            out[f"correct_class_recall_{label}"] = correct
            print(f"    ...of which assigned the CORRECT class : {100 * correct:.2f}%")

    if is_binary:
        print("\n  (binary model — it has no vein/artery concept. The recall numbers above\n"
              "   say whether its single 'vessel' class fires on Mus-V's veins at all.)")
    return out


def probe_av_call(model, n_classes, device, batch: int, ds_name: str):
    """
    On a dataset whose GT does NOT distinguish vein from artery (phantom = 'vessel',
    mendeley = 'artery' only), ask what the model *chooses* to call the vessels it
    finds. This is the free-choice the marginal loss deliberately leaves open.
    """
    if n_classes == 1:
        print(f"\n  [{ds_name}] binary model has no vein/artery head — skipping A/V call.")
        return {}

    ds = ReadDataset('test', batch_size=batch, dataset_name=ds_name)
    loader = ds.get_dataloader(shuffle=False, num_workers=0)
    mode = DATASET_CONFIGS[ds_name]["label_mode"]

    n_vein = n_art = 0
    outside_vein = outside_art = 0     # predicted vessel OUTSIDE the GT mask
    imgs_with_outside_vein = n_imgs = 0

    for _, gt, pr in predict(model, loader, n_classes, device):
        n_vein += int((pr == VEIN).sum())
        n_art  += int((pr == ARTERY).sum())
        for i in range(len(gt)):
            n_imgs += 1
            out_mask = (gt[i] == 0)             # everything the GT calls background
            ov = int(((pr[i] == VEIN) & out_mask).sum())
            oa = int(((pr[i] == ARTERY) & out_mask).sum())
            outside_vein += ov
            outside_art  += oa
            if ov > 200:                        # a real blob, not a few stray pixels
                imgs_with_outside_vein += 1

    tot = n_vein + n_art
    print(f"\n  [{ds_name}] of all pixels the model calls vessel:")
    if tot == 0:
        print("      (model predicted no vessel at all)")
        return {}
    print(f"      called VEIN  : {100 * n_vein / tot:5.1f}%")
    print(f"      called ARTERY: {100 * n_art  / tot:5.1f}%")

    res = {"frac_called_vein": n_vein / tot, "frac_called_artery": n_art / tot}

    if mode == "artery":
        # mendeley: GT labels the carotid ONLY. Vessel predicted outside it is the
        # model volunteering something the annotator never labelled — most plausibly
        # the internal jugular vein.
        print(f"\n      Predicted vessel OUTSIDE the labelled artery "
              f"(candidate unlabelled jugular vein):")
        print(f"        vein-labelled pixels  : {outside_vein:,}")
        print(f"        artery-labelled pixels: {outside_art:,}")
        print(f"        images with a vein blob >200px outside GT: "
              f"{imgs_with_outside_vein}/{n_imgs} "
              f"({100 * imgs_with_outside_vein / max(n_imgs,1):.1f}%)")
        res["imgs_with_unlabelled_vein"] = imgs_with_outside_vein / max(n_imgs, 1)
    return res


def save_overlays(model, n_classes, device, batch: int, ds_name: str,
                  out_dir: Path, n: int = 8):
    ds = ReadDataset('test', batch_size=batch, dataset_name=ds_name)
    loader = ds.get_dataloader(shuffle=False, num_workers=0)
    out_dir.mkdir(parents=True, exist_ok=True)

    is_binary = n_classes == 1
    gt_is_typed = DATASET_CONFIGS[ds_name]["label_mode"] == "full3"
    # A binary prediction is untyped — draw it green so it is never mistaken for a
    # vein. Same for an untyped GT (phantom / mendeley say "vessel"/"artery" only).
    UNTYPED = np.array([90, 200, 90], dtype=np.uint8)

    rows, taken = [], 0
    stride = max(1, len(ds) // (n * 2))
    idx = 0
    for img, gt, pr in predict(model, loader, n_classes, device):
        for i in range(len(img)):
            if idx % stride == 0 and taken < n:
                base = cv2.cvtColor(img[i], cv2.COLOR_GRAY2BGR)

                def blend(lbl, typed):
                    o = base.copy()
                    for c in (VEIN, ARTERY):
                        m = lbl == c
                        if m.any():
                            col = COLORS_BGR[c] if typed else UNTYPED
                            o[m] = (0.45 * o[m] + 0.55 * col).astype(np.uint8)
                    return o

                g = blend(gt[i], gt_is_typed)
                p = blend(pr[i], not is_binary)
                cv2.putText(g, "ground truth" + ("" if gt_is_typed else " (untyped)"),
                            (6, 18), 0, 0.5, (0, 255, 255), 1)
                cv2.putText(p, "prediction" + (" (untyped)" if is_binary else ""),
                            (6, 18), 0, 0.5, (0, 255, 255), 1)
                rows.append(np.hstack([base, g, p]))
                taken += 1
            idx += 1
        if taken >= n:
            break

    if rows:
        path = out_dir / f"probe_{ds_name}.png"
        cv2.imwrite(str(path), np.vstack(rows))
        legend = "green = vessel (untyped)" if is_binary else "red = vein, blue = artery"
        print(f"  overlays → {path}  ({legend})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--datasets", nargs="+",
                    default=["musv", "mendeley", "phantom_taobao"],
                    choices=list(DATASET_CONFIGS))
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--overlay-samples", type=int, default=8)
    ap.add_argument("--no-overlays", action="store_true")
    ap.add_argument("--out", default=None, help="overlay output dir")
    args = ap.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model, n_classes, saved = load_model(args.ckpt, device)

    tag = saved.get("run_name", Path(args.ckpt).stem)
    out_dir = Path(args.out or f"results/unet/probes/{tag}")

    for ds_name in args.datasets:
        print(f"\n{'=' * 66}\n{ds_name}\n{'=' * 66}")
        if DATASET_CONFIGS[ds_name]["label_mode"] == "full3":
            probe_musv(model, n_classes, device, args.batch)
        else:
            probe_av_call(model, n_classes, device, args.batch, ds_name)
        if not args.no_overlays:
            save_overlays(model, n_classes, device, args.batch, ds_name,
                          out_dir, args.overlay_samples)


if __name__ == "__main__":
    main()
