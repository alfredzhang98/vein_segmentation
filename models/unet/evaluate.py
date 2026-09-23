"""
Test script — 在 test 集上评估模型，保存预测可视化图
用法:
    python models/unet/evaluate.py --ckpt models/unet/checkpoints/unet_v10.pth
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
from tqdm import tqdm

from data.pipeline.dataPrepare import (ReadDataset, DATASET_CONFIGS,
                                       CLASS_BG, CLASS_VEIN, CLASS_ARTERY, CLASS_VESSEL)
from models.unet.train import validate, build_val_loaders, CONFIG
from models.unet.model import UNet
from models.unet.postprocess import clean_binary, clean_labels, complete_region


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


# Cleanup is shared with deployment. Defaults retain thin compressed veins and
# allow either class to be absent; joint repair uses closing and enclosed-hole filling.


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
                cln_f = (complete_region(raw_f, min_area=min_area) if keep_largest
                         else clean_binary(raw_f, **kw))
                cln_v = cln_a = None
            else:
                if keep_largest:
                    repaired = clean_labels(p, min_area=min_area)
                    cln_v, cln_a = repaired == CLASS_VEIN, repaired == CLASS_ARTERY
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
                for name, raw, cln, truth in [("vein", raw_v, cln_v, g == CLASS_VEIN),
                                               ("artery", raw_a, cln_a, g == CLASS_ARTERY)]:
                    if not truth.any():
                        add(f"absent_{name}_fp_rate", float(raw.any()), float(cln.any()))
                        add(f"absent_{name}_fp_pixels", float(raw.sum()), float(cln.sum()))
                if not (g > 0).any():
                    add("empty_frame_fp_rate", float(raw_f.any()), float(cln_f.any()))
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
                fg = (complete_region(fg, min_area=min_area) if keep_largest
                      else clean_binary(fg, **kw))
            pred_np = np.where(fg, CLASS_VESSEL, CLASS_BG).astype(np.uint8)
            legend  = "green = vessel (type not asked for on a phantom)"
        elif do_clean:
            if keep_largest:
                repaired = clean_labels(pred_np, min_area=min_area)
                v, a = repaired == CLASS_VEIN, repaired == CLASS_ARTERY
            else:
                v = clean_binary(pred_np == CLASS_VEIN, **kw)
                a = clean_binary(pred_np == CLASS_ARTERY, **kw)
            # Display-only overlap is green; training/metrics remain A/V masks.
            pred_np = v.astype(np.uint8) + 2 * a.astype(np.uint8)
            legend = "red = artery, blue = vein, green = overlap"
        else:
            legend = "red = artery, blue = vein"

        vessel_np = np.where(pred_np > 0, CLASS_VESSEL, CLASS_BG).astype(np.uint8)
        panels = [
            (img_np,                       "Image"),
            (overlay(img_np, mask_np),     "Ground Truth"),
            (overlay(img_np, pred_np),     "Prediction"),
            (overlay(img_np, vessel_np),   "Vessel union"),
        ]
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
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
        Image.fromarray(colorize(vessel_np)).save(out / f"sample_{idx:02d}_vessel.png")

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
    parser.add_argument("--min-area", type=int, default=4,
                        help="评估网格上的连通域最小面积；默认4；可设1保留极小目标。增大会误删受压血管。")
    parser.add_argument("--max-dist", type=float, default=40.0,
                        help="仅 --allow-fragments 生效：保留距最大区域不超过此像素距离的碎片")
    parser.add_argument("--allow-fragments", action="store_true",
                        help="实验选项：允许同一类别多个邻近区域；默认每类最多一个，允许为空")
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
                  f"(默认共用联合类别修复 + 闭运算/填洞；allow-fragments使用旧距离筛选)")
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
        table(f"最终结果 — 后处理后 ({', '.join(rule)})",
              lambda d, k: postproc[d][k][1])
        print("\n注：后处理和 infer.clean() 用的是同一套连通域筛选，所以这一张表"
              "\n    但评估使用模型输入网格，部署使用原图网格；像素阈值不应跨网格直接比较。")


if __name__ == "__main__":
    main()
