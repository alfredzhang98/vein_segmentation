"""
Test script — evaluate model on test set, save prediction visualizations
Usage:
    python test.py --ckpt outputs/checkpoints/your_checkpoint.pth
"""
import os
for _k in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_k, '1')

import argparse
import random
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

from dataPrepare import ReadDataset
from train import validate, CONFIG
from unet import UNet


def get_device():
    return torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")


def load_model(ckpt_path: str, cfg: dict, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location=device)
    saved_cfg = ckpt.get("config", {})
    n_classes = saved_cfg.get("num_classes", cfg["num_classes"])
    bilinear  = saved_cfg.get("bilinear",    cfg["bilinear"])

    model = UNet(n_channels=1, n_classes=n_classes, bilinear=bilinear)
    model.load_state_dict(ckpt["model"])
    model = model.to(device, memory_format=torch.channels_last)
    model.eval()
    print(f"Loaded: {ckpt_path}")
    print(f"  epoch {ckpt.get('epoch','?')}  |  best dice {ckpt.get('best_score',0):.4f}")
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


def save_predictions(model, loader, device, cfg, output_dir: Path, num_samples: int = 10):
    out = output_dir
    out.mkdir(parents=True, exist_ok=True)

    all_samples = []
    with torch.no_grad():
        for imgs, masks, _ in tqdm(loader, desc="Collecting", leave=False):
            imgs  = imgs.to(device, dtype=torch.float32, memory_format=torch.channels_last)
            masks = masks.to(device, dtype=torch.float32)
            with torch.autocast(device.type, enabled=cfg["amp"]):
                logits = model(imgs)
            preds = torch.sigmoid(logits.float())
            for i in range(imgs.size(0)):
                all_samples.append((imgs[i].float().cpu(), masks[i].float().cpu(), preds[i].float().cpu()))

    selected = random.sample(all_samples, min(num_samples, len(all_samples)))
    print(f"Saving {len(selected)} samples to {out}")

    for idx, (img, mask, pred) in enumerate(tqdm(selected, desc="Saving", leave=False)):
        img_np  = (img.squeeze().numpy() * 0.5 + 0.5 ) * 255
        img_np  = img_np.clip(0, 255).astype(np.uint8)
        mask_np = (mask.squeeze().numpy() * 255).astype(np.uint8)
        pred_np = (pred.squeeze().numpy() * 255).astype(np.uint8)

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        for ax, arr, title in zip(axes, [img_np, mask_np, pred_np],
                                   ["Image", "Ground Truth", "Prediction"]):
            ax.imshow(arr, cmap="gray"); ax.set_title(title); ax.axis("off")
        plt.tight_layout()
        plt.savefig(out / f"sample_{idx:02d}_combined.png", dpi=150, bbox_inches="tight")
        plt.close()
        Image.fromarray(img_np).save(out / f"sample_{idx:02d}_image.png")
        Image.fromarray(mask_np).save(out / f"sample_{idx:02d}_mask.png")
        Image.fromarray(pred_np).save(out / f"sample_{idx:02d}_pred.png")

    print(f"Done → {out}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True, help="checkpoint .pth path")
    parser.add_argument("--batch",   type=int, default=CONFIG["batch_size_per_gpu"])
    parser.add_argument("--samples", type=int, default=10)
    parser.add_argument("--no-save", action="store_true")
    parser.add_argument("--dataset", nargs="+",
                        default=["phantom_1", "phantom_2"],
                        choices=["phantom_1", "dataset1", "phantom_2"],
                        help="Test dataset(s)")
    args = parser.parse_args()

    device = get_device()
    model, cfg = load_model(args.ckpt, CONFIG, device)

    for ds_name in args.dataset:
        print(f"\n{'='*60}")
        print(f"Testing on: {ds_name}")
        print(f"{'='*60}")
        test_ds     = ReadDataset(dataset_type="test", batch_size=args.batch,
                                  dataset_name=ds_name)
        test_loader = test_ds.get_dataloader(shuffle=False, num_workers=0)
        print(f"Test samples: {len(test_ds)}")

        measure_fps(model, test_loader, device, cfg)

        test_loss, test_dice = validate(model, test_loader, cfg, device)
        print(f"\n[TEST RESULT — {ds_name}]  loss={test_loss:.4f}  dice={test_dice:.4f}")

        if not args.no_save:
            run_name = CONFIG.get("run_name", "run")
            out_dir = Path("outputs") / f"sample_predictions_{run_name}_{ds_name}"
            save_predictions(model, test_loader, device, cfg,
                             out_dir, num_samples=args.samples)


if __name__ == "__main__":
    main()
