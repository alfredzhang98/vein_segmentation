"""Export a small, reviewable PMC9883282 annotation queue; never invent labels/time alignment.

Run from the repository root. Raw MATLAB files remain untouched. Reruns preserve
the queue and human masks; use a different output directory to change selection.
"""
import argparse
import csv
import json
import re
import sys
from pathlib import Path

import cv2
import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
SUBJECT_SPLITS = {20: "train", 2: "val", 21: "test"}


def inspect_file(path):
    with h5py.File(path) as f:
        images = f["ultrasound_images"]
        force, time = np.asarray(f["force_data"]).ravel(), np.asarray(f["time_data"]).ravel()
        if len(force) != len(time) or not np.all(np.diff(time) > 0):
            raise ValueError(f"Invalid force/time vectors: {path}")
        return dict(source_mat=path.name, subject_id=int(re.match(r"subject(\d+)", path.name)[1]),
                    frames=images.shape[0], hdf5_shape=list(images.shape), fields=list(f.keys()),
                    force_samples=len(force), force_min=float(force.min()), force_max=float(force.max()),
                    force_time_start=float(time[0]), force_time_end=float(time[-1]),
                    force_sample_dt=float(np.median(np.diff(time))),
                    image_timestamps_available=False, pose_available=False, labels_available=False)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", type=Path, default=ROOT / "data/datasets/PMC9883282")
    p.add_argument("--output-dir", type=Path)
    p.add_argument("--frames-per-sequence", type=int, default=30)
    p.add_argument("--label-mode", choices=["full3"], default="full3")
    p.add_argument("--crop", type=int, nargs=4, default=[72, 550, 163, 633],
                   metavar=("Y0", "Y1", "X0", "X1"), help="ROI in transposed 600x800 screenshot")
    p.add_argument("--checkpoint", type=Path, help="Optional v10 suggestions for TRAIN ONLY")
    p.add_argument("--device", default="cpu")
    args = p.parse_args()
    if args.frames_per_sequence < 1:
        p.error("--frames-per-sequence must be positive")
    out = (args.output_dir or args.data_dir).absolute()
    mats = sorted(args.data_dir.glob("*.mat"))
    if not mats:
        p.error("No MATLAB files found")
    audit = [inspect_file(x) for x in mats]
    out.mkdir(parents=True, exist_ok=True)
    for name in ["images", "masks"]:
        (out / name).mkdir(exist_ok=True)
    meta = out / "meta_PMC9883282_1.csv"
    selection = dict(frames_per_sequence=args.frames_per_sequence, crop=args.crop,
                     label_mode=args.label_mode, subject_splits=SUBJECT_SPLITS)
    selection_file = out / "annotation_selection.json"
    serialized = json.dumps(selection, sort_keys=True, indent=2)
    if meta.exists() and (not selection_file.exists() or selection_file.read_text() != serialized):
        raise ValueError("Existing annotation selection differs; use a separate --output-dir.")
    old = {}
    if meta.exists():
        with meta.open(newline="") as f:
            old = {r["filename"]: r for r in csv.DictReader(f)}
    predictor = None
    if args.checkpoint:
        from models.unet.infer import UNetInferencer
        predictor = UNetInferencer(str(args.checkpoint), device=args.device, fit_mode="letterbox")
        if predictor.n_classes != 3:
            raise ValueError("Suggestions require a three-class A/V checkpoint")
    rows = []
    y0, y1, x0, x1 = args.crop
    for path, info in zip(mats, audit):
        subject = info["subject_id"]
        split = SUBJECT_SPLITS[subject]
        with h5py.File(path) as f:
            frames = f["ultrasound_images"]
            if not (0 <= y0 < y1 <= frames.shape[2] and 0 <= x0 < x1 <= frames.shape[1]):
                raise ValueError(f"Invalid ROI {args.crop} for {path}")
            selected = np.unique(np.linspace(0, len(frames)-1, min(args.frames_per_sequence, len(frames)), dtype=int))
            for idx in selected:
                filename = f"{path.stem}_{idx:06d}.png"
                image_path = out / "images" / filename
                image = frames[int(idx)].T[y0:y1, x0:x1].copy()
                if not image_path.exists() and not cv2.imwrite(str(image_path), image):
                    raise OSError(image_path)
                row = old.get(filename, dict(filename=filename, relative_path=str(image_path),
                    subject_id=subject, sequence_id=path.stem, frame_index=int(idx), split=split,
                    label_mode=args.label_mode, mask_status="ND", mask_path="", suggestion_path="",
                    suggestion_checkpoint="", source_mat=path.name, crop_y0=y0, crop_x0=x0,
                    image_timestamp="", force_value="", force_alignment="unverified"))
                # Validation/test annotation starts blank to avoid model-assisted evaluation bias.
                suggestion = out / "suggestions" / f"{Path(filename).stem}_mask.png"
                if predictor and split == "train" and row["mask_status"] == "ND" and not suggestion.exists():
                    suggestion.parent.mkdir(exist_ok=True)
                    pred = predictor.predict(image)
                    mask = np.zeros_like(pred)
                    mask[pred == 1] = 255
                    if args.label_mode == "full3":
                        mask[pred == 2] = 128
                    if not cv2.imwrite(str(suggestion), mask):
                        raise OSError(suggestion)
                    row["suggestion_path"] = str(suggestion)
                    row["suggestion_checkpoint"] = str(args.checkpoint.absolute())
                rows.append(row)
        print(f"{path.name}: {info['frames']} frames -> {len(selected)} queued ({split})", flush=True)
    tmp = meta.with_suffix(".csv.tmp")
    with tmp.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(meta)
    selection_file.write_text(serialized)
    (out / "source_audit.json").write_text(json.dumps(audit, indent=2))
    print(f"Queue: {meta}; {len(rows)} frames. Suggestions are NOT ground truth.")


if __name__ == "__main__":
    main()
