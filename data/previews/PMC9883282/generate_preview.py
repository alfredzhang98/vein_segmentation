"""Generate local PMC9883282 previews from raw MAT files without changing labels.

Paths default to this repository, independent of the working directory.
Use --check-only to validate the downloaded source before encoding.
"""

import argparse
import html
import json
import math
from pathlib import Path
import tempfile


HERE = Path(__file__).resolve().parent
DATA_DIR = HERE.parents[1] / "datasets" / "PMC9883282"
SUFFIX = "_prepped_force_ultrasound.mat"
SEQUENCES = {
    "subject20_16degree": "受试者20：抬高16度、逐渐加压",
    "subject20_Valsalva": "受试者20：Valsalva动作、逐渐加压",
    "subject20_constant": "受试者20：仰卧、恒定接触力",
    "subject20_supine": "受试者20：仰卧、逐渐加压",
    "subject21_supine": "受试者21：仰卧、逐渐加压",
    "subject2_supine": "受试者2：仰卧、逐渐加压",
}
CODECS = {"mp4": "mp4v", "webm": "VP80"}


def inspect_sources(data_dir, sequences):
    """Fail before creating output if the selected raw sources are absent/invalid."""
    paths = [data_dir / (stem + SUFFIX) for stem in sequences]
    missing = [p.name for p in paths if not p.is_file()]
    if missing:
        raise ValueError(
            f"原始数据尚未下载完整，或 --data-dir 指向了错误位置：{data_dir}\n"
            + "缺少：\n  " + "\n  ".join(missing)
            + f"\n请按 {HERE / 'README_CN.md'} 下载并解压 MAT 文件；脚本不会自动下载。"
        )
    import h5py
    import numpy as np

    rows = []
    for path, stem in zip(paths, sequences):
        with h5py.File(path, "r") as source:
            if "ultrasound_images" not in source:
                raise ValueError(f"缺少 ultrasound_images：{path}")
            frames = source["ultrasound_images"]
            if (not isinstance(frames, h5py.Dataset) or frames.ndim != 3
                    or frames.dtype != np.dtype("uint8") or min(frames.shape) < 1):
                raise ValueError(f"需要非空 uint8 [N, width, height] 图像数组：{path}")
            count, width, height = frames.shape
            if width % 2 or height % 2:
                raise ValueError(f"视频编码需要偶数宽高；不自动裁剪原图：{path}")
            # Read both endpoints to catch unreadable/incomplete HDF5 data early.
            frames[0]
            frames[-1]
            rows.append(dict(source_mat=path.name, name=SEQUENCES[stem], stem=stem,
                             frames=count, width=width, height=height,
                             original_frame_timestamps_available=False))
    return rows


def encode(source_path, row, output_dir, formats, fps):
    import cv2
    import h5py
    import numpy as np

    writers = []
    try:
        for ext in formats:
            path = output_dir / f"{row['stem']}.{ext}"
            writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*CODECS[ext]),
                                     fps, (row["width"], row["height"]))
            writers.append(writer)
            if not writer.isOpened():
                raise ValueError(
                    f"当前 OpenCV 无法编码 {ext} ({CODECS[ext]})。"
                    "可用 --format 选择其他格式，或在已有编码支持的本地环境运行；"
                    "不需要安装服务器系统软件。"
                )
        with h5py.File(source_path, "r") as source:
            frames = source["ultrasound_images"]
            for idx in range(row["frames"]):
                # MATLAB/HDF5 axis order: transpose each frame, keep full screenshot.
                gray = np.ascontiguousarray(frames[idx].T)
                frame = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
                label = f"Frame {idx:06d} / {row['frames'] - 1:06d} | preview {fps:g} fps"
                cv2.putText(frame, label, (10, row["height"] - 12),
                            cv2.FONT_HERSHEY_SIMPLEX, .45, (255, 255, 255), 1, cv2.LINE_AA)
                for writer in writers:
                    writer.write(frame)
    finally:
        for writer in writers:
            writer.release()
    for ext in formats:
        path = output_dir / f"{row['stem']}.{ext}"
        capture = cv2.VideoCapture(str(path))
        try:
            ok, frame = capture.read()
            count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            if (not ok or count != row["frames"]
                    or frame.shape[:2] != (row["height"], row["width"])):
                raise ValueError(f"生成的视频未通过帧数／尺寸／解码检查：{path}")
        finally:
            capture.release()


def write_index(output_dir, rows, formats, fps):
    cards = []
    for row in rows:
        # MPEG-4 Part 2 may require a desktop player; prefer VP8 for browsers.
        video = (f'<video controls preload="none" src="{row["stem"]}.webm"></video>'
                 if "webm" in formats else "<p>MP4 请下载后使用本地播放器打开。</p>")
        links = " · ".join(f'<a href="{row["stem"]}.{ext}">{ext.upper()} 下载</a>'
                           for ext in formats)
        cards.append(f'<section><h2>{html.escape(row["name"])}</h2>{video}'
                     f'<p>{row["frames"]} 帧 · 预览 {row["preview_seconds"]:.2f} 秒'
                     f' · {links}</p></section>')
    page = f'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>PMC9883282 视频预览</title><style>
body{{background:#10151c;color:#eee;font:16px system-ui;max-width:1000px;margin:32px auto;padding:0 20px;line-height:1.8}}
video{{width:100%;max-width:800px}}a{{color:#72b8ff}}section{{margin:40px 0}}h2{{font-size:20px}}
</style></head><body><h1>PMC9883282：左颈内静脉超声</h1>
<p>完整原图 · 无 mask · {fps:g} fps 仅为预览速度，不代表原始采集速度。
帧号从 0 开始，与标注文件名对应。没有逐帧时间戳，不配准或叠加 force。</p>
{''.join(cards)}</body></html>'''
    (output_dir / "index.html").write_text(page, encoding="utf-8")
    (output_dir / "manifest.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR,
                        help="Directory directly containing the six original MAT files")
    parser.add_argument("--output-dir", type=Path, default=HERE)
    parser.add_argument("--sequence", nargs="+", choices=list(SEQUENCES),
                        help="Selected recordings; default: all six")
    parser.add_argument("--format", choices=["both", *CODECS], default="both")
    parser.add_argument("--fps", type=float, default=20, help="Playback FPS, NOT acquisition FPS")
    parser.add_argument("--check-only", action="store_true", help="Validate sources without writing previews")
    parser.add_argument("--overwrite", action="store_true", help="Replace selected generated outputs")
    args = parser.parse_args()
    if not math.isfinite(args.fps) or args.fps <= 0:
        parser.error("--fps must be finite and positive")
    try:
        sequences = list(dict.fromkeys(args.sequence or SEQUENCES))
        rows = inspect_sources(args.data_dir, sequences)
        for row in rows:
            print(f"{row['source_mat']}: {row['frames']} 帧，{row['width']}×{row['height']}", flush=True)
        if args.check_only:
            print(f"原始数据检查通过：{len(rows)} 段，共 {sum(r['frames'] for r in rows)} 帧；未生成文件。")
            return
        formats = list(CODECS) if args.format == "both" else [args.format]
        output = args.output_dir.resolve()
        if output == args.data_dir.resolve() or args.data_dir.resolve() in output.parents:
            raise ValueError("--output-dir 必须位于原始数据目录之外，避免混入数据或标注。")
        destinations = [output / f"{r['stem']}.{ext}" for r in rows for ext in formats]
        destinations += [output / "index.html", output / "manifest.json"]
        existing = [p.name for p in destinations if p.exists()]
        if existing and not args.overwrite:
            raise ValueError("已有预览产物：" + ", ".join(existing)
                             + "。如需重建，使用 --overwrite 或另选 --output-dir。")
        for row in rows:
            row.update(preview_fps=args.fps, preview_seconds=row["frames"] / args.fps,
                       formats=formats)
        output.mkdir(parents=True, exist_ok=True)
        # Publish only after every selected recording has encoded successfully.
        with tempfile.TemporaryDirectory(prefix=".preview-", dir=output) as temp:
            staging = Path(temp)
            for row in rows:
                encode(args.data_dir / row["source_mat"], row, staging, formats, args.fps)
                print(f"已编码：{row['stem']}", flush=True)
            write_index(staging, rows, formats, args.fps)
            for destination in destinations:
                (staging / destination.name).replace(destination)
        print(f"完成：{output / 'index.html'}；人工标注与原始数据未修改。")
    except (ImportError, OSError, ValueError) as exc:
        parser.exit(1, f"错误：{exc}\n请使用已有 ml_env 环境，并参见同目录 README_CN.md。\n")


if __name__ == "__main__":
    main()
