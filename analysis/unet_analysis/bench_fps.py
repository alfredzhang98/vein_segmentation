"""
Deployment latency / throughput benchmark, for the paper's real-time claim.

WHY NOT test.py's measure_fps()
-------------------------------
That one times the model on whatever batch the test loader hands it (16), and reports
images/second. A robot does not get batches: frames arrive one at a time and each must
be answered before the next arrives. Batched throughput overstates the achievable rate,
sometimes several-fold, because it amortises kernel-launch overhead across 16 frames
that in reality never coexist. Batch 1 is the number that belongs in the paper.

And the network forward is not the whole cost. What the guidance loop actually waits for
is:

    fit -> normalise -> forward -> argmax -> unfit -> morphology/clean -> moments

The post-processing runs on the CPU (OpenCV), and on a 576x544 mask it is not free. So
this reports BOTH the forward pass alone and the full pipeline through
UNetInferencer.predict()/measure() — the same code the rig runs.

Latency percentiles, not just the mean: a real-time loop is bounded by its slow frames,
not its average one.

Usage:
    python analysis/unet_analysis/bench_fps.py --ckpt models/unet/checkpoints/unet_v10.pth
    python bench_fps.py --ckpt <x>.pth --cpu          # CPU-only fallback
    python bench_fps.py --ckpt <x>.pth --n 300 --warmup 50
"""
import argparse
import platform
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
# Before torch.cuda — see gpu_utils. A benchmark that lands on a GPU another job is
# already saturating measures that job's interference, not this model.
from gpu_utils import pick_idle_gpu
if "--cpu" not in sys.argv:
    pick_idle_gpu()

import numpy as np
import torch

from models.unet.infer import UNetInferencer


def gpu_name():
    if not torch.cuda.is_available():
        return None
    try:
        return torch.cuda.get_device_name(0)
    except Exception:
        return "unknown CUDA device"


def cpu_name():
    try:
        out = subprocess.run(["lscpu"], capture_output=True, text=True).stdout
        for line in out.split("\n"):
            if "Model name" in line:
                return line.split(":", 1)[1].strip()
    except Exception:
        pass
    return platform.processor() or "unknown CPU"


def stats(times_ms):
    t = np.array(times_ms)
    return dict(mean=t.mean(), p50=np.percentile(t, 50), p95=np.percentile(t, 95),
                p99=np.percentile(t, 99), worst=t.max(), fps_mean=1000.0 / t.mean(),
                fps_p95=1000.0 / np.percentile(t, 95))


def show(label, s):
    print(f"  {label:34s} {s['p50']:7.2f} {s['mean']:8.2f} {s['p95']:7.2f} "
          f"{s['p99']:7.2f} {s['worst']:7.2f} │ {s['fps_mean']:6.1f} {s['fps_p95']:7.1f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--n", type=int, default=200, help="计时的帧数")
    ap.add_argument("--warmup", type=int, default=30,
                    help="预热帧数（CUDA 首次调用要编译/分配，不预热会把第一帧的几十 ms 算进去）")
    ap.add_argument("--cpu", action="store_true", help="强制用 CPU（有些机器人没有 GPU）")
    ap.add_argument("--frame", default=None,
                    help="用一张真实图片测（默认用随机噪声，速度上等价）")
    ap.add_argument("--batch-sweep", action="store_true",
                    help="附带跑批量吞吐做参照（注意：这不是实时可达速率）")
    args = ap.parse_args()

    device = "cpu" if args.cpu else ("cuda" if torch.cuda.is_available() else "cpu")
    inf = UNetInferencer(args.ckpt, device=device)

    n_par = sum(p.numel() for p in inf.model.parameters())
    H, W = inf.target_size

    if args.frame:
        import cv2
        frame = cv2.imread(args.frame, cv2.IMREAD_GRAYSCALE)
        if frame is None:
            raise SystemExit(f"读不了: {args.frame}")
    else:
        # Speed does not depend on content — same tensor shape, same kernels. Only the
        # post-processing varies slightly with how many components it finds, which is
        # why --frame exists for a sanity check against a real one.
        frame = (np.random.rand(H, W) * 255).astype(np.uint8)

    print("=" * 88)
    print("部署延迟 benchmark  (batch=1，真机就是一帧一帧来的)")
    print("=" * 88)
    print(f"  checkpoint : {Path(args.ckpt).name}")
    print(f"  模型       : U-Net, {inf.n_classes} 类, {n_par/1e6:.2f}M 参数")
    print(f"  输入       : {H}x{W} 灰度")
    print(f"  设备       : {device.upper()}  |  {gpu_name() if device=='cuda' else cpu_name()}")
    print(f"  AMP        : {bool(inf.cfg.get('amp', True)) and device=='cuda'}")
    print(f"  PyTorch    : {torch.__version__}   CUDA: {torch.version.cuda}")
    print(f"  预热 {args.warmup} 帧, 计时 {args.n} 帧")

    def sync():
        if device == "cuda":
            torch.cuda.synchronize()

    # ---- 1. 网络前向（含 fit/normalise，不含后处理）----
    x = (frame.astype(np.float32) / 255.0 - 0.5) / 0.5
    t = torch.from_numpy(x)[None, None].to(device, dtype=torch.float32,
                                           memory_format=torch.channels_last)
    for _ in range(args.warmup):
        with torch.inference_mode(), torch.autocast(device, enabled=(device == "cuda")):
            inf.model(t)
    sync()
    fwd = []
    for _ in range(args.n):
        t0 = time.perf_counter()
        with torch.inference_mode(), torch.autocast(device, enabled=(device == "cuda")):
            inf.model(t)
        sync()
        fwd.append((time.perf_counter() - t0) * 1000)

    # ---- 2. 完整推理: fit -> forward -> argmax -> unfit ----
    for _ in range(args.warmup):
        inf.predict(frame)
    sync()
    pred = []
    for _ in range(args.n):
        t0 = time.perf_counter()
        inf.predict(frame)
        sync()
        pred.append((time.perf_counter() - t0) * 1000)

    # ---- 3. 端到端: 上面 + clean + 几何测量（机器人真正等的东西）----
    m = inf.predict(frame)
    for _ in range(max(5, args.warmup // 3)):
        inf.measure(m, mm_per_px=0.0832)
    e2e = []
    for _ in range(args.n):
        t0 = time.perf_counter()
        mask = inf.predict(frame)
        inf.measure(mask, mm_per_px=0.0832)
        sync()
        e2e.append((time.perf_counter() - t0) * 1000)

    print("\n" + "-" * 88)
    print(f"  {'阶段':34s} {'p50':>7s} {'mean':>8s} {'p95':>7s} {'p99':>7s} {'最差':>7s} │"
          f" {'FPS':>6s} {'FPS@p95':>7s}")
    print(f"  {'':34s} {'(ms)':>7s} {'(ms)':>8s} {'(ms)':>7s} {'(ms)':>7s} {'(ms)':>7s} │")
    print("-" * 88)
    show("1. 网络前向", stats(fwd))
    show("2. + fit/unfit (predict)", stats(pred))
    show("3. + 后处理 + 几何 (端到端)", stats(e2e))
    print("-" * 88)

    s = stats(e2e)
    print(f"\n  → 论文里报这个: 端到端 {s['p50']:.1f} ms/frame (p50), "
          f"{s['fps_mean']:.0f} FPS, batch=1, {gpu_name() if device=='cuda' else 'CPU'}")
    print(f"  → 实时性论证用 p95 更稳: {s['p95']:.1f} ms → {s['fps_p95']:.0f} FPS 保底")
    pp = stats(e2e)['p50'] - stats(pred)['p50']
    print(f"  → 后处理+几何占 {pp:.1f} ms ({pp/s['p50']*100:.0f}%)，"
          f"跑在 CPU 上（OpenCV），不随 GPU 变快")

    if args.batch_sweep:
        print("\n  批量吞吐（仅作参照 —— 真机拿不到批，别写进实时性论证）")
        print(f"  {'batch':>6s} {'ms/batch':>10s} {'img/s':>9s}")
        for bs in (1, 4, 8, 16, 32):
            tb = torch.from_numpy(x)[None, None].repeat(bs, 1, 1, 1).to(
                device, dtype=torch.float32, memory_format=torch.channels_last)
            for _ in range(10):
                with torch.inference_mode(), torch.autocast(device, enabled=(device == "cuda")):
                    inf.model(tb)
            sync()
            t0 = time.perf_counter()
            for _ in range(20):
                with torch.inference_mode(), torch.autocast(device, enabled=(device == "cuda")):
                    inf.model(tb)
            sync()
            dt = (time.perf_counter() - t0) / 20
            print(f"  {bs:6d} {dt*1000:10.2f} {bs/dt:9.1f}")


if __name__ == "__main__":
    main()
