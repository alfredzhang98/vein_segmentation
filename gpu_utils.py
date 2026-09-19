"""
Pick one idle GPU, without needing CUDA_VISIBLE_DEVICES=N on every command line.

⚠ ORDER IS THE WHOLE POINT. CUDA_VISIBLE_DEVICES is read by the CUDA runtime exactly
once, at the moment torch first touches the driver. Setting it afterwards is silently
ignored — no error, no warning, the process just runs on whatever card it already
picked. train.py had precisely this bug: it called torch.cuda.is_available() before
setting the variable, so for the whole life of this project every training run reported
"selecting GPU 2" and then ran on GPU 0, sharing it with another user's job at 100%
utilisation (measured: 201 img/s -> 74 img/s).

So this module queries nvidia-smi through subprocess — never torch — and must be called
BEFORE the first torch.cuda.* call in the process. Importing torch is fine; touching
torch.cuda is not.

    from gpu_utils import pick_idle_gpu
    pick_idle_gpu()            # <- before anything reads torch.cuda
    import torch               # (or after; only .cuda access matters)
"""
import os
import subprocess
import sys

# Scripts import each other (the figure script pulls clean_binary out of test.py), so
# pick_idle_gpu() gets called more than once per process. The second call is harmless —
# it sees the variable already set and honours it — but it should not narrate that again.
_ALREADY_PICKED = False


def query_gpus():
    """[(index, free_MiB, util_pct)] sorted by free memory, most free first. No torch."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.free,utilization.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=True, timeout=15).stdout.strip()
    except Exception:
        return []
    gpus = []
    for line in out.split("\n"):
        if not line.strip():
            continue
        try:
            i, free, util = (p.strip() for p in line.split(","))
            gpus.append((int(i), int(free), int(util)))
        except ValueError:
            continue
    gpus.sort(key=lambda g: (-g[1], g[2]))
    return gpus


def pick_idle_gpu(min_free_mib=6000, verbose=True):
    """
    Claim exactly ONE GPU — the one with the most free memory — by setting
    CUDA_VISIBLE_DEVICES to it. Returns the physical index, or None for CPU.

    One card, never several: everything that calls this (figures, benchmarks, testing)
    is a single-stream job that cannot use a second GPU, and this box is shared — taking
    cards we will not use is just denying them to someone else.

    Respects a CUDA_VISIBLE_DEVICES the caller already set, so an explicit choice on the
    command line still wins.
    """
    global _ALREADY_PICKED
    pre = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if pre != "":
        if verbose and not _ALREADY_PICKED:
            print(f"[gpu] CUDA_VISIBLE_DEVICES 已由外部设为 '{pre}'，沿用", file=sys.stderr)
        _ALREADY_PICKED = True
        return None if pre.strip() == "" else int(pre.split(",")[0])

    gpus = query_gpus()
    if not gpus:
        if verbose:
            print("[gpu] nvidia-smi 无结果 → CPU", file=sys.stderr)
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        return None

    idx, free, util = gpus[0]
    if free < min_free_mib:
        if verbose:
            print(f"[gpu] 最空闲的 GPU {idx} 也只有 {free} MiB "
                  f"(< {min_free_mib}) → 退回 CPU", file=sys.stderr)
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        return None

    os.environ["CUDA_VISIBLE_DEVICES"] = str(idx)
    if verbose and not _ALREADY_PICKED:
        busy = [f"{i}({u}%)" for i, f, u in gpus[1:] if u > 50]
        print(f"[gpu] 选用 GPU {idx}  ({free/1024:.0f} GB 空闲, {util}% 占用)"
              + (f"  |  忙碌的卡: {', '.join(busy)}" if busy else ""), file=sys.stderr)
    _ALREADY_PICKED = True
    return idx
