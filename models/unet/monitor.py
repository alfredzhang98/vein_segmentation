"""Read-only Linux training monitor; standard library only, no CUDA context."""
import argparse
from datetime import datetime
from pathlib import Path
import re
import subprocess
import time


def process_state(pid):
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        return fields[0], fields[19]  # state, starttime (protect against PID reuse)
    except FileNotFoundError:
        return None, None


def emit(message):
    print(f"{datetime.now().astimezone().isoformat(timespec='seconds')} {message}", flush=True)


def query_gpu(pid):
    def query(flag, columns):
        return subprocess.check_output(
            ["nvidia-smi", f"--query-{flag}={columns}", "--format=csv,noheader,nounits"],
            text=True, timeout=10,
        ).strip().splitlines()

    try:
        apps = [line.split(", ") for line in query("compute-apps", "pid,gpu_uuid,used_memory")]
        owned = {uuid: memory for proc, uuid, memory in apps if proc == str(pid)}
        rows = query("gpu", "uuid,index,utilization.gpu,memory.used,temperature.gpu")
        result = []
        for row in rows:
            uuid, index, utilization, total_memory, temperature = row.split(", ")
            if uuid in owned:
                result.append(f"GPU {index}: utilization={utilization}% (whole GPU), "
                              f"process_memory={owned[uuid]} MiB, GPU_memory={total_memory} MiB, "
                              f"temperature={temperature} C")
        return "; ".join(result) or "No GPU allocation found for this PID"
    except (OSError, subprocess.SubprocessError, ValueError) as exc:
        return f"GPU query unavailable: {exc}"


def read_log(path):
    if path is None or not path.is_file():
        return None
    with path.open("rb") as stream:
        stream.seek(max(0, path.stat().st_size - 262144))
        return stream.read().decode("utf-8", errors="replace")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pid", type=int, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--train-log", type=Path)
    parser.add_argument("--interval", type=float, default=60)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    if args.interval <= 0:
        parser.error("--interval must be positive")
    _, identity = process_state(args.pid)
    if identity is None:
        parser.error("Training PID does not exist")
    emit(f"Monitoring PID={args.pid}; interval={args.interval}s; checkpoints={args.checkpoint_dir}")
    emit("Checkpoint epoch is the best saved epoch, NOT necessarily the current epoch. "
         "Process exit alone does not prove successful completion.")
    previous_checkpoint = None
    previous_summary = None
    while True:
        state, current_identity = process_state(args.pid)
        finished = identity != current_identity or state in ("Z", "X")
        emit(f"process={'EXITED' if finished else state}; " + query_gpu(args.pid))
        if state in ("T", "t"):
            emit("WARNING: training is paused")
        files = list(args.checkpoint_dir.glob("*.pth"))
        try:
            newest = max(files, key=lambda p: p.stat().st_mtime) if files else None
            if newest is not None:
                signature = (newest.name, newest.stat().st_size)
                if signature != previous_checkpoint:
                    emit(f"CHECKPOINT observed: {newest.name}; bytes={signature[1]}")
                    previous_checkpoint = signature
            else:
                emit("No checkpoint observed yet")
        except FileNotFoundError:
            emit("Checkpoint rotated during check; will retry next poll")
        content = read_log(args.train_log)
        if content is None:
            if previous_summary != "missing":
                emit("WARNING: training log unavailable; batch progress/loss/completion reason "
                     "cannot be monitored. See the original training terminal.")
                previous_summary = "missing"
        else:
            lines = re.split(r"[\r\n]+", content)
            summaries = [line for line in lines if any(marker in line for marker in
                         ("PRIMARY(", "Training complete", "Early stop", "Error:",
                          "Traceback", "out of memory"))]
            summary = "\n".join(summaries[-5:])
            if summary and summary != previous_summary:
                emit("TRAIN LOG: " + summary)
                previous_summary = summary
            if time.time() - args.train_log.stat().st_mtime > 600:
                emit("WARNING: training log unchanged for >10 min; check original terminal")
        if finished:
            emit("Monitoring ended: training process ended. Verify the terminal/log for the exit reason.")
            return
        if args.once:
            return
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
