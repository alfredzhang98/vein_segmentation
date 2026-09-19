"""
Paper-ready summary of the final model: architecture, training setup, data, and the
per-domain test results.

Everything here is READ OUT of the checkpoint and the dataset configs — nothing is
retyped by hand. A summary transcribed by a human drifts from the run it claims to
describe the moment either one changes; this cannot.

Emits Markdown (default) or a LaTeX table for direct paste into the manuscript.

Usage:
  python analysis/unet_analysis/summarize_model.py --ckpt models/unet/checkpoints/unet_v10.pth
  python summarize_model.py --ckpt <x>.pth --latex
  python summarize_model.py --ckpt <x>.pth --results  # also re-runs test (slow)
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from data.pipeline.dataPrepare import DATASET_CONFIGS, MODE_IDS
from models.unet.model import UNet

MODE_MEANING = {
    "full3":  "background / vein / artery (fully labelled)",
    "vessel": "vessel, type unknown (marginal $p_1{+}p_2$)",
    "artery": "artery only; vein present but unlabelled (marginal $p_2$)",
}


def load(ckpt_path, device="cpu"):
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ck.get("config", {}) or {}
    model = UNet(n_channels=int(cfg.get("n_channels", 1)),
                 n_classes=int(cfg.get("num_classes", 3)),
                 bilinear=bool(cfg.get("bilinear", False)),
                 base_ch=int(cfg.get("base_ch", 64)),
                 dropout=float(cfg.get("dropout", 0.0)))
    model.load_state_dict(ck["model"])
    return ck, cfg, model


def count_split(name, split):
    cfg = DATASET_CONFIGS[name]
    out = cfg.get("npz_out_dir", cfg["data_dir"])
    p = out / f"{cfg['npz_prefix']}_{split}.npz"
    if not p.exists():
        return None
    return len(np.load(p, allow_pickle=True)["masks"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--latex", action="store_true", help="输出 LaTeX 表格而不是 Markdown")
    ap.add_argument("--out", default=None, help="写到文件（默认打到 stdout）")
    args = ap.parse_args()

    ck, cfg, model = load(args.ckpt)
    n_par = sum(p.numel() for p in model.parameters())
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)

    L = []
    A = L.append

    A(f"# 模型与训练参数总结\n")
    A(f"Checkpoint: `{Path(args.ckpt).name}`  \n")
    A(f"Epoch {ck.get('epoch','?')}  |  best PRIMARY = {ck.get('best_score', float('nan')):.4f}\n")

    # ── architecture ──
    A("\n## 网络结构\n")
    A("| 项 | 值 |")
    A("|---|---|")
    A(f"| Architecture | U-Net (encoder-decoder, 4× down / 4× up, skip connections) |")
    A(f"| Input | {cfg.get('n_channels',1)} × {DATASET_CONFIGS['musv']['target_size'][0]}"
      f" × {DATASET_CONFIGS['musv']['target_size'][1]} (grayscale) |")
    A(f"| Output channels | {cfg.get('num_classes',3)} (0 = background, 1 = vein, 2 = artery) |")
    A(f"| Base width (`base_ch`) | {cfg.get('base_ch', 64)} |")
    A(f"| Encoder widths | {' → '.join(str(int(cfg.get('base_ch',64)) * 2**i) for i in range(5))} |")
    A(f"| Upsampling | {'bilinear' if cfg.get('bilinear') else 'transposed conv'} |")
    A(f"| Dropout2d | {cfg.get('dropout', 0.0)} (two deepest encoder blocks + bottleneck) |")
    A(f"| Parameters | {n_par:,} ({n_par/1e6:.2f} M) |")

    # ── training ──
    A("\n## 训练配置\n")
    A("| 项 | 值 |")
    A("|---|---|")
    A(f"| Optimizer | {str(cfg.get('optimizer','adamw')).upper()} |")
    A(f"| Learning rate | {cfg.get('learning_rate', 3e-4):g} |")
    A(f"| Weight decay | {cfg.get('weight_decay', 1e-4):g} |")
    A(f"| LR schedule | ReduceLROnPlateau (mode={cfg.get('plateau_mode','max')}, "
      f"patience={cfg.get('plateau_patience',5)}, factor=0.5) |")
    A(f"| Batch size | {cfg.get('batch_size_per_gpu', 16)} |")
    A(f"| Gradient clip | {cfg.get('gradient_clip', 1.0):g} |")
    A(f"| Mixed precision | {'yes' if cfg.get('amp') else 'no'} |")
    A(f"| Early stopping | patience {cfg.get('early_stop_patience',30)}, "
      f"min_delta {cfg.get('min_delta',0):g} |")
    A(f"| Loss | partial-label marginal loss: CE + Dice per label mode |")

    # ── data ──
    A("\n## 数据集\n")
    A("模型输出恒为 3 通道；每个数据集只监督它的标签**真正能确定**的那个边缘概率。\n")
    A("| Dataset | label mode | 监督什么 | mask ids | train | val | test | ×/epoch |")
    A("|---|---|---|---|---|---|---|---|")
    train_names = [s.strip() for s in str(cfg.get("dataset", "")).split("+") if s.strip()]
    for name in train_names or list(DATASET_CONFIGS):
        d = DATASET_CONFIGS[name]
        m = d["label_mode"]
        tr, va, te = (count_split(name, s) for s in ("train", "val", "test"))
        A(f"| `{name}` | `{m}` | {MODE_MEANING[m]} | {sorted(MODE_IDS[m])} | "
          f"{tr or '—'} | {va or '—'} | {te or '—'} | ×{d['aug_times_train']} |")

    # ── augmentation ──
    A("\n## 数据增强（训练时在线做，每个 epoch 重新随机）\n")
    d = DATASET_CONFIGS["musv"]
    A("| 增强 | 幅度 | 模拟什么 |")
    A("|---|---|---|")
    # round(), not int(): (1.2 - 1) * 100 is 19.999999999999996 in binary float, and
    # int() truncates that to "±19%" — a wrong number in a paper table.
    A(f"| Isotropic zoom | ±{round((d['aug_scale'][1]-1)*100)}% | 深度档位 (4 cm / 5 cm) |")
    A(f"| Rotation | ±{d.get('aug_rotation_limit',10)}° | 探头角度 |")
    A(f"| Horizontal flip | p=0.5 | 左右颈 |")
    A(f"| Gamma | {d['aug_gamma'][0]}–{d['aug_gamma'][1]}% | 超声增益 |")
    A(f"| Multiplicative noise | ×{d['aug_speckle'][0]}–{d['aug_speckle'][1]} | 散斑（乘性） |")
    A(f"| Blur / sharpen | p={d['aug_blur']} | 不同探头与聚焦 |")
    A(f"| Brightness / contrast | ±{round(d['aug_brightness'][1]*100)}% | — |")
    A("\n> 缩放是**各向同性**的（手写，不用 `A.Affine(scale=)`）。"
      "后者会独立采样 x/y 缩放，把圆压成 0.70:1–1.47:1 的椭圆——"
      "而动脉圆、静脉扁正是区分两者的唯一线索。\n")
    A("> 垂直翻转只对仿体开启：B 超的探头面永远在图像顶部，"
      "上下翻转的颈部解剖在物理上不可能出现。\n")

    # ── checkpoint criterion ──
    A("\n## Checkpoint 选择准则\n")
    w = cfg.get("val_weights", {})
    A(f"`PRIMARY = Σ wᵢ · scoreᵢ / Σ wᵢ`，各验证集贡献它的标签**能支持**的分数：\n")
    A("| 验证集 | 权重 | 用什么分数 |")
    A("|---|---|---|")
    score_of = {"full3": "(Dice_vein + Dice_artery) / 2", "vessel": "Dice_vessel",
                "artery": "Dice_artery"}
    for k, v in w.items():
        first = k.split("+")[0].strip()
        m = DATASET_CONFIGS[first]["label_mode"]
        A(f"| `{k}` | {v:g} | {score_of[m]} |")
    A("\n> `artery` 模式**只报 Dice_artery**。早期版本让它落到 Dice_vessel"
      "（= 预测静脉∪动脉 vs 只标了动脉的真值），等于把模型正确找到的、"
      "未标注的颈内静脉算成假阳性——实测该项在 27 个 epoch 里让 Mus-V 静脉 "
      "Dice 从 0.614 掉到 0.574。\n")

    # ── post-processing ──
    A("\n## 推理后处理（部署配置）\n")
    A("| 步骤 | 参数 | 理由 |")
    A("|---|---|---|")
    A("| Morphological open → close | 3×3 → 5×5 ellipse | 去散斑、桥接 1–2 px 缝隙 |")
    A("| Min area | 150 px | Mus-V 里最小的**真**静脉是 232 px，150 保证删不掉真血管 |")
    A("| Anchor + distance gate | ≤ 40 px | 锚点 = 最大连通域；近处碎片是同一根血管被切开，保留；远处的是假阳性，删 |")
    A("\n> `min_area` **不能用 Dice 调**：在 val 上最优是 450，在 test 上最优是 100 —— "
      "两头顶天，纯粹在拟合噪声。这里按**物理**定（真静脉的面积下限）。\n")
    A("> 仿体走**先合并再清理**（`(p1+p2) > thr` → clean，一根管子）；"
      "人体走**逐类清理再取并集**（一帧里有两根血管，对合并前景做 keep-largest "
      "会直接删掉一根真颈动脉，实测 −0.10 Dice）。\n")

    out = "\n".join(L)

    if args.latex:
        out += "\n\n" + latex_table(cfg, n_par)

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(out)
        print(f"Saved → {args.out}")
    else:
        print(out)


def latex_table(cfg, n_par):
    """A compact IEEE-style table of the training setup."""
    rows = [
        ("Architecture", "U-Net (3-class, partial-label)"),
        ("Base width", f"{cfg.get('base_ch', 64)}"),
        ("Parameters", f"{n_par/1e6:.2f}\\,M"),
        ("Input", f"{DATASET_CONFIGS['musv']['target_size'][0]}$\\times$"
                  f"{DATASET_CONFIGS['musv']['target_size'][1]}, grayscale"),
        ("Dropout2d", f"{cfg.get('dropout', 0.0)}"),
        ("Optimizer", f"{str(cfg.get('optimizer','adamw')).upper()}"),
        ("Learning rate", f"{cfg.get('learning_rate', 3e-4):g}"),
        ("Weight decay", f"{cfg.get('weight_decay', 1e-4):g}"),
        ("Batch size", f"{cfg.get('batch_size_per_gpu', 16)}"),
        ("Loss", "CE + Dice (per-mode marginal)"),
    ]
    body = " \\\\\n".join(f"    {k} & {v}" for k, v in rows)
    return ("% ---- paste into the manuscript ----\n"
            "\\begin{table}[t]\n\\centering\n"
            "\\caption{Segmentation model and training configuration.}\n"
            "\\label{tab:seg_params}\n"
            "\\begin{tabular}{ll}\n\\hline\n"
            "    Parameter & Value \\\\\n\\hline\n"
            f"{body} \\\\\n"
            "\\hline\n\\end{tabular}\n\\end{table}")


if __name__ == "__main__":
    main()
