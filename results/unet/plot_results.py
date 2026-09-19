"""
Grouped bar chart of the STAGE-1 experiments, for the IEEE RA-L paper.

SCOPE — read this before reusing the figure. The numbers below are hard-coded from
Exp-A .. Exp-E, the first-generation **binary** (vessel / background) model. They are
the domain-transfer study that motivated joint training; they are NOT the released
3-class background/vein/artery model, whose results live in the README table next to
this file. Do not relabel this chart as the current model's.

Usage:  python results/unet/plot_results.py
Output: results/unet/figures/exp_results.{pdf,svg,png}
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path

from figure_style import FigureConfig

# ── Data ──────────────────────────────────────────────────────────────────────
experiments = [
    "Phantom",
    "CCA",
    "Mixed",
    "Full FT",
    "Decoder FT",
]

phantom_dice  = [0.8967, 0.4735, 0.9226, 0.9117, 0.9041]
mendeley_dice = [0.3147, 0.9509, 0.9479, 0.2682, 0.2205]

# ── Style ─────────────────────────────────────────────────────────────────────
FigureConfig(figsize=(12.0, 7.0)).apply()

# Colorblind-safe palette (Nature/Science style)
C_PHANTOM  = "#2166AC"   # blue  — target domain
C_MENDELEY = "#D6604D"   # red   — source domain
ALPHA_BAR  = 0.88

# ── Layout ────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots()

n      = len(experiments)
x      = np.arange(n)
width  = 0.32
gap    = 0.04

bars_ph = ax.bar(x - width/2 - gap/2, phantom_dice,  width,
                 color=C_PHANTOM,  alpha=ALPHA_BAR, zorder=3,
                 edgecolor="white", linewidth=0.8)
bars_mn = ax.bar(x + width/2 + gap/2, mendeley_dice, width,
                 color=C_MENDELEY, alpha=ALPHA_BAR, zorder=3,
                 edgecolor="white", linewidth=0.8)

# ── Highlight best experiment ─────────────────────────────────────────────────
best_idx = 2   # Mixed
for bar in [bars_ph[best_idx], bars_mn[best_idx]]:
    bar.set_edgecolor("#1a1a1a")
    bar.set_linewidth(2.0)

# ── Value labels ──────────────────────────────────────────────────────────────
def label_bars(bars, values):
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.012,
                f"{val:.3f}",
                ha="center", va="bottom",
                fontsize=14, fontweight="bold", color="#333333")

label_bars(bars_ph, phantom_dice)
label_bars(bars_mn, mendeley_dice)

# ── Axes ──────────────────────────────────────────────────────────────────────
ax.set_xticks(x)
ax.set_xticklabels(experiments, multialignment="center")
ax.set_ylabel("Dice Score")
ax.set_ylim(0.0, 1.12)
ax.set_yticks(np.arange(0, 1.01, 0.2))
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.1f}"))

ax.xaxis.grid(False)
ax.yaxis.grid(True)
ax.set_axisbelow(True)

for spine in ax.spines.values():
    spine.set_zorder(5)

# ── Best annotation ───────────────────────────────────────────────────────────
ax.annotate("Best",
            xy=(x[best_idx], max(phantom_dice[best_idx], mendeley_dice[best_idx]) + 0.045),
            ha="center", va="bottom", fontsize=16, fontweight="bold",
            color="#1a1a1a",
            arrowprops=dict(arrowstyle="-", color="#1a1a1a", lw=1.2),
            xytext=(x[best_idx], max(phantom_dice[best_idx], mendeley_dice[best_idx]) + 0.085))

# ── Legend ────────────────────────────────────────────────────────────────────
legend_handles = [
    mpatches.Patch(color=C_PHANTOM,  alpha=ALPHA_BAR, label="Phantom"),
    mpatches.Patch(color=C_MENDELEY, alpha=ALPHA_BAR, label="CCA"),
]
ax.legend(handles=legend_handles,
          loc="upper center", bbox_to_anchor=(0.5, -0.10),
          ncol=2, fontsize=18,
          frameon=True, edgecolor="#cccccc", fancybox=True, framealpha=0.6)

# ── Save ──────────────────────────────────────────────────────────────────────
out_dir = Path("results/unet/figures")
fig.tight_layout()
fig.subplots_adjust(bottom=0.18)
saved = FigureConfig.save(fig, out_dir / "exp_results", formats=("pdf", "svg", "png"))
print("Saved → " + "  +  ".join(str(p) for p in saved))
