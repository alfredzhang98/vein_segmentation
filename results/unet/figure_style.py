"""Unified figure styling for analysis plots.

Single source of truth for matplotlib rcParams across every analysis
script. The DEFAULT preset is sized for IEEE 2-column papers (RA-L /
ICRA / T-RO) where figures are downscaled to ~3.5 in / 7.16 in print
width: large fonts at draft size still render clearly after scaling
and remain legible in greyscale print.

Three presets:

    FigureConfig()                   # default — large, bold, Arial
    FigureConfig.ieee_strict()       # small (8–10 pt) for already-final layout
    FigureConfig.presentation()      # even larger for slides / posters

Usage:
    from figure_style import FigureConfig
    FigureConfig().apply()             # at script top, once
    fig, ax = plt.subplots()
    ax.plot(...)
    FigureConfig.save(fig, "out/fig1") # → out/fig1.pdf, out/fig1.svg

Why the default is "big":
    A 22-pt label drafted on a 10×6 in canvas → after IEEE 2-col scaling
    to ~7.16 in, the label prints at ~16 pt apparent — comfortably above
    IEEE's 8 pt minimum and readable at 100% zoom on screen.
    Drafting at IEEE strict 8 pt produces lines/text so thin they tend
    to disappear when figures are reviewed at "fit to page".
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Tuple

import matplotlib.pyplot as plt


# IEEE 2-column paper canonical widths (inches). Use when explicitly
# sizing for the final printed page (figsize=(IEEE_DOUBLE_COL_IN, ...)).
IEEE_SINGLE_COL_IN: float = 3.5      # 88.9 mm
IEEE_DOUBLE_COL_IN: float = 7.16     # 181.8 mm
IEEE_PAGE_HEIGHT_IN: float = 9.0


@dataclass
class FigureConfig:
    """Default = poster / paper-draft style (large bold Arial). Sized so the
    figure stays readable when shrunk to IEEE 2-column print width (~7 in)
    after compositing — drafting at small sizes tends to produce text that
    disappears in greyscale review prints.

    Use FigureConfig.ieee_strict() if your figure is going onto the page
    at exactly its drafted size with no rescaling.
    """
    # ── Size ──
    figsize: Tuple[float, float] = (10.0, 6.0)

    # ── Typography ──
    font_family: str = "sans-serif"
    font_sans: Tuple[str, ...] = ("Arial", "Helvetica", "DejaVu Sans")
    font_size: float = 22.0
    label_size: float = 22.0
    title_size: float = 22.0
    tick_size: float = 20.0
    legend_size: float = 13.0

    bold_labels: bool = True
    bold_title: bool = True
    title_pad: float = 35.0   # gap between subplot title and the data area
                                # (default matplotlib is 6 pt — too tight for paper figures)

    # ── Lines / ticks ──
    axes_line_width: float = 2.0
    plot_line_width: float = 2.0
    tick_width: float = 2.0
    tick_length_major: float = 4.0
    tick_length_minor: float = 2.5
    tick_direction: str = "out"

    # ── Grid ──
    grid: bool = True
    grid_alpha: float = 0.7
    grid_style: str = "--"
    grid_width: float = 1.0
    grid_color: str = "#888"   # darker than matplotlib default so it stays
                                # visible on coloured (pastel) backgrounds

    # ── Legend ──
    legend_frame: bool = False
    legend_loc: str = "best"

    # ── Save ──
    dpi_screen: int = 150
    dpi_save: int = 600
    save_bbox: str = "tight"
    save_transparent: bool = False
    save_formats: Tuple[str, ...] = ("pdf", "svg")

    # ── Apply ──
    def apply(self) -> "FigureConfig":
        """Push these settings into matplotlib's global rcParams."""
        rcp = {
            # Font
            "font.family": self.font_family,
            "font.sans-serif": list(self.font_sans),
            "font.size": self.font_size,
            "font.weight": "bold" if self.bold_labels else "normal",
            "axes.labelsize": self.label_size,
            "axes.labelweight": "bold" if self.bold_labels else "normal",
            "axes.titlesize": self.title_size,
            "axes.titleweight": "bold" if self.bold_title else "normal",
            "axes.titlepad": self.title_pad,
            "xtick.labelsize": self.tick_size,
            "ytick.labelsize": self.tick_size,
            "legend.fontsize": self.legend_size,
            # Lines / ticks
            "axes.linewidth": self.axes_line_width,
            "lines.linewidth": self.plot_line_width,
            "xtick.major.width": self.tick_width,
            "ytick.major.width": self.tick_width,
            "xtick.minor.width": self.tick_width * 0.75,
            "ytick.minor.width": self.tick_width * 0.75,
            "xtick.major.size": self.tick_length_major,
            "ytick.major.size": self.tick_length_major,
            "xtick.minor.size": self.tick_length_minor,
            "ytick.minor.size": self.tick_length_minor,
            "xtick.direction": self.tick_direction,
            "ytick.direction": self.tick_direction,
            # Grid
            "axes.grid": self.grid,
            "grid.linestyle": self.grid_style,
            "grid.linewidth": self.grid_width,
            "grid.alpha": self.grid_alpha,
            "grid.color": self.grid_color,
            # Legend
            "legend.frameon": self.legend_frame,
            "legend.loc": self.legend_loc,
            # Figure / save
            "figure.figsize": self.figsize,
            "figure.dpi": self.dpi_screen,
            "savefig.dpi": self.dpi_save,
            "savefig.bbox": self.save_bbox,
            "savefig.transparent": self.save_transparent,
            # Embed fonts properly so IEEE PDF Express never complains
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
        plt.rcParams.update(rcp)
        return self

    # ── Presets ──
    @classmethod
    def ieee_strict(cls, figsize: Tuple[float, float] = (IEEE_DOUBLE_COL_IN, 4.0),
                    **kw) -> "FigureConfig":
        """Strict IEEE: 8–10 pt fonts, thin lines. Use only when the
        figure is being placed at final page size with no rescaling."""
        defaults = dict(
            figsize=figsize,
            font_size=9.0,
            label_size=9.0,
            title_size=10.0,
            tick_size=8.0,
            legend_size=8.0,
            bold_labels=False,
            bold_title=False,
            axes_line_width=0.8,
            plot_line_width=1.2,
            tick_width=0.8,
            tick_length_major=3.5,
            grid=False,
        )
        defaults.update(kw)
        return cls(**defaults)

    @classmethod
    def presentation(cls, figsize: Tuple[float, float] = (12.0, 7.0),
                     **kw) -> "FigureConfig":
        """Slides / posters: even bigger than the default."""
        defaults = dict(
            figsize=figsize,
            font_size=26.0,
            label_size=26.0,
            title_size=26.0,
            tick_size=22.0,
            legend_size=18.0,
            plot_line_width=2.5,
            axes_line_width=2.0,
        )
        defaults.update(kw)
        return cls(**defaults)

    # Back-compat aliases for the old "ieee_*_col" names: now both map to
    # the default (paper-draft size, large fonts that survive scaling).
    @classmethod
    def ieee_single_col(cls, height: float = 6.0, **kw) -> "FigureConfig":
        kw.setdefault("figsize", (IEEE_SINGLE_COL_IN * 2, height))
        return cls(**kw)

    @classmethod
    def ieee_double_col(cls, height: float = 6.0, **kw) -> "FigureConfig":
        kw.setdefault("figsize", (10.0, height))
        return cls(**kw)

    # ── Save helper ──
    @staticmethod
    def save(fig, path_stem: str | Path,
             formats: Iterable[str] | None = None,
             dpi: int | None = None,
             bbox: str = "tight",
             transparent: bool = False) -> list[Path]:
        """Save `fig` to `path_stem` in every requested format.
        Returns the list of files written."""
        if formats is None:
            formats = ("pdf", "svg")
        if dpi is None:
            dpi = int(plt.rcParams.get("savefig.dpi") or 600)
        path_stem = Path(path_stem)
        path_stem.parent.mkdir(parents=True, exist_ok=True)
        out: list[Path] = []
        for ext in formats:
            ext = ext.lstrip(".")
            p = path_stem.with_suffix(f".{ext}")
            fig.savefig(p, dpi=dpi, bbox_inches=bbox, transparent=transparent)
            out.append(p)
        return out


def apply_default() -> FigureConfig:
    """Apply the default (large-font, Arial-bold) preset. Convenient when
    a script just wants `from figure_style import apply_default; apply_default()`.
    """
    return FigureConfig().apply()
