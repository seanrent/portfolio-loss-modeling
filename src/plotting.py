"""
Chart helpers.

One place for the visual language of the study, so every figure in the report
looks like it came from the same analyst. Nothing here is clever -- it is a
palette, a style function, and a save function. The point is consistency.

Design choices worth knowing:

* Vintage year is an ORDINAL variable (2012 comes before 2013), so vintage
  curves use a single-hue blue ramp light->dark rather than eight unrelated
  colors. The reader gets "older vs newer" from the color without reading the
  legend.
* Categorical comparisons (e.g. logistic vs boosted) use a fixed slot order,
  never a cycled default, so the same entity keeps the same color across figures.
* Loss severity uses a sequential ramp; the tail of the loss distribution is the
  one place a "critical" red appears, because that is the thing the eye should
  find first.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# --------------------------------------------------------------------------- #
# Palette
# --------------------------------------------------------------------------- #

# Chart chrome -- deliberately recessive so the data carries the contrast.
SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
AXISLINE = "#c3c2b7"

# Categorical slots, in fixed order. Assign by position, never by cycling.
CATEGORICAL = [
    "#2a78d6",  # 1 blue
    "#eb6834",  # 2 orange
    "#1baf7a",  # 3 aqua
    "#eda100",  # 4 yellow
    "#e87ba4",  # 5 magenta
    "#008300",  # 6 green
    "#4a3aa7",  # 7 violet
    "#e34948",  # 8 red
]

# Single-hue blue ramp, light -> dark. Used for ordered things: vintage year,
# FICO band, loan grade. The lightest step is held back from the surface so an
# ordinal series stays legible (a sequential heatmap may go lighter -- there,
# "nearly invisible" correctly means "nearly zero").
BLUE_RAMP = [
    "#86b6ef",
    "#5598e7",
    "#3987e5",
    "#2a78d6",
    "#256abf",
    "#1c5cab",
    "#184f95",
    "#104281",
    "#0d366b",
]

# Sequential colormap for heatmaps (loss rate by segment).
SEQ_BLUE = LinearSegmentedColormap.from_list(
    "seq_blue", ["#eaf2fd", "#cde2fb", "#9ec5f4", "#5598e7", "#2a78d6", "#184f95", "#0d366b"]
)

# Reserved status colors. These never stand in for "series 4".
STATUS_CRITICAL = "#d03b3b"
STATUS_WARNING = "#fab219"
STATUS_GOOD = "#0ca30c"

FIGURES_DIR = Path(__file__).resolve().parents[1] / "figures"


def ordinal_colors(n: int) -> list[str]:
    """
    Evenly spaced steps from the blue ramp for `n` ordered categories.

    Used wherever the categories have a natural order (vintage year, FICO band).
    Spreading across the full ramp keeps adjacent lines distinguishable even
    when there are seven or eight of them.
    """
    if n <= 1:
        return [BLUE_RAMP[3]]
    step = (len(BLUE_RAMP) - 1) / (n - 1)
    return [BLUE_RAMP[round(i * step)] for i in range(n)]


def set_style() -> None:
    """
    Apply the study's matplotlib style. Call once at the top of each notebook.

    Everything here is about lowering the visual weight of the chrome: hairline
    gridlines on the value axis only, no top/right spines, muted tick labels.
    The reader should see the data first and the axes second.
    """
    mpl.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "figure.dpi": 110,
            "savefig.dpi": 200,
            "savefig.bbox": "tight",
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
            "font.size": 10,
            "axes.titlesize": 13,
            "axes.titleweight": "semibold",
            "axes.titlecolor": INK_PRIMARY,
            "axes.titlepad": 10,
            "axes.labelsize": 10,
            "axes.labelcolor": INK_SECONDARY,
            "axes.edgecolor": AXISLINE,
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": GRIDLINE,
            "grid.linewidth": 0.7,
            "xtick.color": INK_MUTED,
            "ytick.color": INK_MUTED,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.frameon": False,
            "legend.fontsize": 9,
            "legend.labelcolor": INK_SECONDARY,
            "lines.linewidth": 2.0,
            "lines.markersize": 5,
            "axes.prop_cycle": mpl.cycler(color=CATEGORICAL),
        }
    )


def style_axes(ax, *, xgrid: bool = False, ygrid: bool = True) -> None:
    """
    Per-axes cleanup: grid on the value axis only.

    A gridline exists so the eye can carry a value across to the axis. On a
    time axis it carries nothing, so it comes off by default.
    """
    ax.grid(axis="y", visible=ygrid)
    ax.grid(axis="x", visible=xgrid)


def annotate_last(ax, x, y, label, color, *, dx: float = 0.6, fontsize: int = 9) -> None:
    """
    Direct-label the end of a line.

    Direct labels beat legend hunting when there are only a handful of series.
    We still keep the legend -- identity should never depend on color alone --
    but the label at the line's end is what people actually read.
    """
    ax.annotate(
        label,
        xy=(x, y),
        xytext=(x + dx, y),
        color=color,
        fontsize=fontsize,
        va="center",
        fontweight="semibold",
    )


def pct_axis(ax, axis: str = "y", decimals: int = 1) -> None:
    """Format an axis as percentages (values supplied as fractions)."""
    fmt = mpl.ticker.FuncFormatter(lambda v, _: f"{v * 100:.{decimals}f}%")
    (ax.yaxis if axis == "y" else ax.xaxis).set_major_formatter(fmt)


def save_fig(fig, name: str) -> Path:
    """
    Save a figure into figures/ and return the path.

    Figures are exported rather than only rendered inline because REPORT.md
    embeds them -- the report has to be readable by someone who never opens a
    notebook.
    """
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURES_DIR / name
    fig.savefig(path)
    return path
