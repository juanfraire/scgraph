"""Shared figure style for the paper (IEEE OJ-COMS single-column figures).

Series colors are the validated colorblind-safe categorical palette
(light-mode slots; worst adjacent CVD deltaE 24.2). Assignment is fixed by
entity across ALL figures, never by rank:
  SCG (our router)        -> blue   #2a78d6, solid
  primary baseline        -> red    #e34948, dashed
  secondary baseline      -> orange #eb6834, dash-dot
  reference bound         -> gray   #6e6e6e, dotted
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

COL_SCG = "#2a78d6"
COL_BASE1 = "#e34948"
COL_BASE2 = "#eb6834"
COL_BOUND = "#6e6e6e"

STYLE = {
    "scg": dict(color=COL_SCG, linestyle="-", marker="o"),
    "base1": dict(color=COL_BASE1, linestyle="--", marker="s"),
    "base2": dict(color=COL_BASE2, linestyle="-.", marker="^"),
    "bound": dict(color=COL_BOUND, linestyle=":", marker=""),
}


def setup() -> None:
    plt.rcParams.update(
        {
            "figure.figsize": (3.6, 2.5),
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 8,
            "legend.fontsize": 7,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "lines.linewidth": 1.6,
            "lines.markersize": 3.5,
            "axes.grid": True,
            "grid.color": "#dddddd",
            "grid.linewidth": 0.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
        }
    )


def save(fig, stem: str) -> None:
    """Write <stem>.pdf (paper) and <stem>.png (quick look)."""
    fig.tight_layout(pad=0.4)
    fig.savefig(f"{stem}.pdf")
    fig.savefig(f"{stem}.png")
