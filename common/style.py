"""Matplotlib defaults and a colour-blind-safe palette shared by all figures."""
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt

COLORS = {
    "baseline": "#4C566A",
    "l2": "#0072B2",
    "l6": "#009E73",
    "linf": "#D55E00",
    "setbased": "#0072B2",
    "nominal": "#D55E00",
    "sequential": "#E69F00",
    "accent": "#CC79A7",
    "gray": "#999999",
    "blue_pts": "#2F6FDE",
    "red_pts": "#D1495B",
    "magenta": "#C2185B",
}


def apply_style():
    mpl.rcParams.update({
        "figure.dpi": 120,
        "savefig.dpi": 200,
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "legend.fontsize": 7.5,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.22,
        "lines.linewidth": 1.6,
        "pdf.fonttype": 42,
    })


def save(fig, folder: Path, stem: str):
    folder.mkdir(parents=True, exist_ok=True)
    fig.savefig(folder / f"{stem}.png", bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {folder.name}/{stem}.png")
