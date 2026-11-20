"""Restrained colour-blind-safe Matplotlib defaults."""
from cycler import cycler
import matplotlib as mpl

COLORS = {
    "baseline": "#4C566A",
    "l2": "#0072B2",
    "l4": "#56B4E9",
    "l6": "#009E73",
    "l10": "#E69F00",
    "linf": "#D55E00",
    "task": "#CC79A7",
    "gray": "#999999",
}


def apply_style():
    mpl.rcParams.update({
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.prop_cycle": cycler(color=[COLORS[k] for k in ("baseline", "l2", "l4", "l6", "l10", "linf", "task")]),
        "grid.alpha": 0.22,
        "lines.linewidth": 1.8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
