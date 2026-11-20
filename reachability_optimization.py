"""Optimize and plot the N-link arm's finite-horizon L-infinity reachability.

Run from the repository root with ``python3 reachability_optimization.py``.
Study settings are edited in ``STUDY_OVERRIDES`` below. The dynamics and
numerical integration routines are shared with the N-dof-arm implementation.
"""
from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
LIBRARY = ROOT / "N-dof-arm"
sys.path.insert(0, str(LIBRARY))

# Edit these values to change the experiment. Omitted fields keep config.py defaults.
RUN_ID = "reachability_linf"
RESTARTS = 1
STUDY_OVERRIDES = {
    "baseline_lengths": (2.0, 2.0),
    "total_mass": 2.0,
    "total_torque": 2.0,
    "n_min": 1,
    "n_max": 5,
    "length_min": 0.2,
    "length_max": 4.0,
    "horizon": 2.0,
    "theta_samples": 180,
    "time_samples": 300,
}

import config as _config

_config.CONFIG = replace(_config.CONFIG, **STUDY_OVERRIDES)

from config import CONFIG
from design import BASELINE_LENGTHS
from optimization_variable_n import RESULTS_ROOT, run as run_nested_optimization
from optimization_fixed_n import evaluate_capability
from plotting.style import COLORS, apply_style


def save_figure(fig, output_dir: Path, stem: str) -> None:
    fig.savefig(output_dir / f"{stem}.png", bbox_inches="tight")
    fig.savefig(output_dir / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    run_nested_optimization(
        RUN_ID, criteria=("linf_area",), restarts=RESTARTS,
        actuator_convention="fixed_total",
    )
    run_dir = RESULTS_ROOT / RUN_ID
    metadata = json.loads((run_dir / "metadata.json").read_text())
    result = metadata["capability_optimizations"]["linf_area"]
    rows = [r for r in result["results_by_N"] if r.get("status") in ("optimized", "evaluated")]
    if not rows:
        raise RuntimeError("No feasible morphology was optimized.")
    best = result["best_overall"]
    baseline = evaluate_capability(BASELINE_LENGTHS, "linf_area")
    optimum = evaluate_capability(np.asarray(best["lengths"]), "linf_area")

    apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    axes[0].plot([r["N"] for r in rows], [r["area"] for r in rows], "o-", color=COLORS["linf"])
    axes[0].axhline(baseline["area"], color=COLORS["baseline"], ls="--", label="Baseline")
    axes[0].scatter([best["N"]], [best["area"]], s=70, color=COLORS["task"], zorder=3,
                    label=f"Best: N={best['N']}")
    axes[0].set(xlabel="Number of links, N", ylabel="Reachable area [dimensionless]",
                title="Best reachability by morphology size")
    axes[0].set_xticks([r["N"] for r in rows]); axes[0].legend(frameon=False)
    for item, label, color in ((baseline, "Baseline", COLORS["baseline"]),
                               (optimum, "Reachability optimum", COLORS["linf"])):
        boundary = np.vstack((item["geometry"]["boundary"], item["geometry"]["boundary"][0]))
        axes[1].plot(boundary[:, 0], boundary[:, 1], color=color,
                     label=f"{label} (A={item['area']:.3g})")
    axes[1].set(xlabel="$x_{EE}/L_0$", ylabel="$T\\dot{x}_{EE}/L_0$",
                title="Bounded-torque reachable set")
    axes[1].set_aspect("equal", adjustable="datalim"); axes[1].legend(frameon=False)
    for ax in axes: ax.grid(True)
    fig.tight_layout()
    output_dir = run_dir / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    save_figure(fig, output_dir, "reachability_optimization")
    print(f"Reachability best: N={best['N']}, lengths={np.round(best['lengths'], 4)} m")
    print(f"Area: {best['area']:.7g}; baseline: {baseline['area']:.7g}")
    print(f"Saved results and plots under {run_dir}")


if __name__ == "__main__":
    main()
