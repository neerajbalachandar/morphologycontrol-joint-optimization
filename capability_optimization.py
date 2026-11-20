"""Find the lowest-cost morphologies meeting reachable-area gain targets.

Run from the repository root with ``python3 capability_optimization.py``.
Study settings are edited in ``STUDY_OVERRIDES`` below.
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
RUN_ID = "capability_targets"
RESTARTS = 1
STUDY_OVERRIDES = {
    "baseline_lengths": (2.0, 2.0),
    "total_mass": 2.0,
    "total_torque": 2.0,
    "n_min": 1,
    "n_max": 5,
    "length_min": 0.2,
    "length_max": 4.0,
    "target_gains": (0.05, 0.10, 0.20, 0.30),
    "horizon": 2.0,
    "theta_samples": 180,
    "time_samples": 300,
}

import config as _config

_config.CONFIG = replace(_config.CONFIG, **STUDY_OVERRIDES)

from config import CONFIG
from optimization_variable_n import RESULTS_ROOT, run as run_nested_optimization
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
    capability = metadata["capability_optimizations"]["linf_area"]
    rows = [r for r in capability["results_by_N"] if r.get("status") in ("optimized", "evaluated")]
    targets = metadata["minimum_cost_targets"]
    if not rows:
        raise RuntimeError("No feasible morphology was optimized.")

    apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    n = np.asarray([r["N"] for r in rows])
    gains = np.asarray([100 * (r["area"] / capability["baseline_objective"] - 1) for r in rows])
    costs = np.asarray([r["design_cost"] for r in rows])
    axes[0].plot(n, gains, "o-", color=COLORS["linf"], label="Best gain at each N")
    for target in CONFIG.target_gains:
        axes[0].axhline(100 * target, color=COLORS["gray"], lw=0.9, ls=":")
    axes[0].set(xlabel="Number of links, N", ylabel="Reachable-area improvement [%]",
                title="Capability available by morphology size")
    axes[0].set_xticks(n); axes[0].legend(frameon=False)
    axes[1].plot(n, costs, "o-", color=COLORS["l2"], label="Relative design cost")
    for target in targets:
        if target["status"] == "feasible":
            axes[1].scatter(target["N"], target["design_cost"], s=65,
                            label=f"{100*target['gain_target']:.0f}% target: N={target['N']}")
    axes[1].set(xlabel="Number of links, N", ylabel="Dimensionless design-cost proxy",
                title="Minimum cost meeting each gain target")
    axes[1].set_xticks(n); axes[1].legend(frameon=False, fontsize=7)
    for ax in axes: ax.grid(True)
    fig.tight_layout()
    output_dir = run_dir / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    save_figure(fig, output_dir, "capability_optimization")
    print("Minimum-cost target designs:")
    for item in targets:
        if item["status"] == "feasible":
            print(f"  {100*item['gain_target']:.0f}%: N={item['N']}, "
                  f"lengths={np.round(item['lengths'], 4)} m, cost={item['design_cost']:.4g}")
        else:
            print(f"  {100*item['gain_target']:.0f}%: infeasible in configured N range")
    print(f"Saved results and plots under {run_dir}")


if __name__ == "__main__":
    main()
