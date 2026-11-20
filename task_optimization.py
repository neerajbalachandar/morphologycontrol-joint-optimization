"""Optimize morphology for bounded-torque end-effector tracking and visualize it.

Run from the repository root with ``python3 task_optimization.py``. It compares
the task-optimal morphology against the baseline under the same reference.
"""
from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np

ROOT = Path(__file__).resolve().parent
LIBRARY = ROOT / "N-dof-arm"
sys.path.insert(0, str(LIBRARY))

# Edit these values to change the experiment. Omitted fields keep config.py defaults.
RUN_ID = "task_tracking"
ACTUATOR_CONVENTION = "fixed_total"
STUDY_OVERRIDES = {
    "baseline_lengths": (2.0, 2.0),
    "total_mass": 2.0,
    "total_torque": 2.0,
    "n_min": 1,
    "n_max": 5,
    "length_min": 0.2,
    "length_max": 4.0,
    "task_horizon": 5.0,
    "task_steps": 125,
    "task_reference_amplitude": 0.12,
    "task_reference_frequency": 1.0,
}

import config as _config

_config.CONFIG = replace(_config.CONFIG, **STUDY_OVERRIDES)

from config import CONFIG
from design import BASELINE_LENGTHS
from model import kinematic_points
from optimization_variable_n import RESULTS_ROOT
from task_tracking import constrained_tracking_qp, optimize_task_morphology
from plotting.style import COLORS, apply_style


def _save(fig, output_dir: Path, stem: str) -> None:
    fig.savefig(output_dir / f"{stem}.png", bbox_inches="tight")
    fig.savefig(output_dir / f"{stem}.pdf", bbox_inches="tight")


def _animate(results: dict, lengths_by_name: dict, output_dir: Path) -> Path:
    names = ("baseline", "task_optimum")
    max_length = float(sum(BASELINE_LENGTHS))
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 4.2))
    points = {}
    arms, trails = [], []
    for ax, name in zip(axes, names):
        lengths = lengths_by_name[name]
        q = results[name]["states"][:, :len(lengths)]
        points[name] = np.asarray([kinematic_points(qi, lengths) for qi in q])
        ax.set(xlim=(-1.1 * max_length, 1.1 * max_length),
               ylim=(-1.1 * max_length, 0.25 * max_length),
               xlabel="x [m]", ylabel="y [m]", title=name.replace("_", " ").title())
        ax.set_aspect("equal", adjustable="box"); ax.grid(True, alpha=0.22)
        arm, = ax.plot([], [], "o-", lw=2.5, ms=5, color=COLORS["task"])
        trail, = ax.plot([], [], "--", lw=1.2, alpha=0.6, color=COLORS["linf"])
        arms.append(arm); trails.append(trail)
    time_text = fig.text(0.5, 0.015, "", ha="center")

    def update(frame):
        for i, name in enumerate(names):
            xy = points[name][frame]
            arms[i].set_data(xy[:, 0], xy[:, 1])
            trails[i].set_data(points[name][:frame + 1, -1, 0], points[name][:frame + 1, -1, 1])
        time_text.set_text(f"Bounded-torque tracking, t={results['baseline']['times'][frame]:.2f} s")
        return (*arms, *trails, time_text)

    animation = FuncAnimation(fig, update, frames=len(results["baseline"]["times"]),
                              interval=40, blit=False)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    output_dir.mkdir(parents=True, exist_ok=True)
    video = output_dir / "task_tracking_morphology.mp4"
    try:
        animation.save(video, writer="ffmpeg", fps=25, dpi=140)
    except (RuntimeError, FileNotFoundError):
        video = output_dir / "task_tracking_morphology.gif"
        animation.save(video, writer="pillow", fps=20, dpi=100)
    plt.close(fig)
    return video


def main() -> None:
    designs = []
    for n in range(CONFIG.n_min, CONFIG.n_max + 1):
        try:
            designs.append(optimize_task_morphology(n, actuator_convention=ACTUATOR_CONVENTION))
        except ValueError:
            continue
    if not designs:
        raise RuntimeError("No feasible task morphology was found.")
    optimum = min(designs, key=lambda row: row["objective"])
    lengths_by_name = {
        "baseline": np.asarray(BASELINE_LENGTHS, dtype=float),
        "task_optimum": np.asarray(optimum["lengths"], dtype=float),
    }
    results = {
        name: constrained_tracking_qp(lengths, actuator_convention=ACTUATOR_CONVENTION)
        for name, lengths in lengths_by_name.items()
    }
    run_dir = RESULTS_ROOT / RUN_ID
    output_dir = run_dir / "plots"
    apply_style()

    fig, axes = plt.subplots(2, 2, figsize=(10.2, 7.0))
    for name, color, label in (("baseline", COLORS["baseline"], "Baseline"),
                               ("task_optimum", COLORS["task"], "Task optimum")):
        result = results[name]
        axes[0, 0].plot(result["times"], result["reference"][:, 0], color="black", ls="--", lw=1.2,
                        label="Reference" if name == "baseline" else None)
        axes[0, 0].plot(result["times"], result["output"][:, 0], color=color, label=label)
        axes[0, 1].plot(result["times"], result["reference"][:, 1], color="black", ls="--", lw=1.2,
                        label="Reference" if name == "baseline" else None)
        axes[0, 1].plot(result["times"], result["output"][:, 1], color=color, label=label)
        axes[1, 0].plot(result["times"][:-1], np.max(np.abs(result["torques"]), axis=1),
                        color=color, label=label)
    axes[0, 0].set(title="End-effector position", ylabel="$x_{EE}$ [m]")
    axes[0, 1].set(title="End-effector velocity", ylabel="$\\dot{x}_{EE}$ [m/s]")
    axes[1, 0].set(title="Peak motor torque", ylabel="Max $|\\tau_i|$ [Nm]", xlabel="Time [s]")
    axes[1, 1].bar(["Baseline", "Task optimum"],
                   [results[k]["rms_output_error"] for k in ("baseline", "task_optimum")],
                   color=[COLORS["baseline"], COLORS["task"]])
    axes[1, 1].set(title="Tracking error", ylabel="RMS output error")
    for ax in axes.flat:
        ax.grid(True, alpha=0.22)
        if ax is not axes[1, 1]: ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    _save(fig, output_dir, "task_tracking_comparison")
    plt.close(fig)

    payload = {
        "study_configuration": CONFIG.snapshot(),
        "actuator_budget_convention": ACTUATOR_CONVENTION,
        "selected_task_morphology": {
            "N": len(optimum["lengths"]), "lengths": np.asarray(optimum["lengths"]).tolist(),
            "objective": float(optimum["objective"]),
            "morphology_optimizer_success": bool(optimum.get("morphology_optimizer_success", True)),
            "morphology_optimizer_message": optimum.get("morphology_optimizer_message", "Direct evaluation; N=1 has no free link lengths."),
        },
        "tracking_results": {
            name: {
                "lengths": result["lengths"].tolist(),
                "rms_output_error": result["rms_output_error"],
                "rms_position_error": result["rms_position_error"],
                "rms_velocity_error": result["rms_velocity_error"],
                "tracking_cost": result["tracking_cost"],
                "control_effort_cost": result["control_effort_cost"],
                "objective": result["objective"],
                "optimizer_success": result["optimizer_success"],
                "optimizer_message": result["optimizer_message"],
                "torque_limits": result["torque_limits"].tolist(),
                "actuator_budget_convention": result["actuator_budget_convention"],
            } for name, result in results.items()
        },
        "fixed_n_task_search": [
            {"N": len(item["lengths"]), "lengths": np.asarray(item["lengths"]).tolist(),
             "objective": float(item["objective"]),
             "morphology_optimizer_success": bool(item.get("morphology_optimizer_success", True)),
             "morphology_optimizer_message": item.get("morphology_optimizer_message", "Direct evaluation; N=1 has no free link lengths.")}
            for item in designs
        ],
    }
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "task_optimization.json").write_text(json.dumps(payload, indent=2) + "\n")
    arrays = {}
    for name, result in results.items():
        for key in ("times", "states", "torques", "output", "reference", "tracking_error"):
            arrays[f"{name}_{key}"] = result[key]
    np.savez_compressed(run_dir / "task_optimization_arrays.npz", **arrays)
    video = _animate(results, lengths_by_name, output_dir)
    print(f"Task optimum: N={len(optimum['lengths'])}, lengths={np.round(optimum['lengths'], 4)} m")
    print(f"RMS output error: baseline={results['baseline']['rms_output_error']:.5g}, "
          f"task optimum={results['task_optimum']['rms_output_error']:.5g}")
    print(f"Saved results, plots, and animation under {run_dir}; animation: {video.name}")


if __name__ == "__main__":
    main()
