"""Run the separate common-output tracking comparison for one explicit run."""
from __future__ import annotations

import argparse
import json

import numpy as np

from config import CONFIG
from optimization_variable_n import RESULTS_ROOT
from task_tracking import constrained_tracking_qp, optimize_task_morphology


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    run_dir = RESULTS_ROOT / args.run_id
    metadata_path = run_dir / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Run {args.run_id!r} is missing {metadata_path}")
    metadata = json.loads(metadata_path.read_text())
    reach = metadata["capability_optimizations"]["linf_area"]["best_overall"]
    reach_lengths = np.asarray(reach["lengths"], dtype=float)
    candidates = [
        ("baseline", np.asarray(metadata["baseline"]["lengths"], dtype=float)),
        ("reachability_optimum", reach_lengths),
    ]
    task_designs = []
    for n in range(1, metadata["study_configuration"]["n_max"] + 1):
        try:
            result = optimize_task_morphology(n)
            task_designs.append(result)
        except ValueError:
            continue
    if not task_designs:
        raise RuntimeError("No feasible task-optimized morphology was found.")
    task_best = min(task_designs, key=lambda item: item["objective"])
    candidates.append(("task_optimum", np.asarray(task_best["lengths"], dtype=float)))

    records, arrays = {}, {}
    for label, lengths in candidates:
        result = constrained_tracking_qp(lengths)
        records[label] = {
            "lengths": lengths.tolist(), "N": len(lengths),
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
        }
        for key in ("times", "states", "torques", "output", "reference", "tracking_error"):
            arrays[f"{label}_{key}"] = result[key]
    payload = {
        "study_configuration": CONFIG.snapshot(),
        "experiment": "separate finite-horizon constrained end-effector output tracking QP",
        "reference": "x_d=x0+A*sin(omega*t), xdot_d=A*omega*cos(omega*t), common physical units",
        "task_settings": {
            "horizon_s": CONFIG.task_horizon,
            "steps": CONFIG.task_steps,
            "amplitude_m": CONFIG.task_reference_amplitude,
            "frequency_rad_s": CONFIG.task_reference_frequency,
            "effort_weight": CONFIG.task_effort_weight,
            "actuator_budget_convention": "fixed_total",
        },
        "task_optimum_selection": "minimum QP objective across feasible N after fixed-N length optimization",
        "designs": records,
        "fixed_n_task_search": [
            {"N": int(item["lengths"].size), "lengths": np.asarray(item["lengths"]).tolist(),
             "objective": float(item["objective"]),
             "morphology_optimizer_success": bool(item["morphology_optimizer_success"])}
            for item in task_designs
        ],
    }
    (run_dir / "task_tracking.json").write_text(json.dumps(payload, indent=2) + "\n")
    np.savez_compressed(run_dir / "task_tracking_arrays.npz", **arrays)
    print(f"Saved task comparison under {run_dir}")


if __name__ == "__main__":
    main()
