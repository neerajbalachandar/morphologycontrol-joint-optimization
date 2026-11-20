"""Authoritative nested discrete-continuous reachability optimizer."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from config import CONFIG
from design import (
    ACTUATOR_COST_PER_BASELINE_JOINT,
    BASELINE_LENGTHS,
    BASELINE_TOTAL_LENGTH,
    LINK_COST_PER_BASELINE_LENGTH,
    N_BASELINE,
    N_MAX,
    N_MIN,
    TOTAL_MASS,
    actuator_limits,
)
from optimization_fixed_n import evaluate_capability, feasible_n, optimize_fixed_n


RESULTS_ROOT = Path(__file__).resolve().parent / "results"
CRITERIA = ("l2_trace", "l6_area", "linf_area")
TARGET_GAINS = CONFIG.target_gains


def morphology_cost(n):
    """Common dimensionless proxy: wL*L/L0 + wN*N/N0 (not a price)."""
    return float(
        LINK_COST_PER_BASELINE_LENGTH
        + ACTUATOR_COST_PER_BASELINE_JOINT * n / N_BASELINE
    )


def serialize_record(record):
    """Convert numpy values and remove bulky arrays before metadata export."""
    excluded = {"A", "B", "M", "K", "D", "gramian", "geometry", "evaluations"}

    def convert(value):
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, (np.floating, np.integer, np.bool_)):
            return value.item()
        if isinstance(value, dict):
            return {str(k): convert(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [convert(v) for v in value]
        return value

    return {key: convert(value) for key, value in record.items() if key not in excluded}


def nested_capability(criterion, convention="peak_equivalent",
                      actuator_convention="fixed_total", restarts=1):
    """Optimize links independently for each feasible N and compare objectives."""
    rows, all_candidates = [], {}
    for n in range(N_MIN, N_MAX + 1):
        if not feasible_n(n):
            rows.append({"N": n, "status": "infeasible",
                         "reason": "N*l_min <= L0 <= N*l_max is false."})
            continue
        row = optimize_fixed_n(
            n, criterion=criterion, convention=convention,
            actuator_convention=actuator_convention, restarts=restarts,
        )
        if row["status"] in {"optimized", "evaluated"}:
            lengths = np.asarray(row["lengths"], dtype=float)
            # Independent final objective recomputation at every selected design.
            check = evaluate_capability(lengths, criterion, convention, actuator_convention)
            row["independent_recomputed_objective"] = float(check["objective"])
            row["objective_consistency_error"] = float(check["objective"] - row["objective"])
            row["improvement_vs_baseline_percent"] = None
            row["design_cost"] = morphology_cost(n)
            row["total_length"] = float(lengths.sum())
            row["constraint_residual"] = float(lengths.sum() - BASELINE_TOTAL_LENGTH)
            row["torque_limits"] = actuator_limits(n, actuator_convention).tolist()
            all_candidates[f"{criterion}_N{n}_gramian"] = row["gramian"]
            all_candidates[f"{criterion}_N{n}_gramian_eigenvalues"] = row["gramian_eigenvalues"]
            if row.get("geometry"):
                all_candidates[f"{criterion}_N{n}_boundary"] = row["geometry"]["boundary"]
                all_candidates[f"{criterion}_N{n}_support"] = row["geometry"]["support"]
                all_candidates[f"{criterion}_N{n}_theta"] = row["geometry"]["theta"]
            if row.get("evaluations"):
                all_candidates[f"{criterion}_N{n}_candidate_lengths"] = np.asarray(
                    [entry["lengths"] for entry in row["evaluations"]], dtype=float
                )
                all_candidates[f"{criterion}_N{n}_candidate_objective"] = np.asarray(
                    [entry["objective"] for entry in row["evaluations"]], dtype=float
                )
            print(
                f"{criterion}: N={n}, objective={row['objective']:.8e}, "
                f"links={np.round(lengths, 5)}, solver_success={row['optimizer_success']}"
            )
        rows.append(row)
    eligible = [r for r in rows if r.get("status") in {"optimized", "evaluated"}]
    best = max(eligible, key=lambda item: item["objective"]) if eligible else None
    baseline = evaluate_capability(
        np.asarray(BASELINE_LENGTHS), criterion, convention, actuator_convention
    )
    all_candidates[f"{criterion}_baseline_gramian"] = baseline["gramian"]
    for row in rows:
        if row.get("status") in {"optimized", "evaluated"}:
            row["baseline_objective"] = float(baseline["objective"])
            row["improvement_vs_baseline_percent"] = float(
                100 * (row["objective"] / baseline["objective"] - 1)
            )
    if best:
        best["improvement_vs_baseline_percent"] = float(
            100 * (best["objective"] / baseline["objective"] - 1)
        )
    return {"criterion": criterion, "input_norm_convention": convention,
            "actuator_budget_convention": actuator_convention,
            "baseline_objective": float(baseline["objective"]),
            "best_overall": serialize_record(best) if best else None,
            "results_by_N": [serialize_record(row) for row in rows],
            "candidate_arrays": all_candidates}


def minimum_cost_designs(linf_rows, baseline_area):
    """Compare target-feasible designs; select lowest N then largest margin."""
    target_rows = []
    for gain in TARGET_GAINS:
        target = (1.0 + gain) * baseline_area
        feasible_rows = [r for r in linf_rows
                         if r.get("status") in {"optimized", "evaluated"}
                         and r.get("area", r.get("objective", -np.inf)) >= target]
        # Candidate optimization records store selected morphology; all link
        # redistribution has identical J_m under fixed L0, so capability margin
        # is the transparent tie-breaker after minimizing actuator count.
        if feasible_rows:
            min_n = min(r["N"] for r in feasible_rows)
            at_min_n = [r for r in feasible_rows if r["N"] == min_n]
            selected = max(at_min_n, key=lambda r: r.get("area", r.get("objective", -np.inf)))
            target_rows.append({
                "gain_target": gain, "area_target": target,
                "status": "feasible", "N": min_n,
                "lengths": selected["lengths"],
                "area": float(selected.get("area", selected.get("objective"))),
                "capability_margin": float(selected.get("area", selected.get("objective")) - target),
                "design_cost": morphology_cost(min_n),
                "tie_breaker": "largest capability margin at minimum feasible N",
            })
        else:
            target_rows.append({"gain_target": gain, "area_target": target,
                                "status": "infeasible_within_configured_N_range"})
    return target_rows


def run(run_id="canonical_fixed_total", criteria=CRITERIA, restarts=1,
        actuator_convention="fixed_total"):
    run_dir = RESULTS_ROOT / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    optimization = {}
    arrays = {}
    for criterion in criteria:
        result = nested_capability(criterion, CONFIG.norm_convention,
                                   actuator_convention, restarts)
        optimization[criterion] = {k: v for k, v in result.items() if k != "candidate_arrays"}
        arrays.update(result["candidate_arrays"])

    baseline = evaluate_capability(np.asarray(BASELINE_LENGTHS), "linf_area",
                                   CONFIG.norm_convention, actuator_convention)
    linf_rows = optimization.get("linf_area", {}).get("results_by_N", [])
    targets = minimum_cost_designs(linf_rows, baseline["objective"]) if linf_rows else []
    metadata = {
        "schema_version": "1.0",
        "run_id": run_id,
        "study_configuration": CONFIG.snapshot(),
        "formulation": "nested fixed-N continuous capability optimization, then exhaustive N comparison",
        "primary_actuator_budget_convention": actuator_convention,
        "actuator_limits_by_N": {
            str(n): actuator_limits(n, actuator_convention).tolist()
            for n in range(N_MIN, N_MAX + 1)
        },
        "design_cost": {
            "equation": "J_m = w_L L/L0 + w_N N/N0",
            "w_L": LINK_COST_PER_BASELINE_LENGTH,
            "w_N": ACTUATOR_COST_PER_BASELINE_JOINT,
            "interpretation": "dimensionless relative proxy, not measured manufacturing cost",
            "fixed_length_note": "L=L0 makes the link term constant; actuator/joint count drives variable cost",
        },
        "baseline": {
            "N": len(BASELINE_LENGTHS), "lengths": list(BASELINE_LENGTHS),
            "total_length": BASELINE_TOTAL_LENGTH, "total_mass": TOTAL_MASS,
            "linf_area": baseline["objective"],
            "physical_output_area": baseline["physical_output_area"],
            "gramian_trace_l2": baseline["gramian_trace_l2"],
            "gramian_eigenvalues": baseline["gramian_eigenvalues"].tolist(),
            "gramian_logdet": baseline["gramian_logdet"],
        },
        "numerical_resolution": {
            "angular_samples": CONFIG.theta_samples,
            "time_quadrature_samples": CONFIG.time_samples,
        },
        "capability_optimizations": optimization,
        "minimum_cost_targets": targets,
        "limitations": [
            "DE success reports numerical stopping status, not a global-optimality proof.",
            "Reachability areas are polygonal numerical estimates at the recorded resolution.",
        ],
    }
    (run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n")
    if arrays:
        np.savez_compressed(run_dir / "candidate_and_geometry_arrays.npz", **arrays)
    print(f"Saved run metadata and arrays under {run_dir}")
    return metadata


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default="canonical_fixed_total")
    parser.add_argument("--restarts", type=int, default=1)
    parser.add_argument("--actuator-convention", choices=("fixed_total", "fixed_per_joint"), default="fixed_total")
    parser.add_argument("--criteria", nargs="+", choices=CRITERIA, default=list(CRITERIA))
    args = parser.parse_args()
    run(args.run_id, args.criteria, args.restarts, args.actuator_convention)
