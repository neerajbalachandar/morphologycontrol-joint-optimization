"""Nested fixed-N optimization of bounded-torque reachable-set area.

This script decomposes morphology optimization into an inner continuous
link-length optimization for each feasible joint count and an outer exhaustive
comparison over joint counts. Both levels use the same reachable-area
objective; this is not a morphology-controller bilevel problem.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import LinearConstraint, differential_evolution

from design import (
    BASELINE_LENGTHS,
    BASELINE_TOTAL_LENGTH,
    N_MIN,
    N_MAX,
    TOTAL_MASS,
    actuator_limits,
)
from model import output_matrix, robot_matrices
from reachability import reachable_set
from reachability_optimization_dual import (
    LENGTH_MAX_DUAL,
    LENGTH_MIN_DUAL,
    MAXITER,
    NTHETA,
    NT,
    POPSIZE,
    SEED,
    T,
)


# Mirrors the tolerance passed to differential_evolution in the joint-N script.
DE_TOL = 1e-3
# Set to a larger integer to repeat each fixed-N optimization from more seeds.
INNER_RESTARTS = 1
LENGTH_TOL = 1e-8
AREA_TOL = 1e-9
JSON_DIR = Path(__file__).resolve().parent / "json"
OUTPUT_PATH = JSON_DIR / "reachability_optimization_dual_bilevel_results.json"


def is_feasible_count(n: int, total_length: float = BASELINE_TOTAL_LENGTH) -> bool:
    """Return whether N links can sum to the fixed total within link bounds."""
    return (
        n >= 1
        and n * LENGTH_MIN_DUAL <= total_length + LENGTH_TOL
        and total_length <= n * LENGTH_MAX_DUAL + LENGTH_TOL
    )


def reconstruct_lengths(
    n: int, independent_lengths: np.ndarray | list[float],
    total_length: float = BASELINE_TOTAL_LENGTH,
) -> np.ndarray:
    """Append the dependent final link so the total length is exactly fixed."""
    if n < 1:
        raise ValueError("N must be at least 1.")
    independent = np.asarray(independent_lengths, dtype=float).reshape(-1)
    if n == 1:
        if independent.size != 0:
            raise ValueError("N=1 has no independent link lengths.")
        return np.array([total_length], dtype=float)
    if independent.size != n - 1:
        raise ValueError(f"Expected {n - 1} independent lengths for N={n}.")
    return np.r_[independent, total_length - np.sum(independent)]


def validate_lengths(
    lengths: np.ndarray, n: int, total_length: float = BASELINE_TOTAL_LENGTH,
) -> tuple[bool, str]:
    """Check positivity, configured bounds, and fixed-total-length equality."""
    values = np.asarray(lengths, dtype=float)
    if values.shape != (n,):
        return False, f"expected {n} lengths, received shape {values.shape}"
    if not np.all(np.isfinite(values)):
        return False, "link lengths contain non-finite values"
    if np.any(values <= 0.0):
        return False, "link lengths must be strictly positive"
    if np.any(values < LENGTH_MIN_DUAL - LENGTH_TOL):
        return False, "one or more link lengths are below the configured lower bound"
    if np.any(values > LENGTH_MAX_DUAL + LENGTH_TOL):
        return False, "one or more link lengths exceed the configured upper bound"
    residual = float(np.sum(values) - total_length)
    if abs(residual) > LENGTH_TOL:
        return False, f"total-length residual {residual:.3e} exceeds tolerance"
    return True, "valid"


def evaluate_area(lengths: np.ndarray) -> float:
    """Evaluate end-effector position-velocity reachable-set area."""
    lengths = np.asarray(lengths, dtype=float)
    A, B, *_ = robot_matrices(lengths, total_mass=TOTAL_MASS)
    C = output_matrix(lengths)
    area = reachable_set(
        A,
        B,
        C,
        actuator_limits(len(lengths)),
        T=T,
        ntheta=NTHETA,
        nt=NT,
    )[3]
    return float(area)


def project_to_bounded_sum(
    values: np.ndarray, target: float, lower: float, upper: float,
) -> np.ndarray:
    """Project a vector onto a box intersected with a fixed-sum hyperplane."""
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return values.copy()
    left = lower - float(np.max(values))
    right = upper - float(np.min(values))
    for _ in range(80):
        shift = 0.5 * (left + right)
        candidate = np.clip(values + shift, lower, upper)
        if float(candidate.sum()) < target:
            left = shift
        else:
            right = shift
    projected = np.clip(values + 0.5 * (left + right), lower, upper)
    residual = target - float(projected.sum())
    if abs(residual) > 1e-12:
        slack = upper - projected if residual > 0 else projected - lower
        for i in np.argsort(slack)[::-1]:
            adjustment = np.sign(residual) * min(abs(residual), float(slack[i]))
            projected[i] += adjustment
            residual -= adjustment
            if abs(residual) <= 1e-12:
                break
    return projected


def feasible_population(n: int, seed: int) -> np.ndarray:
    """Create DE's initial population from feasible full-length vectors."""
    dimension = n - 1
    if dimension == 0:
        return np.empty((0, 0), dtype=float)
    rng = np.random.default_rng(seed)
    population_size = max(5, POPSIZE * dimension)
    rows = np.empty((population_size, dimension), dtype=float)
    for row in range(population_size):
        full = rng.uniform(LENGTH_MIN_DUAL, LENGTH_MAX_DUAL, size=n)
        full = project_to_bounded_sum(
            full, BASELINE_TOTAL_LENGTH, LENGTH_MIN_DUAL, LENGTH_MAX_DUAL
        )
        # The final component is reconstructed by the objective, so retain only
        # the independent components from this feasible full vector.
        rows[row] = full[:-1]
    return rows


def optimize_fixed_n(n: int) -> dict[str, Any]:
    """Optimize link lengths for one fixed, feasible joint count."""
    if not is_feasible_count(n):
        return {
            "N": n,
            "status": "infeasible",
            "reason": (
                f"Need N*l_min <= L0 <= N*l_max; got "
                f"[{n * LENGTH_MIN_DUAL:.12g}, {n * LENGTH_MAX_DUAL:.12g}] "
                f"for L0={BASELINE_TOTAL_LENGTH:.12g}."
            ),
        }

    if n == 1:
        lengths = reconstruct_lengths(1, [])
        valid, reason = validate_lengths(lengths, 1)
        if not valid:
            return {"N": n, "status": "infeasible", "reason": reason}
        area = evaluate_area(lengths)
        return make_result(
            n, lengths, area,
            restarts=[{
                "seed": None,
                "optimizer_success": True,
                "message": "Direct evaluation; N=1 has no free length variables.",
                "nfev": 1,
                "best_objective": -area,
                "best_area": area,
                "feasible_evaluations": 1,
                "evaluation_errors": 0,
            }],
            optimizer_success=True,
            message="Direct evaluation; no continuous optimization needed.",
            nfev=1,
        )

    # Degenerate feasible sets (all links forced to a bound) need evaluation,
    # not a nominal zero-width differential-evolution search.
    if (
        abs(BASELINE_TOTAL_LENGTH - n * LENGTH_MIN_DUAL) <= LENGTH_TOL
        or abs(BASELINE_TOTAL_LENGTH - n * LENGTH_MAX_DUAL) <= LENGTH_TOL
    ):
        lengths = np.full(n, BASELINE_TOTAL_LENGTH / n, dtype=float)
        valid, reason = validate_lengths(lengths, n)
        if not valid:
            return {"N": n, "status": "infeasible", "reason": reason}
        area = evaluate_area(lengths)
        return make_result(
            n, lengths, area,
            restarts=[{
                "seed": None,
                "optimizer_success": True,
                "message": "Direct evaluation; feasible length set is a singleton.",
                "nfev": 1,
                "best_objective": -area,
                "best_area": area,
                "feasible_evaluations": 1,
                "evaluation_errors": 0,
            }],
            optimizer_success=True,
            message="Direct evaluation; feasible length set is a singleton.",
            nfev=1,
        )

    lower_sum = BASELINE_TOTAL_LENGTH - LENGTH_MAX_DUAL
    upper_sum = BASELINE_TOTAL_LENGTH - LENGTH_MIN_DUAL
    constraint = LinearConstraint(
        np.ones((1, n - 1)), lower_sum, upper_sum
    )
    bounds = [(LENGTH_MIN_DUAL, LENGTH_MAX_DUAL)] * (n - 1)
    restart_records: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None

    for restart in range(max(1, INNER_RESTARTS)):
        seed = SEED + (n - N_MIN) * max(1, INNER_RESTARTS) + restart
        run_best: dict[str, Any] = {"area": -np.inf, "lengths": None}
        errors: list[str] = []
        feasible_evaluations = 0

        def objective(independent: np.ndarray) -> float:
            nonlocal feasible_evaluations
            lengths = reconstruct_lengths(n, independent)
            valid, why = validate_lengths(lengths, n)
            if not valid:
                return 1e100
            try:
                area = evaluate_area(lengths)
            except Exception as exc:  # Keep the run diagnosable in its JSON.
                if len(errors) < 3:
                    errors.append(f"{type(exc).__name__}: {exc}")
                return 1e100
            if not np.isfinite(area):
                if len(errors) < 3:
                    errors.append("area evaluation returned a non-finite value")
                return 1e100
            feasible_evaluations += 1
            if area > run_best["area"]:
                run_best["area"] = area
                run_best["lengths"] = lengths.copy()
            return -area

        try:
            result = differential_evolution(
                objective,
                bounds=bounds,
                constraints=(constraint,),
                seed=seed,
                maxiter=MAXITER,
                popsize=POPSIZE,
                tol=DE_TOL,
                init=feasible_population(n, seed),
                polish=False,
                updating="immediate",
            )
            result_lengths = reconstruct_lengths(n, result.x)
            result_valid, result_reason = validate_lengths(result_lengths, n)
            if result_valid and run_best["lengths"] is None:
                # Normally result.x was already seen by objective(). This is a
                # fallback for SciPy versions that return an untracked point.
                result_area = evaluate_area(result_lengths)
                if np.isfinite(result_area):
                    run_best["area"] = result_area
                    run_best["lengths"] = result_lengths
            else:
                result_reason = f"optimizer returned an invalid candidate: {result_reason}"
            record = {
                "seed": int(seed),
                "optimizer_success": bool(result.success),
                "message": str(result.message),
                "nfev": int(result.nfev),
                "best_objective": (
                    float(-run_best["area"]) if run_best["lengths"] is not None else None
                ),
                "best_area": (
                    float(run_best["area"]) if run_best["lengths"] is not None else None
                ),
                "feasible_evaluations": int(feasible_evaluations),
                "evaluation_errors": len(errors),
                "evaluation_error_details": errors,
            }
            if not result_valid:
                record["candidate_validation_error"] = result_reason
            if run_best["lengths"] is not None and (
                best is None or run_best["area"] > best["area"]
            ):
                best = {
                    "area": float(run_best["area"]),
                    "lengths": run_best["lengths"].copy(),
                    "optimizer_success": bool(result.success),
                    "message": str(result.message),
                    "nfev": int(result.nfev),
                }
        except Exception as exc:
            record = {
                "seed": int(seed),
                "optimizer_success": False,
                "message": f"{type(exc).__name__}: {exc}",
                "nfev": 0,
                "best_objective": None,
                "best_area": None,
                "feasible_evaluations": int(feasible_evaluations),
                "evaluation_errors": len(errors),
                "evaluation_error_details": errors,
            }
        restart_records.append(record)

    if best is None:
        return {
            "N": n,
            "status": "failed",
            "reason": "No valid finite-area candidate was returned by any restart.",
            "restarts": restart_records,
            "restart_count": len(restart_records),
        }

    # Validate and freshly recompute the retained candidate for consistency.
    valid, reason = validate_lengths(best["lengths"], n)
    if not valid:
        return {
            "N": n,
            "status": "failed",
            "reason": f"Best candidate failed final validation: {reason}",
            "restarts": restart_records,
            "restart_count": len(restart_records),
        }
    area_check = evaluate_area(best["lengths"])
    if not np.isfinite(area_check) or not np.isclose(
        area_check, best["area"], rtol=AREA_TOL, atol=AREA_TOL
    ):
        return {
            "N": n,
            "status": "failed",
            "reason": "Independent final area recomputation did not match the retained area.",
            "recomputed_area": float(area_check),
            "retained_area": float(best["area"]),
            "restarts": restart_records,
            "restart_count": len(restart_records),
        }
    return make_result(
        n,
        best["lengths"],
        area_check,
        restarts=restart_records,
        optimizer_success=best["optimizer_success"],
        message=best["message"],
        nfev=best["nfev"],
    )


def make_result(
    n: int,
    lengths: np.ndarray,
    area: float,
    *,
    restarts: list[dict[str, Any]],
    optimizer_success: bool,
    message: str,
    nfev: int,
) -> dict[str, Any]:
    """Build JSON-ready diagnostics for one independently optimized N."""
    lengths = np.asarray(lengths, dtype=float)
    residual = float(lengths.sum() - BASELINE_TOTAL_LENGTH)
    return {
        "N": int(n),
        "status": "optimized",
        "lengths": lengths.tolist(),
        "area": float(area),
        "improvement_percent": float(100.0 * (area / _baseline_area - 1.0)),
        "length_sum": float(lengths.sum()),
        "length_min": float(lengths.min()),
        "length_max": float(lengths.max()),
        "total_length_residual": residual,
        "optimizer_success": bool(optimizer_success),
        "optimizer_message": str(message),
        "nfev": int(nfev),
        "restart_count": len(restarts),
        "restarts": restarts,
    }


_baseline_area = float("nan")


def run_nested_optimization() -> dict[str, Any]:
    """Run each feasible fixed-N problem, then compare their best areas."""
    global _baseline_area
    _baseline_area = evaluate_area(np.asarray(BASELINE_LENGTHS, dtype=float))
    if not np.isfinite(_baseline_area) or _baseline_area <= 0.0:
        raise RuntimeError(f"Invalid baseline reachable-set area: {_baseline_area}")

    records: list[dict[str, Any]] = []
    infeasible_counts: list[dict[str, Any]] = []
    print("Nested fixed-N reachability optimization")
    print(
        f"L0={BASELINE_TOTAL_LENGTH:g} m; link bounds="
        f"[{LENGTH_MIN_DUAL:g}, {LENGTH_MAX_DUAL:g}] m; "
        f"per-motor torque limits from actuator_limits(N)"
    )
    print(f"Baseline morphology={np.asarray(BASELINE_LENGTHS)}; area={_baseline_area:.8e}")

    for n in range(N_MIN, N_MAX + 1):
        record = optimize_fixed_n(n)
        records.append(record)
        if record["status"] == "infeasible":
            infeasible_counts.append({"N": n, "reason": record["reason"]})
            print(f"N={n}: infeasible — {record['reason']}")
        elif record["status"] == "failed":
            print(f"N={n}: failed — {record['reason']}")
        else:
            print(
                f"N={n}: area={record['area']:.8e}, "
                f"gain={record['improvement_percent']:.3f}%, "
                f"lengths={np.array2string(np.asarray(record['lengths']), precision=5)}, "
                f"converged={record['optimizer_success']}"
            )

    successful = [record for record in records if record["status"] == "optimized"]
    best = max(successful, key=lambda item: item["area"]) if successful else None
    if best:
        print(
            f"\nBest across feasible N: N={best['N']}, "
            f"area={best['area']:.8e}, lengths={best['lengths']}, "
            f"gain={best['improvement_percent']:.3f}%"
        )
    else:
        print("\nNo feasible joint count produced a valid optimization result.")

    print("\nPer-N best feasible areas")
    print(f"{'N':>4} {'status':>12} {'area':>16} {'gain (%)':>12} {'success':>9}")
    for record in records:
        if record["status"] == "optimized":
            print(
                f"{record['N']:>4} {'optimized':>12} {record['area']:>16.8e} "
                f"{record['improvement_percent']:>12.3f} {str(record['optimizer_success']):>9}"
            )
        else:
            print(f"{record['N']:>4} {record['status']:>12} {'—':>16} {'—':>12} {'—':>9}")

    return {
        "objective_definition": (
            "Maximize the finite-horizon area of the bounded-input reachable set "
            "in the end-effector position-velocity plane [x_EE, xdot_EE]."
        ),
        "optimization_method": (
            "Nested computational decomposition: independent differential-evolution "
            "link-length optimization for every feasible integer N, followed by "
            "exhaustive comparison of the optimized areas. Both levels use the same "
            "area objective; this is not a morphology-controller bilevel optimization."
        ),
        "global_optimality_note": (
            "Differential evolution is stochastic; successful termination is numerical "
            "convergence information, not proof of a global optimum."
        ),
        "settings": {
            "N_min": int(N_MIN),
            "N_max": int(N_MAX),
            "baseline_lengths": np.asarray(BASELINE_LENGTHS, dtype=float).tolist(),
            "fixed_total_length": float(BASELINE_TOTAL_LENGTH),
            "link_bounds": [float(LENGTH_MIN_DUAL), float(LENGTH_MAX_DUAL)],
            "total_mass": float(TOTAL_MASS),
            "actuator_limit_policy": "design.actuator_limits(N)",
            "horizon": float(T),
            "reachable_set_directions": int(NTHETA),
            "time_quadrature_points": int(NT),
            "differential_evolution": {
                "maxiter": int(MAXITER),
                "popsize": int(POPSIZE),
                "tol": float(DE_TOL),
                "polish": False,
                "inner_restarts_per_N": int(max(1, INNER_RESTARTS)),
                "seed_convention": "SEED + (N - N_MIN) * restart_count + restart_index",
                "base_seed": int(SEED),
                "initialization": "feasible projected population",
                "constraint": "L0 - l_max <= sum(l_1..l_(N-1)) <= L0 - l_min",
            },
            "validation_tolerance": float(LENGTH_TOL),
        },
        "baseline_area": float(_baseline_area),
        "best_overall": best,
        "results_by_N": records,
        "infeasible_joint_counts": infeasible_counts,
    }


def main() -> None:
    payload = run_nested_optimization()
    JSON_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    print(f"\nSaved results to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
