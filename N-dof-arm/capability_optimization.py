"""Minimize normalized design cost subject to a reachable-area requirement."""
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

from design import (
    N_MIN, N_MAX,
    BASELINE_LENGTHS,
    BASELINE_TOTAL_LENGTH,
    TOTAL_MASS,
    TOTAL_LENGTH_MIN,
    TOTAL_LENGTH_MAX,
    LENGTH_MIN,
    LENGTH_MAX,
    REQUIRED_IMPROVEMENT,
    actuator_limits,
    design_cost,
    feasible_length_interval,
    length_bounds,
)
from model import robot_matrices, output_matrix
from reachability import reachable_set

T = 2.0
NT = 250
NTHETA = 120
N_STARTS = 4
HERE = Path(__file__).resolve().parent


def evaluate_area(lengths):
    n = len(lengths)
    A, B, *_ = robot_matrices(lengths, total_mass=TOTAL_MASS)
    C = output_matrix(lengths)
    return float(reachable_set(
        A, B, C, actuator_limits(n), T=T, ntheta=NTHETA, nt=NT
    )[3])


def project_to_sum(x, target):
    left, right = -2.0 * LENGTH_MAX, 2.0 * LENGTH_MAX
    for _ in range(60):
        shift = 0.5 * (left + right)
        candidate = np.clip(x + shift, LENGTH_MIN, LENGTH_MAX)
        if candidate.sum() < target:
            left = shift
        else:
            right = shift
    return np.clip(x + 0.5 * (left + right), LENGTH_MIN, LENGTH_MAX)


def starts_for_n(n):
    lo, hi = feasible_length_interval(n)
    starts = [np.full(n, np.clip(BASELINE_TOTAL_LENGTH / n, lo, hi))]
    starts.extend([np.full(n, lo), np.full(n, hi)])
    rng = np.random.default_rng(300 + n)
    for _ in range(N_STARTS):
        x = rng.uniform(LENGTH_MIN, LENGTH_MAX, size=n)
        x = project_to_sum(x, np.clip(x.sum(), TOTAL_LENGTH_MIN, TOTAL_LENGTH_MAX))
        starts.append(x)
    unique = []
    for x in starts:
        if not any(np.allclose(x, old) for old in unique):
            unique.append(x)
    return unique[:N_STARTS]


def maximize_area(n):
    """Estimate the best area for N before enforcing the target constraint."""
    constraints = [
        {"type": "ineq", "fun": lambda x: np.sum(x) - TOTAL_LENGTH_MIN},
        {"type": "ineq", "fun": lambda x: TOTAL_LENGTH_MAX - np.sum(x)},
    ]
    best = None
    for x0 in starts_for_n(n):
        result = minimize(
            lambda x: -evaluate_area(x), x0, method="SLSQP",
            bounds=length_bounds(n), constraints=constraints,
            options={"maxiter": 80, "ftol": 1e-7, "disp": False},
        )
        candidates = [x0, result.x]
        for x in candidates:
            if not (np.all(np.isfinite(x)) and TOTAL_LENGTH_MIN - 1e-7 <= np.sum(x) <= TOTAL_LENGTH_MAX + 1e-7):
                continue
            area = evaluate_area(x)
            if best is None or area > best["area"]:
                best = {"lengths": np.asarray(x, dtype=float), "area": area}
    return best


def minimum_cost_design(n, target_area, feasible_seed):
    constraints = [
        {"type": "ineq", "fun": lambda x: evaluate_area(x) - target_area},
        {"type": "ineq", "fun": lambda x: np.sum(x) - TOTAL_LENGTH_MIN},
        {"type": "ineq", "fun": lambda x: TOTAL_LENGTH_MAX - np.sum(x)},
    ]
    result = minimize(
        design_cost, feasible_seed, method="SLSQP",
        bounds=length_bounds(n), constraints=constraints,
        options={"maxiter": 100, "ftol": 1e-7, "disp": False},
    )
    candidates = [(feasible_seed, "feasible capability seed"), (result.x, str(result.message))]
    valid = []
    for x, status in candidates:
        area = evaluate_area(x)
        if (
            np.all(np.isfinite(x))
            and area >= target_area - 1e-6
            and TOTAL_LENGTH_MIN - 1e-7 <= np.sum(x) <= TOTAL_LENGTH_MAX + 1e-7
        ):
            valid.append({"lengths": np.asarray(x), "area": area, "cost": design_cost(x), "solver_status": status})
    return min(valid, key=lambda item: item["cost"]) if valid else None


def main():
    baseline_area = evaluate_area(BASELINE_LENGTHS)
    target_area = (1.0 + REQUIRED_IMPROVEMENT) * baseline_area
    print(f"Baseline reachable area = {baseline_area:.8e}")
    print(f"Required reachable area = {target_area:.8e} (+{100 * REQUIRED_IMPROVEMENT:.1f}%)")
    results = []

    for n in range(N_MIN, N_MAX + 1):
        max_result = maximize_area(n)
        if max_result is None:
            results.append({"N": n, "status": "search_failed"})
            print(f"N={n}: could not find a feasible morphology.")
            continue
        max_gain = 100.0 * (max_result["area"] / baseline_area - 1.0)
        if max_result["area"] < target_area - 1e-6:
            record = {
                "N": n,
                "status": "target_infeasible_in_search",
                "best_found_lengths": max_result["lengths"].tolist(),
                "max_area_found": max_result["area"],
                "max_improvement_found_percent": max_gain,
                "required_area": target_area,
            }
            results.append(record)
            print(
                f"N={n}: best area gain={max_gain:.3f}%; "
                f"target +{100 * REQUIRED_IMPROVEMENT:.1f}% not reached."
            )
            continue

        optimum = minimum_cost_design(n, target_area, max_result["lengths"])
        if optimum is None:
            results.append({
                "N": n,
                "status": "target_feasible_but_cost_search_failed",
                "best_found_lengths": max_result["lengths"].tolist(),
                "max_area_found": max_result["area"],
                "required_area": target_area,
            })
            print(f"N={n}: capability target found, but minimum-cost search did not return a feasible point.")
            continue

        improvement = 100.0 * (optimum["area"] / baseline_area - 1.0)
        record = {
            "N": n,
            "status": "feasible",
            "lengths": optimum["lengths"].tolist(),
            "area": optimum["area"],
            "improvement": improvement,
            "cost": optimum["cost"],
            "solver_status": optimum["solver_status"],
        }
        results.append(record)
        print(
            f"N={n} | lengths={np.round(optimum['lengths'], 5)} | "
            f"area={optimum['area']:.6e} | gain={improvement:.3f}% | "
            f"design cost={optimum['cost']:.4f}"
        )

    feasible = [r for r in results if r.get("status") == "feasible"]
    if feasible:
        best = min(feasible, key=lambda r: r["cost"])
        print(f"\nLowest-cost target-feasible design: N={best['N']}, lengths={best['lengths']}")
    output = HERE / "capability_optimization_results.json"
    output.write_text(json.dumps(results, indent=2))
    print(f"Saved per-N feasibility and design results to {output.name}")


if __name__ == "__main__":
    main()
