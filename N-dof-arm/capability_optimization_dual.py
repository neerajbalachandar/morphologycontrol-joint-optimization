"""Joint mixed-integer capability optimization over N and link lengths.

Minimize a dimensionless design-change cost subject to a target improvement
in bounded-input reachable-set area, fixed total length, and relaxed per-link
bounds that make each configured joint count feasible. Differential evolution
searches N and active link lengths simultaneously.
"""
import json
from pathlib import Path

import numpy as np
from scipy.optimize import differential_evolution

from model import robot_matrices, output_matrix
from reachability import reachable_set
from design import (
    N_MIN, N_MAX, N_BASELINE, BASELINE_LENGTHS, BASELINE_TOTAL_LENGTH,
    TOTAL_MASS, TAU_BUDGET, actuator_limits,
)

T = 2.0
NT = 250
NTHETA = 120
REQUIRED_IMPROVEMENT = 0.10
SEED = 11
MAXITER = 100
POPSIZE = 12
TOL = 1e-10
HERE = Path(__file__).resolve().parent
JSON_DIR = HERE / "json"

# Relaxed only for this dual-N study. The lower bound is below L0/N_MAX so
# N=N_MAX has a nonzero-volume feasible region, rather than one isolated point.
# Replace with real per-link hardware bounds for a physical design study.
LENGTH_MIN_DUAL = min(0.2, BASELINE_TOTAL_LENGTH / N_MAX)
LENGTH_MAX_DUAL = BASELINE_TOTAL_LENGTH / N_MIN


def project_box_sum(values, target):
    left, right = -2.0 * LENGTH_MAX_DUAL, 2.0 * LENGTH_MAX_DUAL
    for _ in range(70):
        shift = 0.5 * (left + right)
        candidate = np.clip(values + shift, LENGTH_MIN_DUAL, LENGTH_MAX_DUAL)
        if candidate.sum() < target:
            left = shift
        else:
            right = shift
    return np.clip(values + 0.5 * (left + right), LENGTH_MIN_DUAL, LENGTH_MAX_DUAL)


def feasible_initial_population():
    """Seed the joint DE population with feasible samples for every N."""
    dimension = N_MAX
    population_size = max(5, POPSIZE * dimension)
    rng = np.random.default_rng(SEED)
    rows = []
    for index in range(population_size):
        n = N_MIN + index % (N_MAX - N_MIN + 1)
        if n == 1:
            lengths = np.array([BASELINE_TOTAL_LENGTH])
        else:
            raw = rng.uniform(LENGTH_MIN_DUAL, LENGTH_MAX_DUAL, size=n)
            lengths = project_box_sum(raw, BASELINE_TOTAL_LENGTH)
        row = np.full(dimension, LENGTH_MIN_DUAL)
        row[0] = n
        row[1:n] = lengths[:-1]
        rows.append(row)
    return np.asarray(rows)


def active_lengths(n, z):
    if n == 1:
        return np.array([BASELINE_TOTAL_LENGTH], dtype=float)
    free = np.asarray(z[:n - 1], dtype=float)
    return np.r_[free, BASELINE_TOTAL_LENGTH - np.sum(free)]


def valid_lengths(lengths):
    return (
        np.all(np.isfinite(lengths))
        and np.all(lengths >= LENGTH_MIN_DUAL - TOL)
        and np.all(lengths <= LENGTH_MAX_DUAL + TOL)
        and abs(np.sum(lengths) - BASELINE_TOTAL_LENGTH) <= TOL
    )


def evaluate_area(lengths):
    n = len(lengths)
    A, B, *_ = robot_matrices(lengths, total_mass=TOTAL_MASS)
    C = output_matrix(lengths)
    return float(reachable_set(
        A, B, C, actuator_limits(n), T=T, ntheta=NTHETA, nt=NT
    )[3])


def design_change_cost(lengths):
    """Penalize changing actuator count and nonuniform morphology.

    The count term is normalized to the baseline actuator count. The shape
    term is the normalized L1 distance from equal link lengths at this N.
    With no BOM or actuator prices supplied, this is a relative design-change
    metric, not a monetary manufacturing cost.
    """
    n = len(lengths)
    count_change = abs(n - N_BASELINE) / N_BASELINE
    equal_lengths = BASELINE_TOTAL_LENGTH / n
    morphology_change = np.sum(np.abs(np.asarray(lengths) - equal_lengths)) / BASELINE_TOTAL_LENGTH
    return float(count_change + morphology_change)


def main():
    baseline_area = evaluate_area(BASELINE_LENGTHS)
    target_area = (1.0 + REQUIRED_IMPROVEMENT) * baseline_area
    bounds = [(N_MIN, N_MAX)] + [(LENGTH_MIN_DUAL, LENGTH_MAX_DUAL)] * (N_MAX - 1)
    best_evaluated = {}

    def objective(v):
        n = int(round(v[0]))
        lengths = active_lengths(n, v[1:])
        violation = np.maximum(LENGTH_MIN_DUAL - lengths, 0.0) + np.maximum(lengths - LENGTH_MAX_DUAL, 0.0)
        if not valid_lengths(lengths):
            return 1e6 + 1e6 * float(np.sum(violation**2))
        area = evaluate_area(lengths)
        cost = design_change_cost(lengths)
        entry = best_evaluated.get(n)
        if entry is None or area > entry["max_area_evaluated"]:
            entry = {
                "N": n,
                "max_area_evaluated": area,
                "max_area_lengths": lengths.tolist(),
                "target_feasible_candidate": None,
            }
            best_evaluated[n] = entry
        if area >= target_area and (
            entry["target_feasible_candidate"] is None
            or cost < entry["target_feasible_candidate"]["cost"]
        ):
            entry["target_feasible_candidate"] = {
                "lengths": lengths.tolist(),
                "area": area,
                "cost": cost,
            }
        # A large shortfall penalty makes every target-feasible design
        # preferable to an infeasible candidate whenever one exists.
        shortfall = max(target_area - area, 0.0) / target_area
        return cost + 1e3 * shortfall

    print("Joint N + link-length capability optimization")
    print(f"Fixed total length={BASELINE_TOTAL_LENGTH:g} m; dual-study link bounds=[{LENGTH_MIN_DUAL:g}, {LENGTH_MAX_DUAL:g}] m")
    print(f"Baseline area={baseline_area:.8e}; target area={target_area:.8e} (+{100*REQUIRED_IMPROVEMENT:g}%)")
    print("Cost = normalized actuator-count change + normalized link-shape change")
    result = differential_evolution(
        objective,
        bounds=bounds,
        integrality=[True] + [False] * (N_MAX - 1),
        seed=SEED,
        maxiter=MAXITER,
        popsize=POPSIZE,
        init=feasible_initial_population(),
        tol=1e-3,
        polish=False,
        disp=True,
    )

    n_star = int(round(result.x[0]))
    lengths_star = active_lengths(n_star, result.x[1:])
    if not valid_lengths(lengths_star):
        raise RuntimeError(f"Optimizer returned an infeasible morphology: N={n_star}, lengths={lengths_star}")
    area_star = evaluate_area(lengths_star)
    improvement = 100.0 * (area_star / baseline_area - 1.0)
    feasible_target = area_star >= target_area - 1e-8

    # Include final best in the diagnostics archive even if it was the last
    # population update and was already evaluated by the solver.
    if n_star not in best_evaluated or area_star > best_evaluated[n_star]["max_area_evaluated"]:
        best_evaluated[n_star] = {
            "N": n_star,
            "max_area_evaluated": area_star,
            "max_area_lengths": lengths_star.tolist(),
            "target_feasible_candidate": None,
        }
    if feasible_target:
        item = best_evaluated[n_star]
        old = item["target_feasible_candidate"]
        final_cost = design_change_cost(lengths_star)
        if old is None or final_cost < old["cost"]:
            item["target_feasible_candidate"] = {
                "lengths": lengths_star.tolist(),
                "area": area_star,
                "cost": final_cost,
            }

    best = {
        "N": n_star,
        "lengths": lengths_star.tolist(),
        "area": float(area_star),
        "improvement_percent": float(improvement),
        "design_change_cost": design_change_cost(lengths_star),
        "target_met": bool(feasible_target),
    }
    payload = {
        "objective": "minimize normalized design change subject to target reachable-area improvement",
        "baseline_lengths": BASELINE_LENGTHS.tolist(),
        "baseline_area": float(baseline_area),
        "target_area": float(target_area),
        "required_improvement_percent": 100.0 * REQUIRED_IMPROVEMENT,
        "fixed_total_length": float(BASELINE_TOTAL_LENGTH),
        "link_bounds": [LENGTH_MIN_DUAL, LENGTH_MAX_DUAL],
        "best": best,
        "best_evaluated_candidate_by_N": [best_evaluated[n] for n in sorted(best_evaluated)],
        "solver_success": bool(result.success),
        "solver_message": str(result.message),
        "function_evaluations": int(result.nfev),
    }
    JSON_DIR.mkdir(parents=True, exist_ok=True)
    output = JSON_DIR / "capability_optimization_joint_results.json"
    output.write_text(json.dumps(payload, indent=2))
    print(f"\nBest design: N={n_star}, lengths={lengths_star}")
    print(f"Area={area_star:.8e}, gain={improvement:.3f}%, design-change cost={best['design_change_cost']:.5f}")
    print(f"Target met: {feasible_target}; solver: {result.message}")
    print(f"Saved results to {output}")


if __name__ == "__main__":
    main()
