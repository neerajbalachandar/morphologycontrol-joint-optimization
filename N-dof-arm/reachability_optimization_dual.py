"""Joint mixed-integer optimization of joint count N and link lengths.

Maximize finite-time bounded-input reachable-set area subject to fixed total
length and relaxed per-link bounds that make every configured N feasible.
Differential evolution searches N and the active lengths simultaneously.
"""
import json
from pathlib import Path

import numpy as np
from scipy.optimize import differential_evolution

from model import robot_matrices, output_matrix
from reachability import reachable_set
from design import (
    N_MIN, N_MAX, BASELINE_LENGTHS, BASELINE_TOTAL_LENGTH,
    TOTAL_MASS, TAU_BUDGET, actuator_limits,
)

T = 2.0
NT = 250
NTHETA = 120
SEED = 7
MAXITER = 100
POPSIZE = 12
TOL = 1e-10
HERE = Path(__file__).resolve().parent
JSON_DIR = HERE / "json"

# Relaxed only for this dual-N study. The lower bound is below L0/N_MAX so
# N=N_MAX has a nonzero-volume feasible region, rather than one isolated point.
# Replace these with hardware bounds when those are known.
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


def main():
    baseline_area = evaluate_area(BASELINE_LENGTHS)
    bounds = [(N_MIN, N_MAX)] + [(LENGTH_MIN_DUAL, LENGTH_MAX_DUAL)] * (N_MAX - 1)
    archive = {}

    def objective(v):
        n = int(round(v[0]))
        lengths = active_lengths(n, v[1:])
        violation = np.maximum(LENGTH_MIN_DUAL - lengths, 0.0) + np.maximum(lengths - LENGTH_MAX_DUAL, 0.0)
        if not valid_lengths(lengths):
            return 1e3 + 1e3 * float(np.sum(violation**2))
        area = evaluate_area(lengths)
        if n not in archive or area > archive[n]["area"]:
            archive[n] = {"N": n, "lengths": lengths.tolist(), "area": area}
        return -area

    print("Joint N + link-length reachability optimization")
    print(f"Fixed total length={BASELINE_TOTAL_LENGTH:g} m; dual-study link bounds=[{LENGTH_MIN_DUAL:g}, {LENGTH_MAX_DUAL:g}] m")
    print(f"Total mass={TOTAL_MASS:g}; total torque budget={TAU_BUDGET:g}; baseline area={baseline_area:.8e}")
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
        raise RuntimeError(f"Optimizer returned an infeasible design: N={n_star}, lengths={lengths_star}")
    area_star = evaluate_area(lengths_star)
    improvement = 100.0 * (area_star / baseline_area - 1.0)

    # Keep the best feasible point encountered for each N during the joint
    # search. These are diagnostics, not independent per-N optima.
    if n_star not in archive or area_star > archive[n_star]["area"]:
        archive[n_star] = {"N": n_star, "lengths": lengths_star.tolist(), "area": area_star}
    best = {
        "N": n_star,
        "lengths": lengths_star.tolist(),
        "area": float(area_star),
        "improvement_percent": float(improvement),
    }
    payload = {
        "objective": "maximize finite-time bounded-input reachable-set area",
        "baseline_lengths": BASELINE_LENGTHS.tolist(),
        "baseline_area": float(baseline_area),
        "fixed_total_length": float(BASELINE_TOTAL_LENGTH),
        "link_bounds": [LENGTH_MIN_DUAL, LENGTH_MAX_DUAL],
        "best": best,
        "best_evaluated_candidate_by_N": [archive[n] for n in sorted(archive)],
        "solver_success": bool(result.success),
        "solver_message": str(result.message),
        "function_evaluations": int(result.nfev),
    }
    JSON_DIR.mkdir(parents=True, exist_ok=True)
    output = JSON_DIR / "reachability_optimization_joint_results.json"
    output.write_text(json.dumps(payload, indent=2))
    print(f"\nBest design: N={n_star}, lengths={lengths_star}, area={area_star:.8e}, gain={improvement:.3f}%")
    print(f"Solver: {result.message}")
    print(f"Saved results to {output}")


if __name__ == "__main__":
    main()
