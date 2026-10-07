"""Task tracking co-design over feasible link lengths for each joint count."""
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

from design import (
    N_MIN, N_MAX,
    BASELINE_TOTAL_LENGTH,
    TOTAL_MASS,
    TOTAL_LENGTH_MIN,
    TOTAL_LENGTH_MAX,
    LENGTH_MIN,
    LENGTH_MAX,
    TASK_DESIGN_WEIGHT,
    TASK_EFFORT_WEIGHT,
    REFERENCE_AMPLITUDE,
    MAX_LINEARIZED_ANGLE,
    actuator_limits,
    design_cost,
    feasible_length_interval,
    length_bounds,
)
from model import robot_matrices
from task import simulate_task, tracking_metrics

T = 5.0
DT = 0.04
N_STARTS = 4
MAX_ITER = 60
HERE = Path(__file__).resolve().parent


def evaluate(lengths):
    """Return dimensionless task score and interpretable components."""
    lengths = np.asarray(lengths, dtype=float)
    n = len(lengths)
    A, B, *_ = robot_matrices(lengths, total_mass=TOTAL_MASS)
    Q = np.diag(np.r_[100.0 * np.ones(n), 10.0 * np.ones(n)])
    R = np.eye(n)
    limits = actuator_limits(n)

    t, X, U = simulate_task(A, B, Q, R, T=T, dt=DT, u_max=limits)
    metrics = tracking_metrics(t, X, U, n, Q, R)

    # Normalize tracking by reference amplitude and joint count so scores
    # remain comparable as N changes.
    tracking_fraction = metrics["rms_error"] / (REFERENCE_AMPLITUDE * np.sqrt(n))

    # RMS actuator utilization, integrated over the zero-order-held commands.
    utilization_sq = np.mean((U[:-1] / limits[None, :]) ** 2, axis=1)
    effort_fraction = np.sqrt(np.sum(utilization_sq * np.diff(t)) / T)
    resource_cost = design_cost(lengths)
    score = (
        tracking_fraction**2
        + TASK_EFFORT_WEIGHT * effort_fraction**2
        + TASK_DESIGN_WEIGHT * resource_cost
    )

    metrics.update({
        "tracking_fraction": float(tracking_fraction),
        "effort_fraction": float(effort_fraction),
        "design_cost": float(resource_cost),
        "score": float(score),
        "max_abs_joint_angle": float(np.max(np.abs(X[:, :n]))),
    })
    return float(score), metrics


def project_to_length_sum(lengths, target):
    """Project onto box bounds and a chosen aggregate-length plane."""
    lo_shift, hi_shift = -2.0 * LENGTH_MAX, 2.0 * LENGTH_MAX
    for _ in range(60):
        shift = 0.5 * (lo_shift + hi_shift)
        candidate = np.clip(lengths + shift, LENGTH_MIN, LENGTH_MAX)
        if candidate.sum() < target:
            lo_shift = shift
        else:
            hi_shift = shift
    return np.clip(lengths + 0.5 * (lo_shift + hi_shift), LENGTH_MIN, LENGTH_MAX)


def make_starts(n):
    lo, hi = feasible_length_interval(n)
    starts = [np.full(n, np.clip(BASELINE_TOTAL_LENGTH / n, lo, hi))]
    starts.extend([np.full(n, lo), np.full(n, hi)])

    rng = np.random.default_rng(100 + n)
    for _ in range(N_STARTS):
        x = rng.uniform(LENGTH_MIN, LENGTH_MAX, size=n)
        target = np.clip(x.sum(), TOTAL_LENGTH_MIN, TOTAL_LENGTH_MAX)
        starts.append(project_to_length_sum(x, target))

    unique = []
    for x in starts:
        if not any(np.allclose(x, old) for old in unique):
            unique.append(x)
    return unique[:N_STARTS]


def optimize_fixed_n(n):
    bounds = length_bounds(n)
    constraints = [
        {"type": "ineq", "fun": lambda x: np.sum(x) - TOTAL_LENGTH_MIN},
        {"type": "ineq", "fun": lambda x: TOTAL_LENGTH_MAX - np.sum(x)},
        {
            "type": "ineq",
            "fun": lambda x: MAX_LINEARIZED_ANGLE - cache_eval(x)["max_abs_joint_angle"],
        },
    ]

    cache = {}

    def objective(x):
        key = np.round(np.asarray(x, dtype=float), decimals=10).tobytes()
        if key not in cache:
            cache[key] = evaluate(x)
        return cache[key][0]

    def cache_eval(x):
        key = np.round(np.asarray(x, dtype=float), decimals=10).tobytes()
        if key not in cache:
            cache[key] = evaluate(x)
        return cache[key][1]

    best = None
    statuses = []
    for start_id, x0 in enumerate(make_starts(n), start=1):
        # Keep the feasible start as a fallback if a local optimizer stops
        # early on the nonsmooth torque-saturation boundary.
        candidates = [(x0, "feasible initial design")]
        result = minimize(
            objective,
            x0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": MAX_ITER, "ftol": 1e-5, "disp": False},
        )
        statuses.append(f"start {start_id}: {result.message}")
        candidates.append((result.x, str(result.message)))

        for x, status in candidates:
            x = np.asarray(x, dtype=float)
            feasible = (
                np.all(np.isfinite(x))
                and np.all(x >= LENGTH_MIN - 1e-7)
                and np.all(x <= LENGTH_MAX + 1e-7)
                and TOTAL_LENGTH_MIN - 1e-7 <= np.sum(x) <= TOTAL_LENGTH_MAX + 1e-7
                and cache_eval(x)["max_abs_joint_angle"] <= MAX_LINEARIZED_ANGLE + 1e-7
            )
            if not feasible:
                continue
            score, metrics = evaluate(x)
            if best is None or score < best["score"]:
                best = {
                    "N": n,
                    "lengths": x.copy(),
                    "score": score,
                    "metrics": metrics,
                    "solver_status": status,
                }
    return best, statuses


def main():
    results = []
    print("Task tracking co-design (sampled LQT, torque-limited)")
    print(f"Horizon={T:g} s, sample interval={DT:g} s, torque budget={2.0:g} N m total")
    print(f"Linearized joint-angle envelope: ±{MAX_LINEARIZED_ANGLE:g} rad")
    print("Design cost is normalized to baseline; actuator count has 2× the link-stock weight.")
    print("Score = normalized tracking² + weighted torque use² + weighted design cost")

    for n in range(N_MIN, N_MAX + 1):
        best, statuses = optimize_fixed_n(n)
        if best is None:
            print(f"N={n}: no feasible candidate; optimizer statuses: {statuses}")
            continue
        results.append(best)
        m = best["metrics"]
        print(
            f"N={n} | lengths={np.round(best['lengths'], 5)} | "
            f"tracking RMS={m['rms_error']:.4g} rad | "
            f"torque use={m['effort_fraction']:.3f} | "
            f"max |q|={m['max_abs_joint_angle']:.3f} rad | "
            f"design cost={m['design_cost']:.3f} | score={best['score']:.5g}"
        )
        if not best["solver_status"].startswith("feasible initial"):
            print(f"  selected optimizer status: {best['solver_status']}")

    if not results:
        print("No feasible design was found; no results file was written.")
        return

    best_global = min(results, key=lambda r: r["score"])
    print("\nBest modeled design across joint counts")
    print(f"N={best_global['N']}, lengths={best_global['lengths']}, score={best_global['score']:.6g}")

    serializable = [
        {
            "N": int(r["N"]),
            "lengths": r["lengths"].tolist(),
            "score": float(r["score"]),
            "metrics": {key: float(value) for key, value in r["metrics"].items()},
            "solver_status": r["solver_status"],
        }
        for r in results
    ]
    output = HERE / "task_optimization_results.json"
    output.write_text(json.dumps(serializable, indent=2))
    print(f"Saved per-N results to {output.name}")


if __name__ == "__main__":
    main()
