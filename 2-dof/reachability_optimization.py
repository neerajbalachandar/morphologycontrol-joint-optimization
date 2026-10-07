"""
Pure capability optimization: maximize finite-time bounded-input reachable-set area.

No prescribed capability target and no minimum-design-cost objective.
For each N, maximize A_R(p) subject to the same length constraints.

The reachable set is the 2-D end-effector output set
    y(T) = C x(T) = [x_EE(T), xdot_EE(T)]^T
under |u_i(t)| <= TAU_BUDGET / N.

SLSQP is a local optimizer, so multiple initial guesses are used.
The reported result is the best converged solution found, not a proof
of global optimality.
"""

import numpy as np
from scipy.optimize import minimize

import json
from pathlib import Path

from model import robot_matrices, output_matrix
from reachability import reachable_set
from design import (
    N_MIN,
    N_MAX,
    BASELINE_LENGTHS,
    BASELINE_TOTAL_LENGTH,
    TOTAL_MASS,
    TOTAL_LENGTH_MIN,
    TOTAL_LENGTH_MAX,
    LENGTH_MIN,
    LENGTH_MAX,
    TAU_BUDGET,
)

T = 2.0
NT = 250
NTHETA = 120
N_STARTS = 8
SEED = 7


def evaluate_area(lengths):
    N = len(lengths)
    A, B, *_ = robot_matrices(lengths, total_mass=TOTAL_MASS)
    C = output_matrix(lengths)
    tau_max = (TAU_BUDGET / N) * np.ones(N)

    _, _, _, area = reachable_set(
        A, B, C, tau_max,
        T=T,
        ntheta=NTHETA,
        nt=NT,
    )
    return float(area)


def make_starts(N):
    rng = np.random.default_rng(SEED + N)
    starts = []

    # Seed only with feasible designs. SLSQP behaves poorly when every
    # initial point violates the aggregate length constraint.
    lower = max(LENGTH_MIN, TOTAL_LENGTH_MIN / N)
    upper = min(LENGTH_MAX, TOTAL_LENGTH_MAX / N)
    starts.append(np.full(N, np.clip(BASELINE_TOTAL_LENGTH / N, lower, upper)))
    starts.append(np.full(N, lower))
    starts.append(np.full(N, upper))

    while len(starts) < N_STARTS:
        x = rng.uniform(LENGTH_MIN, LENGTH_MAX, size=N)
        total = np.sum(x)

        if total < TOTAL_LENGTH_MIN:
            room = LENGTH_MAX - x
            needed = TOTAL_LENGTH_MIN - total
            if np.sum(room) > 0:
                x = x + needed * room / np.sum(room)

        elif total > TOTAL_LENGTH_MAX:
            room = x - LENGTH_MIN
            excess = total - TOTAL_LENGTH_MAX
            if np.sum(room) > 0:
                x = x - excess * room / np.sum(room)

        if (
            np.all(x >= LENGTH_MIN - 1e-10)
            and np.all(x <= LENGTH_MAX + 1e-10)
            and TOTAL_LENGTH_MIN - 1e-10 <= np.sum(x) <= TOTAL_LENGTH_MAX + 1e-10
        ):
            starts.append(x)

    # Remove duplicate starts (common for N=1 and at active bounds).
    unique = []
    for x in starts:
        if not any(np.allclose(x, y) for y in unique):
            unique.append(x)
    return unique


def optimize_fixed_N(N):
    bounds = [(LENGTH_MIN, LENGTH_MAX) for _ in range(N)]

    constraints = [
        {
            "type": "ineq",
            "fun": lambda x: np.sum(x) - TOTAL_LENGTH_MIN,
        },
        {
            "type": "ineq",
            "fun": lambda x: TOTAL_LENGTH_MAX - np.sum(x),
        },
    ]

    def objective(x):
        return -evaluate_area(x)

    best = None

    for k, x0 in enumerate(make_starts(N), start=1):
        result = minimize(
            objective,
            x0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={
                "maxiter": 100,
                "ftol": 1e-7,
                "disp": False,
            },
        )

        if not result.success:
            print(f"  start {k}: failed -- {result.message}")
            continue

        lengths = result.x
        area = evaluate_area(lengths)

        if best is None or area > best["area"]:
            best = {
                "N": N,
                "lengths": lengths,
                "area": area,
                "total_length": np.sum(lengths),
                "start": k,
            }

    return best


print("\n" + "=" * 72)
print("PURE REACHABLE-SET CAPABILITY OPTIMIZATION")
print("=" * 72)
print(f"Baseline lengths        = {BASELINE_LENGTHS}")
print(f"Baseline total length   = {BASELINE_TOTAL_LENGTH:.6f}")
print(f"Total mass              = {TOTAL_MASS:.6f}")
print(f"Total torque budget     = {TAU_BUDGET:.6f}")
print(f"Time horizon            = {T:.3f}")
print(f"Reachability directions = {NTHETA}")
print(f"Time quadrature points  = {NT}")
print(f"Starts per N            = {N_STARTS}")

baseline = BASELINE_LENGTHS
baseline_area = evaluate_area(baseline)
print(f"\nBaseline reachable area = {baseline_area:.8e}")

results = []

for N in range(N_MIN, N_MAX + 1):
    print("\n" + "-" * 72)
    print(f"OPTIMIZING REACHABLE AREA: N = {N}")
    print("-" * 72)

    best = optimize_fixed_N(N)

    if best is None:
        print("No converged feasible solution found.")
        continue

    improvement = (best["area"] / baseline_area - 1.0) * 100.0
    best["improvement_percent"] = improvement
    results.append(best)

    print(f"Best start           = {best['start']}")
    print(f"Optimal lengths      = {best['lengths']}")
    print(f"Total length         = {best['total_length']:.8f}")
    print(f"Reachable area       = {best['area']:.8e}")
    print(f"Improvement vs base  = {improvement:.4f}%")

print("\n" + "=" * 72)
print("REACHABLE-SET OPTIMIZATION SUMMARY")
print("=" * 72)
print(f"{'N':>4s}{'Total L':>14s}{'Area':>18s}{'Gain':>14s}")

for r in results:
    print(
        f"{r['N']:4d}"
        f"{r['total_length']:14.6f}"
        f"{r['area']:18.8e}"
        f"{r['improvement_percent']:13.4f}%"
    )

if results:
    best_global = max(results, key=lambda r: r["area"])

    print("\n" + "=" * 72)
    print("REACHABLE-SET-OPTIMAL MORPHOLOGY")
    print("=" * 72)
    print(f"N*                 = {best_global['N']}")
    print(f"lengths*           = {best_global['lengths']}")
    print(f"total length       = {best_global['total_length']:.8f}")
    print(f"reachable area     = {best_global['area']:.8e}")
    print(f"improvement        = {best_global['improvement_percent']:.4f}%")
    output = Path(__file__).with_name("reachability_optimization_results.json")
    serializable = [
        {
            "N": int(r["N"]),
            "lengths": np.asarray(r["lengths"], dtype=float).tolist(),
            "area": float(r["area"]),
            "total_length": float(r["total_length"]),
            "start": int(r["start"]),
            "improvement_percent": float(r["improvement_percent"]),
        }
        for r in results
    ]
    output.write_text(json.dumps(serializable, indent=2))
    print("Saved optimization results for plotting.")
else:
    print("\nNo converged feasible morphology was found.")
