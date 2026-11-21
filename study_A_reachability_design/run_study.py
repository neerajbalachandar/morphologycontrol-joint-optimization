"""Study A: reachability analysis for morphology design optimization.

Reproduces the structure of Nguyen, Cortes & Kramer, "Reachability Analysis for
Design Optimization" (ACC 2026, arXiv:2603.15865), with the planar arm of
``common/arm.py`` in place of the F-16 longitudinal model:

  A1  (paper Ex. 1, Fig. 1)  one-link arm: exact Theorem-2 boundary vs sampled
                             bang-bang extremals vs zonotope inner approximation;
                             Proposition 1 (one switch) checked, and shown to fail
                             for the two-link arm.
  A2  (paper Ex. 2, Fig. 2)  L^6 reachable set from sampled initial costates.
  A3  (paper Ex. 3, Fig. 3)  Proposition-2 costate ball (guaranteed-inside points).
  A4  (paper Ex. 4)          min l1 + l2  s.t. tr W >= 1.1 tr W0      (L^2 Gramian)
  A5  (paper Ex. 5)          min l1 + l2  s.t. vol R_L6 >= 1.1 vol R0_L6
  A6  (follow-up paper)      min l1 + l2  s.t. vol R_Linf >= 1.1 vol R0_Linf, and
                             max vol R_Linf (volume maximization, "VM")
  A7  validation             bang-bang extremals replayed on the nonlinear arm.

The two-link arm has 4 states and 2 inputs, the same dimensions as the paper's
aircraft model (v_T, alpha, q, theta; throttle, elevator), and 2 design
variables (l1, l2) as the paper has (b, c_bar).

Run from the repository root:  python study_A_reachability_design/run_study.py
"""
from __future__ import annotations

import os

# The matrices are tiny (4 x 4); multithreaded BLAS only adds contention.
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

from functools import lru_cache
import json
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import expm
from scipy.optimize import minimize
from scipy.spatial import ConvexHull

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

import reachability as R                                                     # noqa: E402
from common.arm import (DEFAULT, joint_positions, linearize, reach_margin,  # noqa: E402
                        simulate_nonlinear, total_mass)
from common.style import COLORS, apply_style, save                           # noqa: E402

FIG = HERE / "figures"
RES = HERE / "results"

# ---------------------------------------------------------------- settings
ARM = DEFAULT
T = 0.5                                  # s, horizon (paper: T = 1; see README for the time scaling)
BASELINE = (0.45, 0.45)                  # m, baseline link lengths (paper: b0, c_bar0)
BOUND_FACTORS = (0.5, 1.5)               # paper: 0.5 b0 <= b <= 1.5 b0
MIN_REACH_MARGIN = 0.05                  # p_star stays 5 % inside the workspace annulus
REQUIRED_GAIN = 0.10                     # paper: metric >= 1.1 x baseline
P_NORM = 6.0                             # paper Ex. 2, 3, 5
NT = 401                                 # time samples for the kernel
N_DIRS = 2048                            # costate directions in R^4 for hull volumes
GRID = 31                                # design-space map resolution per axis
ONE_LINK_LENGTH = 0.9                    # m, A1-A3 single link (= baseline total length)
FD_STEP = 1e-5                           # m, SLSQP finite-difference step

DIRS4 = R.sphere_directions(N_DIRS, 4, seed=0)
LP_RADIUS = (2 * T) ** (1 / P_NORM)      # (m T)^(1/p): makes the L^6 ball contain the unit box
METRICS = {
    "trace": "L2 Gramian trace (Ex. 4)",
    "vol_l6": "L6 reachable volume (Ex. 5)",
    "vol_linf": "L-inf (box) reachable volume",
}


# ---------------------------------------------------------------- metrics
@lru_cache(maxsize=None)
def _kernel(l1, l2):
    lin = linearize((l1, l2), ARM)
    t, w, G = R.kernel(lin.A, lin.B, lin.C_task, T, NT, lin.torque_limits)
    return lin, t, w, G


@lru_cache(maxsize=None)
def metric(l1, l2, name):
    """Reachability metric of the task-space set y = [dx, dy, dxdot, dydot]."""
    _, _, w, G = _kernel(round(l1, 12), round(l2, 12))
    if name == "trace":
        return float(np.trace(R.gramian(G, w)))
    if name == "vol_l6":
        return R.hull_volume(R.lp_boundary_points(G, w, DIRS4, P_NORM, LP_RADIUS))
    if name == "vol_linf":
        return R.hull_volume(R.bang_bang_points(G, w, DIRS4))
    raise KeyError(name)


def ratio(l, name):
    return metric(float(l[0]), float(l[1]), name) / metric(*BASELINE, name)


def length_bounds():
    return [(f * b, g * b) for b, f, g in zip(BASELINE, [BOUND_FACTORS[0]] * 2, [BOUND_FACTORS[1]] * 2)]


def feasible_geometry(l):
    return reach_margin(l, ARM.p_star) - MIN_REACH_MARGIN


# ---------------------------------------------------------------- A1-A3
def one_link():
    lin = linearize([ONE_LINK_LENGTH], ARM)
    return lin.A, lin.B, lin.torque_limits[0]


def experiment_a1():
    A, B, umax = one_link()
    hyp = R.planar_assumptions(A, B)
    eta, xi1, xi2 = R.theorem2_boundary(A, B[:, 0], T, -umax, umax)
    exact_area = float(ConvexHull(np.vstack((xi1, xi2))).volume)
    t, w, G = R.kernel(A, B, np.eye(2), T, 2001, [umax])
    D2 = R.sphere_directions(720, 2)
    pts = R.bang_bang_points(G, w, D2)
    zono = {s: R.zoh_zonotope(A, B, T, s, -umax, umax) for s in (5, 20)}
    zono_area = {s: float(ConvexHull(R.zonotope_polygon(*z)).volume) for s, z in zono.items()}
    sw1 = R.switch_counts(G, D2).ravel()
    lin2 = linearize(BASELINE, ARM)
    _, _, G2 = R.kernel(lin2.A, lin2.B, np.eye(4), T, 2001, lin2.torque_limits)
    sw2 = R.switch_counts(G2, R.sphere_directions(4096, 4, seed=1))

    apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.0), gridspec_kw={"width_ratios": [1.5, 1]})
    ax = axes[0]
    for s, color in zip(zono, ("#d9d9d9", "#a6bddb")):
        poly = R.zonotope_polygon(*zono[s])
        ax.fill(poly[:, 0], poly[:, 1], color=color, alpha=0.7,
                label=f"zonotope inner approx., {s} held steps (area {zono_area[s]:.4f})")
    ax.plot(xi1[:, 0], xi1[:, 1], color=COLORS["red_pts"], lw=2.2, label="Theorem 2: ξ₁(η)  (+ then −)")
    ax.plot(xi2[:, 0], xi2[:, 1], color=COLORS["blue_pts"], lw=2.2, label="Theorem 2: ξ₂(η)  (− then +)")
    ax.plot(pts[::6, 0], pts[::6, 1], "k.", ms=3, label="Lemma 1 bang-bang endpoints (sampled c)")
    ax.set(xlabel="joint angle Δq [rad]", ylabel="joint rate Δq̇ [rad/s]",
           title=f"A1: one-link reachable set, T = {T:g} s, |τ| ≤ {umax:g} N m  (exact area {exact_area:.4f})")
    ax.legend(frameon=False, fontsize=7, loc="upper left")
    ax = axes[1]
    kmax = int(max(sw1.max(), sw2.max()))
    f1 = np.bincount(sw1, minlength=kmax + 1) / sw1.size
    f2 = np.bincount(sw2.ravel(), minlength=kmax + 1) / sw2.size
    k = np.arange(kmax + 1)
    ax.bar(k - 0.2, f1, 0.4, color=COLORS["l2"], label="one link (2 states, 1 input)")
    ax.bar(k + 0.2, f2, 0.4, color=COLORS["linf"], label="two links (4 states, 2 inputs)")
    ax.set_xticks(k)
    ax.set(xlabel="switches of ψᵢ(t; c) on [0, T]", ylabel="fraction of sampled (c, i)",
           title="Proposition 1: ≤ 1 switch only in the planar case")
    ax.legend(frameon=False)
    fig.tight_layout()
    save(fig, FIG, "A1_exact_boundary_theorem2")
    return {"assumptions": hyp, "exact_area": exact_area,
            "sampled_bang_bang_area": float(ConvexHull(pts).volume),
            "zonotope_inner_area": zono_area,
            "max_switches_one_link": int(sw1.max()),
            "switch_histogram_two_link": np.bincount(sw2.ravel()).tolist()}


def experiment_a2_a3(rng):
    A, B, umax = one_link()
    assert umax == 1.0, "A2/A3 use the raw unit L^6 ball, as in the paper (tau_max = 1 N m)"
    p, q = P_NORM, P_NORM / (P_NORM - 1)
    # A2: sample lam0; scale each direction so costs straddle 1 (cost(s lam) = s^(1/(p-1)) cost(lam))
    # lam0 = -e^{A^T T} c makes -B^T lam(t) = B^T e^{A^T (T-t)} c, so c is the outward
    # normal of the endpoint; uniform c spreads the samples along the whole boundary.
    eAT = expm(A * T)
    n = 1525                                                    # paper: 1,525 samples
    th = rng.uniform(0, 2 * np.pi, n)
    dirs = -np.column_stack((np.cos(th), np.sin(th))) @ eAT
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    _, unit_cost = R.lp_costate_rollout(A, B, T, dirs, p)
    unit_mag = unit_cost ** (-(p - 1))                         # magnitude with cost exactly 1
    lam0 = dirs * (unit_mag * 10 ** rng.uniform(-1.2, 1.2, n))[:, None]
    xf, cost = R.lp_costate_rollout(A, B, T, lam0, p)
    inside = cost <= 1.0
    # A3: Proposition 2 ball ||lam0||_q^q <= bound
    bound = R.prop2_bound(A, B, T, p)
    th3 = rng.uniform(0, 2 * np.pi, 600)
    d3 = -np.column_stack((np.cos(th3), np.sin(th3))) @ eAT
    scale = (bound / np.sum(np.abs(d3) ** q, axis=1)) ** (1 / q) * rng.uniform(0, 1, 600) ** (p - 1)   # endpoint radius ~ U
    xf3, cost3 = R.lp_costate_rollout(A, B, T, d3 * scale[:, None], p)
    zpoly = R.zonotope_polygon(*R.zoh_zonotope(A, B, T, 40, -1.0, 1.0))

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.0), sharex=True, sharey=True)
    for ax in axes:
        ax.fill(zpoly[:, 0], zpoly[:, 1], color="#bdbdbd", alpha=0.6, label="L∞ set (zonotope inner approx.)")
        ax.plot(xf[inside, 0], xf[inside, 1], ".", ms=2.5, color=COLORS["blue_pts"], label="‖u‖_L6 ≤ 1 (reachable)")
        ax.plot(xf[~inside, 0], xf[~inside, 1], ".", ms=2.5, color=COLORS["red_pts"], label="‖u‖_L6 > 1 (not reachable)")
        ax.set_xlabel("joint angle Δq [rad]")
    axes[1].plot(xf3[:, 0], xf3[:, 1], ".", ms=3, color=COLORS["magenta"], label="Prop. 2 costate ball (guaranteed inside)")
    axes[0].set(ylabel="joint rate Δq̇ [rad/s]", title="A2: L⁶ set from sampled initial costates (Thm 3)")
    axes[1].set_title("A3: sufficient condition of Proposition 2")
    lim = 1.25 * np.abs(xf[inside]).max(axis=0)
    axes[0].set_xlim(-lim[0], lim[0]); axes[0].set_ylim(-lim[1], lim[1])
    for ax in axes:
        ax.legend(frameon=False, fontsize=7, loc="upper left", markerscale=3)
    fig.tight_layout()
    save(fig, FIG, "A2_A3_lp_costate_sampling")
    return {"samples": n, "inside_fraction": float(inside.mean()), "prop2_bound_lam0_q_norm_q": bound,
            "prop2_max_cost": float(cost3.max()),
            "l6_area_sampled_inside_hull": float(ConvexHull(xf[inside]).volume),
            "linf_area_zonotope40": float(ConvexHull(zpoly).volume)}


# ---------------------------------------------------------------- A4-A6
def optimize_min_cost(name):
    """Paper Eq. (19)-(20): min l1 + l2 s.t. metric >= (1 + gain) metric0."""
    cons = [{"type": "ineq", "fun": lambda l: ratio(l, name) - (1 + REQUIRED_GAIN)},
            {"type": "ineq", "fun": feasible_geometry}]
    path = [np.array(BASELINE)]
    res = minimize(lambda l: float(np.sum(l)), BASELINE, method="SLSQP", bounds=length_bounds(),
                   constraints=cons, callback=lambda xk: path.append(xk.copy()),
                   options={"ftol": 1e-9, "eps": FD_STEP, "maxiter": 200})
    return res, np.array(path)


def cobyla_check(name, x0):
    """Derivative-free re-solve from the SQP point (hull volumes are only piecewise smooth)."""
    lo = np.array([b[0] for b in length_bounds()])
    hi = np.array([b[1] for b in length_bounds()])
    cons = [{"type": "ineq", "fun": lambda l: ratio(l, name) - (1 + REQUIRED_GAIN)},
            {"type": "ineq", "fun": feasible_geometry},
            {"type": "ineq", "fun": lambda l: np.r_[l - lo, hi - l]}]
    res = minimize(lambda l: float(np.sum(l)), x0, method="COBYLA", constraints=cons,
                   options={"rhobeg": 0.02, "tol": 1e-6, "maxiter": 400})
    return {"lengths": res.x.tolist(), "cost": float(res.x.sum()), "ratio": float(ratio(res.x, name)),
            "function_evaluations": int(res.nfev)}


def optimize_max_volume():
    """Follow-up paper "VM": maximize the box-input volume within the same bounds."""
    path = [np.array(BASELINE)]
    res = minimize(lambda l: -ratio(l, "vol_linf"), BASELINE, method="SLSQP", bounds=length_bounds(),
                   constraints=[{"type": "ineq", "fun": feasible_geometry}],
                   callback=lambda xk: path.append(xk.copy()),
                   options={"ftol": 1e-10, "eps": FD_STEP, "maxiter": 200})
    return res, np.array(path)


def design_space_grid():
    (a, b), (c, d) = length_bounds()
    L1, L2 = np.meshgrid(np.linspace(a, b, GRID), np.linspace(c, d, GRID), indexing="ij")
    out = {k: np.full(L1.shape, np.nan) for k in METRICS}
    ok = np.zeros(L1.shape, bool)
    for i in range(GRID):
        for j in range(GRID):
            l = (L1[i, j], L2[i, j])
            ok[i, j] = feasible_geometry(l) >= 0
            if reach_margin(l, ARM.p_star) > 0.005:
                for k in METRICS:
                    out[k][i, j] = ratio(l, k)
    return L1, L2, out, ok


def grid_check(L1, L2, vals, ok, name):
    """Cheapest grid design meeting the requirement (guards against local minima)."""
    feas = ok & (vals[name] >= 1 + REQUIRED_GAIN)
    if not feas.any():
        return None
    cost = np.where(feas, L1 + L2, np.inf)
    i, j = np.unravel_index(np.argmin(cost), cost.shape)
    return [float(L1[i, j]), float(L2[i, j])]


def fig_design_space(L1, L2, vals, ok, designs, paths):
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.3), sharey=True)
    for ax, (name, label) in zip(axes, METRICS.items()):
        Z = vals[name]
        cs = ax.contourf(L1, L2, Z, levels=np.linspace(0, np.nanmax(Z), 15), cmap="Blues")
        fig.colorbar(cs, ax=ax, shrink=0.85, label="metric / baseline")
        ax.contour(L1, L2, Z, levels=[1 + REQUIRED_GAIN], colors=[COLORS["linf"]], linewidths=2)
        ax.contourf(L1, L2, (~ok).astype(float), levels=[0.5, 1.5], colors=["#888888"], alpha=0.55)
        for s in np.arange(0.5, 1.5, 0.1):
            ax.plot([s - 1, 1], [1, s - 1], color="white", lw=0.6, alpha=0.7)
        ax.plot(*BASELINE, "ks", ms=7, label="baseline")
        key = {"trace": "A4", "vol_l6": "A5", "vol_linf": "A6"}[name]
        ax.plot(paths[key][:, 0], paths[key][:, 1], ".-", color=COLORS["accent"], ms=4, lw=1, label="SQP iterates")
        ax.plot(*designs[key]["lengths"], "*", color=COLORS["accent"], ms=15, mec="k", label=f"{key} optimum")
        if name == "vol_linf":
            ax.plot(*designs["A6-VM"]["lengths"], "D", color="gold", mec="k", ms=7, label="max volume (VM)")
        ax.set(xlim=(L1.min(), L1.max()), ylim=(L2.min(), L2.max()), xlabel="l₁ [m]", title=label)
        ax.set_aspect("equal")
    axes[0].set_ylabel("l₂ [m]")
    axes[0].legend(frameon=False, fontsize=7, loc="upper right")
    fig.text(0.5, -0.02, "orange: requirement boundary (1.1 x baseline);  grey: p* outside the 5 % workspace margin; "
             "white lines: iso-cost l₁+l₂", ha="center", fontsize=8)
    fig.tight_layout()
    save(fig, FIG, "A3_design_space_maps")


def fig_projections(designs):
    """Follow-up paper Figs 5-7: 2-D projections of the 4-D box-input set."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.3))
    colors = {"baseline": COLORS["baseline"], "A4": COLORS["l2"], "A5": COLORS["l6"],
              "A6": COLORS["linf"], "A6-VM": "goldenrod"}
    for key, d in designs.items():
        _, _, w, G = _kernel(*map(float, d["lengths"]))
        pts = R.bang_bang_points(G, w, DIRS4)
        for ax, sl, unit in zip(axes, (slice(0, 2), slice(2, 4)), (1e3, 1e3)):
            poly = R.hull_polygon(pts[:, sl] * unit)
            ax.plot(poly[:, 0], poly[:, 1], color=colors[key], label=f"{key}: l=({d['lengths'][0]:.3f}, {d['lengths'][1]:.3f})")
    axes[0].set(xlabel="Δx tip [mm]", ylabel="Δy tip [mm]", title=f"tip position reachable in T = {T:g} s")
    axes[1].set(xlabel="Δẋ tip [mm/s]", ylabel="Δẏ tip [mm/s]", title="tip velocity reachable at T")
    for ax in axes:
        ax.set_aspect("equal", adjustable="datalim")
    axes[0].legend(frameon=False, fontsize=7)
    fig.tight_layout()
    save(fig, FIG, "A4_reachable_set_projections")


def fig_morphology(designs):
    fig, ax = plt.subplots(figsize=(5, 4.2))
    colors = {"baseline": COLORS["baseline"], "A4": COLORS["l2"], "A5": COLORS["l6"],
              "A6": COLORS["linf"], "A6-VM": "goldenrod"}
    for key, d in designs.items():
        lin = linearize(d["lengths"], ARM)
        pts = joint_positions(lin.q0, np.asarray(d["lengths"]))
        ax.plot(pts[:, 0], pts[:, 1], "o-", color=colors[key], lw=2.4, ms=5, label=f"{key}  (Σl = {sum(d['lengths']):.3f} m)")
    ax.plot(*ARM.p_star, "k*", ms=13, label="work point p*")
    ax.plot(0, 0, "ks", ms=8)
    ax.set(aspect="equal", xlabel="x [m]", ylabel="y [m]", title="Designs holding the tool at p*")
    ax.legend(frameon=False, fontsize=7)
    save(fig, FIG, "A5_morphologies")


def fig_cross_metrics(designs):
    fig, ax = plt.subplots(figsize=(7.5, 3.4))
    keys = list(designs)
    width = 0.25
    for i, (name, label) in enumerate(METRICS.items()):
        vals = [designs[k]["ratios"][name] for k in keys]
        ax.bar(np.arange(len(keys)) + (i - 1) * width, vals, width,
               color=(COLORS["l2"], COLORS["l6"], COLORS["linf"])[i], label=label)
    ax.axhline(1.0, color="k", lw=0.8)
    ax.axhline(1 + REQUIRED_GAIN, color="k", lw=0.8, ls="--")
    ax.set_xticks(range(len(keys)), [f"{k}\nΣl={sum(designs[k]['lengths']):.3f}" for k in keys])
    ax.set(ylabel="metric / baseline", title="Each design scored on every metric")
    ax.legend(frameon=False, fontsize=7)
    fig.tight_layout()
    save(fig, FIG, "A6_cross_metric_comparison")


# ---------------------------------------------------------------- A7
def experiment_a7(designs):
    """Replay task-plane bang-bang extremals on the nonlinear arm."""
    th = np.linspace(0, 2 * np.pi, 24, endpoint=False)
    C = np.column_stack((np.cos(th), np.sin(th), np.zeros(24), np.zeros(24)))
    out = {}
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.3))
    for ax, key in zip(axes, ("baseline", "A6")):
        l = np.asarray(designs[key]["lengths"])
        lin, t, w, G = _kernel(*map(float, l))
        lin_end = R.bang_bang_points(G, w, C)[:, :2]
        nl_end = []
        for c in C:
            v = np.sign(np.einsum("tki,k->ti", G, c))               # Lemma 1 control on the grid
            tau = v * lin.torque_limits
            fn = (lambda tt, tau=tau: tau[min(int(round(tt / (t[1] - t[0]))), len(t) - 1)])
            sim = simulate_nonlinear(l, lin.q0, fn, T, ARM, n_eval=50)
            nl_end.append(sim["tip"][-1] - np.asarray(ARM.p_star))
        nl_end = np.array(nl_end)
        pts = R.bang_bang_points(G, w, DIRS4)
        poly = R.hull_polygon(pts[:, :2] * 1e3)
        ax.plot(poly[:, 0], poly[:, 1], color=COLORS["gray"], label="linear reachable set (projection)")
        ax.plot(lin_end[:, 0] * 1e3, lin_end[:, 1] * 1e3, "o", mfc="none", color=COLORS["l2"], label="linear endpoints")
        ax.plot(nl_end[:, 0] * 1e3, nl_end[:, 1] * 1e3, "x", color=COLORS["linf"], label="nonlinear endpoints")
        err = np.linalg.norm(nl_end - lin_end, axis=1) / np.linalg.norm(lin_end, axis=1)
        ax.set(aspect="equal", xlabel="Δx tip [mm]", ylabel="Δy tip [mm]",
               title=f"{key}: median endpoint error {100*np.median(err):.1f} %")
        out[key] = {"median_relative_error": float(np.median(err)), "max_relative_error": float(err.max())}
    axes[0].legend(frameon=False, fontsize=7)
    fig.tight_layout()
    save(fig, FIG, "A7_nonlinear_validation")
    return out


# ---------------------------------------------------------------- main
def describe(l):
    l = np.asarray(l, float)
    return {"lengths": l.tolist(), "cost_l1_plus_l2": float(l.sum()),
            "cost_change_percent": float(100 * (l.sum() / sum(BASELINE) - 1)),
            "total_mass_kg": total_mass(l, ARM),
            "reach_margin": float(reach_margin(l, ARM.p_star)),
            "ratios": {k: float(ratio(l, k)) for k in METRICS},
            "absolute": {k: float(metric(float(l[0]), float(l[1]), k)) for k in METRICS}}


def main():
    apply_style()
    rng = np.random.default_rng(0)
    t0 = time.time()
    print("Study A: reachability analysis for design optimization")
    print("[A1] Theorem 2 / Proposition 1"); a1 = experiment_a1()
    print("[A2, A3] L^6 costate sampling, Proposition 2"); a23 = experiment_a2_a3(rng)

    print("[A4-A6] design optimization")
    designs, paths, solver = {"baseline": describe(BASELINE)}, {}, {}
    for key, name in (("A4", "trace"), ("A5", "vol_l6"), ("A6", "vol_linf")):
        res, path = optimize_min_cost(name)
        designs[key], paths[key] = describe(res.x), path
        solver[key] = {"success": bool(res.success), "message": str(res.message), "iterations": int(res.nit),
                       "requirement_met": bool(ratio(res.x, name) >= 1 + REQUIRED_GAIN - 1e-6),
                       "cobyla_check": cobyla_check(name, res.x)}
        print(f"  {key}: l = {np.round(res.x, 4)}, Σl = {res.x.sum():.4f} ({designs[key]['cost_change_percent']:+.1f} %), "
              f"{name} ratio {ratio(res.x, name):.4f}, {res.message}")
    res, path = optimize_max_volume()
    designs["A6-VM"], paths["A6-VM"] = describe(res.x), path
    solver["A6-VM"] = {"success": bool(res.success), "message": str(res.message), "iterations": int(res.nit)}
    print(f"  VM: l = {np.round(res.x, 4)}, volume ratio {ratio(res.x, 'vol_linf'):.4f}")

    print("  design-space grid")
    L1, L2, vals, ok = design_space_grid()
    grid = {key: grid_check(L1, L2, vals, ok, name)
            for key, name in (("A4", "trace"), ("A5", "vol_l6"), ("A6", "vol_linf"))}

    print("[figures]")
    fig_design_space(L1, L2, vals, ok, designs, paths)
    fig_projections(designs)
    fig_morphology(designs)
    fig_cross_metrics(designs)
    print("[A7] nonlinear validation"); a7 = experiment_a7(designs)

    summary = {
        "settings": {"horizon_s": T, "baseline_lengths_m": BASELINE, "bound_factors": BOUND_FACTORS,
                     "min_reach_margin": MIN_REACH_MARGIN, "required_gain": REQUIRED_GAIN,
                     "p_norm": P_NORM, "lp_radius_normalized_inputs": LP_RADIUS,
                     "time_samples": NT, "costate_directions": N_DIRS, "grid": GRID,
                     "one_link_length_m": ONE_LINK_LENGTH, "arm": ARM.__dict__},
        "A1_theorem2": a1, "A2_A3_lp": a23,
        "designs": designs, "solver": solver, "grid_check_cheapest_feasible": grid,
        "A7_nonlinear_validation": a7, "runtime_s": round(time.time() - t0, 1),
    }
    RES.mkdir(exist_ok=True)
    (RES / "summary.json").write_text(json.dumps(summary, indent=2, default=float) + "\n")
    print(f"  wrote results/summary.json ({summary['runtime_s']} s)")


if __name__ == "__main__":
    main()
