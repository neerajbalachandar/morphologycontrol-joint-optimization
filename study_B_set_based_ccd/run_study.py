"""Study B: set-based robust control co-design of the two-link arm.

Reproduces the structure of Bird, Siefert, Pangborn & Jain, "A Set-Based
Approach for Robust Control Co-Design" (ACC 2024) on the arm of ``common/arm.py``:

  1. Nominal simultaneous CCD + LQR (the paper's baseline): link lengths chosen
     for one nominal disturbance scenario, LQR gain from the same weights.
  2. Control-only set-based design on the baseline morphology (sequential
     design; not in the paper, added to isolate the value of co-design).
  3. Set-based robust CCD (Problem 24): link lengths and state-feedback gain
     chosen together so that the reachable tube is safe, respects the torque
     limits and returns into R(0), which certifies safety for all time.

Then validates the certificate by simulation (worst-case and random disturbance
sequences, linear and nonlinear arm) and writes figures and a JSON summary.

Run from the repository root:  python study_B_set_based_ccd/run_study.py
"""
from __future__ import annotations

import os

# The matrices are tiny (4 x 4); multithreaded BLAS only adds contention.
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import json
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

import setbased as sb                                         # noqa: E402
from common.arm import joint_positions, reach_margin, simulate_nonlinear_sampled  # noqa: E402
from common.style import COLORS, apply_style, save            # noqa: E402

FIG = HERE / "figures"
RES = HERE / "results"
S = sb.SETUP

# Nominal scenario used by the baseline design (one initial state, one force history).
NOMINAL_STEPS = 80                     # 0.8 s

# LQR / scenario weights: tip error in units of the tolerance, torque in units of the limit.
LQR_TORQUE_WEIGHT = 20.0
LQR_VELOCITY_WEIGHT = 1e-4


def nominal_force(k):
    """Deterministic force history (N): a smooth x-push and a slower y-sway."""
    t = k * S.dt
    fx = 0.8 * S.force_max * np.sin(np.pi * t / 0.3) ** 2 * (t < 0.3)
    fy = 0.6 * S.force_max * np.sin(2 * np.pi * t / 0.6) * (t < 0.6)
    return np.array([fx, fy])


# --------------------------------------------------------------------------
# Simulation helpers
# --------------------------------------------------------------------------
def simulate_linear(design, forces, x0=None):
    Acl = design.Ad + design.Bd @ design.K
    x = np.zeros((len(forces) + 1, 4))
    if x0 is not None:
        x[0] = x0
    for k, F in enumerate(forces):
        x[k + 1] = Acl @ x[k] + design.Ed @ F
    u = x[:-1] @ design.K.T
    tip = x[:, :2] @ design.lin.J.T
    return {"x": x, "u": u, "tip": tip, "t": np.arange(len(x)) * S.dt}


def lqr_weights(design):
    J = design.lin.J
    Q = np.zeros((4, 4))
    Q[:2, :2] = J.T @ J / S.tip_tolerance**2
    Q[2:, 2:] = LQR_VELOCITY_WEIGHT * np.eye(2)
    R = LQR_TORQUE_WEIGHT * np.diag(1.0 / np.asarray(S.arm.torque_limits) ** 2)
    return Q, R


def lqr_design(lengths):
    d = sb.build(np.r_[lengths, np.zeros(8)])
    Q, R = lqr_weights(d)
    d.K = sb.lqr_gain(d, Q, R)
    return d


def scenario_cost(design):
    """Quadratic cost of the nominal scenario, same weights as the LQR."""
    Q, R = lqr_weights(design)
    sim = simulate_linear(design, [nominal_force(k) for k in range(NOMINAL_STEPS)])
    x, u = sim["x"], sim["u"]
    return float(S.dt * (np.einsum("ti,ij,tj->", x, Q, x) + np.einsum("ti,ij,tj->", u, R, u)))


def _task_direction(design, h):
    """Safety rows are h = [+-J_row, 0]; return the matching unit task axis."""
    J = design.lin.J
    for sign in (1, -1):
        for i in range(2):
            if np.allclose(h[:2], sign * J[i]):
                e = np.zeros(2)
                e[i] = sign
                return e
    raise ValueError("not a safety direction")


def worst_case_sequence(design, direction, k_star):
    """Disturbance and initial vertex that maximize direction^T x(k_star).

    x(k) = Acl^k x0 + sum_j Acl^(k-1-j) Ed w_j is linear, so the maximizer is a
    vertex choice per step (bang-bang), which realizes the support function.
    """
    Acl = design.Ad + design.Bd @ design.K
    lam = direction.copy()
    forces = np.zeros((k_star, 2))
    for j in range(k_star - 1, -1, -1):
        forces[j] = S.force_max * np.sign(design.Ed.T @ lam)
        lam = Acl.T @ lam
    x0 = np.asarray(S.r0) * np.sign(lam)
    return x0, forces


# --------------------------------------------------------------------------
# Designs
# --------------------------------------------------------------------------
def nominal_ccd():
    """Baseline: lengths minimize design cost + nominal scenario cost, K = LQR(lengths)."""
    def f(l):
        return sb.design_cost(l) + scenario_cost(lqr_design(l))
    res = minimize(f, S.baseline_lengths, method="SLSQP", bounds=[S.length_bounds] * 2,
                   constraints=[{"type": "ineq", "fun": lambda l: reach_margin(l, S.arm.p_star) - S.min_reach_margin}],
                   options={"ftol": 1e-10, "eps": 1e-6, "maxiter": 200})
    d = lqr_design(res.x)
    return d, res


def starting_points(lengths_list):
    """LQR gains over a range of weights for each starting morphology."""
    starts = []
    for lengths in lengths_list:
        for qscale in (1.0, 10.0, 100.0):
            d = sb.build(np.r_[lengths, np.zeros(8)])
            Q, R = lqr_weights(d)
            K = sb.lqr_gain(d, qscale * Q, R)
            z = np.clip(sb.pack(lengths, K)[2:], -S.gain_bound, S.gain_bound)
            starts.append(np.r_[lengths, z])
    return starts


def set_based(starts, fixed_lengths=None, label=""):
    """Phase 1 then Problem (24) from every start; keep the best certified design."""
    best, best_slack, log = None, (-np.inf, None), []
    for i, p0 in enumerate(starts):
        t0 = time.time()
        slack, p_feas = sb.find_feasible(p0, fixed_lengths=fixed_lengths)
        entry = {"start": i, "phase1_slack": slack}
        if slack > best_slack[0]:
            best_slack = (slack, p_feas)
        if slack >= 0:
            res, hist = sb.solve(p_feas, fixed_lengths=fixed_lengths)
            cert = min(sb.certificate(res.x).values())
            p_final = res.x if cert >= -1e-6 else p_feas
            J = sb.objective(p_final)
            entry.update({"objective": J, "certificate": cert, "message": str(res.message),
                          "iterations": int(res.nit)})
            if best is None or J < best["objective"]:
                best = {"p": p_final, "objective": J, "history": [sb.objective(p_feas)] + hist}
        entry["seconds"] = round(time.time() - t0, 1)
        log.append(entry)
        print(f"  {label} start {i}: phase-1 slack {slack:+.4f}"
              + (f", J = {entry['objective']:.4f}" if "objective" in entry else ""))
    return best, best_slack, log


# --------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------
def projected_polygon(center, generators, P):
    c, G = P @ center, P @ generators
    th = np.linspace(0, 2 * np.pi, 360, endpoint=False)
    D = np.column_stack((np.cos(th), np.sin(th)))
    pts = c + np.sign(D @ G) @ G.T
    v = pts[ConvexHull(pts).vertices]
    return np.vstack((v, v[:1]))


def fig_tubes(designs):
    """Paper Fig. 3: reachable sets projected on tip deviation (dx, dy) in mm."""
    fig, axes = plt.subplots(1, len(designs), figsize=(4.1 * len(designs), 3.9), sharex=True, sharey=True)
    for ax, (name, d) in zip(np.atleast_1d(axes), designs.items()):
        sets = sb.tube(d)
        P = np.c_[d.lin.J, np.zeros((2, 2))] * 1e3
        cmap = plt.get_cmap("viridis")
        for k, (c, G) in enumerate(sets):
            poly = projected_polygon(c, G, P)
            ax.plot(poly[:, 0], poly[:, 1], color=cmap(k / S.steps), lw=0.7, alpha=0.8)
        r0 = projected_polygon(np.zeros(4), np.diag(S.r0), P)
        ax.plot(r0[:, 0], r0[:, 1], color="black", lw=2.2, label="R(0) (operating region)")
        lim = S.tip_tolerance * 1e3
        ax.plot([-lim, lim, lim, -lim, -lim], [-lim, -lim, lim, lim, -lim], "--", color=COLORS["red_pts"],
                lw=1.5, label="safety bound |dx|,|dy| ≤ δ")
        cert = sb.certificate(sb.pack(d.lengths, d.K))
        ok = all(v >= -1e-6 for v in cert.values())
        ax.set_title(f"{name}\n{'certified safe & invariant' if ok else 'NOT certified'}", fontsize=9)
        ax.set_xlabel("tip deviation dx [mm]")
        ax.set_aspect("equal")
    np.atleast_1d(axes)[0].set_ylabel("tip deviation dy [mm]")
    np.atleast_1d(axes)[0].legend(loc="lower left", frameon=False)
    sm = plt.cm.ScalarMappable(cmap="viridis", norm=plt.Normalize(0, S.steps * S.dt))
    fig.colorbar(sm, ax=axes, label="time of R(k) [s]", shrink=0.8)
    save(fig, FIG, "B1_reachable_tubes_tip")


def fig_usage(designs):
    """Constraint usage along the horizon (support value / limit) for every design."""
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.2))
    for (name, d), color in zip(designs.items(), (COLORS["nominal"], COLORS["sequential"], COLORS["setbased"])):
        r = sb.reach(d)
        Hx, fx = sb.safety_directions(d)
        t = np.arange(1, S.steps + 1) * S.dt
        axes[0].plot(t, (r["rho_x"] / fx).max(axis=1), color=color, label=name)
        tau = np.asarray(S.arm.torque_limits)
        tu = np.arange(S.steps) * S.dt
        axes[1].plot(tu, (r["rho_u"][:, :2] / tau).max(axis=1), color=color)
        _, fr = sb.region_directions()
        axes[2].bar(name.replace(" ", "\n"), (r["rho_r"] / fr).max(), color=color)
    for ax, title in zip(axes, ("safety usage  max ρ_x / δ", "torque usage  max ρ_u / τ_max",
                                "terminal R(N) ⊆ R(0) usage")):
        ax.axhline(1.0, ls="--", color="black", lw=1)
        ax.set_title(title)
    axes[0].set_xlabel("time [s]"); axes[1].set_xlabel("time [s]")
    axes[2].tick_params(axis="x", labelsize=7)
    axes[0].legend(frameon=False)
    fig.tight_layout()
    save(fig, FIG, "B2_constraint_usage")


def fig_nominal_scenario(designs):
    """Paper Fig. 4: nominal-scenario tip trajectories and torques."""
    forces = np.array([nominal_force(k) for k in range(NOMINAL_STEPS)])
    fig, axes = plt.subplots(2, 2, figsize=(10, 5.6), sharex=True)
    for (name, d), color in zip(designs.items(), (COLORS["nominal"], COLORS["sequential"], COLORS["setbased"])):
        sim = simulate_linear(d, forces)
        for j, comp in enumerate("xy"):
            axes[0, j].plot(sim["t"], sim["tip"][:, j] * 1e3, color=color, label=name)
            axes[1, j].plot(sim["t"][:-1], sim["u"][:, j], color=color)
    for j in range(2):
        axes[0, j].axhline(S.tip_tolerance * 1e3, ls="--", color="k", lw=0.8)
        axes[0, j].axhline(-S.tip_tolerance * 1e3, ls="--", color="k", lw=0.8)
        axes[1, j].axhline(S.arm.torque_limits[j], ls="--", color="k", lw=0.8)
        axes[1, j].axhline(-S.arm.torque_limits[j], ls="--", color="k", lw=0.8)
        axes[1, j].set_xlabel("time [s]")
    axes[0, 0].set_ylabel("tip dx [mm]"); axes[0, 1].set_ylabel("tip dy [mm]")
    axes[1, 0].set_ylabel("shoulder torque [N m]"); axes[1, 1].set_ylabel("elbow torque [N m]")
    axes[0, 0].legend(frameon=False)
    ax2 = axes[0, 0].inset_axes([0.62, 0.62, 0.36, 0.33])
    ax2.plot(np.arange(NOMINAL_STEPS) * S.dt, forces, lw=1)
    ax2.set_title("nominal force [N]", fontsize=7); ax2.tick_params(labelsize=6)
    fig.suptitle("Nominal scenario (the only scenario the baseline was designed for)", fontsize=10)
    fig.tight_layout()
    save(fig, FIG, "B3_nominal_scenario")


def worst_case_and_monte_carlo(designs, rng):
    """Realize the binding support direction of each design and sample random sequences."""
    out = {}
    fig, axes = plt.subplots(1, len(designs), figsize=(4.1 * len(designs), 3.4), sharey=True)
    for ax, ((name, d), color) in zip(np.atleast_1d(axes), zip(designs.items(), (COLORS["nominal"], COLORS["sequential"], COLORS["setbased"]))):
        r = sb.reach(d)
        Hx, fx = sb.safety_directions(d)
        usage = r["rho_x"] / fx
        k_idx, h_idx = np.unravel_index(np.argmax(usage), usage.shape)
        k_star = k_idx + 1
        x0, forces = worst_case_sequence(d, Hx[h_idx], k_star)
        forces = np.vstack((forces, np.zeros((S.steps - k_star, 2))))
        sim = simulate_linear(d, forces, x0)
        proj = sim["x"] @ Hx[h_idx]
        # random vertex sequences from random vertices of R(0)
        mc = []
        for _ in range(300):
            x0r = np.asarray(S.r0) * rng.choice([-1, 1], 4)
            fr = S.force_max * rng.choice([-1, 1], (S.steps, 2))
            mc.append(simulate_linear(d, fr, x0r)["x"] @ Hx[h_idx])
        mc = np.array(mc)
        # nonlinear check of the worst case with sampled feedback
        nl = simulate_nonlinear_sampled(d.lengths, d.lin.q0, d.K, forces, x0, S.dt, S.arm)
        # project the true (nonlinear) tip deviation onto the same task-space direction
        nl_proj = nl["tip"] @ _task_direction(d, Hx[h_idx])
        t = sim["t"]
        ax.fill_between(t, mc.min(0) * 1e3, mc.max(0) * 1e3, color=color, alpha=0.18, label="300 random sequences")
        ax.plot(t, proj * 1e3, color=color, label=f"worst case for step {k_star}")
        ax.plot(t, nl_proj * 1e3, ":", color="black", lw=1.2, label="same, nonlinear arm")
        rho = np.r_[np.nan, r["rho_x"][:, h_idx]]
        ax.plot(t, rho * 1e3, color="gray", lw=1, label="support value ρ (bound on the tube)")
        ax.axhline(fx[h_idx] * 1e3, ls="--", color=COLORS["red_pts"], label="safety bound δ")
        ax.set_title(name, fontsize=9)
        ax.set_xlabel("time [s]")
        out[name] = {"k_star": int(k_star), "direction": Hx[h_idx].tolist(),
                     "worst_case_peak_mm": float(proj.max() * 1e3),
                     "nonlinear_worst_case_peak_mm": float(nl_proj.max() * 1e3),
                     "support_bound_mm": float(r["rho_x"][k_idx, h_idx] * 1e3),
                     "monte_carlo_peak_mm": float(mc.max() * 1e3),
                     "tolerance_mm": S.tip_tolerance * 1e3}
    np.atleast_1d(axes)[0].set_ylabel("tip deviation along binding direction [mm]")
    np.atleast_1d(axes)[0].legend(frameon=False, fontsize=6.5, loc="lower right")
    fig.tight_layout()
    save(fig, FIG, "B4_worst_case_validation")
    return out


def fig_morphology(designs):
    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    for (name, d), color in zip(designs.items(), (COLORS["nominal"], COLORS["sequential"], COLORS["setbased"])):
        pts = joint_positions(d.lin.q0, d.lengths)
        ax.plot(pts[:, 0], pts[:, 1], "o-", color=color, lw=2.4, ms=5,
                label=f"{name}: l = ({d.lengths[0]:.3f}, {d.lengths[1]:.3f}) m")
    ax.plot(*S.arm.p_star, "k*", ms=12, label="work point p*")
    ax.plot(0, 0, "ks", ms=8)
    ax.set_aspect("equal"); ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
    ax.legend(frameon=False, fontsize=7, loc="upper left")
    ax.set_title("Arm holding the tool at p*")
    save(fig, FIG, "B5_morphologies")


def fig_convergence(history):
    fig, ax = plt.subplots(figsize=(4.6, 3.0))
    ax.plot(history, "o-", ms=3, color=COLORS["setbased"])
    ax.set_xlabel("SQP iteration"); ax.set_ylabel("objective (24a)")
    ax.set_title("Set-based CCD convergence (from phase-1 point)")
    save(fig, FIG, "B6_convergence")


# --------------------------------------------------------------------------
def summarize(d, extra=None):
    p = sb.pack(d.lengths, d.K)
    J, parts = sb.objective(p, parts=True)
    r = sb.reach(d)
    Hx, fx = sb.safety_directions(d)
    _, fr = sb.region_directions()
    tau = np.asarray(S.arm.torque_limits)
    out = {"lengths_m": d.lengths.tolist(), "q0_rad": d.lin.q0.tolist(), "K": d.K.tolist(),
           "objective_24a": float(J), "objective_terms": {k: float(v) for k, v in parts.items()},
           "certificate_min_slack": sb.certificate(p),
           "certified": bool(min(sb.certificate(p).values()) >= -1e-6),
           "max_safety_usage": float((r["rho_x"] / fx).max()),
           "max_torque_usage": float((r["rho_u"][:, :2] / tau).max()),
           "terminal_usage": float((r["rho_r"] / fr).max()),
           "closed_loop_spectral_radius": float(np.abs(np.linalg.eigvals(d.Ad + d.Bd @ d.K)).max()),
           "nominal_scenario_cost": scenario_cost(d)}
    out.update(extra or {})
    return out


def main():
    apply_style()
    rng = np.random.default_rng(0)
    t_start = time.time()
    print("Study B: set-based robust control co-design")

    print("[1] nominal simultaneous CCD + LQR")
    nominal, res_nom = nominal_ccd()
    print(f"  lengths {np.round(nominal.lengths, 4)}, certificate {sb.certificate(sb.pack(nominal.lengths, nominal.K))}")

    print("[2] control-only set-based design on the baseline morphology")
    l0 = np.asarray(S.baseline_lengths)
    best_ctrl, slack_ctrl, log_ctrl = set_based(starting_points([l0]), fixed_lengths=l0, label="control-only")

    print("[3] set-based robust CCD (lengths + gains)")
    starts = starting_points([l0, nominal.lengths, np.array([0.60, 0.55])])
    best_co, slack_co, log_co = set_based(starts, label="co-design")
    if best_co is None:
        raise RuntimeError("No certified co-design found; relax the setup in setbased.Setup.")

    designs = {"Nominal CCD + LQR": nominal}
    if best_ctrl is not None:
        designs["Control-only set-based"] = sb.build(best_ctrl["p"])
    else:
        designs["Control-only (least violation)"] = sb.build(slack_ctrl[1])
    designs["Set-based CCD"] = sb.build(best_co["p"])

    print("[4] figures and validation")
    fig_tubes(designs)
    fig_usage(designs)
    fig_nominal_scenario(designs)
    validation = worst_case_and_monte_carlo(designs, rng)
    fig_morphology(designs)
    fig_convergence(best_co["history"])

    summary = {
        "setup": {k: (list(v) if isinstance(v, tuple) else v) for k, v in S.__dict__.items() if k != "arm"},
        "arm": S.arm.__dict__,
        "designs": {name: summarize(d) for name, d in designs.items()},
        "control_only_best_phase1_slack": slack_ctrl[0],
        "co_design_best_phase1_slack": slack_co[0],
        "validation": validation,
        "multistart_logs": {"control_only": log_ctrl, "co_design": log_co},
        "runtime_s": round(time.time() - t_start, 1),
    }
    RES.mkdir(exist_ok=True)
    (RES / "summary.json").write_text(json.dumps(summary, indent=2, default=float) + "\n")
    print(f"  wrote results/summary.json  ({summary['runtime_s']} s)")
    for name, s in summary["designs"].items():
        print(f"  {name:32s} l={np.round(s['lengths_m'], 3)}  J={s['objective_24a']:.3f}  "
              f"safety {s['max_safety_usage']:.2f}  torque {s['max_torque_usage']:.2f}  "
              f"terminal {s['terminal_usage']:.2f}  certified={s['certified']}")


if __name__ == "__main__":
    main()
