"""Set-based robust control co-design (after Bird, Siefert, Pangborn & Jain, 2024).

Everything the optimizer needs is computed from support functions of zonotopes.
The closed-loop arm is linear and discrete, so reachable sets are propagated
exactly as zonotopes (no reduction, no template over-approximation in the
recursion):

    R(0)   = box (operating region, H-rep  H_r x <= f_r)
    R(k+1) = A_cl R(k) (+) E_d W,   A_cl = A_d + B_d K
    U(k)   = K R(k)

For a zonotope Z = <c, G>, rho_Z(l) = l^T c + ||G^T l||_1, which makes every
containment test of the paper (Eq. 5) a closed-form inequality.

Notation follows the paper: Algorithm 1 = ``reach``; Proposition 1 = the three
inequality groups returned by ``constraints``; Problem (24) = ``objective`` +
``constraints``. One deliberate fix: the input constraint is also checked at
k = 0 (the paper's Algorithm 1 starts the input sets at k = 1 while
Proposition 1 needs k = 0..N-1).
"""
from __future__ import annotations

from dataclasses import dataclass, field
import sys
from pathlib import Path

import numpy as np
from scipy.linalg import solve_discrete_are
from scipy.optimize import minimize

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.arm import ArmParameters, DEFAULT, linearize, reach_margin, zoh  # noqa: E402


@dataclass(frozen=True)
class Setup:
    arm: ArmParameters = DEFAULT
    baseline_lengths: tuple = (0.45, 0.45)
    dt: float = 0.01                        # s, ZOH sample time (paper: 0.01 s)
    steps: int = 40                         # N, analysed horizon (0.4 s)
    force_max: float = 0.3                  # N, |F_x|, |F_y| <= force_max (set W)
    r0: tuple = (0.002, 0.002, 0.02, 0.02)  # R(0) half-widths [rad, rad, rad/s, rad/s]
    tip_tolerance: float = 0.003            # m, safety: |dx_tip|, |dy_tip| <= delta
    tip_speed_scale: float = 0.05           # m/s, normalizes tip speed in the cost
    length_bounds: tuple = (0.3, 0.8)       # m, per link
    min_reach_margin: float = 0.05          # keep p_star away from the workspace boundary
    gain_bound: float = 30.0                # |z_ij| bound on normalized gains
    length_cost: float = 1.0                # m_p = length_cost * (l1 + l2) / L_baseline
    gammas: tuple = (1.0, 0.05, 0.01)       # gamma_1 (terminal), gamma_2 (running), gamma_3 (input)


SETUP = Setup()


# --------------------------------------------------------------------------
# Plant, controller parametrization
# --------------------------------------------------------------------------
@dataclass
class Design:
    lengths: np.ndarray
    K: np.ndarray                     # u = K x  (2 x 4)
    lin: object = field(repr=False, default=None)
    Ad: np.ndarray = field(repr=False, default=None)
    Bd: np.ndarray = field(repr=False, default=None)
    Ed: np.ndarray = field(repr=False, default=None)


def gain_scale(setup=SETUP):
    """K_ij = z_ij * tau_i / r0_j: z = 1 spends a full motor on one R(0) half-width."""
    tau = np.asarray(setup.arm.torque_limits, float)
    return np.outer(tau, 1.0 / np.asarray(setup.r0, float))


def unpack(p, setup=SETUP):
    p = np.asarray(p, float)
    return p[:2], p[2:].reshape(2, 4) * gain_scale(setup)


def pack(lengths, K, setup=SETUP):
    return np.r_[lengths, (np.asarray(K) / gain_scale(setup)).ravel()]


def build(p, setup=SETUP) -> Design:
    lengths, K = unpack(p, setup)
    lin = linearize(lengths, setup.arm)
    Ad, BE = zoh(lin.A, np.hstack((lin.B, lin.E)), setup.dt)
    return Design(lengths, K, lin, Ad, BE[:, :2], BE[:, 2:])


def lqr_gain(design_or_lengths, Q, R, setup=SETUP):
    if isinstance(design_or_lengths, Design):
        d = design_or_lengths
    else:
        d = build(np.r_[design_or_lengths, np.zeros(8)], setup)
    X = solve_discrete_are(d.Ad, d.Bd, Q, R)
    return -np.linalg.solve(R + d.Bd.T @ X @ d.Bd, d.Bd.T @ X @ d.Ad)


# --------------------------------------------------------------------------
# Zonotope tube and support functions (Algorithm 1)
# --------------------------------------------------------------------------
def support(center, generators, directions):
    """rho_Z(l) for each row l of `directions` (zonotope <center, generators>)."""
    L = np.atleast_2d(directions)
    return L @ center + np.abs(L @ generators).sum(axis=1)


def tube(design: Design, setup=SETUP):
    """Exact zonotopes R(0..N) as (center, generators) for the closed loop."""
    Acl = design.Ad + design.Bd @ design.K
    W = design.Ed * setup.force_max          # disturbance generators (box W)
    c, G = np.zeros(4), np.diag(setup.r0)
    out = [(c, G)]
    for _ in range(setup.steps):
        c, G = Acl @ c, np.hstack((Acl @ G, W))
        out.append((c, G))
    return out


def safety_directions(design: Design, setup=SETUP):
    """H_x for |J dq| <= delta: rows [+-J_row, 0]; f_x = delta."""
    J = design.lin.J
    H = np.vstack((np.c_[J, np.zeros((2, 2))], -np.c_[J, np.zeros((2, 2))]))
    return H, np.full(4, setup.tip_tolerance)


def region_directions(setup=SETUP):
    H = np.vstack((np.eye(4), -np.eye(4)))
    return H, np.r_[setup.r0, setup.r0]


def cost_map(design: Design, setup=SETUP):
    """Q: state -> [tip position / delta, tip velocity / v_scale] (dimensionless)."""
    J = design.lin.J
    return np.block([[J / setup.tip_tolerance, np.zeros((2, 2))],
                     [np.zeros((2, 2)), J / setup.tip_speed_scale]])


def reach(design: Design, setup=SETUP):
    """Algorithm 1: support values used by the cost and by Proposition 1."""
    sets = tube(design, setup)
    Hx, _ = safety_directions(design, setup)
    Hr, _ = region_directions(setup)
    Q = cost_map(design, setup)
    tau = np.asarray(setup.arm.torque_limits, float)
    Lh_x = np.vstack((np.eye(4), -np.eye(4)))       # axis directions of Q R(k)
    Lh_u = np.vstack((np.eye(2), -np.eye(2)))
    rho_x, rho_lx, rho_u, rho_lu = [], [], [], []
    for k, (c, G) in enumerate(sets):
        rho_lx.append(support(Q @ c, Q @ G, Lh_x))
        if k >= 1:
            rho_x.append(support(c, G, Hx))
        if k <= setup.steps - 1:
            uc, uG = design.K @ c, design.K @ G
            rho_lu.append(support(uc / tau, uG / tau[:, None], Lh_u))
            rho_u.append(support(uc, uG, Lh_u))
    rho_r = support(*sets[-1], Hr)
    return {"sets": sets, "rho_lx": np.array(rho_lx), "rho_x": np.array(rho_x),
            "rho_u": np.array(rho_u), "rho_lu": np.array(rho_lu), "rho_r": rho_r}


def radius_bound(rho_pos_neg):
    """Paper Eq. (21): R2_bar = ||d||_2 + ||c||_2 from axis support values."""
    n = rho_pos_neg.shape[-1] // 2
    up, lo = rho_pos_neg[..., :n], rho_pos_neg[..., n:]
    return np.linalg.norm((up + lo) / 2, axis=-1) + np.linalg.norm((up - lo) / 2, axis=-1)


# --------------------------------------------------------------------------
# Problem (24)
# --------------------------------------------------------------------------
def design_cost(lengths, setup=SETUP):
    return setup.length_cost * float(np.sum(lengths)) / float(np.sum(setup.baseline_lengths))


def objective(p, setup=SETUP, parts=False):
    d = build(p, setup)
    r = reach(d, setup)
    g1, g2, g3 = setup.gammas
    ex = radius_bound(r["rho_lx"])
    eu = radius_bound(r["rho_lu"])
    terms = {"design": design_cost(d.lengths, setup), "terminal": g1 * ex[-1],
             "running_state": g2 * ex[:-1].sum(), "running_input": g3 * eu.sum()}
    total = sum(terms.values())
    return (total, terms) if parts else total


def constraints(p, setup=SETUP):
    """Proposition 1 as g(p) >= 0 (normalized slacks)."""
    d = build(p, setup)
    r = reach(d, setup)
    _, fx = safety_directions(d, setup)
    _, fr = region_directions(setup)
    tau = np.asarray(setup.arm.torque_limits, float)
    safety = 1.0 - (r["rho_x"] / fx).ravel()              # (23a)  k = 1..N
    inputs = 1.0 - (r["rho_u"] / np.r_[tau, tau]).ravel()  # (23b)  k = 0..N-1
    invariance = 1.0 - r["rho_r"] / fr                     # (23c)  R(N) in R(0)
    geometry = np.array([reach_margin(d.lengths, setup.arm.p_star) - setup.min_reach_margin])
    return {"safety": safety, "inputs": inputs, "invariance": invariance, "geometry": geometry}


def certificate(p, setup=SETUP):
    """Worst slack in each group; all >= 0 certifies safety for all time."""
    g = constraints(p, setup)
    return {k: float(v.min()) for k, v in g.items()}


def bounds(setup=SETUP, fixed_lengths=None):
    lo, hi = setup.length_bounds
    lb = [(lo, hi)] * 2 if fixed_lengths is None else [(l, l) for l in fixed_lengths]
    return lb + [(-setup.gain_bound, setup.gain_bound)] * 8


GROUPS = ("safety", "inputs", "invariance", "geometry")


def _stacked(p, setup):
    g = constraints(p, setup)
    return np.concatenate([g[k] for k in GROUPS])


def find_feasible(p0, setup=SETUP, fixed_lengths=None, maxiter=300):
    """Phase 1: maximize the worst normalized slack s.t. g(p) >= s.

    Returns (s*, p*). s* >= 0 means a certified design was found; s* < 0 is the
    least-violating design this local search could reach.
    """
    s0 = float(_stacked(p0, setup).min())
    res = minimize(lambda z: -z[-1], np.r_[p0, s0], method="SLSQP",
                   bounds=bounds(setup, fixed_lengths) + [(-10.0, 1.0)],
                   constraints=[{"type": "ineq", "fun": lambda z: _stacked(z[:-1], setup) - z[-1]}],
                   options={"maxiter": maxiter, "ftol": 1e-10, "eps": 1e-6})
    p = res.x[:-1]
    return float(_stacked(p, setup).min()), p


def solve(p0, setup=SETUP, fixed_lengths=None, maxiter=400):
    """SQP solution of Problem (24) (paper: fmincon SQP; here scipy SLSQP)."""
    history = []
    res = minimize(objective, p0, args=(setup,), method="SLSQP", bounds=bounds(setup, fixed_lengths),
                   constraints=[{"type": "ineq", "fun": _stacked, "args": (setup,)}],
                   callback=lambda xk: history.append(objective(xk, setup)),
                   options={"maxiter": maxiter, "ftol": 1e-10, "eps": 1e-6})
    return res, history
