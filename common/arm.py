"""Planar serial arm shared by both studies.

The arm moves in a horizontal plane (SCARA-type), so gravity acts along the
joint axes and does not enter the planar dynamics. Link ``i`` is a uniform rod
of length ``l_i`` and mass ``rho * l_i``. A motor of mass ``motor_mass`` sits at
every joint after the first (the base motor is grounded), and a point payload
sits at the tip. Each joint has viscous damping ``damping`` and optional
passive stiffness ``stiffness``.

Nonlinear dynamics
    M(q) qdd + C(q, qd) qd + D qd + K_s q = tau + J(q)^T F_ext

Linearization about a rest configuration (q0, 0) is exact to first order
because the Coriolis term is quadratic in qd:
    x = [dq; dqd],  xdot = A x + B tau + E F_ext
    A = [[0, I], [-M0^-1 K_s, -M0^-1 D]],  B = [0; M0^-1],  E = [0; M0^-1 J0^T]

The rest configuration is obtained from inverse kinematics of a fixed work
point ``p_star`` (two-link elbow-up solution), so changing the link lengths
changes the configuration, the inertia and the Jacobian together.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import expm


@dataclass(frozen=True)
class ArmParameters:
    """Physical constants. Lengths are the design variables, passed separately."""

    linear_density: float = 2.0            # kg/m, link mass per unit length
    motor_mass: float = 0.4                # kg, motor at every joint except the base
    payload_mass: float = 0.25             # kg, point mass at the tip
    damping: float = 4.0                   # N m s/rad, viscous joint damping (geared motor back-EMF)
    stiffness: float = 0.0                 # N m/rad, passive joint stiffness
    torque_limits: tuple = (1.0, 0.5)      # N m, symmetric motor limits (shoulder, elbow)
    p_star: tuple = (0.55, 0.35)           # m, work point the arm is holding
    elbow_up: bool = True

    def with_(self, **changes) -> "ArmParameters":
        return replace(self, **changes)


DEFAULT = ArmParameters()


# --------------------------------------------------------------------------
# Kinematics
# --------------------------------------------------------------------------
def joint_positions(q, lengths):
    """Base, joint and tip positions (N+1, 2) for relative joint angles q."""
    theta = np.cumsum(q)
    steps = np.column_stack((lengths * np.cos(theta), lengths * np.sin(theta)))
    return np.vstack((np.zeros(2), np.cumsum(steps, axis=0)))


def tip_jacobian(q, lengths):
    """2 x N Jacobian of the tip position with respect to relative angles."""
    theta = np.cumsum(q)
    J = np.zeros((2, len(q)))
    for j in range(len(q)):
        J[0, j] = -np.sum(lengths[j:] * np.sin(theta[j:]))
        J[1, j] = np.sum(lengths[j:] * np.cos(theta[j:]))
    return J


def inverse_kinematics(lengths, p_star=DEFAULT.p_star, elbow_up=True, strict=True):
    """Two-link inverse kinematics.

    With ``strict`` an unreachable p_star raises ValueError; otherwise the
    elbow cosine is clipped (a continuous extension used inside optimizers,
    whose reach-margin constraint rejects such designs anyway).
    """
    l1, l2 = lengths
    px, py = p_star
    c2 = (px**2 + py**2 - l1**2 - l2**2) / (2.0 * l1 * l2)
    if not -1.0 <= c2 <= 1.0:
        if strict:
            raise ValueError("work point is outside the workspace")
        c2 = float(np.clip(c2, -1.0, 1.0))
    q2 = np.arccos(c2) * (1.0 if elbow_up else -1.0)
    q1 = np.arctan2(py, px) - np.arctan2(l2 * np.sin(q2), l1 + l2 * np.cos(q2))
    return np.array([q1, q2])


def reach_margin(lengths, p_star=DEFAULT.p_star):
    """Positive when p_star is strictly inside the annular two-link workspace.

    Returns min(l1 + l2 - r, r - |l1 - l2|) / r, a smooth-enough feasibility
    constraint for the optimizers (r = |p_star|).
    """
    l1, l2 = lengths
    r = float(np.hypot(*p_star))
    return min(l1 + l2 - r, r - abs(l1 - l2)) / r


# --------------------------------------------------------------------------
# Inertia
# --------------------------------------------------------------------------
def _bodies(lengths, params):
    """(mass, rotational inertia, link index, offset along that link)."""
    bodies = []
    for i, li in enumerate(lengths):
        m = params.linear_density * li
        bodies.append((m, m * li**2 / 12.0, i, 0.5 * li))
        if i >= 1:
            bodies.append((params.motor_mass, 0.0, i, 0.0))
    bodies.append((params.payload_mass, 0.0, len(lengths) - 1, lengths[-1]))
    return bodies


def mass_matrix(q, lengths, params=DEFAULT):
    lengths = np.asarray(lengths, float)
    n = len(lengths)
    theta = np.cumsum(q)
    M = np.zeros((n, n))
    for m, inertia, link, offset in _bodies(lengths, params):
        # Position of the body: sum of full links before `link` plus offset.
        arm = np.r_[lengths[:link], offset]
        ang = theta[: link + 1]
        Jv = np.zeros((2, n))
        for j in range(link + 1):
            Jv[0, j] = -np.sum(arm[j:] * np.sin(ang[j:]))
            Jv[1, j] = np.sum(arm[j:] * np.cos(ang[j:]))
        Jw = np.zeros(n)
        Jw[: link + 1] = 1.0
        M += m * Jv.T @ Jv + inertia * np.outer(Jw, Jw)
    return M


def total_mass(lengths, params=DEFAULT):
    return float(sum(b[0] for b in _bodies(np.asarray(lengths, float), params)))


# --------------------------------------------------------------------------
# Linear model
# --------------------------------------------------------------------------
@dataclass
class LinearArm:
    lengths: np.ndarray
    q0: np.ndarray
    A: np.ndarray
    B: np.ndarray
    E: np.ndarray
    J: np.ndarray                  # tip Jacobian at q0
    C_task: np.ndarray             # [dx, dy, dxdot, dydot] = C_task x
    torque_limits: np.ndarray
    M: np.ndarray
    params: ArmParameters = field(repr=False)


def linearize(lengths, params=DEFAULT, q0=None):
    """Linear model about rest at q0 (default: IK of params.p_star)."""
    lengths = np.asarray(lengths, float)
    n = len(lengths)
    if q0 is None:
        q0 = (inverse_kinematics(lengths, params.p_star, params.elbow_up, strict=False)
              if n == 2 else np.zeros(n))
    M = mass_matrix(q0, lengths, params)
    Minv = np.linalg.inv(M)
    D = params.damping * np.eye(n)
    Ks = params.stiffness * np.eye(n)
    J = tip_jacobian(q0, lengths)
    Z, I = np.zeros((n, n)), np.eye(n)
    A = np.block([[Z, I], [-Minv @ Ks, -Minv @ D]])
    B = np.vstack((Z, Minv))
    E = np.vstack((np.zeros((n, 2)), Minv @ J.T))
    C_task = np.block([[J, np.zeros((2, n))], [np.zeros((2, n)), J]])
    tau = np.atleast_1d(np.asarray(params.torque_limits, float))
    tau = tau[:n] if tau.size >= n else np.full(n, tau[0])
    return LinearArm(lengths, np.asarray(q0, float), A, B, E, J, C_task, tau, M, params)


def zoh(A, B, dt):
    """Exact zero-order-hold discretization of (A, B)."""
    n, m = B.shape
    blk = np.zeros((n + m, n + m))
    blk[:n, :n], blk[:n, n:] = A, B
    T = expm(blk * dt)
    return T[:n, :n], T[:n, n:]


# --------------------------------------------------------------------------
# Nonlinear simulation (used to check the linear predictions)
# --------------------------------------------------------------------------
def _coriolis(q, qd, lengths, params, h=1e-6):
    """C(q, qd) qd = Mdot qd - 0.5 d/dq (qd^T M qd), by central differences."""
    n = len(q)
    dM = np.empty((n, n, n))
    for k in range(n):
        e = np.zeros(n)
        e[k] = h
        dM[k] = (mass_matrix(q + e, lengths, params) - mass_matrix(q - e, lengths, params)) / (2 * h)
    Mdot = np.einsum("kij,k->ij", dM, qd)
    grad = 0.5 * np.einsum("i,kij,j->k", qd, dM, qd)
    return Mdot @ qd - grad


def simulate_nonlinear(lengths, q0, torque_fn, horizon, params=DEFAULT, force_fn=None,
                       n_eval=200):
    """Integrate the full nonlinear arm from rest at q0 under tau(t) [, F(t)]."""
    lengths = np.asarray(lengths, float)
    n = len(lengths)

    def rhs(t, s):
        q, qd = s[:n], s[n:]
        tau = np.asarray(torque_fn(t), float)
        if force_fn is not None:
            tau = tau + tip_jacobian(q, lengths).T @ np.asarray(force_fn(t), float)
        M = mass_matrix(q, lengths, params)
        rhs_ = tau - _coriolis(q, qd, lengths, params) - params.damping * qd \
            - params.stiffness * (q - q0)
        return np.r_[qd, np.linalg.solve(M, rhs_)]

    t_eval = np.linspace(0.0, horizon, n_eval)
    sol = solve_ivp(rhs, (0.0, horizon), np.r_[q0, np.zeros(n)], t_eval=t_eval,
                    rtol=1e-8, atol=1e-10, max_step=horizon / 400)
    q, qd = sol.y[:n].T, sol.y[n:].T
    tip = np.array([joint_positions(qi, lengths)[-1] for qi in q])
    tip_vel = np.array([tip_jacobian(qi, lengths) @ qdi for qi, qdi in zip(q, qd)])
    return {"t": sol.t, "q": q, "qd": qd, "tip": tip, "tip_vel": tip_vel}


def simulate_nonlinear_sampled(lengths, q0, K, forces, x0, dt, params=DEFAULT):
    """Nonlinear arm under sampled state feedback tau_k = K x_k and held tip forces.

    x = [q - q0, qd]. Torque saturation is NOT applied, so the result can be
    compared one-to-one with the linear prediction.
    """
    lengths = np.asarray(lengths, float)
    n = len(lengths)
    x = np.zeros((len(forces) + 1, 2 * n))
    x[0] = x0
    for k, F in enumerate(forces):
        tau = K @ x[k]

        def rhs(t, s, tau=tau, F=F):
            q, qd = s[:n], s[n:]
            M = mass_matrix(q, lengths, params)
            f = tau + tip_jacobian(q, lengths).T @ F - _coriolis(q, qd, lengths, params) \
                - params.damping * qd - params.stiffness * (q - q0)
            return np.r_[qd, np.linalg.solve(M, f)]

        s0 = np.r_[q0 + x[k, :n], x[k, n:]]
        sol = solve_ivp(rhs, (0.0, dt), s0, rtol=1e-9, atol=1e-12)
        s1 = sol.y[:, -1]
        x[k + 1] = np.r_[s1[:n] - q0, s1[n:]]
    tip = np.array([joint_positions(q0 + xi[:n], lengths)[-1] for xi in x])
    return {"x": x, "tip": tip - joint_positions(q0, lengths)[-1]}
