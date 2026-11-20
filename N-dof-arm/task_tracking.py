"""Separate constrained tracking QP for a common end-effector reference."""
from __future__ import annotations

import numpy as np
from scipy.linalg import expm
from scipy.optimize import minimize

from config import CONFIG
from design import BASELINE_TOTAL_LENGTH, TOTAL_MASS, actuator_limits
from model import output_matrix, robot_matrices


def common_reference(times, amplitude=CONFIG.task_reference_amplitude,
                     omega=CONFIG.task_reference_frequency,
                     x0=CONFIG.task_reference_offset):
    """Physical horizontal end-effector reference [x, xdot] for all morphologies."""
    phase = omega * np.asarray(times)
    return np.column_stack((x0 + amplitude * np.sin(phase),
                            amplitude * omega * np.cos(phase)))


def constrained_tracking_qp(lengths, horizon=CONFIG.task_horizon, steps=CONFIG.task_steps,
                            actuator_convention="fixed_total", effort_weight=CONFIG.task_effort_weight,
                            amplitude=CONFIG.task_reference_amplitude,
                            omega=CONFIG.task_reference_frequency):
    """Solve a finite-horizon convex tracking QP with box-bounded motor torque.

    Exact ZOH dynamics are condensed into the objective; L-BFGS-B solves the
    resulting convex quadratic with torque bounds. The physical output task
    uses the linearized [x_EE, xdot_EE] map; full planar forward kinematics is
    available separately from ``model.kinematic_points`` for configuration
    visualization.
    """
    lengths = np.asarray(lengths, dtype=float)
    A, B, *_ = robot_matrices(lengths, total_mass=TOTAL_MASS)
    C = output_matrix(lengths)
    dt = horizon / steps
    nstate, ninput = B.shape
    augmented = np.zeros((nstate + ninput, nstate + ninput))
    augmented[:nstate, :nstate] = A
    augmented[:nstate, nstate:] = B
    transition = expm(augmented * dt)
    Ad, Bd = transition[:nstate, :nstate], transition[:nstate, nstate:]
    times = np.linspace(0.0, horizon, steps + 1)
    reference = common_reference(times, amplitude, omega)
    limits = actuator_limits(ninput, actuator_convention)

    # Precompute output response from each held input at each future sample.
    powers = [np.eye(nstate)]
    for _ in range(steps):
        powers.append(Ad @ powers[-1])
    response = [[None for _ in range(steps)] for _ in range(steps + 1)]
    for k in range(1, steps + 1):
        for j in range(k):
            response[k][j] = C @ powers[k - 1 - j] @ Bd
    weights = np.diag([1.0 / BASELINE_TOTAL_LENGTH**2,
                       (horizon / BASELINE_TOTAL_LENGTH)**2])

    def rollout(flat_u, with_gradient=False):
        U = np.asarray(flat_u).reshape(steps, ninput)
        X = np.zeros((steps + 1, nstate))
        for k in range(steps):
            X[k + 1] = Ad @ X[k] + Bd @ U[k]
        Y = X @ C.T
        error = Y - reference
        tracking = 0.5 * dt * np.einsum("ti,ij,tj->", error, weights, error)
        effort = 0.5 * dt * effort_weight * np.sum(U**2)
        value = tracking + effort
        if not with_gradient:
            return value
        output_gradient = dt * error @ weights
        grad = np.zeros_like(U)
        for j in range(steps):
            for k in range(j + 1, steps + 1):
                grad[j] += response[k][j].T @ output_gradient[k]
        grad += dt * effort_weight * U
        return value, grad.reshape(-1)

    x0 = np.zeros((steps, ninput))
    bounds = [(-limits[j], limits[j]) for _ in range(steps) for j in range(ninput)]
    result = minimize(lambda u: rollout(u, True), x0.reshape(-1), jac=True,
                      method="L-BFGS-B", bounds=bounds,
                      options={"maxiter": CONFIG.task_qp_maxiter,
                               "ftol": CONFIG.task_qp_tolerance})
    U = result.x.reshape(steps, ninput)
    X = np.zeros((steps + 1, nstate))
    for k in range(steps):
        X[k + 1] = Ad @ X[k] + Bd @ U[k]
    Y = X @ C.T
    error = Y - reference
    rms = np.sqrt(np.trapezoid(np.sum(error**2, axis=1), times) / horizon)
    tracking_cost = float(0.5 * dt * np.einsum("ti,ij,tj->", error, weights, error))
    control_effort = float(0.5 * dt * effort_weight * np.sum(U**2))
    return {
        "lengths": lengths,
        "times": times,
        "states": X,
        "torques": U,
        "output": Y,
        "reference": reference,
        "tracking_error": error,
        "rms_output_error": float(rms),
        "rms_position_error": float(np.sqrt(np.trapezoid(error[:, 0]**2, times) / horizon)),
        "rms_velocity_error": float(np.sqrt(np.trapezoid(error[:, 1]**2, times) / horizon)),
        "tracking_cost": tracking_cost,
        "control_effort_cost": control_effort,
        "objective": float(result.fun),
        "optimizer_success": bool(result.success),
        "optimizer_message": str(result.message),
        "torque_limits": limits,
        "horizon": float(horizon),
        "steps": int(steps),
        "reference_amplitude": float(amplitude),
        "reference_frequency": float(omega),
        "actuator_budget_convention": actuator_convention,
        "total_length": float(lengths.sum()),
        "total_mass": float(TOTAL_MASS),
        "full_planar_kinematics_note": "Use model.kinematic_points(q, lengths) for nonlinear planar coordinates.",
    }


def optimize_task_morphology(n, initial_lengths=None, horizon=CONFIG.task_horizon, steps=CONFIG.task_steps,
                             actuator_convention="fixed_total"):
    """Task-oriented morphology comparison using the common output-tracking QP."""
    from scipy.optimize import minimize as continuous_minimize
    from optimization_fixed_n import feasible_n, project_box_sum, reconstruct, is_valid

    if not feasible_n(n):
        raise ValueError(f"N={n} is infeasible for the configured fixed total length.")
    if n == 1:
        return constrained_tracking_qp([BASELINE_TOTAL_LENGTH], horizon, steps, actuator_convention)
    if initial_lengths is None:
        initial_lengths = np.full(n, BASELINE_TOTAL_LENGTH / n)

    def objective(free):
        lengths = reconstruct(n, free)
        if not is_valid(lengths):
            return 1e12
        return constrained_tracking_qp(lengths, horizon, steps, actuator_convention)["objective"]

    constraint = {"type": "ineq", "fun": lambda z: np.array([
        z.sum() - (BASELINE_TOTAL_LENGTH - CONFIG.length_max),
        BASELINE_TOTAL_LENGTH - CONFIG.length_min - z.sum(),
    ])}
    initial = project_box_sum(np.asarray(initial_lengths), BASELINE_TOTAL_LENGTH,
                              CONFIG.length_min, CONFIG.length_max)[:-1]
    result = continuous_minimize(
        objective, initial, method="SLSQP",
        bounds=[(CONFIG.length_min, CONFIG.length_max)] * (n - 1),
        constraints=(constraint,), options={
            "maxiter": CONFIG.task_morphology_maxiter,
            "ftol": CONFIG.task_morphology_tolerance,
        },
    )
    lengths = reconstruct(n, result.x)
    record = constrained_tracking_qp(lengths, horizon, steps, actuator_convention)
    record["morphology_optimizer_success"] = bool(result.success)
    record["morphology_optimizer_message"] = str(result.message)
    return record
