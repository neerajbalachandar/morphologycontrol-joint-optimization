"""Fixed-N capability optimizer shared by the primary experiment runner."""
from __future__ import annotations

import numpy as np
from scipy.optimize import LinearConstraint, differential_evolution

from config import CONFIG
from design import BASELINE_TOTAL_LENGTH, TOTAL_MASS, actuator_limits
from model import output_matrix, robot_matrices
from reachability import (
    convex_hull_area,
    lp_boundary_points,
    norm_radius,
    output_scaling,
    reachability_kernel,
)


def feasible_n(n):
    return n >= 1 and n * CONFIG.length_min <= BASELINE_TOTAL_LENGTH <= n * CONFIG.length_max


def reconstruct(n, free, total_length=BASELINE_TOTAL_LENGTH):
    free = np.asarray(free, dtype=float)
    if n == 1:
        return np.array([total_length])
    return np.r_[free, total_length - free.sum()]


def is_valid(lengths, tolerance=1e-8):
    lengths = np.asarray(lengths, dtype=float)
    return bool(
        lengths.ndim == 1 and lengths.size >= 1
        and np.all(np.isfinite(lengths)) and np.all(lengths > 0)
        and np.all(lengths >= CONFIG.length_min - tolerance)
        and np.all(lengths <= CONFIG.length_max + tolerance)
        and abs(lengths.sum() - BASELINE_TOTAL_LENGTH) <= tolerance
    )


def project_box_sum(values, target, lower, upper):
    values = np.asarray(values, dtype=float)
    lo, hi = lower - values.max(), upper - values.min()
    for _ in range(80):
        shift = (lo + hi) / 2
        trial = np.clip(values + shift, lower, upper)
        if trial.sum() < target:
            lo = shift
        else:
            hi = shift
    result = np.clip(values + (lo + hi) / 2, lower, upper)
    residual = target - result.sum()
    for idx in np.argsort((upper - result) if residual > 0 else (result - lower))[::-1]:
        delta = np.sign(residual) * min(abs(residual), (upper - result[idx]) if residual > 0 else (result[idx] - lower))
        result[idx] += delta
        residual -= delta
        if abs(residual) <= 1e-12:
            break
    return result


def evaluate_capability(lengths, criterion="linf_area", convention="peak_equivalent",
                        actuator_convention=None, ntheta=None, nt=None):
    """Compute one objective and reusable geometry/Gramian diagnostics."""
    from config import CONFIG
    from design import actuator_limits
    from reachability import norm_radius

    lengths = np.asarray(lengths, dtype=float)
    A, B, M, K, D, masses, rho = robot_matrices(lengths, total_mass=TOTAL_MASS)
    C = output_matrix(lengths)
    tau = actuator_limits(len(lengths), actuator_convention)
    S = output_scaling(BASELINE_TOTAL_LENGTH, CONFIG.horizon)
    times, kernels = reachability_kernel(
        A, B, C, tau, CONFIG.horizon, nt or CONFIG.time_samples, S
    )
    radius2 = norm_radius(2, len(lengths), CONFIG.horizon, convention)
    W = np.trapezoid(kernels @ np.swapaxes(kernels, 1, 2), times, axis=0)
    scaled_W = radius2**2 * W
    trace = float(np.trace(scaled_W))
    ellipse_area = float(np.pi * np.sqrt(max(np.linalg.det(scaled_W), 0.0)))
    gramian_eigenvalues = np.linalg.eigvalsh(W)
    sign, logdet = np.linalg.slogdet(W)
    if criterion == "l2_trace":
        objective = trace
        geometry = None
    else:
        p = 6.0 if criterion == "l6_area" else np.inf
        ntheta_eff = ntheta or CONFIG.theta_samples
        p_radius = norm_radius(p, len(lengths), CONFIG.horizon, convention)
        # Boundary and support are reconstructed from the already sampled
        # kernel where possible to keep one consistent quadrature per design.
        from reachability import lp_support_samples
        theta, support = lp_support_samples(kernels, times, p, p_radius, ntheta_eff)
        _, boundary = lp_boundary_points(kernels, times, p, p_radius, ntheta_eff)
        geometry = {"theta": theta, "support": support, "boundary": boundary,
                    "area": convex_hull_area(boundary), "gramian": W,
                    "ellipsoid_area": ellipse_area, "radius": p_radius,
                    "p": p, "convention": convention, "nondimensional": True,
                    "times": times, "kernels": kernels}
        objective = geometry["area"]
    return {
        "objective": float(objective),
        "area": float(geometry["area"] if geometry is not None else ellipse_area),
        "physical_output_area": float(
            (geometry["area"] if geometry is not None else ellipse_area)
            * BASELINE_TOTAL_LENGTH**2 / CONFIG.horizon
        ),
        "criterion": criterion,
        "p": 2.0 if criterion == "l2_trace" else (6.0 if criterion == "l6_area" else "infinity"),
        "input_norm_convention": convention,
        "input_norm_radius": float(radius2 if criterion == "l2_trace" else p_radius),
        "actuator_budget_convention": actuator_convention or CONFIG.primary_actuator_convention,
        "horizon": float(CONFIG.horizon),
        "total_length": float(np.sum(lengths)),
        "total_mass": float(TOTAL_MASS),
        "torque_limits": tau,
        "gramian": W,
        "gramian_trace_l2": trace,
        "gramian_eigenvalues": gramian_eigenvalues,
        "gramian_logdet": float(logdet) if sign > 0 else None,
        "l2_ellipsoid_area": ellipse_area,
        "geometry": geometry,
        "A": A, "B": B, "M": M, "K": K, "D": D,
        "masses": masses, "linear_density": float(rho),
    }


def optimize_fixed_n(n, criterion="linf_area", convention="peak_equivalent",
                     actuator_convention=None, seed=None, maxiter=None,
                     popsize=None, tol=None, restarts=1, archive=True):
    """Optimize one continuous fixed-N morphology with total length fixed."""
    if not feasible_n(n):
        return {"N": n, "status": "infeasible", "reason": "N*l_min <= L0 <= N*l_max is false."}
    if n == 1:
        lengths = np.array([BASELINE_TOTAL_LENGTH])
        metrics = evaluate_capability(lengths, criterion, convention, actuator_convention)
        return {"N": n, "status": "evaluated", "lengths": lengths,
                "optimizer_success": True, "message": "N=1 has no free lengths.",
                "nfev": 1, "evaluations": [], **metrics}

    if n * CONFIG.length_min == BASELINE_TOTAL_LENGTH or n * CONFIG.length_max == BASELINE_TOTAL_LENGTH:
        lengths = np.full(n, BASELINE_TOTAL_LENGTH / n)
        metrics = evaluate_capability(lengths, criterion, convention, actuator_convention)
        return {"N": n, "status": "evaluated", "lengths": lengths,
                "optimizer_success": True, "message": "Feasible set is a singleton.",
                "nfev": 1, "evaluations": [], **metrics}

    base_seed = CONFIG.random_seed if seed is None else seed
    best = None
    all_runs, candidates = [], []
    constraint = LinearConstraint(
        np.ones((1, n - 1)),
        BASELINE_TOTAL_LENGTH - CONFIG.length_max,
        BASELINE_TOTAL_LENGTH - CONFIG.length_min,
    )
    bounds = [(CONFIG.length_min, CONFIG.length_max)] * (n - 1)
    for restart in range(max(1, restarts)):
        current_seed = base_seed + restart
        rng = np.random.default_rng(current_seed)
        popsize_eff = popsize or CONFIG.optimizer_popsize
        init = []
        for _ in range(max(5, popsize_eff * (n - 1))):
            full = project_box_sum(
                rng.uniform(CONFIG.length_min, CONFIG.length_max, size=n),
                BASELINE_TOTAL_LENGTH, CONFIG.length_min, CONFIG.length_max,
            )
            init.append(full[:-1])

        def objective(free):
            lengths = reconstruct(n, free)
            if not is_valid(lengths):
                return 1e100
            try:
                record = evaluate_capability(lengths, criterion, convention, actuator_convention)
                value = record["objective"]
                if not np.isfinite(value):
                    return 1e100
                if archive:
                    candidates.append({"lengths": lengths.copy(), "objective": float(value)})
                return -float(value)
            except (ValueError, np.linalg.LinAlgError, FloatingPointError):
                return 1e100

        result = differential_evolution(
            objective, bounds, constraints=(constraint,), seed=current_seed,
            maxiter=maxiter or CONFIG.optimizer_maxiter,
            popsize=popsize_eff, tol=tol or CONFIG.optimizer_tolerance,
            init=np.asarray(init), polish=CONFIG.optimizer_polish,
            strategy=CONFIG.optimizer_strategy, workers=CONFIG.optimizer_workers,
        )
        lengths = reconstruct(n, result.x)
        valid = is_valid(lengths)
        metrics = evaluate_capability(lengths, criterion, convention, actuator_convention) if valid else None
        run = {
            "seed": int(current_seed), "optimizer_success": bool(result.success),
            "message": str(result.message), "nfev": int(result.nfev),
            "objective": metrics["objective"] if metrics else None,
            "valid": bool(valid),
        }
        all_runs.append(run)
        if valid and (best is None or metrics["objective"] > best["objective"]):
            best = {**metrics, "lengths": lengths, "optimizer_success": bool(result.success),
                    "message": str(result.message), "nfev": int(result.nfev)}
    if best is None:
        return {"N": n, "status": "failed", "restarts": all_runs, "evaluations": candidates}
    best.update({"N": n, "status": "optimized", "restarts": all_runs,
                 "evaluations": candidates})
    return best
