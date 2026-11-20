"""Deterministic low-dimensional costate-sampling reference calculation."""
from __future__ import annotations

import numpy as np
from scipy.linalg import expm
from scipy.stats import qmc
from scipy.special import ndtri

from reachability import norm_radius


def sample_costate_reachable_points(A, B, C, tau_max, horizon, p,
                                    convention="peak_equivalent", samples=2048,
                                    seed=7, time_samples=500):
    """Sample PMP extremals from Sobol costates and retain norm-feasible ones.

    Sobol coordinates generate directions in R^(2N), while a logarithmic
    amplitude sweep avoids arbitrary dimensional costate boxes. The amplitude
    scale is normalized by ||B diag(tau)||_F and the horizon. Costate sampling
    is intentionally a reference method: its sampling cost and coverage
    deteriorate as the costate dimension 2N grows.
    """
    nstate, ninput = B.shape
    power = int(np.ceil(np.log2(max(2, samples))))
    unit = qmc.Sobol(d=nstate, scramble=True, seed=seed).random_base2(power)[:samples]
    directions = ndtri(np.clip(unit, 1e-10, 1 - 1e-10))
    directions /= np.maximum(np.linalg.norm(directions, axis=1, keepdims=True), 1e-15)
    scale = max(np.linalg.norm(B @ np.diag(tau_max), ord="fro") * horizon, 1e-12)
    amplitudes = scale * np.geomspace(1e-3, 1e3, samples)
    lambdas = directions * amplitudes[:, None]

    times = np.linspace(0.0, horizon, time_samples)
    dt = np.diff(times)
    Dtau = np.diag(tau_max)
    radius = norm_radius(p, ninput, horizon, convention)
    backward_maps = np.asarray([expm(-A.T * time) for time in times])
    costates = np.einsum("tij,sj->sti", backward_maps, lambdas)
    switching = -np.einsum("mi,sti->stm", B.T, costates) @ Dtau.T
    if np.isinf(p):
        control = np.sign(switching)
        raw_norms = np.max(np.abs(control), axis=(1, 2))
    else:
        exponent = 1.0 / (p - 1.0)
        control = np.sign(switching) * np.abs(switching) ** exponent
        integrand = np.sum(np.abs(control) ** p, axis=2)
        raw_norms = np.trapezoid(integrand, times, axis=1) ** (1.0 / p)
    accepted = raw_norms <= radius * (1.0 + 1e-10)

    dt_mean = float(np.mean(dt))
    augmented = np.zeros((nstate + ninput, nstate + ninput))
    augmented[:nstate, :nstate] = A
    augmented[:nstate, nstate:] = B
    transition = expm(augmented * dt_mean)
    Ad, Bd = transition[:nstate, :nstate], transition[:nstate, nstate:]
    midpoint_controls = 0.5 * (control[:, :-1] + control[:, 1:])
    states = np.zeros((len(lambdas), nstate))
    for k in range(len(dt)):
        applied_torque = midpoint_controls[:, k, :] * tau_max[None, :]
        states = states @ Ad.T + applied_torque @ Bd.T
    outputs = states @ C.T
    return {
        "costates": lambdas,
        "terminal_states": states,
        "terminal_outputs": outputs,
        "input_norms": raw_norms,
        "accepted": np.asarray(accepted, dtype=bool),
        "radius": radius,
        "p": p,
        "costate_dimension": nstate,
        "costate_scale": scale,
    }
