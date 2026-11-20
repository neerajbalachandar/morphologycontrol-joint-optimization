"""Optional outer-zonotope propagation for small-N validation cases."""
from __future__ import annotations

import numpy as np
from scipy.linalg import expm
from scipy.spatial import ConvexHull


def exact_zoh_matrices(A, B, dt):
    """Exact zero-order-hold discretization using one augmented exponential."""
    n, m = B.shape
    block = np.zeros((n + m, n + m))
    block[:n, :n] = A
    block[:n, n:] = B
    transition = expm(block * dt)
    return transition[:n, :n], transition[:n, n:]


def reduce_outer_generators(generators, max_generators):
    """Bound discarded generators by an axis-aligned box (outer approximation).

    Retained generators are selected by Euclidean norm. For each state axis,
    discarded generators' absolute components are summed to create a box
    generator. This box contains their Minkowski sum, so reduction preserves
    outer inclusion.
    """
    if generators.shape[1] <= max_generators:
        return generators
    norms = np.linalg.norm(generators, axis=0)
    keep_count = max(0, max_generators - generators.shape[0])
    kept_idx = np.argsort(norms)[-keep_count:] if keep_count else np.array([], dtype=int)
    discard_idx = np.setdiff1d(np.arange(generators.shape[1]), kept_idx, assume_unique=True)
    kept = generators[:, kept_idx]
    box = np.diag(np.sum(np.abs(generators[:, discard_idx]), axis=1))
    return np.column_stack((kept, box))


def propagate_outer_zonotope(A, B, tau_max, horizon, steps=80,
                             max_generators=120, output=None):
    """Propagate an outer zonotope from zero for sampled box-valued inputs.

    The discretized held-input reachable set is represented by
    Z={c+G xi: ||xi||_inf<=1}. With generator reduction, the result remains an
    outer approximation. Exact finite-time continuous-input inclusion is not
    asserted; temporal hold discretization is separately reported.
    """
    if steps < 1:
        raise ValueError("steps must be positive")
    dt = horizon / steps
    Ad, Bd = exact_zoh_matrices(A, B, dt)
    input_generators = Bd @ np.diag(tau_max)
    center = np.zeros(A.shape[0])
    generators = np.empty((A.shape[0], 0))
    for _ in range(steps):
        center = Ad @ center
        generators = np.column_stack((Ad @ generators, input_generators))
        if max_generators is not None:
            generators = reduce_outer_generators(generators, max_generators)
    if output is not None:
        center = output @ center
        generators = output @ generators
        # Projection may create redundant generators; no reduction is required
        # for the returned 2-D outer polygon approximation.
    return {"center": center, "generators": generators,
            "steps": steps, "dt": dt, "inclusion": "outer"}


def projected_zonotope_boundary(center, generators, directions=360):
    """Sample support points of a 2-D zonotope's exact support map."""
    theta = np.linspace(0, 2 * np.pi, directions, endpoint=False)
    points = np.empty((directions, 2))
    for i, angle in enumerate(theta):
        s = np.array([np.cos(angle), np.sin(angle)])
        projection = s @ generators
        points[i] = center + generators @ np.sign(projection)
    return theta, points


def polygon_area(points):
    points = np.asarray(points)
    try:
        return float(ConvexHull(points).volume)
    except Exception:
        return 0.0
