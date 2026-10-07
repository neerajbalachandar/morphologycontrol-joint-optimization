import numpy as np
from scipy.linalg import expm


def support_function(theta, A, B, C, tau_max, T=2.0, nt=300):
    """Support function of the projected bounded-input reachable set.

    |u_i(t)| <= tau_max[i]

    h(theta) =
        integral_0^T sum_i tau_i,max
        |c(theta)^T C exp(A(T-t)) b_i| dt
    """
    c = np.array([np.cos(theta), np.sin(theta)])
    t = np.linspace(0.0, T, nt)
    values = np.zeros(nt)

    for k, tk in enumerate(t):
        G = C @ expm(A * (T - tk)) @ B
        values[k] = np.sum(tau_max * np.abs(c @ G))

    return np.trapezoid(values, t)


def reachable_boundary(
    A, B, C, tau_max, T=2.0, ntheta=180, nt=300
):
    theta = np.linspace(
        0.0, 2.0 * np.pi, ntheta, endpoint=False
    )
    # Compute the matrix exponential once per quadrature time. Computing it
    # inside every direction made each design evaluation ntheta times slower.
    t = np.linspace(0.0, T, nt)
    responses = np.array([
        C @ expm(A * (T - tk)) @ B for tk in t
    ])
    directions = np.column_stack((np.cos(theta), np.sin(theta)))
    projections = np.einsum("di,tij->tdj", directions, responses)
    weighted_support = np.sum(
        np.abs(projections) * np.asarray(tau_max)[None, None, :], axis=2
    )
    h = np.trapezoid(weighted_support, t, axis=0)
    return theta, h


def support_to_boundary(theta, h):
    """Recover a smooth convex boundary from sampled support values."""
    dtheta = theta[1] - theta[0]
    h_periodic = np.r_[h[-1], h, h[0]]

    dh = (
        h_periodic[2:] - h_periodic[:-2]
    ) / (2.0 * dtheta)

    normal = np.column_stack([
        np.cos(theta),
        np.sin(theta),
    ])
    tangent = np.column_stack([
        -np.sin(theta),
        np.cos(theta),
    ])

    return h[:, None] * normal + dh[:, None] * tangent


def polygon_area(points):
    x = points[:, 0]
    y = points[:, 1]
    return 0.5 * abs(
        np.sum(
            x * np.roll(y, -1)
            - y * np.roll(x, -1)
        )
    )


def reachable_set(
    A, B, C, tau_max, T=2.0, ntheta=180, nt=300
):
    theta, h = reachable_boundary(
        A, B, C, tau_max,
        T=T, ntheta=ntheta, nt=nt
    )
    boundary = support_to_boundary(theta, h)
    area = polygon_area(boundary)
    return theta, h, boundary, area


def output_controllability_gramian(A, B, C, T=2.0, nt=300):
    """Finite-horizon Gramian for the selected output y(T)=C x(T).

    This is the unit-energy reachable ellipsoid matrix. It is distinct from
    the bounded-torque reachable set used by ``reachable_set``.
    """
    t = np.linspace(0.0, T, nt)
    integrand = np.empty((nt, C.shape[0], C.shape[0]))
    for k, ti in enumerate(t):
        G = C @ expm(A * ti) @ B
        integrand[k] = G @ G.T
    return np.trapezoid(integrand, t, axis=0)
