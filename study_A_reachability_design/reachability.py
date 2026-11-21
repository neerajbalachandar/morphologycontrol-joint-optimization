"""Finite-horizon reachable sets of LTI systems (after Nguyen, Cortes & Kramer).

System  xdot = A x + B u,  x(0) = 0,  y = C x,  horizon T.
All inputs are normalized, u = diag(u_max) v, so the kernel is

    G(t) = C e^{A (T - t)} B diag(u_max)          (ny x m)

and every result below is stated for the normalized input v.

Implemented results (paper numbering):
  Lemma 1      exposed points are reached by bang-bang inputs
               v_i(t) = v_hi if psi_i(t; c) >= 0 else v_lo,  psi(t; c) = c^T G(t)
  Prop. 1      planar, single-input, real distinct eigenvalues -> <= 1 switch
  Theorem 2    exact boundary  R = conv(G1 u G2) from one-switch curves xi_1, xi_2
  Theorem 3    L^p-optimal input  v*(t) = |-B^T e^{-A^T t} lam0|^{1/(p-1)} sgn(.)
  Prop. 2      ||lam0||_q^q <= 1 / (m int ||vec(e^{-A t} B)||_p^q dt)  =>  ||v||_Lp <= 1
plus the L^2 output Gramian (trace metric), a zero-order-hold zonotope inner
approximation of the L^inf set, and convex-hull volumes.
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import expm
from scipy.spatial import ConvexHull
from scipy.stats import norm, qmc


# --------------------------------------------------------------------------
# Kernel and quadrature
# --------------------------------------------------------------------------
def time_grid(T, nt):
    t = np.linspace(0.0, T, nt)
    w = np.full(nt, t[1] - t[0])
    w[[0, -1]] *= 0.5
    return t, w


def kernel(A, B, C, T, nt=401, u_max=None):
    """G(t_k) = C e^{A(T-t_k)} B diag(u_max) on a uniform grid (exact transition)."""
    t, w = time_grid(T, nt)
    Bs = B if u_max is None else B * np.asarray(u_max, float)
    step = expm(A * (t[1] - t[0]))
    G = np.empty((nt, C.shape[0], B.shape[1]))
    Phi = np.eye(A.shape[0])                     # e^{A(T - t_k)} for k = nt-1 .. 0
    for k in range(nt - 1, -1, -1):
        G[k] = C @ Phi @ Bs
        Phi = step @ Phi
    return t, w, G


def sphere_directions(n_dirs, dim, seed=0):
    """Deterministic, well-spread unit directions (scrambled Sobol -> Gaussian)."""
    if dim == 2:
        th = np.linspace(0, 2 * np.pi, n_dirs, endpoint=False)
        return np.column_stack((np.cos(th), np.sin(th)))
    m = int(np.ceil(np.log2(max(n_dirs, 2))))
    u = qmc.Sobol(dim, scramble=True, seed=seed).random_base2(m)[:n_dirs]
    z = norm.ppf(np.clip(u, 1e-12, 1 - 1e-12))
    return z / np.linalg.norm(z, axis=1, keepdims=True)


# --------------------------------------------------------------------------
# Lemma 1: bang-bang extremals of the (possibly asymmetric) box
# --------------------------------------------------------------------------
def bang_bang_points(G, w, directions, v_lo=-1.0, v_hi=1.0):
    """Endpoints y(T) of the Lemma-1 controls for each costate direction c."""
    psi = np.einsum("tki,sk->sti", G, directions)            # psi_i(t; c)
    v = np.where(psi >= 0.0, v_hi, v_lo)
    return np.einsum("tki,sti,t->sk", G, v, w)


def support_linf(G, w, directions, v_lo=-1.0, v_hi=1.0):
    """h(c) = int sum_i max(v_lo psi_i, v_hi psi_i) dt (support of the box set)."""
    psi = np.einsum("tki,sk->sti", G, directions)
    return np.einsum("sti,t->s", np.maximum(v_lo * psi, v_hi * psi), w)


def switch_counts(G, directions, tol=1e-12):
    """Number of sign changes of every psi_i(t; c), shape (n_dirs, m)."""
    psi = np.einsum("tki,sk->sti", G, directions)
    s = np.sign(np.where(np.abs(psi) < tol, 0.0, psi))
    # carry the last non-zero sign forward so exact zeros do not count twice
    for k in range(1, s.shape[1]):
        s[:, k] = np.where(s[:, k] == 0, s[:, k - 1], s[:, k])
    return np.sum(s[:, 1:] * s[:, :-1] < 0, axis=1)


# --------------------------------------------------------------------------
# Theorem 2: exact planar single-input boundary
# --------------------------------------------------------------------------
def _input_integral(A, b, s):
    """F(s) = int_0^s e^{A sigma} b d sigma, exactly, via an augmented exponential."""
    n = A.shape[0]
    M = np.zeros((n + 1, n + 1))
    M[:n, :n], M[:n, n] = A, b
    return expm(M * s)[:n, n]


def theorem2_boundary(A, b, T, u_lo, u_hi, n_eta=400):
    """xi_1(eta), xi_2(eta) for eta in [0, T]; R = conv(G1 u G2).

    int_0^eta e^{A(T-tau)} b dtau = F(T) - F(T - eta)
    int_eta^T e^{A(T-tau)} b dtau = F(T - eta)
    """
    eta = np.linspace(0.0, T, n_eta)
    FT = _input_integral(A, b, T)
    tail = np.array([_input_integral(A, b, T - e) for e in eta])
    head = FT - tail
    xi1 = u_hi * head + u_lo * tail
    xi2 = u_lo * head + u_hi * tail
    return eta, xi1, xi2


def planar_assumptions(A, B):
    """Check the hypotheses of Prop. 1 / Theorem 2."""
    ev = np.linalg.eigvals(A)
    return {"dimension_2": A.shape == (2, 2), "single_input": B.shape[1] == 1,
            "real_eigenvalues": bool(np.all(np.abs(ev.imag) < 1e-12)),
            "distinct": bool(abs(ev[0] - ev[1]) > 1e-9), "eigenvalues": ev.real.tolist()}


# --------------------------------------------------------------------------
# Theorem 3 and Proposition 2: L^p inputs from sampled initial costates
# --------------------------------------------------------------------------
def lp_costate_rollout(A, B, T, lam0, p, nt=801):
    """Theorem 3 inputs for each initial costate lam0 (rows); returns endpoints and norms.

    v*(t) = |-B^T e^{-A^T t} lam0|^{1/(p-1)} sgn(-B^T e^{-A^T t} lam0)
    x(T)  = int_0^T e^{A(T-t)} B v*(t) dt
    ||v||_Lp = (int_0^T sum_i |v_i|^p dt)^{1/p}
    """
    t, w = time_grid(T, nt)
    lam0 = np.atleast_2d(lam0)
    step_back = expm(-A.T * (t[1] - t[0]))
    step_fwd = expm(A * (t[1] - t[0]))
    lam = np.empty((nt, A.shape[0], lam0.shape[0]))
    lam[0] = lam0.T
    for k in range(1, nt):
        lam[k] = step_back @ lam[k - 1]                        # e^{-A^T t_k} lam0
    s = -np.einsum("ni,tns->tsi", B, lam)                       # -B^T lam(t)
    v = np.sign(s) * np.abs(s) ** (1.0 / (p - 1.0))
    Phi = np.empty((nt,) + A.shape)                             # e^{A(T - t_k)}
    P = np.eye(A.shape[0])
    for k in range(nt - 1, -1, -1):
        Phi[k] = P
        P = step_fwd @ P
    xf = np.einsum("t,tij,jm,tsm->si", w, Phi, B, v)
    cost = (np.einsum("t,tsi->s", w, np.abs(v) ** p)) ** (1.0 / p)
    return xf, cost


def prop2_bound(A, B, T, p, nt=801):
    """Right-hand side of Prop. 2: ||lam0||_q^q <= 1 / (m int ||vec(e^{-At}B)||_p^q dt)."""
    q = p / (p - 1.0)
    t, w = time_grid(T, nt)
    step = expm(-A * (t[1] - t[0]))
    P = np.eye(A.shape[0])
    vals = np.empty(nt)
    for k in range(nt):
        vals[k] = np.linalg.norm((P @ B).ravel(), ord=p) ** q
        P = step @ P
    return 1.0 / (B.shape[1] * float(w @ vals))


def lp_boundary_points(G, w, directions, p, radius=1.0):
    """Exact L^p boundary points maximizing c^T y(T) (Theorem 3 with lam(T) = c).

    Equivalent to sampling lam0 = e^{A^T T} c and rescaling to unit norm; used for
    volume metrics because every point lies on the boundary.
    """
    q = p / (p - 1.0)
    psi = np.einsum("tki,sk->sti", G, directions)
    Hq = np.einsum("sti,t->s", np.abs(psi) ** q, w) ** (1.0 / q)
    v = radius * np.sign(psi) * np.abs(psi) ** (q - 1.0) / Hq[:, None, None] ** (q - 1.0)
    return np.einsum("tki,sti,t->sk", G, v, w)


# --------------------------------------------------------------------------
# Zonotope inner approximation (piecewise-constant inputs)
# --------------------------------------------------------------------------
def zoh_zonotope(A, B, T, steps, u_lo, u_hi):
    """Reachable set of held inputs on `steps` intervals: an exact zonotope that is
    an inner approximation of the continuous-time box-input set."""
    n, m = B.shape
    blk = np.zeros((n + m, n + m))
    blk[:n, :n], blk[:n, n:] = A, B
    E = expm(blk * (T / steps))
    Ad, Bd = E[:n, :n], E[:n, n:]
    u_lo, u_hi = np.broadcast_to(u_lo, (m,)), np.broadcast_to(u_hi, (m,))
    mid, half = (u_hi + u_lo) / 2, (u_hi - u_lo) / 2
    center, gens, P = np.zeros(n), [], np.eye(n)
    for _ in range(steps):
        center = center + P @ Bd @ mid
        gens.append(P @ Bd * half)
        P = Ad @ P
    return center, np.hstack(gens)


def zonotope_polygon(center, generators, n_dirs=720):
    th = np.linspace(0, 2 * np.pi, n_dirs, endpoint=False)
    D = np.column_stack((np.cos(th), np.sin(th)))
    pts = center + np.sign(D @ generators) @ generators.T
    return pts[ConvexHull(pts).vertices]


# --------------------------------------------------------------------------
# Set metrics
# --------------------------------------------------------------------------
def gramian(G, w):
    return np.einsum("t,tki,tli->kl", w, G, G)


def hull_volume(points):
    return float(ConvexHull(points).volume)


def hull_polygon(points2d):
    h = ConvexHull(points2d)
    v = points2d[h.vertices]
    return np.vstack((v, v[:1]))
