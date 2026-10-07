import numpy as np
from scipy.linalg import expm
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d


def desired_state(t, N, amplitude=0.15, omega=1.0):
    """Reference state [q_d, qdot_d]."""
    amplitudes = amplitude * np.ones(N)
    q = amplitudes * np.sin(omega * t)
    qdot = amplitudes * omega * np.cos(omega * t)
    return np.r_[q, qdot]


def finite_horizon_lqt(A, B, Q, R, T, dt, N):
    """Solve the finite-horizon linear-quadratic tracking problem."""
    n = A.shape[0]
    Qf = Q.copy()
    Rinv = np.linalg.inv(R)

    def rhs(t, z):
        P = z[:n * n].reshape(n, n)
        s = z[n * n:]
        xd = desired_state(t, N)

        Pdot = -(
            A.T @ P
            + P @ A
            - P @ B @ Rinv @ B.T @ P
            + Q
        )

        sdot = -(
            A.T - P @ B @ Rinv @ B.T
        ) @ s + Q @ xd

        return np.r_[Pdot.ravel(), sdot]

    xd_T = desired_state(T, N)
    zT = np.r_[Qf.ravel(), -Qf @ xd_T]

    t_backward = np.arange(T, 0.0, -dt)
    if len(t_backward) == 0 or not np.isclose(t_backward[-1], 0.0):
        t_backward = np.r_[t_backward, 0.0]

    sol = solve_ivp(
        rhs,
        [T, 0.0],
        zT,
        t_eval=t_backward,
        rtol=1e-7,
        atol=1e-9,
    )
    if not sol.success:
        raise RuntimeError(f"LQT Riccati integration failed: {sol.message}")

    t = sol.t[::-1]
    P = sol.y[:n * n, ::-1].T.reshape(-1, n, n)
    s = sol.y[n * n:, ::-1].T

    return t, P, s


def simulate_task(A, B, Q, R, T=5.0, dt=0.01, u_max=None):
    """Simulate sampled-data finite-horizon LQT from x(0)=0.

    Exact zero-order-hold discretization and a backward Riccati recursion
    avoid nested adaptive ODE integrations during design optimization.
    ``u_max`` may be a scalar or per-actuator vector; limits are enforced on
    the torque actually applied to the plant.
    """
    n = A.shape[0]
    N = n // 2
    if T <= 0 or dt <= 0:
        raise ValueError("T and dt must be positive.")
    if n != 2 * B.shape[1] or N < 1:
        raise ValueError("Expected a second-order state and matching input matrix.")

    steps = max(1, int(np.ceil(T / dt)))
    h = T / steps
    t = np.linspace(0.0, T, steps + 1)

    # Exact continuous-to-discrete conversion for zero-order-held input.
    m = B.shape[1]
    augmented = np.zeros((n + m, n + m))
    augmented[:n, :n] = A
    augmented[:n, n:] = B
    transition = expm(augmented * h)
    Ad = transition[:n, :n]
    Bd = transition[:n, n:]

    Q = np.asarray(Q, dtype=float)
    R = np.asarray(R, dtype=float)
    if Q.shape != (n, n) or R.shape != (m, m):
        raise ValueError("Q and R dimensions must match the state and input.")
    if u_max is None:
        limits = None
    else:
        limits = np.broadcast_to(np.asarray(u_max, dtype=float), (m,))
        if np.any(limits <= 0):
            raise ValueError("Torque limits must be positive.")

    XD = np.array([desired_state(tk, N) for tk in t])
    Q_step = h * Q
    R_step = h * R
    P = Q.copy()
    s = -Q @ XD[-1]
    gains = [None] * steps
    offsets = [None] * steps

    # Discrete affine Riccati recursion for tracking a known reference.
    for k in range(steps - 1, -1, -1):
        H = R_step + Bd.T @ P @ Bd
        K = np.linalg.solve(H, Bd.T @ P @ Ad)
        d = np.linalg.solve(H, Bd.T @ s)
        gains[k] = K
        offsets[k] = d

        P_old = Q_step + Ad.T @ P @ Ad - Ad.T @ P @ Bd @ K
        s_old = -Q_step @ XD[k] + Ad.T @ s - Ad.T @ P @ Bd @ d
        P = 0.5 * (P_old + P_old.T)
        s = s_old

    X = np.zeros((steps + 1, n))
    U = np.zeros((steps + 1, m))
    for k in range(steps):
        uk = -(gains[k] @ X[k] + offsets[k])
        if limits is not None:
            uk = np.clip(uk, -limits, limits)
        U[k] = uk
        X[k + 1] = Ad @ X[k] + Bd @ uk
    # The terminal control is undefined; retain zero at the terminal sample.
    return t, X, U


def tracking_metrics(t, X, U, N, Q, R):
    XD = np.array([desired_state(ti, N) for ti in t])
    E = X - XD
    q_error = E[:, :N]

    rms_error = np.sqrt(
        np.trapezoid(np.sum(q_error**2, axis=1), t)
        / (t[-1] - t[0])
    )

    max_error = np.max(
        np.linalg.norm(q_error, axis=1)
    )

    tracking_cost = np.trapezoid(
        np.einsum("ti,ij,tj->t", E, Q, E),
        t,
    )

    control_density = np.einsum("ti,ij,tj->t", U, R, U)
    # Controls are held constant over [t[k], t[k+1]); integrate those
    # applied commands directly instead of trapezoiding an artificial u(T).
    control_cost = np.sum(control_density[:-1] * np.diff(t))

    return {
        "rms_error": rms_error,
        "max_error": max_error,
        "tracking_cost": tracking_cost,
        "control_cost": control_cost,
        "total_cost": tracking_cost + control_cost,
    }
