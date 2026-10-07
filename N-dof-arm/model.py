import numpy as np


def robot_matrices(
    lengths,
    total_mass=2.0,
    g=9.81,
    damping=0.05,
):
    """Linearized N-link planar robot about the downward equilibrium.

    q_i are relative joint angles:
        q = [q_1, ..., q_N]

    The total robot mass is fixed for every morphology:
        sum_i m_i = total_mass

    Dynamics:
        M q_ddot + D q_dot + K q = u
        x_dot = A x + B u
    """
    lengths = np.asarray(lengths, dtype=float)
    N = len(lengths)

    if N < 1:
        raise ValueError("N must be at least 1.")
    if np.any(lengths <= 0):
        raise ValueError("All link lengths must be positive.")

    rho = total_mass / np.sum(lengths)
    masses = rho * lengths
    inertia = masses * lengths**2 / 12.0

    M = np.zeros((N, N))

    # Kinematic Jacobians at q = 0.
    for k in range(N):
        Jv = np.zeros((2, N))
        for j in range(k + 1):
            Jv[0, j] = np.sum(lengths[j:k]) + 0.5 * lengths[k]

        Jw = np.zeros(N)
        Jw[:k + 1] = 1.0

        M += (
            masses[k] * Jv.T @ Jv
            + inertia[k] * np.outer(Jw, Jw)
        )

    # Gravity stiffness about the stable downward equilibrium.
    K = np.zeros((N, N))
    for k in range(N):
        for i in range(k + 1):
            for j in range(k + 1):
                r0 = max(i, j)
                lever = np.sum(lengths[r0:k]) + 0.5 * lengths[k]
                K[i, j] += masses[k] * g * lever

    D = damping * np.eye(N)
    Minv = np.linalg.inv(M)

    A = np.block([
        [np.zeros((N, N)), np.eye(N)],
        [-Minv @ K, -Minv @ D],
    ])

    B = np.vstack([
        np.zeros((N, N)),
        Minv,
    ])

    return A, B, M, K, D, masses, rho


def end_effector_jacobian(lengths):
    """Linearized horizontal end-effector position x_EE = J q."""
    lengths = np.asarray(lengths, dtype=float)
    return np.array([np.sum(lengths[j:]) for j in range(len(lengths))])


def output_matrix(lengths):
    """y = [x_EE, xdot_EE] = C x."""
    lengths = np.asarray(lengths, dtype=float)
    N = len(lengths)
    J = end_effector_jacobian(lengths)

    C = np.zeros((2, 2 * N))
    C[0, :N] = J
    C[1, N:] = J
    return C


def kinematic_points(q, lengths):
    """Cartesian positions of base and joints for relative joint angles q."""
    q = np.asarray(q, dtype=float)
    lengths = np.asarray(lengths, dtype=float)

    points = [np.array([0.0, 0.0])]
    theta = 0.0
    position = np.array([0.0, 0.0])

    for qi, li in zip(q, lengths):
        theta += qi
        position = position + np.array([
            li * np.sin(theta),
            -li * np.cos(theta),
        ])
        points.append(position.copy())

    return np.asarray(points)
