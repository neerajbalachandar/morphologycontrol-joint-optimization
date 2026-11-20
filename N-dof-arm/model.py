import numpy as np
from config import CONFIG


def robot_matrices(
    lengths,
    total_mass=CONFIG.total_mass,
    g=CONFIG.gravity,
    damping=CONFIG.viscous_damping,
):
    """Linearized N-link planar robot about the downward equilibrium.

    q_i are relative joint angles:
        q = [q_1, ..., q_N]

    The total robot mass is fixed for every morphology:
        sum_i m_i = total_mass

    Link i has uniform linear density rho=M/sum(l), mass m_i=rho*l_i,
    centre of mass at l_i/2, and slender-link inertia I_i=m_i*l_i**2/12.
    Joint angles are relative; link k's absolute angle is sum(q[:k+1]).
    The centre-of-mass translational Jacobian has horizontal row
    Jv_x[j]=sum(l[j:k])+l[k]/2 for j<=k, and its angular Jacobian is one
    for all upstream joints. These Jacobians give
    M=sum(m_i Jv_i.T Jv_i + I_i Jw_i.T Jw_i). Potential energy is
    V=sum(m_i*g*y_com_i); its Hessian at the downward equilibrium gives K.
    Thus M is symmetric positive definite for positive lengths and masses.
    Damping is the benchmark's diagonal viscous model, D=damping*I.

    Dynamics:
        M q_ddot + D q_dot + K q = u
        x_dot = A x + B u

    The input vector is generalized relative-joint torque, so B=[0; M^-1].
    This rigid-link benchmark is not an exact continuum soft arm.
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
