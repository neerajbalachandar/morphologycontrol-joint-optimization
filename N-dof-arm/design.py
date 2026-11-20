import numpy as np
from config import CONFIG


# Reference morphology.
BASELINE_LENGTHS = np.asarray(CONFIG.baseline_lengths, dtype=float)
N_BASELINE = len(BASELINE_LENGTHS)

BASELINE_TOTAL_LENGTH = np.sum(BASELINE_LENGTHS)
BASELINE_MEAN_LENGTH = np.mean(BASELINE_LENGTHS)

# Fixed total robot mass.
TOTAL_MASS = CONFIG.total_mass

# Morphology search.
N_MIN = CONFIG.n_min
N_MAX = CONFIG.n_max

# Individual link bounds.
LENGTH_MIN = CONFIG.length_min
LENGTH_MAX = CONFIG.length_max

# Total-size bounds.
TOTAL_LENGTH_MIN = BASELINE_TOTAL_LENGTH
TOTAL_LENGTH_MAX = BASELINE_TOTAL_LENGTH

# Both resource conventions are explicit. Fixed total is the primary study;
# fixed per-joint authority is retained only for sensitivity comparisons.
TAU_PER_BASELINE_JOINT = CONFIG.torque_per_joint
TAU_BUDGET = CONFIG.total_torque

# Relative, dimensionless cost-proxy weights.
# The link term represents stock length/packaging, not material mass (total
# mass is fixed below). Replace these with a BOM and lifecycle estimates for
# a hardware-specific study.
LINK_COST_PER_BASELINE_LENGTH = CONFIG.link_cost_weight
ACTUATOR_COST_PER_BASELINE_JOINT = CONFIG.joint_cost_weight

# Compatibility aliases; new code reads these settings from CONFIG directly.
TASK_EFFORT_WEIGHT = CONFIG.task_effort_weight
REFERENCE_AMPLITUDE = CONFIG.task_reference_amplitude
REQUIRED_IMPROVEMENT = CONFIG.target_gains[1]


def actuator_limits(N, convention=None):
    """Return per-motor limits for ``fixed_total`` or ``fixed_per_joint``.

    The default is the study's primary fixed-total 2 Nm budget. The
    ``fixed_per_joint`` option is a separately labelled sensitivity model.
    """
    if N < 1:
        raise ValueError("N must be at least 1.")
    convention = convention or CONFIG.primary_actuator_convention
    if convention == "fixed_total":
        per_motor_limit = TAU_BUDGET / N
    elif convention == "fixed_per_joint":
        per_motor_limit = TAU_PER_BASELINE_JOINT
    else:
        raise ValueError("convention must be 'fixed_total' or 'fixed_per_joint'.")
    return per_motor_limit * np.ones(N)


def feasible_length_interval(N):
    """Per-link interval after accounting for aggregate length feasibility."""
    if N < 1:
        raise ValueError("N must be at least 1.")
    lo = max(LENGTH_MIN, TOTAL_LENGTH_MIN / N)
    hi = min(LENGTH_MAX, TOTAL_LENGTH_MAX / N)
    if lo > hi:
        raise ValueError(f"No feasible total-length design exists for N={N}.")
    return lo, hi


def design_cost(lengths):
    """Dimensionless design proxy J_m=w_L L/L0+w_N N/N0.

    The proxy accounts for total link envelope and actuator count. Since total
    mass is fixed, it does not pretend that link length is material mass. Its
    coefficients should be replaced with a BOM and lifecycle estimates for a
    hardware-specific study.
    """
    lengths = np.asarray(lengths, dtype=float)
    N = len(lengths)
    return float(
        LINK_COST_PER_BASELINE_LENGTH * np.sum(lengths) / BASELINE_TOTAL_LENGTH
        + ACTUATOR_COST_PER_BASELINE_JOINT * N / N_BASELINE
    )


def length_bounds(N):
    return [(LENGTH_MIN, LENGTH_MAX)] * N
