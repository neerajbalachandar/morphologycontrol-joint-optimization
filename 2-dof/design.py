import numpy as np


# Reference morphology.
BASELINE_LENGTHS = np.array([1.0, 1.0])
N_BASELINE = len(BASELINE_LENGTHS)

BASELINE_TOTAL_LENGTH = np.sum(BASELINE_LENGTHS)
BASELINE_MEAN_LENGTH = np.mean(BASELINE_LENGTHS)

# Fixed total robot mass.
TOTAL_MASS = 2.0

# Morphology search.
N_MIN = 1
N_MAX = 5

# Individual link bounds.
LENGTH_MIN = 0.5 * BASELINE_MEAN_LENGTH
LENGTH_MAX = 1.5 * BASELINE_MEAN_LENGTH

# Total-size bounds.
TOTAL_LENGTH_MIN = 0.5 * BASELINE_TOTAL_LENGTH
TOTAL_LENGTH_MAX = 1.5 * BASELINE_TOTAL_LENGTH

# Fixed total actuator authority.
TAU_PER_BASELINE_JOINT = 1.0
TAU_BUDGET = N_BASELINE * TAU_PER_BASELINE_JOINT

# Relative cost-proxy weights, normalized against the baseline design.
# The link term represents stock length/packaging, not material mass (total
# mass is fixed below). Replace these with a BOM and lifecycle estimates for
# a hardware-specific study.
LINK_COST_PER_BASELINE_LENGTH = 1.0
ACTUATOR_COST_PER_BASELINE_JOINT = 2.0

# Task-design trade-offs. These are explicit tuning parameters, not measured
# prices or safety limits.
TASK_EFFORT_WEIGHT = 0.05
TASK_DESIGN_WEIGHT = 0.05
REFERENCE_AMPLITUDE = 0.15
REQUIRED_IMPROVEMENT = 0.10
MAX_LINEARIZED_ANGLE = 0.35  # rad; keep trajectories near the linearization point


def actuator_limits(N):
    """Equal allocation of the fixed total torque budget."""
    return (TAU_BUDGET / N) * np.ones(N)


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
    """Dimensionless manufacturing proxy, normalized to the baseline.

    The proxy accounts for total link envelope and actuator count. Since total
    mass is fixed, it does not pretend that link length is material mass. Its
    coefficients should be replaced with a BOM and lifecycle estimates for a
    hardware-specific study.
    """
    lengths = np.asarray(lengths, dtype=float)
    N = len(lengths)
    raw = (
        LINK_COST_PER_BASELINE_LENGTH * np.sum(lengths) / BASELINE_TOTAL_LENGTH
        + ACTUATOR_COST_PER_BASELINE_JOINT * N / N_BASELINE
    )
    baseline_raw = LINK_COST_PER_BASELINE_LENGTH + ACTUATOR_COST_PER_BASELINE_JOINT
    return float(raw / baseline_raw)


def length_bounds(N):
    return [(LENGTH_MIN, LENGTH_MAX)] * N
