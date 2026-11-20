"""Compatibility exports for the separate constrained output-tracking study.

The previous identical-joint-angle reference and clipped unconstrained LQT
controller are retired. Use the shared end-effector reference and box-bounded
quadratic tracking implementation below.
"""
from task_tracking import (
    common_reference,
    constrained_tracking_qp,
    optimize_task_morphology,
)

__all__ = ["common_reference", "constrained_tracking_qp", "optimize_task_morphology"]
