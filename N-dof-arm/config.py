"""Single source of physical and numerical settings for the N-link study."""
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class StudyConfig:
    baseline_lengths: tuple[float, ...] = (2.0, 2.0)
    total_mass: float = 2.0
    gravity: float = 9.81
    viscous_damping: float = 0.05
    n_min: int = 1
    n_max: int = 5
    length_min: float = 0.2
    length_max: float = 4.0
    total_torque: float = 2.0
    torque_per_joint: float = 1.0
    primary_actuator_convention: str = "fixed_total"
    link_cost_weight: float = 1.0
    joint_cost_weight: float = 2.0
    horizon: float = 2.0
    theta_samples: int = 180
    time_samples: int = 300
    plot_theta_samples: int = 240
    plot_time_samples: int = 400
    convergence_resolutions: tuple[tuple[int, int], ...] = ((60, 120), (120, 240), (240, 400), (360, 600))
    analytical_sweep_points: int = 64
    redistribution_grid_points: int = 31
    redistribution_random_samples: int = 100
    costate_samples: int = 512
    costate_time_samples: int = 300
    zonotope_steps: int = 50
    zonotope_max_generators: int = 30
    p_values: tuple = (2.0, 4.0, 6.0, 10.0, "infinity")
    norm_convention: str = "peak_equivalent"
    task_horizon: float = 5.0
    task_steps: int = 125
    task_effort_weight: float = 1e-2
    task_qp_maxiter: int = 1000
    task_qp_tolerance: float = 1e-10
    task_morphology_maxiter: int = 50
    task_morphology_tolerance: float = 1e-5
    task_reference_amplitude: float = 0.12
    task_reference_frequency: float = 1.0
    task_reference_offset: float = 0.0
    target_gains: tuple[float, ...] = (0.05, 0.10, 0.20, 0.30)
    analytical_length_min: float = 0.5
    analytical_length_max: float = 4.0
    optimizer_maxiter: int = 200
    optimizer_popsize: int = 12
    optimizer_tolerance: float = 1e-3
    optimizer_strategy: str = "best1bin"
    optimizer_polish: bool = False
    optimizer_workers: int = 1
    random_seed: int = 7

    @property
    def baseline_total_length(self) -> float:
        return float(sum(self.baseline_lengths))

    @property
    def baseline_joint_count(self) -> int:
        return len(self.baseline_lengths)

    def snapshot(self) -> dict:
        data = asdict(self)
        data["baseline_total_length"] = self.baseline_total_length
        data["baseline_joint_count"] = self.baseline_joint_count
        return data


CONFIG = StudyConfig()
