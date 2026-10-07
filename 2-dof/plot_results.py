"""Compare baseline and saved co-design results with compact plots."""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from design import BASELINE_LENGTHS, TOTAL_MASS, actuator_limits
from model import output_matrix, robot_matrices
from reachability import output_controllability_gramian, reachable_set
from task import desired_state, simulate_task, tracking_metrics

HERE = Path(__file__).resolve().parent
PLOTS_DIR = HERE / "plots"


def best_result(filename, key):
    path = HERE / filename
    if not path.exists():
        return None
    entries = json.loads(path.read_text())
    entries = [
        entry for entry in entries
        if entry.get("status", "feasible") == "feasible" and "lengths" in entry
    ]
    return min(entries, key=key) if entries else None


def load_cases():
    results = [
        ("Baseline", BASELINE_LENGTHS),
        ("Task", (best_result("task_optimization_results.json", lambda x: x.get("score", x.get("cost", np.inf))) or {}).get("lengths")),
        ("Capability", (best_result("capability_optimization_results.json", lambda x: x["cost"]) or {}).get("lengths")),
        ("Reachability", (best_result("reachability_optimization_results.json", lambda x: -x["area"]) or {}).get("lengths")),
    ]
    return [(name, np.asarray(lengths, dtype=float)) for name, lengths in results if lengths is not None]


def morphology_metrics(lengths):
    A, B, *_ = robot_matrices(lengths, total_mass=TOTAL_MASS)
    C = output_matrix(lengths)
    W = output_controllability_gramian(A, B, C, T=2.0, nt=240)
    _, _, boundary, area = reachable_set(
        A, B, C, actuator_limits(len(lengths)), T=2.0, ntheta=180, nt=240
    )
    return A, B, C, W, boundary, area


def plot_summary():
    cases = load_cases()
    if len(cases) == 1:
        print("Only baseline data available. Run optimization scripts to add optimized cases.")

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    ax_reach, ax_gramian, ax_morph, ax_track = axes.ravel()
    tracking_data = []

    for name, lengths in cases:
        A, B, C, W, boundary, area = morphology_metrics(lengths)
        closed = np.vstack((boundary, boundary[0]))
        ax_reach.plot(closed[:, 0], closed[:, 1], lw=2, label=f"{name} (A={area:.2g})")

        eig = np.maximum(np.linalg.eigvalsh(W), np.finfo(float).tiny)
        ax_gramian.plot([1, 2], eig, "o-", label=name)

        ax_morph.plot(np.arange(1, len(lengths) + 1), lengths, "o-", label=name)

        n = len(lengths)
        Q = np.diag(np.r_[100.0 * np.ones(n), 10.0 * np.ones(n)])
        R = np.eye(n)
        t, X, U = simulate_task(
            A, B, Q, R, T=5.0, dt=0.02, u_max=actuator_limits(n)
        )
        XD = np.array([desired_state(ti, n) for ti in t])
        metrics = tracking_metrics(t, X, U, n, Q, R)
        ax_track.plot(t, X[:, 0], label=f"{name} (RMS={metrics['rms_error']:.3g})")
        tracking_data.append((name, t, X[:, 0] - XD[:, 0], U, actuator_limits(n)))

    ax_reach.set(title="Bounded-torque reachable set at T=2 s", xlabel="$x_{EE}(T)$", ylabel="$\\dot{x}_{EE}(T)$")
    ax_reach.set_aspect("equal", adjustable="datalim")
    ax_gramian.set(title="Output Gramian eigenvalues", xlabel="Eigenvalue index", ylabel="Eigenvalue (log scale)")
    ax_gramian.set_yscale("log")
    ax_gramian.set_xticks([1, 2], ["smallest", "largest"])
    ax_morph.set(title="Optimized link lengths", xlabel="Link index", ylabel="Length")
    ax_track.set(title="Joint 1 tracking", xlabel="Time [s]", ylabel="$q_1$ [rad]")

    for ax in axes.ravel():
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    # The reference is shared across morphologies (all use the same q_d).
    ref_t = np.linspace(0.0, 5.0, 300)
    ax_track.plot(ref_t, [desired_state(ti, 1)[0] for ti in ref_t], "k--", label="Reference")
    ax_track.legend(fontsize=8)
    fig.tight_layout()

    # A single focused plot shows actuator effort over the task.
    fig_u, (ax_error, ax_u) = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    for name, t, error, U, limit in tracking_data:
        ax_error.plot(t, error, label=name)
        utilization = np.max(np.abs(U) / limit[None, :], axis=1)
        ax_u.plot(t, utilization, label=name)
    ax_error.axhline(0.0, color="k", linewidth=0.8)
    ax_error.set(ylabel="$q_1-q_{1,d}$ [rad]", title="Tracking error")
    ax_u.axhline(1.0, color="k", linestyle="--", linewidth=1, label="Torque limit")
    ax_u.set(title="Actuator limit usage", xlabel="Time [s]", ylabel="Maximum $|u_i|/u_{i,max}$")
    ax_error.grid(True, alpha=0.3)
    ax_u.grid(True, alpha=0.3)
    ax_error.legend(fontsize=8)
    ax_u.legend()
    fig_u.tight_layout()

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    summary_path = PLOTS_DIR / "design_comparison.png"
    tracking_path = PLOTS_DIR / "tracking_and_control.png"
    fig.savefig(summary_path, dpi=220, bbox_inches="tight")
    fig_u.savefig(tracking_path, dpi=220, bbox_inches="tight")
    print(f"Saved plots to {summary_path} and {tracking_path}")
    plt.show()


if __name__ == "__main__":
    plot_summary()
