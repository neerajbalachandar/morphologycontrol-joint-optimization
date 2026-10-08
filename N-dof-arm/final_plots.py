"""Publication-style figures for the three requested N-dof optimizations.

Figures saved by this script:
1. Bounded-torque reachable sets for task, dual capability, and dual reachability optima.
2. Reachability-only comparison: fixed-N optimum versus dual-N optimum.
3. Torque-limited LQT tracking from the task optimizer's recorded initial morphology
   and its optimized morphology, with the same N, reference, and actuator limits.
4. Output controllability Gramian eigenvalues, log determinant, and unit-energy
   output-reachable ellipsoids for baseline and the two dual optima.

The ellipsoid is the centered finite-horizon L2-input reachable ellipsoid
    y.T @ inv(W_y) @ y <= 1,
where W_y = integral_0^T C exp(A t) B B.T exp(A.T t) C.T dt and
y = [x_EE(T), xdot_EE(T)]. It describes inputs with integral(u.T u)dt <= 1,
not the componentwise bounded-torque set shown in figures 1 and 2.
"""
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from design import (
    BASELINE_LENGTHS,
    BASELINE_TOTAL_LENGTH,
    LENGTH_MAX,
    LENGTH_MIN,
    REQUIRED_IMPROVEMENT,
    TOTAL_MASS,
    actuator_limits,
)
from model import output_matrix, robot_matrices
from reachability import output_controllability_gramian, reachable_set
from task import desired_state, simulate_task, tracking_metrics

HERE = Path(__file__).resolve().parent
JSON_DIR = HERE / "json"
PLOTS_DIR = HERE / "plots"
REACH_T = 2.0
REACH_NT = 250
REACH_NTHETA = 120
TASK_T = 5.0
TASK_DT = 0.02

# Okabe-Ito color-blind-safe palette, with stable meaning across figures.
COLORS = {
    "baseline": "#222222",
    "task": "#0072B2",
    "capability": "#009E73",
    "reachability": "#D55E00",
    "fixed": "#CC79A7",
}

mpl.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 10,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "axes.linewidth": 0.8,
    "lines.linewidth": 1.8,
    "savefig.dpi": 350,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


def read_json(name):
    """Use the freshest result file in json/ or the script directory."""
    candidates = [p for p in (JSON_DIR / name, HERE / name) if p.exists()]
    if not candidates:
        raise FileNotFoundError(f"Missing {name}; run its optimization script first.")
    path = max(candidates, key=lambda p: p.stat().st_mtime)
    return json.loads(path.read_text())


def fixed_n_capability_design():
    rows = read_json("capability_optimization_results.json")
    feasible = [r for r in rows if r.get("status") == "feasible" and "lengths" in r]
    if not feasible:
        raise RuntimeError("No target-feasible fixed-N capability morphology is saved.")
    return min(feasible, key=lambda row: row["cost"])


def fixed_n_reachability_design():
    rows = read_json("reachability_optimization_results.json")
    if isinstance(rows, dict):
        rows = rows.get("per_N", [])
    rows = [r for r in rows if "lengths" in r and "area" in r]
    if not rows:
        raise RuntimeError("No fixed-N reachability results with lengths and area are saved.")
    return max(rows, key=lambda row: row["area"])


def dual_best(name):
    data = read_json(name)
    if "best" not in data or "lengths" not in data["best"]:
        raise RuntimeError(f"{name} does not contain a valid best design.")
    return data["best"]


def get_task_design():
    rows = read_json("task_optimization_results.json")
    rows = [r for r in rows if "lengths" in r]
    if not rows:
        raise RuntimeError("No task optimization designs are saved.")
    best = min(rows, key=lambda row: row.get("score", row.get("cost", np.inf)))
    initial = best.get("initial_lengths")
    if initial is None:
        # Backward-compatible reconstruction of the first start in the
        # optimizer: equal baseline-total allocation clipped to feasible bounds.
        n = int(best["N"])
        lo = max(LENGTH_MIN, 0.5 * BASELINE_TOTAL_LENGTH / n)
        hi = min(LENGTH_MAX, 1.5 * BASELINE_TOTAL_LENGTH / n)
        initial = np.full(n, np.clip(BASELINE_TOTAL_LENGTH / n, lo, hi)).tolist()
        print("Note: task results lack initial_lengths; reconstructed the original equal-share start.")
    return best, np.asarray(initial, dtype=float)


def design_matrices(lengths):
    lengths = np.asarray(lengths, dtype=float)
    A, B, *_ = robot_matrices(lengths, total_mass=TOTAL_MASS)
    C = output_matrix(lengths)
    return A, B, C


def bounded_reachable(lengths):
    A, B, C = design_matrices(lengths)
    theta, support, boundary, area = reachable_set(
        A, B, C, actuator_limits(len(lengths)),
        T=REACH_T, ntheta=REACH_NTHETA, nt=REACH_NT,
    )
    return theta, support, boundary, area


def style_axis(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(direction="out", length=3.5, width=0.8)
    ax.grid(True, color="#D9D9D9", linewidth=0.6, alpha=0.65)
    ax.set_axisbelow(True)


def save_figure(fig, stem):
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(PLOTS_DIR / f"{stem}.{ext}", bbox_inches="tight")
    print(f"Saved {stem}.png and {stem}.pdf")


def plot_dual_capability_reachability_task(task_best, cap_dual, reach_dual):
    designs = [
        ("Task optimum", np.asarray(task_best["lengths"], dtype=float), COLORS["task"]),
        ("Capability dual optimum", np.asarray(cap_dual["lengths"], dtype=float), COLORS["capability"]),
        ("Reachability dual optimum", np.asarray(reach_dual["lengths"], dtype=float), COLORS["reachability"]),
    ]
    fig, ax = plt.subplots(figsize=(5.5, 4.2), constrained_layout=True)
    for label, lengths, color in designs:
        _, _, boundary, area = bounded_reachable(lengths)
        closed = np.vstack((boundary, boundary[0]))
        ax.plot(closed[:, 0], closed[:, 1], color=color,
                label=f"{label} (N={len(lengths)}, A={area:.2f})")
    ax.set_xlabel(r"End-effector position $x_{EE}(T)$ [m]")
    ax.set_ylabel(r"End-effector velocity $\dot{x}_{EE}(T)$ [m s$^{-1}$]")
    ax.set_title(r"Bounded-torque reachable set at $T=2$ s")
    ax.legend(frameon=False, loc="best")
    style_axis(ax)
    # Position and velocity have different physical units; do not force an
    # equal aspect ratio between these axes.
    save_figure(fig, "final_reachable_sets_task_capability_dual")
    return fig


def plot_fixed_vs_dual_reachability(fixed_reach, reach_dual):
    designs = [
        ("Fixed-N reachability optimum", np.asarray(fixed_reach["lengths"], dtype=float), COLORS["fixed"],
         f"N={fixed_reach['N']}; total length={fixed_reach.get('total_length', np.sum(fixed_reach['lengths'])):.2f} m"),
        ("Dual-N reachability optimum", np.asarray(reach_dual["lengths"], dtype=float), COLORS["reachability"],
         f"N={reach_dual['N']}; total length={np.sum(reach_dual['lengths']):.2f} m"),
    ]
    fig, ax = plt.subplots(figsize=(5.5, 4.2), constrained_layout=True)
    for label, lengths, color, constraint_note in designs:
        _, _, boundary, area = bounded_reachable(lengths)
        closed = np.vstack((boundary, boundary[0]))
        ax.plot(closed[:, 0], closed[:, 1], color=color,
                label=f"{label} ({constraint_note}, A={area:.2f})")
    ax.set_xlabel(r"End-effector position $x_{EE}(T)$ [m]")
    ax.set_ylabel(r"End-effector velocity $\dot{x}_{EE}(T)$ [m s$^{-1}$]")
    ax.set_title(r"Reachability optimization comparison, $T=2$ s")
    ax.legend(frameon=False, loc="best")
    style_axis(ax)
    save_figure(fig, "final_reachability_fixed_vs_dual")
    return fig


def plot_task_tracking(task_best, initial_lengths):
    optimized_lengths = np.asarray(task_best["lengths"], dtype=float)
    if len(initial_lengths) != len(optimized_lengths):
        raise ValueError("Task initial and optimized morphologies must have the same N for a tracking comparison.")
    n = len(optimized_lengths)
    Q = np.diag(np.r_[100.0 * np.ones(n), 10.0 * np.ones(n)])
    R = np.eye(n)
    tau_max = actuator_limits(n)

    series = []
    for label, lengths, color in (
        ("Initial morphology", initial_lengths, COLORS["baseline"]),
        ("Task-optimized morphology", optimized_lengths, COLORS["task"]),
    ):
        A, B, _ = design_matrices(lengths)
        t, X, U = simulate_task(A, B, Q, R, T=TASK_T, dt=TASK_DT, u_max=tau_max)
        desired = np.array([desired_state(ti, n) for ti in t])
        metrics = tracking_metrics(t, X, U, n, Q, R)
        series.append({"label": label, "lengths": lengths, "color": color,
                       "t": t, "X": X, "U": U, "desired": desired,
                       "metrics": metrics})

    fig, axes = plt.subplots(3, 1, figsize=(6.2, 7.2), sharex=True, constrained_layout=True)
    ax_track, ax_error, ax_torque = axes
    for data in series:
        t = data["t"]
        x = data["X"][:, 0]
        qd = data["desired"][:, 0]
        error = x - qd
        utilization = np.max(np.abs(data["U"][:-1]) / tau_max[None, :], axis=1)
        utilization = np.r_[utilization, utilization[-1]]
        morphology = np.array2string(data["lengths"], precision=2, separator=",")
        label = f"{data['label']} {morphology}; RMS={data['metrics']['rms_error']:.4f} rad"
        ax_track.plot(t, x, color=data["color"], label=label)
        ax_error.plot(t, error, color=data["color"], label=data["label"])
        ax_torque.plot(t, utilization, color=data["color"], label=data["label"])
    ax_track.plot(series[-1]["t"], series[-1]["desired"][:, 0], "--", color="#666666",
                  linewidth=1.3, label="Desired joint-1 trajectory")
    ax_track.set_ylabel(r"Joint angle $q_1$ [rad]")
    ax_track.set_title(f"Torque-limited LQT tracking, same N={n} and torque limits")
    ax_error.axhline(0.0, color="#666666", linewidth=0.8)
    ax_error.set_ylabel(r"Tracking error $q_1-q_{1,d}$ [rad]")
    ax_torque.axhline(1.0, color="#666666", linestyle="--", linewidth=1.0,
                      label="Actuator limit")
    ax_torque.set_ylabel(r"Maximum torque use $\max_i |u_i|/u_{i,max}$")
    ax_torque.set_xlabel("Time [s]")
    for ax in axes:
        style_axis(ax)
        ax.legend(frameon=False, loc="best")
    save_figure(fig, "final_task_tracking_initial_vs_optimized")
    return fig


def gramian_and_ellipsoid(lengths):
    A, B, C = design_matrices(lengths)
    W = output_controllability_gramian(A, B, C, T=REACH_T, nt=REACH_NT)
    W = 0.5 * (W + W.T)
    eigenvalues, eigenvectors = np.linalg.eigh(W)
    if np.any(eigenvalues <= 0):
        raise ValueError(f"Output Gramian is not positive definite: eigenvalues={eigenvalues}")
    sign, logdet = np.linalg.slogdet(W)
    if sign <= 0:
        raise ValueError("Output Gramian determinant is not positive.")
    angles = np.linspace(0.0, 2.0 * np.pi, 361)
    unit_circle = np.vstack((np.cos(angles), np.sin(angles)))
    # Boundary of {y: y^T W^{-1} y <= 1}; it is centered at the origin.
    boundary = eigenvectors @ np.diag(np.sqrt(eigenvalues)) @ unit_circle
    return W, eigenvalues, float(logdet), boundary


def plot_gramian_comparison(cap_dual, reach_dual):
    designs = [
        ("Baseline", np.asarray(BASELINE_LENGTHS, dtype=float), COLORS["baseline"]),
        ("Reachability dual", np.asarray(reach_dual["lengths"], dtype=float), COLORS["reachability"]),
        ("Capability dual", np.asarray(cap_dual["lengths"], dtype=float), COLORS["capability"]),
    ]
    results = [(label, lengths, color, gramian_and_ellipsoid(lengths))
               for label, lengths, color in designs]
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.0), constrained_layout=True)
    ax_eig, ax_det, ax_ell = axes
    positions = np.arange(len(results))
    width = 0.23
    for idx, (label, _, color, (_, eigs, logdet, boundary)) in enumerate(results):
        ax_eig.plot([1, 2], eigs, "o-", color=color, label=label)
        ax_det.scatter(idx, logdet, color=color, s=38, zorder=3)
        ax_det.annotate(f"{logdet:.2f}", (idx, logdet), xytext=(0, 6),
                        textcoords="offset points", ha="center", color=color, fontsize=8)
        ax_ell.plot(boundary[0], boundary[1], color=color,
                    label=f"{label}; N={len(results[idx][1])}")
    ax_eig.set_yscale("log")
    ax_eig.set_xticks([1, 2], ["smallest", "largest"])
    ax_eig.set_ylabel("Eigenvalue of $W_y$ (SI units; log scale)")
    ax_eig.set_title(r"Output Gramian eigenvalues, $T=2$ s")
    ax_eig.legend(frameon=False)
    ax_det.axhline(0.0, color="#777777", linewidth=0.8)
    ax_det.set_xticks(positions, [item[0] for item in results], rotation=15, ha="right")
    ax_det.set_ylabel(r"$\log(\det W_y)$ (computed in SI units)")
    ax_det.set_title("Gramian volume measure")
    ax_ell.set_xlabel(r"$x_{EE}(T)$ [m]")
    ax_ell.set_ylabel(r"$\dot{x}_{EE}(T)$ [m s$^{-1}$]")
    ax_ell.set_title("Unit-energy output ellipsoid")
    ax_ell.legend(frameon=False, fontsize=7)
    for ax in axes:
        style_axis(ax)
    save_figure(fig, "final_output_gramian_comparison")
    return fig


def main():
    task_best, task_initial = get_task_design()
    cap_dual = dual_best("capability_optimization_joint_results.json")
    reach_dual = dual_best("reachability_optimization_joint_results.json")
    fixed_reach = fixed_n_reachability_design()

    print("Selected designs used in the final figures:")
    print(f"  Task initial N={len(task_initial)} lengths={task_initial}")
    print(f"  Task optimum N={task_best['N']} lengths={task_best['lengths']}")
    print(f"  Capability dual N={cap_dual['N']} lengths={cap_dual['lengths']}")
    print(f"  Reachability dual N={reach_dual['N']} lengths={reach_dual['lengths']}")
    print(f"  Reachability fixed-N N={fixed_reach['N']} lengths={fixed_reach['lengths']}")
    print("  Gramian ellipsoid: y.T @ inv(W_y) @ y <= 1 for unit L2 input energy, centered at zero.")

    plot_dual_capability_reachability_task(task_best, cap_dual, reach_dual)
    plot_fixed_vs_dual_reachability(fixed_reach, reach_dual)
    plot_task_tracking(task_best, task_initial)
    plot_gramian_comparison(cap_dual, reach_dual)
    plt.show()


if __name__ == "__main__":
    main()
