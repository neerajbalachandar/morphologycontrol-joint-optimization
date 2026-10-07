"""Compact, defined plots for all five N-dof co-design optimizations."""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from design import (
    BASELINE_LENGTHS, REQUIRED_IMPROVEMENT, TOTAL_MASS, actuator_limits,
)
from model import output_matrix, robot_matrices
from reachability import output_controllability_gramian, reachable_set
from task import desired_state, simulate_task, tracking_metrics

HERE = Path(__file__).resolve().parent
JSON_DIR = HERE / "json"
PLOTS_DIR = HERE / "plots"
TASK_T = 5.0
TASK_DT = 0.04
REACH_T = 2.0


def read_results(name):
    """Read the current json/ location, falling back to legacy script output."""
    candidates = [path for path in (JSON_DIR / name, HERE / name) if path.exists()]
    if not candidates:
        return None
    # Prefer the most recently generated file if both locations contain a copy.
    path = max(candidates, key=lambda candidate: candidate.stat().st_mtime)
    return json.loads(path.read_text())


def selected_cases():
    cases = [("Baseline", np.asarray(BASELINE_LENGTHS, dtype=float))]

    task_rows = read_results("task_optimization_results.json") or []
    task_rows = [r for r in task_rows if "lengths" in r]
    if task_rows:
        row = min(task_rows, key=lambda r: r.get("score", r.get("cost", np.inf)))
        cases.append((f"Task optimum (N={row['N']})", np.asarray(row["lengths"], dtype=float)))

    capability_rows = read_results("capability_optimization_results.json") or []
    capability_rows = [
        r for r in capability_rows
        if r.get("status", "feasible") == "feasible" and "lengths" in r
    ]
    if capability_rows:
        row = min(capability_rows, key=lambda r: r["cost"])
        cases.append((f"Capability fixed N (N={row['N']})", np.asarray(row["lengths"], dtype=float)))

    reachability_rows = read_results("reachability_optimization_results.json") or []
    reachability_rows = [r for r in reachability_rows if "lengths" in r and "area" in r]
    if reachability_rows:
        row = max(reachability_rows, key=lambda r: r["area"])
        cases.append((f"Reachability fixed N (N={row['N']})", np.asarray(row["lengths"], dtype=float)))

    cap_dual = read_results("capability_optimization_joint_results.json")
    if cap_dual and cap_dual.get("best", {}).get("lengths"):
        row = cap_dual["best"]
        cases.append((f"Capability dual (N={row['N']})", np.asarray(row["lengths"], dtype=float)))

    reach_dual = read_results("reachability_optimization_joint_results.json")
    if reach_dual and reach_dual.get("best", {}).get("lengths"):
        row = reach_dual["best"]
        cases.append((f"Reachability dual (N={row['N']})", np.asarray(row["lengths"], dtype=float)))

    if len(cases) == 1:
        raise FileNotFoundError("No optimization result files found. Run the optimization scripts first.")
    return cases


def metrics_for_design(lengths):
    n = len(lengths)
    A, B, *_ = robot_matrices(lengths, total_mass=TOTAL_MASS)
    C = output_matrix(lengths)
    theta, h, boundary, area = reachable_set(
        A, B, C, actuator_limits(n), T=REACH_T, ntheta=120, nt=250
    )
    gramian = output_controllability_gramian(A, B, C, T=REACH_T, nt=250)
    gramian_eigs = np.maximum(np.linalg.eigvalsh(gramian), np.finfo(float).tiny)
    q_weight = np.diag(np.r_[100.0 * np.ones(n), 10.0 * np.ones(n)])
    t, X, U = simulate_task(
        A, B, q_weight, np.eye(n), T=TASK_T, dt=TASK_DT,
        u_max=actuator_limits(n),
    )
    desired = np.array([desired_state(ti, n) for ti in t])
    metrics = tracking_metrics(t, X, U, n, q_weight, np.eye(n))
    return {
        "A": A, "B": B, "C": C, "theta": theta, "support": h,
        "boundary": boundary, "area": area, "gramian_eigs": gramian_eigs,
        "t": t, "X": X, "U": U, "desired": desired, "metrics": metrics,
    }


def per_n_rows():
    return (
        read_results("task_optimization_results.json") or [],
        read_results("capability_optimization_results.json") or [],
        read_results("reachability_optimization_results.json") or [],
    )


def plot_all():
    cases = selected_cases()
    computed = [(name, lengths, metrics_for_design(lengths)) for name, lengths in cases]
    colors = plt.get_cmap("tab10")

    # Figure 1: show the morphology and the three modeled outcomes for the
    # selected solution from each optimization family.
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    ax_reach, ax_morph, ax_track, ax_gramian = axes.ravel()
    for i, (name, lengths, data) in enumerate(computed):
        color = colors(i)
        boundary = np.vstack((data["boundary"], data["boundary"][0]))
        ax_reach.plot(boundary[:, 0], boundary[:, 1], color=color, lw=2,
                      label=f"{name}; area={data['area']:.2f}")
        ax_morph.plot(np.arange(1, len(lengths) + 1), lengths, "o-", color=color, label=name)
        n = len(lengths)
        rms = data["metrics"]["rms_error"]
        ax_track.plot(data["t"], data["X"][:, 0], color=color,
                      label=f"{name}; joint-1 RMS error={rms:.3f} rad")
        ax_gramian.plot([1, 2], data["gramian_eigs"], "o-", color=color, label=name)

    ref_t = np.linspace(0.0, TASK_T, 300)
    ax_track.plot(ref_t, [desired_state(ti, 1)[0] for ti in ref_t], "k--", lw=1.5, label="Joint-1 reference")
    ax_reach.set(title="Bounded-torque reachable set at T=2 s",
                 xlabel="$x_{EE}(T)$ [m]", ylabel="$\\dot{x}_{EE}(T)$ [m/s]")
    ax_reach.set_aspect("equal", adjustable="datalim")
    ax_morph.set(title="Selected link lengths", xlabel="Link index", ylabel="Length [m]")
    ax_track.set(title="Torque-limited LQT tracking over the same reference",
                 xlabel="Time [s]", ylabel="$q_1$ [rad]")
    ax_gramian.set(title="Output controllability Gramian eigenvalues at T=2 s",
                   xlabel="Sorted eigenvalue", ylabel="Eigenvalue [log scale]")
    ax_gramian.set_yscale("log")
    ax_gramian.set_xticks([1, 2], ["smallest", "largest"])
    for ax in axes.ravel():
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=7)
    fig.tight_layout()

    # Figure 2: control and tracking error are shown separately to keep units
    # and interpretations explicit.
    fig_control, (ax_error, ax_util) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    for i, (name, _, data) in enumerate(computed):
        color = colors(i)
        error = data["X"][:, 0] - data["desired"][:, 0]
        utilization = np.max(np.abs(data["U"]) / actuator_limits(data["U"].shape[1])[None, :], axis=1)
        ax_error.plot(data["t"], error, color=color, label=name)
        ax_util.plot(data["t"], utilization, color=color, label=name)
    ax_error.axhline(0.0, color="black", lw=0.8)
    ax_error.set(title="Joint-1 tracking error", ylabel="$q_1-q_{1,d}$ [rad]")
    ax_util.axhline(1.0, color="black", ls="--", lw=1, label="Actuator torque limit")
    ax_util.set(title="Maximum actuator torque utilization",
                xlabel="Time [s]", ylabel="$\\max_i |u_i|/u_{i,max}$")
    for ax in (ax_error, ax_util):
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8)
    fig_control.tight_layout()

    # Figure 3: every fixed-N result row, including infeasible capability
    # cases' best-found areas, is plotted. Dual-N solutions are separate in
    # figures 1 and 2 because they use fixed total length L0.
    task_rows, cap_rows, reach_rows = per_n_rows()
    fig_n, (ax_task, ax_cap, ax_reach_n) = plt.subplots(1, 3, figsize=(15, 4.8))
    if task_rows:
        task_rows = sorted(task_rows, key=lambda r: r["N"])
        ax_task.plot([r["N"] for r in task_rows], [r.get("score", r.get("cost")) for r in task_rows], "o-")
    ax_task.set(title="Task co-design: normalized objective", xlabel="Joint count N", ylabel="Task score (lower is better)")
    if cap_rows:
        cap_rows = sorted(cap_rows, key=lambda r: r["N"])
        ns, areas, feasible = [], [], []
        for row in cap_rows:
            ns.append(row["N"])
            if row.get("status") == "feasible":
                areas.append(row["area"])
                feasible.append(True)
            else:
                areas.append(row.get("max_area_found", np.nan))
                feasible.append(False)
        ax_cap.plot(ns, areas, "o", label="Optimized or best-found area")
        target = (1.0 + REQUIRED_IMPROVEMENT) * metrics_for_design(BASELINE_LENGTHS)["area"]
        ax_cap.axhline(target, color="red", ls="--", label=f"Required area (+{100*REQUIRED_IMPROVEMENT:g}%)")
        for n, area, ok in zip(ns, areas, feasible):
            if not ok and np.isfinite(area):
                ax_cap.annotate("target not met", (n, area), xytext=(0, -16), textcoords="offset points", ha="center", fontsize=7)
        ax_cap.legend(fontsize=7)
    ax_cap.set(title="Capability co-design: reachability target check",
               xlabel="Joint count N", ylabel="Reachable-set area")
    if reach_rows:
        reach_rows = sorted(reach_rows, key=lambda r: r["N"])
        ax_reach_n.plot([r["N"] for r in reach_rows], [r["area"] for r in reach_rows], "o-")
    ax_reach_n.set(title="Reachability co-design: optimized area", xlabel="Joint count N", ylabel="Reachable-set area")
    for ax in (ax_task, ax_cap, ax_reach_n):
        ax.grid(True, alpha=0.25)
        ax.set_xticks(range(1, max([len(r.get("lengths", [])) for r in task_rows + cap_rows + reach_rows] + [5]) + 1))
    fig_n.suptitle("Fixed-N optimization results (these use their scripts' length constraints; dual-N designs are shown separately)", fontsize=10)
    fig_n.tight_layout()

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    paths = [
        PLOTS_DIR / "optimization_designs_and_reachability.png",
        PLOTS_DIR / "optimization_tracking_and_control.png",
        PLOTS_DIR / "optimization_metrics_by_N.png",
    ]
    for figure, path in zip((fig, fig_control, fig_n), paths):
        figure.savefig(path, dpi=220, bbox_inches="tight")
    print("Saved plots:")
    for path in paths:
        print(f"  {path}")
    plt.show()


if __name__ == "__main__":
    plot_all()
