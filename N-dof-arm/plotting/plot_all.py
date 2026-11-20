"""Generate the focused figure set for one explicitly named experiment run."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import cdist
from scipy.spatial import ConvexHull

from config import CONFIG
from design import BASELINE_LENGTHS, TOTAL_MASS, actuator_limits
from model import kinematic_points, output_matrix, robot_matrices
from optimization_variable_n import RESULTS_ROOT, morphology_cost
from reachability import (
    lp_reachable_set,
    output_scaling,
    reachability_kernel,
)
from reachability_costate import sample_costate_reachable_points
from reachability_zonotope import propagate_outer_zonotope, projected_zonotope_boundary
from .style import COLORS, apply_style

FINITE_P = tuple(float(p) for p in CONFIG.p_values if p != "infinity")
NORM_SPECS = [
    (p, key, f"$p={int(p)}$")
    for p, key in zip(FINITE_P, ("l2", "l4", "l6", "l10"))
] + [(np.inf, "linf", "$p=\\infty$")]


def _geometry(lengths, p=np.inf, convention="peak_equivalent", ntheta=180, nt=300):
    lengths = np.asarray(lengths, dtype=float)
    A, B, *_ = robot_matrices(lengths, total_mass=TOTAL_MASS)
    C = output_matrix(lengths)
    return lp_reachable_set(
        A, B, C, actuator_limits(len(lengths), "fixed_total"),
        CONFIG.horizon, p, convention, sum(BASELINE_LENGTHS),
        ntheta, nt, nondimensional=True,
    )


def _row_by_n(metadata, criterion):
    return metadata["capability_optimizations"][criterion]["results_by_N"]


def _save(fig, output_dir, stem):
    fig.savefig(output_dir / f"{stem}.png", bbox_inches="tight")
    fig.savefig(output_dir / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def figure_1_analytical(output_dir):
    lengths = np.linspace(CONFIG.analytical_length_min, CONFIG.analytical_length_max,
                          CONFIG.analytical_sweep_points)
    areas, traces, ellipses, l6, linf, wn, zeta = [], [], [], [], [], [], []
    for length in lengths:
        link = np.array([length])
        A, B, M, K, D, _, _ = robot_matrices(link, total_mass=TOTAL_MASS)
        C = output_matrix(link)
        g2 = lp_reachable_set(A, B, C, actuator_limits(1, "fixed_total"), CONFIG.horizon,
                              2.0, "peak_equivalent", sum(BASELINE_LENGTHS), CONFIG.theta_samples, CONFIG.time_samples)
        g6 = lp_reachable_set(A, B, C, actuator_limits(1, "fixed_total"), CONFIG.horizon,
                              6.0, "peak_equivalent", sum(BASELINE_LENGTHS), CONFIG.theta_samples, CONFIG.time_samples)
        gi = lp_reachable_set(A, B, C, actuator_limits(1, "fixed_total"), CONFIG.horizon,
                              np.inf, "peak_equivalent", sum(BASELINE_LENGTHS), CONFIG.theta_samples, CONFIG.time_samples)
        traces.append(np.trace(g2["gramian"]))
        ellipses.append(g2["ellipsoid_area"])
        l6.append(g6["area"]); linf.append(gi["area"])
        areas.append(gi["area"])
        ev = np.linalg.eigvals(np.linalg.solve(M, K))
        natural = float(np.sqrt(np.max(np.real(ev))))
        wn.append(natural)
        zeta.append(float(D[0, 0] / M[0, 0] / (2 * natural)))

    fig, axes = plt.subplots(2, 3, figsize=(12.2, 7.3))
    ax = axes.ravel()
    selected_lengths = np.asarray(CONFIG.baseline_lengths).sum() * np.array([0.1875, 0.375, 0.625, 0.9375])
    for length, color in zip(selected_lengths,
                             (COLORS["l2"], COLORS["l4"], COLORS["l6"], COLORS["linf"])):
        selected = _geometry([length], np.inf, "peak_equivalent", CONFIG.theta_samples, CONFIG.time_samples)
        b = np.vstack((selected["boundary"], selected["boundary"][0]))
        ax[0].plot(b[:, 0], b[:, 1], color=color, label=f"l={length:g} m")
    ax[0].set(title="N=1, L∞ boundaries at selected lengths", xlabel="$x_{EE}/L_0$",
              ylabel="$T\\dot{x}_{EE}/L_0$")
    ax[0].set_aspect("equal", adjustable="box")
    ax[0].legend(frameon=False)
    ax[1].plot(lengths, areas, color=COLORS["linf"])
    ax[1].set(title="$L_\\infty$ reachable area", xlabel="Link length [m]", ylabel="Area [dimensionless]")
    ax[2].plot(lengths, traces, label="Trace", color=COLORS["l2"])
    ax[2].plot(lengths, ellipses, label="L2 ellipse area", color=COLORS["l4"])
    ax[2].set(title="L2 output Gramian", xlabel="Link length [m]", ylabel="Metric [dimensionless]")
    ax[2].legend(frameon=False)
    ax[3].plot(lengths, l6, color=COLORS["l6"], label="$L_6$ area")
    ax[3].plot(lengths, linf, color=COLORS["linf"], label="$L_\\infty$ area")
    ax[3].set(title="Input-norm capability", xlabel="Link length [m]", ylabel="Area [dimensionless]")
    ax[3].legend(frameon=False)
    ax[4].plot(lengths, wn, color=COLORS["l2"], label="$\\omega_n$")
    ax[4].set(title="Linearized natural frequency", xlabel="Link length [m]", ylabel="Frequency [rad s$^{-1}$]")
    axr = ax[4].twinx(); axr.plot(lengths, zeta, color=COLORS["task"], label="$\\zeta$")
    axr.set_ylabel("Damping ratio")
    ax[5].axis("off")
    ax[5].text(0.02, 0.94,
               "Analytical single-link benchmark\n\n"
               "$M_q=ml^2/3$, $K_q=mgl/2$\n"
               "$\\omega_n=\\sqrt{3g/(2l)}$\n"
               "Total mass held fixed; linear density varies as $M/l$.\n"
               "Actuator budget: fixed total 2 Nm.", va="top", fontsize=10)
    for a in ax[:5]: a.grid(True)
    fig.tight_layout()
    _save(fig, output_dir, "01_one_link_analytical")


def figure_2_norm_geometry(output_dir, morphology):
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.6))
    for ax, convention, title in zip(axes, ("peak_equivalent", "raw_paper"),
                                     ("Peak-equivalent budget (primary)", "Raw paper unit norm (sensitivity)")):
        for p, key, label in NORM_SPECS:
            result = _geometry(morphology, p, convention, CONFIG.plot_theta_samples, CONFIG.plot_time_samples)
            boundary = np.vstack((result["boundary"], result["boundary"][0]))
            area_label = f"{label}; A={result['area']:.3g}"
            if p == 2.0:
                area_label += f", ellipse={result['ellipsoid_area']:.3g}"
            ax.plot(boundary[:, 0], boundary[:, 1], color=COLORS[key], label=area_label)
        ax.set(title=title, xlabel="$x_{EE}/L_0$", ylabel="$T\\dot{x}_{EE}/L_0$")
        ax.set_aspect("equal", adjustable="box"); ax.grid(True); ax.legend(frameon=False)
    fig.suptitle(f"Input norm geometry, lengths={np.round(morphology, 3)} m")
    fig.tight_layout()
    _save(fig, output_dir, "02_lp_vs_linf_geometry")


def figure_3_support(output_dir, morphology):
    fig, ax = plt.subplots(figsize=(7.3, 4.4))
    for p, key, label in NORM_SPECS:
        data = _geometry(morphology, p, "peak_equivalent", CONFIG.plot_theta_samples, CONFIG.plot_time_samples)
        ax.plot(data["theta"], data["support"], color=COLORS[key], label=label)
    ax.set(xlabel="Output direction $\\theta$ [rad]", ylabel="Support $h_p(\\theta)$ [dimensionless]",
           title="Directional end-effector capability")
    ax.grid(True); ax.legend(ncol=3, frameon=False)
    fig.tight_layout(); _save(fig, output_dir, "03_support_functions")


def figure_4_convergence(output_dir, morphology):
    resolutions = CONFIG.convergence_resolutions
    pvals = [p for p, _, _ in NORM_SPECS]
    areas, boundaries = {p: [] for p in pvals}, {p: [] for p in pvals}
    for ntheta, nt in resolutions:
        for p in pvals:
            result = _geometry(morphology, p, "peak_equivalent", ntheta, nt)
            areas[p].append(result["area"]); boundaries[p].append(result["boundary"])
    inf_ref = boundaries[np.inf][-1]
    relative, hausdorff = {}, {}
    for p in pvals[:-1]:
        relative[p] = [abs(a - areas[np.inf][i]) / max(areas[np.inf][i], 1e-15)
                       for i, a in enumerate(areas[p])]
        hausdorff[p] = []
        for pts, infpts in zip(boundaries[p], boundaries[np.inf]):
            distances = cdist(pts, infpts)
            hausdorff[p].append(float(max(distances.min(axis=0).max(), distances.min(axis=1).max())))
    fig, axes = plt.subplots(1, 3, figsize=(12.2, 3.9))
    x = np.arange(len(resolutions))
    for p, key, label in NORM_SPECS:
        axes[0].plot(x, areas[p], "o-", color=COLORS[key], label=label)
    for p, key, _ in NORM_SPECS[:-1]:
        axes[1].plot(x, relative[p], "o-", color=COLORS[key], label=f"p={p:g}")
        axes[2].plot(x, hausdorff[p], "o-", color=COLORS[key], label=f"p={p:g}")
    labels = [f"{a}×{b}" for a, b in resolutions]
    axes[0].set(title="Area versus p", ylabel="Area [dimensionless]")
    axes[1].set(title="Relative area difference", ylabel="$|A_p-A_∞|/A_∞$")
    axes[2].set(title="Boundary Hausdorff estimate", ylabel="Distance [dimensionless]")
    for a in axes:
        a.set_xlabel("Angular × time samples"); a.set_xticks(x, labels, rotation=25)
        a.grid(True); a.legend(frameon=False)
    fig.tight_layout(); _save(fig, output_dir, "04_norm_convergence")


def _extremal_control(kernels, times, p, angle, radius):
    s = np.array([np.cos(angle), np.sin(angle)])
    a = np.einsum("tki,k->ti", kernels, s)
    if np.isinf(p): return np.sign(a)
    q = p / (p - 1)
    H = np.trapezoid(np.sum(np.abs(a)**q, axis=1), times)**(1/q)
    return radius * max(H, 1e-15)**(1-q) * np.sign(a) * np.abs(a)**(q-1)


def figure_5_controls(output_dir, morphology):
    lengths = np.asarray(morphology)
    A, B, *_ = robot_matrices(lengths, total_mass=TOTAL_MASS)
    C = output_matrix(lengths)
    tau = actuator_limits(len(lengths), "fixed_total")
    times, kernels = reachability_kernel(A, B, C, tau, CONFIG.horizon, CONFIG.plot_time_samples,
                                         output_scaling(sum(BASELINE_LENGTHS), CONFIG.horizon))
    fig, axes = plt.subplots(2, 2, figsize=(12, 6.5), sharex="col")
    for col, angle in enumerate((0.35, 1.2)):
        for p, key, label in ((np.inf, "linf", "L∞ saturated"), (6., "l6", "L6"), (2., "l2", "L2 energy")):
            radius = 1.0 if np.isinf(p) else (len(lengths)*CONFIG.horizon)**(1/p)
            normalized_u = _extremal_control(kernels, times, p, angle, radius)
            physical = normalized_u * tau[None, :]
            for j in range(len(lengths)):
                axes[0, col].plot(times, normalized_u[:, j], color=COLORS[key],
                                  linestyle="-" if j == 0 else "--",
                                  label=f"{label}, motor {j+1}")
                axes[1, col].plot(times, physical[:, j], color=COLORS[key],
                                  linestyle="-" if j == 0 else "--",
                                  label=f"{label}, motor {j+1}")
        axes[0, col].set(title=f"Normalized controls, $\\theta={angle:g}$ rad",
                         ylabel="$v_i=u_i/\\tau_{max,i}$")
        axes[1, col].set(title=f"Physical torques, $\\theta={angle:g}$ rad",
                         xlabel="Time [s]", ylabel="Torque $u_i$ [Nm]")
        for limit in tau:
            axes[1, col].axhline(limit, color="black", lw=.65, alpha=.25)
            axes[1, col].axhline(-limit, color="black", lw=.65, alpha=.25)
        for row in range(2):
            axes[row, col].grid(True); axes[row, col].legend(frameon=False, fontsize=7)
    fig.tight_layout(); _save(fig, output_dir, "05_extremal_control_histories")


def figure_6_costate(output_dir):
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    for col, n in enumerate((1, 2)):
        lengths = np.full(n, sum(BASELINE_LENGTHS)/n)
        A, B, *_ = robot_matrices(lengths, total_mass=TOTAL_MASS); C = output_matrix(lengths)
        tau = actuator_limits(n, "fixed_total")
        sample = sample_costate_reachable_points(
            A, B, C, tau, CONFIG.horizon, 6., "peak_equivalent",
            CONFIG.costate_samples, CONFIG.random_seed+n, CONFIG.costate_time_samples,
        )
        S = output_scaling(sum(BASELINE_LENGTHS), CONFIG.horizon)
        points = sample["terminal_outputs"] @ S.T
        direct = _geometry(lengths, 6., "peak_equivalent", CONFIG.plot_theta_samples, CONFIG.plot_time_samples)["boundary"]
        mask = sample["accepted"]
        ax_costate = axes[0, col]
        ax_costate.scatter(sample["costates"][~mask, 0], sample["costates"][~mask, 1],
                           s=8, color=COLORS["gray"], alpha=.35, label="outside L6 bound")
        ax_costate.scatter(sample["costates"][mask, 0], sample["costates"][mask, 1],
                           s=8, color=COLORS["l6"], alpha=.5, label="inside L6 bound")
        ax_costate.set(title=f"N={n}: initial costates, dimension {2*n}",
                       xlabel="$\\lambda_{0,1}$", ylabel="$\\lambda_{0,2}$")
        ax_costate.grid(True); ax_costate.legend(frameon=False)
        if n > 1:
            ax_costate.text(.03, .97, "first two components shown", transform=ax_costate.transAxes,
                            va="top", fontsize=7)
        ax = axes[1, col]
        ax.scatter(points[~mask, 0], points[~mask, 1], s=8, color=COLORS["gray"], alpha=.35, label="outside L6 bound")
        ax.scatter(points[mask, 0], points[mask, 1], s=8, color=COLORS["l6"], alpha=.5, label="accepted costates")
        if np.count_nonzero(mask) >= 3:
            try:
                accepted_hull = points[mask][ConvexHull(points[mask]).vertices]
                accepted_hull = np.vstack((accepted_hull, accepted_hull[0]))
                ax.plot(accepted_hull[:, 0], accepted_hull[:, 1], color=COLORS["l6"],
                        ls=":", lw=1.4, label="accepted-point convex hull")
            except Exception:
                pass
        hull = np.vstack((direct, direct[0]))
        ax.plot(hull[:, 0], hull[:, 1], color=COLORS["baseline"], ls="--",
                label="direct projected L6 boundary")
        ax.set(title=f"N={n}: accepted terminal outputs", xlabel="$x_{EE}/L_0$", ylabel="$T\\dot{x}_{EE}/L_0$")
        ax.set_aspect("equal", adjustable="box"); ax.grid(True); ax.legend(frameon=False)
    fig.tight_layout(); _save(fig, output_dir, "06_costate_sampling_validation")


def figure_7_morphologies(output_dir, baseline, asymmetric):
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    cases = (("Symmetric baseline", np.asarray(baseline), COLORS["baseline"]),
             ("Asymmetric reachability optimum", np.asarray(asymmetric), COLORS["linf"]))
    for col, (title, lengths, color) in enumerate(cases):
        q = np.linspace(0.12, 0.28, len(lengths)) * np.where(np.arange(len(lengths))%2, -1, 1)
        points = kinematic_points(q, lengths)
        ax = axes[0, col]
        ax.plot(points[:, 0], points[:, 1], "o-", color=color, lw=3, ms=5)
        ax.scatter(points[0, 0], points[0, 1], color="black", s=25, zorder=3)
        ax.set(title=f"{title}\nN={len(lengths)}, l={np.round(lengths, 2)} m",
               xlabel="Horizontal position [m]", ylabel="Vertical position [m]")
        ax.set_aspect("equal", adjustable="box"); ax.grid(True)
        reach = _geometry(lengths, np.inf, "peak_equivalent", CONFIG.plot_theta_samples, CONFIG.plot_time_samples)
        boundary = np.vstack((reach["boundary"], reach["boundary"][0]))
        axes[1, col].plot(boundary[:, 0], boundary[:, 1], color=color,
                          label=f"Area={reach['area']:.3g}")
        axes[1, col].set(title=f"{title}: projected reachable boundary",
                          xlabel="$x_{EE}/L_0$", ylabel="$T\\dot{x}_{EE}/L_0$")
        axes[1, col].set_aspect("equal", adjustable="box"); axes[1, col].grid(True)
        axes[1, col].legend(frameon=False)
    fig.tight_layout(); _save(fig, output_dir, "07_morphology_configurations")


def figure_8_capability_by_n(output_dir, metadata):
    criteria = (("l2_trace", COLORS["l2"], "L2 Gramian trace"),
                ("l6_area", COLORS["l6"], "L6 area"),
                ("linf_area", COLORS["linf"], "L∞ area"))
    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    for key, color, label in criteria:
        opt = metadata["capability_optimizations"][key]
        baseline = opt["baseline_objective"]
        rows = [r for r in opt["results_by_N"] if r.get("objective") is not None]
        ax.plot([r["N"] for r in rows], [r["objective"]/baseline for r in rows],
                "o-", color=color, label=label)
    ax.set(xlabel="Joint count N", ylabel="Optimized metric / baseline metric",
           title="Capability optimization under common resource constraints")
    ax.set_xticks(range(CONFIG.n_min, CONFIG.n_max+1)); ax.axhline(1, color="black", lw=1, ls=":")
    ax.grid(True); ax.legend(frameon=False)
    fig.tight_layout(); _save(fig, output_dir, "08_reachability_optimization_by_N")


def figure_9_lengths(output_dir, metadata):
    rows = _row_by_n(metadata, "linf_area")
    valid = [r for r in rows if r.get("lengths")]
    matrix = np.full((len(valid), CONFIG.n_max), np.nan)
    for i, row in enumerate(valid): matrix[i, :len(row["lengths"])] = row["lengths"]
    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    image = ax.imshow(np.ma.masked_invalid(matrix), aspect="auto", cmap="viridis",
                      vmin=CONFIG.length_min, vmax=CONFIG.length_max)
    ax.set(yticks=np.arange(len(valid)), yticklabels=[f"N={r['N']}" for r in valid],
           xticks=np.arange(CONFIG.n_max), xticklabels=np.arange(1, CONFIG.n_max+1),
           xlabel="Link index", title="L∞-optimal link lengths [m]")
    fig.colorbar(image, ax=ax, label="Length [m]")
    fig.tight_layout(); _save(fig, output_dir, "09_optimal_link_distributions")


def figure_10_cost_tradeoff(output_dir, metadata):
    rows = [r for r in _row_by_n(metadata, "linf_area") if r.get("objective") is not None]
    baseline_area = metadata["baseline"]["linf_area"]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.3))
    areas = np.asarray([r["objective"] for r in rows]); costs = np.asarray([morphology_cost(r["N"]) for r in rows])
    gains = 100 * (areas / baseline_area - 1)
    nondominated = np.asarray([
        not any((costs[j] <= costs[i] and gains[j] >= gains[i]
                 and (costs[j] < costs[i] or gains[j] > gains[i]))
                for j in range(len(rows)) if j != i)
        for i in range(len(rows))
    ])
    axes[0].scatter(gains, costs, color=COLORS["gray"], label="Best fixed-N candidates")
    frontier = np.argsort(gains[nondominated])
    axes[0].plot(gains[nondominated][frontier], costs[nondominated][frontier],
                 "o-", color=COLORS["linf"], label="Non-dominated frontier")
    axes[0].set(xlabel="Capability improvement over baseline [%]", ylabel="Design proxy $J_m$",
                title="Cost-capability Pareto candidates")
    targets = metadata["minimum_cost_targets"]
    gains = [100*t["gain_target"] for t in targets]
    for t in targets:
        if t["status"] == "feasible":
            axes[1].scatter(100*t["gain_target"], t["design_cost"], color=COLORS["l6"], s=45)
        else:
            axes[1].scatter(100*t["gain_target"], np.nanmin(costs), marker="x", color=COLORS["linf"])
    axes[1].set(xticks=gains, xlabel="Required reachable-area improvement [%]",
                ylabel="Minimum feasible $J_m$", title="Minimum-cost capability targets")
    axes[0].legend(frameon=False)
    for ax in axes: ax.grid(True)
    fig.tight_layout(); _save(fig, output_dir, "10_design_capability_tradeoff")


def figure_11_geometry_comparison(output_dir, metadata):
    baseline = np.asarray(metadata["baseline"]["lengths"])
    l6best = metadata["capability_optimizations"]["l6_area"]["best_overall"]
    ibest = metadata["capability_optimizations"]["linf_area"]["best_overall"]
    target = next((x for x in metadata["minimum_cost_targets"] if x["status"] == "feasible"), None)
    cases = [("Baseline", baseline, COLORS["baseline"]),
             ("L6-optimal", np.asarray(l6best["lengths"]), COLORS["l6"]),
             ("L∞-optimal", np.asarray(ibest["lengths"]), COLORS["linf"])]
    if target:
        cases.append(("Minimum-cost target", np.asarray(target["lengths"]), COLORS["task"]))
    fig, ax = plt.subplots(figsize=(7, 5.2))
    for name, lengths, color in cases:
        result = _geometry(lengths, np.inf, "peak_equivalent", CONFIG.plot_theta_samples, CONFIG.plot_time_samples)
        boundary = np.vstack((result["boundary"], result["boundary"][0]))
        ax.plot(boundary[:, 0], boundary[:, 1], color=color,
                label=f"{name}, N={len(lengths)}, A={result['area']:.3g}")
    ax.set(xlabel="$x_{EE}/L_0$", ylabel="$T\\dot{x}_{EE}/L_0$",
           title="Reachability before and after morphology optimization")
    ax.set_aspect("equal", adjustable="box"); ax.grid(True); ax.legend(frameon=False)
    fig.tight_layout(); _save(fig, output_dir, "11_geometry_before_after")


def figure_12_task(output_dir, run_dir, metadata):
    task_path = run_dir / "task_tracking.json"
    arrays_path = run_dir / "task_tracking_arrays.npz"
    if not task_path.exists() or not arrays_path.exists():
        return False
    task = json.loads(task_path.read_text()); data = np.load(arrays_path)
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.4))
    colors = {"baseline": COLORS["baseline"], "reachability_optimum": COLORS["linf"], "task_optimum": COLORS["task"]}
    for name, rec in task["designs"].items():
        color = colors[name]
        t = data[f"{name}_times"]
        y, ref, err = data[f"{name}_output"], data[f"{name}_reference"], data[f"{name}_tracking_error"]
        axes[0, 0].plot(t, y[:, 0], color=color,
                        label=f"{name.replace('_', ' ')} (N={rec['N']})")
        axes[0, 0].plot(t, ref[:, 0], color="black", ls="--", lw=.8,
                        label="common desired output" if name == "baseline" else "_nolegend_")
        axes[0, 1].plot(t, err[:, 0], color=color, label=name)
        axes[1, 0].plot(t, err[:, 1], color=color, label=name)
        U = data[f"{name}_torques"]
        limits = rec["torque_limits"]
        for j in range(U.shape[1]):
            axes[1, 1].plot(t[:-1], U[:, j], color=color, ls="-" if j == 0 else "--",
                            label=f"{name}, motor {j+1}")
            axes[1, 1].axhline(limits[j], color=color, lw=.5, alpha=.22)
            axes[1, 1].axhline(-limits[j], color=color, lw=.5, alpha=.22)
    axes[0, 0].set(title="Horizontal output tracking", ylabel="$x_{EE}$ [m]")
    axes[0, 1].set(title="Position tracking error", ylabel="$x-x_d$ [m]")
    axes[1, 0].set(title="Velocity tracking error", ylabel="$\\dot{x}-\\dot{x}_d$ [m s$^{-1}$]", xlabel="Time [s]")
    axes[1, 1].set(title="Bounded motor torques", ylabel="Torque [Nm]", xlabel="Time [s]")
    morphology_areas = []
    for name, rec in task["designs"].items():
        morphology_areas.append((name, _geometry(rec["lengths"], np.inf)["area"], colors[name]))
    axes[0, 2].bar([x[0] for x in morphology_areas], [x[1] for x in morphology_areas],
                   color=[x[2] for x in morphology_areas])
    axes[0, 2].set(title="General capability of each morphology", ylabel="$L_\\infty$ area")
    axes[1, 2].axis("off")
    for ax in axes.ravel():
        ax.grid(True)
        if ax is not axes[1, 2] and ax is not axes[0, 2]: ax.legend(frameon=False)
    fig.tight_layout(); _save(fig, output_dir, "12_task_tracking_comparison")
    return True


def figure_13_redistribution(output_dir, run_dir):
    arrays_path = run_dir / "experiment_arrays.npz"
    if not arrays_path.exists(): return False
    arrays = np.load(arrays_path)
    if "redistribution_samples_3_lengths" not in arrays: return False
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.1))
    n2_x = arrays["redistribution_samples_2_coordinate"]
    n2_y = arrays["redistribution_samples_2_metrics"][:, 2]
    axes[0].plot(n2_x, n2_y, color=COLORS["linf"])
    axes[0].set(title="N=2: $l_1=2+\\delta$", xlabel="$\\delta$ [m]", ylabel="$L_\\infty$ area")
    lengths3 = arrays["redistribution_samples_3_lengths"]
    metrics3 = arrays["redistribution_samples_3_metrics"][:, 2]
    scatter = axes[1].scatter(lengths3[:, 0], lengths3[:, 1], c=metrics3,
                              cmap="viridis", s=16)
    axes[1].set(title="N=3 feasible morphology landscape", xlabel="$l_1$ [m]", ylabel="$l_2$ [m]")
    fig.colorbar(scatter, ax=axes[1], label="$L_\\infty$ area")
    for n, color in ((4, COLORS["l6"]), (5, COLORS["l10"])):
        key_l, key_m = f"redistribution_samples_{n}_lengths", f"redistribution_samples_{n}_metrics"
        if key_l not in arrays: continue
        total_length = np.sum(CONFIG.baseline_lengths)
        nonuniformity = np.sum(np.abs(arrays[key_l] - total_length/n), axis=1) / total_length
        axes[2].scatter(nonuniformity, arrays[key_m][:, 2], s=18, alpha=.7,
                        color=color, label=f"N={n}")
    axes[2].set(title="N=4,5 feasible samples", xlabel="Length nonuniformity descriptor",
                ylabel="$L_\\infty$ area")
    axes[2].legend(frameon=False)
    for ax in axes: ax.grid(True)
    fig.tight_layout(); _save(fig, output_dir, "13_fixed_n_morphology_sweeps")
    return True


def figure_14_zonotope(output_dir):
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.4))
    for n, ax in zip((1, 2), axes):
        lengths = np.full(n, sum(BASELINE_LENGTHS)/n)
        A, B, *_ = robot_matrices(lengths, total_mass=TOTAL_MASS)
        C = output_matrix(lengths)
        S = output_scaling(sum(BASELINE_LENGTHS), CONFIG.horizon)
        zono = propagate_outer_zonotope(
            A, B, actuator_limits(n, "fixed_total"), CONFIG.horizon,
            steps=CONFIG.zonotope_steps,
            max_generators=CONFIG.zonotope_max_generators, output=S @ C,
        )
        _, boundary = projected_zonotope_boundary(zono["center"], zono["generators"], CONFIG.plot_theta_samples)
        direct = _geometry(lengths, np.inf, "peak_equivalent", CONFIG.plot_theta_samples, CONFIG.plot_time_samples)["boundary"]
        zclosed = np.vstack((boundary, boundary[0])); dclosed = np.vstack((direct, direct[0]))
        ax.plot(zclosed[:, 0], zclosed[:, 1], color=COLORS["l4"],
                label="Reduced ZOH zonotope (outer for discretized set)")
        ax.plot(dclosed[:, 0], dclosed[:, 1], color=COLORS["linf"],
                label="Direct continuous-time support")
        ax.set(title=f"N={n} equal links", xlabel="$x_{EE}/L_0$", ylabel="$T\\dot{x}_{EE}/L_0$")
        ax.set_aspect("equal", adjustable="box"); ax.grid(True); ax.legend(frameon=False)
    fig.suptitle("Optional forward zonotope validation; temporal hold relation is not asserted")
    fig.tight_layout(); _save(fig, output_dir, "14_zonotope_validation")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True, help="Exact directory name under N-dof-arm/results/")
    args = parser.parse_args()
    run_dir = RESULTS_ROOT / args.run_id
    path = run_dir / "metadata.json"
    if not path.exists(): raise FileNotFoundError(f"No metadata for explicit run {args.run_id!r}: {path}")
    metadata = json.loads(path.read_text())
    output_dir = run_dir / "plots"; output_dir.mkdir(parents=True, exist_ok=True)
    apply_style()
    baseline = np.asarray(metadata["baseline"]["lengths"], dtype=float)
    reach_best = np.asarray(metadata["capability_optimizations"]["linf_area"]["best_overall"]["lengths"], dtype=float)
    figure_1_analytical(output_dir)
    figure_2_norm_geometry(output_dir, baseline)
    figure_3_support(output_dir, baseline)
    figure_4_convergence(output_dir, baseline)
    figure_5_controls(output_dir, baseline)
    figure_6_costate(output_dir)
    figure_7_morphologies(output_dir, baseline, reach_best)
    figure_8_capability_by_n(output_dir, metadata)
    figure_9_lengths(output_dir, metadata)
    figure_10_cost_tradeoff(output_dir, metadata)
    figure_11_geometry_comparison(output_dir, metadata)
    task_created = figure_12_task(output_dir, run_dir, metadata)
    redistribution_created = figure_13_redistribution(output_dir, run_dir)
    figure_14_zonotope(output_dir)
    print(f"Generated core figures 01–11 and zonotope comparison 14 in {output_dir}")
    print("Generated figure 12 (task tracking)." if task_created else
          "Figure 12 skipped: run task_experiment.py to create task-tracking data.")
    if not redistribution_created:
        print("Figure 13 skipped: generate experiment arrays with experiment_runner.py.")


if __name__ == "__main__":
    main()
