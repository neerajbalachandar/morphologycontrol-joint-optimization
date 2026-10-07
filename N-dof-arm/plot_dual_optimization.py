"""Plot saved joint N/length reachability and capability optimization results."""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from design import BASELINE_LENGTHS, TOTAL_MASS, actuator_limits
from model import output_matrix, robot_matrices
from reachability import output_controllability_gramian, reachable_set

HERE = Path(__file__).resolve().parent
JSON_DIR = HERE / "json"
PLOTS_DIR = HERE / "plots"


def load_json(name):
    path = JSON_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}; run both dual optimization scripts first.")
    return json.loads(path.read_text())


def area_and_metrics(lengths):
    lengths = np.asarray(lengths, dtype=float)
    A, B, *_ = robot_matrices(lengths, total_mass=TOTAL_MASS)
    C = output_matrix(lengths)
    _, _, boundary, area = reachable_set(
        A, B, C, actuator_limits(len(lengths)), T=2.0, ntheta=120, nt=250
    )
    W = output_controllability_gramian(A, B, C, T=2.0, nt=250)
    eigs = np.maximum(np.linalg.eigvalsh(W), np.finfo(float).tiny)
    return boundary, area, eigs


def main():
    reach = load_json("reachability_optimization_joint_results.json")
    capability = load_json("capability_optimization_joint_results.json")
    cases = [("Baseline", BASELINE_LENGTHS)]
    if reach["best"]:
        cases.append(("Reachability optimum", reach["best"]["lengths"]))
    if capability["best"]:
        cases.append(("Capability optimum", capability["best"]["lengths"]))

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    ax_set, ax_lengths, ax_area, ax_gramian = axes.ravel()
    colors = plt.get_cmap("tab10")
    for index, (name, lengths) in enumerate(cases):
        color = colors(index)
        boundary, area, eigs = area_and_metrics(lengths)
        closed = np.vstack([boundary, boundary[0]])
        ax_set.plot(closed[:, 0], closed[:, 1], color=color, lw=2,
                    label=f"{name}: N={len(lengths)}, A={area:.2f}")
        positions = np.arange(1, len(lengths) + 1) + (index - (len(cases) - 1) / 2) * 0.12
        ax_lengths.bar(positions, lengths, width=0.12, color=color, label=name)
        ax_gramian.plot([1, 2], eigs, "o-", color=color, label=name)
        ax_area.bar(index, area, color=color)
        ax_area.text(index, area, f" {area:.2f}", va="center", fontsize=8)

    ax_set.set(title="Bounded-torque reachable sets at T=2 s",
               xlabel="$x_{EE}(T)$", ylabel="$\\dot{x}_{EE}(T)$")
    ax_set.set_aspect("equal", adjustable="datalim")
    ax_lengths.set(title="Link lengths in selected designs", xlabel="Link index", ylabel="Length [m]")
    ax_lengths.set_xticks(range(1, max(len(x) for _, x in cases) + 1))
    ax_area.set(title="Reachable-set area", ylabel="Area in $(x_{EE}, \\dot{x}_{EE})$ coordinates")
    ax_area.set_xticks(range(len(cases)), [name.replace(" ", "\n") for name, _ in cases])
    ax_gramian.set(title="Output controllability Gramian eigenvalues", xlabel="Sorted eigenvalue", ylabel="Eigenvalue [log scale]", yscale="log")
    ax_gramian.set_xticks([1, 2], ["smallest", "largest"])

    for ax in axes.ravel():
        ax.grid(True, alpha=0.25)
        if ax.get_legend_handles_labels()[0]:
            ax.legend(fontsize=8)
    fig.tight_layout()

    # Capability target and cost make the constrained objective explicit.
    fig2, (ax_target, ax_cost, ax_by_n) = plt.subplots(1, 3, figsize=(15, 4.5))
    baseline_area = capability["baseline_area"]
    target_area = capability["target_area"]
    target_labels = ["Baseline", "Reachability\noptimum", "Capability\noptimum"]
    target_values = [baseline_area]
    if reach.get("best"):
        target_values.append(reach["best"]["area"])
    else:
        target_values.append(np.nan)
    if capability.get("best"):
        target_values.append(capability["best"]["area"])
    else:
        target_values.append(np.nan)
    bars = ax_target.bar(target_labels, target_values, color=["gray", "tab:blue", "tab:orange"])
    ax_target.axhline(target_area, color="red", linestyle="--", label=f"Capability target ({target_area:.2f})")
    ax_target.bar_label(bars, fmt="%.2f", padding=2)
    ax_target.set(title="Capability target check", ylabel="Reachable-set area")
    ax_target.legend(fontsize=8)

    cap_best = capability.get("best", {})
    cost = cap_best.get("design_change_cost", np.nan)
    ax_cost.bar(["Capability\noptimum"], [cost], color="tab:orange")
    if np.isfinite(cost):
        ax_cost.text(0, cost, f" {cost:.3f}", va="center")
    ax_cost.set(title="Normalized design-change cost", ylabel="Count change + morphology redistribution")
    reach_by_n = {r["N"]: r for r in reach.get("best_evaluated_candidate_by_N", [])}
    cap_by_n = {r["N"]: r for r in capability.get("best_evaluated_candidate_by_N", [])}
    joint_counts = sorted(set(reach_by_n) | set(cap_by_n))
    x = np.arange(len(joint_counts))
    reach_areas = [reach_by_n.get(n, {}).get("area", np.nan) for n in joint_counts]
    cap_areas = [cap_by_n.get(n, {}).get("max_area_evaluated", np.nan) for n in joint_counts]
    ax_by_n.bar(x - 0.18, reach_areas, width=0.36, label="Reachability search")
    ax_by_n.bar(x + 0.18, cap_areas, width=0.36, label="Capability search")
    ax_by_n.axhline(target_area, color="red", linestyle="--", label="Capability target")
    ax_by_n.set(title="Best area evaluated by N", xlabel="Joint count N", ylabel="Reachable-set area")
    ax_by_n.set_xticks(x, joint_counts)
    ax_by_n.legend(fontsize=8)
    for ax in (ax_target, ax_cost, ax_by_n):
        ax.grid(True, axis="y", alpha=0.25)
    fig2.tight_layout()

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    out1 = PLOTS_DIR / "dual_design_comparison.png"
    out2 = PLOTS_DIR / "dual_capability_target_cost.png"
    fig.savefig(out1, dpi=220, bbox_inches="tight")
    fig2.savefig(out2, dpi=220, bbox_inches="tight")
    print(f"Saved plots: {out1} and {out2}")
    plt.show()


if __name__ == "__main__":
    main()
