"""Animate the selected design from each optimization family side by side."""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

from design import BASELINE_LENGTHS, TOTAL_MASS, actuator_limits
from model import kinematic_points, robot_matrices
from task import simulate_task

HERE = Path(__file__).resolve().parent
JSON_DIR = HERE / "json"
PLOTS_DIR = HERE / "plots"
T = 5.0
DT = 0.04


def read_results(name):
    candidates = [path for path in (JSON_DIR / name, HERE / name) if path.exists()]
    if not candidates:
        return None
    path = max(candidates, key=lambda candidate: candidate.stat().st_mtime)
    return json.loads(path.read_text())


def animation_cases():
    cases = [("Baseline", np.asarray(BASELINE_LENGTHS, dtype=float))]
    task = read_results("task_optimization_results.json") or []
    task = [r for r in task if "lengths" in r]
    if task:
        row = min(task, key=lambda r: r.get("score", r.get("cost", 1e300)))
        cases.append((f"Task optimum · N={row['N']}", np.asarray(row["lengths"], dtype=float)))

    cap = read_results("capability_optimization_results.json") or []
    cap = [r for r in cap if r.get("status", "feasible") == "feasible" and "lengths" in r]
    if cap:
        row = min(cap, key=lambda r: r["cost"])
        cases.append((f"Capability fixed N · N={row['N']}", np.asarray(row["lengths"], dtype=float)))

    reach = read_results("reachability_optimization_results.json") or []
    reach = [r for r in reach if "lengths" in r and "area" in r]
    if reach:
        row = max(reach, key=lambda r: r["area"])
        cases.append((f"Reachability fixed N · N={row['N']}", np.asarray(row["lengths"], dtype=float)))

    cap_dual = read_results("capability_optimization_joint_results.json")
    if cap_dual and cap_dual.get("best", {}).get("lengths"):
        row = cap_dual["best"]
        cases.append((f"Capability dual · N={row['N']}", np.asarray(row["lengths"], dtype=float)))

    reach_dual = read_results("reachability_optimization_joint_results.json")
    if reach_dual and reach_dual.get("best", {}).get("lengths"):
        row = reach_dual["best"]
        cases.append((f"Reachability dual · N={row['N']}", np.asarray(row["lengths"], dtype=float)))
    return cases


def build_animation():
    cases = animation_cases()
    data = []
    for label, lengths in cases:
        n = len(lengths)
        A, B, *_ = robot_matrices(lengths, total_mass=TOTAL_MASS)
        Q = np.diag(np.r_[100.0 * np.ones(n), 10.0 * np.ones(n)])
        t, X, _ = simulate_task(
            A, B, Q, np.eye(n), T=T, dt=DT, u_max=actuator_limits(n)
        )
        points = np.array([kinematic_points(x[:n], lengths) for x in X])
        data.append((label, lengths, t, points))

    # All cases have common T and dt; derive a shared frame count defensively.
    frames = min(len(item[2]) for item in data)
    total_length = max(np.sum(lengths) for _, lengths, _, _ in data)
    limit = 1.12 * total_length
    rows, columns = 2, 3
    fig, axes = plt.subplots(rows, columns, figsize=(14, 8), squeeze=False)
    lines, trails = [], []
    for idx, ax in enumerate(axes.ravel()):
        if idx >= len(data):
            ax.set_visible(False)
            continue
        label, lengths, _, _ = data[idx]
        ax.set(
            xlim=(-limit, limit), ylim=(-limit, 0.2 * limit),
            xlabel="x [m]", ylabel="y [m]",
            title=f"{label}\nlinks: {np.round(lengths, 2)} m",
        )
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.25)
        ax.plot([0], [0], "ks", ms=4)
        arm, = ax.plot([], [], "o-", lw=2.5, ms=5)
        trail, = ax.plot([], [], "--", lw=1, alpha=0.6)
        lines.append(arm)
        trails.append(trail)

    clock_text = fig.text(0.5, 0.015, "", ha="center")

    def update(frame):
        for i, (arm, trail) in enumerate(zip(lines, trails)):
            points = data[i][3]
            arm.set_data(points[frame, :, 0], points[frame, :, 1])
            trail.set_data(points[:frame + 1, -1, 0], points[:frame + 1, -1, 1])
        clock_text.set_text(f"Same bounded-torque tracking simulation · t={data[0][2][frame]:.2f} s")
        return (*lines, *trails, clock_text)

    # Figure-level time text and TkAgg blitting are incompatible in some
    # Matplotlib releases; use normal frame redraws for backend portability.
    animation = FuncAnimation(fig, update, frames=frames, interval=40, blit=False)
    fig.suptitle("Selected morphology from each optimization result")
    fig.tight_layout(rect=(0, 0.04, 1, 0.96))
    return fig, animation


if __name__ == "__main__":
    fig, animation = build_animation()
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    video_path = PLOTS_DIR / "all_optimization_designs.mp4"
    try:
        animation.save(video_path, writer="ffmpeg", fps=25, dpi=150)
        print(f"Saved animation to {video_path}")
    except (RuntimeError, FileNotFoundError) as error:
        gif_path = PLOTS_DIR / "all_optimization_designs.gif"
        animation.save(gif_path, writer="pillow", fps=25, dpi=100)
        print(f"MP4 export unavailable ({error}); saved GIF to {gif_path}")
    plt.show()
