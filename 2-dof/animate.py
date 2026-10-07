"""Animate the baseline and saved optimized arms tracking the same motion."""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

from design import BASELINE_LENGTHS, TOTAL_MASS, actuator_limits
from model import kinematic_points, robot_matrices
from task import simulate_task

HERE = Path(__file__).resolve().parent


def best_lengths(filename, key):
    path = HERE / filename
    if not path.exists():
        return None
    results = json.loads(path.read_text())
    results = [
        result for result in results
        if result.get("status", "feasible") == "feasible" and "lengths" in result
    ]
    return min(results, key=key)["lengths"] if results else None


def build_animation():
    cases = [
        ("Baseline", BASELINE_LENGTHS),
        (
            "Task optimal",
            best_lengths("task_optimization_results.json", lambda r: r.get("score", r.get("cost", 1e300))),
        ),
        ("Capability optimal", best_lengths("capability_optimization_results.json", lambda r: r["cost"])),
        ("Reachability optimal", best_lengths("reachability_optimization_results.json", lambda r: -r["area"])),
    ]
    cases = [(name, np.asarray(lengths, dtype=float)) for name, lengths in cases if lengths is not None]

    data = []
    for label, lengths in cases:
        n = len(lengths)
        A, B, *_ = robot_matrices(lengths, total_mass=TOTAL_MASS)
        Q = np.diag(np.r_[100.0 * np.ones(n), 10.0 * np.ones(n)])
        t, X, _ = simulate_task(
            A, B, Q, np.eye(n), T=5.0, dt=0.04, u_max=actuator_limits(n)
        )
        points = np.array([kinematic_points(x[:n], lengths) for x in X])
        data.append((label, lengths, t, X, points))

    max_reach = max(np.sum(lengths) for _, lengths, *_ in data)
    limit = 1.15 * max_reach
    fig, axes = plt.subplots(1, len(data), figsize=(4.5 * len(data), 5), squeeze=False)
    lines, trails = [], []
    for ax, (label, lengths, _, _, points) in zip(axes[0], data):
        ax.set(xlim=(-limit, limit), ylim=(-limit, 0.2 * limit), xlabel="x [m]", ylabel="y [m]", title=f"{label} · {len(lengths)} links")
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.25)
        ax.plot([0], [0], "ks", ms=5)
        arm, = ax.plot([], [], "o-", lw=3, ms=6)
        trail, = ax.plot([], [], "--", lw=1, alpha=0.6)
        lines.append(arm)
        trails.append(trail)

    clock_text = fig.text(0.5, 0.02, "", ha="center")

    def update(frame):
        for arm, trail, (_, _, _, _, points) in zip(lines, trails, data):
            arm.set_data(points[frame, :, 0], points[frame, :, 1])
            trail.set_data(points[: frame + 1, -1, 0], points[: frame + 1, -1, 1])
        clock_text.set_text(f"Tracking motion · t = {data[0][2][frame]:.2f} s")
        return (*lines, *trails, clock_text)

    animation = FuncAnimation(fig, update, frames=len(data[0][2]), interval=40, blit=True)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    return fig, animation


if __name__ == "__main__":
    fig, animation = build_animation()
    # Optional export: animation.save(HERE / "morphology_tracking.gif", writer="pillow", fps=25)
    plt.show()
