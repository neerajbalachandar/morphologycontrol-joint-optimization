"""Animate the three common-reference designs from one explicit experiment run."""
import argparse
import json

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

from model import kinematic_points
from optimization_variable_n import RESULTS_ROOT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    run_dir = RESULTS_ROOT / args.run_id
    required = (
        run_dir / "metadata.json",
        run_dir / "task_tracking.json",
        run_dir / "task_tracking_arrays.npz",
    )
    missing = [path.name for path in required if not path.is_file()]
    if missing:
        if "metadata.json" in missing:
            raise FileNotFoundError(
                f"Run {args.run_id!r} is missing {', '.join(missing)} under {run_dir}. "
                "Create the run first with: python3 reachability_optimization_dual.py "
                f"--run-id {args.run_id}"
            )
        raise FileNotFoundError(
            f"Run {args.run_id!r} is missing {', '.join(missing)} under {run_dir}. "
            "Generate the task tracking comparison first with: "
            f"python3 task_experiment.py --run-id {args.run_id}"
        )
    metadata = json.loads(required[0].read_text())
    task = json.loads(required[1].read_text())
    arrays = np.load(required[2])
    names = ("baseline", "reachability_optimum", "task_optimum")
    study = metadata.get("study_configuration", {})
    total_length = study.get("baseline_total_length", study.get("fixed_total_length"))
    if total_length is None and "baseline_lengths" in study:
        total_length = sum(study["baseline_lengths"])
    if total_length is None or total_length <= 0:
        raise ValueError(
            f"Run {args.run_id!r} has no positive baseline total length in its "
            "study_configuration metadata."
        )
    limit = 1.1 * float(total_length)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.6))
    arms, trails = [], []
    points = {}
    for ax, name in zip(axes, names):
        record = task["designs"][name]
        lengths = np.asarray(record["lengths"], dtype=float)
        states = arrays[f"{name}_states"]
        q = states[:, :len(lengths)]
        points[name] = np.asarray([kinematic_points(qi, lengths) for qi in q])
        ax.set(xlim=(-limit, limit), ylim=(-limit, 0.25*limit),
               xlabel="x [m]", ylabel="y [m]",
               title=f"{name.replace('_', ' ').title()}\nN={len(lengths)}, l={np.round(lengths, 2)} m")
        ax.set_aspect("equal", adjustable="box"); ax.grid(True, alpha=.2)
        ax.plot([0], [0], "ks", ms=4)
        arm, = ax.plot([], [], "o-", lw=2.5, ms=5)
        trail, = ax.plot([], [], "--", lw=1, alpha=.55)
        arms.append(arm); trails.append(trail)
    time_text = fig.text(.5, .015, "", ha="center")

    def update(frame):
        for i, name in enumerate(names):
            xy = points[name][frame]
            arms[i].set_data(xy[:, 0], xy[:, 1])
            trails[i].set_data(points[name][:frame+1, -1, 0], points[name][:frame+1, -1, 1])
        time_text.set_text(f"Common end-effector tracking task, t={arrays['baseline_times'][frame]:.2f} s")
        return (*arms, *trails, time_text)

    animation = FuncAnimation(fig, update, frames=len(arrays["baseline_times"]),
                              interval=40, blit=False)
    fig.tight_layout(rect=(0, .04, 1, 1))
    plot_dir = run_dir / "plots"; plot_dir.mkdir(parents=True, exist_ok=True)
    out = plot_dir / "morphology_tracking.mp4"
    try:
        animation.save(out, writer="ffmpeg", fps=25, dpi=150)
    except (RuntimeError, FileNotFoundError):
        out = plot_dir / "morphology_tracking.gif"
        animation.save(out, writer="pillow", fps=20, dpi=100)
    plt.close(fig)
    print(f"Saved animation to {out}")


if __name__ == "__main__":
    main()
