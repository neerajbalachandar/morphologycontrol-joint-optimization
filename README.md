# Morphology–control optimization

Use one root-level script per optimization workflow. Each script sets its study
parameters at the top, runs that optimization, saves its result, and creates
the matching plots. The shared dynamics and numerical methods remain in
`N-dof-arm/` so the physical model has one implementation.

Run from the repository root:

```bash
python3 reachability_optimization.py
python3 capability_optimization.py
python3 task_optimization.py
```

The first script maximizes the bounded-torque L-infinity reachable area while
comparing feasible link counts. The second finds the lowest-cost morphology
meeting each configured reachability-improvement target. The third minimizes
the common-reference tracking objective and compares its trajectory and motor
torques with the baseline morphology.

Each script writes JSON/NPZ data and PNG/PDF plots below
`N-dof-arm/results/<run-id>/`; task optimization also writes an MP4 animation,
or a GIF if ffmpeg is unavailable. Edit `STUDY_OVERRIDES` and `RUN_ID` near the
top of the corresponding script to change its study parameters and output
folder. The original detailed numerical modules and legacy command wrappers
remain in `N-dof-arm/` for compatibility and reference.
