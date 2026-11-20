# N-link reachability and morphology study

The primary workflow compares end-effector reachability at fixed total length,
mass, and total actuator torque. All study settings live in `config.py`.
`optimization_variable_n.py` is the only primary variable-N optimizer: it
solves each feasible fixed-N link problem and then compares the results.
Tracking is separate and optional.

## Recommended command order

Run commands from this folder:

```bash
cd N-dof-arm
```

1. Optimize the three capability criteria and make a named results run:

```bash
python3 optimization_variable_n.py --run-id canonical_fixed_total
```

This independently optimizes link lengths for every feasible N for L2 Gramian
trace, L6 reachable area, and L-infinity reachable area. It also checks the
5%, 10%, 20%, and 30% minimum-cost targets. Default settings are one seeded
differential-evolution run per N.

2. Generate sensitivity, validation, and morphology-sweep arrays for that run:

```bash
python3 experiment_runner.py --run-id canonical_fixed_total --skip-optimization
```

This calculates the one-link sweep, Lp norm comparisons (including raw-paper
norm sensitivity), Sobol costate samples for N=1,2, the optional reduced outer
zonotope comparison, and fixed-N redistribution samples. Some of these
calculations are more expensive than the core optimization.

3. Optionally run the separate bounded tracking QP comparison:

```bash
python3 task_experiment.py --run-id canonical_fixed_total
```

It compares the baseline, L-infinity reachability morphology, and a
task-specific morphology under one common physical end-effector reference.

4. Generate the publication-style PNG and PDF figures for exactly that run:

```bash
python3 -m plotting.plot_all --run-id canonical_fixed_total
```

Figures are written to `results/canonical_fixed_total/plots/`. Figures 01–11
and 14 are generated from the reachability run; Figure 12 appears when step 3
has created task data. Figure 13 appears when step 2 has created redistribution
arrays.

5. Optionally animate the three tracking morphologies:

```bash
python3 animate.py --run-id canonical_fixed_total
```

This saves an MP4, or a GIF when ffmpeg is unavailable, under the same run's
`plots/` directory.

## Sensitivity and output files

To compare the different actuator resource assumption, run the optimizer with
`--actuator-convention fixed_per_joint`. This gives every motor 1 Nm and is a
sensitivity study; aggregate authority therefore grows with N. Primary results
use `fixed_total`, with each motor limited to `2/N Nm`.

Each `results/<run-id>/` contains `metadata.json`, optimizer candidate and
boundary arrays in NPZ, auxiliary experiment arrays, optional task arrays, and
the plots. Always pass the desired run ID to the plotting command; it does not
select files based on modification time.

Old optimizer and plotting command names remain as compatibility entry points
into the nested optimizer or unified plotting pipeline. The padded
mixed-integer algorithm, old identical-joint-angle objective, and clipped LQT
controller are retired. Existing result JSON and image/video files are left
in place and are not silently reinterpreted.
