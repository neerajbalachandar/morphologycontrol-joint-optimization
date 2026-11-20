"""Generate canonical capability, norm-comparison, and validation datasets."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from config import CONFIG
from design import BASELINE_LENGTHS, TOTAL_MASS, actuator_limits
from model import output_matrix, robot_matrices
from optimization_fixed_n import evaluate_capability
from optimization_variable_n import RESULTS_ROOT, run as run_optimization
from reachability import lp_reachable_set, output_scaling
from reachability_costate import sample_costate_reachable_points
from reachability_zonotope import propagate_outer_zonotope, projected_zonotope_boundary


def one_link_sensitivity(lengths=CONFIG.analytical_sweep_points):
    """Vary one-link length at fixed mass, recording changed line density."""
    records = []
    for length in np.linspace(CONFIG.analytical_length_min, CONFIG.analytical_length_max, lengths):
        link = np.array([length])
        A, B, M, K, D, masses, rho = robot_matrices(link, total_mass=TOTAL_MASS)
        C = output_matrix(link)
        measures = {}
        for label, p in (("l2", 2.0), ("l6", 6.0), ("linf", np.inf)):
            result = lp_reachable_set(
                A, B, C, actuator_limits(1, "fixed_total"), CONFIG.horizon,
                p, "peak_equivalent", sum(BASELINE_LENGTHS), CONFIG.theta_samples, CONFIG.time_samples,
            )
            measures[label] = result["area"]
            if label == "l2":
                measures["gramian_trace"] = float(np.trace(result["gramian"]))
                measures["ellipsoid_area"] = result["ellipsoid_area"]
        eigenvalues = np.linalg.eigvals(np.linalg.solve(M, K))
        wn = float(np.sqrt(np.max(np.real(eigenvalues))))
        damping_rate = float(D[0, 0] / M[0, 0])
        records.append({
            "length": float(length), "mass": float(masses[0]),
            "linear_density": float(rho), "l2_gramian_trace": measures["gramian_trace"],
            "l2_ellipsoid_area": measures["ellipsoid_area"],
            "l6_area": measures["l6"], "linf_area": measures["linf"],
            "natural_frequency_rad_s": wn,
            "damping_ratio": float(damping_rate / (2.0 * wn)),
        })
    return records


def norm_comparison(lengths=None, ntheta=CONFIG.plot_theta_samples, nt=CONFIG.plot_time_samples):
    """Peak-equivalent and raw-paper Lp geometry for one morphology."""
    lengths = np.asarray(BASELINE_LENGTHS if lengths is None else lengths, dtype=float)
    A, B, *_ = robot_matrices(lengths, total_mass=TOTAL_MASS)
    C = output_matrix(lengths)
    output = {}
    for convention in ("peak_equivalent", "raw_paper"):
        output[convention] = {}
        for p in tuple(float(v) for v in CONFIG.p_values if v != "infinity") + (np.inf,):
            data = lp_reachable_set(
                A, B, C, actuator_limits(len(lengths), "fixed_total"),
                CONFIG.horizon, p, convention, sum(BASELINE_LENGTHS),
                ntheta, nt, nondimensional=True,
            )
            output[convention]["inf" if np.isinf(p) else str(int(p))] = {
                "p": "infinity" if np.isinf(p) else p,
                "radius": data["radius"], "area": data["area"],
                "ellipsoid_area": data["ellipsoid_area"],
                "theta": data["theta"], "support": data["support"],
                "boundary": data["boundary"], "gramian": data["gramian"],
            }
    return {"lengths": lengths, "horizon": CONFIG.horizon,
            "actuator_limits": actuator_limits(len(lengths), "fixed_total"),
            "output_coordinates": "[x_EE/L0, T*xdot_EE/L0]",
            "norms": output}


def redistribution_samples(seed=CONFIG.random_seed, random_samples=CONFIG.redistribution_random_samples):
    """Sample fixed-total morphology spaces for N=2..5 before optimization."""
    rng = np.random.default_rng(seed)
    all_samples = {}
    for n in range(2, CONFIG.n_max + 1):
        if n == 2:
            half_length = sum(BASELINE_LENGTHS) / 2.0
            delta_lo = max(CONFIG.length_min - half_length, half_length - CONFIG.length_max)
            delta_hi = min(CONFIG.length_max - half_length, half_length - CONFIG.length_min)
            deltas = np.linspace(delta_lo, delta_hi, CONFIG.redistribution_grid_points)
            morphologies = [np.array([half_length + delta, half_length - delta]) for delta in deltas]
            coordinates = deltas
        elif n == 3:
            grid = np.linspace(CONFIG.length_min, CONFIG.length_max, CONFIG.redistribution_grid_points)
            morphologies = []
            coordinates = []
            L0 = sum(BASELINE_LENGTHS)
            for l1 in grid:
                for l2 in grid:
                    l3 = L0 - l1 - l2
                    if CONFIG.length_min <= l3 <= CONFIG.length_max:
                        morphologies.append(np.array([l1, l2, l3]))
                        coordinates.append([l1, l2])
        else:
            morphologies = []
            while len(morphologies) < random_samples:
                raw = rng.dirichlet(np.ones(n)) * sum(BASELINE_LENGTHS)
                if np.all(raw >= CONFIG.length_min) and np.all(raw <= CONFIG.length_max):
                    morphologies.append(raw)
            coordinates = np.arange(len(morphologies))
        metrics = []
        for lengths in morphologies:
            l2 = evaluate_capability(lengths, "l2_trace")
            l6 = evaluate_capability(lengths, "l6_area")
            linf = evaluate_capability(lengths, "linf_area")
            metrics.append([l2["objective"], l6["objective"], linf["objective"]])
        all_samples[str(n)] = {"lengths": np.asarray(morphologies),
                               "coordinate": np.asarray(coordinates),
                               "metrics": np.asarray(metrics)}
    return all_samples


def costate_and_zonotope():
    """Generate small-N PMP samples and optional outer-zonotope comparisons."""
    output = {}
    for n in (1, 2):
        lengths = np.full(n, sum(BASELINE_LENGTHS) / n)
        A, B, *_ = robot_matrices(lengths, total_mass=TOTAL_MASS)
        C = output_matrix(lengths)
        tau = actuator_limits(n, "fixed_total")
        costate = sample_costate_reachable_points(
            A, B, C, tau, CONFIG.horizon, 6.0, "peak_equivalent",
            samples=CONFIG.costate_samples, seed=CONFIG.random_seed + n,
            time_samples=CONFIG.costate_time_samples,
        )
        costate["direct_support"] = lp_reachable_set(
            A, B, C, tau, CONFIG.horizon, 6.0, "peak_equivalent",
            sum(BASELINE_LENGTHS), CONFIG.plot_theta_samples, CONFIG.plot_time_samples,
        )
        zono = propagate_outer_zonotope(
            A, B, tau, CONFIG.horizon, steps=CONFIG.zonotope_steps,
            max_generators=CONFIG.zonotope_max_generators,
            output=output_scaling(sum(BASELINE_LENGTHS), CONFIG.horizon) @ C,
        )
        ztheta, zboundary = projected_zonotope_boundary(zono["center"], zono["generators"])
        output[str(n)] = {"lengths": lengths, "costate": costate,
                          "zonotope": {**zono, "theta": ztheta, "boundary": zboundary}}
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default="canonical_fixed_total")
    parser.add_argument("--skip-optimization", action="store_true")
    parser.add_argument("--skip-redistribution", action="store_true")
    args = parser.parse_args()
    run_dir = RESULTS_ROOT / args.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    if not args.skip_optimization:
        run_optimization(args.run_id)
    auxiliary = {
        "one_link_sensitivity": one_link_sensitivity(),
        "norm_comparison": norm_comparison(),
        "costate_and_zonotope": costate_and_zonotope(),
    }
    if not args.skip_redistribution:
        auxiliary["redistribution_samples"] = redistribution_samples()
    numeric = {}

    def flatten(prefix, value):
        if isinstance(value, dict):
            for key, child in value.items():
                flatten(f"{prefix}_{key}" if prefix else str(key), child)
        elif isinstance(value, np.ndarray):
            if value.dtype != object:
                numeric[prefix] = value
        elif isinstance(value, (float, int, np.floating, np.integer)):
            numeric[prefix] = np.asarray(value)
        elif isinstance(value, list) and value and isinstance(value[0], dict):
            for i, child in enumerate(value):
                flatten(f"{prefix}_{i}", child)
        elif isinstance(value, list):
            try:
                numeric[prefix] = np.asarray(value)
            except (TypeError, ValueError):
                pass

    flatten("", auxiliary)
    np.savez_compressed(run_dir / "experiment_arrays.npz", **numeric)
    (run_dir / "experiment_metadata.json").write_text(json.dumps({
        "run_id": args.run_id,
        "study_configuration": CONFIG.snapshot(),
        "dataset_contents": ["one-link length sensitivity", "Lp and L-infinity boundaries under peak-equivalent and raw-paper norms",
                             "Sobol costate samples for N=1,2", "optional reduced outer zonotope", "fixed-N morphology samples"],
        "zonotope_inclusion": "outer approximation after exact-ZOH discretization; generator truncation is enclosed by an axis-aligned box",
        "one_link_mass_note": "mass is held at configured total mass, so linear density changes with link length",
    }, indent=2) + "\n")
    print(f"Saved auxiliary experiment arrays under {run_dir}")


if __name__ == "__main__":
    main()
