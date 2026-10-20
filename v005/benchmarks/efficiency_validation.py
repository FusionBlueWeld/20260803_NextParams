"""Read-only historical replay and bounded, process-separated efficiency checks.

Historical replay uses the predictions which existed before each actual batch.
It does not retrain, alter historical trials, or treat unmeasured points as known.
The press benchmark exposes separate prepare/suggest/ingest commands so Codex
can relay a JSON request to an independent simulator process between rounds.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import inspect
import json
import os
from pathlib import Path
import platform
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

TRIALS = ROOT / "v005" / "trials"
RESULTS = Path(__file__).resolve().parent / "results" / "efficiency_20261010"
PRESS_SOURCE = TRIALS / "trial_single_press_forming_20261010"


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def csv_rows(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_rows(path, columns, rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def input_key(row, names):
    """Match CSV-published grid values, never nearby arbitrary observations."""
    return tuple(round(float(row[name]), 8) for name in names)


def peak_process_memory_bytes():
    """Report this worker's peak resident memory, with no system-wide polling."""
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes
        class Counters(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD)] + [
                (name, ctypes.c_size_t) for name in (
                    "PeakWorkingSetSize", "WorkingSetSize", "QuotaPeakPagedPoolUsage",
                    "QuotaPagedPoolUsage", "QuotaPeakNonPagedPoolUsage", "QuotaNonPagedPoolUsage",
                    "PagefileUsage", "PeakPagefileUsage")]
        counters = Counters()
        counters.cb = ctypes.sizeof(counters)
        kernel = ctypes.WinDLL("kernel32")
        kernel.GetCurrentProcess.restype = ctypes.c_void_p
        psapi = ctypes.WinDLL("psapi")
        psapi.GetProcessMemoryInfo.argtypes = [ctypes.c_void_p, ctypes.POINTER(Counters), wintypes.DWORD]
        if not psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
            return None
        return int(counters.PeakWorkingSetSize)
    import resource
    raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(raw if sys.platform == "darwin" else raw * 1024)


def observations(rows, problem):
    from v005.src.validation import ExperimentData
    return ExperimentData(
        [r["experiment_id"] for r in rows],
        [[float(r[p.column]) for p in problem.parameters] for r in rows],
        {v.column: [float(r[v.column]) for r in rows] for v in problem.result_variables},
    )


class SavedPredictionModel:
    """A model adapter backed solely by a saved pre-measurement response CSV."""
    def __init__(self, problem, points, values, support, output_scales):
        import numpy as np
        self.results = values
        self.support = support
        self.lookup = {tuple(np.round(row, 10)): i for i, row in enumerate(points)}
        self.gp_models = {v.column: SimpleNamespace(y_scale=float(output_scales[v.column]))
                          for v in problem.result_variables}

    def predict(self, points):
        import numpy as np
        from v005.src.hybrid.model import HybridPrediction
        indices = np.array([self.lookup[tuple(np.round(row, 10))] for row in points])
        return HybridPrediction(self.support[indices],
            {name: {field: data[indices] for field, data in fields.items()}
             for name, fields in self.results.items()})


def optimization_options(config=None):
    """Keep validation runnable while the production policy interface evolves."""
    from v005.src.hybrid.optimizer import run_optimization
    if config is None:
        from v005.src.search_config import SearchConfig
        config = SearchConfig()
    parameters = inspect.signature(run_optimization).parameters
    for name in ("search_config", "search_settings", "recommendation_config"):
        if name in parameters:
            return {name: config}
    return {}


def replay_case(trial_name, run_number):
    import numpy as np
    from v005.src.hybrid.optimizer import run_optimization
    from v005.src.preprocessing import normalize_parameters, prepare_experiments
    from v005.src.validation import load_and_validate_problem
    started = time.perf_counter()
    trial = TRIALS / trial_name
    contract = read_json(trial / "simulator_contract.json")
    problem = load_and_validate_problem(trial / "problem.csv")
    path = trial / "output" / "response_spaces" / f"hybrid_response_space_run_{run_number:04d}.csv"
    # Numeric arrays are retained rather than all dictionaries from a large CSV.
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        first = next(reader)
        source_rows = int(first["data_rows"])
        columns = [p.column for p in problem.parameters]
        mapping = {
            "hybrid_mean": "hybrid_mean", "hybrid_std": "hybrid_std",
            "acquisition_std": "acquisition_std", "nn_pred": "nn_pred",
            "nn_std": "nn_std", "raw_hybrid_std": "raw_hybrid_std",
            "uncertainty_calibration_scale": "calibration_scale",
        }
        wanted = columns + ["gp_support"] + [f"{v.column}_{suffix}"
                   for v in problem.result_variables for suffix in mapping.values()]
        import itertools
        numeric = (float(row[c]) for row in itertools.chain([first], reader) for c in wanted)
        matrix = np.fromiter(numeric, dtype=float).reshape(-1, len(wanted))
    grid = matrix[:, :len(columns)]
    positions = {name: i for i, name in enumerate(wanted)}
    values = {v.column: {field: matrix[:, positions[f"{v.column}_{suffix}"]]
              for field, suffix in mapping.items()} for v in problem.result_variables}
    support = matrix[:, positions["gp_support"]]
    data = csv_rows(trial / "data" / "experiments.csv")
    experiments = observations(data[:source_rows], problem)
    prepared = prepare_experiments(experiments, problem)
    bundle_dir = trial / "output" / "stage_bundles" / f"stage_bundle_run_run_{run_number:04d}"
    if (bundle_dir / "stage_manifest.json").exists():
        manifest = read_json(bundle_dir / "stage_manifest.json")
        with np.load(bundle_dir / manifest["predictor"], allow_pickle=False) as archive:
            scales = {name: float(archive[f"gp_{i}_settings"][1])
                      for i, name in enumerate(manifest["predicted_outputs"])}
        scale_source = "saved safe model weights"
    else:
        # Native trials did not export stage bundles. Reconstruct only the GP
        # residual scale from the saved NN values at the already measured grid
        # points; this is not fitting a model or obtaining any new physics data.
        from v005.src.settings import DEFAULT_NOISE_RATIO
        control_columns = [columns.index(n) for n in contract["controls"]]
        lookup = {tuple(np.round(row[control_columns], 8)): i for i, row in enumerate(grid)}
        at_data = np.array([lookup[tuple(np.round(row[control_columns], 8))]
                            for row in prepared.raw_parameters])
        scales = {}
        for v in problem.result_variables:
            residual = prepared.result_means[v.column] - values[v.column]["nn_pred"][at_data]
            scale = float(np.std(residual))
            scales[v.column] = scale if scale >= 1e-12 else max(abs(float(np.mean(residual))) * DEFAULT_NOISE_RATIO, 1.0)
        scale_source = "reconstructed residual scale from CSV-published NN predictions at measured points"
    model = SavedPredictionModel(problem, normalize_parameters(grid, problem), values, support, scales)
    result = run_optimization(experiments, prepared, problem, 5, model,
        candidate_grid=grid,
        candidate_identity_columns=np.array([columns.index(n) for n in contract["controls"]]),
        **optimization_options())
    previous = csv_rows(trial / "steps" / f"step_{run_number:03d}" / "recommendations.csv")
    measured_next = data[source_rows:source_rows + len(previous)]
    truth = {input_key(row, contract["controls"]): row for row in measured_next}
    def passed(row):
        return all(float(row[c["name"]]) <= c["target"] + 1e-12 if c["direction"] == "less_equal"
                   else float(row[c["name"]]) >= c["target"] - 1e-12 for c in contract["constraints"])
    selected = []
    for row in result.recommendations:
        observed = truth.get(input_key(row, contract["controls"]))
        selected.append({"rank": int(row["rank"]), "selection_role": row.get("selection_role"),
            "feasibility_probability": float(row["feasibility_probability"]),
            "score": float(row["recommendation_score"]),
            "controls": {n: row[n] for n in contract["controls"]},
            "actual_passed_if_previously_measured": None if observed is None else passed(observed)})
    selected_keys = {input_key(row, contract["controls"]) for row in result.recommendations}
    old_low = [row for row in previous if float(row["feasibility_probability"]) < 0.05]
    old_failed_keys = {input_key(row, contract["controls"]) for row in measured_next if not passed(row)}
    old_passed_keys = {input_key(row, contract["controls"]) for row in measured_next if passed(row)}
    return {"trial": trial_name, "run": run_number, "observations": source_rows,
        "had_feasible_observation": result.had_feasible_observation,
        "previous_count": len(previous), "new_count": len(selected),
        "previous_actual_passed": sum(passed(row) for row in measured_next),
        "previous_low_probability_count": len(old_low),
        "old_low_probability_retained": sum(input_key(row, contract["controls"]) in selected_keys for row in old_low),
        "previous_actual_failed_retained": len(selected_keys & old_failed_keys),
        "previous_actual_passed_retained": len(selected_keys & old_passed_keys),
        "new_low_probability_count": sum(row["feasibility_probability"] < 0.05 for row in selected),
        "selected": selected, "warnings": result.warnings,
        "seconds": time.perf_counter() - started,
        "scale_source": scale_source,
        "response_sha256": file_sha256(path),
        "interpretation": "same historical model and observations; newly selected unmeasured points have unknown truth"}


REPLAY_CASES = [("trial_assembly_20261010", run) for run in range(17, 21)] + [
    ("trial_single_laser_welding_20261010", 7), ("trial_single_laser_welding_20261010", 8),
    ("trial_interconnect_welding_20261010", 1)]


def replay(output):
    cases = []
    for name, run in REPLAY_CASES:
        result = replay_case(name, run)
        cases.append(result)
        print(json.dumps({k: result[k] for k in ("trial", "run", "new_count", "old_low_probability_retained", "seconds")}), flush=True)
        write_json(output, {"scope": "historical replay without retraining", "cases": cases})


def prepare_benchmark(output, seeds=(11, 29, 47), initial=12, budget=32):
    """Stdlib file relay plus local RNG; simulator is intentionally not imported."""
    import itertools
    import numpy as np
    contract = read_json(PRESS_SOURCE / "simulator_contract.json")
    points = list(itertools.product(*(contract["axes"][n] for n in contract["controls"])))
    folders = []
    for seed in seeds:
        indices = np.random.default_rng(seed).choice(len(points), initial, replace=False)
        conditions = [{"experiment_id": f"initial_{i:03d}", **dict(zip(contract["controls"], points[j]))}
                      for i, j in enumerate(indices)]
        for method in ("random", "nn", "hybrid"):
            trial = Path(output) / f"press_seed_{seed}_{method}"
            trial.mkdir(parents=True, exist_ok=True)
            # Preserve the exact simulator contract and problem used in previous evaluation.
            for name in ("problem.csv", "simulator_contract.json", "spec.json"):
                (trial / name).write_bytes((PRESS_SOURCE / name).read_bytes())
            write_json(trial / "benchmark.json", {"method": method, "seed": seed, "initial": initial,
                "budget": budget, "max_training_samples": 64, "max_candidates": 1521,
                "max_nn_epochs": 300, "ensemble_size": 2,
                "process_protocol": "suggest -> request JSON -> independent simulator -> measurement JSON -> ingest"})
            request = trial / "exchange" / "request_000.json"
            write_json(request, {"step": 0, "conditions": conditions})
            folders.append(str(trial.relative_to(ROOT)))
    write_json(Path(output) / "plan.json", {"trials": folders, "seeds": list(seeds),
        "same_initial_conditions_per_seed": True, "training_parallelism": 1,
        "limits": {"observations": budget, "candidates": 1521, "epochs": 300, "ensemble": 2},
        "simulator_command": "python -m v005.trials.trial_single_press_forming_20261010.simulator_worker evaluate --trial TRIAL --request REQUEST"})
    write_json(Path(output) / "environment.json", {"python": sys.version, "numpy": np.__version__,
        "platform": platform.platform(), "logical_cpus": os.cpu_count(),
        "blas_environment": {n: os.environ.get(n) for n in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")},
        "parallel_trainings": 1, "simulator_independent_process": True})
    print(json.dumps({"prepared": len(folders), "initial": initial, "budget": budget}))


def ingest_benchmark(trial, request):
    """Only file operations; no optimizer or simulator import in this process."""
    trial = Path(trial)
    request = Path(request)
    measurement = request.with_name(request.name.replace("request_", "measurement_"))
    received = read_json(measurement)
    if received["request_sha256"] != hashlib.sha256(request.read_bytes()).hexdigest():
        raise ValueError("measurement request hash mismatch")
    contract = read_json(trial / "simulator_contract.json")
    path = trial / "data" / "experiments.csv"
    data = csv_rows(path) if path.exists() else []
    ids = {r["experiment_id"] for r in data}
    for result in received["results"]:
        if result["experiment_id"] in ids:
            raise ValueError("same measurement must not be ingested twice")
        data.append({"experiment_id": result["experiment_id"], **result["inputs"],
                     **{n: result["outputs"][n] for n in contract["outputs"]}})
    write_rows(path, ["experiment_id", *contract["parameters"], *contract["outputs"]], data)
    write_json(trial / "state.json", {"step": int(received["step"]), "observations": len(data)})
    print(json.dumps({"ingested": len(received["results"]), "observations": len(data)}))


def suggest_benchmark(trial):
    import itertools
    import numpy as np
    from v005.src.hybrid.optimizer import run_optimization, remove_measured_candidates
    from v005.src.hybrid.model import train_hybrid_model
    from v005.src.preprocessing import prepare_experiments
    from v005.src.validation import load_and_validate_problem
    from v005.src.search_config import SearchConfig
    trial = Path(trial)
    settings = read_json(trial / "benchmark.json")
    contract = read_json(trial / "simulator_contract.json")
    state = read_json(trial / "state.json")
    data = csv_rows(trial / "data" / "experiments.csv")
    remaining = settings["budget"] - len(data)
    if remaining <= 0:
        print(json.dumps({"status": "BUDGET", "observations": len(data)}))
        return
    started = time.perf_counter()
    cpu_started = time.process_time()
    problem = load_and_validate_problem(trial / "problem.csv")
    experiments = observations(data, problem)
    prepared = prepare_experiments(experiments, problem)
    grid = np.array(list(itertools.product(*(contract["axes"][n] for n in contract["controls"]))))
    maximum = min(5, remaining)
    if settings["method"] == "random":
        candidates = remove_measured_candidates(grid, prepared.raw_parameters)
        rng = np.random.default_rng(settings["seed"] + 1009 * (state["step"] + 1))
        selected = candidates[rng.choice(len(candidates), maximum, replace=False)]
        recommendations = [{"rank": i + 1, "selection_role": "UNIFORM_RANDOM", **dict(zip(contract["controls"], row))}
                           for i, row in enumerate(selected)]
        training_seconds = 0.0
    else:
        trained = train_hybrid_model(prepared, problem, ensemble_size=2, max_training_samples=64,
                                     max_nn_epochs=300, max_learning_seconds=10.0)
        model = trained.model
        if settings["method"] == "nn":
            model = NNOnlyPredictionModel(model, prepared)
        config = SearchConfig(max_candidates=1521, max_training_samples=64, max_nn_epochs=300,
                              max_learning_seconds=10.0, seed=settings["seed"])
        optimized = run_optimization(experiments, prepared, problem, maximum, model,
                                     candidate_grid=grid, **optimization_options(config))
        recommendations = optimized.recommendations
        training_seconds = trained.learning_seconds
    step = state["step"] + 1
    folder = trial / "steps" / f"step_{step:03d}"
    write_json(folder / "recommendations.json", recommendations)
    conditions = [{"experiment_id": f"step_{step:03d}_rank_{r['rank']}",
                   **{n: float(r[n]) for n in contract["parameters"]}} for r in recommendations]
    request = trial / "exchange" / f"request_{step:03d}.json"
    if conditions:
        write_json(request, {"step": step, "conditions": conditions})
    write_json(folder / "runtime.json", {"seconds": time.perf_counter() - started,
        "cpu_seconds": time.process_time() - cpu_started,
        "peak_resident_memory_bytes": peak_process_memory_bytes(),
        "training_seconds": training_seconds, "source_observations": len(data),
        "returned_count": len(conditions), "requested_maximum": maximum,
        "note": "NN-only comparison fits the same ensemble/residual backend but disables GP prediction; runtime is not a pure NN training cost comparison"})
    print(json.dumps({"status": "READY" if conditions else "NO_VALUE", "step": step,
                      "count": len(conditions), "request": str(request)}))


class NNOnlyPredictionModel:
    """NN mean and ensemble spread with observed residual floor, no GP correction."""
    def __init__(self, trained, prepared):
        import numpy as np
        self.trained = trained
        self.gp_models = trained.gp_models
        at_data = trained.nn_model.predict(prepared.normalized_parameters)
        self.noise = {name: max(prepared.result_noise_std[name],
                              float(np.sqrt(np.mean((prepared.result_means[name] - at_data[:, i]) ** 2))))
                      for i, name in enumerate(trained.result_columns)}

    def predict(self, points):
        import numpy as np
        from v005.src.hybrid.model import HybridPrediction
        values = self.trained.nn_model.predict_members(points)
        mean = np.mean(values, axis=0)
        spread = np.std(values, axis=0, ddof=1)
        result = {}
        for i, name in enumerate(self.trained.result_columns):
            raw = np.sqrt(spread[:, i] ** 2 + self.noise[name] ** 2)
            result[name] = {"hybrid_mean": mean[:, i], "nn_pred": mean[:, i], "nn_std": spread[:, i],
                "hybrid_std": self.trained.uncertainty_calibration_scale * raw,
                "acquisition_std": self.trained.acquisition_uncertainty_scale * raw,
                "raw_hybrid_std": raw,
                "uncertainty_calibration_scale": np.full(len(points), self.trained.uncertainty_calibration_scale)}
        return HybridPrediction(self.trained.support_model.predict(points), result)


def summarize_benchmark(output):
    import numpy as np
    summaries = []
    optimum = read_json(PRESS_SOURCE / "posthoc_grid_audit.json")["best_value"]
    for trial in sorted(Path(output).glob("press_seed_*")):
        settings = read_json(trial / "benchmark.json")
        contract = read_json(trial / "simulator_contract.json")
        data = csv_rows(trial / "data" / "experiments.csv")
        requested_ids = []
        for request in sorted((trial / "exchange").glob("request_*.json")):
            received = read_json(request.with_name(request.name.replace("request_", "measurement_")))
            if received["request_sha256"] != file_sha256(request):
                raise ValueError("historical benchmark measurement hash mismatch")
            requested_ids.extend(row["experiment_id"] for row in received["results"])
        if requested_ids != [row["experiment_id"] for row in data]:
            raise ValueError("measurement ledger and experiment CSV order mismatch")
        identities = [input_key(row, contract["controls"]) for row in data]
        if len(set(identities)) != len(identities):
            raise ValueError("duplicate physical condition in benchmark")
        best = None
        history = []
        failures = 0
        first_optimum = None
        for i, row in enumerate(data, 1):
            passed = all(float(row[c["name"]]) <= c["target"] + 1e-12 if c["direction"] == "less_equal"
                         else float(row[c["name"]]) >= c["target"] - 1e-12 for c in contract["constraints"])
            failures += not passed
            if passed:
                value = float(row[contract["objective"]])
                best = value if best is None else min(best, value)
                if first_optimum is None and abs(best - optimum) <= 1e-10:
                    first_optimum = i
            history.append({"observations": i, "best": best, "passed": passed})
        runtime = [read_json(p) for p in (trial / "steps").glob("step_*/runtime.json")]
        summary = {"trial": trial.name, "seed": settings["seed"], "method": settings["method"],
            "observations": len(data), "initial_best": history[settings["initial"] - 1]["best"],
            "final_best": best, "grid_optimum": optimum, "gap": None if best is None else best - optimum,
            "first_optimum_observation": first_optimum, "failed_measurements": failures,
            "suggestion_seconds": sum(r["seconds"] for r in runtime),
            "training_seconds": sum(r["training_seconds"] for r in runtime), "history": history}
        summary["cpu_seconds"] = sum(r.get("cpu_seconds", 0.0) for r in runtime)
        summary["peak_resident_memory_bytes"] = max((r.get("peak_resident_memory_bytes") or 0 for r in runtime), default=0)
        summary["unique_conditions"] = len(set(identities))
        summaries.append(summary)
    write_json(Path(output) / "summary.json", {"scope": "3 seed short-budget no-noise press smoke benchmark; not high-dimension generalization proof",
        "nn_baseline": "same fitted ensemble; GP correction disabled; uncertainty floor is in-sample residual RMSE; training cost includes unused GP fit",
        "oracle_used_only_after_search": True, "trials": summaries})
    write_json(Path(output) / "environment.json", {"python": sys.version, "numpy": np.__version__,
        "platform": platform.platform(), "logical_cpus": os.cpu_count(),
        "blas_environment": {n: os.environ.get(n) for n in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")},
        "parallel_trainings": 1, "simulator_independent_process": True,
        "total_measured_conditions": sum(r["observations"] for r in summaries),
        "training_and_suggestion_seconds": sum(r["suggestion_seconds"] for r in summaries),
        "total_suggestion_cpu_seconds": sum(r["cpu_seconds"] for r in summaries),
        "peak_suggestion_resident_memory_bytes": max(r["peak_resident_memory_bytes"] for r in summaries),
        "hashes": {str(path.relative_to(ROOT)): file_sha256(path) for path in
                   [ROOT / "v005" / "src" / "hybrid" / "optimizer.py", ROOT / "v005" / "src" / "hybrid" / "model.py",
                    ROOT / "v005" / "src" / "search_config.py", Path(__file__)]}})
    print(json.dumps([{k: r[k] for k in ("seed", "method", "observations", "final_best", "failed_measurements", "suggestion_seconds")}
                     for r in summaries]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("replay", "prepare", "suggest", "ingest", "summarize"))
    parser.add_argument("--output", type=Path, default=RESULTS)
    parser.add_argument("--trial", type=Path)
    parser.add_argument("--request", type=Path)
    args = parser.parse_args()
    if args.command == "replay":
        replay(args.output / "historical_replay.json")
    elif args.command == "prepare":
        prepare_benchmark(args.output)
    elif args.command == "suggest":
        suggest_benchmark(args.trial)
    elif args.command == "ingest":
        ingest_benchmark(args.trial, args.request)
    else:
        summarize_benchmark(args.output)


if __name__ == "__main__":
    main()
