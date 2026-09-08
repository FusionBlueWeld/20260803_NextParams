"""Closed-loop multi-process benchmarks through the real optimizer CLI."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .simulators import ROOT, PROBLEM_HEADER, Simulator, measure, read_csv, write_csv, write_json

PROJECT_ROOT = ROOT.parent


def versions() -> list[str]:
    return sorted(p.name for p in PROJECT_ROOT.glob("v[0-9][0-9][0-9]") if (p / "src" / "cli.py").is_file())


def cli(version: str, *args: str) -> str:
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    # Small matrices run more consistently without BLAS oversubscription.
    for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        env[key] = "1"
    result = subprocess.run([sys.executable, str(PROJECT_ROOT / "main.py"), "--version", version, *args],
                            cwd=PROJECT_ROOT, env=env, capture_output=True, text=True, encoding="utf-8",
                            errors="replace", timeout=600)
    if result.returncode:
        raise RuntimeError(f"{version} {' '.join(args)}\n{result.stderr or result.stdout}")
    return result.stdout


def source_hashes(folder: Path) -> dict[str, str]:
    return {str(p.relative_to(folder)).replace("\\", "/"): hashlib.sha256(
        p.read_bytes().replace(b"\r\n", b"\n")).hexdigest() for p in sorted(folder.rglob("*.py"))}


def regression_metrics(prediction, truth, span: float) -> dict:
    error = np.asarray(prediction, dtype=float) - np.asarray(truth, dtype=float)
    if not np.all(np.isfinite(error)):
        raise ValueError("Nonfinite prediction error")
    if not error.size:
        return {"count": 0, "mae": None, "rmse": None, "normalized_rmse": None}
    rmse = float(np.sqrt(np.mean(error**2)))
    return {"count": int(error.size), "mae": float(np.mean(np.abs(error))), "rmse": rmse,
            "normalized_rmse": rmse / span if span > 0 else None}


def regret(sim: Simulator, value: float | None, optimum: float, span: float) -> float | None:
    if value is None:
        return None
    gap = optimum - value if sim.objective.direction == "maximize" else value - optimum
    return max(0.0, gap) / span if span > 0 else 0.0


def summarize_best(sim: Simulator, points, outputs) -> dict | None:
    idx = sim.best_index(outputs)
    if idx is None:
        return None
    return {"parameters": {n: float(points[idx, j]) for j, n in enumerate(sim.parameter_columns)},
            "outputs": {n: float(outputs[n][idx]) for n in sim.output_columns}}


def audit(sim: Simulator) -> dict:
    grid = sim.grid()
    values = sim.evaluate_points(grid)
    initial = sim.initial_design()
    initial_values = sim.evaluate_points(initial)
    best = summarize_best(sim, grid, values)
    return {"simulator": sim.id, "name": sim.manifest["name"], "provenance": sim.provenance(),
            "parameter_count": len(sim.parameters), "initial_design": sim.initial_design_method(),
            "candidate_count": len(grid), "feasible_count": int(sim.feasible(values).sum()),
            "initial_count": len(initial), "initial_feasible_count": int(sim.feasible(initial_values).sum()),
            "objective": sim.objective.column, "direction": sim.objective.direction,
            "grid_best": best, "initial_best": summarize_best(sim, initial, initial_values),
            "output_ranges": {n: {"min": float(np.min(values[n])), "max": float(np.max(values[n]))}
                              for n in sim.output_columns}}


def response_metrics(sim: Simulator, trial: Path, seen_points: np.ndarray, spans: dict) -> dict:
    # v000 has no full response-space export. Never substitute recommendation-only errors.
    files = sorted((trial / "output").rglob("*response_space_run_*.csv"))
    if not files:
        return {"status": "not_available", "reason": "This version did not export a response space"}
    path = max(files, key=lambda p: int(re.search(r"response_space_run_(\d+)\.csv$", p.name).group(1)))
    rows = read_csv(path)
    if not rows:
        raise ValueError("Empty response space")
    if "data_rows" in rows[0] and any(int(row["data_rows"]) != len(seen_points) for row in rows):
        return {"status": "not_available", "reason": "Response space predates the final observations"}
    points = np.array([[float(row[n]) for n in sim.parameter_columns] for row in rows])
    truth = sim.evaluate_points(points)
    seen = {tuple(np.round(p, 8)) for p in seen_points}
    unobserved = np.array([tuple(np.round(p, 8)) not in seen for p in points])
    metrics = {}
    selected_prediction = {}
    for n in sim.output_columns:
        keys = [f"{n}_hybrid_mean", f"{n}_mean", f"{n}_nn_pred"]
        key = next((key for key in keys if key in rows[0]), None)
        if key is None:
            raise ValueError(f"Missing prediction for {n} in {path.name}")
        predicted = np.array([float(row[key]) for row in rows])
        selected_prediction[n] = predicted
        metrics[n] = {"prediction_column": key,
                      "all_grid": regression_metrics(predicted, truth[n], spans[n]),
                      "unobserved": regression_metrics(predicted[unobserved], truth[n][unobserved], spans[n])}
    # Check the export actually covers the declared candidate grid, not a partial file.
    keys = {tuple(np.round(p, 8)) for p in points}
    expected = {tuple(np.round(p, 8)) for p in sim.grid()}
    if len(keys) != len(points) or keys != expected:
        raise ValueError("Response-space export is not the complete unique candidate grid")
    return {"status": "ok", "file": str(path.relative_to(PROJECT_ROOT)), "row_count": len(rows),
            "unobserved_count": int(unobserved.sum()), "outputs": metrics,
            "feasibility_classification_accuracy": float(np.mean(sim.feasible(selected_prediction) == sim.feasible(truth)))}


def validate_recommendations(sim: Simulator, points: np.ndarray, seen: set) -> None:
    sim.evaluate_points(points)  # Input domain and model contract.
    batch = set()
    for point in points:
        key = tuple(np.round(point, 8))
        if key in batch or key in seen:
            raise ValueError("Optimizer recommended an already observed or duplicate condition")
        for value, axis in zip(point, sim.axes):
            if not np.any(np.isclose(value, axis, rtol=1e-9, atol=1e-9)):
                raise ValueError("Optimizer recommended an off-grid condition")
        batch.add(key)


def run_optimizer(version: str, trial_name: str, trial: Path, recommendations: int) -> dict | None:
    cli(version, "--run", trial_name, "--n", str(recommendations))
    status_path = trial / "output" / "stopping_status.json"
    if status_path.is_file():
        status = json.loads(status_path.read_text(encoding="utf-8"))
        # STOP_RECOMMENDED is advisory: this benchmark deliberately uses a fixed budget.
        # STOP_REQUIRED can return success without updating recommendations.csv.
        if status.get("status") == "STOP_REQUIRED":
            return status
    return None


def run(sim: Simulator, version: str, trial_name: str, result_dir: Path,
        iterations: int = 15, recommendations: int = 1, seed: int = 0, noise: float = 0.0) -> dict:
    existed = result_dir.exists()
    try:
        return _run(sim, version, trial_name, result_dir, iterations, recommendations, seed, noise)
    except Exception as error:
        # Include setup failures, but never touch an output directory predating this call.
        if not existed and result_dir.is_dir():
            try:
                path = result_dir / "run.json"
                metadata = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
                write_json(path, {**metadata, "status": "failed", "error": str(error)})
            except OSError:
                pass  # Do not mask the original error if failure reporting is also unwritable.
        raise


def _run(sim: Simulator, version: str, trial_name: str, result_dir: Path,
        iterations: int = 15, recommendations: int = 1, seed: int = 0, noise: float = 0.0) -> dict:
    if version not in versions():
        raise ValueError(f"Unknown optimizer version: {version}")
    if not re.fullmatch(r"trial_[A-Za-z0-9_-]+", trial_name):
        raise ValueError("Trial name must start with trial_ and contain only letters, digits, _ or -")
    if iterations < 1 or recommendations < 1 or seed < 0 or not math.isfinite(noise) or noise < 0:
        raise ValueError("Invalid iterations, recommendations, seed or noise")
    trial = PROJECT_ROOT / version / "trials" / trial_name
    if trial.exists() or result_dir.exists():
        raise FileExistsError(f"Refusing to overwrite trial or result: {trial}, {result_dir}")
    started = time.monotonic()
    grid = sim.grid()
    grid_truth = sim.evaluate_points(grid)
    best = summarize_best(sim, grid, grid_truth)
    if best is None:
        raise ValueError("No feasible point on the declared oracle grid")
    spans = {n: float(np.ptp(grid_truth[n])) for n in sim.output_columns}
    objective = sim.objective.column
    optimum = best["outputs"][objective]
    points = sim.initial_design(seed)
    # One final --run is performed after the last observations, so leave candidates for it.
    if len(points) + iterations * recommendations + recommendations > len(grid):
        raise ValueError("Experiment budget leaves too few candidates for final prediction")
    truth = {n: v.copy() for n, v in sim.evaluate_points(points).items() if n in spans}
    rng = np.random.default_rng(seed)
    observed = measure(truth, spans, noise, rng)
    rows = sim.experiment_rows(points, observed, "initial")
    truth_rows = sim.experiment_rows(points, truth, "initial")
    initial_best = summarize_best(sim, points, truth)
    initial_value = initial_best["outputs"][objective] if initial_best else None
    initial_normalized_regret = regret(sim, initial_value, optimum, spans[objective])
    result_dir.mkdir(parents=True, exist_ok=False)
    metadata = {"simulator": sim.id, "optimizer": version, "trial": trial_name,
                "seed": seed, "noise_fraction": noise, "noise_sigma": {n: noise*s for n, s in spans.items()},
                "iterations": iterations, "recommendations": recommendations,
                "started_at": datetime.now(timezone.utc).isoformat(),
                "simulator_provenance": sim.provenance(), "numpy_version": np.__version__,
                "optimizer_sha256_lf": source_hashes(PROJECT_ROOT / version / "src"),
                "runner_sha256_lf": {p.name: hashlib.sha256(p.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
                                     for p in [ROOT / "simulators.py", ROOT / "benchmark.py", ROOT / "run.py", PROJECT_ROOT / "main.py"]},
                "python_version": sys.version, "initial_design": sim.initial_design_method(seed)}
    write_json(result_dir / "run.json", {**metadata, "status": "running"})
    write_csv(result_dir / "problem.csv", PROBLEM_HEADER, sim.rows)
    write_csv(result_dir / "observations.csv", sim.experiment_header, rows)
    write_csv(result_dir / "truth.csv", sim.experiment_header, truth_rows)
    history = []
    stop_status = None
    seen = {tuple(np.round(p, 8)) for p in points}
    try:
        cli(version, "--new", trial_name)
        write_csv(trial / "problem.csv", PROBLEM_HEADER, sim.rows)
        cli(version, "--prepare", trial_name)
        write_csv(trial / "data" / "experiments.csv", sim.experiment_header, rows)
        write_csv(result_dir / "truth.csv", sim.experiment_header, truth_rows)
        for iteration in range(1, iterations + 1):
            stop_status = run_optimizer(version, trial_name, trial, recommendations)
            if stop_status is not None:
                break
            recommended = sorted(read_csv(trial / "output" / "recommendations.csv"), key=lambda row: int(row["rank"]))
            if len(recommended) != recommendations:
                raise ValueError(f"Expected {recommendations} recommendations, got {len(recommended)}")
            batch = np.array([[float(row[n]) for n in sim.parameter_columns] for row in recommended])
            validate_recommendations(sim, batch, seen)
            batch_truth = sim.evaluate_points(batch)
            measurements = measure(batch_truth, spans, noise, rng)
            for slot, (row, point) in enumerate(zip(recommended, batch)):
                predicted = {n: float(row[f"{n}_mean"]) for n in sim.output_columns}
                if not all(math.isfinite(v) for v in predicted.values()):
                    raise ValueError("Nonfinite recommendation prediction")
                probability = float(row["feasibility_probability"])
                if not math.isfinite(probability) or not 0 <= probability <= 1:
                    raise ValueError("Invalid feasibility probability")
                history.append({"iteration": iteration, "slot": slot + 1,
                                "parameters": dict(zip(sim.parameter_columns, map(float, point))),
                                "predicted": predicted,
                                "truth": {n: float(batch_truth[n][slot]) for n in sim.output_columns},
                                "observed": {n: float(measurements[n][slot]) for n in sim.output_columns},
                                "true_feasible": bool(sim.feasible(batch_truth)[slot]),
                                "observed_feasible": bool(sim.feasible(measurements)[slot]),
                                "predicted_feasibility_probability": probability})
            points = np.vstack([points, batch])
            truth = {n: np.concatenate([truth[n], batch_truth[n]]) for n in sim.output_columns}
            observed = {n: np.concatenate([observed[n], measurements[n]]) for n in sim.output_columns}
            seen.update(tuple(np.round(p, 8)) for p in batch)
            rows.extend(sim.experiment_rows(batch, measurements, f"bo_{iteration:03d}"))
            truth_rows.extend(sim.experiment_rows(batch, batch_truth, f"bo_{iteration:03d}"))
            write_csv(trial / "data" / "experiments.csv", sim.experiment_header, rows)
            write_csv(result_dir / "observations.csv", sim.experiment_header, rows)
            write_csv(result_dir / "truth.csv", sim.experiment_header, truth_rows)
            current_best = summarize_best(sim, points, truth)
            value = current_best["outputs"][objective] if current_best else None
            for item in history[-recommendations:]:
                item["best_true_feasible_so_far"] = value
                item["normalized_regret"] = regret(sim, value, optimum, spans[objective])
            write_json(result_dir / "history.json", history)
            print(f"[{sim.id}/{version}/seed={seed}] {iteration}/{iterations}: best={value}", flush=True)
        if stop_status is None:
            stop_status = run_optimizer(version, trial_name, trial, recommendations)
        write_json(result_dir / "history.json", history)
        final_best = summarize_best(sim, points, truth)
        # What would the user choose using measurements, and how good is it in truth?
        chosen_idx = sim.best_index(observed)
        chosen = None if chosen_idx is None else {
            "parameters": {n: float(points[chosen_idx, j]) for j, n in enumerate(sim.parameter_columns)},
            "observed_outputs": {n: float(observed[n][chosen_idx]) for n in sim.output_columns},
            "true_outputs": {n: float(truth[n][chosen_idx]) for n in sim.output_columns},
            "true_feasible": bool(sim.feasible(truth)[chosen_idx])}
        value = final_best["outputs"][objective] if final_best else None
        normalized = regret(sim, value, optimum, spans[objective])
        regret_reduction = (initial_normalized_regret - normalized
                            if initial_normalized_regret is not None and normalized is not None else None)
        errors = {n: regression_metrics([h["predicted"][n] for h in history],
                                       [h["truth"][n] for h in history], spans[n]) for n in sim.output_columns}
        summary = {**metadata, "status": "complete", "elapsed_seconds": time.monotonic() - started,
                   "completed_iterations": len(history) // recommendations,
                   "termination_reason": "optimizer_stop_required" if stop_status else "budget_completed",
                   "optimizer_stop_status": stop_status,
                   "objective": objective, "direction": sim.objective.direction,
                   "parameter_count": len(sim.parameters),
                   "candidate_count": len(grid), "grid_feasible_count": int(sim.feasible(grid_truth).sum()),
                   "initial_count": len(sim.initial_design(seed)), "final_count": len(points),
                   "grid_best": best, "initial_best": initial_best, "final_best": final_best,
                   "observed_selected_condition": chosen, "normalized_regret": normalized,
                   "initial_normalized_regret": initial_normalized_regret,
                   "normalized_regret_reduction": regret_reduction,
                   "optimum_found": value is not None and bool(np.isclose(value, optimum, rtol=1e-8, atol=1e-10)),
                   "true_feasible_recommendation_rate": float(np.mean([h["true_feasible"] for h in history])) if history else None,
                   "feasibility_brier_score": float(np.mean([(h["predicted_feasibility_probability"] - h["true_feasible"])**2 for h in history])) if history else None,
                   "recommendation_prediction_metrics": errors,
                   "response_space_metrics": response_metrics(sim, trial, points, spans)}
        write_json(result_dir / "summary.json", summary)
        write_json(result_dir / "run.json", {**metadata, "status": "complete"})
        return summary
    except Exception as error:
        write_json(result_dir / "run.json", {**metadata, "status": "failed", "error": str(error)})
        raise


def suite_report(directory: Path, results: list[dict]) -> None:
    columns = ["simulator", "optimizer", "seed", "noise_fraction", "status", "parameter_count",
               "initial_normalized_regret", "normalized_regret", "normalized_regret_reduction",
               "true_feasible_recommendation_rate", "feasibility_brier_score", "optimum_found", "elapsed_seconds", "error"]
    write_csv(directory / "comparison.csv", columns, [{n: r.get(n, "") for n in columns} for r in results])
    write_json(directory / "comparison.json", results)
    lines = ["# 多工程ベンチマーク", "", "同一工程・seed・ノイズ条件でバージョンを比較します。",
             "regretは候補グリッド上の最良値との差を目的値の全グリッド幅で正規化した値です。小さいほど良好。",
             "解空間の誤差はsummary.jsonのresponse_space_metricsを参照してください。v000は解空間出力なし。", "",
             "|工程|入力数|version|seed|状態|初期regret|最終regret|改善量|制約適合率|",
             "|---|---:|---|---:|---|---:|---:|---:|---:|"]
    def fmt(value):
        return "—" if value is None else f"{value:.4f}"
    for r in results:
        lines.append(f"|{r['simulator']}|{r.get('parameter_count', '—')}|{r['optimizer']}|{r['seed']}|{r['status']}|"
                     f"{fmt(r.get('initial_normalized_regret'))}|{fmt(r.get('normalized_regret'))}|"
                     f"{fmt(r.get('normalized_regret_reduction'))}|{fmt(r.get('true_feasible_recommendation_rate'))}|")
    (directory / "comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
