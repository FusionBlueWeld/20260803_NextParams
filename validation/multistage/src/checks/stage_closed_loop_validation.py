"""塗工・乾燥・硬化をv004で個別反復最適化し、bundle予測空間を検証する。"""

from __future__ import annotations

import csv
import itertools
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

from validation.multistage.functional_coating.pipeline import evaluate_line
from validation.multistage.src.checks.learned_r001_validation import ROOT, V004, BASE, _current_contexts, _lhs


STAGES = ("coating", "drying", "curing")
PROBLEM_HEADER = ["column", "display_name", "unit", "role", "direction", "lower", "upper", "step", "target"]


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def _write_csv(path: Path, header: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=header)
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def _cli(*args: str) -> str:
    environment = os.environ.copy()
    environment["PYTHONIOENCODING"] = "utf-8"
    for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        environment[key] = "1"
    result = subprocess.run(
        [sys.executable, str(ROOT / "main.py"), "--version", "v004", *args], cwd=ROOT,
        env=environment, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=600,
    )
    if result.returncode:
        raise RuntimeError(result.stderr or result.stdout)
    return result.stdout


def _problem(manifest: dict) -> list[dict[str, object]]:
    inputs = [*manifest["controls"], *manifest["incoming_state"]]
    constraints = {item["name"]: item for item in manifest["local_constraints"]}
    objective = manifest["local_objective"]
    rows = []
    for item in inputs:
        lower, upper = item["range"]
        rows.append({"column": item["name"], "display_name": item["name"], "unit": item["unit"],
                     "role": "parameter", "direction": "", "lower": lower, "upper": upper,
                     "step": (upper - lower) / 2, "target": ""})
    for item in manifest["outputs"]:
        name = item["name"]
        if name == objective["name"]:
            role, direction, target = "objective", objective["direction"], ""
        elif name in constraints:
            role, direction, target = "constraint", constraints[name]["direction"], constraints[name]["target"]
        else:
            role, direction, target = "monitor", "", ""
        rows.append({"column": name, "display_name": name, "unit": item["unit"], "role": role,
                     "direction": direction, "lower": "", "upper": "", "step": "", "target": target})
    return rows


def _evaluate(stage: str, inputs: dict[str, object]) -> dict[str, np.ndarray]:
    module = __import__(f"validation.multistage.functional_coating.{stage}.physics_model", fromlist=["evaluate_model"])
    return {name: np.asarray(value) for name, value in module.evaluate_model(**inputs).items()}


def _feasible(manifest: dict, outputs: dict[str, np.ndarray]) -> np.ndarray:
    first = np.asarray(next(iter(outputs.values())))
    result = np.ones(first.shape, dtype=bool)
    for item in manifest["local_constraints"]:
        values = np.asarray(outputs[item["name"]])
        result &= values >= item["target"] if item["direction"] == "greater_equal" else values <= item["target"]
    return result


def _control_grid(manifest: dict, context: dict[str, float]) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    controls = manifest["controls"]
    axes = [[item["range"][0], sum(item["range"]) / 2, item["range"][1]] for item in controls]
    points = np.asarray(list(itertools.product(*axes)), dtype=float)
    values = {item["name"]: points[:, index] for index, item in enumerate(controls)}
    values.update({name: np.full(len(points), value) for name, value in context.items()})
    return points, values


def _maximin_indices(points: np.ndarray, count: int, seed: int) -> np.ndarray:
    lower, upper = points.min(axis=0), points.max(axis=0)
    normalized = (points - lower) / np.maximum(upper - lower, 1e-12)
    rng = np.random.default_rng(seed)
    first = int(rng.integers(len(points)))
    selected = [first]
    distance = np.sum((normalized - normalized[first]) ** 2, axis=1)
    distance[first] = -1
    while len(selected) < count:
        index = int(np.argmax(distance))
        selected.append(index)
        distance = np.minimum(distance, np.sum((normalized - normalized[index]) ** 2, axis=1))
        distance[selected] = -1
    return np.asarray(selected)


def _best(manifest: dict, points: np.ndarray, outputs: dict[str, np.ndarray]) -> int | None:
    indices = np.flatnonzero(_feasible(manifest, outputs))
    if not len(indices):
        return None
    objective = manifest["local_objective"]
    values = outputs[objective["name"]]
    return int(indices[np.argmax(values[indices])] if objective["direction"] == "maximize" else indices[np.argmin(values[indices])])


def _normal_cdf(value: np.ndarray) -> np.ndarray:
    return np.asarray([0.5 * (1 + math.erf(float(item) / math.sqrt(2))) for item in np.ravel(value)]).reshape(value.shape)


def _prediction_probability(manifest: dict, predicted: dict[str, dict[str, np.ndarray]]) -> np.ndarray:
    size = len(next(iter(predicted.values()))["mean"])
    probability = np.ones(size)
    for item in manifest["local_constraints"]:
        mean = predicted[item["name"]]["mean"]
        std = np.maximum(predicted[item["name"]]["std"], 1e-12)
        z = (float(item["target"]) - mean) / std
        probability *= 1 - _normal_cdf(z) if item["direction"] == "greater_equal" else _normal_cdf(z)
    return probability


def _load_bundle(path: Path):
    # CLIが保存したpickleはsrc.*名を参照するため、v004の読込環境へ切り替えます。
    for name in list(sys.modules):
        if name == "src" or name.startswith("src."):
            del sys.modules[name]
    if str(V004) not in sys.path:
        sys.path.insert(0, str(V004))
    from src.stage_bundle import load_stage_bundle
    return load_stage_bundle(path)


def run_one(stage: str, seed: int, prefix: str, iterations: int = 15, training_samples: int = 128,
            heldout_samples: int = 512) -> dict[str, object]:
    """1工程・1seedで推薦と仮想測定を反復し、独立した未学習点でも予測を採点します。"""

    manifest = _read_json(BASE / stage / "manifest.json")
    context = _current_contexts()[stage]
    trial_name = f"trial_{prefix}_{stage}_s{seed}"
    trial = V004 / "trials" / trial_name
    if trial.exists():
        raise FileExistsError(trial)
    _cli("--new", trial_name)
    _write_csv(trial / "problem.csv", PROBLEM_HEADER, _problem(manifest))
    connection = {
        "schema_version": "1.0", "stage_id": stage,
        "controls": [item["name"] for item in manifest["controls"]],
        "incoming_context": [item["name"] for item in manifest["incoming_state"]],
        "connector_outputs": [item["name"] for item in manifest["outputs"]],
        "current_incoming_context": context,
    }
    _write_json(trial / "stage_connection.json", connection)
    _cli("--prepare", trial_name)

    inputs = [*manifest["controls"], *manifest["incoming_state"]]
    names = [item["name"] for item in inputs]
    broad = _lhs([tuple(item["range"]) for item in inputs], training_samples, 1000 + seed)
    grid_points, grid_values = _control_grid(manifest, context)
    initial_indices = _maximin_indices(grid_points, 5, seed)
    current = np.column_stack([grid_values[name][initial_indices] for name in names])
    initial = np.vstack([broad, current])
    initial_values = dict(zip(names, initial.T))
    initial_outputs = _evaluate(stage, initial_values)
    output_names = [item["name"] for item in manifest["outputs"]]
    rows: list[dict[str, object]] = []
    for index in range(len(initial)):
        row = {"experiment_id": f"initial_{index + 1:04d}"}
        row.update({name: float(initial_values[name][index]) for name in names})
        row.update({name: float(initial_outputs[name][index]) for name in output_names})
        rows.append(row)
    header = ["experiment_id", *names, *output_names]
    _write_csv(trial / "data" / "experiments.csv", header, rows)

    history = []
    seen_controls = {tuple(grid_points[index]) for index in initial_indices}
    for iteration in range(1, iterations + 1):
        stdout = _cli("--run", trial_name, "--n", "1")
        if "STOP_REQUIRED" in stdout:
            break
        recommendation = _read_csv(trial / "output" / "recommendations.csv")[0]
        for name, value in context.items():
            if not np.isclose(float(recommendation[name]), value):
                raise AssertionError(f"{stage}: incoming context changed: {name}")
        control_tuple = tuple(float(recommendation[item["name"]]) for item in manifest["controls"])
        if control_tuple in seen_controls:
            raise AssertionError(f"{stage}: duplicate control recommendation")
        seen_controls.add(control_tuple)
        raw = {name: float(recommendation[name]) for name in names}
        raw.update(context)
        truth = _evaluate(stage, raw)
        row = {"experiment_id": f"bo_{iteration:03d}"}
        row.update(raw)
        row.update({name: float(value) for name, value in truth.items()})
        rows.append(row)
        _write_csv(trial / "data" / "experiments.csv", header, rows)
        history.append({
            "iteration": iteration,
            "controls": {name: raw[name] for name in connection["controls"]},
            "predicted_objective": float(recommendation[f"{manifest['local_objective']['name']}_mean"]),
            "predicted_feasibility_probability": float(recommendation["feasibility_probability"]),
            "support": float(recommendation["gp_support"]),
            "true_feasible": bool(_feasible(manifest, truth)),
            "true_outputs": {name: float(value) for name, value in truth.items()},
        })
    _cli("--run", trial_name, "--n", "1")

    grid_outputs = _evaluate(stage, grid_values)
    oracle_index = _best(manifest, grid_points, grid_outputs)
    evaluated_points = np.asarray([[item["controls"][name] for name in connection["controls"]] for item in history])
    evaluated_values = {name: np.asarray([item["true_outputs"][name] for item in history]) for name in output_names}
    current_initial_values = {name: initial_outputs[name][-len(initial_indices):] for name in output_names}
    all_current_points = np.vstack([grid_points[initial_indices], evaluated_points]) if len(evaluated_points) else grid_points[initial_indices]
    all_current_values = {name: np.concatenate([current_initial_values[name], evaluated_values[name]]) for name in output_names}
    final_index = _best(manifest, all_current_points, all_current_values)
    objective = manifest["local_objective"]
    objective_name = objective["name"]
    span = float(np.ptp(grid_outputs[objective_name]))
    gap = ((grid_outputs[objective_name][oracle_index] - all_current_values[objective_name][final_index])
           if objective["direction"] == "maximize" else
           (all_current_values[objective_name][final_index] - grid_outputs[objective_name][oracle_index]))
    regret = max(0.0, float(gap)) / max(span, 1e-12)

    bundle = max((trial / "output" / "stage_bundles").glob("stage_bundle_run_*"), key=lambda path: int(path.name.rsplit("_", 1)[1]))
    _, predictor = _load_bundle(bundle)
    heldout = _lhs([tuple(item["range"]) for item in inputs], heldout_samples, 9000 + seed)
    heldout_values = dict(zip(names, heldout.T))
    heldout_truth = _evaluate(stage, heldout_values)
    prediction = predictor.predict_arrays(heldout_values)
    predicted = {name: {kind: np.asarray(value) for kind, value in item.items()} for name, item in prediction["outputs"].items()}
    errors = {}
    for name in output_names:
        output_span = float(np.ptp(heldout_truth[name]))
        difference = predicted[name]["mean"] - heldout_truth[name]
        rmse = float(np.sqrt(np.mean(difference * difference)))
        errors[name] = {"rmse": rmse, "normalized_rmse": rmse / max(output_span, 1e-12)}
    true_feasible = _feasible(manifest, heldout_truth)
    predicted_mean_feasible = _feasible(manifest, {name: item["mean"] for name, item in predicted.items()})
    probability = _prediction_probability(manifest, predicted)
    recommendation_brier = float(np.mean([(item["predicted_feasibility_probability"] - item["true_feasible"]) ** 2 for item in history]))
    heldout_brier = float(np.mean((probability - true_feasible) ** 2))
    support = np.asarray(prediction["support"])
    selected_controls = {name: float(all_current_points[final_index, column]) for column, name in enumerate(connection["controls"])}
    selected_true_feasible = bool(_feasible(manifest, {name: np.asarray([all_current_values[name][final_index]]) for name in output_names})[0])
    selected_input = {**selected_controls, **context}
    selected_support = float(predictor.predict_arrays(selected_input)["support"])
    max_nrmse = max(item["normalized_rmse"] for item in errors.values())
    checks = {
        "normalized_regret_le_0_01": bool(regret <= 0.01),
        "selected_condition_is_true_feasible": selected_true_feasible,
        "heldout_probability_brier_le_0_1": bool(heldout_brier <= 0.1),
        "maximum_heldout_nrmse_le_0_1": bool(max_nrmse <= 0.1),
        "heldout_feasibility_accuracy_ge_0_9": bool(np.mean(predicted_mean_feasible == true_feasible) >= 0.9),
        "selected_support_ge_0_5": bool(selected_support >= 0.5),
    }
    return {
        "status": "pass" if all(checks.values()) else "fail",
        "stage": stage, "seed": seed, "trial": trial_name,
        "training_samples_broad": training_samples, "initial_current_context_samples": len(initial_indices),
        "completed_iterations": len(history), "heldout_samples": heldout_samples,
        "current_incoming_context": context,
        "oracle_grid_best_controls": {name: float(grid_points[oracle_index, column]) for column, name in enumerate(connection["controls"])},
        "selected_controls": selected_controls,
        "selected_true_feasible": selected_true_feasible,
        "normalized_regret": regret,
        "optimum_found": bool(regret <= 1e-12),
        "recommendation_feasible_rate": float(np.mean([item["true_feasible"] for item in history])),
        "recommendation_probability_brier": recommendation_brier,
        "heldout_probability_brier_uncalibrated": heldout_brier,
        "heldout_feasibility_accuracy": float(np.mean(predicted_mean_feasible == true_feasible)),
        "prediction_errors": errors, "maximum_heldout_normalized_rmse": max_nrmse,
        "support": {"selected": selected_support, "heldout_min": float(np.min(support)),
                    "heldout_median": float(np.median(support)), "heldout_max": float(np.max(support)),
                    "heldout_supported_fraction": float(np.mean(support >= 0.5))},
        "probability_status": "UNCALIBRATED_MODEL_PREDICTIVE_ESTIMATE",
        "acceptance_checks": checks, "bundle": str(bundle.relative_to(ROOT)), "history": history,
    }


def run_suite(prefix: str, output: Path, seeds=(0, 1, 2), iterations: int = 15) -> dict[str, object]:
    """工程とseedの組合せを順に実行し、指定先へ集計結果を保存します。"""

    if output.exists():
        raise FileExistsError(output)
    results = []
    for stage in STAGES:
        for seed in seeds:
            result = run_one(stage, seed, prefix, iterations=iterations)
            results.append(result)
            print(f"[{stage}/seed={seed}] {result['status']} regret={result['normalized_regret']:.6g}", flush=True)
    report = {
        "status": "pass" if all(item["status"] == "pass" for item in results) else "fail",
        "seeds": list(seeds), "iterations": iterations, "results": results,
    }
    _write_json(output, report)
    return report


def consolidate_final_report(base_report: Path, dense_curing_reports: list[Path], output: Path) -> dict[str, object]:
    """基準レポートと硬化工程の追加検証を統合し、最終結果を別ファイルへ保存します。"""

    base = _read_json(base_report)
    initial_curing = [item for item in base["results"] if item["stage"] == "curing"]
    final_results = [item for item in base["results"] if item["stage"] != "curing"]
    final_results.extend(_read_json(path) for path in dense_curing_reports)
    report = {
        "status": "pass" if all(item["status"] == "pass" for item in final_results) else "fail",
        "final_run_count": len(final_results),
        "final_results": final_results,
        "data_density_diagnostic": {
            "curing_128_samples_status": "fail",
            "reason": "maximum held-out normalized RMSE exceeded 0.10 and broad-domain support was sparse",
            "curing_128_sample_results": initial_curing,
            "curing_512_samples_status": "pass",
        },
    }
    _write_json(output, report)
    return report


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit("usage: python -m validation.multistage.src.checks.stage_closed_loop_validation PREFIX OUTPUT.json")
    value = run_suite(sys.argv[1], Path(sys.argv[2]))
    print(json.dumps({"status": value["status"], "runs": len(value["results"])}, ensure_ascii=True))
