"""In-process v003 GP-residual ablation on the deterministic physics simulators.

This module intentionally lives under ``validation/single``.  It imports the
v003 training and acquisition code, but does not create a v003 trial or alter
the production workflow.  The no-GP arm is an adapter around the *same* model
prediction: only ``hybrid_mean`` is replaced by ``nn_pred``; GP standard
deviation, support, NN and the initial design remain unchanged.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from ..simulators import Simulator, load

# v003 is a normal importable package (the directory name is a valid module
# name), so the validation runner can call the production building blocks
# directly without subprocesses or temporary trials.
from v003.src.hybrid.model import HybridModel, HybridPrediction, train_hybrid_model
from v003.src.hybrid.optimizer import run_optimization
from v003.src.knowledge import KnowledgeRule
from v003.src.preprocessing import prepare_experiments, normalize_parameters
from v003.src.settings import NN_SEED, ProblemDefinition, VariableDefinition
from v003.src.validation import ExperimentData


DEFAULT_SIMULATORS = ("thermal_curing", "press_forming", "convection_drying")
# Three default seeds keep the reference run practical; callers can pass five
# or more seeds for a higher-powered reproducibility analysis.
DEFAULT_SEEDS = (0, 1, 2)
DEFAULT_LOCAL_RADIUS = 0.20
STD_FLOOR_FRACTION = 1e-6


class NoGpMeanAdapter:
    """Expose a HybridModel-like object with the GP mean correction removed."""

    def __init__(self, base: HybridModel):
        self.base = base
        # run_optimization uses this scale when there is no feasible observation.
        self.gp_models = base.gp_models

    def predict(self, normalized_x: np.ndarray):
        prediction = self.base.predict(normalized_x)
        results = {}
        for column, values in prediction.results.items():
            copied = dict(values)
            copied["hybrid_mean"] = np.asarray(values["nn_pred"], dtype=float).copy()
            # Keep GP std and support exactly as the trained model supplied them.
            results[column] = copied
        return HybridPrediction(support=prediction.support, results=results)


@dataclass(frozen=True)
class MetricSet:
    outputs: dict[str, dict[str, float]]
    macro_nrmse: float
    macro_nlpd: float
    macro_coverage95: float
    feasible_accuracy: float


def make_problem(sim: Simulator) -> ProblemDefinition:
    """Translate a validation simulator definition to the v003 data contract."""

    def convert(variable) -> VariableDefinition:
        return VariableDefinition(
            column=variable.column,
            display_name=variable.display_name,
            unit=variable.unit,
            role=variable.role,
            direction=variable.direction,
            lower=variable.lower,
            upper=variable.upper,
            step=variable.step,
            target=variable.target,
        )

    parameters = [convert(item) for item in sim.parameters]
    outputs = [convert(item) for item in sim.outputs]
    return ProblemDefinition(
        parameters=parameters,
        objective=convert(sim.objective),
        constraints=[convert(item) for item in sim.constraints],
        monitors=[item for item in outputs if item.role == "monitor"],
        result_order=outputs,
        candidate_count=len(sim.grid()),
    )


def knowledge_rules(problem: ProblemDefinition) -> list[KnowledgeRule]:
    """Use one minimal, explicit, physically obvious common prior.

    The objective is the only output constrained here.  Applying a separate
    lower-bound rule to every monitor/constraint materially changes the NN
    loss and confounds the GP-residual ablation; the physical simulator's own
    feasibility constraints remain active in acquisition and scoring.
    """

    return [
        KnowledgeRule(
            rule_id=f"physical_nonnegative_{problem.objective.column}",
            type="lower_bound",
            target=problem.objective.column,
            value=0.0,
            strength=2,
            enabled=True,
            note="validation common rule: the physical objective is non-negative",
        )
    ]


def experiment_data(sim: Simulator, points: np.ndarray, values: Mapping[str, np.ndarray]) -> ExperimentData:
    return ExperimentData(
        experiment_ids=[f"ablation_{i:05d}" for i in range(len(points))],
        parameter_rows=np.asarray(points, dtype=float).tolist(),
        result_rows={name: np.asarray(values[name], dtype=float).tolist() for name in sim.output_columns},
    )


def _safe_span(truth: np.ndarray) -> float:
    return max(float(np.ptp(truth)), 1e-12)


def contains_point(points: np.ndarray, target: np.ndarray) -> bool:
    """Return whether an observed design contains the exact oracle grid point."""

    return bool(np.any(np.all(np.isclose(np.asarray(points), np.asarray(target)), axis=1)))


def _metric_for_mask(
    predicted: Mapping[str, np.ndarray],
    truth: Mapping[str, np.ndarray],
    std: Mapping[str, np.ndarray],
    feasible_truth: np.ndarray,
    feasible_prediction: np.ndarray,
    mask: np.ndarray,
) -> MetricSet:
    outputs: dict[str, dict[str, float]] = {}
    nrmse_values: list[float] = []
    nlpd_values: list[float] = []
    coverage_values: list[float] = []
    for name, actual_raw in truth.items():
        actual = np.asarray(actual_raw, dtype=float)[mask]
        estimate = np.asarray(predicted[name], dtype=float)[mask]
        uncertainty = np.maximum(np.asarray(std[name], dtype=float)[mask], _safe_span(np.asarray(truth[name], dtype=float)) * STD_FLOOR_FRACTION)
        span = _safe_span(np.asarray(truth[name], dtype=float))
        error = estimate - actual
        mae = float(np.mean(np.abs(error)))
        rmse = float(np.sqrt(np.mean(np.square(error))))
        nrmse = rmse / span
        normalized_error = error / span
        normalized_std = uncertainty / span
        nlpd = 0.5 * (np.log(2.0 * math.pi * np.square(normalized_std)) + np.square(normalized_error / normalized_std))
        coverage = np.abs(error) <= 1.96 * uncertainty
        outputs[name] = {
            "mae": mae,
            "rmse": rmse,
            "nrmse": float(nrmse),
            "nlpd": float(np.mean(nlpd)),
            "coverage95": float(np.mean(coverage)),
        }
        nrmse_values.append(float(nrmse))
        nlpd_values.append(float(np.mean(nlpd)))
        coverage_values.append(float(np.mean(coverage)))
    return MetricSet(
        outputs=outputs,
        macro_nrmse=float(np.mean(nrmse_values)) if nrmse_values else float("nan"),
        macro_nlpd=float(np.mean(nlpd_values)) if nlpd_values else float("nan"),
        macro_coverage95=float(np.mean(coverage_values)) if coverage_values else float("nan"),
        feasible_accuracy=float(np.mean(np.asarray(feasible_truth)[mask] == np.asarray(feasible_prediction)[mask])),
    )


def calculate_metrics(
    predicted: Mapping[str, np.ndarray],
    truth: Mapping[str, np.ndarray],
    std: Mapping[str, np.ndarray],
    feasible_truth: np.ndarray,
    feasible_prediction: np.ndarray,
    normalized_grid: np.ndarray,
    optimum_normalized: np.ndarray,
    local_radius: float = DEFAULT_LOCAL_RADIUS,
) -> dict[str, Any]:
    """Calculate global/local accuracy, likelihood and feasibility metrics."""

    distances = np.linalg.norm(normalized_grid - optimum_normalized[None, :], axis=1)
    local_mask = distances <= local_radius + 1e-12
    if not np.any(local_mask):
        local_mask[np.argmin(distances)] = True
    global_mask = np.ones(len(normalized_grid), dtype=bool)
    global_metrics = _metric_for_mask(predicted, truth, std, feasible_truth, feasible_prediction, global_mask)
    local_metrics = _metric_for_mask(predicted, truth, std, feasible_truth, feasible_prediction, local_mask)
    return {
        "local_radius": local_radius,
        "local_count": int(local_mask.sum()),
        "global": {"outputs": global_metrics.outputs, "macro_nrmse": global_metrics.macro_nrmse, "macro_nlpd": global_metrics.macro_nlpd, "macro_coverage95": global_metrics.macro_coverage95, "feasible_accuracy": global_metrics.feasible_accuracy},
        "local": {"outputs": local_metrics.outputs, "macro_nrmse": local_metrics.macro_nrmse, "macro_nlpd": local_metrics.macro_nlpd, "macro_coverage95": local_metrics.macro_coverage95, "feasible_accuracy": local_metrics.feasible_accuracy},
    }


def _prediction_arrays(model: Any, grid: np.ndarray, problem: ProblemDefinition) -> tuple[dict, dict, dict]:
    prediction = model.predict(normalize_parameters(grid, problem))
    means = {item.column: prediction.results[item.column]["hybrid_mean"] for item in problem.result_variables}
    std = {item.column: np.maximum(prediction.results[item.column]["hybrid_std"], 0.0) for item in problem.result_variables}
    return prediction, means, std


def run_arm(
    sim: Simulator,
    arm: str,
    seed: int,
    iterations: int,
    local_radius: float,
) -> dict[str, Any]:
    problem = make_problem(sim)
    grid = sim.grid()
    normalized_grid = normalize_parameters(grid, problem)
    truth = sim.evaluate_points(grid)
    true_feasible = sim.feasible(truth)
    optimum_index = sim.best_index(truth)
    if optimum_index is None:
        raise ValueError(f"{sim.id} has no feasible grid optimum")
    optimum_point = grid[optimum_index]
    optimum_normalized = normalized_grid[optimum_index]
    initial = sim.initial_design(seed)
    observed = sim.evaluate_points(initial)
    points = initial.copy()
    history: list[dict[str, Any]] = []
    arrival_step: int | None = 0 if contains_point(initial, optimum_point) else None
    checkpoint_step: int | None = None
    checkpoint_metrics: dict[str, Any] | None = None

    def checkpoint(step: int) -> None:
        nonlocal checkpoint_step, checkpoint_metrics
        data = experiment_data(sim, points, observed)
        prepared = prepare_experiments(data, problem)
        training = train_hybrid_model(prepared, problem, knowledge_rules(problem))
        model = training.model if arm == "gp" else NoGpMeanAdapter(training.model)
        prediction, means, std = _prediction_arrays(model, grid, problem)
        predicted_feasible = problem_feasible(means, problem)
        checkpoint_step = step
        checkpoint_metrics = calculate_metrics(
            means, truth, std, true_feasible, predicted_feasible,
            normalized_grid, optimum_normalized, local_radius,
        )
        checkpoint_metrics["training"] = {
            "nn_epochs": training.nn_epochs,
            "nn_normalized_mse": training.nn_normalized_mse,
            "nn_knowledge_loss": training.nn_knowledge_loss,
        }
        checkpoint_metrics["gp_mean_enabled"] = arm == "gp"
        checkpoint_metrics["gp_std_definition"] = "v003 HybridModel gp_std/hybrid_std (shared with adapter)"

    if arrival_step == 0:
        # The initial design itself may contain the oracle optimum; its
        # checkpoint is the post-initial-design retraining, not a later budget
        # snapshot.
        checkpoint(0)

    for step in range(1, iterations + 1) if arrival_step is None else ():
        data = experiment_data(sim, points, observed)
        prepared = prepare_experiments(data, problem)
        training = train_hybrid_model(prepared, problem, knowledge_rules(problem))
        model = training.model if arm == "gp" else NoGpMeanAdapter(training.model)
        result = run_optimization(data, prepared, problem, recommendation_count=1, model=model)
        recommendation = result.recommendations[0]
        next_point = np.array([[float(recommendation[item.column]) for item in problem.parameters]], dtype=float)
        next_truth = sim.evaluate_points(next_point)
        points = np.vstack([points, next_point])
        observed = {name: np.concatenate([observed[name], next_truth[name]]) for name in sim.output_columns}
        reached = bool(np.all(np.isclose(next_point[0], optimum_point)))
        if arrival_step is None and reached:
            arrival_step = step
        history.append({"step": step, "point": next_point[0].tolist(), "is_optimum": reached, "recommendation_score": float(recommendation["recommendation_score"])})
        if checkpoint_step is None and arrival_step == step:
            checkpoint(step)
        print(f"[{sim.id}/{arm}/seed={seed}] {step}/{iterations}", flush=True)
        if reached:
            break

    if checkpoint_step is None:
        checkpoint(iterations)
    return {
        "simulator": sim.id,
        "arm": arm,
        "seed": seed,
        "iterations": iterations,
        "initial_count": len(initial),
        "arrival_step": arrival_step,
        "found": arrival_step is not None,
        "censored": arrival_step is None,
        "checkpoint_step": checkpoint_step,
        "optimum_index": int(optimum_index),
        "optimum_point": optimum_point.tolist(),
        "history": history,
        "metrics": checkpoint_metrics,
        "knowledge_constraints": {"type": "lower_bound", "target": "objective only (physical non-negative)", "value": 0.0, "strength": 2, "enabled": True},
    }


def problem_feasible(values: Mapping[str, np.ndarray], problem: ProblemDefinition) -> np.ndarray:
    mask = np.ones(len(next(iter(values.values()))), dtype=bool)
    for constraint in problem.constraints:
        candidate = np.asarray(values[constraint.column])
        if constraint.direction == "greater_equal":
            mask &= candidate >= float(constraint.target)
        else:
            mask &= candidate <= float(constraint.target)
    return mask


def aggregate(results: list[dict[str, Any]], iterations: int) -> dict[str, Any]:
    paired: list[dict[str, Any]] = []
    for simulator in sorted({r["simulator"] for r in results}):
        for seed in sorted({r["seed"] for r in results if r["simulator"] == simulator}):
            arms = {(r["arm"]): r for r in results if r["simulator"] == simulator and r["seed"] == seed}
            if set(arms) != {"gp", "no_gp"}:
                continue
            gp, no_gp = arms["gp"], arms["no_gp"]
            gp_arrival = gp["arrival_step"] if gp["arrival_step"] is not None else iterations + 1
            no_arrival = no_gp["arrival_step"] if no_gp["arrival_step"] is not None else iterations + 1
            row = {"simulator": simulator, "seed": seed, "gp_found": gp["found"], "no_gp_found": no_gp["found"], "gp_arrival": gp_arrival, "no_gp_arrival": no_arrival, "arrival_delta_gp_minus_no_gp": gp_arrival - no_arrival}
            for scope in ("global", "local"):
                for metric, lower_is_better in (("macro_nrmse", True), ("macro_nlpd", True), ("macro_coverage95", False), ("feasible_accuracy", False)):
                    gp_value = float(gp["metrics"][scope][metric])
                    no_value = float(no_gp["metrics"][scope][metric])
                    delta = gp_value - no_value
                    row[f"{scope}_{metric}_delta_gp_minus_no_gp"] = delta
                    row[f"metric_winner_{scope}_{metric}"] = (
                        "gp" if (delta < -1e-12 if lower_is_better else delta > 1e-12)
                        else "no_gp" if (delta > 1e-12 if lower_is_better else delta < -1e-12)
                        else "tie"
                    )
            paired.append(row)
    summary: dict[str, Any] = {"iterations": iterations, "paired": paired, "by_simulator": {}}
    for simulator in sorted({row["simulator"] for row in paired}):
        rows = [row for row in paired if row["simulator"] == simulator]
        win_loss = {}
        for scope in ("global", "local"):
            for metric in ("macro_nrmse", "macro_nlpd", "macro_coverage95", "feasible_accuracy"):
                key = f"metric_winner_{scope}_{metric}"
                win_loss[f"{scope}_{metric}"] = {name: sum(row[key] == name for row in rows) for name in ("gp", "no_gp", "tie")}
        summary["by_simulator"][simulator] = {"pairs": len(rows), "gp_found_rate": float(np.mean([row["gp_found"] for row in rows])), "no_gp_found_rate": float(np.mean([row["no_gp_found"] for row in rows])), "median_gp_arrival": float(np.median([row["gp_arrival"] for row in rows])), "median_no_gp_arrival": float(np.median([row["no_gp_arrival"] for row in rows])), "win_tie_loss": win_loss, "seed_consistency_gp_found": len({row["gp_found"] for row in rows}) == 1, "seed_consistency_no_gp_found": len({row["no_gp_found"] for row in rows}) == 1}
    all_rows = paired
    if all_rows:
        summary["overall"] = {"pairs": len(all_rows), "gp_found_rate": float(np.mean([r["gp_found"] for r in all_rows])), "no_gp_found_rate": float(np.mean([r["no_gp_found"] for r in all_rows])), "win_tie_loss": {f"{scope}_{metric}": {name: sum(r[f"metric_winner_{scope}_{metric}"] == name for r in all_rows) for name in ("gp", "no_gp", "tie")} for scope in ("global", "local") for metric in ("macro_nrmse", "macro_nlpd", "macro_coverage95", "feasible_accuracy")}}
    return summary


def _write_csv(path: Path, results: list[dict[str, Any]]) -> None:
    rows: list[dict[str, Any]] = []
    for result in results:
        metrics = result["metrics"]
        rows.append({
            "simulator": result["simulator"], "arm": result["arm"], "seed": result["seed"],
            "found": result["found"], "censored": result["censored"],
            "arrival_step": result["arrival_step"], "checkpoint_step": result["checkpoint_step"],
            "global_macro_nrmse": metrics["global"]["macro_nrmse"],
            "global_macro_nlpd": metrics["global"]["macro_nlpd"],
            "global_macro_coverage95": metrics["global"]["macro_coverage95"],
            "global_feasible_accuracy": metrics["global"]["feasible_accuracy"],
            "local_macro_nrmse": metrics["local"]["macro_nrmse"],
            "local_macro_nlpd": metrics["local"]["macro_nlpd"],
            "local_macro_coverage95": metrics["local"]["macro_coverage95"],
            "local_feasible_accuracy": metrics["local"]["feasible_accuracy"],
        })
    fields = list(rows[0]) if rows else ["simulator", "arm", "seed"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _write_trajectories(path: Path, results: list[dict[str, Any]]) -> None:
    fields = ["simulator", "arm", "seed", "step", "point", "is_optimum", "recommendation_score"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for result in results:
            for step in result["history"]:
                writer.writerow({"simulator": result["simulator"], "arm": result["arm"], "seed": result["seed"], **step})


def _write_report(path: Path, payload: dict[str, Any]) -> None:
    summary = payload["summary"]
    lines = ["# v003 GP残差アブレーション", "", "GPありは v003 の `hybrid_mean = nn_pred + support * gp_mean`、GPなしは同じ予測の `hybrid_mean = nn_pred` とした adapter です。GP std、support、NN、初期設計、ノイズなし物理真値は共通です。", "", "知識制約: 物理的に明白な objective の lower_bound=0 を1ルールだけ、strength=2、enabled=true で両armに共通適用（最小限のactive knowledge prior）。物理制約のfeasibility判定は各simulatorの問題定義を共通使用。", "", "## paired summary", "", "未到達のarrivalは `iterations + 1` として中央値を計算し、個別結果では `censored=true` と明記しています。", ""]
    for simulator, row in summary.get("by_simulator", {}).items():
        paired = [item for item in summary["paired"] if item["simulator"] == simulator]
        deltas = ", ".join(
            f"{name}={np.mean([item[name] for item in paired]):+.4g}"
            for name in ("global_macro_nrmse_delta_gp_minus_no_gp", "global_macro_nlpd_delta_gp_minus_no_gp", "local_macro_nrmse_delta_gp_minus_no_gp", "local_macro_nlpd_delta_gp_minus_no_gp")
        )
        lines.append(f"- {simulator}: found GP={row['gp_found_rate']:.3f}, no-GP={row['no_gp_found_rate']:.3f}; median arrival GP={row['median_gp_arrival']:.1f}, no-GP={row['median_no_gp_arrival']:.1f}; mean GP-noGP deltas ({deltas}); win/tie/loss={row['win_tie_loss']}")
    if summary.get("overall"):
        lines += ["", f"全体: {summary['overall']}"]
    lines += ["", "## 指標定義", "", "MAE/RMSE/NRMSEは全候補格子と最適点から正規化距離<=local_radiusのlocal windowで出力別に計算。macro NRMSEは出力NRMSEの単純平均。尤度は出力幅で正規化したGaussian NLPD、coverageは |error|<=1.96*std。feasible accuracyは物理制約による分類一致率です。"]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_experiment(simulators: list[str], seeds: list[int], iterations: int, output: Path, local_radius: float) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing output directory: {output}")
    output.mkdir(parents=True)
    results = []
    for simulator_id in simulators:
        sim = load(simulator_id)
        for seed in seeds:
            for arm in ("gp", "no_gp"):
                results.append(run_arm(sim, arm, seed, iterations, local_radius))
    payload = {
        "config": {
            "optimizer": "v003",
            "simulators": simulators,
            "seeds": seeds,
            "seed_role": "initial design selection; the production NN seed remains fixed",
            "nn_seed": NN_SEED,
            "iterations": iterations,
            "local_radius": local_radius,
            "noise": 0.0,
        },
        "simulator_provenance": {
            simulator_id: load(simulator_id).provenance()
            for simulator_id in simulators
        },
        "results": results,
        "summary": aggregate(results, iterations),
    }
    (output / "raw_results.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_csv(output / "raw_results.csv", results)
    _write_trajectories(output / "trajectories.csv", results)
    _write_report(output / "report.md", payload)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="v003 GP残差あり/なし物理シミュレータ比較")
    parser.add_argument("--simulators", nargs="+", default=list(DEFAULT_SIMULATORS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument("--iterations", type=int, default=15)
    parser.add_argument("--local-radius", type=float, default=DEFAULT_LOCAL_RADIUS)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.iterations < 1 or not math.isfinite(args.local_radius) or args.local_radius <= 0 or not args.seeds or any(seed < 0 for seed in args.seeds):
        raise SystemExit("iterations/seeds/local-radius must be positive")
    run_experiment(args.simulators, args.seeds, args.iterations, args.output, args.local_radius)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
