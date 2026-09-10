"""製品版 v002 と v003 の同条件物理シミュレータ比較。

v002 は ``直接 GP + NN の support blend``、v003 は ``NN + support * 残差 GP``
に v003 の現在の最小知識 (objective の lower_bound=0, strength=2) を加えた
実装を、そのまま同一プロセスから呼び出す。この比較は GP 残差だけの要因分解
ではなく、v002 から v003 への総合的な進化比較である。
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from ..simulators import Simulator, load
from .gp_residual_ablation import _metric_for_mask, calculate_metrics

# Package names are deliberately qualified.  Both versions have modules called
# ``src`` and importing either one through a bare path would silently collide.
from v002.src.hybrid.model import train_hybrid_model as train_v002
from v002.src.hybrid.optimizer import run_optimization as optimize_v002
from v002.src.preprocessing import normalize_parameters as normalize_v002
from v002.src.preprocessing import prepare_experiments as prepare_v002
from v002.src.settings import NN_SEED as V002_NN_SEED
from v002.src.settings import ProblemDefinition as V002Problem
from v002.src.settings import VariableDefinition as V002Variable
from v002.src.validation import ExperimentData as V002ExperimentData
from v003.src.hybrid.model import train_hybrid_model as train_v003
from v003.src.hybrid.optimizer import run_optimization as optimize_v003
from v003.src.knowledge import KnowledgeRule
from v003.src.preprocessing import normalize_parameters as normalize_v003
from v003.src.preprocessing import prepare_experiments as prepare_v003
from v003.src.settings import NN_SEED as V003_NN_SEED
from v003.src.settings import ProblemDefinition as V003Problem
from v003.src.settings import VariableDefinition as V003Variable
from v003.src.validation import ExperimentData as V003ExperimentData


DEFAULT_SIMULATORS = (
    "thermal_curing",
    "press_forming",
    "convection_drying",
    "electroplating",
    "milling",
)
DEFAULT_SEEDS = (0, 1, 2)
DEFAULT_ITERATIONS = 15
DEFAULT_LOCAL_RADIUS = 0.20
_METRICS = ("macro_nrmse", "macro_nlpd", "macro_coverage95", "feasible_accuracy")
_SCOPES = ("global", "local", "near_optimum_k64")
_EPS = 1e-12
_OPTIMUM_RTOL = 1e-9
_OPTIMUM_ATOL = 1e-10


def _convert_variable(variable: Any, cls: type) -> Any:
    return cls(
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


def make_problem(sim: Simulator, version: str) -> V002Problem | V003Problem:
    """Translate the validation simulator to the selected production contract."""

    variable_cls = V002Variable if version == "v002" else V003Variable
    problem_cls = V002Problem if version == "v002" else V003Problem
    outputs = [_convert_variable(item, variable_cls) for item in sim.outputs]
    return problem_cls(
        parameters=[_convert_variable(item, variable_cls) for item in sim.parameters],
        objective=_convert_variable(sim.objective, variable_cls),
        constraints=[_convert_variable(item, variable_cls) for item in sim.constraints],
        monitors=[item for item in outputs if item.role == "monitor"],
        result_order=outputs,
        candidate_count=len(sim.grid()),
    )


def knowledge_rules(problem: V003Problem) -> list[KnowledgeRule]:
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


def _experiment_data(sim: Simulator, points: np.ndarray, values: Mapping[str, np.ndarray], version: str) -> Any:
    cls = V002ExperimentData if version == "v002" else V003ExperimentData
    return cls(
        experiment_ids=[f"evolution_{i:05d}" for i in range(len(points))],
        parameter_rows=np.asarray(points, dtype=float).tolist(),
        result_rows={name: np.asarray(values[name], dtype=float).tolist() for name in sim.output_columns},
    )


def _problem_feasible(values: Mapping[str, np.ndarray], problem: Any) -> np.ndarray:
    mask = np.ones(len(next(iter(values.values()))), dtype=bool)
    for constraint in problem.constraints:
        candidate = np.asarray(values[constraint.column])
        if constraint.direction == "greater_equal":
            mask &= candidate >= float(constraint.target)
        else:
            mask &= candidate <= float(constraint.target)
    return mask


def _true_regret(sim: Simulator, outputs: Mapping[str, np.ndarray], global_best: float, span: float) -> float | None:
    feasible = sim.feasible(outputs)
    if not np.any(feasible):
        return None
    observed = np.asarray(outputs[sim.objective.column], dtype=float)[feasible]
    best = float(np.min(observed) if sim.objective.direction == "minimize" else np.max(observed))
    gap = best - global_best if sim.objective.direction == "minimize" else global_best - best
    return float(max(gap, 0.0) / max(span, 1e-12))


def _optimal_mask(sim: Simulator, outputs: Mapping[str, np.ndarray], global_best: float) -> np.ndarray:
    """Return every feasible row attaining the best objective, including ties."""

    objective = np.asarray(outputs[sim.objective.column], dtype=float)
    return sim.feasible(outputs) & np.isclose(
        objective,
        global_best,
        rtol=_OPTIMUM_RTOL,
        atol=_OPTIMUM_ATOL,
    )


def _version_step(
    sim: Simulator,
    version: str,
    problem: Any,
    points: np.ndarray,
    observed: Mapping[str, np.ndarray],
    recommendation_count: int = 1,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    data = _experiment_data(sim, points, observed, version)
    if version == "v002":
        prepared = prepare_v002(data, problem)
        trained = train_v002(prepared, problem)
        result = optimize_v002(data, prepared, problem, recommendation_count=recommendation_count, model=trained.model)
    else:
        prepared = prepare_v003(data, problem)
        trained = train_v003(prepared, problem, knowledge_rules(problem))
        result = optimize_v003(data, prepared, problem, recommendation_count=recommendation_count, model=trained.model)
    next_points = np.array(
        [[float(rec[item.column]) for item in problem.parameters] for rec in result.recommendations],
        dtype=float,
    )
    infos = [
        {
            "recommendation_score": float(rec["recommendation_score"]),
            "selection_role": str(rec["selection_role"]),
            "training": trained,
        }
        for rec in result.recommendations
    ]
    return next_points, infos


def run_arm(sim: Simulator, version: str, seed: int, iterations: int, local_radius: float, batch_size: int = 1) -> dict[str, Any]:
    problem = make_problem(sim, version)
    grid = sim.grid()
    truth = sim.evaluate_points(grid)
    normalized_grid = normalize_v003(grid, make_problem(sim, "v003"))
    true_feasible = sim.feasible(truth)
    optimum_index = sim.best_index(truth)
    if optimum_index is None:
        raise ValueError(f"{sim.id} has no feasible grid optimum")
    optimum = grid[optimum_index]
    canonical_optimum_normalized = normalized_grid[optimum_index]
    objective_truth = np.asarray(truth[sim.objective.column], dtype=float)
    global_best = float(objective_truth[optimum_index])
    truth_span = max(float(np.ptp(objective_truth)), 1e-12)
    initial = sim.initial_design(seed)
    observed = sim.evaluate_points(initial)
    points = initial.copy()
    initial_regret = _true_regret(sim, {k: np.asarray(v) for k, v in sim.evaluate_points(initial).items()}, global_best, truth_span)
    history: list[dict[str, Any]] = []
    initial_optimal = _optimal_mask(sim, observed, global_best)
    arrival_step: int | None = 0 if np.any(initial_optimal) else None
    arrival_round: int | None = 0 if arrival_step == 0 else None
    arrival_consumed_conditions: int | None = 0 if arrival_step == 0 else None
    arrival_point: np.ndarray | None = initial[np.flatnonzero(initial_optimal)[0]].copy() if arrival_step == 0 else None
    checkpoint_step: int | None = None
    checkpoint_metrics: dict[str, Any] | None = None

    def checkpoint(step: int) -> None:
        nonlocal checkpoint_step, checkpoint_metrics
        current_problem = make_problem(sim, version)
        data = _experiment_data(sim, points, observed, version)
        if version == "v002":
            prepared = prepare_v002(data, current_problem)
            trained = train_v002(prepared, current_problem)
            model = trained.model
            predicted = model.predict(normalize_v002(grid, current_problem))
        else:
            prepared = prepare_v003(data, current_problem)
            trained = train_v003(prepared, current_problem, knowledge_rules(current_problem))
            model = trained.model
            predicted = model.predict(normalize_v003(grid, current_problem))
        means = {item.column: predicted.results[item.column]["hybrid_mean"] for item in current_problem.result_variables}
        std = {item.column: np.maximum(predicted.results[item.column]["hybrid_std"], 0.0) for item in current_problem.result_variables}
        predicted_feasible = _problem_feasible(means, current_problem)
        checkpoint_step = step
        center_point = optimum if arrival_point is None else arrival_point
        center_normalized = normalize_v003(center_point.reshape(1, -1), make_problem(sim, "v003"))[0]
        checkpoint_metrics = calculate_metrics(means, truth, std, true_feasible, predicted_feasible, normalized_grid, center_normalized, local_radius)
        # In 8D the radius=.20 window often contains only the exact optimum.
        # Keep the requested global/local metrics, and add a stable local proxy
        # consisting of the 64 nearest grid points for high-dimensional cases.
        nearest_count = min(64, len(grid))
        distances = np.linalg.norm(normalized_grid - center_normalized[None, :], axis=1)
        nearest_mask = np.zeros(len(grid), dtype=bool)
        nearest_mask[np.argsort(distances, kind="stable")[:nearest_count]] = True
        nearest = _metric_for_mask(means, truth, std, true_feasible, predicted_feasible, nearest_mask)
        checkpoint_metrics["near_optimum_k64"] = {"count": nearest_count, "effective_radius": float(np.max(distances[nearest_mask])), "outputs": nearest.outputs, "macro_nrmse": nearest.macro_nrmse, "macro_nlpd": nearest.macro_nlpd, "macro_coverage95": nearest.macro_coverage95, "feasible_accuracy": nearest.feasible_accuracy}
        checkpoint_metrics["training"] = {"nn_epochs": trained.nn_epochs, "nn_normalized_mse": trained.nn_normalized_mse}
        if version == "v003":
            checkpoint_metrics["training"]["nn_knowledge_loss"] = trained.nn_knowledge_loss
        checkpoint_metrics["regret"] = _true_regret(sim, sim.evaluate_points(points), global_best, truth_span)

    if arrival_step == 0:
        checkpoint(0)
    evaluated = 0
    batch_round = 0
    while arrival_step is None and evaluated < iterations:
        recommendation_count = min(batch_size, iterations - evaluated)
        next_points, infos = _version_step(
            sim, version, problem, points, observed, recommendation_count
        )
        next_truth = sim.evaluate_points(next_points)
        reached_mask = _optimal_mask(sim, next_truth, global_best)
        batch_round += 1
        for rank, (next_point, info) in enumerate(zip(next_points, infos), start=1):
            prefix_observed = {
                name: np.concatenate([observed[name], next_truth[name][:rank]])
                for name in sim.output_columns
            }
            regret = _true_regret(sim, prefix_observed, global_best, truth_span)
            reached = bool(reached_mask[rank - 1])
            history.append(
                {
                    "step": evaluated + rank,
                    "batch_round": batch_round,
                    "batch_rank": rank,
                    "point": next_point.tolist(),
                    "is_optimum": reached,
                    "selection_role": info["selection_role"],
                    "recommendation_score": info["recommendation_score"],
                    "best_true_feasible_regret": regret,
                }
            )
        points = np.vstack([points, next_points])
        observed = {name: np.concatenate([observed[name], next_truth[name]]) for name in sim.output_columns}
        if np.any(reached_mask):
            first_rank = int(np.flatnonzero(reached_mask)[0]) + 1
            arrival_step = evaluated + first_rank
            arrival_round = batch_round
            arrival_consumed_conditions = evaluated + recommendation_count
            arrival_point = next_points[first_rank - 1].copy()
        evaluated += recommendation_count
        print(
            f"[{sim.id}/{version}/seed={seed}/batch={batch_size}] "
            f"round={batch_round}, conditions={evaluated}/{iterations}",
            flush=True,
        )
        if arrival_step is not None:
            checkpoint(evaluated)
            break
    if checkpoint_step is None:
        checkpoint(iterations)
    final_regret = checkpoint_metrics["regret"] if checkpoint_metrics else None
    return {
        "simulator": sim.id,
        "complexity_group": "simple" if len(sim.parameters) <= 3 else "high_dim",
        "version": version,
        "seed": seed,
        "iterations": iterations,
        "batch_size": batch_size,
        "parameter_count": len(sim.parameters),
        "candidate_count": len(grid),
        "initial_count": len(initial),
        "arrival_step": arrival_step,
        "arrival_round": arrival_round,
        "arrival_consumed_conditions": arrival_consumed_conditions,
        "found": arrival_step is not None,
        "censored": arrival_step is None,
        "checkpoint_step": checkpoint_step,
        "optimum_index": int(optimum_index),
        "optimum_point": optimum.tolist(),
        "reached_optimum_point": None if arrival_point is None else arrival_point.tolist(),
        "initial_regret": initial_regret,
        "final_regret": final_regret,
        "regret_reduction": (initial_regret - final_regret) if initial_regret is not None and final_regret is not None else None,
        "history": history,
        "metrics": checkpoint_metrics,
        "knowledge": "none" if version == "v002" else "objective lower_bound=0, strength=2",
    }


def _winner(delta: float, lower_is_better: bool) -> str:
    if abs(delta) <= _EPS:
        return "tie"
    return "v003" if (delta < 0 if lower_is_better else delta > 0) else "v002"


def _paired_rows(results: list[dict[str, Any]], iterations: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    keys = sorted({(r["simulator"], r["seed"]) for r in results})
    for simulator, seed in keys:
        arms = {(r["version"]): r for r in results if r["simulator"] == simulator and r["seed"] == seed}
        if set(arms) != {"v002", "v003"}:
            continue
        old, new = arms["v002"], arms["v003"]
        old_arrival = old["arrival_step"] if old["arrival_step"] is not None else iterations + 1
        new_arrival = new["arrival_step"] if new["arrival_step"] is not None else iterations + 1
        row: dict[str, Any] = {"simulator": simulator, "complexity_group": old["complexity_group"], "seed": seed, "v002_found": old["found"], "v003_found": new["found"], "v002_arrival": old_arrival, "v003_arrival": new_arrival, "arrival_delta_v003_minus_v002": new_arrival - old_arrival, "arrival_winner": _winner(float(new_arrival - old_arrival), True), "v002_initial_regret": old["initial_regret"], "v003_initial_regret": new["initial_regret"], "v002_final_regret": old["final_regret"], "v003_final_regret": new["final_regret"], "v002_regret_reduction": old["regret_reduction"], "v003_regret_reduction": new["regret_reduction"]}
        for name, lower in (("final_regret", True), ("regret_reduction", False)):
            a, b = old[name], new[name]
            delta = float(b - a) if a is not None and b is not None else None
            row[f"{name}_delta_v003_minus_v002"] = delta
            row[f"{name}_winner"] = "unavailable" if delta is None else _winner(delta, lower)
        for scope in _SCOPES:
            for metric in _METRICS:
                a, b = float(old["metrics"][scope][metric]), float(new["metrics"][scope][metric])
                delta = b - a
                row[f"{scope}_{metric}_delta_v003_minus_v002"] = delta
                row[f"{scope}_{metric}_winner"] = _winner(delta, metric in {"macro_nrmse", "macro_nlpd"})
        rows.append(row)
    return rows


def _group_summary(rows: list[dict[str, Any]], iterations: int) -> dict[str, Any]:
    def one(group_rows: list[dict[str, Any]]) -> dict[str, Any]:
        if not group_rows:
            return {"pairs": 0}
        out: dict[str, Any] = {"pairs": len(group_rows)}
        for version in ("v002", "v003"):
            out[f"{version}_arrival_rate"] = float(np.mean([r[f"{version}_found"] for r in group_rows]))
            out[f"{version}_median_censored_arrival"] = float(np.median([r[f"{version}_arrival"] for r in group_rows]))
        out["arrival_win_tie_loss"] = {x: sum(r["arrival_winner"] == x for r in group_rows) for x in ("v003", "tie", "v002")}
        for name in ("final_regret", "regret_reduction"):
            deltas = [r[f"{name}_delta_v003_minus_v002"] for r in group_rows if r[f"{name}_delta_v003_minus_v002"] is not None]
            out[f"{name}_mean_delta_v003_minus_v002"] = float(np.mean(deltas)) if deltas else None
            out[f"{name}_win_tie_loss"] = {x: sum(r[f"{name}_winner"] == x for r in group_rows) for x in ("v003", "tie", "v002", "unavailable")}
        for scope in _SCOPES:
            for metric in _METRICS:
                key = f"{scope}_{metric}_delta_v003_minus_v002"
                out[f"{key}_mean"] = float(np.mean([r[key] for r in group_rows]))
                out[f"{scope}_{metric}_win_tie_loss"] = {x: sum(r[f"{scope}_{metric}_winner"] == x for r in group_rows) for x in ("v003", "tie", "v002")}
        return out

    groups = {"simple": [r for r in rows if r["complexity_group"] == "simple"], "high_dim": [r for r in rows if r["complexity_group"] == "high_dim"], "overall": rows}
    return {name: one(group) for name, group in groups.items()}


def aggregate(results: list[dict[str, Any]], iterations: int) -> dict[str, Any]:
    rows = _paired_rows(results, iterations)
    by_simulator = {name: _group_summary([r for r in rows if r["simulator"] == name], iterations)["overall"] for name in sorted({r["simulator"] for r in rows})}
    return {"iterations": iterations, "paired": rows, "by_simulator": by_simulator, "by_complexity": _group_summary(rows, iterations), "overall": _group_summary(rows, iterations)["overall"]}


def _write_results_csv(path: Path, results: list[dict[str, Any]]) -> None:
    rows: list[dict[str, Any]] = []
    for result in results:
        row = {k: result[k] for k in ("simulator", "complexity_group", "version", "seed", "batch_size", "parameter_count", "candidate_count", "found", "censored", "arrival_step", "arrival_round", "arrival_consumed_conditions", "checkpoint_step", "initial_regret", "final_regret", "regret_reduction")}
        row["local_count"] = result["metrics"].get("local_count")
        row["near_optimum_k64_count"] = result["metrics"]["near_optimum_k64"]["count"]
        row["near_optimum_k64_effective_radius"] = result["metrics"]["near_optimum_k64"]["effective_radius"]
        for scope in _SCOPES:
            for metric in _METRICS:
                row[f"{scope}_{metric}"] = result["metrics"][scope][metric]
        for output, values in result["metrics"]["global"]["outputs"].items():
            for metric, value in values.items():
                row[f"global_{output}_{metric}"] = value
        rows.append(row)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    if not fields:
        fields = ["simulator", "version", "seed"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _write_trajectories(path: Path, results: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["simulator", "complexity_group", "version", "seed", "batch_size", "step", "batch_round", "batch_rank", "point", "is_optimum", "selection_role", "recommendation_score", "best_true_feasible_regret"])
        writer.writeheader()
        for result in results:
            for step in result["history"]:
                writer.writerow({"simulator": result["simulator"], "complexity_group": result["complexity_group"], "version": result["version"], "seed": result["seed"], "batch_size": result["batch_size"], **step})


def _source_hash(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted((root / "src").rglob("*.py")):
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(path.read_bytes().replace(b"\r\n", b"\n"))
    return digest.hexdigest()


def _write_report(path: Path, payload: dict[str, Any]) -> None:
    summary = payload["summary"]
    batch_size = payload["config"]["batch_size"]
    lines = ["# v002 vs v003 製品版進化比較", "", f"この比較は v002 の直接GP+NN support blend と、v003 の残差GP + objective lower_bound=0 (strength=2) を、同じ物理シミュレータ・seed・初期設計・ノイズ0・推薦batch={batch_size}件・総取得予算で比較する総合比較です。GP残差単独の要因分解ではありません。", "", "未到達は arrival=budget+1 として censored median に含めました。到達は制約付き大域最適目的値を持つ任意の格子点（同率最適を含む）で判定します。予測指標は到達した最適点を中心に到達バッチ反映直後、未到達は代表最適点を中心にbudget末のcheckpointで算出します。", ""]
    for group, item in payload["summary"].get("by_complexity", {}).items():
        if item.get("pairs"):
            lines.append(f"- {group}: pairs={item['pairs']}, arrival={item['arrival_win_tie_loss']}, final_regret Δ(v003-v002)={item['final_regret_mean_delta_v003_minus_v002']}, reduction Δ={item['regret_reduction_mean_delta_v003_minus_v002']}")
    lines += ["", "## 工程別", ""]
    for name, item in summary.get("by_simulator", {}).items():
        if item.get("pairs"):
            lines.append(f"- {name}: {item}")
    lines += ["", "## 指標", "", "MAE/RMSE/NRMSE、出力幅正規化Gaussian NLPD、95% coverage、feasible accuracyを全候補とcheckpoint中心から正規化距離<=0.20のlocal windowで出力別およびmacroで算出しました。regretは真の制約付き最適目的値に対する、観測済み最良真値の正規化gapです。8Dではlocal_countとnearest-64のeffective_radiusを保存し、半径0.20の1点評価だけで結論を出さないようにしました。"]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_experiment(simulators: list[str], seeds: list[int], iterations: int, output: Path, local_radius: float = DEFAULT_LOCAL_RADIUS, batch_size: int = 1) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing output directory: {output}")
    if not simulators or not seeds or any(seed < 0 for seed in seeds) or iterations < 1 or batch_size < 1 or not math.isfinite(local_radius) or local_radius <= 0:
        raise ValueError("simulators/seeds must be nonempty, seeds nonnegative, iterations/batch_size/local_radius positive")
    output.mkdir(parents=True)
    results: list[dict[str, Any]] = []
    for simulator_id in simulators:
        sim = load(simulator_id)
        for seed in seeds:
            for version in ("v002", "v003"):
                results.append(run_arm(sim, version, seed, iterations, local_radius, batch_size))
    repo_root = Path(__file__).resolve().parents[4]
    payload = {"config": {"versions": ["v002", "v003"], "simulators": simulators, "seeds": seeds, "seed_role": "initial design selection", "nn_seeds": {"v002": V002_NN_SEED, "v003": V003_NN_SEED, "equal": V002_NN_SEED == V003_NN_SEED}, "iterations": iterations, "batch_size": batch_size, "local_radius": local_radius, "noise": 0.0, "comparison_scope": "version evolution; not GP-residual-only factorization"}, "simulator_provenance": {sim.id: sim.provenance() for sim in [load(x) for x in simulators]}, "version_provenance": {"v002": {"source_hash": _source_hash(repo_root / "v002")}, "v003": {"source_hash": _source_hash(repo_root / "v003")}}, "results": results, "summary": aggregate(results, iterations)}
    (output / "raw_results.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    _write_results_csv(output / "raw_results.csv", results)
    _write_trajectories(output / "trajectories.csv", results)
    _write_report(output / "report.md", payload)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="v002/v003 製品版進化比較")
    parser.add_argument("--simulators", nargs="+", default=list(DEFAULT_SIMULATORS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument("--iterations", type=int, default=DEFAULT_ITERATIONS)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--local-radius", type=float, default=DEFAULT_LOCAL_RADIUS)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.simulators or not args.seeds or any(seed < 0 for seed in args.seeds) or args.iterations < 1 or args.batch_size < 1 or not math.isfinite(args.local_radius) or args.local_radius <= 0:
        raise SystemExit("simulators/seeds must be nonempty, seeds nonnegative, iterations/batch-size/local-radius positive")
    run_experiment(args.simulators, args.seeds, args.iterations, args.output, args.local_radius, args.batch_size)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
