"""Hybrid推薦とrun別解空間CSVを出力します。"""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path
from typing import Iterable
from uuid import uuid4

import numpy as np

from ..data_loader import write_csv_atomic
from ..preprocessing import PreparedData
from ..settings import (
    HYBRID_RESPONSE_FILE_PREFIX,
    HYBRID_ACQUISITION_UNCERTAINTY_SCALE,
    HYBRID_UNCERTAINTY_CALIBRATION_SCALE,
    NN_ENSEMBLE_SIZE,
    NN_SEED,
    ProblemDefinition,
)
from ..validation import ExperimentData
from .model import HybridPrediction
from .optimizer import OptimizationResult


def archive_recommendations(path: Path) -> Path | None:
    """再実行前に古い推薦を履歴へ移し、最新の推薦との取り違えを防ぎます。"""
    if not path.exists():
        return None
    archive = path.parent / "recommendations_history"
    archive.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S_%f")
    destination = archive / f"recommendations_{stamp}_{uuid4().hex[:8]}.csv"
    path.rename(destination)
    return destination


def print_run_result(
    trial_name: str,
    experiments: ExperimentData,
    problem: ProblemDefinition,
    result: OptimizationResult,
    recommendation_path: Path,
) -> None:
    """Hybrid探索結果を日本語で表示します。"""

    unique_conditions = len({tuple(row) for row in experiments.parameter_rows})
    print("ParamOptimizer v004")
    print("-------------------")
    print(f"trial: {trial_name}")
    print(f"実験データ: {experiments.row_count}件")
    print(f"固有条件: {unique_conditions}件")
    print(f"入力パラメータ: {', '.join(item.display_name for item in problem.parameters)}")
    print(f"目的: {format_objective(problem)}")
    print(f"制約: {format_constraints(problem)}")
    print(f"評価した未測定候補: {result.evaluated_candidate_count:,}件")
    print(f"今回の推薦数: {result.requested_recommendation_count}件")
    print(
        f"自動分散比率: {100 * result.diversity_weight:.1f}% "
        f"（不確かさ={result.diversity_uncertainty_signal:.3f}, "
        f"未被覆距離={result.diversity_coverage_signal:.3f}）"
    )

    if result.observed_best is not None:
        print("\n実測済みの最良条件:")
        print("  " + format_parameter_values(result.observed_best, problem))
        print(
            f"  {problem.objective.display_name}="
            f"{format_number(result.observed_best[problem.objective.column])}"
        )
    elif problem.constraints:
        print("\n実測済みデータには、全制約を満たす条件がまだありません。")

    print("\nHybridによる次に試す候補:")
    for recommendation in result.recommendations:
        print(
            f"{recommendation['rank']}位  "
            f"{format_parameter_values(recommendation, problem)}"
        )
        print("      " + format_prediction_values(recommendation, problem))
        print(f"      GPデータ支持度={float(recommendation['gp_support']):.3f}")
        print(f"      理由: {recommendation['recommendation_reason']}")

    for warning in result.warnings:
        print(f"\n[注意] {warning}")
    print("\nHybrid予測値は実測値ではありません。実測による確認が必要です。")
    print(f"\n推薦出力:\n{recommendation_path}")


def write_recommendations(
    path: Path,
    experiments: ExperimentData,
    problem: ProblemDefinition,
    result: OptimizationResult,
) -> None:
    """最新のHybrid推薦をv000互換列と追加根拠列で保存します。"""

    generated_at = datetime.now().astimezone().isoformat(timespec="seconds")
    header = [
        "generated_at",
        "data_rows",
        "requested_recommendations",
        "auto_diversity_weight",
        "auto_uncertainty_signal",
        "auto_coverage_signal",
        "feasibility_guard_active",
        "rank",
        "selection_role",
    ]
    header.extend(item.column for item in problem.parameters)
    header.append("gp_support")
    for variable in problem.result_variables:
        header.extend(
            [
                f"{variable.column}_nn_pred",
                f"{variable.column}_nn_std",
                f"{variable.column}_raw_std",
                f"{variable.column}_calibration_scale",
                f"{variable.column}_acquisition_std",
                f"{variable.column}_mean",
                f"{variable.column}_std",
            ]
        )
        if variable.role == "constraint":
            header.append(f"{variable.column}_probability")
    header.extend(
        [
            "feasibility_probability",
            "expected_improvement",
            "base_recommendation_score",
            "preferred_multiplier",
            "recommendation_score",
            "nearest_distance",
            "recommendation_reason",
        ]
    )
    rows: list[dict[str, object]] = []
    for recommendation in result.recommendations:
        row = {
            "generated_at": generated_at,
            "data_rows": experiments.row_count,
            "requested_recommendations": result.requested_recommendation_count,
            "auto_diversity_weight": result.diversity_weight,
            "auto_uncertainty_signal": result.diversity_uncertainty_signal,
            "auto_coverage_signal": result.diversity_coverage_signal,
            "feasibility_guard_active": result.feasibility_guard_active,
            **recommendation,
        }
        rows.append({key: csv_value(row.get(key, "")) for key in header})
    write_csv_atomic(path, header, rows)


def next_run_id(response_directory: Path) -> str:
    pattern = re.compile(rf"^{re.escape(HYBRID_RESPONSE_FILE_PREFIX)}(\d{{4}})\.csv$")
    numbers: list[int] = []
    if response_directory.is_dir():
        for path in response_directory.iterdir():
            match = pattern.fullmatch(path.name)
            if match:
                numbers.append(int(match.group(1)))
    return f"run_{max(numbers, default=0) + 1:04d}"


def response_space_path(response_directory: Path, run_id: str) -> Path:
    number = run_id.removeprefix("run_")
    return response_directory / f"{HYBRID_RESPONSE_FILE_PREFIX}{number}.csv"


def write_response_space(
    path: Path,
    run_id: str,
    experiments: ExperimentData,
    prepared: PreparedData,
    problem: ProblemDefinition,
    raw_grid: np.ndarray,
    prediction: HybridPrediction,
    experiment_allowed: np.ndarray | None = None,
    preferred_multiplier: np.ndarray | None = None,
    ensemble_size: int = NN_ENSEMBLE_SIZE,
    uncertainty_calibration_scale: float = HYBRID_UNCERTAINTY_CALIBRATION_SCALE,
    acquisition_uncertainty_scale: float = HYBRID_ACQUISITION_UNCERTAINTY_SCALE,
) -> None:
    """NNグリッド、GP支持度、Hybrid予測を同じ行へ保存します。"""

    header = [
        "run_id",
        "data_rows",
        "unique_conditions",
        "nn_seed",
        "nn_ensemble_size",
        "uncertainty_calibration_scale",
        "acquisition_uncertainty_scale",
    ]
    header.extend(item.column for item in problem.parameters)
    header.extend(["experiment_allowed", "preferred_multiplier", "gp_support"])
    allowed = (
        np.ones(len(raw_grid), dtype=bool)
        if experiment_allowed is None
        else np.asarray(experiment_allowed, dtype=bool)
    )
    multipliers = (
        np.ones(len(raw_grid), dtype=float)
        if preferred_multiplier is None
        else np.asarray(preferred_multiplier, dtype=float)
    )
    for result in problem.result_variables:
        header.extend(
            [
                f"{result.column}_nn_pred",
                f"{result.column}_nn_std",
                f"{result.column}_hybrid_mean",
                f"{result.column}_raw_hybrid_std",
                f"{result.column}_calibration_scale",
                f"{result.column}_acquisition_std",
                f"{result.column}_hybrid_std",
            ]
        )

    def rows() -> Iterable[dict[str, object]]:
        for row_index in range(len(raw_grid)):
            row: dict[str, object] = {
                "run_id": run_id,
                "data_rows": experiments.row_count,
                "unique_conditions": prepared.unique_condition_count,
                "nn_seed": NN_SEED,
                "nn_ensemble_size": ensemble_size,
                "uncertainty_calibration_scale": uncertainty_calibration_scale,
                "acquisition_uncertainty_scale": acquisition_uncertainty_scale,
                "experiment_allowed": allowed[row_index],
                "preferred_multiplier": f"{multipliers[row_index]:.10g}",
                "gp_support": f"{prediction.support[row_index]:.10g}",
            }
            for column_index, parameter in enumerate(problem.parameters):
                row[parameter.column] = f"{raw_grid[row_index, column_index]:.10g}"
            for result in problem.result_variables:
                values = prediction.results[result.column]
                row[f"{result.column}_nn_pred"] = f"{values['nn_pred'][row_index]:.10g}"
                row[f"{result.column}_nn_std"] = f"{values['nn_std'][row_index]:.10g}"
                row[f"{result.column}_hybrid_mean"] = (
                    f"{values['hybrid_mean'][row_index]:.10g}"
                )
                row[f"{result.column}_raw_hybrid_std"] = (
                    f"{values['raw_hybrid_std'][row_index]:.10g}"
                )
                row[f"{result.column}_calibration_scale"] = (
                    f"{values['uncertainty_calibration_scale'][row_index]:.10g}"
                )
                row[f"{result.column}_acquisition_std"] = (
                    f"{values['acquisition_std'][row_index]:.10g}"
                )
                row[f"{result.column}_hybrid_std"] = (
                    f"{values['hybrid_std'][row_index]:.10g}"
                )
            yield row

    write_csv_atomic(path, header, rows())


def format_objective(problem: ProblemDefinition) -> str:
    action = "最小化" if problem.objective.direction == "minimize" else "最大化"
    return f"{problem.objective.display_name}を{action}"


def format_constraints(problem: ProblemDefinition) -> str:
    if not problem.constraints:
        return "なし"
    parts: list[str] = []
    for constraint in problem.constraints:
        symbol = ">=" if constraint.direction == "greater_equal" else "<="
        parts.append(f"{constraint.display_name} {symbol} {format_number(constraint.target)}")
    return "、".join(parts)


def format_parameter_values(values: dict[str, object], problem: ProblemDefinition) -> str:
    parts: list[str] = []
    for parameter in problem.parameters:
        text = f"{parameter.display_name}={format_number(values[parameter.column])}"
        if parameter.unit:
            text += f" {parameter.unit}"
        parts.append(text)
    return ", ".join(parts)


def format_prediction_values(values: dict[str, object], problem: ProblemDefinition) -> str:
    parts: list[str] = []
    for result in [problem.objective, *problem.constraints]:
        mean = format_number(values[f"{result.column}_mean"])
        std = format_number(values[f"{result.column}_std"])
        text = f"{result.display_name}Hybrid予測={mean} ± {std}"
        if result.role == "constraint":
            probability = 100 * float(values[f"{result.column}_probability"])
            text += f"（達成確率{probability:.1f}%）"
        parts.append(text)
    return " / ".join(parts)


def format_number(value: object) -> str:
    if value is None:
        return "-"
    return f"{float(value):.8g}"


def csv_value(value: object) -> object:
    return f"{value:.10g}" if isinstance(value, float) else value
