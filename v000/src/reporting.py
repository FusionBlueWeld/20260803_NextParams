"""CLI表示とrecommendations.csvの出力を担当する機能。"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from .data_loader import write_csv_atomic
from .optimizer import OptimizationResult
from .settings import ProblemDefinition
from .validation import ExperimentData


# ---------------------------------------------------------------------------
# CLIへ人間向けの概要を表示する機能
# ---------------------------------------------------------------------------


def print_run_result(
    trial_name: str,
    experiments: ExperimentData,
    problem: ProblemDefinition,
    result: OptimizationResult,
    output_path: Path,
) -> None:
    """探索結果を、詳細すぎない日本語で表示します。"""

    unique_conditions = len({tuple(row) for row in experiments.parameter_rows})
    objective_text = format_objective(problem)
    constraint_text = format_constraints(problem)

    print("ParamOptimizer v000")
    print("-------------------")
    print(f"trial: {trial_name}")
    print(f"実験データ: {experiments.row_count}件")
    print(f"固有条件: {unique_conditions}件")
    print(f"入力パラメータ: {', '.join(item.display_name for item in problem.parameters)}")
    print(f"目的: {objective_text}")
    print(f"制約: {constraint_text}")
    print(f"評価した未測定候補: {result.evaluated_candidate_count:,}件")
    print(f"今回の推薦数: {result.requested_recommendation_count}件")
    print(
        f"自動分散比率: {100 * result.diversity_weight:.1f}% "
        f"（不確かさ={result.diversity_uncertainty_signal:.3f}, "
        f"未被覆距離={result.diversity_coverage_signal:.3f}）"
    )
    if result.feasibility_guard_active:
        print("制約確率ガード: 有効（2位以降）")

    if result.observed_best is not None:
        print("\n実測済みの最良条件:")
        print("  " + format_parameter_values(result.observed_best, problem))
        print(
            f"  {problem.objective.display_name}="
            f"{format_number(result.observed_best[problem.objective.column])}"
        )
    elif problem.constraints:
        print("\n実測済みデータには、全制約を満たす条件がまだありません。")

    print("\n次に試す候補:")
    for recommendation in result.recommendations:
        rank = recommendation["rank"]
        print(f"{rank}位  {format_parameter_values(recommendation, problem)}")
        print("      " + format_prediction_values(recommendation, problem))
        print(f"      理由: {recommendation['recommendation_reason']}")

    for warning in result.warnings:
        print(f"\n[注意] {warning}")

    print("\n推薦値はモデル予測です。実測による確認が必要です。")
    print(f"\n出力:\n{output_path}")


def format_objective(problem: ProblemDefinition) -> str:
    """目的変数を日本語で説明します。"""

    action = "最小化" if problem.objective.direction == "minimize" else "最大化"
    return f"{problem.objective.display_name}を{action}"


def format_constraints(problem: ProblemDefinition) -> str:
    """複数制約を1行の日本語へまとめます。"""

    if not problem.constraints:
        return "なし"
    parts: list[str] = []
    for constraint in problem.constraints:
        symbol = ">=" if constraint.direction == "greater_equal" else "<="
        parts.append(f"{constraint.display_name} {symbol} {format_number(constraint.target)}")
    return "、".join(parts)


def format_parameter_values(values: dict[str, object], problem: ProblemDefinition) -> str:
    """入力条件を「名前=値」の形で表示します。"""

    parts: list[str] = []
    for parameter in problem.parameters:
        text = f"{parameter.display_name}={format_number(values[parameter.column])}"
        if parameter.unit:
            text += f" {parameter.unit}"
        parts.append(text)
    return ", ".join(parts)


def format_prediction_values(values: dict[str, object], problem: ProblemDefinition) -> str:
    """目的と制約の予測を短い1行にまとめます。"""

    parts: list[str] = []
    for result in [problem.objective, *problem.constraints]:
        mean = format_number(values[f"{result.column}_mean"])
        std = format_number(values[f"{result.column}_std"])
        text = f"{result.display_name}予測={mean} ± {std}"
        if result.role == "constraint":
            probability = 100 * float(values[f"{result.column}_probability"])
            text += f"（達成確率{probability:.1f}%）"
        parts.append(text)
    return " / ".join(parts)


def format_number(value: object) -> str:
    """CSVやCLIで読みやすい桁数へ整えます。"""

    if value is None:
        return "-"
    return f"{float(value):.8g}"


# ---------------------------------------------------------------------------
# recommendations.csvを書き込む機能
# ---------------------------------------------------------------------------


def write_recommendations(
    path: Path,
    experiments: ExperimentData,
    problem: ProblemDefinition,
    result: OptimizationResult,
) -> None:
    """最新推薦だけを、入力・出力数に合わせたCSVへ保存します。"""

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

    for variable in problem.result_variables:
        header.extend([f"{variable.column}_mean", f"{variable.column}_std"])
        if variable.role == "constraint":
            header.append(f"{variable.column}_probability")

    header.extend(
        [
            "feasibility_probability",
            "expected_improvement",
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


def csv_value(value: object) -> object:
    """浮動小数点を長すぎない文字列にしてCSVへ渡します。"""

    if isinstance(value, float):
        return f"{value:.10g}"
    return value
