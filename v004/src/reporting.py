"""v004の問題設定と実行結果の表示。判定や学習は行いません。"""

from __future__ import annotations

from .settings import ProblemDefinition


def print_problem_summary(problem: ProblemDefinition) -> None:
    """prepare完了時に、読み取った問題設定を確認表示します。"""

    parameters = problem.parameters
    objective = problem.objective
    constraints = problem.constraints

    print("\n入力パラメータ:")
    for parameter in parameters:
        print(f"- {parameter.display_name} ({parameter.column})")

    objective_action = "最小化" if objective.direction == "minimize" else "最大化"
    print(f"\n目的:\n- {objective.display_name}を{objective_action}")

    print("\n制約:")
    if not constraints:
        print("- なし")
    for constraint in constraints:
        symbol = ">=" if constraint.direction == "greater_equal" else "<="
        print(f"- {constraint.display_name} {symbol} {constraint.target:g}")

    print(f"\n探索候補数: {problem.candidate_count:,}件")

