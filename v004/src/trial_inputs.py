"""prepareとrunが共有する入力設定の読込。

問題・工程接続・知識・探索領域・停止条件を同じ順序で検査します。
実験CSVはrunでだけ必要なため、ここでは読み込みません。

"""

from dataclasses import dataclass
from pathlib import Path

from .connection import CONNECTION_FILE_NAME, StageConnection, load_stage_connection
from .knowledge import KnowledgeRule, load_knowledge_constraints
from .policies import load_search_regions
from .policies.regions import RegionPolicy
from .settings import PROBLEM_FILE_NAME, ProblemDefinition
from .stopping import StopConfig, load_stop_config
from .validation import UserInputError, load_and_validate_problem


@dataclass(frozen=True)
class TrialInputs:
    """検査済みの入力設定。学習結果や出力ファイルは含めません。"""

    problem: ProblemDefinition
    connection: StageConnection | None
    knowledge_rules: list[KnowledgeRule]
    region_policy: RegionPolicy
    stop_config: StopConfig


def load_trial_inputs(trial: Path) -> TrialInputs:
    """任意設定がない場合も、各読込器の既定動作を維持します。"""

    problem = load_and_validate_problem(trial / PROBLEM_FILE_NAME)
    connection = load_stage_connection(trial / CONNECTION_FILE_NAME, problem)
    knowledge_rules = load_knowledge_constraints(trial / "knowledge_constraints.csv", problem)
    try:
        region_policy = load_search_regions(
            trial / "search_regions.csv", [item.column for item in problem.parameters],
        )
        stop_config = load_stop_config(
            trial / "stop_settings.csv", objective_direction=problem.objective.direction,
        )
    except ValueError as error:
        raise UserInputError(str(error)) from error
    return TrialInputs(problem, connection, knowledge_rules, region_policy, stop_config)
