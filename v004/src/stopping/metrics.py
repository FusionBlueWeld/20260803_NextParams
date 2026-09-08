"""終了判定に必要な目標到達と予測最適条件を計算します。"""

from __future__ import annotations

import numpy as np

from ..settings import ProblemDefinition


def objective_target_reached(
    observed_best: dict[str, float] | None,
    problem: ProblemDefinition,
) -> bool:
    """制約適合した実測最良値が、任意設定の目的目標へ達したか返します。"""

    target = problem.objective.target
    if observed_best is None or target is None:
        return False
    value = observed_best[problem.objective.column]
    if problem.objective.direction == "maximize":
        return value >= target
    return value <= target


def predicted_optimum_index(
    prediction,
    problem: ProblemDefinition,
    allowed_mask: np.ndarray,
) -> int | None:
    """最終予測平均が制約を満たす許可候補から、予測最適点を返します。"""

    feasible = np.asarray(allowed_mask, dtype=bool).copy()
    for constraint in problem.constraints:
        values = prediction.results[constraint.column]["hybrid_mean"]
        if constraint.direction == "greater_equal":
            feasible &= values >= float(constraint.target)
        else:
            feasible &= values <= float(constraint.target)
    indices = np.flatnonzero(feasible)
    if not len(indices):
        return None
    objective = prediction.results[problem.objective.column]["hybrid_mean"]
    local = (
        int(np.argmax(objective[indices]))
        if problem.objective.direction == "maximize"
        else int(np.argmin(objective[indices]))
    )
    return int(indices[local])
