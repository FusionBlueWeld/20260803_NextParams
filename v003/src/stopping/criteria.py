"""実験継続・終了を判定する純粋関数。

停止には二つの層があります。予算・候補枯渇・実測目標到達は即時の
``STOP_REQUIRED``、モデルの収束は複数runの履歴を要する
``STOP_RECOMMENDED``です。収束判定は安全側に倒し、履歴が不足している
場合は必ず``CONTINUE``を返します。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Sequence

import numpy as np


class StopStatus(str, Enum):
    CONTINUE = "CONTINUE"
    STOP_RECOMMENDED = "STOP_RECOMMENDED"
    STOP_REQUIRED = "STOP_REQUIRED"


@dataclass(frozen=True)
class StopConfig:
    """停止判定の閾値。スコアや目的値の単位は呼出側の単位を使います。"""

    max_additional_experiments: int | None = None
    min_history: int = 5
    patience: int = 3
    support_threshold: float = 0.8
    coverage_threshold: float = 0.8
    recent_improvement_tolerance: float = 1e-3
    top_score_threshold: float | None = 0.05
    top_score_tolerance: float = 1e-3
    condition_tolerance: float = 0.05
    objective_direction: str = "maximize"

    def __post_init__(self) -> None:
        if self.max_additional_experiments is not None and self.max_additional_experiments < 0:
            raise ValueError("max_additional_experimentsは0以上です")
        if self.min_history < 1 or self.patience < 1:
            raise ValueError("min_history/patienceは1以上です")
        if not 0 <= self.support_threshold <= 1 or not 0 <= self.coverage_threshold <= 1:
            raise ValueError("support/coverage閾値は0〜1です")
        if self.recent_improvement_tolerance < 0 or self.top_score_tolerance < 0:
            raise ValueError("変動許容値は0以上です")
        if self.condition_tolerance < 0:
            raise ValueError("condition_toleranceは0以上です")
        if self.objective_direction not in {"maximize", "minimize"}:
            raise ValueError("objective_directionはmaximize/minimizeです")


@dataclass(frozen=True)
class StopDecision:
    status: StopStatus
    reasons: tuple[str, ...] = field(default_factory=tuple)
    metrics: dict[str, float | bool | None] = field(default_factory=dict)

    @property
    def should_stop(self) -> bool:
        return self.status != StopStatus.CONTINUE


def support_coverage(
    support: Sequence[float] | np.ndarray | None,
    *,
    threshold: float = 0.8,
    allowed_mask: Sequence[bool] | np.ndarray | None = None,
) -> float | None:
    """実験可能候補に対するGP支持度カバー率を計算します。"""

    if support is None:
        return None
    values = np.asarray(support, dtype=float).reshape(-1)
    if values.size == 0:
        return None
    if allowed_mask is not None:
        mask = np.asarray(allowed_mask, dtype=bool).reshape(-1)
        if mask.size != values.size:
            raise ValueError("allowed_maskとsupportの長さが違います")
        values = values[mask]
    if values.size == 0:
        return None
    return float(np.mean(values >= threshold))


def normalized_condition_distance(a: Sequence[float], b: Sequence[float]) -> float:
    """入力条件間のユークリッド距離（各軸0〜1正規化済み想定）。"""

    left, right = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if left.shape != right.shape:
        raise ValueError("条件ベクトルの形状が違います")
    if left.size == 0:
        return 0.0
    return float(np.linalg.norm(left - right) / np.sqrt(left.size))


def _all_explored(
    allowed_candidates: np.ndarray | None,
    explored_candidates: np.ndarray | None,
    *,
    atol: float = 1e-9,
) -> bool | None:
    if allowed_candidates is None or explored_candidates is None:
        return None
    allowed = np.asarray(allowed_candidates, dtype=float)
    explored = np.asarray(explored_candidates, dtype=float)
    if allowed.ndim == 1:
        allowed = allowed.reshape(1, -1)
    if explored.size == 0:
        return allowed.shape[0] == 0
    if explored.ndim == 1:
        explored = explored.reshape(1, -1)
    if allowed.shape[1:] != explored.shape[1:]:
        raise ValueError("allowed/explored candidatesの形状が違います")
    return all(
        np.any(
            np.all(np.isclose(explored, row, atol=atol, rtol=0), axis=1)
        )
        for row in allowed
    )


def _recent_stagnation(
    objective_values: Sequence[float] | None,
    config: StopConfig,
) -> tuple[bool | None, float | None]:
    if objective_values is None or len(objective_values) < config.patience:
        return None, None
    values = np.asarray(objective_values, dtype=float)
    if values.size < config.patience:
        return None, None
    values = values[-config.patience :]
    best = float(
        np.max(values)
        if config.objective_direction == "maximize"
        else np.min(values)
    )
    first = float(values[0])
    improvement = (
        best - first
        if config.objective_direction == "maximize"
        else first - best
    )
    return improvement <= config.recent_improvement_tolerance + 1e-12, improvement


def _score_stability(
    scores: Sequence[float] | None,
    config: StopConfig,
) -> tuple[bool | None, float | None]:
    if scores is None or len(scores) < config.patience:
        return None, None
    latest = np.asarray(scores, dtype=float)[-config.patience :]
    spread = float(np.max(latest) - np.min(latest))
    small = (
        config.top_score_threshold is None
        or float(np.max(latest)) <= config.top_score_threshold
    )
    return bool(small and spread <= config.top_score_tolerance), spread


def _condition_stability(
    conditions: Sequence[Sequence[float]] | np.ndarray | None,
    config: StopConfig,
) -> tuple[bool | None, float | None]:
    if conditions is None or len(conditions) < config.patience:
        return None, None
    latest = np.asarray(conditions, dtype=float)[-config.patience :]
    distances = [
        normalized_condition_distance(left, right)
        for left, right in zip(latest, latest[1:])
    ]
    maximum = float(max(distances, default=0.0))
    return maximum <= config.condition_tolerance, maximum


def evaluate_stop(
    *,
    config: StopConfig,
    additional_experiments: int | None = None,
    allowed_candidates: np.ndarray | None = None,
    explored_candidates: np.ndarray | None = None,
    best_feasible_objective_history: Sequence[float] | None = None,
    target_reached: bool = False,
    support: Sequence[float] | np.ndarray | None = None,
    top_scores: Sequence[float] | None = None,
    predicted_optimum_conditions: Sequence[Sequence[float]] | np.ndarray | None = None,
) -> StopDecision:
    """現在の履歴から停止状態を返します（状態を変更しません）。

    ``best_feasible_objective_history`` は各runでの「制約適合した実測値の
    best」を渡します。``predicted_optimum_conditions`` は既測定候補を
    除外した推薦点ではなく、全許可グリッド上で毎run再計算した予測最適
    条件です。そのため、実験済み点の除外による見かけの揺れを避けられます。
    """

    reasons: list[str] = []
    metrics: dict[str, float | bool | None] = {}
    count = (
        additional_experiments
        if additional_experiments is not None
        else len(best_feasible_objective_history or ())
    )
    if (
        config.max_additional_experiments is not None
        and count >= config.max_additional_experiments
    ):
        reasons.append("最大追加実験回数に到達")
    explored = _all_explored(allowed_candidates, explored_candidates)
    metrics["all_candidates_explored"] = explored
    if explored is True:
        reasons.append("許可された候補を全探索")
    if target_reached:
        reasons.append("制約適合した実測値が目標に到達")
    if reasons:
        return StopDecision(StopStatus.STOP_REQUIRED, tuple(reasons), metrics)

    coverage = support_coverage(support, threshold=config.support_threshold)
    metrics["support_coverage"] = coverage
    coverage_ok = coverage is not None and coverage >= config.coverage_threshold
    metrics["coverage_sufficient"] = coverage_ok
    stagnant, improvement = _recent_stagnation(best_feasible_objective_history, config)
    metrics["recent_improvement"] = improvement
    metrics["objective_stagnant"] = stagnant
    stable_score, spread = _score_stability(top_scores, config)
    metrics["top_score_spread"] = spread
    metrics["top_score_stable_and_small"] = stable_score
    stable_condition, distance = _condition_stability(
        predicted_optimum_conditions,
        config,
    )
    metrics["condition_distance"] = distance
    metrics["predicted_optimum_stable"] = stable_condition
    histories = (
        best_feasible_objective_history,
        top_scores,
        predicted_optimum_conditions,
    )
    history_size = min(len(history) if history is not None else 0 for history in histories)
    metrics["minimum_history_reached"] = history_size >= config.min_history
    enough = history_size >= config.min_history
    converged = bool(
        enough
        and coverage_ok
        and stagnant is True
        and stable_score is True
        and stable_condition is True
    )
    metrics["converged"] = converged
    if converged:
        return StopDecision(StopStatus.STOP_RECOMMENDED, ("収束条件を満たした",), metrics)
    return StopDecision(StopStatus.CONTINUE, tuple(), metrics)


__all__ = [
    "StopConfig", "StopDecision", "StopStatus", "evaluate_stop",
    "normalized_condition_distance", "support_coverage",
]
