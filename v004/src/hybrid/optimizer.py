

"""探索候補を作り、次に試す条件を選ぶ機能。

このファイルが、v004のベイズ最適化の中心です。処理を小さな関数に分け、
候補生成、予測、点数計算、上位選択の順番が追えるようにしています。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .gp_component import constraint_probability, expected_improvement
from .model import HybridModel
from ..policies import RegionPolicy
from ..preprocessing import PreparedData, normalize_parameters
from ..parameter_space import make_axis, make_parameter_grid
from ..settings import (
    DIVERSITY_COVERAGE_REFERENCE,
    DIVERSITY_TOP_FRACTION,
    FAR_FROM_MEASUREMENTS_DISTANCE,
    MAX_DIVERSITY_WEIGHT,
    MIN_DIVERSITY_WEIGHT,
    MIN_RECOMMENDATION_FEASIBILITY,
    MIN_RECOMMENDATION_DISTANCE,
    ProblemDefinition,
)
from ..validation import ExperimentData, UserInputError


@dataclass(frozen=True)
class OptimizationResult:
    """探索処理からCLIとCSVへ渡す最終結果です。"""

    recommendations: list[dict[str, object]]
    requested_recommendation_count: int
    diversity_weight: float
    diversity_uncertainty_signal: float
    diversity_coverage_signal: float
    feasibility_guard_active: bool
    evaluated_candidate_count: int
    allowed_candidate_count: int
    forbidden_candidate_count: int
    observed_best: dict[str, float] | None
    had_feasible_observation: bool
    warnings: list[str]


# ---------------------------------------------------------------------------
# 探索候補を作る機能
# ---------------------------------------------------------------------------


def make_candidates(problem: ProblemDefinition) -> np.ndarray:
    """problem.csvの範囲と刻みから全候補の組み合わせを作ります。"""

    return make_parameter_grid(problem)


def remove_measured_candidates(
    candidates: np.ndarray,
    measured_parameters: np.ndarray,
    identity_columns: np.ndarray | None = None,
) -> np.ndarray:
    """既に測定した入力条件を、次実験の候補から除きます。"""

    candidate_identity = candidates if identity_columns is None else candidates[:, identity_columns]
    measured_identity = measured_parameters if identity_columns is None else measured_parameters[:, identity_columns]
    measured_keys = {tuple(np.round(row, 12)) for row in measured_identity}
    keep_rows = [tuple(np.round(row, 12)) not in measured_keys for row in candidate_identity]
    return candidates[np.array(keep_rows, dtype=bool)]


# ---------------------------------------------------------------------------
# 実測済みの最良条件を確認する機能
# ---------------------------------------------------------------------------


def observed_feasible_mask(
    prepared: PreparedData,
    problem: ProblemDefinition,
    region_policy: RegionPolicy | None = None,
) -> np.ndarray:
    """実測平均が全制約を満たし、現在の方針で使用できる条件を返します。"""

    feasible = np.ones(prepared.unique_condition_count, dtype=bool)
    for constraint in problem.constraints:
        values = prepared.result_means[constraint.column]
        assert constraint.target is not None
        if constraint.direction == "greater_equal":
            feasible &= values >= constraint.target
        else:
            feasible &= values <= constraint.target
    if region_policy is not None:
        feasible &= region_policy.evaluate(
            prepared.raw_parameters, [item.column for item in problem.parameters]
        ).allowed_mask
    return feasible


def find_observed_best(
    prepared: PreparedData,
    problem: ProblemDefinition,
    feasible: np.ndarray,
) -> dict[str, float] | None:
    """制約を満たす実測条件の中から、目的変数が最良の条件を返します。"""

    if not np.any(feasible):
        return None

    objective_values = prepared.result_means[problem.objective.column]
    feasible_indices = np.flatnonzero(feasible)
    if problem.objective.direction == "minimize":
        best_index = feasible_indices[np.argmin(objective_values[feasible])]
    else:
        best_index = feasible_indices[np.argmax(objective_values[feasible])]

    result: dict[str, float] = {}
    for column_index, parameter in enumerate(problem.parameters):
        result[parameter.column] = float(prepared.raw_parameters[best_index, column_index])
    for variable in problem.result_variables:
        result[variable.column] = float(prepared.result_means[variable.column][best_index])
    return result


# ---------------------------------------------------------------------------
# 候補を評価し、上位条件を選ぶ機能
# ---------------------------------------------------------------------------


def run_optimization(
    experiments: ExperimentData,
    prepared: PreparedData,
    problem: ProblemDefinition,
    recommendation_count: int,
    model: HybridModel,
    region_policy: RegionPolicy | None = None,
    candidate_grid: np.ndarray | None = None,
    observed_context_mask: np.ndarray | None = None,
    candidate_identity_columns: np.ndarray | None = None,
) -> OptimizationResult:
    """全候補を評価し、次実験の推薦条件を返します。"""

    parameter_names = [item.column for item in problem.parameters]
    all_candidates = make_candidates(problem) if candidate_grid is None else candidate_grid
    policy = region_policy or RegionPolicy()
    policy_result = policy.evaluate(all_candidates, parameter_names)
    forbidden_candidate_count = int(np.sum(~policy_result.allowed_mask))
    allowed_candidate_count = int(np.sum(policy_result.allowed_mask))
    raw_candidates = all_candidates[policy_result.allowed_mask]
    measured_parameters = prepared.raw_parameters
    if observed_context_mask is not None:
        measured_parameters = measured_parameters[np.asarray(observed_context_mask, dtype=bool)]
    raw_candidates = remove_measured_candidates(
        raw_candidates, measured_parameters, candidate_identity_columns
    )
    if len(raw_candidates) == 0:
        raise UserInputError("探索範囲内の全条件が測定済みです。未測定の候補がありません。")
    if len(raw_candidates) < recommendation_count:
        raise UserInputError(
            f"未測定候補は{len(raw_candidates):,}件ですが、"
            f"--nで{recommendation_count:,}件が指定されています。\n"
            "--nを未測定候補数以下にしてください。"
        )

    candidate_policy = policy.evaluate(raw_candidates, parameter_names)
    normalized_candidates = normalize_parameters(raw_candidates, problem)
    hybrid_prediction = model.predict(normalized_candidates)
    predictions: dict[str, dict[str, np.ndarray]] = {}
    for result in problem.result_variables:
        values = hybrid_prediction.results[result.column]
        predictions[result.column] = {
            "mean": values["hybrid_mean"],
            "std": values["hybrid_std"],
            "nn_pred": values["nn_pred"],
        }

    # 各制約の達成確率を掛け合わせます。制約がなければ常に1です。
    total_feasibility = np.ones(len(raw_candidates), dtype=float)
    for constraint in problem.constraints:
        assert constraint.target is not None
        probability = constraint_probability(
            mean=predictions[constraint.column]["mean"],
            std=predictions[constraint.column]["std"],
            direction=constraint.direction,
            target=constraint.target,
        )
        predictions[constraint.column]["probability"] = probability
        total_feasibility *= probability

    feasible_observed = observed_feasible_mask(prepared, problem, policy)
    if observed_context_mask is not None:
        context_mask = np.asarray(observed_context_mask, dtype=bool)
        if context_mask.shape != feasible_observed.shape:
            raise ValueError("observed_context_maskの形状が実測条件数と一致しません。")
        feasible_observed &= context_mask
    observed_best = find_observed_best(prepared, problem, feasible_observed)
    objective_prediction = predictions[problem.objective.column]

    if observed_best is not None:
        improvement = expected_improvement(
            mean=objective_prediction["mean"],
            std=objective_prediction["std"],
            best_observed=observed_best[problem.objective.column],
            direction=problem.objective.direction,
        )
        base_score = total_feasibility * improvement
    else:
        # 合格実測がまだない場合は、制約達成確率と不確かさを優先します。
        improvement = np.zeros(len(raw_candidates), dtype=float)
        uncertainty_scale = max(model.gp_models[problem.objective.column].y_scale, 1e-12)
        normalized_uncertainty = np.clip(objective_prediction["std"] / uncertainty_scale, 0.0, 1.0)
        base_score = total_feasibility * (0.25 + 0.75 * normalized_uncertainty)

    # preferredは科学的な獲得価値を置き換えず、有界な倍率だけを掛ける。
    score = base_score * candidate_policy.preferred_multiplier

    nearest_distance = nearest_measurement_distance(
        normalized_candidates,
        prepared.normalized_parameters,
    )

    preferred_candidates: np.ndarray | None = None
    best_score_index = int(np.argmax(score))
    if (
        observed_best is not None
        and problem.constraints
        and total_feasibility[best_score_index] < MIN_RECOMMENDATION_FEASIBILITY
    ):
        preferred_candidates = total_feasibility >= MIN_RECOMMENDATION_FEASIBILITY

    diversity_state = calculate_diversity_state(
        score=score,
        objective_std=objective_prediction["std"],
        objective_scale=model.gp_models[problem.objective.column].y_scale,
        nearest_distance=nearest_distance,
        force_narrow=preferred_candidates is not None,
    )

    selected_indices = select_diverse_top_candidates(
        normalized_candidates,
        score,
        recommendation_count,
        prepared.normalized_parameters,
        diversity_weight=diversity_state.weight,
        preferred_candidates=preferred_candidates,
    )

    recommendations: list[dict[str, object]] = []
    for rank, candidate_index in enumerate(selected_indices, start=1):
        recommendation: dict[str, object] = {
            "rank": rank,
            "selection_role": "BEST_SCORE" if rank == 1 else "DIVERSITY",
            "feasibility_probability": float(total_feasibility[candidate_index]),
            "expected_improvement": float(improvement[candidate_index]),
            "base_recommendation_score": float(base_score[candidate_index]),
            "recommendation_score": float(score[candidate_index]),
            "preferred_multiplier": float(
                candidate_policy.preferred_multiplier[candidate_index]
            ),
            "nearest_distance": float(nearest_distance[candidate_index]),
            "gp_support": float(hybrid_prediction.support[candidate_index]),
        }

        for column_index, parameter in enumerate(problem.parameters):
            recommendation[parameter.column] = float(raw_candidates[candidate_index, column_index])

        for result in problem.result_variables:
            recommendation[f"{result.column}_mean"] = float(
                predictions[result.column]["mean"][candidate_index]
            )
            recommendation[f"{result.column}_std"] = float(
                predictions[result.column]["std"][candidate_index]
            )
            recommendation[f"{result.column}_nn_pred"] = float(
                predictions[result.column]["nn_pred"][candidate_index]
            )
            if result.role == "constraint":
                recommendation[f"{result.column}_probability"] = float(
                    predictions[result.column]["probability"][candidate_index]
                )

        recommendation["recommendation_reason"] = make_recommendation_reason(
            feasibility=float(total_feasibility[candidate_index]),
            improvement=float(improvement[candidate_index]),
            nearest_distance=float(nearest_distance[candidate_index]),
            has_feasible_observation=observed_best is not None,
            has_constraints=bool(problem.constraints),
            is_diversity_candidate=rank > 1,
        )
        recommendations.append(recommendation)

    warnings = list(prepared.warnings)
    if forbidden_candidate_count:
        warnings.append(
            f"使用禁止範囲により{forbidden_candidate_count:,}候補を推薦対象から除外しました。"
        )
    if observed_best is None and problem.constraints:
        warnings.append(
            "制約を満たす実測条件がまだありません。推薦は改善候補ではなく探索候補として扱ってください。"
        )

    return OptimizationResult(
        recommendations=recommendations,
        requested_recommendation_count=recommendation_count,
        diversity_weight=diversity_state.weight,
        diversity_uncertainty_signal=diversity_state.uncertainty_signal,
        diversity_coverage_signal=diversity_state.coverage_signal,
        feasibility_guard_active=preferred_candidates is not None,
        evaluated_candidate_count=len(raw_candidates),
        allowed_candidate_count=allowed_candidate_count,
        forbidden_candidate_count=forbidden_candidate_count,
        observed_best=observed_best,
        had_feasible_observation=observed_best is not None,
        warnings=warnings,
    )


def nearest_measurement_distance(
    candidates: np.ndarray,
    measured: np.ndarray,
    chunk_size: int = 10_000,
) -> np.ndarray:
    """各候補から最も近い実測条件までの正規化距離を計算します。"""

    distances: list[np.ndarray] = []
    for start in range(0, len(candidates), chunk_size):
        stop = min(start + chunk_size, len(candidates))
        differences = candidates[start:stop, None, :] - measured[None, :, :]
        squared = np.sum(differences * differences, axis=2)
        distances.append(np.sqrt(np.min(squared, axis=1)))
    return np.concatenate(distances)


@dataclass(frozen=True)
class DiversityState:
    """モデル状態から決めた、今回の分散制御値です。"""

    weight: float
    uncertainty_signal: float
    coverage_signal: float


def calculate_diversity_state(
    score: np.ndarray,
    objective_std: np.ndarray,
    objective_scale: float,
    nearest_distance: np.ndarray,
    force_narrow: bool = False,
) -> DiversityState:
    """上位候補周辺の不確かさと未被覆距離から分散比率を自動決定します。

    全探索空間ではなく推薦スコア上位5%を見ることで、目的・制約上の有望領域が
    どの程度モデルに支持されているかを判断します。不確かで実測点から遠いほど
    分散を強め、十分観測されているほど有望度を優先します。
    """

    candidate_count = len(score)
    if candidate_count == 0:
        return DiversityState(MIN_DIVERSITY_WEIGHT, 0.0, 0.0)

    top_count = min(
        candidate_count,
        max(100, int(np.ceil(candidate_count * DIVERSITY_TOP_FRACTION))),
    )
    top_indices = np.argsort(-score, kind="stable")[:top_count]
    safe_scale = max(float(objective_scale), 1e-12)
    uncertainty_signal = float(
        np.clip(np.median(objective_std[top_indices]) / safe_scale, 0.0, 1.0)
    )
    coverage_signal = float(
        np.clip(
            np.median(nearest_distance[top_indices]) / DIVERSITY_COVERAGE_REFERENCE,
            0.0,
            1.0,
        )
    )

    if force_narrow:
        weight = MIN_DIVERSITY_WEIGHT
    else:
        state_signal = 0.5 * uncertainty_signal + 0.5 * coverage_signal
        weight = MIN_DIVERSITY_WEIGHT + (
            MAX_DIVERSITY_WEIGHT - MIN_DIVERSITY_WEIGHT
        ) * state_signal

    return DiversityState(
        weight=float(np.clip(weight, MIN_DIVERSITY_WEIGHT, MAX_DIVERSITY_WEIGHT)),
        uncertainty_signal=uncertainty_signal,
        coverage_signal=coverage_signal,
    )


def select_diverse_top_candidates(
    normalized_candidates: np.ndarray,
    score: np.ndarray,
    count: int,
    measured: np.ndarray,
    diversity_weight: float,
    preferred_candidates: np.ndarray | None = None,
) -> list[int]:
    """1位は最高点、2位以降は有望度と空間被覆を両立する点を選びます。

    分散候補は、実測点および先に選んだ推薦点からの最短距離を使う
    greedy maximin方式です。推薦スコアも幾何平均へ残すため、単に探索範囲の
    端を選ぶのではなく、改善・制約達成の価値がある未観測領域を優先します。
    """

    if count < 1:
        return []

    if preferred_candidates is None or not np.any(preferred_candidates):
        preferred = np.ones(len(normalized_candidates), dtype=bool)
    else:
        preferred = np.asarray(preferred_candidates, dtype=bool)

    # 1位は分散・制約ガードを加えず、元の推薦スコア最大点を選びます。
    # stableな最大値選択により、同点時も候補生成順で結果を再現できます。
    first_index = int(np.argmax(score))
    selected = [first_index]
    if count == 1:
        return selected

    maximum_score = float(np.max(score))
    if maximum_score > 0.0:
        normalized_score = np.clip(score / maximum_score, 0.0, 1.0)
    else:
        # 全候補の改善スコアが0の場合は、未観測領域の被覆だけで選びます。
        normalized_score = np.ones(len(score), dtype=float)

    available = np.ones(len(normalized_candidates), dtype=bool)
    available[first_index] = False
    maximum_distance = max(np.sqrt(normalized_candidates.shape[1]), 1e-12)

    # 候補×全実測点の距離配列を保持せず、最近傍距離だけを残します。
    # これにより候補上限20万件でもメモリ使用量を抑えられます。
    minimum_distance = nearest_measurement_distance(normalized_candidates, measured)
    distance_from_first = np.sqrt(
        np.sum(
            np.square(normalized_candidates - normalized_candidates[first_index]),
            axis=1,
        )
    )
    minimum_distance = np.minimum(minimum_distance, distance_from_first)

    while len(selected) < count:
        preferred_available = available & preferred
        candidate_pool = preferred_available if np.any(preferred_available) else available
        eligible = candidate_pool & (minimum_distance >= MIN_RECOMMENDATION_DISTANCE)
        if not np.any(eligible):
            # 厳密な最小距離では指定件数を満たせない場合だけ、距離条件を緩和します。
            eligible = candidate_pool

        distance_utility = np.clip(minimum_distance / maximum_distance, 0.0, 1.0)
        combined_utility = (
            np.power(normalized_score, 1.0 - diversity_weight)
            * np.power(distance_utility, diversity_weight)
        )
        combined_utility[~eligible] = -1.0

        next_index = int(np.argmax(combined_utility))
        selected.append(next_index)
        available[next_index] = False
        distance_from_selected = np.sqrt(
            np.sum(
                np.square(normalized_candidates - normalized_candidates[next_index]),
                axis=1,
            )
        )
        minimum_distance = np.minimum(minimum_distance, distance_from_selected)

    return selected


def make_recommendation_reason(
    feasibility: float,
    improvement: float,
    nearest_distance: float,
    has_feasible_observation: bool,
    has_constraints: bool,
    is_diversity_candidate: bool,
) -> str:
    """数値だけでは分かりにくい推薦理由を、日本語の短文にします。"""

    reasons: list[str] = []
    if is_diversity_candidate:
        reasons.append("探索空間を広くカバーするための分散候補")
    if has_constraints:
        if feasibility >= 0.8:
            reasons.append("制約を満たす可能性が高い")
        elif feasibility >= 0.5:
            reasons.append("制約境界を確認できる")
        else:
            reasons.append("制約達成の不確かさが大きい")

    if has_feasible_observation and improvement > 0:
        reasons.append("目的値の改善が期待できる")
    if not has_feasible_observation:
        reasons.append("制約を満たす領域を探すための候補")
    if nearest_distance >= FAR_FROM_MEASUREMENTS_DISTANCE:
        reasons.append("実測条件から離れているため慎重な確認が必要")
    if not reasons:
        reasons.append("推薦スコアが高い")
    return "。".join(reasons) + "。"
