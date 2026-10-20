

"""探索候補を作り、次に試す条件を選ぶ機能。

このファイルが、v005のベイズ最適化の中心です。処理を小さな関数に分け、
候補生成、予測、点数計算、上位選択の順番が追えるようにしています。
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from .gp_component import constraint_probability, expected_improvement
from .model import HybridModel, HybridPrediction
from .efficient_selection import select_efficient_candidates
from ..candidate_identity import CandidateIdentity
from ..search_config import SearchConfig, effective_search_config
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
    top_score: float = 0.0
    useful_candidate_count: int = 0
    batch_reduction_reason: str = ""


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
    problem: ProblemDefinition | None = None,
    fixed_context: dict[str,float] | None = None,
) -> np.ndarray:
    """既に測定した入力条件を、次実験の候補から除きます。"""

    if problem is not None:
        # Current-context filtering happens before this call. Full identity still
        # retains fixed incoming columns and the declared axis numbers.
        return CandidateIdentity(problem,fixed_context).exclude_observed(candidates,measured_parameters)
    candidate_identity = candidates if identity_columns is None else candidates[:, identity_columns]
    measured_identity = measured_parameters if identity_columns is None else measured_parameters[:, identity_columns]
    measured_keys = {tuple(row) for row in measured_identity}
    keep_rows = [tuple(row) not in measured_keys for row in candidate_identity]
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
    search_config: SearchConfig | None = None,
    candidate_prediction: HybridPrediction | None = None,
    fixed_context: dict[str,float] | None = None,
    candidate_focus: dict | None = None,
    confirmation_slots: int = 0,
) -> OptimizationResult:
    """全候補を評価し、次実験の推薦条件を返します。"""

    config=effective_search_config(search_config or SearchConfig(),problem,experiments)
    parameter_names = [item.column for item in problem.parameters]
    all_candidates = make_candidates(problem) if candidate_grid is None else candidate_grid
    policy = region_policy or RegionPolicy()
    policy_result = policy.evaluate(all_candidates, parameter_names)
    forbidden_candidate_count = int(np.sum(~policy_result.allowed_mask))
    allowed_candidate_count = int(np.sum(policy_result.allowed_mask))
    measured_parameters = prepared.raw_parameters
    if observed_context_mask is not None:
        measured_parameters = measured_parameters[np.asarray(observed_context_mask, dtype=bool)]
    measured_mask=CandidateIdentity(problem,fixed_context).measured_mask(all_candidates,measured_parameters)
    candidate_mask=policy_result.allowed_mask & ~measured_mask
    raw_candidates=all_candidates[candidate_mask]
    if len(raw_candidates) == 0:
        raise UserInputError("探索範囲内の全条件が測定済みです。未測定の候補がありません。")

    candidate_policy = policy.evaluate(raw_candidates, parameter_names)
    normalized_candidates = normalize_parameters(raw_candidates, problem)
    if candidate_prediction is None:
        hybrid_prediction = model.predict(normalized_candidates)
    else:
        if len(candidate_prediction.support)!=len(all_candidates):
            raise ValueError("candidate_predictionはcandidate_gridと同じ行順・件数が必要です。")
        hybrid_prediction=HybridPrediction(candidate_prediction.support[candidate_mask],
            {name:{key:np.asarray(value)[candidate_mask] for key,value in values.items()}
             for name,values in candidate_prediction.results.items()})
    predictions: dict[str, dict[str, np.ndarray]] = {}
    for result in problem.result_variables:
        values = hybrid_prediction.results[result.column]
        predictions[result.column] = {
            "mean": values["hybrid_mean"],
            "std": values["hybrid_std"],
            "acquisition_std": values.get("acquisition_std", values["hybrid_std"]),
            "nn_pred": values["nn_pred"],
            "nn_std": values.get("nn_std", np.zeros_like(values["hybrid_mean"])),
            "raw_std": values.get("raw_hybrid_std", values["hybrid_std"]),
            "calibration_scale": values.get(
                "uncertainty_calibration_scale",
                np.ones_like(values["hybrid_mean"]),
            ),
        }

    # 各制約の達成確率を掛け合わせます。制約がなければ常に1です。
    total_feasibility = np.ones(len(raw_candidates), dtype=float)
    for constraint in problem.constraints:
        assert constraint.target is not None
        probability = constraint_probability(
            mean=predictions[constraint.column]["mean"],
            std=predictions[constraint.column]["std"],
            direction=constraint.direction,
            target=constraint_probability_target(constraint,config),
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
            std=objective_prediction["acquisition_std"],
            best_observed=observed_best[problem.objective.column],
            direction=problem.objective.direction,
        )
        base_score = total_feasibility * improvement
    else:
        # 合格実測がまだない場合は、制約達成確率と不確かさを優先します。
        improvement = np.zeros(len(raw_candidates), dtype=float)
        uncertainty_scale = max(model.gp_models[problem.objective.column].y_scale, 1e-12)
        normalized_uncertainty = np.clip(
            objective_prediction["acquisition_std"] / uncertainty_scale, 0.0, 1.0
        )
        base_score = total_feasibility * (0.25 + 0.75 * normalized_uncertainty)

    # preferredは科学的な獲得価値を置き換えず、有界な倍率だけを掛ける。
    score = base_score * candidate_policy.preferred_multiplier
    unweighted_score = score.copy()
    focus_main = np.ones(len(raw_candidates), dtype=bool)
    confirmation_roles = np.full(len(raw_candidates), '', dtype=object)
    if candidate_focus is not None:
        focus_main = np.asarray(candidate_focus['main_mask'], dtype=bool)[candidate_mask]
        confirmation_roles = np.asarray(candidate_focus['confirmation_roles'], dtype=object)[candidate_mask]
        score *= np.asarray(candidate_focus['weights'], dtype=float)[candidate_mask]
        # 確認専用点を改善枠へ混ぜず、別の根拠で選ぶ。
        score[~focus_main] = 0.

    nearest_distance = nearest_measurement_distance(
        normalized_candidates,
        prepared.normalized_parameters,
        memory_mb=config.distance_memory_mb,
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
        objective_std=objective_prediction["acquisition_std"],
        objective_scale=model.gp_models[problem.objective.column].y_scale,
        nearest_distance=nearest_distance,
        force_narrow=preferred_candidates is not None,
    )

    objective_scale=float(np.std(prepared.result_means[problem.objective.column]))
    if objective_scale<1e-12:
        objective_scale=max(abs(float(np.mean(prepared.result_means[problem.objective.column])))*0.05,1.0)
    confirmation_indices = []
    if candidate_focus is not None and confirmation_slots:
        eligible = (confirmation_roles != '') & np.isfinite(total_feasibility)
        eligible &= total_feasibility >= config.min_information_feasibility
        # 改善枠があるなら最低1件を残す。--nは確認を含む総件数の上限。
        keep_improvement = int(np.any(focus_main & (total_feasibility >= config.min_recommendation_feasibility)
                                     & (score / objective_scale >= config.min_normalized_improvement)))
        limit = min(confirmation_slots, max(0, recommendation_count - keep_improvement))
        priorities = {'FROZEN_CHECK': 3., 'SCREENING': 2., 'GLOBAL_CHECK': 4.}
        utility = np.array([priorities.get(role, 0.) for role in confirmation_roles])
        utility += np.clip(nearest_distance, 0., 1.)
        for _ in range(limit):
            if not np.any(eligible):
                break
            j = int(np.argmax(np.where(eligible, utility, -np.inf)))
            confirmation_indices.append(j)
            eligible[j] = False
        score[confirmation_indices] = 0.
    improvement_count = recommendation_count - len(confirmation_indices)
    search_indices = np.flatnonzero(focus_main)
    search_indices = search_indices[~np.isin(search_indices, confirmation_indices)]
    if improvement_count <= 0 or len(search_indices) == 0:
        selected_indices=[];roles=[];useful_count=0;top_score=0.;reduction='改善枠に候補がありません'
    elif config.adaptive_batch:
        selected=select_efficient_candidates(normalized_candidates[search_indices],score[search_indices],total_feasibility[search_indices],
            improvement[search_indices],objective_scale,nearest_distance[search_indices],objective_prediction['acquisition_std'][search_indices],
            hybrid_prediction.support[search_indices],improvement_count,config,
            has_feasible_observation=observed_best is not None,diversity_weight=diversity_state.weight)
        selected_indices=[int(search_indices[j]) for j in selected.indices];roles=selected.roles
        useful_count=selected.useful_count;top_score=selected.max_score;reduction=selected.reduction_reason
    else:
        local_indices = select_diverse_top_candidates(
            normalized_candidates[search_indices],score[search_indices],min(improvement_count,len(search_indices)),
            prepared.normalized_parameters,diversity_weight=diversity_state.weight,
            preferred_candidates=None if preferred_candidates is None else preferred_candidates[search_indices])
        selected_indices = [int(search_indices[j]) for j in local_indices]
        roles=['BEST_SCORE' if rank==0 else 'DIVERSITY' for rank in range(len(selected_indices))]
        useful_count=len(raw_candidates);top_score=float(np.max(score));reduction=''

    if candidate_focus is not None:
        # 固定確認はEI閾値に依存しない。改善枠との重複は除き、根拠をCSVに残す。
        for j in confirmation_indices:
            selected_indices.append(j)
            roles.append(str(confirmation_roles[j]))
        useful_count += len(confirmation_indices)
        # 配分重みで見かけ上スコアを下げ、停止根拠にしてはいけない。
        top_score = float(np.max(unweighted_score))

    recommendations: list[dict[str, object]] = []
    for rank, candidate_index in enumerate(selected_indices, start=1):
        recommendation: dict[str, object] = {
            "rank": rank,
            "selection_role": roles[rank-1],
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
            recommendation[f"{result.column}_acquisition_std"] = float(
                predictions[result.column]["acquisition_std"][candidate_index]
            )
            recommendation[f"{result.column}_nn_pred"] = float(
                predictions[result.column]["nn_pred"][candidate_index]
            )
            recommendation[f"{result.column}_nn_std"] = float(
                predictions[result.column]["nn_std"][candidate_index]
            )
            recommendation[f"{result.column}_raw_std"] = float(
                predictions[result.column]["raw_std"][candidate_index]
            )
            recommendation[f"{result.column}_calibration_scale"] = float(
                predictions[result.column]["calibration_scale"][candidate_index]
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
        if roles[rank-1]=='INFORMATION':
            recommendation['recommendation_reason']='制約境界・予測の不確かさを確認する情報収集候補。'+recommendation['recommendation_reason']
        elif roles[rank-1] in ('SCREENING', 'FROZEN_CHECK', 'GLOBAL_CHECK'):
            recommendation['recommendation_reason']='変数の作用・固定の妥当性を確認する実験。改善スコアとは別の選択基準。'
        recommendations.append(recommendation)

    warnings = list(prepared.warnings)
    if reduction:warnings.append(reduction)
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
        top_score=top_score,useful_candidate_count=useful_count,batch_reduction_reason=reduction,
    )


def nearest_measurement_distance(
    candidates: np.ndarray,
    measured: np.ndarray,
    chunk_size: int = 10_000,
    memory_mb: int = 16,
) -> np.ndarray:
    """各候補から最も近い実測条件までの正規化距離を計算します。"""

    if not len(candidates):return np.empty(0)
    if not len(measured):return np.full(len(candidates),np.inf)
    # Pairwise matrix only: 65k candidates × 124 observations × 50 inputs
    # previously formed hundreds of MB of temporary three-dimensional arrays.
    work_width=max(2*len(measured),len(measured)+2*candidates.shape[1])
    chunk_size=max(1,min(chunk_size,int(memory_mb*1024**2//(8*work_width))))
    squared_measured=np.einsum('ij,ij->i',measured,measured)
    result=np.empty(len(candidates))
    for start in range(0,len(candidates),chunk_size):
        chunk=candidates[start:start+chunk_size]
        squared=chunk @ measured.T
        squared *= -2
        squared += np.einsum('ij,ij->i',chunk,chunk)[:,None]
        squared += squared_measured[None,:]
        np.maximum(squared,0,out=squared)
        nearest=np.argmin(squared,axis=1)
        # Re-evaluate just the nearest pairs by subtraction to recover exact
        # zero distances and avoid cancellation in the inner-product formula.
        delta=chunk-measured[nearest]
        result[start:start+len(chunk)]=np.sqrt(np.einsum('ij,ij->i',delta,delta))
    return result


def constraint_probability_target(constraint,config):
    """Continuity correction only for explicitly declared ordinal steps."""
    step=config.ordinal_constraint_steps.get(constraint.column)
    if step is None:return float(constraint.target)
    target=float(constraint.target)
    ratio=target/step
    nearest=round(ratio)
    if abs(ratio-nearest)<=4*math.ulp(ratio):ratio=float(nearest)
    if constraint.direction=='less_equal':return np.floor(ratio)*step+0.5*step
    return np.ceil(ratio)*step-0.5*step


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
