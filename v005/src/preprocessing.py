
"""実験データを学習しやすい形へ整える機能。

同じ入力条件を複数回測定した場合は、入力条件ごとに結果の平均を計算します。
元データは変更せず、学習用の配列だけを新しく作ります。
"""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Mapping

import numpy as np

from .settings import DEFAULT_NOISE_RATIO, ProblemDefinition
from .validation import ExperimentData, UserInputError
from .candidate_identity import CandidateIdentity


@dataclass(frozen=True)
class PreparedData:
    """反復測定を集計し、学習用に準備したデータです。"""

    raw_parameters: np.ndarray
    normalized_parameters: np.ndarray
    result_means: dict[str, np.ndarray]
    result_noise_std: dict[str, float]
    repeat_counts: np.ndarray
    warnings: list[str]

    @property
    def unique_condition_count(self) -> int:
        return len(self.raw_parameters)


# ---------------------------------------------------------------------------
# パラメータを0～1へ変換する機能
# ---------------------------------------------------------------------------


def parameter_bounds(problem: ProblemDefinition) -> tuple[np.ndarray, np.ndarray]:
    """problem.csvの探索下限と上限をNumPy配列で返します。"""

    lower = np.array([item.lower for item in problem.parameters], dtype=float)
    upper = np.array([item.upper for item in problem.parameters], dtype=float)
    return lower, upper


def normalize_parameters(raw: np.ndarray, problem: ProblemDefinition) -> np.ndarray:
    """パラメータを探索範囲に対して0～1へそろえます。"""

    lower, upper = parameter_bounds(problem)
    return (raw - lower) / (upper - lower)


# ---------------------------------------------------------------------------
# 同一条件の反復測定をまとめる機能
# ---------------------------------------------------------------------------


def prepare_experiments(
    experiments: ExperimentData,
    problem: ProblemDefinition,
    fixed_context: Mapping[str, float] | None = None,
) -> PreparedData:
    """実験行を入力条件ごとにまとめ、平均と測定ばらつきを計算します。"""

    identity = CandidateIdentity(problem, fixed_context)
    grouped_indices = identity.group_indices(experiments.parameter_rows)
    canonical_rows = identity.canonicalize(experiments.parameter_rows)
    raw_parameters = canonical_rows[[indices[0] for indices in grouped_indices.values()]]
    repeat_counts = np.array([len(indices) for indices in grouped_indices.values()], dtype=int)

    if len(raw_parameters) < 2:
        raise UserInputError(
            "探索には異なる入力条件が2条件以上必要です。\n"
            "experiments.csvへ別の入力条件の実験結果を追加してください。"
        )

    result_means: dict[str, np.ndarray] = {}
    result_noise_std: dict[str, float] = {}

    for result in problem.result_variables:
        raw_results = np.array(experiments.result_rows[result.column], dtype=float)
        means: list[float] = []
        repeated_residuals: list[float] = []

        for indices in grouped_indices.values():
            group_values = raw_results[indices]
            group_mean = float(np.mean(group_values))
            means.append(group_mean)

            # 反復測定がある場合、平均との差を測定ノイズの参考にします。
            if len(group_values) >= 2:
                repeated_residuals.extend((group_values - group_mean).tolist())

        mean_array = np.array(means, dtype=float)
        result_means[result.column] = mean_array

        if repeated_residuals:
            noise_std = float(np.sqrt(np.mean(np.square(repeated_residuals))))
        else:
            # 反復測定がなければ、結果全体のばらつきの5%を既定値にします。
            overall_scale = float(np.std(mean_array))
            if overall_scale < 1e-12:
                overall_scale = max(abs(float(np.mean(mean_array))), 1.0)
            noise_std = DEFAULT_NOISE_RATIO * overall_scale
        result_noise_std[result.column] = max(noise_std, 1e-9)

    warnings = build_data_warnings(raw_parameters, repeat_counts, problem, fixed_context)

    return PreparedData(
        raw_parameters=raw_parameters,
        normalized_parameters=normalize_parameters(raw_parameters, problem),
        result_means=result_means,
        result_noise_std=result_noise_std,
        repeat_counts=repeat_counts,
        warnings=warnings,
    )


# ---------------------------------------------------------------------------
# データ量や探索範囲に関する注意を作る機能
# ---------------------------------------------------------------------------


def build_data_warnings(
    raw_parameters: np.ndarray,
    repeat_counts: np.ndarray,
    problem: ProblemDefinition,
    fixed_context: Mapping[str, float] | None = None,
) -> list[str]:
    """処理は続けられるものの、解釈に注意が必要な点を返します。"""

    warnings: list[str] = []
    parameter_count = sum(p.column not in (fixed_context or {}) for p in problem.parameters)
    unique_count = len(raw_parameters)
    recommended_count = max(5, parameter_count * 3)

    if parameter_count > 5:
        warnings.append(
            f"入力パラメータが{parameter_count}個あります。高次元では実測点から離れた候補が増えやすくなります。"
        )
    if unique_count < recommended_count:
        warnings.append(
            f"固有条件は{unique_count}件です。{parameter_count}パラメータでは"
            f"{recommended_count}件以上を目安に、結果を慎重に解釈してください。"
        )
    if int(np.max(repeat_counts)) == 1:
        warnings.append(
            "同一条件の反復測定がありません。予測の測定ノイズには既定値を使用します。"
        )

    observed_min = np.min(raw_parameters, axis=0)
    observed_max = np.max(raw_parameters, axis=0)
    for index, parameter in enumerate(problem.parameters):
        assert parameter.lower is not None
        assert parameter.upper is not None
        if parameter.lower < observed_min[index] or parameter.upper > observed_max[index]:
            warnings.append(
                f"{parameter.display_name}の探索範囲には、実測した最小値・最大値の外側が含まれます。"
            )

    return warnings
