"""初期実験計画の診断と、有限候補集合からの適応的な設計推薦。"""

from __future__ import annotations

import math

import numpy as np

from .candidate_identity import CandidateIdentity
from .parameter_space import make_parameter_grid
from .settings import ProblemDefinition


def _controls(problem: ProblemDefinition, connection) -> list[int]:
    names = [item.column for item in problem.parameters]
    controls = set(names) if connection is None else set(connection.controls)
    return [index for index, name in enumerate(names) if name in controls]


def _matrix(raw, width: int) -> np.ndarray:
    if hasattr(raw, "parameter_rows"):
        raw = raw.parameter_rows
    array = np.asarray(raw, dtype=float)
    if array.size == 0:
        return array.reshape(0, width)
    if array.ndim != 2 or array.shape[1] != width:
        raise ValueError("実測入力の列数がproblemの入力数と一致しません。")
    return array


def _current_rows(raw, problem, connection) -> np.ndarray:
    matrix = _matrix(raw, len(problem.parameters))
    if connection is not None:
        names = [item.column for item in problem.parameters]
        for name, value in connection.current_incoming_context.items():
            matrix = matrix[matrix[:, names.index(name)] == value]
    if not len(matrix):
        return matrix
    identity = CandidateIdentity(problem, {} if connection is None else connection.current_incoming_context)
    canonical = identity.canonicalize(matrix)
    return np.unique(canonical, axis=0)


def _is_categorical(variable) -> bool:
    """明示的なカテゴリ単位だけを名義尺度として扱います。"""

    unit = str(variable.unit or "").strip().casefold()
    return unit in {"category", "categorical", "カテゴリ", "名義", "名義尺度"}


def _basis_size(variable) -> int:
    levels = min(4, _axis_count(variable))
    if _is_categorical(variable):
        return max(0, levels - 1)
    if levels < 2:
        return 0
    return 1 if levels == 2 else 2


def _design_matrix(rows: np.ndarray, problem: ProblemDefinition,
                   control_indices: list[int]) -> tuple[np.ndarray, int]:
    if not len(rows) or not control_indices:
        return np.ones((len(rows), 1)), 1
    columns = [np.ones(len(rows), dtype=float)]
    for index in control_indices:
        values = rows[:, index]
        distinct = np.unique(values)
        if len(distinct) < 2:
            continue
        variable = problem.parameters[index]
        if _is_categorical(variable):
            # カテゴリ間に順序や等間隔を仮定しない水準指示列を使います。
            columns.extend((values == level).astype(float) for level in distinct[1:])
            continue
        scale = float(np.ptp(values))
        centered = (values - float(np.mean(values))) / scale
        columns.append(centered)
        if len(distinct) >= 3:
            columns.append(centered * centered)
    design = np.column_stack(columns)
    return design, int(np.linalg.matrix_rank(design))


def design_target(problem: ProblemDefinition, connection, config) -> int:
    """操作入力数に応じた初期設計目標を、設定上限内で返します。"""

    dimension = len(_controls(problem, connection))
    target = max(config.initial_min_conditions, 2 * dimension + 4)
    # 4水準を均等に回す設計を想定して、目標を4件単位へそろえます。
    target = int(math.ceil(target / 4) * 4)
    finite_conditions = math.prod(_axis_count(problem.parameters[i]) for i in _controls(problem, connection))
    return min(target, config.initial_max_conditions, finite_conditions)


def design_diagnostics(raw, problem: ProblemDefinition, connection) -> dict:
    """現行の流入状態で、固有条件・水準・線形二次項の識別性を診断します。"""

    rows = _current_rows(raw, problem, connection)
    indices = _controls(problem, connection)
    design, rank = _design_matrix(rows, problem, indices)
    coverage: dict[str, dict[str, int]] = {}
    target_columns = 1
    for index in indices:
        variable = problem.parameters[index]
        distinct = int(np.unique(rows[:, index]).size) if len(rows) else 0
        span = float(np.ptp(rows[:, index]) / (variable.upper - variable.lower)) if len(rows) else 0.
        coverage[variable.column] = {"observed_levels": distinct, "target_levels": min(4, _axis_count(variable)),
                                     "observed_range_fraction": span}
        target_columns += _basis_size(variable)
    target_rows = max(12, 2 * len(indices) + 4)
    enough_coverage = all(value["observed_levels"] >= value["target_levels"] for value in coverage.values())
    enough_coverage &= all(_is_categorical(problem.parameters[index]) or coverage[problem.parameters[index].column]['observed_range_fraction'] >= .75 for index in indices)
    return {
        "current_context_unique_conditions": int(len(rows)),
        "level_coverage": coverage,
        "design_rank": rank,
        "design_columns": int(design.shape[1]),
        "target_rank": target_columns,
        "design_ready": bool(len(rows) >= target_rows and rank >= target_columns and enough_coverage),
    }


def _axis_count(parameter) -> int:
    count = int(math.floor((parameter.upper - parameter.lower) / parameter.step)) + 1
    last = parameter.lower + parameter.step * (count - 1)
    return count + int(parameter.upper - last > max(1e-10, abs(parameter.upper) * 1e-12))


def recommend_design(problem: ProblemDefinition, connection, region_policy, measured, count: int,
                     config, search_config, *, seed_offset: int = 0) -> np.ndarray:
    """全入力列を保ったまま、未測定の安全な設計条件を決定的に選びます。"""

    width = len(problem.parameters)
    if type(count) is not int or count < 0:
        raise ValueError("countは0以上の整数にしてください。")
    empty = np.empty((0, width), dtype=float)
    if count == 0 or not problem.parameters:
        return empty
    cap = min(int(config.max_screening_samples), int(search_config.max_candidates))
    if cap < 1:
        return empty
    pool = make_parameter_grid(problem, connection, max_candidates=cap,
                               seed=int(search_config.seed) + int(seed_offset), mode="budget")
    if not len(pool):
        return empty
    names = [item.column for item in problem.parameters]
    policy = region_policy.evaluate(pool, names)
    pool = pool[policy.allowed_mask]
    observed = _matrix(measured, width)
    identity = CandidateIdentity(problem, {} if connection is None else connection.current_incoming_context)
    pool = identity.exclude_observed(pool, observed)
    if not len(pool):
        return empty
    indices = _controls(problem, connection)
    if not indices:
        return pool[:min(count, len(pool))].copy()

    lower = np.array([problem.parameters[i].lower for i in indices], dtype=float)
    span = np.array([max(problem.parameters[i].upper - problem.parameters[i].lower, 1e-15) for i in indices])
    scaled_pool = (pool[:, indices] - lower) / span
    current = _current_rows(observed, problem, connection)
    scaled_known = ((current[:, indices] - lower) / span) if len(current) else np.empty((0, len(indices)))
    selected: list[int] = []
    rng = np.random.default_rng(int(search_config.seed) + int(seed_offset))
    tie_order = rng.permutation(len(pool))
    tie_rank = np.empty(len(pool), dtype=float)
    tie_rank[tie_order] = np.arange(len(pool))

    # 先に不足する格子水準を補い、その後は局所的な重複を避けて広く配置。
    level_sets = [np.unique(scaled_pool[:, j]) for j in range(len(indices))]
    observed_levels = [set(np.round(scaled_known[:, j], 12)) for j in range(len(indices))]
    chosen_levels = [set(values) for values in observed_levels]
    limit = min(count, len(pool))
    for _ in range(limit):
        coverage_gain = np.zeros(len(pool), dtype=float)
        for dimension in range(len(indices)):
            levels = level_sets[dimension]
            target = min(4, len(levels))
            if len(chosen_levels[dimension]) >= target:
                continue
            level_values = np.round(scaled_pool[:, dimension], 12)
            missing = set(np.round(levels, 12)) - chosen_levels[dimension]
            if missing:
                each_gain = 1.0 / max(1, len(missing))
                coverage_gain += np.fromiter((each_gain if value in missing else 0.0 for value in level_values), float, len(pool))
        reference = scaled_known if not selected else np.vstack((scaled_known, scaled_pool[selected]))
        if len(reference):
            # 二次元の候補×実測×入力テンソルを作らず、距離行列だけを使います。
            squared = (np.sum(scaled_pool * scaled_pool, axis=1)[:, None]
                       + np.sum(reference * reference, axis=1)[None, :]
                       - 2.0 * scaled_pool @ reference.T)
            squared = np.maximum(squared, 0.0)
            separation = np.sqrt(np.min(squared, axis=1)) / math.sqrt(len(indices))
        else:
            separation = np.zeros(len(pool))
        score = coverage_gain + 0.25 * separation + tie_rank * 1e-12
        if selected:
            score[selected] = -np.inf
        best = int(np.argmax(score))
        selected.append(best)
        for dimension in range(len(indices)):
            chosen_levels[dimension].add(round(float(scaled_pool[best, dimension]), 12))
    return pool[selected].copy()
