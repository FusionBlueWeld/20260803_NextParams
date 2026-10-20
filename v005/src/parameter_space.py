
"""GPとNNが共通して使うパラメータ空間を生成します。"""

from __future__ import annotations

import itertools
import math

import numpy as np

from .settings import ProblemDefinition


def make_axis(lower: float, upper: float, step: float) -> list[float]:
    """上下限を必ず含む1軸分の離散候補を作ります。"""

    count = int(np.floor((upper - lower) / step)) + 1
    values = [lower + step * index for index in range(count)]
    tolerance = max(1e-10, abs(upper) * 1e-12)
    if upper - values[-1] > tolerance:
        values.append(upper)
    else:
        values[-1] = upper
    return values


def _axis_count(parameter) -> int:
    count = int(math.floor((parameter.upper - parameter.lower) / parameter.step)) + 1
    last = parameter.lower + parameter.step * (count - 1)
    return count + int(parameter.upper - last > max(1e-10, abs(parameter.upper) * 1e-12))


def candidate_metadata(problem, connection=None, *, max_candidates=4096, mode="auto") -> dict:
    fixed = {} if connection is None else connection.current_incoming_context
    counts = [1 if p.column in fixed else _axis_count(p) for p in problem.parameters]
    total = math.prod(counts)
    exhaustive = total <= max_candidates and mode != "budget"
    return {"nominal_all_input_grid_count": problem.candidate_count,
            "control_grid_count": total, "fixed_incoming_context": dict(fixed),
            "max_candidates": max_candidates, "exhaustive": exhaustive,
            "candidate_mode": "grid" if exhaustive else "budget",
            "global_optimum_guaranteed": False}


def make_parameter_grid(problem: ProblemDefinition, connection=None, *,
                        max_candidates: int = 4096, seed: int = 42, mode: str = "auto") -> np.ndarray:
    """流入状態を先に固定。巨大な軸・直積を確保せず有限予算で点を作る。"""
    if max_candidates < 1 or mode not in ("auto", "grid", "budget"):
        raise ValueError("候補予算または生成方式が不正です。")
    fixed = {} if connection is None else connection.current_incoming_context
    counts = [1 if p.column in fixed else _axis_count(p) for p in problem.parameters]
    total = math.prod(counts)
    if mode == "grid" and total > max_candidates:
        raise ValueError(f"操作条件の格子は{total:,}点です。列挙前に候補予算{max_candidates:,}を超えました。")
    if total <= max_candidates and mode != "budget":
        indices = np.asarray(list(itertools.product(*[range(count) for count in counts])), dtype=float)
    else:
        rng = np.random.default_rng(seed)
        indices = np.column_stack([np.floor((rng.permutation(max_candidates) + rng.random(max_candidates))
                                           / max_candidates * count) for count in counts])
    result = np.empty_like(indices, dtype=float)
    for index, parameter in enumerate(problem.parameters):
        if parameter.column in fixed:
            result[:, index] = fixed[parameter.column]
        else:
            result[:, index] = np.where(indices[:, index] >= counts[index] - 1, parameter.upper,
                                         parameter.lower + parameter.step * indices[:, index])
    return np.unique(result, axis=0)
