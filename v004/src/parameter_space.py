
"""GPとNNが共通して使うパラメータ空間を生成します。"""

from __future__ import annotations

import itertools

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


def make_parameter_grid(problem: ProblemDefinition) -> np.ndarray:
    """problem.csvの範囲と刻みから全交点を作ります。"""

    axes: list[list[float]] = []
    for parameter in problem.parameters:
        assert parameter.lower is not None
        assert parameter.upper is not None
        assert parameter.step is not None
        axes.append(make_axis(parameter.lower, parameter.upper, parameter.step))
    return np.array(list(itertools.product(*axes)), dtype=float)
