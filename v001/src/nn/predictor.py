"""学習済みNNで共通グリッド全体を予測します。"""

from __future__ import annotations

import numpy as np

from ..preprocessing import normalize_parameters
from ..settings import ProblemDefinition
from .model import ResidualNeuralNetwork


def predict_response_space(
    model: ResidualNeuralNetwork,
    raw_grid: np.ndarray,
    problem: ProblemDefinition,
) -> np.ndarray:
    """生のパラメータグリッドを正規化し、元単位の予測値を返します。"""

    normalized_grid = normalize_parameters(raw_grid, problem)
    return model.predict(normalized_grid)
