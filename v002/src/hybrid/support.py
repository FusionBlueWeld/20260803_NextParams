"""GP事後分散の減少量から、入力位置のデータ支持度を計算します。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..settings import (
    GP_LENGTH_SCALE,
    HYBRID_SUPPORT_MULTIPLIER,
    HYBRID_SUPPORT_NOISE,
)
from .gp_component import ModelTrainingError, rbf_kernel, stable_cholesky


@dataclass
class SupportModel:
    """全出力で共有する、入力空間だけに基づくGP支持度モデルです。"""

    train_x: np.ndarray
    cholesky: np.ndarray
    length_scale: float
    multiplier: float

    @classmethod
    def fit(cls, train_x: np.ndarray) -> "SupportModel":
        if len(train_x) < 1:
            raise ModelTrainingError("データ支持度の計算には実測条件が必要です。")
        kernel = rbf_kernel(train_x, train_x, GP_LENGTH_SCALE)
        kernel += HYBRID_SUPPORT_NOISE * np.eye(len(train_x))
        return cls(
            train_x=train_x,
            cholesky=stable_cholesky(kernel),
            length_scale=GP_LENGTH_SCALE,
            multiplier=HYBRID_SUPPORT_MULTIPLIER,
        )

    def predict(self, query_x: np.ndarray, chunk_size: int = 10_000) -> np.ndarray:
        values: list[np.ndarray] = []
        for start in range(0, len(query_x), chunk_size):
            query = query_x[start : start + chunk_size]
            cross = rbf_kernel(query, self.train_x, self.length_scale)
            solved = np.linalg.solve(self.cholesky, cross.T)
            reduction = np.sum(solved * solved, axis=0)
            values.append(np.clip(self.multiplier * reduction, 0.0, 1.0))
        return np.concatenate(values)
