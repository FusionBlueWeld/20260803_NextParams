"""Gaussian Processによる予測を担当する機能。

外部の機械学習ライブラリへ計算を隠さず、v000で必要なRBFカーネルの
Gaussian ProcessだけをNumPyで実装しています。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .settings import DEFAULT_NOISE_RATIO, GP_LENGTH_SCALE


class ModelTrainingError(Exception):
    """数値計算上の理由でモデルを作れなかった場合のエラーです。"""


@dataclass
class GaussianProcess:
    """1つの検査結果を予測するGaussian Processです。"""

    train_x: np.ndarray
    y_mean: float
    y_scale: float
    length_scale: float
    noise_ratio: float
    cholesky: np.ndarray
    alpha: np.ndarray

    @classmethod
    def fit(
        cls,
        train_x: np.ndarray,
        train_y: np.ndarray,
        noise_std: float,
    ) -> "GaussianProcess":
        """実測入力と実測結果からモデルを作ります。"""

        y_mean = float(np.mean(train_y))
        y_scale = float(np.std(train_y))
        if y_scale < 1e-12:
            y_scale = max(abs(y_mean) * DEFAULT_NOISE_RATIO, 1.0)

        normalized_y = (train_y - y_mean) / y_scale
        noise_ratio = max(noise_std / y_scale, 1e-6)

        kernel = rbf_kernel(train_x, train_x, GP_LENGTH_SCALE)
        kernel += (noise_ratio**2) * np.eye(len(train_x))

        # ほぼ同じ条件が多い場合にも計算できるよう、小さな値を段階的に加えます。
        jitter = 1e-10
        cholesky: np.ndarray | None = None
        for _ in range(8):
            try:
                cholesky = np.linalg.cholesky(kernel + jitter * np.eye(len(train_x)))
                break
            except np.linalg.LinAlgError:
                jitter *= 10

        if cholesky is None:
            raise ModelTrainingError(
                "Gaussian Processの計算が安定しませんでした。入力条件の重複や値を確認してください。"
            )

        # K^-1 yを直接逆行列で求めず、三角行列を2回解いて安定させます。
        intermediate = np.linalg.solve(cholesky, normalized_y)
        alpha = np.linalg.solve(cholesky.T, intermediate)

        return cls(
            train_x=train_x,
            y_mean=y_mean,
            y_scale=y_scale,
            length_scale=GP_LENGTH_SCALE,
            noise_ratio=noise_ratio,
            cholesky=cholesky,
            alpha=alpha,
        )

    def predict(self, query_x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """候補条件ごとの予測平均と予測標準偏差を返します。"""

        cross_kernel = rbf_kernel(query_x, self.train_x, self.length_scale)
        normalized_mean = cross_kernel @ self.alpha

        solved = np.linalg.solve(self.cholesky, cross_kernel.T)
        latent_variance = np.maximum(1.0 - np.sum(solved * solved, axis=0), 0.0)

        # 次の実験結果を予測するため、モデルだけでなく測定ノイズも含めます。
        predictive_variance = latent_variance + self.noise_ratio**2

        mean = self.y_mean + self.y_scale * normalized_mean
        std = self.y_scale * np.sqrt(predictive_variance)
        return mean, std


def rbf_kernel(left: np.ndarray, right: np.ndarray, length_scale: float) -> np.ndarray:
    """2つの点の集まりから、距離に応じた類似度を計算します。"""

    differences = left[:, None, :] - right[None, :, :]
    squared_distance = np.sum(differences * differences, axis=2)
    return np.exp(-0.5 * squared_distance / (length_scale**2))
