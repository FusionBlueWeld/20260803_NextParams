"""Hybrid内部で予測平均・標準偏差を与えるGaussian Process部品。"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ..settings import DEFAULT_NOISE_RATIO, GP_LENGTH_SCALE


class ModelTrainingError(Exception):
    """数値計算上の理由でモデルを作れなかった場合のエラーです。"""


@dataclass
class GaussianProcess:
    """1つの結果変数を予測するRBF Gaussian Processです。"""

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
        y_mean = float(np.mean(train_y))
        y_scale = float(np.std(train_y))
        if y_scale < 1e-12:
            y_scale = max(abs(y_mean) * DEFAULT_NOISE_RATIO, 1.0)

        normalized_y = (train_y - y_mean) / y_scale
        noise_ratio = max(noise_std / y_scale, 1e-6)
        kernel = rbf_kernel(train_x, train_x, GP_LENGTH_SCALE)
        kernel += (noise_ratio**2) * np.eye(len(train_x))
        cholesky = stable_cholesky(kernel)
        intermediate = np.linalg.solve(cholesky, normalized_y)
        alpha = np.linalg.solve(cholesky.T, intermediate)
        return cls(
            train_x,
            y_mean,
            y_scale,
            GP_LENGTH_SCALE,
            noise_ratio,
            cholesky,
            alpha,
        )

    def predict(self, query_x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        cross_kernel = rbf_kernel(query_x, self.train_x, self.length_scale)
        normalized_mean = cross_kernel @ self.alpha
        solved = np.linalg.solve(self.cholesky, cross_kernel.T)
        latent_variance = np.maximum(1.0 - np.sum(solved * solved, axis=0), 0.0)
        predictive_variance = latent_variance + self.noise_ratio**2
        mean = self.y_mean + self.y_scale * normalized_mean
        std = self.y_scale * np.sqrt(predictive_variance)
        return mean, std


def stable_cholesky(kernel: np.ndarray) -> np.ndarray:
    """小さなjitterを段階的に加え、安定してCholesky分解します。"""

    jitter = 1e-10
    for _ in range(8):
        try:
            return np.linalg.cholesky(kernel + jitter * np.eye(len(kernel)))
        except np.linalg.LinAlgError:
            jitter *= 10
    raise ModelTrainingError(
        "Gaussian Processの計算が安定しませんでした。入力条件の重複や値を確認してください。"
    )


def rbf_kernel(left: np.ndarray, right: np.ndarray, length_scale: float) -> np.ndarray:
    differences = left[:, None, :] - right[None, :, :]
    squared_distance = np.sum(differences * differences, axis=2)
    return np.exp(-0.5 * squared_distance / (length_scale**2))


def predict_in_chunks(
    model: GaussianProcess,
    candidates: np.ndarray,
    chunk_size: int = 10_000,
) -> tuple[np.ndarray, np.ndarray]:
    means: list[np.ndarray] = []
    standard_deviations: list[np.ndarray] = []
    for start in range(0, len(candidates), chunk_size):
        mean, std = model.predict(candidates[start : start + chunk_size])
        means.append(mean)
        standard_deviations.append(std)
    return np.concatenate(means), np.concatenate(standard_deviations)


def normal_pdf(value: np.ndarray) -> np.ndarray:
    return np.exp(-0.5 * value * value) / math.sqrt(2.0 * math.pi)


def normal_cdf(value: np.ndarray) -> np.ndarray:
    absolute = np.abs(value)
    t = 1.0 / (1.0 + 0.2316419 * absolute)
    polynomial = t * (
        0.319381530
        + t * (-0.356563782 + t * (1.781477937 + t * (-1.821255978 + t * 1.330274429)))
    )
    tail = normal_pdf(absolute) * polynomial
    positive_side = 1.0 - tail
    return np.where(value >= 0, positive_side, 1.0 - positive_side)


def constraint_probability(
    mean: np.ndarray,
    std: np.ndarray,
    direction: str,
    target: float,
) -> np.ndarray:
    safe_std = np.maximum(std, 1e-12)
    z_score = (
        (mean - target) / safe_std
        if direction == "greater_equal"
        else (target - mean) / safe_std
    )
    return np.clip(normal_cdf(z_score), 1e-12, 1.0)


def expected_improvement(
    mean: np.ndarray,
    std: np.ndarray,
    best_observed: float,
    direction: str,
) -> np.ndarray:
    safe_std = np.maximum(std, 1e-12)
    improvement = (
        best_observed - mean if direction == "minimize" else mean - best_observed
    )
    z_score = improvement / safe_std
    result = improvement * normal_cdf(z_score) + safe_std * normal_pdf(z_score)
    return np.maximum(result, 0.0)
