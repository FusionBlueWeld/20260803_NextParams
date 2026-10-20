"""小規模NNの読出し学習と共通残差カーネルによる、費用上限付き工程模型。

隠れ層を固定して読出しを学習する。従来NNの知見損失経路とは別のbackendで、
この模型の標準偏差は校正済み確率ではない。全出力で同じ残差カーネルを使う。
"""

from __future__ import annotations

from dataclasses import dataclass
import time

import numpy as np


def squared_distances(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """候補×学習点×次元の三次元配列を作らない。"""
    return np.maximum(np.sum(left * left, axis=1)[:, None]
                      + np.sum(right * right, axis=1)[None, :] - 2 * left @ right.T, 0.0)


def triangular_solve(matrix: np.ndarray, values: np.ndarray, *, lower: bool = True) -> np.ndarray:
    """汎用solveによる再分解を避け、既知の三角行列だけを解く。"""
    result = np.array(values, dtype=float, copy=True)
    indices = range(len(matrix)) if lower else range(len(matrix) - 1, -1, -1)
    for index in indices:
        previous = slice(None, index) if lower else slice(index + 1, None)
        result[index] = (result[index] - matrix[index, previous] @ result[previous]) / matrix[index, index]
    return result


def _features(x, hidden_w, hidden_b, pairs):
    interactions = x[:, pairs[:, 0]] * x[:, pairs[:, 1]]
    return np.column_stack((np.ones(len(x)), x, x * x,
                            interactions, np.tanh(x @ hidden_w + hidden_b)))


def _input_transform(x):
    """観測入力の変動だけから尺度を決め、出力値には依存させません。"""
    return np.mean(x, axis=0), np.maximum(np.std(x, axis=0), 0.02)


@dataclass
class FastStageModel:
    train_x: np.ndarray
    output_mean: np.ndarray
    output_scale: np.ndarray
    hidden_w: np.ndarray
    hidden_b: np.ndarray
    pairs: np.ndarray
    readouts: np.ndarray
    cholesky: np.ndarray
    residual_alpha: np.ndarray
    residual_scale: np.ndarray
    error_floor: np.ndarray
    precision: np.ndarray
    support_radius: float
    length_scale: float
    feature_center: np.ndarray
    feature_scale: np.ndarray
    fit_seconds: float = 0.0

    @classmethod
    def fit(cls, x: np.ndarray, y: np.ndarray, *, seed: int = 0) -> "FastStageModel":
        started = time.perf_counter()
        n, dimensions = x.shape
        rng = np.random.default_rng(seed)
        # 物理boundsは保存したまま、実際に観測した変動の尺度で基底を作る。
        # 未変動の流入状態も0.02の床で扱い、ゼロ除算・過大な感度を避ける。
        feature_center, feature_scale = _input_transform(x)
        feature_x = (x - feature_center) / feature_scale
        mean, scale = np.mean(y, axis=0), np.std(y, axis=0)
        scale = np.maximum(scale, np.maximum(np.abs(mean) * 1e-6, 1e-8))
        normalized_y = (y - mean) / scale
        pairs = np.array([(i, j) for i in range(dimensions) for j in range(i + 1, dimensions)], dtype=int).reshape(-1, 2)
        if len(pairs) > 64:
            pairs = pairs[np.sort(rng.choice(len(pairs), 64, replace=False))]
        hidden_w = rng.normal(0, 1.5 / np.sqrt(dimensions), (3, dimensions, 32))
        hidden_b = rng.normal(0, 0.7, (3, 32))
        readouts, at_data = [], []
        for w, b in zip(hidden_w, hidden_b):
            features = _features(feature_x, w, b, pairs)
            # 出力軸をまとめて解き、NN ensembleを低い一定予算で作る。
            sample_weight = rng.uniform(0.7, 1.3, n)
            weighted = features * sample_weight[:, None]
            gram = features.T @ weighted
            gram += 0.03 * np.eye(gram.shape[0])
            factor = np.linalg.cholesky(gram)
            rhs = weighted.T @ normalized_y
            readout = triangular_solve(factor.T, triangular_solve(factor, rhs), lower=False)
            readouts.append(readout)
            at_data.append(features @ readout)
        nn_mean = np.mean(at_data, axis=0)
        residual = normalized_y - nn_mean
        kernel_distances = squared_distances(feature_x, feature_x)
        # 変換後距離のmedianを共通尺度とし、出力値やoracleへ合わせない。
        length_scale = max(0.35, float(np.sqrt(np.median(kernel_distances[np.triu_indices(n, 1)]))))
        kernel = np.exp(-0.5 * kernel_distances / length_scale ** 2)
        cholesky = np.linalg.cholesky(kernel + 0.01 * np.eye(n))
        alpha = triangular_solve(cholesky.T, triangular_solve(cholesky, residual), lower=False)
        corrected = nn_mean + kernel @ alpha
        error_floor = np.maximum(np.sqrt(np.mean((normalized_y - corrected) ** 2, axis=0)), 0.015)
        covariance = np.atleast_2d(np.cov(feature_x, rowvar=False))
        precision = np.linalg.solve(covariance + 0.015 * np.eye(dimensions), np.eye(dimensions))
        transformed = feature_x @ precision
        distances = np.maximum(np.sum(transformed * feature_x, axis=1)[:, None]
                               + np.sum(transformed * feature_x, axis=1)[None, :] - 2 * transformed @ feature_x.T, 0)
        np.fill_diagonal(distances, np.inf)
        radius = max(float(np.quantile(np.sqrt(np.min(distances, axis=1) / dimensions), 0.9)) * 1.3, 0.10)
        return cls(x, mean, scale, hidden_w, hidden_b, pairs, np.asarray(readouts),
                   cholesky, alpha, np.maximum(np.std(residual, axis=0), 0.03),
                   error_floor, precision, radius, length_scale, feature_center, feature_scale,
                   time.perf_counter() - started)

    def predict(self, x: np.ndarray, *, chunk_size: int = 256):
        means, deviations, supports = [], [], []
        feature_reference = (self.train_x - self.feature_center) / self.feature_scale
        reference = feature_reference @ self.precision
        reference_norm = np.sum(reference * feature_reference, axis=1)
        for start in range(0, len(x), chunk_size):
            query = (x[start:start + chunk_size] - self.feature_center) / self.feature_scale
            members = np.stack([_features(query, w, b, self.pairs) @ readout
                                for w, b, readout in zip(self.hidden_w, self.hidden_b, self.readouts)])
            transformed = query @ self.precision
            distances = np.maximum(np.sum(transformed * query, axis=1)[:, None]
                                   + reference_norm[None, :] - 2 * transformed @ feature_reference.T, 0)
            support = np.exp(-0.5 * np.min(distances, axis=1) / len(query.T) / self.support_radius ** 2)
            cross = np.exp(-0.5 * squared_distances(query, feature_reference) / self.length_scale ** 2)
            solved = triangular_solve(self.cholesky, cross.T)
            latent = np.maximum(1 - np.sum(solved * solved, axis=0), 0)
            normalized_mean = np.mean(members, axis=0) + support[:, None] * (cross @ self.residual_alpha)
            variance = np.var(members, axis=0, ddof=1) + latent[:, None] * self.residual_scale ** 2 + self.error_floor ** 2
            means.append(self.output_mean + self.output_scale * normalized_mean)
            deviations.append(self.output_scale * np.sqrt(variance))
            supports.append(support)
        columns = len(self.output_mean)
        return (np.vstack(means) if means else np.empty((0, columns)),
                np.vstack(deviations) if deviations else np.empty((0, columns)),
                np.concatenate(supports) if supports else np.empty(0))

    def arrays(self) -> dict[str, np.ndarray]:
        return {name: np.asarray(value) for name, value in vars(self).items()}

    @classmethod
    def from_arrays(cls, arrays: dict[str, np.ndarray]) -> "FastStageModel":
        fields = set(cls.__dataclass_fields__)
        transform_fields = {"feature_center", "feature_scale"}
        if set(arrays) not in (fields, fields - transform_fields):
            raise ValueError("高速stage模型の配列項目が一致しません。")
        values = dict(arrays)
        if not transform_fields <= set(values):
            # 旧模型のNNは2x-1、kernel/supportはx。等価変換で予測を保つ。
            if values["train_x"].ndim != 2:
                raise ValueError("高速stage模型の学習配列shapeが不正です。")
            dimensions = values["train_x"].shape[1]
            values["feature_center"] = np.full(dimensions, 0.5)
            values["feature_scale"] = np.full(dimensions, 0.5)
            values["length_scale"] = np.asarray(values["length_scale"]) * 2
            values["precision"] = np.asarray(values["precision"]) * 0.25
        for name in ("support_radius", "length_scale", "fit_seconds"):
            values[name] = float(values[name])
        values["pairs"] = values["pairs"].astype(int)
        return cls(**values)
