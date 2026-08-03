
"""候補条件の価値を計算する機能。

制約を満たす確率と、目的変数が改善する期待値を組み合わせ、
「次に測定する価値」を数値にします。
"""

from __future__ import annotations

import math

import numpy as np


# ---------------------------------------------------------------------------
# 正規分布の確率計算
# ---------------------------------------------------------------------------


def normal_pdf(value: np.ndarray) -> np.ndarray:
    """標準正規分布の確率密度を返します。"""

    return np.exp(-0.5 * value * value) / math.sqrt(2.0 * math.pi)


def normal_cdf(value: np.ndarray) -> np.ndarray:
    """標準正規分布の累積確率を近似計算します。"""

    absolute = np.abs(value)
    t = 1.0 / (1.0 + 0.2316419 * absolute)
    polynomial = t * (
        0.319381530
        + t * (-0.356563782 + t * (1.781477937 + t * (-1.821255978 + t * 1.330274429)))
    )
    tail = normal_pdf(absolute) * polynomial
    positive_side = 1.0 - tail
    return np.where(value >= 0, positive_side, 1.0 - positive_side)


# ---------------------------------------------------------------------------
# 目的変数と制約の評価
# ---------------------------------------------------------------------------


def constraint_probability(
    mean: np.ndarray,
    std: np.ndarray,
    direction: str,
    target: float,
) -> np.ndarray:
    """予測分布が制約を満たす確率を返します。"""

    safe_std = np.maximum(std, 1e-12)
    if direction == "greater_equal":
        z_score = (mean - target) / safe_std
    else:
        z_score = (target - mean) / safe_std
    return np.clip(normal_cdf(z_score), 1e-12, 1.0)


def expected_improvement(
    mean: np.ndarray,
    std: np.ndarray,
    best_observed: float,
    direction: str,
) -> np.ndarray:
    """現在の実測最良値を、どの程度改善できそうか計算します。"""

    safe_std = np.maximum(std, 1e-12)
    if direction == "minimize":
        improvement = best_observed - mean
    else:
        improvement = mean - best_observed

    z_score = improvement / safe_std
    result = improvement * normal_cdf(z_score) + safe_std * normal_pdf(z_score)
    return np.maximum(result, 0.0)


