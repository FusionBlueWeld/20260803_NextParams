"""複合工程の物理モデルが共有する入力・出力検査と数値処理。

入力名・有限性・範囲を確認し、同じ仮想製品を表す配列の形をそろえます。
モデル固有の物理式や工程接続はここには置きません。"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
from numpy.typing import ArrayLike


Bounds = Mapping[str, tuple[float, float]]


def validate_and_broadcast(
    values: Mapping[str, ArrayLike], bounds: Bounds
) -> dict[str, np.ndarray]:
    """入力名・有限性・範囲を検査し、同じ形のfloat配列へそろえます。"""

    if set(values) != set(bounds):
        missing = sorted(set(bounds) - set(values))
        extra = sorted(set(values) - set(bounds))
        raise ValueError(f"Input mismatch; missing={missing}, extra={extra}")
    converted: list[np.ndarray] = []
    for name in bounds:
        try:
            array = np.asarray(values[name], dtype=float)
        except (TypeError, ValueError) as error:
            raise ValueError(f"{name} must contain finite numeric values.") from error
        if np.any(~np.isfinite(array)):
            raise ValueError(f"{name} must contain only finite values.")
        lower, upper = bounds[name]
        if np.any((array < lower) | (array > upper)):
            raise ValueError(
                f"{name} must be in the inclusive range [{lower}, {upper}]."
            )
        converted.append(array)
    broadcast = np.broadcast_arrays(*converted)
    return {name: np.asarray(array, dtype=float) for name, array in zip(bounds, broadcast)}


def sigmoid(value: np.ndarray | float) -> np.ndarray:
    """指数のオーバーフローを抑えながら、滑らかな0〜1の遷移を計算します。"""

    return 1.0 / (1.0 + np.exp(-np.clip(value, -60.0, 60.0)))


def checked_outputs(
    outputs: Mapping[str, ArrayLike], shape: tuple[int, ...]
) -> dict[str, np.ndarray]:
    """全出力が入力と同じ形で有限値になっていることを確認します。"""

    checked: dict[str, np.ndarray] = {}
    for name, raw in outputs.items():
        values = np.asarray(raw, dtype=float)
        if values.shape != shape or np.any(~np.isfinite(values)):
            raise ValueError(f"Invalid output shape or value: {name}")
        checked[name] = values
    return checked
