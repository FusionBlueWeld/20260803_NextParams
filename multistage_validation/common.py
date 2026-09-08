"""Shared validation helpers for the multi-stage physics oracles."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
from numpy.typing import ArrayLike


Bounds = Mapping[str, tuple[float, float]]


def validate_and_broadcast(
    values: Mapping[str, ArrayLike], bounds: Bounds
) -> dict[str, np.ndarray]:
    """Validate an exact input contract and return broadcast float arrays."""

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
    """Numerically stable logistic transition."""

    return 1.0 / (1.0 + np.exp(-np.clip(value, -60.0, 60.0)))


def checked_outputs(
    outputs: Mapping[str, ArrayLike], shape: tuple[int, ...]
) -> dict[str, np.ndarray]:
    """Reject nonfinite values and accidental output broadcasting."""

    checked: dict[str, np.ndarray] = {}
    for name, raw in outputs.items():
        values = np.asarray(raw, dtype=float)
        if values.shape != shape or np.any(~np.isfinite(values)):
            raise ValueError(f"Invalid output shape or value: {name}")
        checked[name] = values
    return checked
