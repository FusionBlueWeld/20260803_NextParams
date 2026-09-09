"""Physics-structured synthetic model for resin/adhesive thermal curing.

The model is deliberately a transparent benchmark, not a production cure
simulator.  It combines a first-order lumped thermal response with a
temperature-dependent first-order cure integral and a competing degradation
integral.  All calculations are deterministic and NumPy-only.
"""

from __future__ import annotations

from typing import Final

import numpy as np
from numpy.typing import ArrayLike


# These limits mirror problem.csv.  Keeping them here makes direct use of the
# truth function safe even when a caller does not first parse problem.csv.
OVEN_TEMPERATURE_BOUNDS: Final[tuple[float, float]] = (100.0, 180.0)
HOLD_TIME_BOUNDS: Final[tuple[float, float]] = (5.0, 65.0)
LAYER_THICKNESS_BOUNDS: Final[tuple[float, float]] = (0.10, 0.50)

# Transparent benchmark coefficients.  Times are minutes, temperatures are K
# in the Arrhenius expressions, and lengths are mm unless noted otherwise.
_AMBIENT_TEMPERATURE_C: Final[float] = 25.0
_THERMAL_TIME_CONSTANT_MIN_AT_1MM: Final[float] = 0.8
_THERMAL_THICKNESS_COEFFICIENT: Final[float] = 8.0
_CURE_REFERENCE_TEMPERATURE_K: Final[float] = 393.15  # 120 °C
_CURE_REFERENCE_RATE_PER_MIN: Final[float] = 0.018
_CURE_ACTIVATION_ENERGY_J_PER_MOL: Final[float] = 65_000.0
_DEGRADATION_REFERENCE_TEMPERATURE_K: Final[float] = 443.15  # 170 °C
_DEGRADATION_REFERENCE_RATE_PER_MIN: Final[float] = 0.003
_DEGRADATION_ACTIVATION_ENERGY_J_PER_MOL: Final[float] = 80_000.0
_GAS_CONSTANT_J_PER_MOL_K: Final[float] = 8.314462618
_INTEGRATION_STEPS: Final[int] = 192


def _as_finite_in_range(values: ArrayLike, name: str, bounds: tuple[float, float]) -> np.ndarray:
    """Convert an input to float and reject nonfinite or out-of-domain values."""

    try:
        array = np.asarray(values, dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must contain finite numeric values.") from error
    if np.any(~np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values.")
    lower, upper = bounds
    if np.any((array < lower) | (array > upper)):
        raise ValueError(f"{name} must be in the inclusive range [{lower}, {upper}].")
    return array


def _arrhenius_rate(
    temperature_k: np.ndarray,
    reference_temperature_k: float,
    reference_rate_per_min: float,
    activation_energy_j_per_mol: float,
) -> np.ndarray:
    """Return a reference-normalized Arrhenius rate in 1/min."""

    exponent = activation_energy_j_per_mol / _GAS_CONSTANT_J_PER_MOL_K * (
        1.0 / reference_temperature_k - 1.0 / temperature_k
    )
    return reference_rate_per_min * np.exp(np.clip(exponent, -60.0, 60.0))


def evaluate_model(
    oven_temperature_c: ArrayLike,
    hold_time_min: ArrayLike,
    layer_thickness_mm: ArrayLike,
) -> dict[str, np.ndarray]:
    """Evaluate cure, damage, bond strength, and peak temperature.

    Parameters
    ----------
    oven_temperature_c:
        Oven set point in degrees Celsius, inclusive range 100–180 °C.
    hold_time_min:
        Nominal hold duration in minutes, inclusive range 5–65 min.
    layer_thickness_mm:
        Fixed-material resin/adhesive layer thickness in millimetres, inclusive
        range 0.10–0.50 mm.

    Returns
    -------
    dict[str, numpy.ndarray]
        Arrays with the common NumPy broadcast shape.  Keys are
        ``bond_strength_mpa``, ``cure_fraction``, ``degradation_fraction``,
        and ``peak_temperature_c``.

    Notes
    -----
    Temperature follows ``T = T_ambient + (T_oven - T_ambient)
    (1-exp(-t/tau))``.  Cure and degradation use exact first-order conversion
    for the midpoint-integrated, time-varying Arrhenius rates.  This keeps the
    high-temperature strength loss smooth while retaining a clear numerical
    integration interpretation.
    """

    oven_c = _as_finite_in_range(oven_temperature_c, "oven_temperature_c", OVEN_TEMPERATURE_BOUNDS)
    hold_min = _as_finite_in_range(hold_time_min, "hold_time_min", HOLD_TIME_BOUNDS)
    thickness_mm = _as_finite_in_range(
        layer_thickness_mm, "layer_thickness_mm", LAYER_THICKNESS_BOUNDS
    )
    oven_c, hold_min, thickness_mm = np.broadcast_arrays(oven_c, hold_min, thickness_mm)

    # The thickness dependence is a benchmark lumped approximation of the
    # longer heat-up time of a thicker layer.  The quadratic form is a compact
    # surrogate for a characteristic diffusion time proportional to L^2.
    thickness_ratio = thickness_mm / LAYER_THICKNESS_BOUNDS[1]
    tau_min = _THERMAL_TIME_CONSTANT_MIN_AT_1MM + _THERMAL_THICKNESS_COEFFICIENT * thickness_ratio**2

    # Midpoint quadrature over each individual hold interval.  Appending the
    # interval axis costs little for the <= 1,053-point configured grid.
    midpoint_fraction = (np.arange(_INTEGRATION_STEPS, dtype=float) + 0.5) / _INTEGRATION_STEPS
    time_min = hold_min[..., None] * midpoint_fraction
    temperature_c = _AMBIENT_TEMPERATURE_C + (oven_c[..., None] - _AMBIENT_TEMPERATURE_C) * (
        1.0 - np.exp(-time_min / tau_min[..., None])
    )
    temperature_k = temperature_c + 273.15
    dt_min = hold_min / _INTEGRATION_STEPS

    cure_rate_per_min = _arrhenius_rate(
        temperature_k,
        _CURE_REFERENCE_TEMPERATURE_K,
        _CURE_REFERENCE_RATE_PER_MIN,
        _CURE_ACTIVATION_ENERGY_J_PER_MOL,
    )
    degradation_rate_per_min = _arrhenius_rate(
        temperature_k,
        _DEGRADATION_REFERENCE_TEMPERATURE_K,
        _DEGRADATION_REFERENCE_RATE_PER_MIN,
        _DEGRADATION_ACTIVATION_ENERGY_J_PER_MOL,
    )
    cure_exposure = np.sum(cure_rate_per_min, axis=-1) * dt_min
    degradation_exposure = np.sum(degradation_rate_per_min, axis=-1) * dt_min
    cure_fraction = -np.expm1(-cure_exposure)
    degradation_fraction = -np.expm1(-degradation_exposure)

    # A broad optimum around 0.24 mm represents a synthetic balance between
    # cohesive bulk and thin-film defects.  Cure raises strength; degradation
    # penalizes it continuously, which makes temperature/time nonmonotonic.
    thickness_factor = 0.83 + 0.17 * np.exp(-((thickness_mm - 0.24) / 0.16) ** 2)
    cure_gain = np.power(np.maximum(cure_fraction, 0.0), 0.72)
    damage_penalty = np.exp(-3.0 * degradation_fraction)
    bond_strength_mpa = 4.0 + 62.0 * thickness_factor * cure_gain * damage_penalty

    # The response is monotone during the hold, so the terminal temperature is
    # also the peak; exposing it is useful as a monitor without extra state.
    peak_temperature_c = _AMBIENT_TEMPERATURE_C + (oven_c - _AMBIENT_TEMPERATURE_C) * (
        1.0 - np.exp(-hold_min / tau_min)
    )

    return {
        "bond_strength_mpa": np.asarray(bond_strength_mpa, dtype=float),
        "cure_fraction": np.asarray(cure_fraction, dtype=float),
        "degradation_fraction": np.asarray(degradation_fraction, dtype=float),
        "peak_temperature_c": np.asarray(peak_temperature_c, dtype=float),
    }
