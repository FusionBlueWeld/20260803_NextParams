"""Physics-structured synthetic model for press blanking.

The model is intentionally a deterministic validation oracle, not a calibrated
production press-forming calculation.  It combines a perimeter--thickness--
shear-strength load estimate with smooth empirical corrections for clearance,
stroke rate, and blank-holder force.
"""

from __future__ import annotations

from typing import Final

import numpy as np
from numpy.typing import ArrayLike


# Bounds are duplicated from problem.csv so that the truth function is
# self-contained and does not perform file I/O during evaluation.
CLEARANCE_PCT_LOWER: Final[float] = 3.0
CLEARANCE_PCT_UPPER: Final[float] = 15.0
STROKE_SPEED_LOWER: Final[float] = 80.0
STROKE_SPEED_UPPER: Final[float] = 320.0
HOLDER_FORCE_LOWER: Final[float] = 20.0
HOLDER_FORCE_UPPER: Final[float] = 100.0

# Fixed process and material conditions.
SHEET_THICKNESS_M: Final[float] = 2.0e-3
SHEAR_STRENGTH_PA: Final[float] = 320.0e6
CUT_PERIMETER_M: Final[float] = 20.0e-3
FIXED_TOOL_WEAR_FRACTION: Final[float] = 0.20


def _broadcast_inputs(
    clearance_pct: ArrayLike,
    stroke_speed_mm_s: ArrayLike,
    blank_holder_force_kn: ArrayLike,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert three inputs to float arrays, broadcast them, and validate."""

    try:
        values = np.broadcast_arrays(
            np.asarray(clearance_pct, dtype=float),
            np.asarray(stroke_speed_mm_s, dtype=float),
            np.asarray(blank_holder_force_kn, dtype=float),
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("Inputs must be numeric and broadcastable.") from exc

    names_and_bounds = (
        ("clearance_pct", values[0], CLEARANCE_PCT_LOWER, CLEARANCE_PCT_UPPER),
        ("stroke_speed_mm_s", values[1], STROKE_SPEED_LOWER, STROKE_SPEED_UPPER),
        ("blank_holder_force_kn", values[2], HOLDER_FORCE_LOWER, HOLDER_FORCE_UPPER),
    )
    for name, value, lower, upper in names_and_bounds:
        if not np.all(np.isfinite(value)):
            raise ValueError(f"{name} must contain only finite values.")
        if np.any((value < lower) | (value > upper)):
            raise ValueError(f"{name} is outside [{lower}, {upper}].")
    return values[0], values[1], values[2]


def evaluate_model(
    clearance_pct: ArrayLike,
    stroke_speed_mm_s: ArrayLike,
    blank_holder_force_kn: ArrayLike,
) -> dict[str, np.ndarray]:
    """Evaluate press-blanking quality and load metrics.

    Parameters
    ----------
    clearance_pct:
        Die clearance as a percentage of the fixed 2.0 mm sheet thickness,
        bounded to 3--15 percent.
    stroke_speed_mm_s:
        Punch stroke speed in millimetres per second, bounded to 80--320.
    blank_holder_force_kn:
        Blank-holder force in kilonewtons, bounded to 20--100.

    Returns
    -------
    dict[str, numpy.ndarray]
        ``burr_height_mm`` (objective), ``peak_force_kn`` and
        ``flatness_error_mm`` (constraints).  Every array has the common
        broadcast shape; scalar inputs therefore return zero-dimensional
        NumPy arrays.

    Raises
    ------
    ValueError
        If inputs are not broadcastable, finite, or within ``problem.csv``'s
        inclusive bounds.
    """

    clearance, speed, holder = _broadcast_inputs(
        clearance_pct, stroke_speed_mm_s, blank_holder_force_kn
    )

    # Base shear load: F = perimeter * sheet thickness * shear strength.
    # Units are m * m * Pa = N, converted to kN.
    base_shear_force_kn = (
        CUT_PERIMETER_M * SHEET_THICKNESS_M * SHEAR_STRENGTH_PA / 1.0e3
    )

    clearance_deviation = (clearance - 8.0) / 5.0
    low_clearance_penalty = np.maximum(-clearance_deviation, 0.0)
    speed_fraction = (speed - STROKE_SPEED_LOWER) / (
        STROKE_SPEED_UPPER - STROKE_SPEED_LOWER
    )
    holder_fraction = (holder - HOLDER_FORCE_LOWER) / (
        HOLDER_FORCE_UPPER - HOLDER_FORCE_LOWER
    )

    # Experimental blanking studies show a weak decrease in peak force as
    # clearance increases, with a steeper confinement penalty only at very
    # small clearance. This monotone form replaces the former U-shaped load
    # correction, whose high-clearance branch had the wrong sign.
    clearance_load_factor = (
        1.0
        - 0.005 * (clearance - 8.0)
        + 0.18 * np.exp(-(clearance - CLEARANCE_PCT_LOWER) / 2.5)
    )
    rate_load_factor = 1.0 + 0.12 * speed_fraction + 0.03 * speed_fraction**2
    holder_friction_factor = 1.0 + 0.04 * holder_fraction
    holder_contact_load_kn = 0.014 * holder + 0.10 * holder_fraction**2
    peak_force_kn = (
        base_shear_force_kn
        * clearance_load_factor
        * rate_load_factor
        * holder_friction_factor
        + holder_contact_load_kn
    )

    # Flatness is dominated by fracture mismatch and is restrained by the
    # holder.  The residual and rate terms are empirical synthetic terms.
    clearance_distortion_mm = (
        0.020
        + 0.085 * clearance_deviation**2
        + 0.012 * low_clearance_penalty
    )
    rate_distortion_mm = 0.030 * speed_fraction**1.3
    holder_restraint = 1.0 + 0.90 * holder_fraction
    overholding_distortion_mm = 0.006 * holder_fraction**2
    flatness_error_mm = (
        0.018
        + (clearance_distortion_mm + rate_distortion_mm) / holder_restraint
        + overholding_distortion_mm
    )

    # Burr height increases with clearance over the experimentally supported
    # 4--15 percent range. A small penalty below 4 percent represents
    # secondary shear/fracture mismatch without asserting a universal 8
    # percent optimum. Speed and holder terms remain synthetic process terms.
    clearance_above_reference = clearance - 4.0
    very_low_clearance = np.maximum(4.0 - clearance, 0.0)
    burr_height_mm = (
        0.018
        + 0.010 * FIXED_TOOL_WEAR_FRACTION
        + 0.0040 * clearance_above_reference
        + 0.00065 * clearance_above_reference**2
        + 0.0040 * very_low_clearance**2
        + 0.004 * ((speed - 200.0) / 120.0) ** 2
        + 0.003 * ((holder - 60.0) / 40.0) ** 2
    )

    return {
        "burr_height_mm": np.asarray(burr_height_mm, dtype=float),
        "peak_force_kn": np.asarray(peak_force_kn, dtype=float),
        "flatness_error_mm": np.asarray(flatness_error_mm, dtype=float),
    }
