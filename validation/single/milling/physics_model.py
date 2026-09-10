"""Physics-structured synthetic model for high-dimensional face milling.

This module is a deterministic validation oracle, not a calibrated machining
model. It combines metric milling relationships with a small, explicit
multi-mode forced-response surrogate for arithmetic-average roughness.
"""

from __future__ import annotations

import numpy as np


# Keep these bounds synchronized with problem.csv. The oracle intentionally
# has no filesystem dependency and never writes data.
_INPUT_BOUNDS: dict[str, tuple[float, float]] = {
    "spindle_speed_rpm": (1800.0, 7200.0),
    "feed_per_tooth_mm": (0.040, 0.160),
    "axial_depth_mm": (0.50, 3.50),
    "radial_engagement_mm": (5.0, 20.0),
    "cutter_diameter_mm": (30.0, 60.0),
    "tooth_count": (2.0, 8.0),
    "nose_radius_mm": (0.40, 1.60),
    "tool_wear_mm": (0.0, 0.30),
}


def _ideal_feed_mark_ra_um(
    feed_per_tooth_mm: np.ndarray | float,
    nose_radius_mm: np.ndarray | float,
) -> np.ndarray:
    """Return the ideal arithmetic-average feed-mark roughness in µm.

    For a circular tool nose and small feed relative to the radius, the
    peak-to-valley height is approximately ``1000*fz**2/(8*r)``.  The
    corresponding arithmetic-average roughness is one quarter of that value.
    """

    feed = np.asarray(feed_per_tooth_mm, dtype=float)
    radius = np.asarray(nose_radius_mm, dtype=float)
    return 1000.0 * feed**2 / (32.0 * radius)


def _broadcast_and_validate(
    spindle_speed_rpm: np.ndarray | float,
    feed_per_tooth_mm: np.ndarray | float,
    axial_depth_mm: np.ndarray | float,
    radial_engagement_mm: np.ndarray | float,
    cutter_diameter_mm: np.ndarray | float,
    tooth_count: np.ndarray | float,
    nose_radius_mm: np.ndarray | float,
    tool_wear_mm: np.ndarray | float,
) -> tuple[np.ndarray, ...]:
    """Convert inputs to broadcast arrays and enforce problem.csv domains."""

    names = tuple(_INPUT_BOUNDS)
    arrays = np.broadcast_arrays(
        *[
            np.asarray(value, dtype=float)
            for value in (
                spindle_speed_rpm,
                feed_per_tooth_mm,
                axial_depth_mm,
                radial_engagement_mm,
                cutter_diameter_mm,
                tooth_count,
                nose_radius_mm,
                tool_wear_mm,
            )
        ]
    )
    validated: list[np.ndarray] = []
    for name, values in zip(names, arrays):
        lower, upper = _INPUT_BOUNDS[name]
        if np.any(~np.isfinite(values)):
            raise ValueError(f"{name} must contain only finite values.")
        # Inclusive decimal grids can be a few ulps outside an endpoint after
        # repeated binary floating-point additions, so allow only round-off.
        tolerance = 1.0e-12 * max(1.0, abs(lower), abs(upper))
        if np.any(values < lower - tolerance) or np.any(values > upper + tolerance):
            raise ValueError(f"{name} must be within [{lower:g}, {upper:g}].")
        validated.append(np.clip(values, lower, upper))
    return tuple(validated)


def evaluate_model(
    spindle_speed_rpm: np.ndarray | float,
    feed_per_tooth_mm: np.ndarray | float,
    axial_depth_mm: np.ndarray | float,
    radial_engagement_mm: np.ndarray | float,
    cutter_diameter_mm: np.ndarray | float,
    tooth_count: np.ndarray | float,
    nose_radius_mm: np.ndarray | float,
    tool_wear_mm: np.ndarray | float,
) -> dict[str, np.ndarray]:
    """Evaluate MRR, roughness, and spindle power on broadcastable inputs.

    Parameters
    ----------
    spindle_speed_rpm:
        Spindle speed, 1800--7200 rpm.
    feed_per_tooth_mm:
        Feed per tooth, 0.040--0.160 mm/tooth.
    axial_depth_mm:
        Axial depth of cut, 0.50--3.50 mm.
    radial_engagement_mm:
        Radial width of cut ``ae``, 5--20 mm.
    cutter_diameter_mm:
        Cutter diameter, 30--60 mm.
    tooth_count:
        Number of cutter teeth, 2--8. Grid values are integral tooth counts.
    nose_radius_mm:
        Effective nose radius, 0.40--1.60 mm.
    tool_wear_mm:
        Flank-wear surrogate, 0--0.30 mm.

    Returns
    -------
    dict[str, numpy.ndarray]
        ``material_removal_rate_cm3_min`` in cm³/min,
        ``roughness_ra_um`` in µm, and ``spindle_power_kw`` in kW. All output
        arrays have exactly the common broadcast shape, including ``()`` for
        scalar inputs.

    Notes
    -----
    The cutter and material are no longer fixed inputs: ``ae``, cutter
    diameter, tooth count, nose radius, and wear are explicit variables. A
    representative steel ``kc`` is modified smoothly by cutting speed, chip
    loading, engagement ratio, and wear. The roughness resonance term is a
    three-mode damped forced-response surrogate, not a regenerative-chatter
    stability-lobe calculation.
    """

    (
        speed,
        feed,
        axial_depth,
        radial_engagement,
        cutter_diameter,
        teeth,
        nose_radius,
        tool_wear,
    ) = _broadcast_and_validate(
        spindle_speed_rpm,
        feed_per_tooth_mm,
        axial_depth_mm,
        radial_engagement_mm,
        cutter_diameter_mm,
        tooth_count,
        nose_radius_mm,
        tool_wear_mm,
    )

    # Metric milling quantities. Q is mm³/min divided by 1000 to obtain
    # cm³/min, exactly as in the Sandvik metric milling relationship.
    material_removal_rate = (
        radial_engagement
        * axial_depth
        * feed
        * speed
        * teeth
        / 1000.0
    )
    cutting_speed_m_min = np.pi * cutter_diameter * speed / 1000.0
    engagement_ratio = radial_engagement / cutter_diameter

    # Representative steel specific cutting force. Speed, chip loading,
    # engagement, and wear are explicit synthetic modifiers of kc, so power is
    # coupled to all eight inputs rather than being only a copy of MRR.
    specific_cutting_force = 1850.0
    specific_cutting_force *= 1.0 + 0.055 * (cutting_speed_m_min / 600.0) ** 0.35
    specific_cutting_force *= 1.0 + 0.040 * (feed / 0.10) ** 0.15
    specific_cutting_force *= 1.0 + 0.060 * engagement_ratio
    specific_cutting_force *= 1.0 + 0.25 * (tool_wear / 0.30) ** 0.70
    spindle_power = (
        material_removal_rate * specific_cutting_force / 60000.0
    )

    # Arithmetic-average feed-mark geometry with wear and a small
    # cutting-depth/engagement contribution. Units: mm are converted to
    # micrometres by 1000. The 1/32 coefficient is Ra; 1/8 would be the
    # ideal peak-to-valley height Rt and must not be used for this output.
    wear_multiplier = 1.0 + 0.55 * tool_wear / 0.30
    geometric_ra = _ideal_feed_mark_ra_um(feed, nose_radius) * wear_multiplier
    ploughing_ra = (
        0.10
        + 0.055 * axial_depth
        + 0.10 * engagement_ratio
        + 0.05 * tool_wear / 0.30
    )

    # Tooth passing frequency with three explicit lightly damped modes. The
    # separated peaks intentionally produce nonmonotonic speed response.
    tooth_frequency_hz = speed * teeth / 60.0
    modal_frequency_hz = np.array([160.0, 300.0, 440.0])
    damping_ratio = np.array([0.065, 0.050, 0.065])
    modal_weight = np.array([0.36, 0.32, 0.24])
    frequency_ratio = tooth_frequency_hz[..., None] / modal_frequency_hz
    modal_transfer = 1.0 / np.sqrt(
        (1.0 - frequency_ratio**2) ** 2
        + (2.0 * damping_ratio * frequency_ratio) ** 2
    )
    forced_response = np.sum(modal_weight * modal_transfer, axis=-1)
    off_resonance_response = float(np.sum(modal_weight))

    load_index = (material_removal_rate / 80.0) ** 0.50
    load_index *= (specific_cutting_force / 2000.0) ** 0.30
    load_index *= 1.0 + 0.15 * engagement_ratio
    load_index *= 1.0 + 0.20 * tool_wear / 0.30
    dynamic_ra = (
        0.17
        * load_index
        * forced_response
        / off_resonance_response
    )
    roughness_ra = geometric_ra + ploughing_ra + dynamic_ra

    return {
        "material_removal_rate_cm3_min": np.asarray(
            material_removal_rate, dtype=float
        ),
        "roughness_ra_um": np.asarray(roughness_ra, dtype=float),
        "spindle_power_kw": np.asarray(spindle_power, dtype=float),
    }
