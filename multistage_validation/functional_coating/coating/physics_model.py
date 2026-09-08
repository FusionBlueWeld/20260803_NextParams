"""Deterministic knife/slot coating reduced-order physics model.

The model represents a pre-mixed polymer solution deposited on a moving web.
It uses capillary-number bead stability, gap-controlled volume deposition,
web-tension disturbance, and entrained-bubble carryover.  Coefficients are
transparent benchmark assumptions rather than a calibration to one machine.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from multistage_validation.common import checked_outputs, sigmoid, validate_and_broadcast


CONTROL_BOUNDS = {
    "coating_gap_um": (120.0, 300.0),
    "line_speed_m_min": (5.0, 30.0),
    "web_tension_n": (40.0, 160.0),
}

INCOMING_STATE_BOUNDS = {
    "incoming_viscosity_pa_s": (0.8, 3.0),
    "incoming_solids_fraction": (0.35, 0.65),
    "incoming_bubble_fraction": (0.0, 0.03),
}

INPUT_BOUNDS = {**CONTROL_BOUNDS, **INCOMING_STATE_BOUNDS}

_SURFACE_TENSION_N_M = 0.034
_DRY_POLYMER_DENSITY_KG_M3 = 1250.0
_SOLVENT_DENSITY_KG_M3 = 790.0


def evaluate_model(
    coating_gap_um: ArrayLike,
    line_speed_m_min: ArrayLike,
    web_tension_n: ArrayLike,
    incoming_viscosity_pa_s: ArrayLike,
    incoming_solids_fraction: ArrayLike,
    incoming_bubble_fraction: ArrayLike,
) -> dict[str, np.ndarray]:
    """Evaluate deposited film state and coating defects.

    All inputs broadcast to one shape.  The returned state can be passed
    directly to the drying model; no random terms or hidden mutable state are
    used.
    """

    x = validate_and_broadcast(
        {
            "coating_gap_um": coating_gap_um,
            "line_speed_m_min": line_speed_m_min,
            "web_tension_n": web_tension_n,
            "incoming_viscosity_pa_s": incoming_viscosity_pa_s,
            "incoming_solids_fraction": incoming_solids_fraction,
            "incoming_bubble_fraction": incoming_bubble_fraction,
        },
        INPUT_BOUNDS,
    )
    gap = x["coating_gap_um"]
    speed = x["line_speed_m_min"]
    tension = x["web_tension_n"]
    viscosity = x["incoming_viscosity_pa_s"]
    solids = x["incoming_solids_fraction"]
    bubbles = x["incoming_bubble_fraction"]

    web_speed_m_s = speed / 60.0
    capillary_number = viscosity * web_speed_m_s / _SURFACE_TENSION_N_M
    log_ca = np.log(np.maximum(capillary_number, 1.0e-12) / 8.0)

    # Gap determines the principal deposited volume.  Viscous drag raises the
    # transfer ratio weakly while excessive web tension thins the wet bead.
    transfer_ratio = (
        0.72
        + 0.075 * np.tanh(log_ca / 1.15)
        + 0.025 * np.log(viscosity / 1.6)
        - 0.00045 * (tension - 100.0)
    )
    transfer_ratio = np.clip(transfer_ratio, 0.58, 0.86)
    wet_thickness_um = gap * transfer_ratio * (1.0 - 0.35 * bubbles)

    # A finite coating window exists in capillary number.  Low values promote
    # ribbing/dewetting; high values entrain air.  Tension away from the machine
    # centre increases cross-web thickness variation.
    low_ca_instability = sigmoid((-0.55 - log_ca) / 0.42)
    high_ca_instability = sigmoid((log_ca - 1.35) / 0.42)
    tension_deviation = ((tension - 100.0) / 70.0) ** 2
    thickness_cv_fraction = np.clip(
        0.010
        + 0.050 * low_ca_instability
        + 0.060 * high_ca_instability
        + 0.022 * tension_deviation
        + 1.15 * bubbles,
        0.0,
        0.35,
    )

    coating_defect_index = np.clip(
        0.42 * low_ca_instability
        + 0.36 * high_ca_instability
        + 0.12 * tension_deviation
        + 5.0 * bubbles
        + 0.40 * thickness_cv_fraction,
        0.0,
        1.0,
    )

    wet_volume_m3_m2 = wet_thickness_um * 1.0e-6
    # Ideal volume additivity keeps the mass split consistent with the
    # downstream dry-polymer and solvent densities.
    wet_density_kg_m3 = 1.0 / (
        solids / _DRY_POLYMER_DENSITY_KG_M3
        + (1.0 - solids) / _SOLVENT_DENSITY_KG_M3
    )
    wet_mass_kg_m2 = wet_volume_m3_m2 * wet_density_kg_m3
    wet_solvent_g_m2 = 1000.0 * wet_mass_kg_m2 * (1.0 - solids)
    wet_solids_g_m2 = 1000.0 * wet_mass_kg_m2 * solids
    coated_solids_rate_kg_m_h = wet_solids_g_m2 * speed / 1000.0 * 60.0

    return checked_outputs(
        {
            "wet_thickness_um": wet_thickness_um,
            "wet_solvent_g_m2": wet_solvent_g_m2,
            "wet_solids_g_m2": wet_solids_g_m2,
            "solids_fraction": solids,
            "thickness_cv_fraction": thickness_cv_fraction,
            "coating_defect_index": coating_defect_index,
            "capillary_number": capillary_number,
            "coated_solids_rate_kg_m_h": coated_solids_rate_kg_m_h,
        },
        gap.shape,
    )
