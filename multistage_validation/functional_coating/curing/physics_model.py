"""Deterministic thermal-cure model with inherited film state.

The model integrates a lumped heat-up trajectory, Arrhenius cure and thermal
degradation, and residual-solvent escape.  Incoming dry-film defects and stress
affect final adhesion and appearance, which makes upstream errors propagate
without introducing arbitrary inter-stage penalties.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from multistage_validation.common import checked_outputs, sigmoid, validate_and_broadcast


CONTROL_BOUNDS = {
    "oven_temperature_c": (90.0, 180.0),
    "hold_time_min": (5.0, 65.0),
    "nip_pressure_mpa": (0.05, 0.60),
}

INCOMING_STATE_BOUNDS = {
    # Wide enough to accept every physically valid output from the declared
    # drying domain.  Poor states remain evaluable and fail final specifications
    # instead of being mistaken for an API/domain error.
    "incoming_dry_thickness_um": (15.0, 310.0),
    "incoming_residual_solvent_pct": (0.0, 70.0),
    "incoming_internal_stress_mpa": (0.0, 8.0),
    "incoming_drying_defect_index": (0.0, 1.0),
    "incoming_thickness_cv_fraction": (0.0, 0.40),
}

INPUT_BOUNDS = {**CONTROL_BOUNDS, **INCOMING_STATE_BOUNDS}

_AMBIENT_TEMPERATURE_C = 25.0
_GAS_CONSTANT_J_MOL_K = 8.314462618
_INTEGRATION_STEPS = 96


def _arrhenius_rate(
    temperature_k: np.ndarray,
    reference_temperature_k: float,
    reference_rate_per_min: float,
    activation_energy_j_mol: float,
) -> np.ndarray:
    exponent = activation_energy_j_mol / _GAS_CONSTANT_J_MOL_K * (
        1.0 / reference_temperature_k - 1.0 / temperature_k
    )
    return reference_rate_per_min * np.exp(np.clip(exponent, -60.0, 60.0))


def evaluate_model(
    oven_temperature_c: ArrayLike,
    hold_time_min: ArrayLike,
    nip_pressure_mpa: ArrayLike,
    incoming_dry_thickness_um: ArrayLike,
    incoming_residual_solvent_pct: ArrayLike,
    incoming_internal_stress_mpa: ArrayLike,
    incoming_drying_defect_index: ArrayLike,
    incoming_thickness_cv_fraction: ArrayLike,
) -> dict[str, np.ndarray]:
    """Evaluate final film cure, damage, adhesion, thickness, and defects."""

    x = validate_and_broadcast(
        {
            "oven_temperature_c": oven_temperature_c,
            "hold_time_min": hold_time_min,
            "nip_pressure_mpa": nip_pressure_mpa,
            "incoming_dry_thickness_um": incoming_dry_thickness_um,
            "incoming_residual_solvent_pct": incoming_residual_solvent_pct,
            "incoming_internal_stress_mpa": incoming_internal_stress_mpa,
            "incoming_drying_defect_index": incoming_drying_defect_index,
            "incoming_thickness_cv_fraction": incoming_thickness_cv_fraction,
        },
        INPUT_BOUNDS,
    )
    oven = x["oven_temperature_c"]
    hold = x["hold_time_min"]
    pressure = x["nip_pressure_mpa"]
    dry_thickness = x["incoming_dry_thickness_um"]
    residual_solvent = x["incoming_residual_solvent_pct"]
    incoming_stress = x["incoming_internal_stress_mpa"]
    incoming_defect = x["incoming_drying_defect_index"]
    incoming_cv = x["incoming_thickness_cv_fraction"]

    thickness_ratio = dry_thickness / 80.0
    thermal_time_constant_min = 0.9 + 2.8 * thickness_ratio**2
    midpoint_fraction = (
        np.arange(_INTEGRATION_STEPS, dtype=float) + 0.5
    ) / _INTEGRATION_STEPS
    time_min = hold[..., None] * midpoint_fraction
    temperature_c = _AMBIENT_TEMPERATURE_C + (
        oven[..., None] - _AMBIENT_TEMPERATURE_C
    ) * (1.0 - np.exp(-time_min / thermal_time_constant_min[..., None]))
    temperature_k = temperature_c + 273.15
    dt_min = hold / _INTEGRATION_STEPS

    cure_rate = _arrhenius_rate(
        temperature_k,
        reference_temperature_k=393.15,
        reference_rate_per_min=0.020,
        activation_energy_j_mol=62_000.0,
    )
    degradation_rate = _arrhenius_rate(
        temperature_k,
        reference_temperature_k=443.15,
        reference_rate_per_min=0.0022,
        activation_energy_j_mol=82_000.0,
    )
    solvent_escape_rate = _arrhenius_rate(
        temperature_k,
        reference_temperature_k=373.15,
        reference_rate_per_min=0.055,
        activation_energy_j_mol=36_000.0,
    ) / np.maximum(thickness_ratio[..., None] ** 1.45, 0.20)

    cure_exposure = np.sum(cure_rate, axis=-1) * dt_min
    degradation_exposure = np.sum(degradation_rate, axis=-1) * dt_min
    solvent_escape_exposure = np.sum(solvent_escape_rate, axis=-1) * dt_min
    cure_fraction = -np.expm1(-cure_exposure)
    degradation_fraction = -np.expm1(-degradation_exposure)
    final_residual_solvent_pct = residual_solvent * np.exp(-solvent_escape_exposure)

    terminal_temperature_c = _AMBIENT_TEMPERATURE_C + (
        oven - _AMBIENT_TEMPERATURE_C
    ) * (1.0 - np.exp(-hold / thermal_time_constant_min))

    # Rapid gel formation can trap vapor before it diffuses out.  Thickness and
    # temperature increase blister risk; nip pressure improves contact but very
    # high pressure can imprint a nonuniform incoming film.
    gel_before_escape = cure_exposure / np.maximum(
        cure_exposure + solvent_escape_exposure, 1.0e-12
    )
    blister_index = np.clip(
        (residual_solvent / 18.0)
        * gel_before_escape
        * sigmoid((terminal_temperature_c - 118.0) / 9.0)
        * np.maximum(thickness_ratio, 0.35),
        0.0,
        1.0,
    )
    pressure_contact = 0.90 + 0.10 * (1.0 - np.exp(-pressure / 0.13))
    excessive_pressure = sigmoid((pressure - 0.48) / 0.045) * incoming_cv
    thermal_relaxation = 1.0 - np.exp(
        -hold / (18.0 + 15.0 * np.maximum(thickness_ratio, 0.25))
    )
    cure_shrinkage = 0.022 * cure_fraction
    compaction = 0.006 * (1.0 - np.exp(-pressure / 0.18))
    final_thickness_um = dry_thickness * (1.0 - cure_shrinkage - compaction)

    final_internal_stress_mpa = np.clip(
        incoming_stress * (1.0 - 0.52 * thermal_relaxation)
        + 2.7 * cure_fraction * thickness_ratio * (1.0 - thermal_relaxation)
        + 1.4 * degradation_fraction,
        0.0,
        10.0,
    )
    final_defect_index = np.clip(
        0.58 * incoming_defect
        + 0.72 * blister_index
        + 0.52 * degradation_fraction
        + 0.30 * excessive_pressure,
        0.0,
        1.0,
    )

    thickness_factor = np.exp(-((final_thickness_um - 72.0) / 45.0) ** 2)
    solvent_penalty = np.exp(-0.055 * final_residual_solvent_pct)
    stress_penalty = np.exp(-0.12 * final_internal_stress_mpa)
    defect_penalty = np.exp(-1.6 * final_defect_index)
    bond_strength_mpa = (
        5.0
        + 60.0
        * np.maximum(cure_fraction, 0.0) ** 0.72
        * np.exp(-3.0 * degradation_fraction)
        * pressure_contact
        * (0.82 + 0.18 * thickness_factor)
        * solvent_penalty
        * stress_penalty
        * defect_penalty
    )

    dimensional_change_um = (
        dry_thickness - final_thickness_um
        + 0.22 * final_internal_stress_mpa
        + 0.45 * incoming_cv * dry_thickness
    )
    curing_energy_kj_m2 = (
        0.42
        * (oven - _AMBIENT_TEMPERATURE_C)
        * hold
        / 0.74
        + 1.8 * pressure * hold
    )

    return checked_outputs(
        {
            "bond_strength_mpa": bond_strength_mpa,
            "cure_fraction": cure_fraction,
            "degradation_fraction": degradation_fraction,
            "final_residual_solvent_pct": final_residual_solvent_pct,
            "final_thickness_um": final_thickness_um,
            "final_internal_stress_mpa": final_internal_stress_mpa,
            "final_defect_index": final_defect_index,
            "dimensional_change_um": dimensional_change_um,
            "blister_index": blister_index,
            "terminal_film_temperature_c": terminal_temperature_c,
            "curing_energy_kj_m2": curing_energy_kj_m2,
        },
        oven.shape,
    )
