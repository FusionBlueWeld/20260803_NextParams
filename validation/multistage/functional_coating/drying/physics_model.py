"""Deterministic convective drying model for a deposited polymer film.

The reduced-order model combines lumped film heating, temperature-dependent
internal solvent diffusion, an external convective mass-transfer resistance,
and surface/core drying imbalance.  Incoming coating state is explicit so the
same model works both by itself and as the second stage of the line.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from validation.multistage.src.common import checked_outputs, sigmoid, validate_and_broadcast


CONTROL_BOUNDS = {
    "air_temperature_c": (50.0, 120.0),
    "air_speed_m_s": (0.5, 6.0),
    "residence_time_min": (1.0, 15.0),
}

INCOMING_STATE_BOUNDS = {
    "incoming_wet_thickness_um": (60.0, 270.0),
    "incoming_solids_fraction": (0.35, 0.65),
    "incoming_thickness_cv_fraction": (0.0, 0.35),
    "incoming_coating_defect_index": (0.0, 1.0),
}

INPUT_BOUNDS = {**CONTROL_BOUNDS, **INCOMING_STATE_BOUNDS}

_AMBIENT_TEMPERATURE_C = 25.0
_DRY_POLYMER_DENSITY_KG_M3 = 1250.0
_SOLVENT_DENSITY_KG_M3 = 790.0
_SOLVENT_LATENT_HEAT_J_KG = 8.5e5


def evaluate_model(
    air_temperature_c: ArrayLike,
    air_speed_m_s: ArrayLike,
    residence_time_min: ArrayLike,
    incoming_wet_thickness_um: ArrayLike,
    incoming_solids_fraction: ArrayLike,
    incoming_thickness_cv_fraction: ArrayLike,
    incoming_coating_defect_index: ArrayLike,
) -> dict[str, np.ndarray]:
    """Evaluate heat/mass transfer, dry-film state, and drying damage."""

    x = validate_and_broadcast(
        {
            "air_temperature_c": air_temperature_c,
            "air_speed_m_s": air_speed_m_s,
            "residence_time_min": residence_time_min,
            "incoming_wet_thickness_um": incoming_wet_thickness_um,
            "incoming_solids_fraction": incoming_solids_fraction,
            "incoming_thickness_cv_fraction": incoming_thickness_cv_fraction,
            "incoming_coating_defect_index": incoming_coating_defect_index,
        },
        INPUT_BOUNDS,
    )
    air_temperature = x["air_temperature_c"]
    air_speed = x["air_speed_m_s"]
    residence = x["residence_time_min"]
    wet_thickness_um = x["incoming_wet_thickness_um"]
    solids_fraction = x["incoming_solids_fraction"]
    incoming_cv = x["incoming_thickness_cv_fraction"]
    incoming_defect = x["incoming_coating_defect_index"]

    time_s = residence * 60.0
    wet_thickness_m = wet_thickness_um * 1.0e-6

    # The wet film and supporting web form a lumped thermal mass.  Thickness
    # increases the heat-up time; airflow increases h with a square-root trend.
    h_w_m2_k = 13.0 + 34.0 * np.sqrt(air_speed)
    wet_density_kg_m3 = 1.0 / (
        solids_fraction / _DRY_POLYMER_DENSITY_KG_M3
        + (1.0 - solids_fraction) / _SOLVENT_DENSITY_KG_M3
    )
    areal_heat_capacity_j_m2_k = 520.0 + wet_thickness_m * wet_density_kg_m3 * 1900.0
    thermal_time_constant_s = areal_heat_capacity_j_m2_k / h_w_m2_k
    terminal_film_temperature_c = _AMBIENT_TEMPERATURE_C + (
        air_temperature - _AMBIENT_TEMPERATURE_C
    ) * (1.0 - np.exp(-time_s / thermal_time_constant_s))
    mean_film_temperature_c = _AMBIENT_TEMPERATURE_C + (
        air_temperature - _AMBIENT_TEMPERATURE_C
    ) * (
        1.0
        - thermal_time_constant_s
        / np.maximum(time_s, 1.0e-12)
        * (1.0 - np.exp(-time_s / thermal_time_constant_s))
    )

    # Internal diffusion has a characteristic L^2 resistance.  External
    # transfer has a saturating airflow benefit and vapor-pressure drive.
    diffusivity_m2_s = 3.0e-12 * np.exp(
        0.035 * (mean_film_temperature_c - _AMBIENT_TEMPERATURE_C)
    )
    half_thickness_m = np.maximum(wet_thickness_m / 2.0, 1.0e-9)
    internal_rate_s = np.pi**2 * diffusivity_m2_s / (4.0 * half_thickness_m**2)
    vapor_drive = np.exp(0.028 * (mean_film_temperature_c - 70.0))
    external_rate_s = (
        0.0030
        + 0.0120 * (1.0 - np.exp(-air_speed / 1.45))
    ) * np.clip(vapor_drive, 0.25, 4.0)
    overall_rate_s = 1.0 / (
        1.0 / np.maximum(internal_rate_s, 1.0e-15)
        + 1.0 / np.maximum(external_rate_s, 1.0e-15)
    )
    remaining_solvent_fraction = np.exp(-overall_rate_s * time_s)

    wet_mass_kg_m2 = wet_thickness_m * wet_density_kg_m3
    dry_solids_kg_m2 = wet_mass_kg_m2 * solids_fraction
    initial_solvent_kg_m2 = wet_mass_kg_m2 * (1.0 - solids_fraction)
    remaining_solvent_kg_m2 = initial_solvent_kg_m2 * remaining_solvent_fraction
    evaporated_solvent_kg_m2 = initial_solvent_kg_m2 - remaining_solvent_kg_m2
    residual_solvent_pct = 100.0 * remaining_solvent_kg_m2 / (
        dry_solids_kg_m2 + remaining_solvent_kg_m2
    )

    solid_thickness_m = dry_solids_kg_m2 / _DRY_POLYMER_DENSITY_KG_M3
    solvent_thickness_m = remaining_solvent_kg_m2 / _SOLVENT_DENSITY_KG_M3
    dry_thickness_um = 1.0e6 * (solid_thickness_m + solvent_thickness_m)

    # Surface drying follows external transfer while the core follows the
    # series-limited rate.  Their conversion difference is a skinning measure.
    surface_removed = 1.0 - np.exp(-external_rate_s * time_s)
    bulk_removed = 1.0 - remaining_solvent_fraction
    skinning_index = np.clip(surface_removed - bulk_removed, 0.0, 1.0)
    thermal_aggression = sigmoid((terminal_film_temperature_c - 92.0) / 8.0)
    shrink_fraction = np.clip(
        (wet_thickness_um - dry_thickness_um) / wet_thickness_um, 0.0, 1.0
    )

    internal_stress_mpa = np.clip(
        0.15
        + 3.4 * skinning_index
        + 1.1 * shrink_fraction * thermal_aggression
        + 2.0 * incoming_cv,
        0.0,
        8.0,
    )
    drying_defect_index = np.clip(
        0.50 * incoming_defect
        + 0.52 * skinning_index
        + 0.18 * thermal_aggression * shrink_fraction
        + 0.65 * incoming_cv,
        0.0,
        1.0,
    )

    # Dimensional energy account per square metre: wet-film sensible heat,
    # latent evaporation, and a small cubic fan-work surrogate.
    sensible_kj_m2 = (
        wet_mass_kg_m2
        * 1900.0
        * np.maximum(terminal_film_temperature_c - _AMBIENT_TEMPERATURE_C, 0.0)
        / 0.72
        / 1000.0
    )
    latent_kj_m2 = evaporated_solvent_kg_m2 * _SOLVENT_LATENT_HEAT_J_KG / 0.78 / 1000.0
    fan_kj_m2 = 0.10 * air_speed**3 * time_s / 1000.0
    drying_energy_kj_m2 = sensible_kj_m2 + latent_kj_m2 + fan_kj_m2

    # Cross-web variation grows slightly as nonuniform wet regions dry at
    # different rates; it is retained as a connector to the curing stage.
    dry_thickness_cv_fraction = np.clip(
        incoming_cv * (1.0 + 0.55 * shrink_fraction)
        + 0.035 * skinning_index,
        0.0,
        0.40,
    )

    return checked_outputs(
        {
            "dry_thickness_um": dry_thickness_um,
            "residual_solvent_pct": residual_solvent_pct,
            "internal_stress_mpa": internal_stress_mpa,
            "drying_defect_index": drying_defect_index,
            "dry_thickness_cv_fraction": dry_thickness_cv_fraction,
            "terminal_film_temperature_c": terminal_film_temperature_c,
            "skinning_index": skinning_index,
            "drying_energy_kj_m2": drying_energy_kj_m2,
        },
        air_temperature.shape,
    )
