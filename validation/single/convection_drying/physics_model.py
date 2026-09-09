"""Deterministic, physics-structured model for hot-air coating drying.

The model is deliberately a synthetic process surrogate.  It preserves useful
engineering structure (heat-up, vapor-pressure driving force, internal and
external mass-transfer resistances, and positive energy accounting) without
claiming calibrated production accuracy.
"""

from __future__ import annotations

import numpy as np


# The fixed film is a representative waterborne polymer coating on a flat
# metal web.  Moisture is declared on a wet basis throughout the outputs.
FILM_THICKNESS_M = 3.5e-4
DRY_SOLIDS_MASS_KG_M2 = 5.0e-2
INITIAL_MOISTURE_WET_BASIS = 0.28
INITIAL_WATER_MASS_KG_M2 = (
    DRY_SOLIDS_MASS_KG_M2
    * INITIAL_MOISTURE_WET_BASIS
    / (1.0 - INITIAL_MOISTURE_WET_BASIS)
)
INITIAL_WET_MASS_KG_M2 = DRY_SOLIDS_MASS_KG_M2 + INITIAL_WATER_MASS_KG_M2


def _saturation_pressure_pa(temperature_c: np.ndarray) -> np.ndarray:
    """Return water saturation pressure in Pa using an Antoine form."""

    # Antoine coefficients for water over the process-temperature range;
    # pressure is returned in Pa after the conventional mmHg conversion.
    pressure_mmhg = 10.0 ** (
        8.07131 - 1730.63 / (233.426 + temperature_c)
    )
    return pressure_mmhg * 133.322


def _sigmoid(value: np.ndarray) -> np.ndarray:
    """Numerically stable logistic transition."""

    return 1.0 / (1.0 + np.exp(-np.clip(value, -60.0, 60.0)))


def evaluate_model(
    air_temperature_c: float | np.ndarray,
    air_speed_m_s: float | np.ndarray,
    residence_time_min: float | np.ndarray,
) -> dict[str, np.ndarray]:
    """Evaluate the synthetic hot-air drying process.

    Parameters
    ----------
    air_temperature_c:
        Dry-bulb air temperature in degrees Celsius, inclusive range 45--105.
    air_speed_m_s:
        Air velocity over the film in m/s, inclusive range 0.5--6.0.
    residence_time_min:
        Residence time in minutes, inclusive range 1--12.

    Returns
    -------
    dict[str, numpy.ndarray]
        Arrays with one common broadcast shape and keys
        ``energy_kj_m2``, ``residual_moisture_pct``, and ``defect_index``.
        Residual moisture is wet-basis mass percent.  Defect index is bounded
        to [0, 1] and represents surface/core drying imbalance.

    Raises
    ------
    ValueError
        If an input is non-finite or outside the declared inclusive bounds.
    """

    temperature, speed, residence = np.broadcast_arrays(
        np.asarray(air_temperature_c, dtype=float),
        np.asarray(air_speed_m_s, dtype=float),
        np.asarray(residence_time_min, dtype=float),
    )

    for name, values, lower, upper in (
        ("air_temperature_c", temperature, 45.0, 105.0),
        ("air_speed_m_s", speed, 0.5, 6.0),
        ("residence_time_min", residence, 1.0, 12.0),
    ):
        if not np.all(np.isfinite(values)):
            raise ValueError(f"{name} must contain only finite values.")
        if np.any((values < lower) | (values > upper)):
            raise ValueError(
                f"{name} must be within the inclusive range "
                f"[{lower}, {upper}]."
            )

    shape = temperature.shape
    time_s = residence * 60.0
    air_temperature_k = temperature + 273.15
    reference_temperature_c = 25.0

    # Heat-up: a convective coefficient with a saturating velocity benefit.
    # The coefficient is an empirical flat-web surrogate in W/(m2 K).
    h_w_m2_k = 12.0 + 42.0 * (1.0 - np.exp(-speed / 1.25))
    # Effective film + web thermal inertia, not the wet coating mass used below.
    # This lumped warm-up surrogate is not a closed full-dryer heat balance.
    coating_heat_capacity_j_m2_k = 0.37 * 1700.0
    surface_warmup = 1.0 - np.exp(
        -h_w_m2_k * time_s / coating_heat_capacity_j_m2_k
    )
    surface_temperature_c = reference_temperature_c + (
        temperature - reference_temperature_c
    ) * surface_warmup
    mean_temperature_c = 0.55 * surface_temperature_c + 0.45 * temperature

    # Internal diffusion uses an Arrhenius-like temperature dependence.  It is
    # intentionally bounded to remain a smooth synthetic surrogate.
    diffusivity_m2_s = 7.0e-12 * np.exp(
        0.042 * (mean_temperature_c - reference_temperature_c)
    )
    internal_rate_s = np.pi**2 * diffusivity_m2_s / (
        4.0 * FILM_THICKNESS_M**2
    )

    # External transfer: Sherwood/Reynolds/Schmidt structure plus a fixed
    # surface-exchange factor representing the porous coating interface.
    rho_air_kg_m3 = 1.19 * 298.15 / air_temperature_k
    viscosity_pa_s = 1.85e-5 * (air_temperature_k / 298.15) ** 0.72
    vapor_diffusivity_m2_s = 2.45e-5 * (air_temperature_k / 298.15) ** 1.81
    characteristic_length_m = 1.2e-2
    reynolds = rho_air_kg_m3 * speed * characteristic_length_m / viscosity_pa_s
    schmidt = viscosity_pa_s / (rho_air_kg_m3 * vapor_diffusivity_m2_s)
    sherwood = 2.0 + 0.90 * np.sqrt(reynolds) * schmidt ** (1.0 / 3.0)
    mass_transfer_m_s = sherwood * vapor_diffusivity_m2_s / characteristic_length_m
    surface_exchange_factor = 0.12
    relative_humidity = 0.35
    vapor_pressure_driving_pa = np.maximum(
        (1.0 - relative_humidity)
        * _saturation_pressure_pa(mean_temperature_c),
        0.0,
    )
    vapor_density_driving_kg_m3 = (
        vapor_pressure_driving_pa * 0.01801528
        / (8.314462618 * (mean_temperature_c + 273.15))
    )
    external_flux_kg_m2_s = (
        surface_exchange_factor
        * mass_transfer_m_s
        * vapor_density_driving_kg_m3
    )
    external_rate_s = external_flux_kg_m2_s / INITIAL_WATER_MASS_KG_M2

    # Series resistance: 1/k_overall = 1/k_internal + 1/k_external.
    overall_rate_s = 1.0 / (
        1.0 / np.maximum(internal_rate_s, 1.0e-15)
        + 1.0 / np.maximum(external_rate_s, 1.0e-15)
    )
    remaining_fraction = np.exp(-overall_rate_s * time_s)
    remaining_water_kg_m2 = INITIAL_WATER_MASS_KG_M2 * remaining_fraction
    residual_moisture_pct = 100.0 * remaining_water_kg_m2 / (
        DRY_SOLIDS_MASS_KG_M2 + remaining_water_kg_m2
    )

    # A fast surface layer is compared with the series-resistance bulk result.
    # This creates a bounded, deterministic defect surrogate, not a measured
    # defect probability.
    aggression = 0.58 * _sigmoid((mean_temperature_c - 73.0) / 8.0) + 0.42 * (
        1.0 - np.exp(-speed / 1.6)
    )
    surface_rate_s = internal_rate_s * (1.0 + 2.25 * aggression)
    surface_fraction = 1.0 - np.exp(-surface_rate_s * time_s)
    bulk_fraction = 1.0 - remaining_fraction
    imbalance = np.maximum(surface_fraction - bulk_fraction, 0.0)
    thermal_shock = _sigmoid((mean_temperature_c - 78.0) / 9.0) * (
        1.0 - np.exp(-speed / 2.0)
    )
    short_residence_penalty = np.exp(-residence / 5.5) * aggression
    defect_index = np.clip(
        1.35 * imbalance + 0.075 * thermal_shock + 0.075 * short_residence_penalty,
        0.0,
        1.0,
    )

    # Energy accounting is positive and dimensional: sensible heat, latent
    # evaporation heat, and fan work per square metre of web.
    evaporated_water_kg_m2 = np.maximum(
        INITIAL_WATER_MASS_KG_M2 - remaining_water_kg_m2,
        0.0,
    )
    sensible_kj_m2 = (
        INITIAL_WET_MASS_KG_M2
        * 1700.0
        * np.maximum(temperature - reference_temperature_c, 0.0)
        / 0.76
        / 1000.0
    )
    latent_kj_m2 = evaporated_water_kg_m2 * 2.35e6 / 0.80 / 1000.0
    fan_kj_m2 = 0.085 * speed**3 * time_s / 1000.0
    energy_kj_m2 = sensible_kj_m2 + latent_kj_m2 + fan_kj_m2

    return {
        "energy_kj_m2": np.asarray(energy_kj_m2, dtype=float),
        "residual_moisture_pct": np.asarray(residual_moisture_pct, dtype=float),
        "defect_index": np.asarray(defect_index, dtype=float),
    }
