"""Deterministic, physics-structured synthetic copper electroplating model.

Faraday's law is the central thickness relation.  Mass transport, cell
voltage, current efficiency, and roughness are smooth engineering closures for
a fixed bath and tool configuration.  The empirical coefficients are intended
for optimization benchmarks, not calibrated production physics.
"""

from __future__ import annotations

from typing import Final

import numpy as np


# These limits intentionally mirror problem.csv.
INPUT_BOUNDS: Final[dict[str, tuple[float, float]]] = {
    "current_density_a_dm2": (1.0, 7.0),
    "bath_temperature_c": (20.0, 50.0),
    "plating_time_min": (5.0, 35.0),
    "copper_concentration_mol_l": (0.4, 1.6),
    "agitation_speed_m_s": (0.05, 0.35),
    "electrode_gap_cm": (0.5, 2.0),
    "duty_cycle": (0.4, 1.0),
    "bath_ph": (1.5, 3.0),
}

COPPER_MOLAR_MASS_KG_MOL: Final[float] = 0.063546
COPPER_VALENCE: Final[float] = 2.0
COPPER_DENSITY_KG_M3: Final[float] = 8960.0
FARADAY_CONSTANT_C_MOL: Final[float] = 96485.0
TARGET_THICKNESS_UM: Final[float] = 20.0


def _sigmoid(value: np.ndarray) -> np.ndarray:
    """Return a numerically stable logistic transform."""

    return 1.0 / (1.0 + np.exp(-np.clip(value, -60.0, 60.0)))


def _broadcast_and_validate(
    current_density_a_dm2: float | np.ndarray,
    bath_temperature_c: float | np.ndarray,
    plating_time_min: float | np.ndarray,
    copper_concentration_mol_l: float | np.ndarray,
    agitation_speed_m_s: float | np.ndarray,
    electrode_gap_cm: float | np.ndarray,
    duty_cycle: float | np.ndarray,
    bath_ph: float | np.ndarray,
) -> tuple[np.ndarray, ...]:
    """Broadcast eight inputs and reject nonfinite or out-of-range values."""

    raw = (
        current_density_a_dm2,
        bath_temperature_c,
        plating_time_min,
        copper_concentration_mol_l,
        agitation_speed_m_s,
        electrode_gap_cm,
        duty_cycle,
        bath_ph,
    )
    try:
        arrays = np.broadcast_arrays(*(np.asarray(value, dtype=float) for value in raw))
    except ValueError as exc:
        raise ValueError("Inputs must be mutually broadcastable.") from exc

    for name, values in zip(INPUT_BOUNDS, arrays):
        lower, upper = INPUT_BOUNDS[name]
        if np.any(~np.isfinite(values)):
            raise ValueError(f"{name} must contain only finite values.")
        if np.any(values < lower) or np.any(values > upper):
            raise ValueError(
                f"{name} must be within the inclusive range [{lower}, {upper}]."
            )
    return tuple(np.asarray(values, dtype=float) for values in arrays)


def evaluate_model(
    current_density_a_dm2: float | np.ndarray,
    bath_temperature_c: float | np.ndarray,
    plating_time_min: float | np.ndarray,
    copper_concentration_mol_l: float | np.ndarray,
    agitation_speed_m_s: float | np.ndarray,
    electrode_gap_cm: float | np.ndarray,
    duty_cycle: float | np.ndarray,
    bath_ph: float | np.ndarray,
) -> dict[str, np.ndarray]:
    """Evaluate copper-plating thickness, quality, and cell-voltage outputs.

    Parameters are, in order: current density [A/dm²], bath temperature [°C],
    plating time [min], copper concentration [mol/L], agitation speed [m/s],
    electrode gap [cm], duty cycle [1], and bath pH [1].  All inputs are
    broadcast together and must lie within the inclusive bounds in
    ``problem.csv``.

    Returns five arrays with one common broadcast shape for
    ``thickness_error_um`` (minimize), ``roughness_ra_um`` (less-equal
    constraint), ``current_efficiency`` (greater-equal constraint),
    ``deposit_thickness_um`` (monitor), and ``cell_voltage_v`` (monitor).
    """

    (
        current,
        temperature,
        time,
        concentration,
        agitation,
        gap,
        duty,
        ph,
    ) = _broadcast_and_validate(
        current_density_a_dm2,
        bath_temperature_c,
        plating_time_min,
        copper_concentration_mol_l,
        agitation_speed_m_s,
        electrode_gap_cm,
        duty_cycle,
        bath_ph,
    )

    # Fixed-tool mass-transfer closure.  Concentration, temperature, and
    # agitation raise the limiting current; a larger gap weakens renewal.
    agitation_factor = 1.0 + 0.80 * (1.0 - np.exp(-agitation / 0.12))
    gap_transport_factor = 1.0 / np.sqrt(1.0 + 0.10 * gap)
    limiting_current = (
        7.2
        * concentration**0.65
        * (1.0 + 0.010 * (temperature - 35.0))
        * agitation_factor
        * gap_transport_factor
    )
    transport_ratio = current / limiting_current
    transport_utilization = np.tanh(transport_ratio) / transport_ratio

    # Relative conductivity is a synthetic electrolyte closure used only to
    # translate concentration, temperature, pH, and gap into cell voltage.
    conductivity_relative = (
        concentration**0.70
        * (1.0 + 0.018 * (temperature - 25.0))
        * (1.0 + 0.50 * (2.2 - ph))
    )
    cell_voltage_v = (
        1.55
        + 0.018 * current
        + 0.025 * current * gap / conductivity_relative
        + 0.060 * (ph - 2.2) ** 2
        + 0.003 * (35.0 - temperature)
        + 0.015 * (1.0 - duty)
    )
    electrical_factor = 1.0 / (1.0 + 0.12 * np.maximum(cell_voltage_v - 1.7, 0.0))

    # Pulsing gives a small synthetic replenishment benefit, while peak current
    # still controls transport crowding and the burning transition.
    burning_gate = _sigmoid((transport_ratio - 0.78) / 0.12) * _sigmoid(
        (cell_voltage_v - 2.0) / 0.35
    )
    base_efficiency = (
        0.965
        + 0.0006 * (temperature - 35.0)
        + 0.018 * (1.0 - duty)
        - 0.018 * np.abs(ph - 2.2)
        + 0.008 * np.log(concentration)
    )
    burning_factor = 1.0 - 0.18 * burning_gate
    current_efficiency = np.clip(
        base_efficiency
        * transport_utilization
        * electrical_factor
        * burning_factor,
        0.05,
        0.995,
    )

    # Faraday thickness: h = eta * j * t * M/(z F rho), with unit conversion
    # from A/dm² and minutes to A/m² and seconds, then to micrometres.
    faraday_um_per_coulomb_m2 = (
        COPPER_MOLAR_MASS_KG_MOL
        / (COPPER_VALENCE * FARADAY_CONSTANT_C_MOL * COPPER_DENSITY_KG_M3)
        * 1.0e6
    )
    deposit_thickness_um = (
        current_efficiency
        * current
        * 100.0
        * time
        * 60.0
        * duty
        * faraday_um_per_coulomb_m2
    )
    thickness_error_um = np.abs(deposit_thickness_um - TARGET_THICKNESS_UM)

    # Surface roughness is an explicit synthetic empirical correction, not a
    # first-principles morphology solution.
    roughness_ra_um = (
        0.11
        + 0.035 * np.sqrt(current)
        + 0.045 * transport_ratio**2
        + 0.50 * burning_gate
        + 0.050 * (duty - 0.4) / 0.6
        + 0.030 * np.abs(ph - 2.2)
        + 0.020 * np.maximum(cell_voltage_v - 2.2, 0.0)
    )

    return {
        "thickness_error_um": np.asarray(thickness_error_um, dtype=float),
        "roughness_ra_um": np.asarray(roughness_ra_um, dtype=float),
        "current_efficiency": np.asarray(current_efficiency, dtype=float),
        "deposit_thickness_um": np.asarray(deposit_thickness_um, dtype=float),
        "cell_voltage_v": np.asarray(cell_voltage_v, dtype=float),
    }
