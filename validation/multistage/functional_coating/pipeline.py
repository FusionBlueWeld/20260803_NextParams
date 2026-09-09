"""End-to-end composition of the three functional-coating stages."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from validation.multistage.src.common import validate_and_broadcast
from validation.multistage.functional_coating.coating import physics_model as coating
from validation.multistage.functional_coating.curing import physics_model as curing
from validation.multistage.functional_coating.drying import physics_model as drying


MATERIAL_BOUNDS = {
    "material_viscosity_pa_s": coating.INCOMING_STATE_BOUNDS["incoming_viscosity_pa_s"],
    "material_solids_fraction": coating.INCOMING_STATE_BOUNDS["incoming_solids_fraction"],
    "material_bubble_fraction": coating.INCOMING_STATE_BOUNDS["incoming_bubble_fraction"],
}

DEFAULT_MATERIAL_STATE = {
    "material_viscosity_pa_s": 1.6,
    "material_solids_fraction": 0.50,
    "material_bubble_fraction": 0.005,
}

LINE_INPUT_BOUNDS = {
    **coating.CONTROL_BOUNDS,
    **drying.CONTROL_BOUNDS,
    **curing.CONTROL_BOUNDS,
    **MATERIAL_BOUNDS,
}

FINAL_SPECIFICATIONS: dict[str, tuple[str, object]] = {
    "bond_strength_mpa": ("greater_equal", 32.0),
    "cure_fraction": ("greater_equal", 0.80),
    "degradation_fraction": ("less_equal", 0.20),
    "final_residual_solvent_pct": ("less_equal", 2.0),
    "final_defect_index": ("less_equal", 0.25),
    "dimensional_change_um": ("less_equal", 6.0),
    "final_thickness_um": ("between", (55.0, 85.0)),
}


def evaluate_line(conditions: Mapping[str, ArrayLike]) -> dict[str, Any]:
    """Run coating -> drying -> curing with explicit state hand-offs.

    ``conditions`` must contain every line control and the three incoming raw
    material state variables.  Scalars and arrays may be mixed and are first
    broadcast across the whole line so every intermediate has one common shape.
    """

    x = validate_and_broadcast(conditions, LINE_INPUT_BOUNDS)

    coating_result = coating.evaluate_model(
        coating_gap_um=x["coating_gap_um"],
        line_speed_m_min=x["line_speed_m_min"],
        web_tension_n=x["web_tension_n"],
        incoming_viscosity_pa_s=x["material_viscosity_pa_s"],
        incoming_solids_fraction=x["material_solids_fraction"],
        incoming_bubble_fraction=x["material_bubble_fraction"],
    )
    drying_result = drying.evaluate_model(
        air_temperature_c=x["air_temperature_c"],
        air_speed_m_s=x["air_speed_m_s"],
        residence_time_min=x["residence_time_min"],
        incoming_wet_thickness_um=coating_result["wet_thickness_um"],
        incoming_solids_fraction=coating_result["solids_fraction"],
        incoming_thickness_cv_fraction=coating_result["thickness_cv_fraction"],
        incoming_coating_defect_index=coating_result["coating_defect_index"],
    )
    curing_result = curing.evaluate_model(
        oven_temperature_c=x["oven_temperature_c"],
        hold_time_min=x["hold_time_min"],
        nip_pressure_mpa=x["nip_pressure_mpa"],
        incoming_dry_thickness_um=drying_result["dry_thickness_um"],
        incoming_residual_solvent_pct=drying_result["residual_solvent_pct"],
        incoming_internal_stress_mpa=drying_result["internal_stress_mpa"],
        incoming_drying_defect_index=drying_result["drying_defect_index"],
        incoming_thickness_cv_fraction=drying_result["dry_thickness_cv_fraction"],
    )

    final = dict(curing_result)
    final["total_thermal_energy_kj_m2"] = (
        drying_result["drying_energy_kj_m2"]
        + curing_result["curing_energy_kj_m2"]
    )
    final["line_throughput_m2_h"] = x["line_speed_m_min"] * 60.0
    final["quality_margin"] = quality_margin(final)
    final["feasible"] = final_feasible(final)

    return {
        "inputs": x,
        "coating": coating_result,
        "drying": drying_result,
        "curing": curing_result,
        "final": final,
    }


def final_feasible(outputs: Mapping[str, ArrayLike]) -> np.ndarray:
    """Return the final-product acceptance mask."""

    first = np.asarray(outputs["bond_strength_mpa"])
    mask = np.ones(first.shape, dtype=bool)
    for name, (direction, target) in FINAL_SPECIFICATIONS.items():
        values = np.asarray(outputs[name])
        if direction == "greater_equal":
            mask &= values >= float(target)
        elif direction == "less_equal":
            mask &= values <= float(target)
        else:
            lower, upper = target
            mask &= (values >= lower) & (values <= upper)
    return mask


def quality_margin(outputs: Mapping[str, ArrayLike]) -> np.ndarray:
    """Return the smallest normalized distance to a final specification.

    Positive values satisfy every specification; negative values expose how
    far the worst condition is beyond its threshold.  This is a diagnostic,
    not a calibrated failure probability.
    """

    margins = [
        (np.asarray(outputs["bond_strength_mpa"]) - 32.0) / 32.0,
        (np.asarray(outputs["cure_fraction"]) - 0.80) / 0.20,
        (0.20 - np.asarray(outputs["degradation_fraction"])) / 0.20,
        (2.0 - np.asarray(outputs["final_residual_solvent_pct"])) / 2.0,
        (0.25 - np.asarray(outputs["final_defect_index"])) / 0.25,
        (6.0 - np.asarray(outputs["dimensional_change_um"])) / 6.0,
        (np.asarray(outputs["final_thickness_um"]) - 55.0) / 15.0,
        (85.0 - np.asarray(outputs["final_thickness_um"])) / 15.0,
    ]
    return np.min(np.stack(margins, axis=0), axis=0)
