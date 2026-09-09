"""Reference grids demonstrating local and end-to-end coating-line choices."""

from __future__ import annotations

import hashlib
import itertools
from pathlib import Path

import numpy as np

from .coating.physics_model import evaluate_model as evaluate_coating
from .curing.physics_model import evaluate_model as evaluate_curing
from .drying.physics_model import evaluate_model as evaluate_drying
from .fault_scenarios import evaluate_fault_scenarios
from .pipeline import DEFAULT_MATERIAL_STATE, evaluate_line
from .robustness import estimate_robustness


CONTROL_AXES = {
    "coating_gap_um": (140.0, 210.0, 280.0),
    "line_speed_m_min": (8.0, 18.0, 28.0),
    "web_tension_n": (60.0, 100.0, 140.0),
    "air_temperature_c": (65.0, 90.0, 115.0),
    "air_speed_m_s": (1.0, 3.25, 5.5),
    "residence_time_min": (3.0, 8.0, 13.0),
    "oven_temperature_c": (105.0, 140.0, 175.0),
    "hold_time_min": (10.0, 35.0, 60.0),
    "nip_pressure_mpa": (0.10, 0.325, 0.55),
}


def _grid(names: list[str]) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    points = np.asarray(list(itertools.product(*(CONTROL_AXES[name] for name in names))))
    return points, {name: points[:, i] for i, name in enumerate(names)}


def _row(names: list[str], points: np.ndarray, index: int) -> dict[str, float]:
    return {name: float(points[index, i]) for i, name in enumerate(names)}


def _scalar_outputs(outputs: dict[str, np.ndarray]) -> dict[str, float | bool]:
    result: dict[str, float | bool] = {}
    for name, value in outputs.items():
        array = np.asarray(value)
        if array.dtype == bool:
            result[name] = bool(array)
        else:
            result[name] = float(array)
    return result


def individual_best_chain() -> dict[str, object]:
    """Optimize the three stages sequentially using local objectives only."""

    coating_names = ["coating_gap_um", "line_speed_m_min", "web_tension_n"]
    coating_points, cx = _grid(coating_names)
    coating_outputs = evaluate_coating(
        **cx,
        incoming_viscosity_pa_s=DEFAULT_MATERIAL_STATE["material_viscosity_pa_s"],
        incoming_solids_fraction=DEFAULT_MATERIAL_STATE["material_solids_fraction"],
        incoming_bubble_fraction=DEFAULT_MATERIAL_STATE["material_bubble_fraction"],
    )
    coating_ok = (
        (coating_outputs["thickness_cv_fraction"] <= 0.08)
        & (coating_outputs["coating_defect_index"] <= 0.25)
    )
    coating_score = np.where(
        coating_ok, coating_outputs["coated_solids_rate_kg_m_h"], -np.inf
    )
    coating_index = int(np.argmax(coating_score))
    if not np.isfinite(coating_score[coating_index]):
        raise RuntimeError("Coating reference grid has no locally feasible point")

    drying_names = ["air_temperature_c", "air_speed_m_s", "residence_time_min"]
    drying_points, dx = _grid(drying_names)
    drying_outputs = evaluate_drying(
        **dx,
        incoming_wet_thickness_um=coating_outputs["wet_thickness_um"][coating_index],
        incoming_solids_fraction=coating_outputs["solids_fraction"][coating_index],
        incoming_thickness_cv_fraction=coating_outputs["thickness_cv_fraction"][coating_index],
        incoming_coating_defect_index=coating_outputs["coating_defect_index"][coating_index],
    )
    drying_ok = (
        (drying_outputs["residual_solvent_pct"] <= 5.0)
        & (drying_outputs["drying_defect_index"] <= 0.30)
    )
    drying_score = np.where(drying_ok, drying_outputs["drying_energy_kj_m2"], np.inf)
    drying_index = int(np.argmin(drying_score))
    if not np.isfinite(drying_score[drying_index]):
        raise RuntimeError("Drying reference grid has no locally feasible point")

    curing_names = ["oven_temperature_c", "hold_time_min", "nip_pressure_mpa"]
    curing_points, ux = _grid(curing_names)
    curing_outputs = evaluate_curing(
        **ux,
        incoming_dry_thickness_um=drying_outputs["dry_thickness_um"][drying_index],
        incoming_residual_solvent_pct=drying_outputs["residual_solvent_pct"][drying_index],
        incoming_internal_stress_mpa=drying_outputs["internal_stress_mpa"][drying_index],
        incoming_drying_defect_index=drying_outputs["drying_defect_index"][drying_index],
        incoming_thickness_cv_fraction=drying_outputs["dry_thickness_cv_fraction"][drying_index],
    )
    curing_ok = (
        (curing_outputs["cure_fraction"] >= 0.80)
        & (curing_outputs["degradation_fraction"] <= 0.20)
        & (curing_outputs["final_residual_solvent_pct"] <= 2.0)
        & (curing_outputs["final_defect_index"] <= 0.25)
    )
    curing_score = np.where(curing_ok, curing_outputs["bond_strength_mpa"], -np.inf)
    curing_index = int(np.argmax(curing_score))
    if not np.isfinite(curing_score[curing_index]):
        raise RuntimeError("Curing reference grid has no locally feasible point")

    conditions = {
        **DEFAULT_MATERIAL_STATE,
        **_row(coating_names, coating_points, coating_index),
        **_row(drying_names, drying_points, drying_index),
        **_row(curing_names, curing_points, curing_index),
    }
    line_result = evaluate_line(conditions)
    return {
        "conditions": conditions,
        "local_objectives": {
            "coating": {
                "objective": "maximize coated_solids_rate_kg_m_h",
                "value": float(coating_outputs["coated_solids_rate_kg_m_h"][coating_index]),
            },
            "drying": {
                "objective": "minimize drying_energy_kj_m2",
                "value": float(drying_outputs["drying_energy_kj_m2"][drying_index]),
            },
            "curing": {
                "objective": "maximize bond_strength_mpa",
                "value": float(curing_outputs["bond_strength_mpa"][curing_index]),
            },
        },
        "final": _scalar_outputs(line_result["final"]),
    }


def full_line_grid() -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, object]]:
    """Evaluate the 3^9 deterministic coarse reference space."""

    names = list(CONTROL_AXES)
    points, conditions = _grid(names)
    line_conditions = {
        **conditions,
        **{name: np.full(len(points), value) for name, value in DEFAULT_MATERIAL_STATE.items()},
    }
    return points, conditions, evaluate_line(line_conditions)


def _candidate_summary(
    names: list[str], points: np.ndarray, result: dict[str, object], index: int
) -> dict[str, object]:
    return {
        "conditions": {**DEFAULT_MATERIAL_STATE, **_row(names, points, index)},
        "final": {
            name: (
                bool(np.asarray(value)[index])
                if np.asarray(value).dtype == bool
                else float(np.asarray(value)[index])
            )
            for name, value in result["final"].items()
        },
    }


def provenance() -> dict[str, str]:
    root = Path(__file__).resolve().parents[1]
    files = sorted(
        path for path in root.rglob("*")
        if path.is_file()
        and path.suffix in {".py", ".json", ".csv"}
        and "tests" not in path.relative_to(root).parts
        and "results" not in path.relative_to(root).parts
    )
    return {
        path.relative_to(root).as_posix(): hashlib.sha256(
            path.read_bytes().replace(b"\r\n", b"\n")
        ).hexdigest()
        for path in files
    }


def audit(robustness_samples: int = 512, seed: int = 0) -> dict[str, object]:
    """Produce deterministic references plus seeded robustness comparisons."""

    points, _, result = full_line_grid()
    names = list(CONTROL_AXES)
    final = result["final"]
    feasible = np.asarray(final["feasible"])
    if not np.any(feasible):
        raise RuntimeError("Full-line reference grid has no feasible point")

    margin_index = int(np.argmax(final["quality_margin"]))
    feasible_indices = np.flatnonzero(feasible)
    energy_index = int(feasible_indices[np.argmin(final["total_thermal_energy_kj_m2"][feasible])])
    throughput_index = int(feasible_indices[np.argmax(final["line_throughput_m2_h"][feasible])])

    local = individual_best_chain()
    margin = _candidate_summary(names, points, result, margin_index)
    energy = _candidate_summary(names, points, result, energy_index)
    throughput = _candidate_summary(names, points, result, throughput_index)
    local["robustness"] = estimate_robustness(
        local["conditions"], samples=robustness_samples, seed=seed
    )
    margin["robustness"] = estimate_robustness(
        margin["conditions"], samples=robustness_samples, seed=seed
    )

    return {
        "model": "functional_coating",
        "grid": {
            "levels_per_control": 3,
            "control_count": len(names),
            "candidate_count": len(points),
            "feasible_count": int(np.sum(feasible)),
            "infeasible_count": int(np.sum(~feasible)),
        },
        "individual_best_chain": local,
        "global_max_quality_margin": margin,
        "global_min_energy_feasible": energy,
        "global_max_throughput_feasible": throughput,
        "known_source_drift_scenarios": evaluate_fault_scenarios(),
        "provenance_sha256_lf": provenance(),
    }
