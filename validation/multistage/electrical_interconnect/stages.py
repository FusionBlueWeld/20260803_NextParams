"""6工程の入力契約と軽量oracle。

測定誤差はここへ混ぜません。式の係数は検証用の合成係数であり、実設備の
加工条件を保証する校正値ではありません。物性と工程構造の出典はREADME参照。
"""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy

import numpy as np

from validation.multistage.src.common import checked_outputs, sigmoid, validate_and_broadcast


MATERIAL_PROPERTIES = {
    "cu": {"material_is_cu": 1.0, "stock_yield_mpa": 90.0, "elastic_modulus_gpa": 118.0,
           "resistivity_nohm_m": 17.2, "conductivity_w_m_k": 390.0,
           "density_g_cm3": 8.94, "melting_temperature_c": 1083.0},
    "al": {"material_is_cu": 0.0, "stock_yield_mpa": 85.0, "elastic_modulus_gpa": 69.0,
           "resistivity_nohm_m": 28.2, "conductivity_w_m_k": 222.0,
           "density_g_cm3": 2.71, "melting_temperature_c": 660.0},
}

SOURCE_REFERENCES = {
    "welding_anchor": "https://www.mdpi.com/2075-4701/8/7/510",
    "mechanical_cut_edge": "https://pmc.ncbi.nlm.nih.gov/articles/PMC7412372/",
    "copper_forming": "https://live.copper.org/publications/pub_list/pdf/a1360.pdf",
    "al_sheet_properties": "https://www.hulamin.com/sites/default/files/downloads/1050%20Data%20Sheet.pdf",
    "assembly_gap": "https://pmc.ncbi.nlm.nih.gov/articles/PMC9921288/",
    "busbar_electrical_path": "https://www.kupfer.de/wp-content/uploads/2019/11/Copper-for-Busbars.pdf",
}

# 各tupleは下限・上限・単位・代表値。material_is_cuは固定カテゴリです。
STOCK_FIELDS = {
    "stock_thickness_mm": (0.9, 1.1, "mm", 1.0),
    "stock_yield_mpa": (45.0, 180.0, "MPa", 90.0),
    "elastic_modulus_gpa": (65.0, 125.0, "GPa", 118.0),
    "material_is_cu": (0.0, 1.0, "category", 1.0),
    "stock_camber_mm": (-0.3, 0.3, "mm", 0.025),
    "stock_oxide_nm": (0.0, 80.0, "nm", 8.0),
    "stock_contamination_fraction": (0.0, 0.6, "fraction", 0.03),
}

CUT_CONTROLS = {
    "cut_speed_m_min": (40.0, 160.0, "m/min", 110.0),
    "tooth_feed_mm": (0.01, 0.09, "mm/tooth", 0.035),
    "clamp_force_n": (150.0, 1200.0, "N", 700.0),
    "finish_allowance_mm": (0.05, 0.6, "mm", 0.25),
    "finish_feed_mm_rev": (0.02, 0.16, "mm/rev", 0.06),
    "deburr_depth_mm": (0.0, 0.25, "mm", 0.05),
    "length_offset_mm": (-0.4, 0.4, "mm", 0.0),
}

PART_FIELDS = {
    "thickness_mm": (0.65, 1.15, "mm", 1.0),
    "length_mm": (78.0, 82.0, "mm", 80.0),
    "width_mm": (38.0, 42.0, "mm", 40.0),
    "burr_mm": (0.0, 0.4, "mm", 0.01),
    "roughness_um": (0.1, 25.0, "um", 2.0),
    "edge_damage_fraction": (0.0, 1.0, "fraction", 0.06),
    "edge_hardening_fraction": (0.0, 1.0, "fraction", 0.1),
    "camber_mm": (-0.6, 0.8, "mm", 0.025),
    "oxide_nm": (0.0, 100.0, "nm", 8.0),
    "contamination_fraction": (0.0, 1.0, "fraction", 0.08),
    "yield_strength_mpa": (30.0, 260.0, "MPa", 90.0),
    "elastic_modulus_gpa": (65.0, 125.0, "GPa", 118.0),
    "material_is_cu": (0.0, 1.0, "category", 1.0),
}

FORM_CONTROLS = {
    "first_overbend_deg": (0.0, 5.0, "degree", 0.65),
    "second_overbend_deg": (0.0, 5.0, "degree", 0.65),
    "punch_radius_mm": (1.0, 4.0, "mm", 2.0),
    "die_opening_mm": (8.0, 20.0, "mm", 12.0),
    "holding_force_n": (150.0, 1500.0, "N", 800.0),
    "ram_speed_mm_s": (5.0, 50.0, "mm/s", 20.0),
    "coining_pressure_mpa": (0.0, 90.0, "MPa", 25.0),
}

FORM_OUTPUT_FIELDS = {
    **PART_FIELDS,
    "first_angle_error_deg": (-4.0, 5.0, "degree", 0.0),
    "second_angle_error_deg": (-4.0, 5.0, "degree", 0.0),
    "pad_height_error_mm": (-3.0, 4.0, "mm", 0.04),
    "pad_slope_deg": (-4.0, 5.0, "degree", 0.0),
    "plastic_strain": (0.0, 0.7, "strain", 0.2),
    "residual_stress_mpa": (0.0, 260.0, "MPa", 35.0),
    "form_damage_fraction": (0.0, 1.0, "fraction", 0.1),
    "thickness_loss_fraction": (0.0, 0.3, "fraction", 0.03),
}

ASSEMBLY_CONTROLS = {
    "x_correction_mm": (-0.6, 0.6, "mm", 0.0),
    "y_correction_mm": (-0.6, 0.6, "mm", 0.0),
    "rotation_correction_deg": (-1.0, 1.0, "degree", 0.0),
    "bottom_clamp_force_n": (200.0, 1600.0, "N", 900.0),
    "top_clamp_force_n": (200.0, 1600.0, "N", 900.0),
    "clamp_span_mm": (10.0, 30.0, "mm", 20.0),
    "bottom_cleaning_dose": (0.0, 1.0, "dose", 0.6),
    "top_cleaning_dose": (0.0, 1.0, "dose", 0.7),
}

WELD_CONTROLS = {
    "laser_power_kw": (1.5, 3.25, "kW", 2.5),
    "travel_speed_m_min": (4.0, 8.0, "m/min", 6.0),
    "focus_offset_mm": (-2.0, 2.0, "mm", 0.0),
    "oscillation_amplitude_mm": (0.25, 1.0, "mm", 0.5),
    "oscillation_frequency_hz": (200.0, 1000.0, "Hz", 500.0),
    "seam_length_mm": (25.0, 35.0, "mm", 35.0),
    "seam_offset_mm": (-3.0, 3.0, "mm", 0.0),
    "ramp_length_mm": (0.0, 6.0, "mm", 2.0),
}

WELD_CONTEXT_FIELDS = {
    "bottom_thickness_mm": (0.6, 1.2, "mm", 0.97),
    "top_thickness_mm": (0.6, 1.2, "mm", 0.97),
    "bottom_material_is_cu": (0.0, 1.0, "category", 1.0),
    "top_material_is_cu": (0.0, 1.0, "category", 0.0),
    "bottom_yield_strength_mpa": (30.0, 260.0, "MPa", 100.0),
    "top_yield_strength_mpa": (30.0, 260.0, "MPa", 100.0),
    "bottom_damage_fraction": (0.0, 1.0, "fraction", 0.1),
    "top_damage_fraction": (0.0, 1.0, "fraction", 0.1),
    "bottom_residual_stress_mpa": (0.0, 260.0, "MPa", 35.0),
    "top_residual_stress_mpa": (0.0, 260.0, "MPa", 35.0),
    "gap_start_mm": (0.0, 5.0, "mm", 0.02),
    "gap_center_mm": (0.0, 5.0, "mm", 0.01),
    "gap_end_mm": (0.0, 5.0, "mm", 0.02),
    "gap_max_mm": (0.0, 5.0, "mm", 0.02),
    "contact_fraction": (0.0, 1.0, "fraction", 0.8),
    "contact_pressure_mpa": (0.0, 20.0, "MPa", 3.0),
    "fixture_heat_loss_fraction": (0.0, 0.5, "fraction", 0.12),
    "bottom_oxide_nm": (0.0, 100.0, "nm", 1.5),
    "top_oxide_nm": (0.0, 100.0, "nm", 1.5),
    "bottom_contamination_fraction": (0.0, 1.0, "fraction", 0.02),
    "top_contamination_fraction": (0.0, 1.0, "fraction", 0.02),
    "surface_roughness_um": (0.1, 25.0, "um", 2.0),
    "overlap_mm": (8.0, 16.0, "mm", 12.0),
    "lateral_error_mm": (-1.0, 1.0, "mm", 0.0),
    "assembly_plastic_set_mm": (0.0, 2.0, "mm", 0.0),
    "assembly_stored_energy_mj": (0.0, 1000.0, "mJ", 10.0),
}

WELD_OUTPUT_UNITS = {
    "line_energy_j_mm": "J/mm", "absorbed_energy_j_mm": "J/mm", "peak_intensity_kw_mm2": "kW/mm2",
    "top_penetration_mm": "mm", "bottom_penetration_mm": "mm", "joint_width_mm": "mm",
    "bonded_area_mm2": "mm2", "mixing_fraction": "fraction", "imc_proxy_um": "um",
    "porosity_fraction": "fraction", "lack_of_fusion_fraction": "fraction", "spatter_fraction": "fraction",
    "base_path_resistance_uohm": "uohm", "interface_resistance_uohm": "uohm",
    "cold_unclamped_resistance_uohm": "uohm", "shear_force_n": "N", "temperature_rise_k": "K",
    "post_release_distortion_mm": "mm", "weld_time_s": "s", "fusion_completion_fraction": "fraction",
}

LATENT_NAMES = {"edge_damage_fraction", "edge_hardening_fraction", "plastic_strain", "residual_stress_mpa",
                "yield_strength_mpa", "oxide_nm", "contamination_fraction",
                "stock_yield_mpa", "stock_oxide_nm", "stock_contamination_fraction",
                "form_damage_fraction", "bottom_damage_fraction", "top_damage_fraction",
                "bottom_residual_stress_mpa", "top_residual_stress_mpa", "contact_fraction",
                "fixture_heat_loss_fraction", "assembly_stored_energy_mj", "mixing_fraction",
                "spatter_fraction", "lack_of_fusion_fraction", "fusion_completion_fraction",
                "absorbed_energy_j_mm", "base_path_resistance_uohm", "interface_resistance_uohm"}
DESTRUCTIVE_NAMES = {"imc_proxy_um", "porosity_fraction", "top_penetration_mm", "bottom_penetration_mm",
                     "joint_width_mm", "bonded_area_mm2", "shear_force_n"}


def is_latent(name: str) -> bool:
    """枝prefixが付いた状態も、元の検査不能状態と同じ扱いにします。"""
    return name in LATENT_NAMES or any(name.endswith(f"_{base}") for base in LATENT_NAMES)


def _manifest(stage_id, controls, incoming, outputs, objective, constraints):
    """物理式とは独立に、モデル入出力・操作軸・検査可能性を宣言します。"""
    fields = {**controls, **incoming}
    units = outputs if all(isinstance(v, str) for v in outputs.values()) else {n: f[2] for n, f in outputs.items()}
    visible = [n for n in units if not is_latent(n) and n not in DESTRUCTIVE_NAMES]
    return {"stage_id": stage_id, "controls": list(controls), "incoming_context": list(incoming),
            "predicted_outputs": list(units), "input_bounds": {n: [f[0], f[1]] for n, f in fields.items()},
            "input_units": {n: f[2] for n, f in fields.items()}, "output_units": units,
            "default_inputs": {n: f[3] for n, f in fields.items()},
            "candidate_axes": {n: np.linspace(f[0], f[1], 4).tolist() for n, f in controls.items()},
            "local_objective": {"name": objective, "direction": "minimize"}, "local_constraints": constraints,
            "observation_policy": {"nondestructive": visible,
                                   "destructive": [n for n in units if n in DESTRUCTIVE_NAMES],
                                   "latent": [n for n in units if is_latent(n)],
                                   "incoming_latent": [n for n in incoming if is_latent(n)],
                                   "proxy_policy": "nominal context; never expose latent oracle truth"},
            "model_status": "SYNTHETIC_UNCALIBRATED_PHYSICS_BENCHMARK"}


def _upper(name, target, scale):
    return {"name": name, "direction": "less_equal", "target": target, "margin_scale": scale}


CUT_OUTPUT_FIELDS = {**PART_FIELDS, "length_error_mm": (0.0, 1.0, "mm", 0.0),
                     "cut_time_s": (0.0, 120.0, "s", 10.0)}
FORM_ALL_OUTPUT_FIELDS = {**FORM_OUTPUT_FIELDS, "angle_error_max_deg": (0.0, 8.0, "degree", 0.0),
                          "form_time_s": (0.0, 60.0, "s", 3.0)}
ASSEMBLY_INCOMING = {f"{side}_{name}": field for side in ("bottom", "top") for name, field in FORM_OUTPUT_FIELDS.items()}
STAGE_MANIFESTS = {}
for _material in ("cu", "al"):
    _stock = deepcopy(STOCK_FIELDS)
    _stock["stock_yield_mpa"] = (*_stock["stock_yield_mpa"][:3], MATERIAL_PROPERTIES[_material]["stock_yield_mpa"])
    _stock["elastic_modulus_gpa"] = (*_stock["elastic_modulus_gpa"][:3], MATERIAL_PROPERTIES[_material]["elastic_modulus_gpa"])
    _stock["material_is_cu"] = (0.0, 1.0, "category", float(_material == "cu"))
    STAGE_MANIFESTS[f"cut_{_material}"] = _manifest(f"cut_{_material}", CUT_CONTROLS, _stock, CUT_OUTPUT_FIELDS,
        "cut_time_s", [_upper("burr_mm", 0.06, 0.06), _upper("edge_damage_fraction", 0.35, 0.35), _upper("length_error_mm", 0.45, 0.45)])
    _incoming = deepcopy(PART_FIELDS)
    _incoming["material_is_cu"] = (0.0, 1.0, "category", float(_material == "cu"))
    _incoming["yield_strength_mpa"] = (30.0, 260.0, "MPa", MATERIAL_PROPERTIES[_material]["stock_yield_mpa"])
    _incoming["elastic_modulus_gpa"] = (65.0, 125.0, "GPa", MATERIAL_PROPERTIES[_material]["elastic_modulus_gpa"])
    STAGE_MANIFESTS[f"form_{_material}"] = _manifest(f"form_{_material}", FORM_CONTROLS, _incoming, FORM_ALL_OUTPUT_FIELDS,
        "form_time_s", [_upper("angle_error_max_deg", 2.0, 2.0), _upper("form_damage_fraction", 0.35, 0.35), _upper("thickness_loss_fraction", 0.12, 0.12)])
STAGE_MANIFESTS["assembly"] = _manifest("assembly", ASSEMBLY_CONTROLS, ASSEMBLY_INCOMING,
    {**WELD_CONTEXT_FIELDS, "assembly_time_s": (0.0, 80.0, "s", 10.0)}, "assembly_time_s",
    [_upper("gap_max_mm", 0.15, 0.15), _upper("assembly_plastic_set_mm", 0.3, 0.3)])
STAGE_MANIFESTS["welding"] = _manifest("welding", WELD_CONTROLS, WELD_CONTEXT_FIELDS, WELD_OUTPUT_UNITS,
    "weld_time_s", [_upper("lack_of_fusion_fraction", 0.2, 0.2), _upper("spatter_fraction", 0.25, 0.25)])

# 同じ数量を別々に回帰すると、個々の角度がNGなのに最大角度だけ合格する等の
# 矛盾が生じます。既知の定義関係は数値モデル側でも維持する契約として渡します。
for _sid in ("form_cu", "form_al"):
    STAGE_MANIFESTS[_sid]["derived_outputs"] = {
        "angle_error_max_deg": {"operation": "max_abs", "sources": ["first_angle_error_deg", "second_angle_error_deg"]}}
STAGE_MANIFESTS["assembly"]["derived_outputs"] = {
    "gap_max_mm": {"operation": "max", "sources": ["gap_start_mm", "gap_center_mm", "gap_end_mm"]}}
STAGE_MANIFESTS["welding"]["derived_outputs"] = {
    "cold_unclamped_resistance_uohm": {"operation": "sum", "sources": ["base_path_resistance_uohm", "interface_resistance_uohm"]}}
for _manifest_item in STAGE_MANIFESTS.values():
    _manifest_item["source_references"] = dict(SOURCE_REFERENCES)
    _manifest_item["coefficient_provenance"] = "synthetic dimensionally consistent coefficients; no fit to cited experiments"
# 公開順序は、2材料枝の合流を含む共通topological順序に固定します。
STAGE_MANIFESTS = {sid: STAGE_MANIFESTS[sid] for sid in ("cut_cu", "cut_al", "form_cu", "form_al", "assembly", "welding")}


def _cut(x):
    """切削負荷・端部仕上げから、バリ・損傷・寸法と加工時間を算出します。"""
    load = (x["tooth_feed_mm"] / 0.04) * np.sqrt(x["cut_speed_m_min"] / 90.0) * np.sqrt(x["stock_yield_mpa"] / 90.0)
    removal = np.exp(-x["finish_allowance_mm"] / 0.18)
    burr = np.maximum(0.0, (0.012 + 0.045 * load + 0.012 * ((x["cut_speed_m_min"] - 110.0) / 80.0) ** 2) * removal - 0.7 * x["deburr_depth_mm"])
    damage = 1.0 - np.exp(-0.17 * load * removal - 0.12 * (x["deburr_depth_mm"] / x["stock_thickness_mm"]) ** 2)
    camber = x["stock_camber_mm"] + 0.025 * load / (x["clamp_force_n"] / 600.0 + 0.3)
    error = x["length_offset_mm"] + 0.022 * load / (x["clamp_force_n"] / 600.0 + 0.3)
    return {"thickness_mm": x["stock_thickness_mm"], "length_mm": 80.0 + error,
            "width_mm": np.full_like(error, 40.0), "length_error_mm": np.abs(error),
            "burr_mm": burr, "roughness_um": 0.7 + 7.0 * (x["finish_feed_mm_rev"] / 0.12) ** 2,
            "edge_damage_fraction": damage, "edge_hardening_fraction": 1.0 - np.exp(-0.25 * load * removal),
            "camber_mm": camber, "oxide_nm": x["stock_oxide_nm"],
            "contamination_fraction": np.clip(x["stock_contamination_fraction"] + 0.035 * load, 0.0, 1.0),
            "yield_strength_mpa": x["stock_yield_mpa"], "elastic_modulus_gpa": x["elastic_modulus_gpa"],
            "material_is_cu": x["material_is_cu"],
            "cut_time_s": 7.0 * 110.0 / x["cut_speed_m_min"] * 0.035 / x["tooth_feed_mm"]
                          + 1.8 * x["finish_allowance_mm"] / x["finish_feed_mm_rev"] + 14.0 * x["deburr_depth_mm"]}


def _form(x):
    """弾塑性の簡約式でspringback、薄肉化、接合面姿勢と残留応力を求めます。"""
    radius = np.maximum(x["punch_radius_mm"], 0.12 * x["die_opening_mm"])
    strain = x["thickness_mm"] / (2.0 * radius + x["thickness_mm"])
    yield_after = x["yield_strength_mpa"] * (1.0 + 0.55 * strain + 0.12 * x["edge_hardening_fraction"])
    springback = 270.0 * yield_after / (1000.0 * x["elastic_modulus_gpa"]) * (radius / x["thickness_mm"] + 0.5)
    springback *= (1.0 + 0.045 * x["ram_speed_mm_s"] / 20.0) / (1.0 + 0.15 * x["coining_pressure_mpa"] / 90.0)
    first = x["first_overbend_deg"] - springback
    second = x["second_overbend_deg"] - springback * (1.0 + 0.08 * strain)
    thinning = 1.0 - np.exp(-0.14 * strain * (1.0 + x["coining_pressure_mpa"] / 120.0))
    new_damage = 1.0 - (1.0 - x["edge_damage_fraction"]) * np.exp(-0.28 * np.maximum(strain - 0.10, 0.0) * (1.0 + 3.0 * x["edge_damage_fraction"]))
    camber = x["camber_mm"] / (1.0 + x["coining_pressure_mpa"] / 35.0) + 0.018 * (x["ram_speed_mm_s"] / 20.0) / (x["holding_force_n"] / 300.0 + 0.4)
    result = {n: x[n] for n in PART_FIELDS}
    result.update({"thickness_mm": x["thickness_mm"] * (1.0 - thinning),
        "yield_strength_mpa": yield_after, "camber_mm": camber,
        "first_angle_error_deg": first, "second_angle_error_deg": second,
        "angle_error_max_deg": np.maximum(np.abs(first), np.abs(second)),
        "pad_height_error_mm": 18.0 * np.sin(np.deg2rad(first + second)) + camber,
        "pad_slope_deg": second - 0.2 * first, "plastic_strain": strain,
        "residual_stress_mpa": yield_after * strain * (0.65 + 0.35 * x["coining_pressure_mpa"] / 90.0),
        "form_damage_fraction": new_damage, "thickness_loss_fraction": thinning,
        "form_time_s": 60.0 / x["ram_speed_mm_s"] + 0.012 * x["coining_pressure_mpa"] + 0.0003 * x["holding_force_n"]})
    return result


def _assembly(x):
    """2部材の形状と治具荷重から隙間field・接触・表面状態を導出します。"""
    force = x["bottom_clamp_force_n"] + x["top_clamp_force_n"]
    span = x["clamp_span_mm"]
    stiffness = (x["bottom_elastic_modulus_gpa"] * x["bottom_thickness_mm"] ** 3 + x["top_elastic_modulus_gpa"] * x["top_thickness_mm"] ** 3) * 40.0 / span ** 3
    height = x["top_pad_height_error_mm"] - x["bottom_pad_height_error_mm"]
    tilt = x["top_pad_slope_deg"] - x["bottom_pad_slope_deg"] + x["rotation_correction_deg"]
    burr = x["top_burr_mm"] + x["bottom_burr_mm"]
    closure = force / (500.0 + 1800.0 * stiffness)
    # 荷重を無限に増やして全てを救済できないよう、材料降伏後の永久変形を区別。
    force_capacity = 12.0 * (x["bottom_yield_strength_mpa"] * x["bottom_thickness_mm"] ** 2 + x["top_yield_strength_mpa"] * x["top_thickness_mm"] ** 2) / span
    plastic_set = np.maximum(force - force_capacity, 0.0) / 15000.0
    uneven = np.abs(x["bottom_clamp_force_n"] - x["top_clamp_force_n"]) / 50000.0
    center_raw = np.abs(height) + burr + 0.05 * np.abs(x["y_correction_mm"])
    start_raw = np.abs(height - 6.0 * np.tan(np.deg2rad(tilt))) + burr + uneven
    end_raw = np.abs(height + 6.0 * np.tan(np.deg2rad(tilt))) + burr + uneven
    gaps = [np.maximum(raw - closure, 0.0) + burr * 0.12 for raw in (start_raw, center_raw, end_raw)]
    mean_gap = sum(gaps) / 3.0
    contact = np.exp(-mean_gap / 0.04) * (0.65 + 0.35 * sigmoid((force - 500.0) / 350.0))
    overlap = 12.0 + x["x_correction_mm"] + 0.5 * (x["bottom_length_mm"] + x["top_length_mm"] - 160.0)
    result = {
        "bottom_thickness_mm": x["bottom_thickness_mm"], "top_thickness_mm": x["top_thickness_mm"],
        "bottom_material_is_cu": x["bottom_material_is_cu"], "top_material_is_cu": x["top_material_is_cu"],
        "bottom_yield_strength_mpa": x["bottom_yield_strength_mpa"], "top_yield_strength_mpa": x["top_yield_strength_mpa"],
        "bottom_damage_fraction": x["bottom_form_damage_fraction"], "top_damage_fraction": x["top_form_damage_fraction"],
        "bottom_residual_stress_mpa": x["bottom_residual_stress_mpa"], "top_residual_stress_mpa": x["top_residual_stress_mpa"],
        "gap_start_mm": gaps[0], "gap_center_mm": gaps[1], "gap_end_mm": gaps[2],
        "gap_max_mm": np.maximum.reduce(gaps), "contact_fraction": contact,
        "contact_pressure_mpa": force / (40.0 * overlap),
        "fixture_heat_loss_fraction": 0.04 + 0.24 * contact * force / (force + 1800.0),
        "surface_roughness_um": (x["bottom_roughness_um"] + x["top_roughness_um"]) / 2.0 + 0.4 * (x["bottom_cleaning_dose"] + x["top_cleaning_dose"]),
        "overlap_mm": overlap, "lateral_error_mm": x["y_correction_mm"],
        "assembly_plastic_set_mm": plastic_set,
        "assembly_stored_energy_mj": np.minimum(force * np.minimum(closure, center_raw) / 2.0, 1000.0),
        "assembly_time_s": 4.0 + 7.0 * (x["bottom_cleaning_dose"] + x["top_cleaning_dose"]) + 0.0015 * force}
    for side in ("bottom", "top"):
        dose = x[f"{side}_cleaning_dose"]
        result[f"{side}_oxide_nm"] = 0.8 + x[f"{side}_oxide_nm"] * np.exp(-3.0 * dose)
        result[f"{side}_contamination_fraction"] = x[f"{side}_contamination_fraction"] * np.exp(-4.0 * dose)
    return result


def _weld(x):
    """CW照射・接触放熱・融込みと異材混合から、治具解除後の品質を算出します。"""
    speed = x["travel_speed_m_min"] * 1000.0 / 60.0
    spot = 0.280 * np.sqrt(1.0 + (x["focus_offset_mm"] / 5.0) ** 2)
    intensity = x["laser_power_kw"] / (np.pi * (spot / 2.0) ** 2)
    top_cu, bottom_cu = x["top_material_is_cu"], x["bottom_material_is_cu"]
    initial_absorption = (0.14 - 0.075 * top_cu) * (1.0 + 0.015 * x["surface_roughness_um"] + 0.003 * x["top_oxide_nm"])
    coupling = np.clip(initial_absorption + (0.43 - initial_absorption) * sigmoid((intensity - (28.0 + 13.0 * top_cu)) / 5.0), 0.02, 0.65)
    line_energy = 1000.0 * x["laser_power_kw"] / speed
    ramp_factor = 1.0 - x["ramp_length_mm"] / (2.0 * x["seam_length_mm"])
    absorbed = line_energy * coupling * ramp_factor * (1.0 - x["fixture_heat_loss_fraction"])
    amp, frequency = x["oscillation_amplitude_mm"], x["oscillation_frequency_hz"]
    coverage = 1.0 - np.exp(-2.0 * amp * frequency / speed)
    swept_width = spot + 2.0 * amp * coverage
    thermal_demand = 5.1 + 2.0 * top_cu + 0.5 * bottom_cu
    depth = absorbed / (thermal_demand * swept_width)
    top_pen = np.minimum(depth, x["top_thickness_mm"])
    bottom_pen = np.minimum(np.maximum(depth - x["top_thickness_mm"], 0.0) * (0.6 + 0.35 * x["contact_fraction"]), x["bottom_thickness_mm"])
    offset_fraction = np.abs(x["seam_offset_mm"]) / 3.0
    endpoint = np.where(x["seam_offset_mm"] < 0.0, x["gap_start_mm"], x["gap_end_mm"])
    gap = (1.0 - offset_fraction) * x["gap_center_mm"] + offset_fraction * endpoint
    gap_loss = 1.0 - np.exp(-gap / (0.045 + 0.025 * amp))
    fusion = sigmoid((depth - x["top_thickness_mm"] - 0.025) / 0.045) * (1.0 - gap_loss)
    width = swept_width * np.sqrt(np.clip(depth / x["top_thickness_mm"], 0.0, 2.0)) * fusion
    contamination = (x["top_contamination_fraction"] + x["bottom_contamination_fraction"]) / 2.0
    porosity = np.clip(0.012 + 0.15 * contamination + 0.001 * (x["top_oxide_nm"] + x["bottom_oxide_nm"]) + 0.05 * sigmoid((depth - 1.8) / 0.25), 0.0, 0.7)
    spatter = np.clip(0.02 + 0.24 * sigmoid((depth - 1.7) / 0.24) + 0.10 * sigmoid((intensity - 49.0) / 7.0) * (1.0 - coverage), 0.0, 0.7)
    # 一定幅断面の簡約。下材混合量は体積比ではなく密度を掛けた質量比です。
    top_mass = (2.71 + 6.23 * top_cu) * top_pen
    bottom_mass = (2.71 + 6.23 * bottom_cu) * bottom_pen
    mixing = bottom_mass / np.maximum(top_mass + bottom_mass, 1e-9)
    dissimilar = np.abs(top_cu - bottom_cu)
    # IMCは実測厚さを再現する反応拡散計算ではなく、異材混合・熱滞在のproxy。
    imc = dissimilar * (0.4 + 15.0 * mixing ** 1.4 * np.sqrt(100.0 / speed)) * fusion
    area = width * x["seam_length_mm"] * (1.0 - porosity) * (1.0 - spatter)
    rho_top, rho_bottom = 28.2 - 11.0 * top_cu, 28.2 - 11.0 * bottom_cu
    base_r = rho_top * 40.0 / (40.0 * x["top_thickness_mm"]) + rho_bottom * 40.0 / (40.0 * x["bottom_thickness_mm"])
    interface_r = ((rho_top + rho_bottom) / 2.0) * 0.22 / np.maximum(area, 0.02) * (1.0 + 0.25 * imc + 2.0 * porosity)
    resistance = base_r + interface_r
    parent_limit = np.minimum(x["top_yield_strength_mpa"] * 40.0 * x["top_thickness_mm"] * (1.0 - x["top_damage_fraction"]),
                              x["bottom_yield_strength_mpa"] * 40.0 * x["bottom_thickness_mm"] * (1.0 - x["bottom_damage_fraction"]))
    joint_force = area * (62.0 + 28.0 * (top_cu + bottom_cu) / 2.0) * np.exp(-0.10 * imc)
    force = np.minimum(parent_limit, joint_force)
    # 電気発熱I^2Rと外部熱抵抗。試験電流200 A、冷却境界0.9 K/Wは固定fixture。
    temperature = 200.0 ** 2 * resistance * 1e-6 * 0.9
    stress = (x["bottom_residual_stress_mpa"] + x["top_residual_stress_mpa"]) / 2.0
    distortion = x["assembly_plastic_set_mm"] + 0.0007 * stress + 0.006 * absorbed + 0.00009 * x["assembly_stored_energy_mj"]
    return {"line_energy_j_mm": line_energy, "absorbed_energy_j_mm": absorbed, "peak_intensity_kw_mm2": intensity,
            "top_penetration_mm": top_pen, "bottom_penetration_mm": bottom_pen, "joint_width_mm": width,
            "bonded_area_mm2": area, "mixing_fraction": mixing, "imc_proxy_um": imc,
            "porosity_fraction": porosity, "lack_of_fusion_fraction": 1.0 - fusion, "spatter_fraction": spatter,
            "base_path_resistance_uohm": base_r, "interface_resistance_uohm": interface_r,
            "cold_unclamped_resistance_uohm": resistance, "shear_force_n": force,
            "temperature_rise_k": temperature, "post_release_distortion_mm": distortion,
            "weld_time_s": x["seam_length_mm"] / speed + 0.15, "fusion_completion_fraction": fusion}


def evaluate_stage(stage_id: str, inputs: Mapping) -> dict[str, np.ndarray]:
    """必須入力・カテゴリ・有限性・broadcastを検査し、同じ形のoracle値を返します。"""
    if stage_id not in STAGE_MANIFESTS:
        raise ValueError(f"Unknown electrical-interconnect stage: {stage_id}")
    manifest = STAGE_MANIFESTS[stage_id]
    x = validate_and_broadcast(inputs, manifest["input_bounds"])
    for name in x:
        if name.endswith("material_is_cu") and np.any((x[name] != 0.0) & (x[name] != 1.0)):
            raise ValueError(f"{name} is a fixed material category and must equal 0 or 1")
    model = _cut if stage_id.startswith("cut_") else _form if stage_id.startswith("form_") else _assembly if stage_id == "assembly" else _weld
    result = checked_outputs(model(x), next(iter(x.values())).shape)
    if set(result) != set(manifest["predicted_outputs"]):
        raise RuntimeError(f"Output contract mismatch for {stage_id}")
    return result


def default_inputs(stage_id: str) -> dict[str, float]:
    """独立工程の契約を満たす代表入力のコピーを返します。"""
    return dict(STAGE_MANIFESTS[stage_id]["default_inputs"])


def incoming_proxy(stage_id: str, inputs: Mapping, regime: str = "proxy") -> dict:
    """現場proxy regimeでは不可視な流入真値を固定代表値へ置換します。"""
    if regime not in ("proxy", "complete"):
        raise ValueError("regime must be proxy or complete")
    result = dict(inputs)
    if regime == "complete":
        return result
    manifest = STAGE_MANIFESTS[stage_id]
    for name in manifest["incoming_context"]:
        if is_latent(name):
            result[name] = np.full_like(np.asarray(inputs[name], dtype=float), manifest["default_inputs"][name])
    return result
