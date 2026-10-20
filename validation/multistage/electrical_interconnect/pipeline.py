"""2材料枝の合流、最終仕様、操作契約。全工程の直積は生成しません。"""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
import hashlib
from pathlib import Path

import numpy as np

from .stages import MATERIAL_PROPERTIES, PART_FIELDS, FORM_OUTPUT_FIELDS, STAGE_MANIFESTS, evaluate_stage


STAGE_ORDER = ("cut_cu", "cut_al", "form_cu", "form_al", "assembly", "welding")
VARIANTS = {"cu_al": ("cu", "al"), "cu_cu": ("cu", "cu"), "al_al": ("al", "al")}
DEFAULT_CONTROLS = {f"{sid}.{name}": STAGE_MANIFESTS[sid]["default_inputs"][name]
                    for sid in STAGE_ORDER for name in STAGE_MANIFESTS[sid]["controls"]}
DEFAULT_MATERIAL_CONTEXT = {f"{sid}.{name}": STAGE_MANIFESTS[sid]["default_inputs"][name]
                            for sid in ("cut_cu", "cut_al") for name in STAGE_MANIFESTS[sid]["incoming_context"]}
FINAL_SPECIFICATIONS = [
    {"name": "cold_unclamped_resistance_uohm", "direction": "less_equal", "target": 60.0, "margin_scale": 15.0},
    {"name": "shear_force_n", "direction": "greater_equal", "target": 1300.0, "margin_scale": 1300.0},
    {"name": "temperature_rise_k", "direction": "less_equal", "target": 2.3, "margin_scale": 0.7},
    {"name": "lack_of_fusion_fraction", "direction": "less_equal", "target": 0.15, "margin_scale": 0.15},
    {"name": "porosity_fraction", "direction": "less_equal", "target": 0.09, "margin_scale": 0.09},
    {"name": "imc_proxy_um", "direction": "less_equal", "target": 6.0, "margin_scale": 6.0},
    {"name": "post_release_distortion_mm", "direction": "less_equal", "target": 0.35, "margin_scale": 0.35},
]

CONNECTIONS = []
for _cut, _form in (("cut_cu", "form_cu"), ("cut_al", "form_al")):
    CONNECTIONS.extend({"from": f"{_cut}.{name}", "to": f"{_form}.{name}"} for name in PART_FIELDS)
for _form, _side in (("form_cu", "bottom"), ("form_al", "top")):
    CONNECTIONS.extend({"from": f"{_form}.{name}", "to": f"assembly.{_side}_{name}"} for name in FORM_OUTPUT_FIELDS)
CONNECTIONS.extend({"from": f"assembly.{name}", "to": f"welding.{name}"}
                   for name in STAGE_MANIFESTS["welding"]["incoming_context"])
SOURCE_OUTPUTS = {connection["to"]: connection["from"] for connection in CONNECTIONS}


def axes(levels: int = 4) -> dict[str, list[float]]:
    """独立操作軸の水準を返します。直積展開を行わず、境界両端を含めます。"""
    if not isinstance(levels, int) or not 2 <= levels <= 20:
        raise ValueError("levels must be an integer between 2 and 20")
    return {f"{sid}.{name}": np.linspace(*STAGE_MANIFESTS[sid]["input_bounds"][name], levels).tolist()
            for sid in STAGE_ORDER for name in STAGE_MANIFESTS[sid]["controls"]}


def material_context(variant: str = "cu_al", overrides: Mapping | None = None) -> dict:
    """材種・調質の代表物性を枝へ設定し、許可された原材状態上書きを適用します。"""
    if variant not in VARIANTS:
        raise ValueError(f"variant must be one of {tuple(VARIANTS)}")
    result = dict(DEFAULT_MATERIAL_CONTEXT)
    for sid, material in zip(("cut_cu", "cut_al"), VARIANTS[variant]):
        properties = MATERIAL_PROPERTIES[material]
        for name in ("stock_yield_mpa", "elastic_modulus_gpa", "material_is_cu"):
            result[f"{sid}.{name}"] = properties[name]
    if overrides:
        extra = set(overrides) - set(result)
        if extra:
            raise ValueError(f"Unknown material context: {sorted(extra)}")
        for sid, material in zip(("cut_cu", "cut_al"), VARIANTS[variant]):
            key = f"{sid}.material_is_cu"
            if key in overrides and np.any(np.asarray(overrides[key]) != float(material == "cu")):
                raise ValueError("material identity is fixed by variant")
        result.update(overrides)
    return result


_resolve_material_context = material_context


def line_config(variant: str = "cu_al", levels: int = 4) -> dict:
    """r002へ接続するbundle未指定の設定を返します。材料contextは外部入力です。"""
    return {"schema_version": "2.0", "case_id": "electrical_interconnect", "variant": variant,
            "stages": [{"id": sid} for sid in STAGE_ORDER], "connections": deepcopy(CONNECTIONS),
            "external_context": material_context(variant), "candidate_axes": axes(levels),
            "final_stage": "welding", "final_specifications": deepcopy(FINAL_SPECIFICATIONS),
            "enforce_local_constraints": True, "use_for_selection": True,
            "quality_label": "SYNTHETIC_UNCALIBRATED_PHYSICS_BENCHMARK"}


def constraint_margin(outputs: Mapping, specifications: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    """仕様余裕を規定尺度で正規化し、有限値と全制約の合否を返します。"""
    first = np.asarray(next(iter(outputs.values())), dtype=float)
    margins = []
    valid = np.ones(first.shape, dtype=bool)
    for spec in specifications:
        values = np.asarray(outputs[spec["name"]], dtype=float)
        valid &= np.isfinite(values)
        scale = float(spec["margin_scale"])
        if scale <= 0.0:
            raise ValueError("margin_scale must be positive")
        margin = (values - spec["target"]) / scale
        margins.append(margin if spec["direction"] == "greater_equal" else -margin)
    minimum = np.minimum.reduce(margins) if margins else np.zeros(first.shape)
    minimum = np.where(valid, minimum, -np.inf)
    return minimum, valid & (minimum >= 0.0)


def evaluate_line(controls: Mapping, material_context: Mapping | None = None, variant: str = "cu_al") -> dict:
    """同じ仮想ワークの2枝を順に計算し、合流後の最終品質を返します。

    指定されていない操作はDEFAULT_CONTROLS。scalarと配列はライン全体でbroadcast。
    完成ワークのみ下流へ渡す選別はベンチマークdriverが担当し、oracleはNGも計算します。
    """
    extra = set(controls) - set(DEFAULT_CONTROLS)
    if extra:
        raise ValueError(f"Unknown qualified controls: {sorted(extra)}")
    values = {**DEFAULT_CONTROLS, **controls}
    context = _resolve_material_context(variant, material_context)
    names = [*values, *context]
    arrays = np.broadcast_arrays(*[np.asarray(v, dtype=float) for v in [*values.values(), *context.values()]])
    all_values = dict(zip(names, arrays))
    outputs = {}
    stage_inputs = {}
    local = {}
    for sid in STAGE_ORDER:
        manifest = STAGE_MANIFESTS[sid]
        inputs = {name: all_values[f"{sid}.{name}"] for name in manifest["controls"]}
        for name in manifest["incoming_context"]:
            key = f"{sid}.{name}"
            if key in SOURCE_OUTPUTS:
                source, output = SOURCE_OUTPUTS[key].split(".", 1)
                inputs[name] = outputs[source][output]
            else:
                inputs[name] = all_values[key]
        outputs[sid] = evaluate_stage(sid, inputs)
        stage_inputs[sid] = inputs
        local[sid] = constraint_margin(outputs[sid], manifest["local_constraints"])[1]
    final = dict(outputs["welding"])
    final["quality_margin"], final["feasible"] = constraint_margin(final, FINAL_SPECIFICATIONS)
    final["all_local_feasible"] = np.logical_and.reduce(list(local.values()))
    final["line_feasible"] = final["feasible"] & final["all_local_feasible"]
    return {"stage_inputs": stage_inputs, "stage_outputs": outputs, "local_feasible": local,
            "final": final, "variant": variant, "controls": {k: all_values[k] for k in values}}


def source_hashes() -> dict[str, str]:
    """実行した物理・観測契約を結果へ記録するためのSHA-256を返します。"""
    directory = Path(__file__).resolve().parent
    return {name: hashlib.sha256((directory / name).read_bytes()).hexdigest()
            for name in ("stages.py", "pipeline.py", "inspection.py", "README.md")}
