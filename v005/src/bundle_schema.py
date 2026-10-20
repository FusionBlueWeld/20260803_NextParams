"""JSON工程契約の検査。動的importやpickleには依存しない。"""

from __future__ import annotations

import copy
import math

from .derived_outputs import validate_derived_outputs


def validate_manifest(source: dict) -> dict:
    if not isinstance(source, dict):
        raise ValueError("stage manifestはobjectにしてください。")
    value = copy.deepcopy(source)
    if value.get("bundle_schema_version") != "2.0":
        raise ValueError("bundle_schema_versionは2.0にしてください。")
    if not isinstance(value.get("stage_id"), str) or not value["stage_id"].strip():
        raise ValueError("stage_idが必要です。")
    groups = [value.get(name) for name in ("controls", "incoming_context", "predicted_outputs")]
    if any(not isinstance(group, list) or any(not isinstance(item, str) or not item for item in group) for group in groups):
        raise ValueError("入出力の役割は変数名listにしてください。")
    controls, incoming, outputs = groups
    names = controls + incoming
    # 状態の更新や引継ぎでは入力と出力に同名が現れる。各namespace内だけ検査。
    if not controls or not outputs or len(set(names)) != len(names) or len(set(outputs)) != len(outputs):
        raise ValueError("入力内および出力内の名前は重複なく定義してください。")
    bounds = value.get("input_bounds", {})
    if not isinstance(bounds, dict) or set(bounds) != set(names):
        raise ValueError("input_boundsは全入力と一致させてください。")
    for name in names:
        pair = bounds[name]
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            raise ValueError(f"{name}の範囲は[下限,上限]にしてください。")
        lo, hi = map(float, pair)
        if not math.isfinite(lo) or not math.isfinite(hi) or lo >= hi or not math.isfinite(hi - lo):
            raise ValueError(f"{name}の範囲が不正です。")
        bounds[name] = [lo, hi]
    for field, expected in (("input_units", names), ("output_units", outputs)):
        units = value.get(field, {})
        if not isinstance(units, dict) or set(units) != set(expected) or any(not isinstance(v, str) for v in units.values()):
            raise ValueError(f"{field}は全変数の単位文字列を指定してください。")
        # 単工程CSVの空単位は無次元。連結先では明示的な「1」へ統一する。
        value[field] = {name: unit.strip() or "1" for name, unit in units.items()}
    value["derived_outputs"] = validate_derived_outputs(
        value.get("derived_outputs", {}), set(outputs), value["output_units"])
    objective = value.get("local_objective", {})
    if not isinstance(objective, dict) or objective.get("name") not in outputs or objective.get("direction") not in ("minimize", "maximize"):
        raise ValueError("local_objectiveは予測出力の最小化または最大化にしてください。")
    constraints = value.get("local_constraints", [])
    if not isinstance(constraints, list):
        raise ValueError("local_constraintsはlistにしてください。")
    for constraint in constraints:
        if not isinstance(constraint, dict) or constraint.get("name") not in outputs or constraint.get("direction") not in ("less_equal", "greater_equal"):
            raise ValueError("local_constraintsの出力または方向が不正です。")
        if not math.isfinite(float(constraint.get("target", float("nan")))):
            raise ValueError("制約targetは有限の値にしてください。")
        constraint.setdefault("margin_scale", max(abs(float(constraint["target"])) * 0.1, 1e-6))
        if not math.isfinite(float(constraint["margin_scale"])) or float(constraint["margin_scale"]) <= 0:
            raise ValueError("margin_scaleは有限で正の値にしてください。")
        constraint["target"], constraint["margin_scale"] = float(constraint["target"]), float(constraint["margin_scale"])
    if not isinstance(value.get("capabilities", {}), dict):
        raise ValueError("capabilitiesはobjectにしてください。")
    return value
