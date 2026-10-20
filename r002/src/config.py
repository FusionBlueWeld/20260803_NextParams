"""工程グラフとbundle契約を、予測や大きな配列生成の前に検査します。"""

from __future__ import annotations

import copy
import json
import math
from pathlib import Path

import numpy as np

from .settings import DEFAULT_SEARCH


class ConfigurationError(ValueError):
    """設定または工程接続の不整合。"""


def read_config(source: dict | str | Path) -> dict:
    """辞書またはtrial/pipeline.jsonから設定を読み、相対bundle基準を保持します。"""
    if isinstance(source, dict):
        return copy.deepcopy(source)
    path = Path(source).resolve()
    path = path / "pipeline.json" if path.is_dir() else path
    config = json.loads(path.read_text(encoding="utf-8-sig"))
    config["_base_path"] = str(path.parent)
    return config


def qualified(value: str) -> tuple[str, str]:
    """工程IDを含む変数参照だけを許可し、同名操作の衝突を防ぎます。"""
    if not isinstance(value, str) or value.count(".") != 1:
        raise ConfigurationError(f"変数はstage_id.nameで指定してください: {value}")
    stage, name = value.split(".")
    if not stage or not name:
        raise ConfigurationError(f"工程名または変数名が空です: {value}")
    return stage, name


def validate_spec(spec: dict, outputs: set[str]) -> None:
    """仕様の方向、有限目標、余裕の尺度を検査します。"""
    if spec.get("name") not in outputs:
        raise ConfigurationError(f"仕様出力が未定義です: {spec.get('name')}")
    direction = spec.get("direction")
    target = spec.get("target")
    if direction == "between":
        if not isinstance(target, (list, tuple)) or len(target) != 2:
            raise ConfigurationError("betweenのtargetは[下限,上限]です。")
        lower, upper = map(float, target)
        if not np.isfinite([lower, upper]).all() or lower >= upper:
            raise ConfigurationError("betweenの目標範囲が不正です。")
    elif direction in {"greater_equal", "less_equal"}:
        if not np.isfinite(float(target)):
            raise ConfigurationError("目標値は有限値です。")
    else:
        raise ConfigurationError(f"仕様方向が不正です: {direction}")
    scale = float(spec.get("margin_scale", 1.0))
    if not math.isfinite(scale) or scale <= 0:
        raise ConfigurationError("margin_scaleは正の有限値です。")


def validate_config(config: dict, predictors: dict) -> dict:
    """分岐合流DAG・供給元・単位・予算・実施済み操作を検査します。"""
    if config.get("schema_version") != "2.0":
        raise ConfigurationError("pipeline schema_versionは2.0にしてください。")
    stages = config.get("stages", [])
    ids = [item.get("id") for item in stages]
    if not ids or len(set(ids)) != len(ids) or any(not name or "." in name for name in ids):
        raise ConfigurationError("stage idは空欄・重複・ドットなしで指定します。")
    if set(predictors) != set(ids):
        raise ConfigurationError("予測器の工程IDがstagesと一致しません。")
    controls, inputs, outputs = {}, {}, {}
    for stage in ids:
        manifest, _ = predictors[stage]
        if manifest.get("stage_id") != stage:
            raise ConfigurationError(f"bundleのstage_idが工程IDと一致しません: {stage}")
        if str(manifest.get("bundle_schema_version")) != "2.0":
            raise ConfigurationError("r002はbundle2.0を使用します。旧bundleはv005で再出力してください。")
        if not manifest.get("capabilities", {}).get("mean_prediction"):
            raise ConfigurationError(f"{stage}は平均予測に対応していません。")
        names = list(manifest.get("controls", [])) + list(manifest.get("incoming_context", []))
        if len(set(names)) != len(names):
            raise ConfigurationError(f"{stage}の操作と流入状態が重複しています。")
        for name in names:
            if not name or "." in name:
                raise ConfigurationError("bundleの変数名は空欄・ドットなしにしてください。")
            bounds = manifest.get("input_bounds", {}).get(name)
            if bounds is None or len(bounds) != 2 or not np.isfinite(bounds).all() or bounds[0] >= bounds[1]:
                raise ConfigurationError(f"{stage}.{name}の入力範囲が不正です。")
            inputs[f"{stage}.{name}"] = tuple(map(float, bounds))
        if set(manifest.get("input_units", {})) != set(names):
            raise ConfigurationError(f"{stage}の入力単位定義が入力変数と一致しません。")
        if any(not isinstance(unit, str) or not unit for unit in manifest["input_units"].values()):
            raise ConfigurationError("入力単位は空でない文字列です。無次元は1を使います。")
        controls.update({f"{stage}.{name}": inputs[f"{stage}.{name}"] for name in manifest.get("controls", [])})
        outputs[stage] = set(manifest.get("predicted_outputs", []))
        if not outputs[stage] or len(outputs[stage]) != len(manifest.get("predicted_outputs", [])):
            raise ConfigurationError(f"{stage}の出力がありません。")
        if set(manifest.get("output_units", {})) != outputs[stage] or any(not isinstance(unit, str) or not unit for unit in manifest["output_units"].values()):
            raise ConfigurationError(f"{stage}の出力単位定義が出力変数と一致しません。")
        objective = manifest.get("local_objective", {})
        if objective.get("name") not in outputs[stage] or objective.get("direction") not in {"maximize", "minimize"}:
            raise ConfigurationError(f"{stage}の局所目的が不正です。")
        for spec in manifest.get("local_constraints", []):
            validate_spec(spec, outputs[stage])
    edges, dependencies = {}, {name: set() for name in ids}
    for row in config.get("connections", []):
        source_stage, source_name = qualified(row["from"])
        target_stage, target_name = qualified(row["to"])
        if source_name not in outputs.get(source_stage, set()) or target_stage not in predictors:
            raise ConfigurationError(f"接続の工程・出力が未定義です: {row}")
        target_manifest = predictors[target_stage][0]
        if target_name not in target_manifest.get("incoming_context", []):
            raise ConfigurationError(f"接続先は流入状態です: {row['to']}")
        if row["to"] in edges:
            raise ConfigurationError(f"入力供給元が重複しています: {row['to']}")
        source_unit = predictors[source_stage][0].get("output_units", {}).get(source_name)
        target_unit = target_manifest.get("input_units", {}).get(target_name)
        if source_unit is None or source_unit != target_unit:
            raise ConfigurationError(f"接続単位が一致しません: {row}")
        dependencies[target_stage].add(source_stage)
        edges[row["to"]] = row["from"]
    order = []
    pending = list(ids)
    while pending:
        ready = [name for name in pending if dependencies[name] <= set(order)]
        if not ready:
            raise ConfigurationError("工程グラフに循環があります。物理状態の伝播はDAGにしてください。")
        order.extend(ready)
        pending = [name for name in pending if name not in ready]
    external_required = set(inputs) - set(controls) - set(edges)
    external = config.get("external_context", {})
    if set(external) != external_required:
        raise ConfigurationError(f"external_contextが一致しません: required={sorted(external_required)}")
    for name, value in external.items():
        lo, hi = inputs[name]
        if not np.isfinite(float(value)) or not lo <= float(value) <= hi:
            raise ConfigurationError(f"外部流入状態が適用範囲外です: {name}")
    axes = config.get("candidate_axes", {})
    if not controls:
        raise ConfigurationError("操作条件を1つ以上定義してください。")
    if set(axes) != set(controls):
        raise ConfigurationError("candidate_axesは全stage_id.controlを指定してください。")
    for name, axis in axes.items():
        array = np.asarray(axis, dtype=float)
        lo, hi = controls[name]
        if array.ndim != 1 or not len(array) or not np.isfinite(array).all() or np.any((array < lo) | (array > hi)):
            raise ConfigurationError(f"候補軸が不正です: {name}")
    fixed = config.get("fixed_controls", {})
    if not set(fixed) <= set(controls):
        raise ConfigurationError("fixed_controlsに未定義操作があります。")
    for name, value in fixed.items():
        lo, hi = controls[name]
        if not np.isfinite(float(value)) or not lo <= float(value) <= hi:
            raise ConfigurationError(f"固定操作が適用範囲外です: {name}")
    observed_stages = set(config.get("completed_stages", []))
    supplied_outputs = {**config.get("observed_outputs", {}), **config.get("hypothetical_outputs", {})}
    if set(config.get("observed_outputs", {})) & set(config.get("hypothetical_outputs", {})):
        raise ConfigurationError("同じ出力を実測値と仮想状態の両方に指定できません。")
    for name, value in supplied_outputs.items():
        stage, variable = qualified(name)
        if variable not in outputs.get(stage, set()) or not np.isfinite(float(value)):
            raise ConfigurationError(f"観測出力が不正です: {name}")
        observed_stages.add(stage)
    if not observed_stages <= set(ids):
        raise ConfigurationError("completed_stagesに未定義工程があります。")
    # 測定済み工程の上流まで実施済みとみなし、後から変更しないようにします。
    completed = set(observed_stages)
    for stage in reversed(order):
        if stage in completed:
            completed.update(dependencies[stage])
    required_fixed = {name for name in controls if qualified(name)[0] in completed}
    if not required_fixed <= set(fixed):
        raise ConfigurationError(f"実施済み工程と上流の全操作を固定してください: {sorted(required_fixed - set(fixed))}")
    for spec in config.get("final_specifications", []):
        stage = spec.get("stage", order[-1])
        if stage not in outputs:
            raise ConfigurationError(f"最終仕様の工程が未定義です: {stage}")
        validate_spec(spec, outputs[stage])
    if not config.get("final_specifications"):
        raise ConfigurationError("最終仕様を1つ以上指定してください。")
    search = {**DEFAULT_SEARCH, **config.get("search", {})}
    for key in ("max_evaluations", "population_size", "batch_size"):
        if int(search[key]) != search[key] or int(search[key]) < 1:
            raise ConfigurationError(f"search.{key}は正の整数です。")
    if int(search["population_size"]) > 2048:
        raise ConfigurationError("population_sizeは2048以下です。")
    if int(search["seed"]) != search["seed"] or int(search["seed"]) < 0:
        raise ConfigurationError("seedは非負整数です。")
    if int(search.get("cache_size", 4096)) < 1 or int(search.get("cache_size", 4096)) > 65536:
        raise ConfigurationError("cache_sizeは1以上65536以下です。")
    if int(search["batch_size"]) > 512:
        raise ConfigurationError("batch_sizeは512以下です。")
    if not np.isfinite(float(search["max_seconds"])) or float(search["max_seconds"]) <= 0:
        raise ConfigurationError("max_secondsは正の有限値です。")
    if not 0 <= float(search["support_threshold"]) < 1:
        raise ConfigurationError("support_thresholdは0以上1未満です。")
    multiplier = float(search.get("conservative_std_multiplier", 0.0))
    if not math.isfinite(multiplier) or multiplier < 0:
        raise ConfigurationError("conservative_std_multiplierは非負有限値です。校正済み確率ではありません。")
    continuous = set(config.get("continuous_controls", []))
    if not continuous <= set(controls):
        raise ConfigurationError("continuous_controlsに未定義操作があります。")
    return {"order": order, "edges": edges, "controls": controls, "search": search, "completed_stages": completed}
