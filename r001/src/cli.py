"""r001: v004 stage bundleを直列接続し、固定候補から全体条件を選びます。"""

from __future__ import annotations

import argparse
import copy
import csv
import importlib
import itertools
import json
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

from .window_calibration import interval_bounds, validate_calibration


VERSION_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = VERSION_ROOT.parent
TRIALS_ROOT = VERSION_ROOT / "trials"
CONFIG_NAME = "pipeline.json"


class UserInputError(Exception):
    pass


def _trial(name: str) -> Path:
    if not name.startswith("trial_") or not all(c.isalnum() or c in "_-" for c in name):
        raise UserInputError("trial名はtrial_で始め、半角英数字、_、-だけで指定してください。")
    return TRIALS_ROOT / name


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    header = list(rows[0]) if rows else []
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=header)
        if header:
            writer.writeheader()
            writer.writerows(rows)
    temporary.replace(path)


def create_trial(name: str) -> None:
    path = _trial(name)
    if path.exists():
        raise UserInputError(f"{name}は既に存在します。")
    TRIALS_ROOT.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{name}_", dir=TRIALS_ROOT))
    try:
        (temporary / "output").mkdir()
        template = {
            "schema_version": "1.0",
            "stages": [{"id": "stage_1", "bundle": "../../../v004/trials/trial_example/output/stage_bundles/stage_bundle_run_001"}],
            "connections": [],
            "external_context": {},
            "candidate_axes": {},
            "final_specifications": [],
            "connected_window": {
                "support_threshold": 0.5,
                "interval_method": "std_multiplier",
                "confidence_z": 1.645,
                "enforce_local_constraints": True,
                "use_for_selection": True,
                "scan_points": 121,
            },
            "window_center_selection": {
                "enabled": False,
                "candidate_limit": 64,
                "coarse_scan_points": 41,
                "required_variations": {},
            },
            "simultaneous_window": {
                "enabled": False,
                "samples": 4096,
                "seed": 20260923,
                "radius_scales": [0.5, 1.0, 1.5],
                "pair_grid_points": 31,
                "pair_count": 3,
            },
        }
        _write_json(temporary / CONFIG_NAME, template)
        temporary.rename(path)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    print(f"{name}を作成しました。\n設定: {path / CONFIG_NAME}")


def _read_config(path: Path) -> dict[str, object]:
    try:
        config = json.loads((path / CONFIG_NAME).read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as error:
        raise UserInputError(f"{CONFIG_NAME}を読み取れません: {error}") from error
    if config.get("schema_version") != "1.0":
        raise UserInputError("pipeline.jsonのschema_versionは1.0にしてください。")
    stages = config.get("stages")
    if not isinstance(stages, list) or not stages:
        raise UserInputError("stagesを1件以上指定してください。")
    ids = [item.get("id") for item in stages]
    if len(set(ids)) != len(ids) or any(not item for item in ids):
        raise UserInputError("stage idは空欄・重複なしで指定してください。")
    return config


def _bundle_path(trial: Path, value: str) -> Path:
    path = Path(value)
    return (path if path.is_absolute() else trial / path).resolve()


def _manifest(path: Path) -> dict[str, object]:
    try:
        value = json.loads((path / "stage_manifest.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise UserInputError(f"stage bundleを読み取れません: {path}: {error}") from error
    if value.get("bundle_schema_version") != "1.0" or not value.get("capabilities", {}).get("mean_prediction"):
        raise UserInputError(f"平均予測に対応しないstage bundleです: {path}")
    return value


def _validate(trial: Path, config: dict[str, object]) -> list[tuple[dict, Path, dict]]:
    stages = []
    for item in config["stages"]:
        path = _bundle_path(trial, str(item.get("bundle", "")))
        stages.append((item, path, _manifest(path)))
    stage_ids = {str(item[0]["id"]) for item in stages}
    stage_order = {str(item[0]["id"]): index for index, item in enumerate(stages)}
    produced = {(str(item[0]["id"]), name) for item in stages for name in item[2]["connector_outputs"]}
    targets: set[tuple[str, str]] = set()
    for connection in config.get("connections", []):
        try:
            source_stage, source_name = str(connection["from"]).split(".", 1)
            target_stage, target_name = str(connection["to"]).split(".", 1)
        except (KeyError, ValueError) as error:
            raise UserInputError("connectionsのfrom/toはstage.variable形式で指定してください。") from error
        if (source_stage, source_name) not in produced:
            raise UserInputError(f"接続元がbundleのconnector_outputsにありません: {connection['from']}")
        target = next((item for item in stages if str(item[0]["id"]) == target_stage), None)
        if target is None or target_name not in target[2]["incoming_context"]:
            raise UserInputError(f"接続先がbundleのincoming_contextにありません: {connection['to']}")
        if stage_order[source_stage] >= stage_order[target_stage]:
            raise UserInputError(f"初版は順方向の直列接続だけを受け付けます: {connection['from']} -> {connection['to']}")
        key = (target_stage, target_name)
        if key in targets:
            raise UserInputError(f"接続先が重複しています: {connection['to']}")
        targets.add(key)
        source = next(item for item in stages if str(item[0]["id"]) == source_stage)
        source_unit = source[2].get("output_units", {}).get(source_name)
        target_unit = target[2].get("input_units", {}).get(target_name)
        if source_unit is None or target_unit is None or source_unit != target_unit:
            raise UserInputError(f"接続単位が一致しません: {connection['from']}[{source_unit}] -> {connection['to']}[{target_unit}]")
    axes = config.get("candidate_axes", {})
    all_controls = {name for _, _, manifest in stages for name in manifest["controls"]}
    if set(axes) != all_controls:
        raise UserInputError(f"candidate_axesは全controlを指定してください: required={sorted(all_controls)}")
    for name, values in axes.items():
        if not isinstance(values, list) or not values:
            raise UserInputError(f"candidate_axes.{name}は1件以上の配列にしてください。")
    external = set(config.get("external_context", {}))
    required_external = {
        f"{item[0]['id']}.{name}"
        for item in stages for name in item[2]["incoming_context"]
        if (str(item[0]["id"]), name) not in targets
    }
    if external != required_external:
        raise UserInputError(f"external_contextが一致しません: required={sorted(required_external)}")
    final_outputs = set(stages[-1][2]["predicted_outputs"])
    for spec in config.get("final_specifications", []):
        if spec.get("name") not in final_outputs:
            raise UserInputError(f"最終仕様の出力が最終stageにありません: {spec.get('name')}")
        if spec.get("direction") not in {"greater_equal", "less_equal", "between"}:
            raise UserInputError("最終仕様のdirectionが不正です。")
        if "margin_scale" in spec and (not np.isfinite(float(spec["margin_scale"])) or float(spec["margin_scale"]) <= 0):
            raise UserInputError("最終仕様のmargin_scaleは0より大きい有限値にしてください。")
    variation = config.get("process_variation")
    if variation is not None:
        if not isinstance(variation, dict):
            raise UserInputError("process_variationはobjectで指定してください。")
        if variation.get("distribution") != "independent_clipped_normal":
            raise UserInputError("process_variation.distributionはindependent_clipped_normalにしてください。")
        samples = int(variation.get("samples", 0))
        candidate_limit = int(variation.get("candidate_limit", 0))
        seed = int(variation.get("seed", -1))
        if samples < 100 or candidate_limit < 1 or seed < 0:
            raise UserInputError("process_variationはsamples>=100、candidate_limit>=1、seed>=0にしてください。")
        deviations = variation.get("standard_deviations")
        varied_names = set(axes) | required_external
        if not isinstance(deviations, dict) or set(deviations) != varied_names:
            raise UserInputError(f"process_variation.standard_deviationsが一致しません: required={sorted(varied_names)}")
        if any(not np.isfinite(float(value)) or float(value) < 0 for value in deviations.values()):
            raise UserInputError("process_variationの標準偏差は0以上の有限値にしてください。")
    window = config.get("connected_window")
    if window is not None:
        if not isinstance(window, dict):
            raise UserInputError("connected_windowはobjectで指定してください。")
        threshold = float(window.get("support_threshold", 0.5))
        interval_method = str(window.get("interval_method", "std_multiplier"))
        confidence_z = float(window.get("confidence_z", 0.0))
        scan_points = int(window.get("scan_points", 121))
        if not np.isfinite(threshold) or not 0 <= threshold < 1:
            raise UserInputError("connected_window.support_thresholdは0以上1未満にしてください。")
        if interval_method not in {"std_multiplier", "residual_quantile", "standardized_residual_quantile"}:
            raise UserInputError("connected_window.interval_methodが不正です。")
        if not np.isfinite(confidence_z) or not 0 <= confidence_z <= 5:
            raise UserInputError("connected_window.confidence_zは0以上5以下にしてください。")
        if scan_points < 11 or scan_points > 2001:
            raise UserInputError("connected_window.scan_pointsは11以上2001以下にしてください。")
        for key in ("enforce_local_constraints", "use_for_selection"):
            if key in window and not isinstance(window[key], bool):
                raise UserInputError(f"connected_window.{key}はtrue/falseにしてください。")
        if interval_method in {"residual_quantile", "standardized_residual_quantile"}:
            raw_path = window.get("calibration_file")
            if not isinstance(raw_path, str) or not raw_path:
                raise UserInputError("residual_quantileにはconnected_window.calibration_fileが必要です。")
            calibration_path = Path(raw_path)
            calibration_path = (calibration_path if calibration_path.is_absolute() else trial / calibration_path).resolve()
            try:
                calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as error:
                raise UserInputError(f"連結窓の残差調整情報を読み取れません: {error}") from error
            try:
                validate_calibration(calibration, config, stages)
            except (ValueError, TypeError, KeyError) as error:
                raise UserInputError(str(error)) from error
            config["_connected_window_calibration"] = calibration
    center_selection = config.get("window_center_selection")
    if center_selection is not None:
        if not isinstance(center_selection, dict) or not isinstance(center_selection.get("enabled", False), bool):
            raise UserInputError("window_center_selectionの形式が不正です。")
        limit = int(center_selection.get("candidate_limit", 0))
        coarse = int(center_selection.get("coarse_scan_points", 0))
        if limit < 1 or coarse < 11 or coarse > 501:
            raise UserInputError("window_center_selectionのcandidate_limit>=1、coarse_scan_points=11..501が必要です。")
        variations = center_selection.get("required_variations", {})
        if center_selection.get("enabled") and set(variations) != set(axes):
            raise UserInputError("window_center_selection.required_variationsは全controlを指定してください。")
        if any(not np.isfinite(float(value)) or float(value) <= 0 for value in variations.values()):
            raise UserInputError("required_variationsは0より大きい有限値にしてください。")
    simultaneous = config.get("simultaneous_window")
    if simultaneous is not None:
        if not isinstance(simultaneous, dict) or not isinstance(simultaneous.get("enabled", False), bool):
            raise UserInputError("simultaneous_windowの形式が不正です。")
        if int(simultaneous.get("samples", 0)) < 100 or int(simultaneous.get("seed", -1)) < 0:
            raise UserInputError("simultaneous_windowはsamples>=100、seed>=0にしてください。")
        scales = simultaneous.get("radius_scales", [])
        if not isinstance(scales, list) or not scales or any(not np.isfinite(float(v)) or float(v) <= 0 for v in scales):
            raise UserInputError("simultaneous_window.radius_scalesは正の数値配列にしてください。")
        if int(simultaneous.get("pair_grid_points", 0)) < 5 or int(simultaneous.get("pair_count", 0)) < 1:
            raise UserInputError("simultaneous_windowのpair_grid_points>=5、pair_count>=1が必要です。")
    return stages


def prepare_trial(name: str) -> None:
    trial = _trial(name)
    if not trial.is_dir():
        raise UserInputError(f"{name}が見つかりません。")
    stages = _validate(trial, _read_config(trial))
    print(f"pipeline.jsonを確認しました。工程数={len(stages)}")


def _load_bundle_runtime():
    """r001のsrc名前空間を退避し、v004が保存した型でbundleを復元します。"""
    for name in list(sys.modules):
        if name == "src" or name.startswith("src."):
            del sys.modules[name]
    sys.path.insert(0, str(PROJECT_ROOT / "v004"))
    sys.path.insert(1, str(PROJECT_ROOT))
    return importlib.import_module("src.stage_bundle").load_stage_bundle


def _margin_and_feasible(outputs: dict[str, np.ndarray], specs: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    count = len(next(iter(outputs.values())))
    feasible = np.ones(count, dtype=bool)
    margins = []
    for spec in specs:
        values = np.asarray(outputs[spec["name"]], dtype=float)
        direction, target = spec["direction"], spec["target"]
        configured_scale = spec.get("margin_scale")
        if direction == "greater_equal":
            feasible &= values >= float(target)
            scale = abs(float(target)) if configured_scale is None else float(configured_scale)
            margins.append((values - float(target)) / max(scale, 1e-12))
        elif direction == "less_equal":
            feasible &= values <= float(target)
            scale = abs(float(target)) if configured_scale is None else float(configured_scale)
            margins.append((float(target) - values) / max(scale, 1e-12))
        else:
            lower, upper = map(float, target)
            feasible &= (values >= lower) & (values <= upper)
            scale = max((upper - lower) / 2 if configured_scale is None else float(configured_scale), 1e-12)
            margins.extend([(values - lower) / scale, (upper - values) / scale])
    margin = np.min(np.stack(margins), axis=0) if margins else np.zeros(count)
    finite = np.all(np.stack([np.isfinite(value) for value in outputs.values()]), axis=0)
    return np.where(finite, margin, -np.inf), feasible & finite


def _predict_allowing_invalid(predictor, manifest: dict[str, object], inputs: dict[str, object]) -> dict[str, object]:
    """範囲外接続を候補単位で除外し、他の候補の探索を継続します。"""
    names = [*manifest["controls"], *manifest["incoming_context"]]
    arrays = np.broadcast_arrays(*[np.asarray(inputs[name], dtype=float) for name in names])
    shape = arrays[0].shape
    flat = {name: array.reshape(-1) for name, array in zip(names, arrays)}
    if hasattr(predictor, "problem"):
        bounds = {item.column: (item.lower, item.upper) for item in predictor.problem.parameters}
    else:
        bounds = predictor.bounds
    valid = np.ones(len(next(iter(flat.values()))), dtype=bool)
    for name, values in flat.items():
        lower, upper = bounds[name]
        valid &= np.isfinite(values) & (values >= lower) & (values <= upper)
    outputs = {
        name: {"mean": np.full(valid.size, np.nan), "std": np.full(valid.size, np.nan)}
        for name in manifest["predicted_outputs"]
    }
    support = np.zeros(valid.size)
    if np.any(valid):
        predicted = predictor.predict_arrays({name: values[valid] for name, values in flat.items()})
        for name in outputs:
            outputs[name]["mean"][valid] = np.asarray(predicted["outputs"][name]["mean"]).reshape(-1)
            outputs[name]["std"][valid] = np.asarray(predicted["outputs"][name]["std"]).reshape(-1)
        support[valid] = np.asarray(predicted["support"]).reshape(-1)
    return {
        "outputs": {name: {kind: value.reshape(shape) for kind, value in result.items()} for name, result in outputs.items()},
        "support": support.reshape(shape),
        "valid_connection": valid.reshape(shape),
    }


def _run_stage_chain(predictors, config: dict[str, object], connections: dict[str, str], values: dict[str, object]):
    """同じ候補形状を保ったまま、全stageの平均予測を順方向に伝播します。"""
    stage_outputs: dict[str, dict[str, np.ndarray]] = {}
    supports = []
    valid_connections = []
    for item, manifest, predictor in predictors:
        stage_id = str(item["id"])
        inputs = {name: values[name] for name in manifest["controls"]}
        for incoming in manifest["incoming_context"]:
            target = f"{stage_id}.{incoming}"
            if target in connections:
                source_stage, source_name = connections[target].split(".", 1)
                inputs[incoming] = stage_outputs[source_stage][source_name]
            else:
                inputs[incoming] = values.get(target, float(config["external_context"][target]))
        predicted = _predict_allowing_invalid(predictor, manifest, inputs)
        stage_outputs[stage_id] = {key: np.asarray(value["mean"]) for key, value in predicted["outputs"].items()}
        supports.append(np.asarray(predicted["support"]))
        valid_connections.append(np.asarray(predicted["valid_connection"], dtype=bool))
    return stage_outputs, np.min(np.stack(supports), axis=0), np.all(np.stack(valid_connections), axis=0)


def _input_bounds(predictor) -> dict[str, tuple[float, float]]:
    if hasattr(predictor, "problem"):
        return {item.column: (float(item.lower), float(item.upper)) for item in predictor.problem.parameters}
    return {name: tuple(map(float, bounds)) for name, bounds in predictor.bounds.items()}


def _constraint_margin_components(
    outputs: dict[str, np.ndarray], specifications: list[dict], prefix: str,
    standard_deviations: dict[str, np.ndarray] | None = None, confidence_z: float = 0.0,
    residual_offsets: dict[str, dict[str, float]] | None = None,
    interval_method: str = "std_multiplier",
) -> dict[str, np.ndarray]:
    """Return signed normalized distance to every constraint boundary."""
    components: dict[str, np.ndarray] = {}
    for spec in specifications:
        name = str(spec["name"])
        values = np.asarray(outputs[name], dtype=float)
        std = np.zeros_like(values) if standard_deviations is None else np.asarray(standard_deviations[name], dtype=float)
        offset = (residual_offsets or {}).get(name)
        lower_value, upper_value = interval_bounds(values, std, interval_method, offset, confidence_z)
        direction, target = spec["direction"], spec["target"]
        configured = spec.get("margin_scale")
        if direction == "greater_equal":
            scale = abs(float(target)) if configured is None else float(configured)
            components[f"{prefix}.{name}:lower"] = (lower_value - float(target)) / max(scale, 1e-12)
        elif direction == "less_equal":
            scale = abs(float(target)) if configured is None else float(configured)
            components[f"{prefix}.{name}:upper"] = (float(target) - upper_value) / max(scale, 1e-12)
        else:
            lower, upper = map(float, target)
            scale = (upper - lower) / 2 if configured is None else float(configured)
            scale = max(scale, 1e-12)
            components[f"{prefix}.{name}:lower"] = (lower_value - lower) / scale
            components[f"{prefix}.{name}:upper"] = (upper - upper_value) / scale
    return components


def _run_connected_window(
    predictors, config: dict[str, object], connections: dict[str, str], values: dict[str, object],
) -> dict[str, object]:
    """Evaluate the intersection of stage, connection, final, and support windows."""
    stage_outputs: dict[str, dict[str, np.ndarray]] = {}
    stage_stds: dict[str, dict[str, np.ndarray]] = {}
    stage_support: dict[str, np.ndarray] = {}
    valid_rows = []
    process_components: dict[str, np.ndarray] = {}
    window_config = config.get("connected_window") or {}
    enforce_local = bool(window_config.get("enforce_local_constraints", False))
    threshold = float(window_config.get("support_threshold", 0.0))
    confidence_z = float(window_config.get("confidence_z", 0.0))
    interval_method = window_config.get("interval_method", "std_multiplier")
    residual_mode = interval_method in {"residual_quantile", "standardized_residual_quantile"}
    calibration_offsets = (config.get("_connected_window_calibration") or {}).get("offsets", {}) if residual_mode else {}
    if residual_mode:
        for name, expected in config["external_context"].items():
            if not np.all(np.asarray(values.get(name, expected)) == expected):
                raise UserInputError("残差調整情報の固定原料状態と評価入力が異なります。再校正してください。")
    for item, manifest, predictor in predictors:
        stage_id = str(item["id"])
        inputs = {name: values[name] for name in manifest["controls"]}
        bounds = _input_bounds(predictor)
        for incoming in manifest["incoming_context"]:
            target_name = f"{stage_id}.{incoming}"
            if target_name in connections:
                source_stage, source_name = connections[target_name].split(".", 1)
                inputs[incoming] = stage_outputs[source_stage][source_name]
                incoming_std = stage_stds[source_stage][source_name]
                incoming_offset = calibration_offsets.get(source_stage, {}).get(source_name)
            else:
                inputs[incoming] = values.get(target_name, float(config["external_context"][target_name]))
                incoming_std = np.zeros_like(np.asarray(inputs[incoming], dtype=float))
                incoming_offset = None
            incoming_values = np.asarray(inputs[incoming], dtype=float)
            lower, upper = bounds[incoming]
            scale = max((upper - lower) / 2, 1e-12)
            if target_name in connections:
                incoming_lower, incoming_upper = interval_bounds(
                    incoming_values, incoming_std, interval_method, incoming_offset, confidence_z)
            else:
                incoming_lower = incoming_upper = incoming_values
            process_components[f"connection.{target_name}:lower"] = (incoming_lower - lower) / scale
            process_components[f"connection.{target_name}:upper"] = (upper - incoming_upper) / scale
        predicted = _predict_allowing_invalid(predictor, manifest, inputs)
        stage_outputs[stage_id] = {
            name: np.asarray(value["mean"], dtype=float) for name, value in predicted["outputs"].items()
        }
        stage_stds[stage_id] = {
            name: np.asarray(value["std"], dtype=float) for name, value in predicted["outputs"].items()
        }
        stage_support[stage_id] = np.asarray(predicted["support"], dtype=float)
        valid_rows.append(np.asarray(predicted["valid_connection"], dtype=bool))
        if enforce_local:
            process_components.update(_constraint_margin_components(
                stage_outputs[stage_id], manifest.get("local_constraints", []), f"local.{stage_id}",
                stage_stds[stage_id], confidence_z,
                calibration_offsets.get(stage_id),
                interval_method,
            ))
    final_stage = str(predictors[-1][0]["id"])
    process_components.update(_constraint_margin_components(
        stage_outputs[final_stage], config.get("final_specifications", []), "final",
        stage_stds[final_stage], confidence_z,
        calibration_offsets.get(final_stage),
        interval_method,
    ))
    component_names = list(process_components)
    component_arrays = [np.asarray(process_components[name], dtype=float) for name in component_names]
    shape = np.broadcast_arrays(*component_arrays)[0].shape
    finite_components = [np.where(np.isfinite(value), value, -np.inf) for value in component_arrays]
    process_stack = np.stack(np.broadcast_arrays(*finite_components), axis=0)
    process_margin = np.min(process_stack, axis=0)
    bottleneck_indices = np.argmin(process_stack, axis=0)
    bottleneck = np.asarray(component_names, dtype=object)[bottleneck_indices]
    valid = np.all(np.stack(np.broadcast_arrays(*valid_rows), axis=0), axis=0)
    minimum_support = np.min(np.stack(np.broadcast_arrays(*stage_support.values()), axis=0), axis=0)
    support_margin = (minimum_support - threshold) / max(1.0 - threshold, 1e-12)
    trusted_margin = np.minimum(process_margin, support_margin)
    process_feasible = valid & (process_margin >= 0)
    trusted_feasible = process_feasible & (minimum_support >= threshold)
    return {
        "stage_outputs": stage_outputs, "stage_stds": stage_stds, "stage_support": stage_support,
        "valid_connection": valid, "minimum_support": minimum_support,
        "process_components": process_components, "process_margin": process_margin,
        "process_feasible": process_feasible, "support_margin": support_margin,
        "trusted_margin": trusted_margin, "trusted_feasible": trusted_feasible,
        "bottleneck": np.asarray(bottleneck, dtype=object).reshape(shape),
    }


def _contiguous_true_interval(axis: np.ndarray, mask: np.ndarray, center: float) -> tuple[float | None, float | None]:
    """Return the feasible grid segment containing the point nearest center."""
    nearest = int(np.argmin(np.abs(axis - center)))
    if not mask[nearest]:
        return None, None
    lower = upper = nearest
    while lower > 0 and mask[lower - 1]:
        lower -= 1
    while upper + 1 < len(mask) and mask[upper + 1]:
        upper += 1
    return float(axis[lower]), float(axis[upper])


def _all_true_intervals(axis: np.ndarray, mask: np.ndarray) -> list[list[float]]:
    """Keep disconnected feasible islands separate."""
    intervals: list[list[float]] = []
    start = None
    for index, passed in enumerate(np.asarray(mask, dtype=bool)):
        if passed and start is None:
            start = index
        if start is not None and (not passed or index == len(mask) - 1):
            end = index if passed and index == len(mask) - 1 else index - 1
            intervals.append([float(axis[start]), float(axis[end])])
            start = None
    return intervals


def _boundary_diagnostic(evaluated: dict[str, object], index: int) -> dict[str, object]:
    bottleneck = str(np.asarray(evaluated["bottleneck"])[index])
    row: dict[str, object] = {
        "bottleneck": bottleneck,
        "margin": float(np.asarray(evaluated["process_margin"])[index]),
        "minimum_support": float(np.asarray(evaluated["minimum_support"])[index]),
        "stage_support": {stage: float(np.asarray(value)[index]) for stage, value in evaluated["stage_support"].items()},
    }
    head = bottleneck.split(":", 1)[0]
    parts = head.split(".")
    if parts[0] == "local" and len(parts) >= 3:
        stage, output = parts[1], ".".join(parts[2:])
    elif parts[0] == "final" and len(parts) >= 2:
        stage, output = next(reversed(evaluated["stage_outputs"])), ".".join(parts[1:])
    else:
        stage = output = None
    if stage is not None and output in evaluated["stage_outputs"][stage]:
        mean = float(np.asarray(evaluated["stage_outputs"][stage][output])[index])
        std = float(np.asarray(evaluated["stage_stds"][stage][output])[index])
        row.update({"output_mean": mean, "output_std": std})
    return row


def _connected_window_profiles(
    predictors, config: dict[str, object], connections: dict[str, str], center: dict[str, float],
) -> dict[str, object]:
    """Scan every control axis and report connected-window headroom around center."""
    scan_points = int((config.get("connected_window") or {}).get("scan_points", 121))
    external = dict(config["external_context"])
    bounds = {}
    for _, manifest, predictor in predictors:
        stage_bounds = _input_bounds(predictor)
        bounds.update({name: stage_bounds[name] for name in manifest["controls"]})
    profiles = {}
    mean_config = copy.deepcopy(config)
    mean_config["connected_window"] = {**(mean_config.get("connected_window") or {}),
                                       "interval_method": "std_multiplier", "confidence_z": 0.0,
                                       "support_threshold": 0.0}
    mean_config.pop("_connected_window_calibration", None)
    for name, (lower, upper) in bounds.items():
        axis = np.unique(np.append(np.linspace(lower, upper, scan_points), center[name]))
        values = {control: np.full(len(axis), value) for control, value in center.items()}
        values[name] = axis
        values.update({key: np.full(len(axis), value) for key, value in external.items()})
        mean_evaluated = _run_connected_window(predictors, mean_config, connections, values)
        evaluated = _run_connected_window(predictors, config, connections, values)
        mean_low, mean_high = _contiguous_true_interval(axis, mean_evaluated["process_feasible"], center[name])
        process_low, process_high = _contiguous_true_interval(axis, evaluated["process_feasible"], center[name])
        trust_low, trust_high = _contiguous_true_interval(axis, evaluated["trusted_feasible"], center[name])
        span = upper - lower
        def boundary_rows(result, low, high):
            rows = {}
            for label, boundary in (("lower", low), ("upper", high)):
                if boundary is not None:
                    index = int(np.argmin(np.abs(axis - boundary)))
                    rows[label] = {"control_value": boundary, **_boundary_diagnostic(result, index)}
            return rows
        profiles[name] = {
            "model_input_bounds": [lower, upper],
            "mean_only_window": [mean_low, mean_high],
            "buffered_process_window": [process_low, process_high],
            "process_window": [process_low, process_high],
            "trusted_window": [trust_low, trust_high],
            "mean_only_all_intervals": _all_true_intervals(axis, mean_evaluated["process_feasible"]),
            "buffered_all_intervals": _all_true_intervals(axis, evaluated["process_feasible"]),
            "trusted_all_intervals": _all_true_intervals(axis, evaluated["trusted_feasible"]),
            "boundary_diagnostics": {
                "mean_only": boundary_rows(mean_evaluated, mean_low, mean_high),
                "buffered": boundary_rows(evaluated, process_low, process_high),
                "trusted": boundary_rows(evaluated, trust_low, trust_high),
            },
            "scan_evaluation_points": len(axis),
            "evaluation_unavailable_count": int(np.sum(~np.asarray(evaluated["valid_connection"], dtype=bool))),
            "lower_headroom": None if trust_low is None else center[name] - trust_low,
            "upper_headroom": None if trust_high is None else trust_high - center[name],
            "nearest_normalized_headroom": None if trust_low is None else min(center[name] - trust_low, trust_high - center[name]) / span,
        }
    scalar_values = {**center, **external}
    at_center = _run_connected_window(predictors, config, connections, scalar_values)
    return {
        "definition": "One-control-at-a-time connected feasible interval; interactions require joint scans.",
        "center": center, "center_process_margin": float(at_center["process_margin"]),
        "center_trusted_margin": float(at_center["trusted_margin"]),
        "center_minimum_support": float(at_center["minimum_support"]),
        "center_bottleneck": str(at_center["bottleneck"]), "controls": profiles,
    }


def _diverse_candidate_indices(points: np.ndarray, eligible: np.ndarray, order: np.ndarray, limit: int) -> np.ndarray:
    """Mix high-margin and maximin candidates without claiming global coverage."""
    available = np.flatnonzero(eligible)
    if len(available) <= limit:
        return available
    top_count = max(1, limit // 2)
    chosen = list(np.asarray([index for index in order if eligible[index]][:top_count], dtype=int))
    normalized = (points - points.min(axis=0)) / np.maximum(np.ptp(points, axis=0), 1e-12)
    distance = np.min([np.sum((normalized - normalized[index]) ** 2, axis=1) for index in chosen], axis=0)
    distance[~eligible] = -1
    distance[chosen] = -1
    while len(chosen) < limit:
        index = int(np.argmax(distance))
        chosen.append(index)
        distance = np.minimum(distance, np.sum((normalized - normalized[index]) ** 2, axis=1))
        distance[~eligible] = -1
        distance[chosen] = -1
    return np.asarray(chosen, dtype=int)


def _window_center_comparisons(
    predictors, config: dict[str, object], connections: dict[str, str], points: np.ndarray,
    axis_names: list[str], eligible: np.ndarray, base_order: np.ndarray,
) -> list[dict[str, object]]:
    setting = config["window_center_selection"]
    indices = _diverse_candidate_indices(points, eligible, base_order, int(setting["candidate_limit"]))
    scan_config = copy.deepcopy(config)
    scan_config["connected_window"]["scan_points"] = int(setting["coarse_scan_points"])
    required = {name: float(value) for name, value in setting["required_variations"].items()}
    rows = []
    for index in indices:
        center = {name: float(points[index, column]) for column, name in enumerate(axis_names)}
        profile = _batched_window_headrooms(predictors, scan_config, connections, center)
        normalized_headrooms = {}
        valid = True
        for name, item in profile["controls"].items():
            if item["lower_headroom"] is None:
                valid = False
                normalized_headrooms[name] = None
            else:
                normalized_headrooms[name] = min(item["lower_headroom"], item["upper_headroom"]) / required[name]
        rho = min(value for value in normalized_headrooms.values() if value is not None) if valid else -np.inf
        row = {
            "candidate_index": int(index), "symmetric_headroom_rho": float(rho),
            "connected_window_margin": float(profile["center_trusted_margin"]),
            "minimum_support": float(profile["center_minimum_support"]),
            "bottleneck": profile["center_bottleneck"], "scan_points_per_axis": int(setting["coarse_scan_points"]),
            "normalized_headrooms": normalized_headrooms,
        }
        row.update(center)
        rows.append(row)
    rows.sort(key=lambda row: (row["symmetric_headroom_rho"], row["connected_window_margin"], row["minimum_support"]), reverse=True)
    return rows


def _batched_window_headrooms(
    predictors, config: dict[str, object], connections: dict[str, str], center: dict[str, float],
) -> dict[str, object]:
    """Evaluate all one-axis coarse scans for one center in a single model batch."""
    scan_points = int(config["connected_window"]["scan_points"])
    bounds = {}
    for _, manifest, predictor in predictors:
        stage_bounds = _input_bounds(predictor)
        bounds.update({name: stage_bounds[name] for name in manifest["controls"]})
    axes = {name: np.unique(np.append(np.linspace(*bounds[name], scan_points), center[name])) for name in center}
    offsets = {}
    total = 0
    for name, axis in axes.items():
        offsets[name] = (total, total + len(axis))
        total += len(axis)
    values = {name: np.full(total, value) for name, value in center.items()}
    for name, axis in axes.items():
        start, end = offsets[name]
        values[name][start:end] = axis
    values.update({name: np.full(total, value) for name, value in config["external_context"].items()})
    evaluated = _run_connected_window(predictors, config, connections, values)
    controls = {}
    for name, axis in axes.items():
        start, end = offsets[name]
        low, high = _contiguous_true_interval(axis, evaluated["trusted_feasible"][start:end], center[name])
        controls[name] = {
            "lower_headroom": None if low is None else center[name] - low,
            "upper_headroom": None if high is None else high - center[name],
        }
    first_name = next(iter(axes))
    start, end = offsets[first_name]
    center_index = start + int(np.argmin(np.abs(axes[first_name] - center[first_name])))
    return {
        "controls": controls,
        "center_trusted_margin": float(evaluated["trusted_margin"][center_index]),
        "center_minimum_support": float(evaluated["minimum_support"][center_index]),
        "center_bottleneck": str(evaluated["bottleneck"][center_index]),
    }


def _simultaneous_window_evaluation(
    predictors, config: dict[str, object], connections: dict[str, str], center: dict[str, float],
    rho: float, normalized_headrooms: dict[str, float],
) -> dict[str, object]:
    """Sample a joint box separately from one-axis window claims."""
    setting = config["simultaneous_window"]
    required = {name: float(value) for name, value in config["window_center_selection"]["required_variations"].items()}
    bounds = {}
    for _, manifest, predictor in predictors:
        stage_bounds = _input_bounds(predictor)
        bounds.update({name: stage_bounds[name] for name in manifest["controls"]})
    rng = np.random.default_rng(int(setting["seed"]))

    def evaluate_controls(control_values: dict[str, np.ndarray]) -> dict[str, object]:
        size = len(next(iter(control_values.values())))
        values = dict(control_values)
        values.update({name: np.full(size, value) for name, value in config["external_context"].items()})
        result = _run_connected_window(predictors, config, connections, values)
        good = np.asarray(result["trusted_feasible"], dtype=bool)
        return {"count": size, "predicted_pass_count": int(good.sum()), "predicted_pass_fraction": float(good.mean())}

    random_rows = []
    samples = int(setting["samples"])
    for scale in map(float, setting["radius_scales"]):
        controls = {}
        clipped = np.zeros(samples, dtype=bool)
        for name in center:
            raw = center[name] + rng.uniform(-1, 1, samples) * rho * scale * required[name]
            controls[name] = np.clip(raw, *bounds[name])
            clipped |= controls[name] != raw
        row = {"radius_scale": scale, "clipped_fraction": float(clipped.mean()), **evaluate_controls(controls)}
        random_rows.append(row)

    names = list(center)
    signs = np.asarray(list(itertools.product((-1.0, 1.0), repeat=len(names))))
    vertex_controls = {
        name: np.clip(center[name] + signs[:, column] * rho * required[name], *bounds[name])
        for column, name in enumerate(names)
    }
    vertices = evaluate_controls(vertex_controls)

    limiting = sorted(names, key=lambda name: normalized_headrooms[name])[:max(2, int(setting["pair_count"]) + 1)]
    pairs = list(itertools.combinations(limiting, 2))[:int(setting["pair_count"])]
    pair_rows = []
    grid_points = int(setting["pair_grid_points"])
    unit = np.linspace(-1, 1, grid_points)
    mesh_a, mesh_b = np.meshgrid(unit, unit, indexing="ij")
    for first, second in pairs:
        size = mesh_a.size
        controls = {name: np.full(size, value) for name, value in center.items()}
        controls[first] = np.clip(center[first] + mesh_a.reshape(-1) * rho * required[first], *bounds[first])
        controls[second] = np.clip(center[second] + mesh_b.reshape(-1) * rho * required[second], *bounds[second])
        pair_rows.append({"controls": [first, second], **evaluate_controls(controls)})
    return {
        "definition": "Finite samples inside the joint +/-rho*required_variation box; not exact 9-D volume.",
        "rho": rho, "required_variations": required, "random_seed": int(setting["seed"]),
        "random_uniform_boxes": random_rows, "rho_box_vertices": vertices,
        "pairwise_sections": pair_rows,
    }


def _wilson_interval(successes: np.ndarray, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """二項比率の95% Wilson区間（モデル誤差は含まずMC誤差だけ）。"""
    z = 1.959963984540054
    probability = successes / samples
    denominator = 1.0 + z * z / samples
    center = (probability + z * z / (2 * samples)) / denominator
    radius = z * np.sqrt(probability * (1 - probability) / samples + z * z / (4 * samples * samples)) / denominator
    low = np.where(successes == 0, 0.0, center - radius)
    high = np.where(successes == samples, 1.0, center + radius)
    return np.clip(low, 0.0, 1.0), np.clip(high, 0.0, 1.0)


def _process_variation_rows(
    predictors, config: dict[str, object], connections: dict[str, str], axis_names: list[str],
    values: dict[str, np.ndarray], order: np.ndarray, deterministic_margin: np.ndarray,
) -> list[dict[str, object]]:
    """上位中心条件を工程ばらつきで一括評価し、予測良品率の下限で再順位付けします。"""
    variation = config["process_variation"]
    samples = int(variation["samples"])
    candidate_indices = np.asarray(order[:min(int(variation["candidate_limit"]), len(order))], dtype=int)
    candidate_count = len(candidate_indices)
    rng = np.random.default_rng(int(variation["seed"]))
    deviations = {name: float(value) for name, value in variation["standard_deviations"].items()}

    bounds_by_name: dict[str, tuple[float, float]] = {}
    for item, manifest, predictor in predictors:
        bounds = _input_bounds(predictor)
        bounds_by_name.update({name: bounds[name] for name in manifest["controls"]})
        stage_id = str(item["id"])
        bounds_by_name.update({f"{stage_id}.{name}": bounds[name] for name in manifest["incoming_context"]})

    scenario_values: dict[str, np.ndarray] = {}
    for name in axis_names:
        centers = values[name][candidate_indices, None]
        draws = centers + deviations[name] * rng.standard_normal(samples)[None, :]
        scenario_values[name] = np.clip(draws, *bounds_by_name[name]).reshape(-1)
    for name, nominal in config["external_context"].items():
        draws = float(nominal) + deviations[name] * rng.standard_normal(samples)
        scenario_values[name] = np.tile(np.clip(draws, *bounds_by_name[name]), candidate_count)

    stage_outputs, _, valid = _run_stage_chain(predictors, config, connections, scenario_values)
    final = stage_outputs[str(predictors[-1][0]["id"])]
    _, feasible = _margin_and_feasible(final, config.get("final_specifications", []))
    good = (feasible & valid).reshape(candidate_count, samples)
    valid_fraction = valid.reshape(candidate_count, samples).mean(axis=1)
    successes = good.sum(axis=1)
    probability = successes / samples
    low, high = _wilson_interval(successes, samples)
    robust_order = np.lexsort((-deterministic_margin[candidate_indices], -probability, -low))
    rows: list[dict[str, object]] = []
    for rank, local_index in enumerate(robust_order, start=1):
        point_index = candidate_indices[local_index]
        row: dict[str, object] = {
            "robust_rank": rank,
            "deterministic_candidate_index": int(point_index),
            "predicted_good_probability": float(probability[local_index]),
            "mc95_low": float(low[local_index]),
            "mc95_high": float(high[local_index]),
            "valid_chain_fraction": float(valid_fraction[local_index]),
            "deterministic_quality_margin": float(deterministic_margin[point_index]),
        }
        row.update({name: float(values[name][point_index]) for name in axis_names})
        rows.append(row)
    return rows


def _individual_best_chain(predictors, config: dict[str, object], connections: dict[str, str]) -> dict[str, object]:
    """各工程の局所目的だけを順に最適化する比較基準を同じbundleで計算します。"""
    selected_controls: dict[str, float] = {}
    stage_outputs: dict[str, dict[str, float]] = {}
    local_objectives: dict[str, dict[str, object]] = {}
    for item, manifest, predictor in predictors:
        stage_id = str(item["id"])
        control_names = list(manifest["controls"])
        points = np.asarray(list(itertools.product(*(config["candidate_axes"][name] for name in control_names))), dtype=float)
        inputs: dict[str, object] = {name: points[:, index] for index, name in enumerate(control_names)}
        for incoming in manifest["incoming_context"]:
            target = f"{stage_id}.{incoming}"
            if target in connections:
                source_stage, source_name = connections[target].split(".", 1)
                inputs[incoming] = stage_outputs[source_stage][source_name]
            else:
                inputs[incoming] = float(config["external_context"][target])
        predicted = _predict_allowing_invalid(predictor, manifest, inputs)
        outputs = {name: np.asarray(value["mean"]) for name, value in predicted["outputs"].items()}
        feasible = np.ones(len(points), dtype=bool)
        for constraint in manifest.get("local_constraints", []):
            values = outputs[constraint["name"]]
            if constraint["direction"] == "greater_equal":
                feasible &= values >= float(constraint["target"])
            else:
                feasible &= values <= float(constraint["target"])
        if not np.any(feasible):
            return {
                "status": "NOT_AVAILABLE",
                "reason": f"{stage_id}の局所制約を満たす予測候補がないため、個別Best連結を構成できません。",
                "failed_stage": stage_id,
                "controls": selected_controls,
                "local_objectives": local_objectives,
            }
        objective = manifest["local_objective"]
        score = outputs[objective["name"]]
        masked = np.where(feasible, score, -np.inf if objective["direction"] == "maximize" else np.inf)
        index = int(np.argmax(masked) if objective["direction"] == "maximize" else np.argmin(masked))
        selected_controls.update({name: float(points[index, column]) for column, name in enumerate(control_names)})
        stage_outputs[stage_id] = {name: float(value[index]) for name, value in outputs.items()}
        local_objectives[stage_id] = {"name": objective["name"], "direction": objective["direction"], "value": float(score[index])}
    final = {name: np.asarray([value]) for name, value in stage_outputs[str(predictors[-1][0]["id"])].items()}
    margin, feasible = _margin_and_feasible(final, config.get("final_specifications", []))
    return {"status": "AVAILABLE", "controls": selected_controls, "local_objectives": local_objectives, "predicted_feasible": bool(feasible[0]), "predicted_quality_margin": float(margin[0])}


def run_trial(name: str, recommendation_count: int) -> None:
    trial = _trial(name)
    config = _read_config(trial)
    stages = _validate(trial, config)
    axis_names = list(config["candidate_axes"])
    points = np.asarray(list(itertools.product(*(config["candidate_axes"][key] for key in axis_names))), dtype=float)
    if len(points) > 200_000:
        raise UserInputError("全体候補が200,000件を超えます。")
    values = {name: points[:, index] for index, name in enumerate(axis_names)}
    load_bundle = _load_bundle_runtime()
    predictors = [(item, manifest, load_bundle(path)[1]) for item, path, manifest in stages]
    connections = {str(row["to"]): str(row["from"]) for row in config.get("connections", [])}
    individual_chain = _individual_best_chain(predictors, config, connections)
    window_result = _run_connected_window(predictors, config, connections, values)
    stage_outputs = window_result["stage_outputs"]
    support = window_result["minimum_support"]
    final_outputs = stage_outputs[str(stages[-1][0]["id"])]
    final_margin, final_feasible = _margin_and_feasible(final_outputs, config.get("final_specifications", []))
    use_window = bool((config.get("connected_window") or {}).get("use_for_selection", False))
    ranking_margin = window_result["trusted_margin"] if use_window else final_margin
    feasible = window_result["trusted_feasible"] if use_window else final_feasible
    order = np.lexsort((-support, -ranking_margin))
    baseline_selected_index = int(order[0])
    center_comparisons = None
    center_setting = config.get("window_center_selection") or {}
    if center_setting.get("enabled"):
        center_comparisons = _window_center_comparisons(
            predictors, config, connections, points, axis_names,
            np.asarray(window_result["trusted_feasible"], dtype=bool), order,
        )
        if center_comparisons:
            selected_index = int(center_comparisons[0]["candidate_index"])
            order = np.concatenate(([selected_index], order[order != selected_index]))
    rows = []
    for rank, index in enumerate(order[:min(recommendation_count, len(order))], start=1):
        row: dict[str, object] = {
            "rank": rank, "predicted_feasible": bool(feasible[index]),
            "predicted_quality_margin": float(final_margin[index]),
            "connected_window_margin": float(window_result["trusted_margin"][index]),
            "connected_process_margin": float(window_result["process_margin"][index]),
            "connected_window_bottleneck": str(window_result["bottleneck"][index]),
            "minimum_stage_support": float(support[index]),
            "good_product_probability": "NOT_EVALUATED",
        }
        row.update({key: float(values[key][index]) for key in axis_names})
        row.update({f"final_{key}_mean": float(value[index]) for key, value in final_outputs.items()})
        rows.append(row)
    output = trial / "output"
    _write_csv(output / "system_recommendations.csv", rows)
    if center_comparisons is not None:
        flattened = []
        for rank, item in enumerate(center_comparisons, start=1):
            row = {key: value for key, value in item.items() if key != "normalized_headrooms"}
            row["window_rank"] = rank
            row.update({f"headroom_ratio_{name}": value for name, value in item["normalized_headrooms"].items()})
            flattened.append(row)
        _write_csv(output / "window_center_candidates.csv", flattened)
    robust_rows = None
    if config.get("process_variation") is not None:
        robust_rows = _process_variation_rows(predictors, config, connections, axis_names, values, order, final_margin)
        _write_csv(output / "robust_recommendations.csv", robust_rows)
    margin_improvement = (
        float(final_margin[order[0]] - individual_chain["predicted_quality_margin"])
        if individual_chain["status"] == "AVAILABLE" else None
    )
    summary = {
        "schema_version": "1.0", "candidate_count": len(points),
        "predicted_feasible_count": int(np.sum(feasible)),
        "best_predicted_quality_margin": float(final_margin[order[0]]),
        "probability_status": "UNCALIBRATED_PROCESS_VARIATION_ESTIMATE" if robust_rows is not None else "NOT_EVALUATED",
        "probability_reason": (
            "設定した独立正規工程ばらつきを平均予測モデルへ伝播。Wilson区間はMonte Carlo誤差だけで、モデル誤差は含まない。"
            if robust_rows is not None else "stage bundleが同時サンプリングと校正済み工程確率に未対応"
        ),
        "individual_best_chain": individual_chain,
        "quality_margin_improvement_over_individual_chain": margin_improvement,
        "connected_window": {
            "enabled_for_selection": use_window,
            "interval_method": str((config.get("connected_window") or {}).get("interval_method", "std_multiplier")),
            "support_threshold": float((config.get("connected_window") or {}).get("support_threshold", 0.0)),
            "confidence_z": float((config.get("connected_window") or {}).get("confidence_z", 0.0)),
            "enforce_local_constraints": bool((config.get("connected_window") or {}).get("enforce_local_constraints", False)),
            "feasible_count": int(np.sum(window_result["trusted_feasible"])),
            "selected_margin": float(window_result["trusted_margin"][order[0]]),
            "selected_process_margin": float(window_result["process_margin"][order[0]]),
            "selected_bottleneck": str(window_result["bottleneck"][order[0]]),
        },
        "selected": rows[0],
    }
    if robust_rows is not None:
        summary["process_variation"] = {
            "distribution": "independent_clipped_normal",
            "samples": int(config["process_variation"]["samples"]),
            "seed": int(config["process_variation"]["seed"]),
            "candidate_limit": int(config["process_variation"]["candidate_limit"]),
            "selection_metric": "mc95_low_then_probability_then_deterministic_margin",
            "selected": robust_rows[0],
        }
    if center_comparisons is not None:
        baseline_comparison = next(
            (item for item in center_comparisons if item["candidate_index"] == baseline_selected_index), None
        )
        summary["window_center_selection"] = {
            "status": "EVALUATED" if center_comparisons else "NO_TRUSTED_CANDIDATES",
            "selection_metric": "symmetric_headroom_rho_then_connected_margin_then_support",
            "evaluated_candidates": len(center_comparisons),
            "eligible_candidates": int(np.sum(window_result["trusted_feasible"])),
            "required_variations": center_setting["required_variations"],
            "selected": center_comparisons[0] if center_comparisons else None,
            "baseline_connected_margin_selection": baseline_comparison,
            "rho_improvement_over_baseline": (
                center_comparisons[0]["symmetric_headroom_rho"] - baseline_comparison["symmetric_headroom_rho"]
                if center_comparisons and baseline_comparison is not None else None
            ),
            "claim": "budgeted_diverse_candidate_comparison_not_global_optimum",
        }
    profiles = None
    if config.get("connected_window") is not None:
        center = {name: float(values[name][order[0]]) for name in axis_names}
        profiles = _connected_window_profiles(predictors, config, connections, center)
        _write_json(output / "connected_window.json", profiles)
    simultaneous_setting = config.get("simultaneous_window") or {}
    if simultaneous_setting.get("enabled"):
        if not center_comparisons:
            raise UserInputError("simultaneous_windowには有効なwindow_center_selection結果が必要です。")
        selected_window = center_comparisons[0]
        simultaneous = _simultaneous_window_evaluation(
            predictors, config, connections, center,
            float(selected_window["symmetric_headroom_rho"]), selected_window["normalized_headrooms"],
        )
        _write_json(output / "simultaneous_window.json", simultaneous)
        summary["simultaneous_window"] = {
            "status": "EVALUATED", "rho": simultaneous["rho"],
            "random_uniform_boxes": simultaneous["random_uniform_boxes"],
            "rho_box_vertices": simultaneous["rho_box_vertices"],
            "pairwise_sections": simultaneous["pairwise_sections"],
            "claim": "finite_sample_joint_evaluation_not_exact_volume",
        }
    _write_json(output / "run_summary.json", summary)
    print(f"{name}の全体条件を{len(points):,}候補から評価しました。")
    print(f"予測適合候補: {summary['predicted_feasible_count']:,}件")
    if margin_improvement is None:
        print(f"個別Best連結: 未評価（{individual_chain['reason']}）")
    else:
        print(f"個別Best連結からの仕様余裕改善: {margin_improvement:.6g}")
    print(f"出力: {output / 'system_recommendations.csv'}")
    if robust_rows is None:
        print("良品確率: 未評価（平均予測bundleのため）")
    else:
        print(f"工程ばらつき込み未校正予測良品率: {robust_rows[0]['predicted_good_probability']:.3%}")
        print(f"確率最適化出力: {output / 'robust_recommendations.csv'}")


def positive_integer(value: str) -> int:
    result = int(value)
    if result < 1:
        raise argparse.ArgumentTypeError("1以上の整数を指定してください")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="v004 stage bundleを接続して全体条件を調整します。")
    actions = parser.add_mutually_exclusive_group(required=True)
    actions.add_argument("--new", metavar="TRIAL")
    actions.add_argument("--prepare", metavar="TRIAL")
    actions.add_argument("--run", metavar="TRIAL")
    parser.add_argument("--n", type=positive_integer, default=3)
    args = parser.parse_args()
    try:
        if args.new:
            create_trial(args.new)
        elif args.prepare:
            prepare_trial(args.prepare)
        else:
            run_trial(args.run, args.n)
        return 0
    except (UserInputError, ValueError, OSError) as error:
        print(f"[入力エラー]\n{error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
