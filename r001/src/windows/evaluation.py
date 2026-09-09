"""連結窓の判定核。工程制約・最終仕様・入力範囲・支持度の余裕を統合します。

校正された残差区間または指定の標準偏差係数を使い、最も厳しい制約を
ボトルネックとして返します。窓の走査や候補選択は別モジュールが担当します。
"""

from __future__ import annotations

import numpy as np
from ..window_calibration import interval_bounds
from ..validation import UserInputError
from ..pipeline import input_bounds, predict_allowing_invalid


def constraint_margin_components(
    outputs: dict[str, np.ndarray], specifications: list[dict], prefix: str,
    standard_deviations: dict[str, np.ndarray] | None = None, confidence_z: float = 0.0,
    residual_offsets: dict[str, dict[str, float]] | None = None,
    interval_method: str = "std_multiplier",
) -> dict[str, np.ndarray]:
    """指定された区間方式で制約ごとの余裕を計算し、最も厳しい条件を選べる形にします。"""

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


def evaluate_connected_window(
    predictors, config: dict[str, object], connections: dict[str, str], values: dict[str, object],
) -> dict[str, object]:
    """工程制約・接続範囲・最終仕様・支持度の共通部分を評価し、余裕と合否を返します。"""

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
        bounds = input_bounds(predictor)
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
        predicted = predict_allowing_invalid(predictor, manifest, inputs)
        stage_outputs[stage_id] = {
            name: np.asarray(value["mean"], dtype=float) for name, value in predicted["outputs"].items()
        }
        stage_stds[stage_id] = {
            name: np.asarray(value["std"], dtype=float) for name, value in predicted["outputs"].items()
        }
        stage_support[stage_id] = np.asarray(predicted["support"], dtype=float)
        valid_rows.append(np.asarray(predicted["valid_connection"], dtype=bool))
        if enforce_local:
            process_components.update(constraint_margin_components(
                stage_outputs[stage_id], manifest.get("local_constraints", []), f"local.{stage_id}",
                stage_stds[stage_id], confidence_z,
                calibration_offsets.get(stage_id),
                interval_method,
            ))
    final_stage = str(predictors[-1][0]["id"])
    process_components.update(constraint_margin_components(
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

