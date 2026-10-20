"""同じ候補・同じ製品の状態を、分岐合流工程へ順に伝播します。"""

from __future__ import annotations

import numpy as np

from .config import qualified, validate_config


def specification_margin(value: np.ndarray, spec: dict) -> np.ndarray:
    """物理単位の目標から無次元の合格余裕を計算します。"""
    target, direction = spec["target"], spec["direction"]
    if direction == "between":
        lower, upper = map(float, target)
        scale = float(spec.get("margin_scale", (upper - lower) / 2))
        return np.minimum(value - lower, upper - value) / scale
    scale = float(spec.get("margin_scale", max(abs(float(target)), 1e-12)))
    return ((value - float(target)) if direction == "greater_equal" else (float(target) - value)) / scale


def conservative_specification_margin(value: np.ndarray, std: np.ndarray, spec: dict, multiplier: float) -> np.ndarray:
    """未校正stdによる任意の余裕減算。信頼区間や制約保証とは扱いません。"""
    mean_margin = specification_margin(value, spec)
    if spec["direction"] == "between":
        lower, upper = map(float, spec["target"])
        scale = float(spec.get("margin_scale", (upper - lower) / 2))
    else:
        scale = float(spec.get("margin_scale", max(abs(float(spec["target"])), 1e-12)))
    return mean_margin - multiplier * std / scale


def evaluate_pipeline(config: dict, predictors: dict, values: dict, *, validated: dict | None = None) -> dict:
    """小さい候補配列を評価し、工程制約・最終仕様・支持度をまとめて返します。"""
    plan = validated or validate_config(config, predictors)
    controls = plan["controls"]
    if set(values) != set(controls):
        raise ValueError("評価する操作辞書は全stage_id.controlと一致させてください。")
    arrays = np.broadcast_arrays(*[np.asarray(values[name], dtype=float) for name in controls])
    shape = arrays[0].shape
    flat_values = {name: array.reshape(-1) for name, array in zip(controls, arrays)}
    count = arrays[0].size
    outputs, stds, supports, validity, stage_inputs = {}, {}, {}, {}, {}
    observation_status, unobserved = {}, {}
    components, conservative_components = {}, {}
    multiplier = float(plan["search"].get("conservative_std_multiplier", 0.0))
    hypothetical = config.get("hypothetical_outputs", {})
    observed = {**config.get("observed_outputs", {}), **hypothetical}
    prediction_rows = 0
    for stage in plan["order"]:
        manifest, predictor = predictors[stage]
        stage_outputs = {name: np.full(count, np.nan) for name in manifest["predicted_outputs"]}
        stage_stds = {name: np.full(count, np.nan) for name in manifest["predicted_outputs"]}
        measured = {name: float(observed[f"{stage}.{name}"]) for name in stage_outputs if f"{stage}.{name}" in observed}
        unobserved[stage] = sorted(set(stage_outputs) - set(measured))
        observation_status[stage] = ("FULLY_MEASURED_MEANS" if len(measured) == len(stage_outputs) else
                                     "PARTIALLY_MEASURED_MODEL_FOR_UNOBSERVED" if measured else "MODEL_PREDICTED")
        if any(f"{stage}.{name}" in hypothetical for name in stage_outputs):
            observation_status[stage] = "HYPOTHETICAL_STATE_MODEL_FOR_UNSUPPLIED"
        support = np.zeros(count)
        valid = np.ones(count, dtype=bool)
        inputs = {name: flat_values[f"{stage}.{name}"] for name in manifest["controls"]}
        for name in manifest["incoming_context"]:
            target = f"{stage}.{name}"
            if target in plan["edges"]:
                upstream, variable = qualified(plan["edges"][target])
                inputs[name] = outputs[upstream][variable]
            else:
                inputs[name] = np.full(count, float(config["external_context"][target]))
        for name, value in inputs.items():
            lower, upper = manifest["input_bounds"][name]
            valid &= np.isfinite(value) & (value >= lower) & (value <= upper)
        stage_inputs[stage] = inputs
        if len(measured) == len(stage_outputs):
            # 完全実測された出力は予測で置き換えず、残り工程に同じ測定平均を渡します。
            for name, value in measured.items():
                stage_outputs[name][:] = value
                stage_stds[name][:] = 0.0
            support[:] = 1.0
            valid[:] = True
        elif np.any(valid):
            prediction_rows += int(np.sum(valid))
            predicted_rows = np.flatnonzero(valid)
            prediction = predictor.predict_arrays({name: value[valid] for name, value in inputs.items()})
            for name in stage_outputs:
                mean = np.asarray(prediction["outputs"][name]["mean"], dtype=float)
                std = np.asarray(prediction["outputs"][name]["std"], dtype=float)
                stage_outputs[name][valid] = np.broadcast_to(mean, (int(np.sum(valid)),))
                stage_stds[name][valid] = np.broadcast_to(std, (int(np.sum(valid)),))
            support[valid] = np.broadcast_to(np.asarray(prediction["support"], dtype=float), (int(np.sum(valid)),))
            if "valid" in prediction:
                declared = np.broadcast_to(np.asarray(prediction["valid"], dtype=bool), (len(predicted_rows),))
                valid[predicted_rows] &= declared
            valid &= np.all(np.stack([np.isfinite(value) for value in stage_outputs.values()]), axis=0)
            valid &= np.all(np.stack([np.isfinite(value) & (value >= 0) for value in stage_stds.values()]), axis=0)
            valid &= np.isfinite(support) & (support >= 0) & (support <= 1)
        for name, value in measured.items():
            stage_outputs[name][:] = value
            stage_stds[name][:] = 0.0
        outputs[stage], stds[stage], supports[stage], validity[stage] = stage_outputs, stage_stds, support, valid
        for spec in manifest.get("local_constraints", []):
            components[f"{stage}.{spec['name']}:local"] = specification_margin(stage_outputs[spec["name"]], spec)
            conservative_components[f"{stage}.{spec['name']}:local"] = conservative_specification_margin(stage_outputs[spec["name"]], stage_stds[spec["name"]], spec, multiplier)
    for spec in config["final_specifications"]:
        stage = spec.get("stage", plan["order"][-1])
        components[f"{stage}.{spec['name']}:final"] = specification_margin(outputs[stage][spec["name"]], spec)
        conservative_components[f"{stage}.{spec['name']}:final"] = conservative_specification_margin(outputs[stage][spec["name"]], stds[stage][spec["name"]], spec, multiplier)
    valid = np.all(np.stack(list(validity.values())), axis=0)
    process_margin = np.min(np.stack(list(components.values())), axis=0)
    conservative_margin = np.min(np.stack(list(conservative_components.values())), axis=0)
    minimum_support = np.min(np.stack(list(supports.values())), axis=0)
    threshold = float(plan["search"]["support_threshold"])
    support_margin = (minimum_support - threshold) / max(1 - threshold, 1e-12)
    trusted_margin = np.minimum(conservative_margin, support_margin)
    trusted_margin = np.where(valid & np.isfinite(trusted_margin), trusted_margin, -1e6)
    feasible = valid & (conservative_margin >= 0) & (minimum_support >= threshold)
    names = list(components) + ["support"]
    matrix = np.stack([*conservative_components.values(), support_margin])
    bottleneck = np.asarray(names, dtype=object)[np.argmin(np.nan_to_num(matrix, nan=-1e6), axis=0)]
    return {
        "stage_outputs": outputs, "stage_inputs": stage_inputs, "stage_stds": stds, "stage_support": supports,
        "stage_validity": validity, "valid_connection": valid,
        "process_margin": process_margin, "minimum_support": minimum_support,
        "conservative_process_margin": conservative_margin,
        "trusted_margin": trusted_margin, "trusted_feasible": feasible,
        "bottleneck": bottleneck, "constraint_margins": components,
        "conservative_constraint_margins": conservative_components,
        "uncertainty_policy": "uncalibrated_heuristic" if multiplier else "mean_prediction_only",
        "conservative_std_multiplier": multiplier,
        "probability_status": "NOT_EVALUATED",
        "observation_status": observation_status, "modelled_unobserved_outputs": unobserved,
        "model_prediction_rows": prediction_rows,
        "observation_noise_status": "NOT_PROPAGATED_AS_PHYSICAL_VARIATION",
    }
