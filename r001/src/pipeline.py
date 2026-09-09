"""各工程の平均予測を接続し、最終仕様と個別Best連結の基準を評価します。

配列の同じ行は同じ仮想製品です。流入範囲外の行は評価不能として残し、
有効な行だけを予測器へ渡します。平均予測を良品確率とは扱いません。
"""

from __future__ import annotations

import itertools
import numpy as np


def margin_and_feasible(outputs: dict[str, np.ndarray], specs: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    """各仕様の余裕を尺度で正規化し、最小余裕と全仕様の合否を返します。"""

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


def predict_allowing_invalid(predictor, manifest: dict[str, object], inputs: dict[str, object]) -> dict[str, object]:
    """範囲内の行だけを予測し、評価不能行は無効値として元の行位置へ戻します。"""

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


def run_stage_chain(predictors, config: dict[str, object], connections: dict[str, str], values: dict[str, object]):
    """同じ行の上流予測平均を下流の流入状態へ渡し、工程別出力を返します。"""

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
        predicted = predict_allowing_invalid(predictor, manifest, inputs)
        stage_outputs[stage_id] = {key: np.asarray(value["mean"]) for key, value in predicted["outputs"].items()}
        supports.append(np.asarray(predicted["support"]))
        valid_connections.append(np.asarray(predicted["valid_connection"], dtype=bool))
    return stage_outputs, np.min(np.stack(supports), axis=0), np.all(np.stack(valid_connections), axis=0)


def input_bounds(predictor) -> dict[str, tuple[float, float]]:
    """学習済み予測器またはoracle予測器から、宣言された入力範囲を取得します。"""

    if hasattr(predictor, "problem"):
        return {item.column: (float(item.lower), float(item.upper)) for item in predictor.problem.parameters}
    return {name: tuple(map(float, bounds)) for name, bounds in predictor.bounds.items()}


def individual_best_chain(predictors, config: dict[str, object], connections: dict[str, str]) -> dict[str, object]:
    """各工程の局所目的を順に最適化した条件を接続し、全体調整の比較基準を作ります。"""

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
        predicted = predict_allowing_invalid(predictor, manifest, inputs)
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
    margin, feasible = margin_and_feasible(final, config.get("final_specifications", []))
    return {"status": "AVAILABLE", "controls": selected_controls, "local_objectives": local_objectives, "predicted_feasible": bool(feasible[0]), "predicted_quality_margin": float(margin[0])}

