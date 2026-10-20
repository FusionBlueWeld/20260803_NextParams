"""手元にあるワークから実行可能な部分実験だけを提案します。"""

from __future__ import annotations

import math
import time

import numpy as np

from .config import qualified, validate_config
from .pipeline import specification_margin
from .search import latin_hypercube


def _available_inputs(stage: str, config: dict, plan: dict) -> list[tuple[dict, list[dict]]]:
    """合流は明示されたassembly_pair_idで合わせ、在庫の自由な直積を作りません。"""
    inventory = [item for item in config.get("inventory", []) if item.get("available", True) and not item.get("consumed", False)]
    manifest_incoming = plan["manifests"][stage].get("incoming_context", [])
    sources = {qualified(plan["edges"][f"{stage}.{name}"])[0] for name in manifest_incoming if f"{stage}.{name}" in plan["edges"]}
    groups = []
    if sources:
        if len(sources) == 1:
            source = next(iter(sources))
            groups = [[item] for item in inventory if item.get("stage_id") == source]
        else:
            pairs = {}
            for item in inventory:
                pair = item.get("assembly_pair_id")
                if pair is not None and item.get("stage_id") in sources:
                    pairs.setdefault(str(pair), {}).setdefault(item["stage_id"], []).append(item)
            for items in pairs.values():
                if set(items) == sources and all(len(entries) == 1 for entries in items.values()):
                    groups.append([entries[0] for entries in items.values()])
    else:
        groups = [[item] for item in inventory if item.get("target_stage_id", item.get("stage_id")) == stage and "incoming_context" in item]
    result = []
    for group in groups:
        inputs = {}
        valid = all(item.get("workpiece_id") for item in group) and len({item.get("workpiece_id") for item in group}) == len(group)
        by_stage = {item.get("stage_id"): item for item in group}
        for name in manifest_incoming:
            target = f"{stage}.{name}"
            if target in plan["edges"]:
                source, variable = qualified(plan["edges"][target])
                value = by_stage[source].get("outputs", {}).get(variable)
            else:
                value = group[0].get("incoming_context", {}).get(name, config.get("external_context", {}).get(target))
            if value is None or not math.isfinite(float(value)):
                valid = False
                break
            inputs[name] = float(value)
        if valid:
            result.append((inputs, group))
    return result


def propose_experiments(config: dict, predictors: dict) -> list[dict]:
    """支持不足・境界・出力stdの代理指標を費用で割り、追加測定候補を返します。"""
    plan = validate_config(config, predictors)
    plan["manifests"] = {stage: item[0] for stage, item in predictors.items()}
    setting = config.get("additional_experiments", {})
    total = int(setting.get("max_evaluations", 256))
    per_work = int(setting.get("conditions_per_workpiece", 8))
    limit = int(setting.get("recommendation_count", 8))
    seconds = float(setting.get("max_seconds", 30.0))
    if total < 1 or per_work < 1 or limit < 1 or not math.isfinite(seconds) or seconds <= 0:
        raise ValueError("追加実験の予算は正にしてください。")
    rng = np.random.default_rng(int(setting.get("seed", plan["search"]["seed"])))
    started, evaluated = time.monotonic(), 0
    candidates = []
    for stage in plan["order"]:
        if stage in plan["completed_stages"]:
            continue
        manifest, predictor = predictors[stage]
        names = manifest["controls"]
        cost_setting = setting.get("costs", {}).get(stage, {})
        if cost_setting.get("allowed", True) is False:
            continue
        cost = float(cost_setting.get("cost", 1.0))
        duration = float(cost_setting.get("seconds", 1.0))
        if not math.isfinite(cost) or cost <= 0 or not math.isfinite(duration) or duration <= 0:
            raise ValueError("実験費用・所要時間は正の有限値です。")
        for incoming, workpieces in _available_inputs(stage, config, plan):
            if evaluated >= total or time.monotonic() - started >= seconds:
                break
            if any(not manifest["input_bounds"][name][0] <= value <= manifest["input_bounds"][name][1] for name, value in incoming.items()):
                continue
            count = min(per_work, total - evaluated, int(plan["search"]["batch_size"]))
            sample = latin_hypercube(count, len(names), rng)
            inputs = {name: np.full(count, value) for name, value in incoming.items()}
            for index, name in enumerate(names):
                axis = np.asarray(config["candidate_axes"][f"{stage}.{name}"], dtype=float)
                if f"{stage}.{name}" in config.get("fixed_controls", {}):
                    inputs[name] = np.full(count, float(config["fixed_controls"][f"{stage}.{name}"]))
                else:
                    inputs[name] = axis[np.minimum((sample[:, index] * len(axis)).astype(int), len(axis) - 1)]
            prediction = predictor.predict_arrays(inputs)
            evaluated += count
            support = np.asarray(prediction["support"], dtype=float)
            uncertainty = np.zeros(count)
            for name in manifest["predicted_outputs"]:
                mean = np.asarray(prediction["outputs"][name]["mean"], dtype=float)
                std = np.asarray(prediction["outputs"][name]["std"], dtype=float)
                uncertainty += np.clip(std / np.maximum(np.abs(mean), 1.0), 0.0, 1.0)
            margins = [specification_margin(np.asarray(prediction["outputs"][spec["name"]]["mean"]), spec) for spec in manifest.get("local_constraints", [])]
            proximity = 1 / (1 + np.min(np.abs(np.stack(margins)), axis=0)) if margins else np.zeros(count)
            # 下流へ影響する工程を優先する代理重み。厳密な情報価値とは表示しません。
            outgoing = sum(qualified(source)[0] == stage for source in plan["edges"].values())
            acquisition = ((1 - support) + uncertainty + 0.25 * proximity) * (1 + 0.1 * outgoing) / (cost * (1 + math.log1p(duration)))
            for row in range(count):
                if not math.isfinite(float(acquisition[row])):
                    continue
                candidates.append({"stage_id": stage, "workpiece_ids": [item["workpiece_id"] for item in workpieces],
                                   "assembly_pair_id": workpieces[0].get("assembly_pair_id"),
                                   "lot_ids": [item.get("lot_id") for item in workpieces],
                                   "controls": {name: float(inputs[name][row]) for name in names},
                                   "incoming_context": incoming, "score": float(acquisition[row]),
                                   "predicted_support": float(support[row]), "cost": cost, "seconds": duration,
                                   "destructive": bool(cost_setting.get("destructive", True) or any(item.get("destructive", False) for item in workpieces)),
                                   "consumes_workpiece": bool(cost_setting.get("consumes_workpiece", True)),
                                   "reason": "cost_weighted_uncertainty_boundary_proxy", "status": "PROPOSED_NOT_EXECUTED"})
    candidates.sort(key=lambda item: item["score"], reverse=True)
    chosen, consumed, seen_conditions = [], set(), set()
    budget = float(setting.get("max_experiment_cost", math.inf))
    time_budget = float(setting.get("max_experiment_seconds", math.inf))
    spent, experiment_seconds = 0.0, 0.0
    for item in candidates:
        key = (item["stage_id"], tuple(sorted(item["workpiece_ids"])), tuple(sorted(item["controls"].items())))
        if key in seen_conditions:
            continue
        seen_conditions.add(key)
        if consumed.intersection(item["workpiece_ids"]) or spent + item["cost"] > budget or experiment_seconds + item["seconds"] > time_budget:
            continue
        chosen.append(item)
        spent += item["cost"]
        experiment_seconds += item["seconds"]
        # 非破壊検査でも加工後のワーク状態は変わります。検査の破壊性とは別に
        # 払い出しIDを予約し、同じ加工前ワークへ異なる条件を割り当てません。
        if item["consumes_workpiece"] or item["destructive"]:
            consumed.update(item["workpiece_ids"])
        if len(chosen) >= limit:
            break
    return chosen
