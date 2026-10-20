"""上流変更の追従と、在庫実験→結果追加→再学習を検証します。"""

from __future__ import annotations

import copy
from pathlib import Path

import numpy as np

from .interconnect_data import available_inventory, generate_workpieces, local_margin, physical_contract, train_bundles
from .interconnect_metrics import compact_search, oracle_predictors, score_controls


def recovery_checks(config: dict, predictors: dict, controls: dict, budget: int) -> dict:
    """製作済みワークの操作を固定し、観測状態を使って残り工程を再探索します。"""
    from r002.src.api import evaluate_acceptance, run_search
    from r002.src.pipeline import evaluate_pipeline
    result = evaluate_pipeline(config, oracle_predictors(), controls)
    _, manifests, _ = physical_contract()
    upstream = [stage["id"] for stage in config["stages"] if stage["id"] != "welding"]
    fixed = {name: value for name, value in controls.items() if name.split(".")[0] in upstream}
    observed = {f"{stage}.{name}": float(values[0]) for stage in upstream
                for name, values in result["stage_outputs"][stage].items() if np.isfinite(values[0])}
    complete_finite = all(np.all(np.isfinite(value)) for stage in upstream for value in result["stage_outputs"][stage].values())
    upstream_passed = all(np.all(local_margin(result["stage_outputs"][stage], manifests[stage]) >= 0) for stage in upstream)
    if not complete_finite or not upstream_passed:
        return {"status": "NO_VALID_UPSTREAM_WORKPIECE"}
    local = copy.deepcopy(config)
    local.update(fixed_controls=fixed, observed_outputs=observed)
    local["search"] = {**config["search"], "max_evaluations": budget}
    complete = run_search(local, predictors, recommendation_count=1)
    if complete["best"]:
        truth_config = copy.deepcopy(config)
        truth_config.update(fixed_controls=fixed, observed_outputs=observed)
        complete.update(score_controls(truth_config, complete["best"]["controls"]))
    # 現場で不可視な値をoracleから黙って与えない。部分観測はモデル補完を使うが、
    # 実測による品質証明としては扱わないことを結果に残します。
    visible = {f"{stage}.{name}" for stage in upstream
               for name in manifests[stage]["observation_policy"]["nondestructive"]}
    partial = {name: value for name, value in observed.items() if name in visible}
    unknown = sorted(set(observed) - set(partial))
    local["observed_outputs"] = partial
    partial_result = run_search(local, predictors, recommendation_count=1)
    partial_result["field_quality_status"] = "UNKNOWN_WITH_UNOBSERVED_STATE" if unknown else "OBSERVED"
    partial_result["unobserved_outputs"] = unknown
    partial_result["interpretation"] = "model_completion_is_not_validation_of_hidden_physical_state"
    # ロット物性の小変更も、接続前の流入条件を変えて再探索できます。
    drift = copy.deepcopy(config)
    drift["search"] = {**config["search"], "max_evaluations": budget}
    changed = {}
    for name, value in list(drift["external_context"].items()):
        if name.endswith("stock_yield_mpa"):
            changed[name] = float(value * 1.1)
            drift["external_context"][name] = changed[name]
    drift_result = run_search(drift, predictors, recommendation_count=1)
    hypothetical = copy.deepcopy(config)
    hypothetical["reference_controls"] = fixed
    acceptance = evaluate_acceptance(hypothetical, {"outputs": observed, "reachable": True}, predictors, budget=budget)
    payload = {"complete_observation_recovery": complete, "partial_observation_recovery": partial_result,
               "local_acceptance_query": acceptance,
               "material_drift": {"changed_context": changed, "status": drift_result["status"],
                                  "evaluation_count": drift_result["evaluation_count"], "best": drift_result["best"]}}
    if drift_result["best"]:
        payload["material_drift"].update(score_controls(drift, drift_result["best"]["controls"]))
    return payload


def experiment_loop(config: dict, predictors: dict, dataset: dict, destination: Path,
                    sample_cap: int, seed: int) -> dict:
    """未溶接在庫で測定し、消費IDを記録して溶接モデルだけ更新します。"""
    from r002.src.api import propose_additional_experiments, run_search
    from v005.src.stage_bundle import load_stage_bundle, train_stage_bundle
    from ...electrical_interconnect.stages import evaluate_stage
    from ...electrical_interconnect.inspection import inspect_final
    inventory_data = generate_workpieces(max(2048, dataset["requested_count"]), seed + 10000,
                                         stop_after="assembly", external_context=config["external_context"])
    inventory = available_inventory(inventory_data, maximum=12)
    local = copy.deepcopy(config)
    local["inventory"] = inventory
    local["additional_experiments"] = {"max_evaluations": 96, "conditions_per_workpiece": 8,
                                       "recommendation_count": 3, "seed": seed,
                                       "max_experiment_cost": 3.0,
                                       "costs": {"welding": {"cost": 1.0, "seconds": 60.0, "destructive": True}}}
    proposals = propose_additional_experiments(local, predictors)
    consumed, records = set(), []
    block = copy.deepcopy(dataset["blocks"]["welding"])
    for proposal in proposals:
        ids = proposal["workpiece_ids"]
        if consumed.intersection(ids):
            raise AssertionError("同じワークを2回溶接する実験が提案されました。")
        inputs = {**proposal["controls"], **proposal["incoming_context"]}
        measurement = evaluate_stage("welding", inputs)
        consumed.update(ids)
        for name in block["inputs"]:
            block["inputs"][name] = np.append(block["inputs"][name], inputs[name])
        for name in block["outputs"]:
            block["outputs"][name] = np.append(block["outputs"][name], measurement[name])
        records.append({**proposal, "status": "EXECUTED_SYNTHETIC_MEASUREMENT",
                        "inspection": inspect_final(measurement, destructive=True, noise=False),
                        "synthetic_training_labels": {name: float(value) for name, value in measurement.items()},
                        "label_regime": "PRIVILEGED_SYNTHETIC_LABELS_NOT_ALL_PHYSICALLY_MEASURABLE",
                        "destructive_inspection": "sacrificial_coupon_not_reusable_inventory"})
    for item in local["inventory"]:
        if item["workpiece_id"] in consumed:
            item.update(consumed=True, available=False)
    if not records:
        return {"status": "NO_AVAILABLE_FEASIBLE_WORKPIECES", "inventory_count": len(inventory), "executed": []}
    manifest = copy.deepcopy(predictors["welding"][0])
    manifest["provenance"] = {"source": "synthetic_partial_experiment_update", "consumed_ids": sorted(consumed)}
    updated = train_stage_bundle(destination / "welding_updated", manifest, block["inputs"], block["outputs"],
                                 seed=seed, max_training_samples=sample_cap)
    new_manifest, new_predictor = load_stage_bundle(updated)
    order = new_manifest["controls"] + new_manifest["incoming_context"]
    lower = np.array([new_manifest["input_bounds"][name][0] for name in order])
    upper = np.array([new_manifest["input_bounds"][name][1] for name in order])
    retained = new_predictor.model.train_x
    incorporated = True
    for record in records:
        inputs = {**record["controls"], **record["incoming_context"]}
        normalized = (np.array([inputs[name] for name in order]) - lower) / (upper - lower)
        incorporated &= bool(np.any(np.all(np.isclose(retained, normalized), axis=1)))
    # 消費済みワークが、次の提案に紛れ込まないことも確認します。
    remaining = propose_additional_experiments(local, {**predictors, "welding": load_stage_bundle(updated)})
    no_reuse = all(not consumed.intersection(item["workpiece_ids"]) for item in remaining)
    feedback_config = copy.deepcopy(config)
    feedback_config["search"] = {**config["search"], "max_evaluations": min(256, config["search"]["max_evaluations"])}
    feedback = run_search(feedback_config, {**predictors, "welding": (new_manifest, new_predictor)}, recommendation_count=1)
    return {"status": "MEASURED_APPENDED_RETRAINED", "inventory_count": len(inventory),
            "executed": records, "consumed_workpiece_ids": sorted(consumed),
            "consumed_workpieces_excluded": no_reuse, "updated_bundle": str(updated.resolve()),
            "new_measurements_retained_in_model": incorporated,
            "original_data_sha256": predictors["welding"][0]["training"]["data_sha256"],
            "updated_data_sha256": new_manifest["training"]["data_sha256"],
            "retained_training_samples": new_manifest["training"]["retained_samples"],
            "updated_model_search": compact_search(feedback, feedback_config),
            "claim": "successful_update_does_not_imply_prediction_error_improved"}


def candidate_confirmation_loop(config: dict, predictors: dict, dataset: dict, initial_search: dict,
                                destination: Path, sample_cap: int, *, rounds: int = 4,
                                candidates_per_round: int = 4) -> dict:
    """推薦条件を新しいワークで確認し、上流NGで停止して観測済み工程だけ更新します。

    正解値を直接探索器に渡すことはありません。確認実験として得た入力/出力だけを
    学習データへ追記します。各条件は別の製作依頼であり、同一部品の使い回しではありません。
    """
    from r002.src.api import run_search
    if any(config.get(name) for name in ("observed_outputs", "hypothetical_outputs", "completed_stages")):
        raise ValueError("全工程の製作確認には観測済み工程を含めません。既存ワークの確認は在庫実験を使ってください。")
    working_data = copy.deepcopy(dataset)
    search = initial_search
    records, history = [], []
    confirmed_best = None
    evaluations = 0
    for round_id in range(rounds):
        completed_ids = set()
        for candidate_id, candidate in enumerate(search["candidates"][:candidates_per_round]):
            request_id = f"confirmation-{dataset['seed']}-{round_id}-{candidate_id}"
            work = generate_workpieces(1, dataset["seed"] + 20000 + len(records),
                                       selected_controls=candidate["controls"], external_context=config["external_context"])
            truth = score_controls(config, candidate["controls"])
            record = {"request_id": request_id, "controls": candidate["controls"],
                      "predicted_feasible": candidate["predicted_feasible"], **truth,
                      "measured_stages": [stage for stage, block in work["blocks"].items() if block["admitted_count"]],
                      "observation_regime": "PRIVILEGED_SYNTHETIC_CONFIRMATION"}
            records.append(record)
            if truth["oracle_feasible"] and (confirmed_best is None or truth["oracle_margin"] > confirmed_best["oracle_margin"]):
                confirmed_best = record
            for stage, measured in work["blocks"].items():
                if not measured["admitted_count"]:
                    continue
                block = working_data["blocks"][stage]
                for name in block["inputs"]:
                    block["inputs"][name] = np.concatenate((block["inputs"][name], measured["inputs"][name]))
                for name in block["outputs"]:
                    block["outputs"][name] = np.concatenate((block["outputs"][name], measured["outputs"][name]))
                block["admitted_count"] += measured["admitted_count"]
                block["passed_count"] += measured["passed_count"]
                completed_ids.add(stage)
        history.append({"round": round_id + 1, "search_evaluations": search["evaluation_count"],
                        "confirmed_feasible_found": confirmed_best is not None,
                        "observed_stages": sorted(completed_ids),
                        "retrain_before_next_round": confirmed_best is None and round_id + 1 < rounds})
        evaluations += search["evaluation_count"]
        if confirmed_best is not None or round_id + 1 == rounds:
            break
        # 最近の測定はv005の保持枠へ入ります。生成元の旧CSVや元bundleは変えません。
        _, predictors = train_bundles(working_data, destination / f"round_{round_id + 1}", sample_cap)
        local = copy.deepcopy(config)
        local["search"] = {**config["search"], "seed": dataset["seed"] + round_id + 1,
                           "max_evaluations": min(1024, config["search"]["max_evaluations"])}
        search = run_search(local, predictors, recommendation_count=candidates_per_round)
    return {"status": "CONFIRMED_SYNTHETIC_FEASIBLE" if confirmed_best else "NO_CONFIRMED_FEASIBLE_WITHIN_BUDGET",
            "confirmed_best": confirmed_best, "confirmation_count": len(records),
            "total_search_evaluations_including_initial": evaluations,
            "history": history, "measurements": records,
            "claim": "measurement_feedback_is_not_field_calibration_or_global_optimum"}
