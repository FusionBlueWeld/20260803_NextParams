"""CLIと検証コードが共通して使うr002の公開API。"""

from __future__ import annotations

import copy
import math
import time

from .bundle_runtime import load_predictors
from .config import qualified, read_config, validate_config
from .experiments import propose_experiments
from .search import run_budgeted_search
from .variation import reevaluate_variation


def load_pipeline(source: dict | str, predictors: dict | None = None) -> tuple[dict, dict]:
    """設定と予測器を読み、予測前に契約を確認します。"""
    config = read_config(source)
    models = load_predictors(config) if predictors is None else predictors
    validate_config(config, models)
    return config, models


def run_search(config: dict | str, predictors: dict | None = None, *, recommendation_count: int = 8) -> dict:
    """全工程、または固定・実測工程より下流の条件を評価予算内で探索します。"""
    prepared, models = load_pipeline(config, predictors)
    result = run_budgeted_search(prepared, models, recommendation_count=recommendation_count)
    if prepared.get("process_variation") and result["candidates"]:
        result["input_variation"] = reevaluate_variation(prepared, models, result["candidates"])
    if prepared.get("inventory"):
        result["additional_experiments"] = propose_experiments(prepared, models)
    return result


def propose_additional_experiments(config: dict | str, predictors: dict | None = None) -> list[dict]:
    """現場で準備できる在庫ワークと費用から追加実験を選びます。"""
    prepared, models = load_pipeline(config, predictors)
    return propose_experiments(prepared, models)


def evaluate_acceptance(config: dict | str, state_overrides: list[dict] | dict, predictors: dict | None = None, *, budget: int | None = None) -> dict:
    """少数の上流状態ごとに残り工程を再探索し、局所的な受入情報を返します。"""
    prepared, models = load_pipeline(config, predictors)
    plan = validate_config(prepared, models)
    overrides = state_overrides if isinstance(state_overrides, list) else [state_overrides]
    if not overrides:
        return {"states": [], "evaluation_count": 0}
    settings = prepared.get("acceptance", {})
    if len(overrides) > int(settings.get("state_limit", 16)):
        raise ValueError("受入状態の件数がstate_limitを超えます。全域の事前列挙は行いません。")
    maximum = int(budget if budget is not None else settings.get("max_evaluations", 512))
    if maximum < len(overrides):
        raise ValueError("受入評価予算は状態数以上にしてください。")
    rows, evaluated = [], 0
    seconds = float(settings.get("max_seconds", 60.0))
    if not math.isfinite(seconds) or seconds <= 0:
        raise ValueError("受入状態評価のmax_secondsは正の有限値です。")
    started = time.monotonic()
    for index, override in enumerate(overrides):
        remaining_seconds = seconds - (time.monotonic() - started)
        if remaining_seconds <= 0:
            break
        local = copy.deepcopy(prepared)
        local.pop("process_variation", None)
        local.pop("inventory", None)
        state = override.get("outputs", override)
        if not isinstance(state, dict):
            raise ValueError("受入状態は修飾名と値の辞書です。")
        local["hypothetical_outputs"] = {**local.get("hypothetical_outputs", {}), **state}
        frozen_stages = {qualified(name)[0] for name in state} | set(plan["completed_stages"])
        for stage in reversed(plan["order"]):
            if stage in frozen_stages:
                frozen_stages.update(qualified(source)[0] for target, source in plan["edges"].items() if qualified(target)[0] == stage)
        reference = {name: value for name, value in prepared.get("reference_controls", {}).items() if qualified(name)[0] in frozen_stages}
        local["fixed_controls"] = {**local.get("fixed_controls", {}), **reference, **override.get("fixed_controls", {})}
        local["search"] = {**local.get("search", {}), "max_evaluations": max(1, (maximum - evaluated) // (len(overrides) - index)),
                           "max_seconds": min(float(local.get("search", {}).get("max_seconds", 120.0)), remaining_seconds)}
        result = run_budgeted_search(local, models, recommendation_count=1)
        evaluated += result["evaluation_count"]
        rows.append({"incoming_outputs": state, "workpiece_id": override.get("workpiece_id"),
                     "reachable_state": bool(override.get("reachable", False)), "status": result["status"],
                     "best": result["best"], "evaluation_count": result["evaluation_count"],
                     "interpretation": "local_budgeted_recourse_not_exact_acceptance_region"})
    return {"states": rows, "evaluation_count": evaluated, "unvisited_state_count": len(overrides) - len(rows),
            "claim": "hypothetical_states_are_not_executable_experiment_inventory"}
