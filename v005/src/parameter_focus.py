"""実測から探索の配分を更新する。小さい作用も永久には削除しない。

補助回帰は説明と実験配分専用。Hybridの学習列・物理制約は変更しない。
upper_effectsは再標本化・予測誤差を含む保守的な目安で、信頼区間ではない。
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass, field
import hashlib
import json
from pathlib import Path

import numpy as np

from .candidate_identity import CandidateIdentity
from .connection import current_context_observation_mask
from .parameter_space import _axis_count, make_parameter_grid
from .preprocessing import normalize_parameters


REVISION = "numpy-sequential-focus-1"


@dataclass
class ParameterAssessment:
    """変数ごとの判断。保存時は明示的な辞書へ変換する。"""
    column: str
    state: str = "UNDECIDED"
    effects: dict = field(default_factory=dict)
    upper_effects: dict = field(default_factory=dict)
    weight: float = 1.
    stable_updates: int = 0
    fixed_value: float | None = None
    background_checks: int = 0
    reason: str = "作用を判断する実測が不足"


def _fingerprint(value):
    encoded = json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def control_columns(problem, connection):
    """操作対象と流入状態を混同しない。接続なしなら全入力を操作対象とする。"""
    names = [p.column for p in problem.parameters]
    controls = names if connection is None else list(connection.controls)
    return [names.index(name) for name in controls]


def _basis(z, quadratic):
    columns = [np.ones(len(z))]
    for j in range(z.shape[1]):
        columns.append(z[:, j])
        if quadratic[j]:
            columns.append(z[:, j] ** 2)
    return np.column_stack(columns)


def _fit(features, values):
    """切片は罰せず、弱いL2正則化を最小二乗の追加行で表す。"""
    penalty = np.sqrt(1e-3) * np.eye(features.shape[1])
    penalty[0, 0] = 0
    return np.linalg.lstsq(np.vstack((features, penalty)),
        np.vstack((values, np.zeros((len(penalty), values.shape[1])))), rcond=None)[0]


def _comparison_checks(x, y, column, problem, context, thresholds, anchor=None):
    """他入力が同じ実測同士を比較。複数背景と中間水準の確認を要求する。"""
    keys = CandidateIdentity(problem, context).keys(x)
    groups = defaultdict(list)
    for i, key in enumerate(keys):
        groups[key[:column] + key[column + 1:]].append(i)
    checks = 0
    violated = False
    anchor_checked = False
    anchor_key = None if anchor is None else CandidateIdentity(problem, context).keys([anchor])[0]
    anchor_background = None if anchor_key is None else anchor_key[:column] + anchor_key[column + 1:]
    variable = problem.parameters[column]
    count = _axis_count(variable)
    required_levels = {variable.upper if i == count - 1 else variable.lower + i * variable.step
                       for i in range(min(count, 16))}
    for background, indices in groups.items():
        if not required_levels.issubset(set(x[indices, column])):
            continue
        difference = np.ptp(y[indices], axis=0)
        if np.any(difference > thresholds):
            violated = True
        else:
            checks += 1
            anchor_checked |= background == anchor_background
    return checks, violated, anchor_checked


def analyze_focus(prepared, problem, connection, config, previous=None, region_policy=None):
    """現在の流入状態の実測だけで、出力別作用と一時固定の根拠を評価する。"""
    controls = control_columns(problem, connection)
    context = {} if connection is None else connection.current_incoming_context
    mask = current_context_observation_mask(prepared.raw_parameters, problem, connection)
    x = prepared.raw_parameters[mask]
    outputs = [problem.objective, *problem.constraints]
    y = np.column_stack([prepared.result_means[o.column][mask] for o in outputs])
    settings = _fingerprint({"revision": REVISION, "problem": asdict(problem),
                             "context": context, "config": asdict(config),
                             "regions": None if region_policy is None else asdict(region_policy)})
    data = _fingerprint({"x": x.tolist(), "y": y.tolist()})
    previous = previous or {}
    if previous.get("settings_fingerprint") != settings:
        previous = {}
    observations = {_fingerprint(row.tolist()): _fingerprint(value.tolist()) for row, value in zip(x, y)}
    prior_observations = previous.get('observation_fingerprints', {})
    if any(observations.get(key) != value for key, value in prior_observations.items()):
        previous = {}  # 訂正・削除・反復平均の変更を、独立条件の追加と混同しない。
        prior_observations = {}
    old = {p["column"]: p for p in previous.get("parameters", [])}
    changed = bool(set(observations) - set(prior_observations))
    state = {"revision": REVISION, "settings_fingerprint": settings,
             "data_fingerprint": data, "data_rows": len(x), "parameters": [],
             "fixed_values": {}, "anchor": None, "phase": "SCREENING",
             "observation_fingerprints": observations,
             "update_index": int(previous.get('update_index', 0)) + int(changed),
             "uncertainty_interpretation": "diagnostic_upper_effect_not_confidence_bound"}

    # 固定値は現在の流入状態で実測済みの合格条件から選ぶ。
    feasible = np.ones(len(x), dtype=bool)
    for i, c in enumerate(problem.constraints, start=1):
        feasible &= y[:, i] <= c.target if c.direction == "less_equal" else y[:, i] >= c.target
    if region_policy is not None and len(x):
        feasible &= region_policy.evaluate(x, [p.column for p in problem.parameters]).allowed_mask
    good = np.flatnonzero(feasible)
    if len(good):
        local = np.argmin(y[good, 0]) if problem.objective.direction == "minimize" else np.argmax(y[good, 0])
        state["anchor"] = x[good[local]].tolist()

    # 分析コストにも上限。対比確認には間引く前の全実測を使う。
    indices = np.arange(len(x))
    if len(indices) > config.max_screening_samples:
        indices = np.sort(np.random.default_rng(42).choice(indices, config.max_screening_samples, replace=False))
    fit_x, fit_y = x[indices], y[indices]
    z = 2 * normalize_parameters(fit_x, problem)[:, controls] - 1 if len(fit_x) else np.empty((0, len(controls)))
    quadratic = [_axis_count(problem.parameters[j]) >= 3 for j in controls]
    features = _basis(z, quadratic)
    enough = len(fit_x) >= features.shape[1] + 4
    categorical = any(str(problem.parameters[j].unit).casefold() in
                      {'category', 'categorical', 'カテゴリ', '名義', '名義尺度'} for j in controls)
    rank_ok = bool(enough and not categorical and np.linalg.matrix_rank(features) == features.shape[1])
    fits, errors = [], np.zeros(len(outputs))
    if rank_ok:
        fits.append(_fit(features, fit_y))
        rng = np.random.default_rng(42)
        for _ in range(config.resamples):
            keep = rng.choice(len(fit_x), max(features.shape[1] + 1, int(len(fit_x) * .8)), replace=False)
            model = _fit(features[keep], fit_y[keep])
            fits.append(model)
            held = np.setdiff1d(np.arange(len(fit_x)), keep)
            if len(held):
                errors = np.maximum(errors, np.quantile(np.abs(features[held] @ model - fit_y[held]), .9, axis=0))
        errors = np.maximum(errors, np.quantile(np.abs(features @ fits[0] - fit_y), .9, axis=0))
    state["analysis"] = {"conditions_used": len(fit_x), "basis_columns": features.shape[1],
                          "identifiable": rank_ok, "prediction_errors": dict(zip([o.column for o in outputs], errors.tolist()))}

    for local, column in enumerate(controls):
        variable = problem.parameters[column]
        entry = asdict(ParameterAssessment(variable.column))
        prior = old.get(variable.column, {})
        observed_levels = len(np.unique(x[:, column])) if len(x) else 0
        if observed_levels < 2:
            entry.update(state="UNTESTED", reason="操作変数がまだ変動していない。無効とは判定しない")
        elif rank_ok and variable.unit != "category":
            # 実際の離散水準を使用。巨大な軸では最大16代表水準を使い固定に慎重になる。
            count = _axis_count(variable)
            level_indices = np.unique(np.linspace(0, count - 1, min(count, 16)).astype(int))
            levels = np.where(level_indices == count - 1, variable.upper, variable.lower + level_indices * variable.step)
            probe = np.tile(np.mean(z, axis=0), (len(levels), 1))
            probe[:, local] = 2 * (levels - variable.lower) / (variable.upper - variable.lower) - 1
            prediction = np.stack([_basis(probe, quadratic) @ model for model in fits])
            ranges = np.ptp(prediction, axis=1)
            effect = ranges[0]
            noise = np.array([prepared.result_noise_std.get(o.column, 0.) for o in outputs])
            upper = np.max(ranges, axis=0) + 2 * errors + 2 * noise
            entry["effects"] = dict(zip([o.column for o in outputs], effect.tolist()))
            entry["upper_effects"] = dict(zip([o.column for o in outputs], upper.tolist()))
            scale = np.maximum(np.std(fit_y, axis=0), 1e-9)
            relevance = float(np.max(effect / scale))
            # 判断の不確かさが大きければ配分を落とさない。
            uncertain = bool(np.any(errors > .2 * scale))
            entry.update(state="ACTIVE" if relevance >= .25 else "UNDECIDED",
                         weight=1. if uncertain else float(np.clip(relevance, .2, 1.)),
                         reason="出力別の作用と未観測誤差を評価。組合せ作用は確認実験で検査")
            declared = all(o.column in config.effect_tolerances for o in outputs)
            if declared:
                thresholds = np.array([config.effect_tolerances[o.column] for o in outputs])
                checks, violation, anchor_checked = _comparison_checks(x, y, column, problem, context, thresholds, state['anchor'])
                entry["background_checks"] = checks
                # 制約余裕が許容作用より小さい場合は固定を許さない。
                margin_ok = state["anchor"] is not None
                if margin_ok:
                    anchor_key = CandidateIdentity(problem, context).keys([state["anchor"]])[0]
                    anchor_index = CandidateIdentity(problem, context).keys(x).index(anchor_key)
                    for j, c in enumerate(problem.constraints, 1):
                        margin = c.target - y[anchor_index, j] if c.direction == "less_equal" else y[anchor_index, j] - c.target
                        margin_ok &= margin > upper[j]
                small = bool(np.all(upper < thresholds) and margin_ok and not violation
                             and variable.column not in config.never_fix and count <= 16)
                if small:
                    streak = int(prior.get("stable_updates", 0)) + int(changed)
                    entry.update(state="FIX_CANDIDATE", stable_updates=streak, weight=.2,
                                 reason="全出力の作用上限が許容差未満。複数背景の実測確認が必要")
                    if (config.temporary_fix_enabled and not prior.get('hybrid_recheck', False)
                            and streak >= config.stability_updates
                            and checks >= config.min_background_checks and anchor_checked):
                        fixed = state["anchor"][column]
                        entry.update(state="TEMPORARILY_FIXED", fixed_value=float(fixed), weight=.2,
                                     reason="複数背景の水準比較と判断の安定を確認。一時固定し定期確認する")
                        state["fixed_values"][variable.column] = float(fixed)
                elif prior.get("state") == "TEMPORARILY_FIXED":
                    entry["reason"] = "作用増大・誤差・制約余裕の不足により一時固定を解除"
        state["parameters"].append(entry)
    if rank_ok:
        state["phase"] = "FOCUSED_SEARCH" if state["fixed_values"] else "ADAPTIVE_SEARCH"
    state["state_changed"] = bool(previous and previous.get("fixed_values", {}) != state["fixed_values"])
    return state


def save_focus(path: Path, state):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(state, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def focus_candidate_plan(grid, prepared, problem, connection, state, config, search_config, region_policy):
    """全入力を保持した改善候補と確認候補を、同じ候補予算内で合成する。"""
    context = {} if connection is None else connection.current_incoming_context
    names = [p.column for p in problem.parameters]
    entries = state["parameters"]
    anchor = state.get("anchor")
    fixed = state.get("fixed_values", {})
    main = np.array(grid, copy=True)
    # 全格子生成後に代入しても計算量は減らないので、固定時は縮小軸を先に生成する。
    if fixed:
        from .connection import StageConnection
        controls = tuple(p.column for p in problem.parameters if p.column not in context and p.column not in fixed)
        if controls:
            focused_connection = StageConnection("focus", controls, tuple((*context, *fixed)), (), {**context, **fixed})
            main = make_parameter_grid(problem, focused_connection, max_candidates=search_config.max_candidates,
                                       seed=search_config.seed, mode=search_config.candidate_mode)
        else:
            main = np.asarray([anchor], dtype=float)
            for name, value in fixed.items():
                main[:, names.index(name)] = value
    probes, reasons = [], {}
    if anchor is not None:
        for entry in entries:
            if entry["state"] == "ACTIVE":
                continue
            column = names.index(entry["column"])
            p = problem.parameters[column]
            count = _axis_count(p)
            for index in np.unique(np.linspace(0, count - 1, min(16, count)).astype(int)):
                row = np.asarray(anchor, dtype=float).copy()
                row[column] = p.upper if index == count - 1 else p.lower + index * p.step
                probes.append(row)
                reasons[tuple(row)] = "FROZEN_CHECK" if entry["state"] == "TEMPORARILY_FIXED" else "SCREENING"
    # 定期的に別の背景も確認。直近最適点の一変数対比だけに閉じ込めない。
    due_global = state.get('update_index', 1) % config.global_check_interval == 0
    global_pool = make_parameter_grid(problem, connection, max_candidates=min(64, search_config.max_candidates),
                                      seed=search_config.seed + len(prepared.raw_parameters), mode="budget")
    global_indices = np.unique(np.linspace(0, len(global_pool) - 1,
        min(len(global_pool), max(8, config.confirmation_slots * 2))).astype(int))
    if due_global or anchor is None:
        for row in global_pool[global_indices]:
            probes.append(row)
            reasons[tuple(row)] = 'GLOBAL_CHECK'
    # 背景比較用に、広域の既測定条件からも一変数を振る。
    context_rows = prepared.raw_parameters[current_context_observation_mask(prepared.raw_parameters, problem, connection)]
    for background in context_rows[:config.min_background_checks]:
        for entry in entries:
            if entry["state"] not in ("FIX_CANDIDATE", "TEMPORARILY_FIXED", "UNDECIDED", "UNTESTED"):
                continue
            j = names.index(entry["column"])
            p = problem.parameters[j]
            for index in np.unique(np.linspace(0, _axis_count(p) - 1, min(16, _axis_count(p))).astype(int)):
                row = background.copy()
                row[j] = p.upper if index == _axis_count(p) - 1 else p.lower + index * p.step
                probes.append(row)
                reasons.setdefault(tuple(row), "FROZEN_CHECK" if entry["state"] == "TEMPORARILY_FIXED" else "SCREENING")
    if probes:
        probe_array = np.unique(np.asarray(probes), axis=0)
        valid = region_policy.evaluate(probe_array, names).allowed_mask
        valid &= ~CandidateIdentity(problem, context).measured_mask(probe_array, prepared.raw_parameters)
        probe_array = probe_array[valid][:min(128, search_config.max_candidates)]
    else:
        probe_array = np.empty((0, len(names)))
    # 確認候補を優先して予算に収める。完全列挙の事実は別途検査する。
    existing_keys = set(map(tuple, main))
    extra_count = sum(tuple(row) not in existing_keys for row in probe_array)
    available = search_config.max_candidates - len(probe_array) if len(main) + extra_count > search_config.max_candidates else len(main)
    if len(main) > available:
        index = np.linspace(0, len(main) - 1, available).astype(int)
        main = main[index]
    pool = np.unique(np.vstack((main, probe_array)), axis=0)
    main_keys = set(map(tuple, main))
    probe_keys = set(map(tuple, probe_array))
    roles = [reasons.get(tuple(row), "SCREENING") if tuple(row) in probe_keys else "" for row in pool]
    main_mask = np.array([tuple(row) in main_keys for row in pool])
    # 弱い変数を動かす候補の優先度を緩やかに下げる。保留変数は完全には除外しない。
    weights = np.ones(len(pool))
    if anchor is not None:
        for entry in entries:
            j = names.index(entry["column"])
            deviation = np.abs(pool[:, j] - anchor[j]) / (problem.parameters[j].upper - problem.parameters[j].lower)
            weights *= 1 - (1 - entry["weight"]) * deviation * .5
    return pool, {"main_mask": main_mask, "confirmation_roles": roles, "weights": weights,
                  "candidate_space_scope": "focused_with_checks" if fixed else "full",
                  "pool_changed": bool(fixed or len(pool) != len(grid) or not np.array_equal(pool, grid))}


def check_hybrid_effects(state, grid, prediction, plan, problem, config):
    """確認用点のHybrid予測との食い違いを記録し、次のrunの固定を保留する。

同じ背景の点を比較するだけで追加学習・追加予測は行わない。
食い違いがないことは、未確認の組合せ作用がない証明ではない。
"""
    selected = np.flatnonzero(np.asarray(plan['confirmation_roles'], dtype=object) != '')
    names = [p.column for p in problem.parameters]
    for entry in state['parameters']:
        if entry['state'] not in ('FIX_CANDIDATE', 'TEMPORARILY_FIXED'):
            continue
        j = names.index(entry['column'])
        groups = defaultdict(list)
        for i in selected:
            row = tuple(grid[i])
            groups[row[:j] + row[j + 1:]].append(i)
        effects = {}
        for output in [problem.objective, *problem.constraints]:
            ranges = [float(np.ptp(prediction.results[output.column]['hybrid_mean'][indices]))
                      for indices in groups.values() if len(indices) >= 2]
            if ranges:
                effects[output.column] = max(ranges)
        entry['hybrid_effects'] = effects
        entry['hybrid_recheck'] = any(value > config.effect_tolerances[name] for name, value in effects.items())
        if entry['hybrid_recheck']:
            entry['reason'] += '。Hybridの水準比較で作用が大きい可能性があり、次runの固定を保留'
