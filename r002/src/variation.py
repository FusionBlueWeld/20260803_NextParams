"""工程入力ばらつきだけを同じシナリオで再評価します。モデル誤差とは分けます。"""

from __future__ import annotations

import copy
import math
import time

import numpy as np

from .config import validate_config
from .pipeline import evaluate_pipeline


def wilson_interval(successes: int, samples: int) -> tuple[float, float]:
    """Monte Carlo標本誤差だけのWilson 95%区間を返します。"""
    if samples == 0:
        return 0.0, 1.0
    z, p = 1.959963984540054, successes / samples
    denominator = 1 + z * z / samples
    centre = (p + z * z / (2 * samples)) / denominator
    radius = z * math.sqrt(p * (1 - p) / samples + z * z / (4 * samples * samples)) / denominator
    return max(0.0, centre - radius), min(1.0, centre + radius)


def reevaluate_variation(config: dict, predictors: dict, candidates: list[dict]) -> dict:
    """予算内の候補を入力変動で比較します。適用範囲外をクリップしません。"""
    setting = config.get("process_variation", {})
    limit = min(len(candidates), int(setting.get("candidate_limit", 8)))
    samples = int(setting.get("samples", 256))
    total_limit = int(setting.get("max_evaluations", max(1, samples * limit)))
    seconds = float(setting.get("max_seconds", 60.0))
    if limit < 1 or samples < 1 or total_limit < 1 or not math.isfinite(seconds) or seconds <= 0:
        raise ValueError("工程ばらつき評価の候補数・標本数・予算を正にしてください。")
    requested_samples = samples
    limit = min(limit, total_limit)
    samples = min(samples, max(1, total_limit // limit))
    plan = validate_config(config, predictors)
    deviations = setting.get("standard_deviations", {})
    permitted = set(plan["controls"]) | set(config.get("external_context", {}))
    if not set(deviations) <= permitted or any(not math.isfinite(float(value)) or float(value) < 0 for value in deviations.values()):
        raise ValueError("工程ばらつきは操作・外部状態の非負有限標準偏差で指定します。")
    rng = np.random.default_rng(int(setting.get("seed", 0)))
    # 候補間で同じ入力偏差を使い、順位比較に余分な乱数差を持ち込みません。
    deltas = {name: rng.normal(0.0, float(std), samples) for name, std in deviations.items()}
    for factor in setting.get("shared_factors", []):
        draw = rng.normal(0.0, 1.0, samples)
        for name, loading in factor.get("loadings", {}).items():
            if name not in permitted or not math.isfinite(float(loading)):
                raise ValueError("共通ロット変動のloadingsが不正です。")
            deltas[name] = deltas.get(name, np.zeros(samples)) + float(loading) * draw
    started = time.monotonic()
    evaluations = 0
    rows = []
    for candidate in candidates[:limit]:
        passes = physical_passes = valid_count = support_count = evaluated = 0
        controls = candidate.get("controls", candidate)
        for start in range(0, samples, int(plan["search"]["batch_size"])):
            if evaluations >= total_limit or time.monotonic() - started >= seconds:
                break
            count = min(int(plan["search"]["batch_size"]), samples - start, total_limit - evaluations)
            local = copy.deepcopy(config)
            values = {name: np.full(count, float(controls[name])) + deltas.get(name, np.zeros(samples))[start:start + count] for name in plan["controls"]}
            # 外部状態の変動は小標本ごとに評価します。観測ノイズはここへ入れません。
            external_varied = any(name in deltas for name in config.get("external_context", {}))
            if external_varied:
                for index in range(count):
                    for name, value in config.get("external_context", {}).items():
                        local["external_context"][name] = float(value) + float(deltas.get(name, np.zeros(samples))[start + index])
                    one = {name: value[index:index + 1] for name, value in values.items()}
                    result = evaluate_pipeline(local, predictors, one, validated=plan)
                    passes += int(np.sum(result["trusted_feasible"]))
                    physical_passes += int(np.sum(result["valid_connection"] & (result["process_margin"] >= 0)))
                    valid_count += int(np.sum(result["valid_connection"]))
                    support_count += int(np.sum(result["minimum_support"] >= plan["search"]["support_threshold"]))
            else:
                result = evaluate_pipeline(local, predictors, values, validated=plan)
                passes += int(np.sum(result["trusted_feasible"]))
                physical_passes += int(np.sum(result["valid_connection"] & (result["process_margin"] >= 0)))
                valid_count += int(np.sum(result["valid_connection"]))
                support_count += int(np.sum(result["minimum_support"] >= plan["search"]["support_threshold"]))
            evaluations += count
            evaluated += count
        if evaluated:
            lower, upper = wilson_interval(passes, evaluated)
            rows.append({"controls": controls, "samples": evaluated, "trusted_pass_fraction": passes / evaluated,
                         "mean_model_specification_pass_fraction": physical_passes / evaluated,
                         "valid_connection_fraction": valid_count / evaluated, "supported_fraction": support_count / evaluated,
                         "mc95_lower": lower, "mc95_upper": upper})
    rows.sort(key=lambda row: (row["mc95_lower"], row["trusted_pass_fraction"]), reverse=True)
    return {"status": "UNCALIBRATED_INPUT_VARIATION_ESTIMATE", "candidates": rows,
            "requested_samples_per_candidate": requested_samples,
            "evaluation_count": evaluations, "elapsed_seconds": time.monotonic() - started,
            "distribution": "unclipped_normal_with_optional_shared_lot_factors",
            "interval_scope": "monte_carlo_sampling_error_only",
            "excluded_uncertainty": ["model_error", "joint_output_uncertainty", "measurement_noise"],
            "observation_noise_status": "NOT_PROPAGATED_AS_PHYSICAL_VARIATION"}
