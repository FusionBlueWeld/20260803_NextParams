"""設定した工程入力のばらつきを平均予測へ伝播し、標本上の良品率を比較します。

Wilson区間はMonte Carlo標本誤差だけを表し、学習モデル誤差や実機校正を含みません。
"""

from __future__ import annotations

import numpy as np
from .pipeline import input_bounds, margin_and_feasible, run_stage_chain


def wilson_interval(successes: np.ndarray, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """標本数と成功数から良品比率の95%Wilson区間を求めます。モデル誤差は含みません。"""

    z = 1.959963984540054
    probability = successes / samples
    denominator = 1.0 + z * z / samples
    center = (probability + z * z / (2 * samples)) / denominator
    radius = z * np.sqrt(probability * (1 - probability) / samples + z * z / (4 * samples * samples)) / denominator
    low = np.where(successes == 0, 0.0, center - radius)
    high = np.where(successes == samples, 1.0, center + radius)
    return np.clip(low, 0.0, 1.0), np.clip(high, 0.0, 1.0)


def process_variation_rows(
    predictors, config: dict[str, object], connections: dict[str, str], axis_names: list[str],
    values: dict[str, np.ndarray], order: np.ndarray, deterministic_margin: np.ndarray,
) -> list[dict[str, object]]:
    """同じseedで入力ばらつきを加え、良品率の区間下限を優先して候補を並べます。"""

    variation = config["process_variation"]
    samples = int(variation["samples"])
    candidate_indices = np.asarray(order[:min(int(variation["candidate_limit"]), len(order))], dtype=int)
    candidate_count = len(candidate_indices)
    rng = np.random.default_rng(int(variation["seed"]))
    deviations = {name: float(value) for name, value in variation["standard_deviations"].items()}

    bounds_by_name: dict[str, tuple[float, float]] = {}
    for item, manifest, predictor in predictors:
        bounds = input_bounds(predictor)
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

    stage_outputs, _, valid = run_stage_chain(predictors, config, connections, scenario_values)
    final = stage_outputs[str(predictors[-1][0]["id"])]
    _, feasible = margin_and_feasible(final, config.get("final_specifications", []))
    good = (feasible & valid).reshape(candidate_count, samples)
    valid_fraction = valid.reshape(candidate_count, samples).mean(axis=1)
    successes = good.sum(axis=1)
    probability = successes / samples
    low, high = wilson_interval(successes, samples)
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

