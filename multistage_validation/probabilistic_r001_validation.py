"""r001の工程ばらつき確率を、独立seedの連結物理oracleで校正検証する。"""

from __future__ import annotations

import csv
import itertools
import json
import sys
from pathlib import Path

import numpy as np

from multistage_validation.functional_coating.benchmark import CONTROL_AXES
from multistage_validation.functional_coating.pipeline import DEFAULT_MATERIAL_STATE, LINE_INPUT_BOUNDS, evaluate_line
from multistage_validation.functional_coating.robustness import DEFAULT_STANDARD_DEVIATIONS, sample_conditions


ROOT = Path(__file__).resolve().parents[1]
CONTROL_NAMES = (
    "coating_gap_um", "line_speed_m_min", "web_tension_n",
    "air_temperature_c", "air_speed_m_s", "residence_time_min",
    "oven_temperature_c", "hold_time_min", "nip_pressure_mpa",
)


def _wilson(successes: int, samples: int) -> tuple[float, float]:
    z = 1.959963984540054
    p = successes / samples
    denominator = 1 + z * z / samples
    center = (p + z * z / (2 * samples)) / denominator
    radius = z * np.sqrt(p * (1 - p) / samples + z * z / (4 * samples * samples)) / denominator
    low = 0.0 if successes == 0 else float(center - radius)
    high = 1.0 if successes == samples else float(center + radius)
    return max(0.0, low), min(1.0, high)


def _exhaustive_oracle_baseline(samples: int, seed: int, chunk_size: int = 128) -> dict[str, object]:
    """全19,683中心条件を共通乱数で評価する物理oracle比較基準。"""
    points = np.asarray(list(itertools.product(*(CONTROL_AXES[name] for name in CONTROL_NAMES))), dtype=float)
    rng = np.random.default_rng(seed)
    standard_normal = {name: rng.normal(size=samples) for name in LINE_INPUT_BOUNDS}
    probability = np.empty(len(points))
    for start in range(0, len(points), chunk_size):
        stop = min(start + chunk_size, len(points))
        conditions: dict[str, np.ndarray] = {}
        for column, name in enumerate(CONTROL_NAMES):
            lower, upper = LINE_INPUT_BOUNDS[name]
            values = points[start:stop, column, None] + DEFAULT_STANDARD_DEVIATIONS[name] * standard_normal[name][None, :]
            conditions[name] = np.clip(values, lower, upper).reshape(-1)
        for name, center in DEFAULT_MATERIAL_STATE.items():
            lower, upper = LINE_INPUT_BOUNDS[name]
            values = center + DEFAULT_STANDARD_DEVIATIONS[name] * standard_normal[name][None, :]
            conditions[name] = np.tile(np.clip(values, lower, upper), stop - start)
        good = np.asarray(evaluate_line(conditions)["final"]["feasible"], dtype=bool)
        probability[start:stop] = good.reshape(stop - start, samples).mean(axis=1)
    best = float(np.max(probability))
    best_indices = np.flatnonzero(probability == best)
    first = int(best_indices[0])
    return {
        "candidate_count": len(points), "samples_per_candidate": samples, "seed": seed,
        "best_probability": best, "best_tie_count": len(best_indices),
        "one_best_controls": {name: float(points[first, column]) for column, name in enumerate(CONTROL_NAMES)},
        "probabilities": probability,
        "points": points,
    }


def run(trial_name: str, report_path: Path, samples: int = 4096, seed: int = 20260909, exhaustive_samples: int = 1024) -> dict[str, object]:
    trial = ROOT / "r001" / "trials" / trial_name
    with (trial / "output" / "robust_recommendations.csv").open(encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file))
    if not rows:
        raise ValueError("robust_recommendations.csv is empty")

    details = []
    for index, row in enumerate(rows):
        nominal = {**DEFAULT_MATERIAL_STATE, **{name: float(row[name]) for name in CONTROL_NAMES}}
        scenarios = sample_conditions(nominal, samples=samples, seed=seed)
        result = evaluate_line(scenarios)
        good = np.asarray(result["final"]["feasible"], dtype=bool)
        successes = int(good.sum())
        oracle_probability = successes / samples
        predicted_probability = float(row["predicted_good_probability"])
        oracle_low, oracle_high = _wilson(successes, samples)
        details.append({
            "robust_rank": int(row["robust_rank"]),
            "predicted_probability": predicted_probability,
            "predicted_mc95_low": float(row["mc95_low"]),
            "predicted_mc95_high": float(row["mc95_high"]),
            "oracle_probability": oracle_probability,
            "oracle_mc95_low": oracle_low,
            "oracle_mc95_high": oracle_high,
            "calibration_error": predicted_probability - oracle_probability,
            "predicted_interval_contains_oracle_probability": float(row["mc95_low"]) <= oracle_probability <= float(row["mc95_high"]),
            "oracle_mean_quality_margin": float(np.mean(result["final"]["quality_margin"])),
            "oracle_p05_quality_margin": float(np.quantile(result["final"]["quality_margin"], 0.05)),
            "controls": {name: nominal[name] for name in CONTROL_NAMES},
        })

    predicted = np.asarray([item["predicted_probability"] for item in details])
    actual = np.asarray([item["oracle_probability"] for item in details])
    oracle_order = np.argsort(-actual, kind="stable")
    selected = details[0]
    oracle_best = details[int(oracle_order[0])]
    exhaustive = _exhaustive_oracle_baseline(exhaustive_samples, seed)
    selected_point = np.asarray([selected["controls"][name] for name in CONTROL_NAMES])
    selected_global_index = int(np.flatnonzero(np.all(np.isclose(exhaustive["points"], selected_point), axis=1))[0])
    selected_global_probability = float(exhaustive["probabilities"][selected_global_index])
    selected_global_rank = int(1 + np.sum(exhaustive["probabilities"] > selected_global_probability))
    # Bernoulli outcomeに対するBrierを、各候補のoracle達成率から厳密に集約。
    brier = np.mean(actual * (1 - predicted) ** 2 + (1 - actual) * predicted ** 2)
    mae = float(np.mean(np.abs(predicted - actual)))
    rmse = float(np.sqrt(np.mean((predicted - actual) ** 2)))
    evaluated_probability_regret = float(oracle_best["oracle_probability"] - selected["oracle_probability"])
    global_probability_regret = float(exhaustive["best_probability"] - selected_global_probability)
    thresholds = {"mean_absolute_error_max": 0.05, "bernoulli_brier_score_max": 0.10, "selection_probability_regret_max": 0.05}
    checks = {
        "mean_absolute_error": mae <= thresholds["mean_absolute_error_max"],
        "bernoulli_brier_score": float(brier) <= thresholds["bernoulli_brier_score_max"],
        "selection_probability_regret": global_probability_regret <= thresholds["selection_probability_regret_max"],
    }
    report = {
        "status": "pass" if all(checks.values()) else "fail",
        "trial": trial_name,
        "candidate_count": len(details),
        "oracle_samples_per_candidate": samples,
        "oracle_seed": seed,
        "independent_from_optimization_seed": True,
        "selected": selected,
        "selected_oracle_rank_within_evaluated_candidates": int(np.flatnonzero(oracle_order == 0)[0] + 1),
        "oracle_best_evaluated_candidate": oracle_best,
        "oracle_probability_regret_within_evaluated_candidates": evaluated_probability_regret,
        "exhaustive_oracle_baseline": {
            "candidate_count": exhaustive["candidate_count"],
            "samples_per_candidate": exhaustive["samples_per_candidate"],
            "seed": exhaustive["seed"],
            "best_probability": exhaustive["best_probability"],
            "best_tie_count": exhaustive["best_tie_count"],
            "one_best_controls": exhaustive["one_best_controls"],
            "selected_probability": selected_global_probability,
            "selected_rank": selected_global_rank,
            "selected_probability_regret": global_probability_regret,
        },
        "acceptance": {"thresholds": thresholds, "checks": checks},
        "calibration": {
            "mean_error": float(np.mean(predicted - actual)),
            "mean_absolute_error": mae,
            "root_mean_squared_error": rmse,
            "bernoulli_brier_score": float(brier),
            "predicted_mc95_interval_coverage": float(np.mean([item["predicted_interval_contains_oracle_probability"] for item in details])),
            "note": "predicted interval is Monte Carlo-only and is not expected to cover learned-model error",
        },
        "candidates": details,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit("usage: python -m multistage_validation.probabilistic_r001_validation TRIAL REPORT.json")
    value = run(sys.argv[1], Path(sys.argv[2]))
    print(json.dumps({key: value[key] for key in value if key != "candidates"}, ensure_ascii=True, indent=2))
