"""学習済み工程の接続予測、NG診断、条件修正後の回復を検証します。

診断が受け取るのは期待レシピ、観測レシピ、学習済み予測です。
既知の故障ラベルは診断後の採点にだけ使い、診断への正解漏洩を防ぎます。"""

from __future__ import annotations

import itertools
import json
import math
import sys
from pathlib import Path

import numpy as np

from validation.multistage.functional_coating.benchmark import CONTROL_AXES
from validation.multistage.functional_coating.coating import physics_model as coating_oracle
from validation.multistage.functional_coating.curing import physics_model as curing_oracle
from validation.multistage.functional_coating.drying import physics_model as drying_oracle
from validation.multistage.functional_coating.fault_scenarios import FAULT_SCENARIOS, REFERENCE_NOMINAL
from validation.multistage.functional_coating.pipeline import (
    FINAL_SPECIFICATIONS,
    LINE_INPUT_BOUNDS,
    evaluate_line,
)
from r001.src.bundle_runtime import load_bundle_runtime
from r001.src.pipeline import margin_and_feasible, run_stage_chain


from ..settings import PROJECT_ROOT as ROOT
PIPELINE_PATH = ROOT / "r001/trials/trial_validated_individual_chain_r001/pipeline.json"
REPORT_PATH = ROOT / "validation/multistage/results/r001_diagnostic_validation.json"
STAGE_CONTROL_NAMES = {
    "coating": ("coating_gap_um", "line_speed_m_min", "web_tension_n"),
    "drying": ("air_temperature_c", "air_speed_m_s", "residence_time_min"),
    "curing": ("oven_temperature_c", "hold_time_min", "nip_pressure_mpa"),
}


def _json_number(value: object) -> float | bool:
    array = np.asarray(value)
    return bool(array) if array.dtype == bool else float(array)


def _load_runtime() -> tuple[dict, list, dict[str, str]]:
    """固定trialの設定とbundleを読み込み、予測器一覧と接続先の対応を返します。"""

    config = json.loads(PIPELINE_PATH.read_text(encoding="utf-8"))
    load_stage_bundle = load_bundle_runtime()
    predictors = []
    for item in config["stages"]:
        manifest, predictor = load_stage_bundle(Path(item["bundle"]))
        predictors.append((item, manifest, predictor))
    connections = {item["to"]: item["from"] for item in config["connections"]}
    return config, predictors, connections


def _lhs(bounds: dict[str, tuple[float, float]], count: int, seed: int) -> dict[str, np.ndarray]:
    """各入力範囲を等分し、区間内乱数と並べ替えで再現可能な標本を作ります。"""

    rng = np.random.default_rng(seed)
    values: dict[str, np.ndarray] = {}
    for name, (lower, upper) in bounds.items():
        unit = (np.arange(count) + rng.random(count)) / count
        rng.shuffle(unit)
        values[name] = lower + (upper - lower) * unit
    return values


def _oracle_valid_holdout(count: int = 4096, seed: int = 20260908) -> tuple[dict, dict, float]:
    """Generate independent line inputs and retain physically connectable rows."""
    raw_count = count * 4
    x = _lhs(LINE_INPUT_BOUNDS, raw_count, seed)
    coat = coating_oracle.evaluate_model(
        coating_gap_um=x["coating_gap_um"], line_speed_m_min=x["line_speed_m_min"],
        web_tension_n=x["web_tension_n"], incoming_viscosity_pa_s=x["material_viscosity_pa_s"],
        incoming_solids_fraction=x["material_solids_fraction"], incoming_bubble_fraction=x["material_bubble_fraction"],
    )
    dry_bounds = drying_oracle.INCOMING_STATE_BOUNDS
    keep = np.ones(raw_count, dtype=bool)
    dry_inputs = {
        "incoming_wet_thickness_um": coat["wet_thickness_um"],
        "incoming_solids_fraction": coat["solids_fraction"],
        "incoming_thickness_cv_fraction": coat["thickness_cv_fraction"],
        "incoming_coating_defect_index": coat["coating_defect_index"],
    }
    for name, value in dry_inputs.items():
        keep &= (value >= dry_bounds[name][0]) & (value <= dry_bounds[name][1])
    idx = np.flatnonzero(keep)
    dry = drying_oracle.evaluate_model(
        air_temperature_c=x["air_temperature_c"][idx], air_speed_m_s=x["air_speed_m_s"][idx],
        residence_time_min=x["residence_time_min"][idx],
        **{name: value[idx] for name, value in dry_inputs.items()},
    )
    cure_bounds = curing_oracle.INCOMING_STATE_BOUNDS
    cure_inputs = {
        "incoming_dry_thickness_um": dry["dry_thickness_um"],
        "incoming_residual_solvent_pct": dry["residual_solvent_pct"],
        "incoming_internal_stress_mpa": dry["internal_stress_mpa"],
        "incoming_drying_defect_index": dry["drying_defect_index"],
        "incoming_thickness_cv_fraction": dry["dry_thickness_cv_fraction"],
    }
    keep2 = np.ones(len(idx), dtype=bool)
    for name, value in cure_inputs.items():
        keep2 &= (value >= cure_bounds[name][0]) & (value <= cure_bounds[name][1])
    physically_connectable = idx[np.flatnonzero(keep2)]
    acceptance = float(len(physically_connectable) / raw_count)
    idx = physically_connectable[:count]
    if len(idx) < count:
        raise RuntimeError(f"Only {len(idx)} physically connectable holdout rows")
    conditions = {name: values[idx] for name, values in x.items()}
    truth = evaluate_line(conditions)
    return conditions, truth, acceptance


def _metric_rows(truth: dict[str, np.ndarray], predicted: dict[str, np.ndarray]) -> dict[str, dict[str, float]]:
    result = {}
    for name in predicted:
        actual = np.asarray(truth[name], dtype=float)
        guess = np.asarray(predicted[name], dtype=float)
        mask = np.isfinite(guess)
        rmse = float(np.sqrt(np.mean((guess[mask] - actual[mask]) ** 2))) if np.any(mask) else math.inf
        scale = max(float(np.ptp(actual)), 1e-12)
        result[name] = {"rmse": rmse, "nrmse": rmse / scale, "valid_fraction": float(mask.mean())}
    return result


def _teacher_forced_stage(item, manifest, predictor, conditions: dict, oracle_inputs: dict) -> dict[str, np.ndarray]:
    inputs = {name: conditions[name] for name in manifest["controls"]}
    inputs.update(oracle_inputs)
    predicted = predictor.predict_arrays(inputs)
    return {name: np.asarray(predicted["outputs"][name]["mean"]) for name in manifest["predicted_outputs"]}


def _end_to_end(config: dict, predictors: list, connections: dict[str, str]) -> dict[str, object]:
    conditions, truth, acceptance = _oracle_valid_holdout()
    values = {
        **{name: conditions[name] for name in CONTROL_AXES},
        "coating.incoming_viscosity_pa_s": conditions["material_viscosity_pa_s"],
        "coating.incoming_solids_fraction": conditions["material_solids_fraction"],
        "coating.incoming_bubble_fraction": conditions["material_bubble_fraction"],
    }
    free, support, valid = run_stage_chain(predictors, config, connections, values)
    teacher = {}
    teacher["coating"] = _teacher_forced_stage(*predictors[0], conditions, {
        "incoming_viscosity_pa_s": conditions["material_viscosity_pa_s"],
        "incoming_solids_fraction": conditions["material_solids_fraction"],
        "incoming_bubble_fraction": conditions["material_bubble_fraction"],
    })
    teacher["drying"] = _teacher_forced_stage(*predictors[1], conditions, {
        "incoming_wet_thickness_um": truth["coating"]["wet_thickness_um"],
        "incoming_solids_fraction": truth["coating"]["solids_fraction"],
        "incoming_thickness_cv_fraction": truth["coating"]["thickness_cv_fraction"],
        "incoming_coating_defect_index": truth["coating"]["coating_defect_index"],
    })
    teacher["curing"] = _teacher_forced_stage(*predictors[2], conditions, {
        "incoming_dry_thickness_um": truth["drying"]["dry_thickness_um"],
        "incoming_residual_solvent_pct": truth["drying"]["residual_solvent_pct"],
        "incoming_internal_stress_mpa": truth["drying"]["internal_stress_mpa"],
        "incoming_drying_defect_index": truth["drying"]["drying_defect_index"],
        "incoming_thickness_cv_fraction": truth["drying"]["dry_thickness_cv_fraction"],
    })
    per_stage = {}
    for stage in ("coating", "drying", "curing"):
        teacher_metrics = _metric_rows(truth[stage], teacher[stage])
        free_metrics = _metric_rows(truth[stage], free[stage])
        per_stage[stage] = {
            "teacher_forced": teacher_metrics,
            "free_running": free_metrics,
            "max_teacher_forced_nrmse": max(row["nrmse"] for row in teacher_metrics.values()),
            "max_free_running_nrmse": max(row["nrmse"] for row in free_metrics.values()),
        }
    predicted_margin, predicted_ok = margin_and_feasible(free["curing"], config["final_specifications"])
    true_ok = np.asarray(truth["final"]["feasible"], dtype=bool)
    usable = np.asarray(valid, dtype=bool) & np.isfinite(predicted_margin)
    accuracy = float(np.mean(predicted_ok[usable] == true_ok[usable]))
    confusion = {
        "true_positive": int(np.sum(predicted_ok[usable] & true_ok[usable])),
        "true_negative": int(np.sum(~predicted_ok[usable] & ~true_ok[usable])),
        "false_positive": int(np.sum(predicted_ok[usable] & ~true_ok[usable])),
        "false_negative": int(np.sum(~predicted_ok[usable] & true_ok[usable])),
    }
    tp, tn = confusion["true_positive"], confusion["true_negative"]
    fp, fn = confusion["false_positive"], confusion["false_negative"]
    classification = {
        "accuracy": accuracy,
        "precision": tp / (tp + fp) if tp + fp else None,
        "recall": tp / (tp + fn) if tp + fn else None,
        "specificity": tn / (tn + fp) if tn + fp else None,
        "balanced_accuracy": 0.5 * (tp / (tp + fn) + tn / (tn + fp)) if tp + fn and tn + fp else None,
    }
    bands = {}
    for label, mask in {
        "support_ge_0_8": support >= 0.8,
        "support_0_5_to_0_8": (support >= 0.5) & (support < 0.8),
        "support_lt_0_5": support < 0.5,
    }.items():
        subset = usable & mask
        bands[label] = {
            "count": int(np.sum(subset)),
            "accuracy": float(np.mean(predicted_ok[subset] == true_ok[subset])) if np.any(subset) else None,
            "margin_mae": float(np.mean(np.abs(predicted_margin[subset] - truth["final"]["quality_margin"][subset]))) if np.any(subset) else None,
        }
    margin_bands = {}
    true_margin = np.asarray(truth["final"]["quality_margin"], dtype=float)
    for label, mask in {
        "near_boundary_abs_margin_lt_0_1": np.abs(true_margin) < 0.1,
        "clear_margin_abs_ge_0_1": np.abs(true_margin) >= 0.1,
        "clear_margin_abs_ge_0_2": np.abs(true_margin) >= 0.2,
    }.items():
        subset = usable & mask
        margin_bands[label] = {
            "count": int(np.sum(subset)),
            "accuracy": float(np.mean(predicted_ok[subset] == true_ok[subset])) if np.any(subset) else None,
            "margin_mae": float(np.mean(np.abs(predicted_margin[subset] - true_margin[subset]))) if np.any(subset) else None,
        }
    release_policies = {}
    for threshold in (0.0, 0.05, 0.1, 0.2):
        release = usable & (support >= 0.5) & (predicted_margin >= threshold)
        released = int(np.sum(release))
        release_policies[str(threshold)] = {
            "released_count": released,
            "oracle_good_fraction": float(np.mean(true_ok[release])) if released else None,
            "false_release_count": int(np.sum(release & ~true_ok)),
        }
    curing_tf = per_stage["curing"]["max_teacher_forced_nrmse"]
    curing_free = per_stage["curing"]["max_free_running_nrmse"]
    return {
        "holdout_count": len(usable), "physical_connection_acceptance_fraction": acceptance,
        "learned_chain_valid_fraction": float(usable.mean()), "minimum_support_quantiles": {
            str(q): float(np.quantile(support, q)) for q in (0.0, 0.1, 0.5, 0.9, 1.0)
        },
        "per_stage": per_stage, "final_feasibility_accuracy": accuracy, "classification": classification,
        "confusion": confusion, "support_bands": bands, "true_margin_bands": margin_bands,
        "release_policy_support_ge_0_5": release_policies,
        "upstream_error_amplification_max_nrmse": curing_free - curing_tf,
    }


def _scalar_prediction(config: dict, predictors: list, connections: dict[str, str], conditions: dict) -> tuple[dict, dict]:
    values = {name: float(conditions[name]) for name in CONTROL_AXES}
    values.update({
        "coating.incoming_viscosity_pa_s": conditions["material_viscosity_pa_s"],
        "coating.incoming_solids_fraction": conditions["material_solids_fraction"],
        "coating.incoming_bubble_fraction": conditions["material_bubble_fraction"],
    })
    means, support, valid = run_stage_chain(predictors, config, connections, values)
    final = {name: np.asarray(value).reshape(1) for name, value in means["curing"].items()}
    margin, feasible = margin_and_feasible(final, config["final_specifications"])
    return means, {"margin": float(margin[0]), "feasible": bool(feasible[0]), "support": float(support), "valid": bool(valid)}


def _constraint_probability(mean: float, std: float, direction: str, target: object) -> float:
    if std <= 1e-12:
        if direction == "greater_equal": return float(mean >= float(target))
        if direction == "less_equal": return float(mean <= float(target))
        low, high = target
        return float(low <= mean <= high)
    cdf = lambda z: 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))
    if direction == "greater_equal": return 1.0 - cdf((float(target) - mean) / std)
    if direction == "less_equal": return cdf((float(target) - mean) / std)
    low, high = target
    return max(0.0, cdf((high - mean) / std) - cdf((low - mean) / std))


def _local_risk(predictors: list, conditions: dict, means: dict) -> dict[str, object]:
    result = {}
    prior = {}
    for item, manifest, predictor in predictors:
        stage = item["id"]
        inputs = {name: conditions[name] for name in manifest["controls"]}
        if stage == "coating":
            inputs.update({"incoming_viscosity_pa_s": conditions["material_viscosity_pa_s"],
                           "incoming_solids_fraction": conditions["material_solids_fraction"],
                           "incoming_bubble_fraction": conditions["material_bubble_fraction"]})
        elif stage == "drying":
            inputs.update({"incoming_wet_thickness_um": means["coating"]["wet_thickness_um"],
                           "incoming_solids_fraction": means["coating"]["solids_fraction"],
                           "incoming_thickness_cv_fraction": means["coating"]["thickness_cv_fraction"],
                           "incoming_coating_defect_index": means["coating"]["coating_defect_index"]})
        else:
            inputs.update({"incoming_dry_thickness_um": means["drying"]["dry_thickness_um"],
                           "incoming_residual_solvent_pct": means["drying"]["residual_solvent_pct"],
                           "incoming_internal_stress_mpa": means["drying"]["internal_stress_mpa"],
                           "incoming_drying_defect_index": means["drying"]["drying_defect_index"],
                           "incoming_thickness_cv_fraction": means["drying"]["dry_thickness_cv_fraction"]})
        pred = predictor.predict_arrays(inputs)
        rows = []
        for constraint in manifest["local_constraints"]:
            output = pred["outputs"][constraint["name"]]
            p = _constraint_probability(float(output["mean"]), float(output["std"]), constraint["direction"], constraint["target"])
            rows.append({"name": constraint["name"], "satisfaction_probability": p})
        result[stage] = {"minimum_constraint_satisfaction": min((r["satisfaction_probability"] for r in rows), default=1.0), "constraints": rows}
    return result


def _candidate_values(conditions: dict, varying_stages: set[str], locked_controls: set[str] | None = None) -> dict[str, np.ndarray]:
    locked_controls = locked_controls or set()
    names = list(CONTROL_AXES)
    axes = []
    for name in names:
        stage = next(stage for stage, controls in STAGE_CONTROL_NAMES.items() if name in controls)
        if stage not in varying_stages or name in locked_controls:
            axes.append((float(conditions[name]),))
        else:
            axes.append(tuple(sorted(set(map(float, CONTROL_AXES[name])) | {float(conditions[name])})))
    points = np.asarray(list(itertools.product(*axes)), dtype=float)
    return {name: points[:, i] for i, name in enumerate(names)}


def _search(config: dict, predictors: list, connections: dict[str, str], conditions: dict,
            varying_stages: set[str], locked_controls: set[str] | None = None,
            minimal_change: bool = False) -> dict[str, object]:
    candidates = _candidate_values(conditions, varying_stages, locked_controls)
    values = dict(candidates)
    size = len(next(iter(candidates.values())))
    values.update({
        "coating.incoming_viscosity_pa_s": np.full(size, conditions["material_viscosity_pa_s"]),
        "coating.incoming_solids_fraction": np.full(size, conditions["material_solids_fraction"]),
        "coating.incoming_bubble_fraction": np.full(size, conditions["material_bubble_fraction"]),
    })
    outputs, support, valid = run_stage_chain(predictors, config, connections, values)
    margin, feasible = margin_and_feasible(outputs["curing"], config["final_specifications"])
    eligible = feasible & valid & (support >= 0.5)
    if not np.any(eligible):
        eligible = valid & np.isfinite(margin)
    indices = np.flatnonzero(eligible)
    if minimal_change and np.any(feasible & valid & (support >= 0.5)):
        indices = np.flatnonzero(feasible & valid & (support >= 0.5))
        changes = np.zeros(len(indices), dtype=int)
        distance = np.zeros(len(indices))
        for name, bounds in LINE_INPUT_BOUNDS.items():
            if name not in candidates: continue
            delta = np.abs(candidates[name][indices] - conditions[name])
            changes += delta > 1e-10
            distance += delta / (bounds[1] - bounds[0])
        order = np.lexsort((-margin[indices], -support[indices], distance, changes))
        index = int(indices[order[0]])
    else:
        index = int(indices[np.lexsort((-support[indices], -margin[indices]))[0]])
    selected = {name: float(value[index]) for name, value in candidates.items()}
    oracle_conditions = {**conditions, **selected}
    oracle = evaluate_line(oracle_conditions)["final"]
    adjustments = {name: selected[name] for name in selected if abs(selected[name] - conditions[name]) > 1e-10}
    return {
        "candidate_count": size, "predicted_margin": float(margin[index]),
        "predicted_feasible": bool(feasible[index] and valid[index]), "minimum_support": float(support[index]),
        "adjustments": adjustments,
        "oracle_margin": float(oracle["quality_margin"]), "oracle_feasible": bool(oracle["feasible"]),
    }


def _failed_specs(final: dict) -> list[str]:
    failed = []
    for name, (direction, target) in FINAL_SPECIFICATIONS.items():
        value = float(final[name])
        if direction == "greater_equal" and value < target: failed.append(name)
        elif direction == "less_equal" and value > target: failed.append(name)
        elif direction == "between" and not (target[0] <= value <= target[1]): failed.append(name)
    return failed


def _fault_diagnostics(config: dict, predictors: list, connections: dict[str, str]) -> dict[str, object]:
    cases = {}
    for case_name, definition in FAULT_SCENARIOS.items():
        conditions = {**REFERENCE_NOMINAL, **definition["overrides"]}
        means, current = _scalar_prediction(config, predictors, connections, conditions)
        oracle = evaluate_line(conditions)["final"]
        local = _local_risk(predictors, conditions, means)
        reset_rows = []
        stage_only = {}
        for stage in STAGE_CONTROL_NAMES:
            reset = dict(conditions)
            for control in STAGE_CONTROL_NAMES[stage]: reset[control] = REFERENCE_NOMINAL[control]
            _, reset_prediction = _scalar_prediction(config, predictors, connections, reset)
            reset_rows.append({"stage": stage, "predicted_margin_gain": reset_prediction["margin"] - current["margin"]})
            stage_only[stage] = _search(config, predictors, connections, conditions, {stage})
        reset_rows.sort(key=lambda row: row["predicted_margin_gain"], reverse=True)
        known_sources = set(definition["source_stages"])
        predicted_sources = {row["stage"] for row in reset_rows if row["predicted_margin_gain"] > 0.01}
        locked = set(definition["overrides"])
        cases[case_name] = {
            "oracle_current": {"feasible": bool(oracle["feasible"]), "margin": float(oracle["quality_margin"]), "failed_specs": _failed_specs(oracle)},
            "predicted_current": current, "local_constraint_risk": local,
            "diagnostic_ranking_by_reference_reset_gain": reset_rows,
            "diagnostic_score_only": {"known_sources": sorted(known_sources), "positive_gain_stages": sorted(predicted_sources),
                                      "source_recall": len(known_sources & predicted_sources) / len(known_sources)},
            "stage_only_recovery": stage_only,
            "minimal_change_all_stage_recovery": _search(config, predictors, connections, conditions, set(STAGE_CONTROL_NAMES), minimal_change=True),
            "best_margin_all_stage_recovery": _search(config, predictors, connections, conditions, set(STAGE_CONTROL_NAMES)),
            "compensation_with_faulty_controls_locked": _search(config, predictors, connections, conditions, set(STAGE_CONTROL_NAMES), locked_controls=locked, minimal_change=True),
        }
    return cases


def run() -> dict[str, object]:
    """固定trialの予測・故障診断・回復を採点し、REPORT_PATHへ結果を保存します。"""

    config, predictors, connections = _load_runtime()
    end_to_end = _end_to_end(config, predictors, connections)
    faults = _fault_diagnostics(config, predictors, connections)
    recovery_ok = all(case["minimal_change_all_stage_recovery"]["oracle_feasible"] for case in faults.values())
    report = {
        "status": "pass" if end_to_end["final_feasibility_accuracy"] >= 0.95 and recovery_ok else "fail",
        "interpretation": {
            "local_constraint_probability": "Conditional model diagnostic, not a root-cause probability.",
            "reference_reset_gain": "Counterfactual repair evidence using the expected recipe; known labels are hidden until scoring.",
            "uncertainty_limit": "Stage bundles have marginal predictive standard deviations but no joint sampling or inter-stage error correlation, so a calibrated end-to-end failure probability is not claimed.",
        },
        "end_to_end_holdout": end_to_end, "fault_cases": faults,
    }
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    return report


if __name__ == "__main__":
    result = run()
    print(json.dumps({"status": result["status"], "end_to_end": result["end_to_end_holdout"],
                      "fault_summary": {name: {"oracle_current": case["oracle_current"],
                                               "ranking": case["diagnostic_ranking_by_reference_reset_gain"],
                                               "minimal_recovery": case["minimal_change_all_stage_recovery"],
                                               "locked_recovery": case["compensation_with_faulty_controls_locked"]}
                                        for name, case in result["fault_cases"].items()}}, ensure_ascii=True, indent=2))
