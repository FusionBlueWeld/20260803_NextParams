"""oracleの校正用標本から、連結予測の残差区間を推定します。

校正と最終評価ではseedを分けます。校正ファイルは予測器ハッシュなどの
適用契約に結び付け、異なるモデルへ流用しません。"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import numpy as np

from validation.multistage.functional_coating.pipeline import LINE_INPUT_BOUNDS, evaluate_line
from validation.multistage.src.checks.r001_diagnostic_validation import _lhs, _load_runtime
from r001.src.cli import _run_connected_window
from r001.src.window_calibration import calibration_contract


from ..settings import PROJECT_ROOT as ROOT
TRIAL = ROOT / "r001/trials/trial_validated_individual_chain_r001"
OUTPUT = TRIAL / "connected_window_calibration.json"


def _dataset(config: dict, samples: int, seed: int) -> tuple[dict, dict, dict]:
    control_bounds = {name: LINE_INPUT_BOUNDS[name] for name in config["candidate_axes"]}
    conditions = _lhs(control_bounds, samples, seed)
    conditions.update({
        "material_viscosity_pa_s": np.full(samples, config["external_context"]["coating.incoming_viscosity_pa_s"]),
        "material_solids_fraction": np.full(samples, config["external_context"]["coating.incoming_solids_fraction"]),
        "material_bubble_fraction": np.full(samples, config["external_context"]["coating.incoming_bubble_fraction"]),
    })
    truth = evaluate_line(conditions)
    values = {
        **{name: conditions[name] for name in config["candidate_axes"]},
        "coating.incoming_viscosity_pa_s": conditions["material_viscosity_pa_s"],
        "coating.incoming_solids_fraction": conditions["material_solids_fraction"],
        "coating.incoming_bubble_fraction": conditions["material_bubble_fraction"],
    }
    return conditions, truth, values


def _fit_offsets(predictors: list, truth: dict, predicted: dict, alpha: float) -> tuple[dict, dict]:
    """支持された同一標本で残差分位点を求めます。真値・予測の元配列は変更しません。"""

    offsets = {}
    sample_counts = {}
    for item, manifest, _ in predictors:
        stage = str(item["id"])
        # Use the same fully supported chain population for every output/alpha.
        # Never mutate predicted['valid_connection'] through a numpy view.
        mask = np.asarray(predicted["valid_connection"], dtype=bool).copy()
        for support in predicted["stage_support"].values():
            mask &= np.asarray(support) >= 0.5
        offsets[stage] = {}
        sample_counts[stage] = int(mask.sum())
        if sample_counts[stage] < 100:
            raise RuntimeError(f"Insufficient supported calibration rows for {stage}")
        for output in manifest["predicted_outputs"]:
            error = np.asarray(truth[stage][output], dtype=float)[mask] - np.asarray(predicted["stage_outputs"][stage][output], dtype=float)[mask]
            std = np.asarray(predicted["stage_stds"][stage][output], dtype=float)[mask]
            std_floor = max(float(np.median(std)) * 0.05, 1e-9)
            standardized = error / np.maximum(std, std_floor)
            offsets[stage][output] = {
                "lower": float(np.quantile(error, alpha, method="higher")),
                "upper": float(np.quantile(error, 1 - alpha, method="higher")),
                "median": float(np.median(error)),
                "rmse": float(np.sqrt(np.mean(error ** 2))),
                "std_floor": std_floor,
                "lower_z": max(0.0, float(-np.quantile(standardized, alpha, method="higher"))),
                "upper_z": max(0.0, float(np.quantile(standardized, 1 - alpha, method="higher"))),
            }
    return offsets, sample_counts


def _truth_feasible(predictors: list, config: dict, truth: dict) -> np.ndarray:
    size = len(next(iter(truth["coating"].values())))
    feasible = np.ones(size, dtype=bool)
    for item, manifest, _ in predictors:
        stage = str(item["id"])
        for constraint in manifest.get("local_constraints", []):
            value = np.asarray(truth[stage][constraint["name"]])
            if constraint["direction"] == "greater_equal":
                feasible &= value >= float(constraint["target"])
            else:
                feasible &= value <= float(constraint["target"])
    feasible &= np.asarray(truth["final"]["feasible"], dtype=bool)
    return feasible


def run(samples: int = 4096, seed: int = 20260921) -> dict[str, object]:
    """固定trialで校正・方式比較を行い、OUTPUTの校正ファイルを更新します。"""

    config, predictors, connections = _load_runtime()
    prediction_config = copy.deepcopy(config)
    prediction_config["connected_window"] = {
        **(prediction_config.get("connected_window") or {}),
        "interval_method": "std_multiplier", "confidence_z": 0.0, "support_threshold": 0.0,
    }
    prediction_config.pop("_connected_window_calibration", None)
    _, truth, values = _dataset(config, samples, seed)
    predicted = _run_connected_window(predictors, prediction_config, connections, values)
    _, tuning_truth, tuning_values = _dataset(config, samples, seed + 1)
    tuning_true = _truth_feasible(predictors, config, tuning_truth)
    comparisons = []
    fitted = {}
    sample_counts = {}
    for alpha in (0.05, 0.075, 0.10, 0.125, 0.15, 0.20):
        offsets, counts = _fit_offsets(predictors, truth, predicted, alpha)
        for method in ("residual_quantile", "standardized_residual_quantile"):
            candidate_config = copy.deepcopy(config)
            candidate_config["connected_window"] = {
                **candidate_config["connected_window"], "interval_method": method,
            }
            candidate_config["_connected_window_calibration"] = {"offsets": offsets}
            tuning_prediction = _run_connected_window(predictors, candidate_config, connections, tuning_values)
            selected = np.asarray(tuning_prediction["trusted_feasible"], dtype=bool)
            tp = int(np.sum(selected & tuning_true))
            fp = int(np.sum(selected & ~tuning_true))
            fn = int(np.sum(~selected & tuning_true))
            comparisons.append({
                "interval_method": method,
                "alpha": alpha, "true_positive": tp, "false_positive": fp, "false_negative": fn,
                "recall": tp / max(tp + fn, 1), "selected_count": int(selected.sum()),
            })
        fitted[alpha] = offsets
        sample_counts = counts
    zero_false_positive = [row for row in comparisons if row["false_positive"] == 0]
    configured_method = config["connected_window"].get("interval_method", "standardized_residual_quantile")
    configured_pool = [row for row in zero_false_positive if row["interval_method"] == configured_method]
    if not configured_pool:
        raise RuntimeError("No calibration method meets zero false positives; no artifact was published.")
    # Compare both methods, but do not silently change the deployed pipeline's method.
    chosen = max(configured_pool, key=lambda row: (row["true_positive"], row["alpha"]))
    best_comparison = max(zero_false_positive, key=lambda row: (row["true_positive"], row["alpha"]))
    alpha = float(chosen["alpha"])
    offsets = fitted[alpha]
    signatures = {
        str(item["id"]): hashlib.sha256(Path(item["bundle"]).joinpath("predictor.pkl").read_bytes()).hexdigest()
        for item in config["stages"]
    }
    result = {
        "schema_version": "1.1", "method": "connected_free_running_empirical_residual_quantiles",
        "selected_interval_method": chosen["interval_method"],
        "best_tuning_comparison": best_comparison,
        "contract": calibration_contract(config, [
            (item, Path(item["bundle"]), manifest) for item, manifest, _ in predictors
        ]),
        "alpha_per_output_tail": alpha, "calibration_seed": seed, "tuning_seed": seed + 1,
        "requested_samples_per_split": samples,
        "physical_connection_acceptance_fraction": 1.0,
        "calibration_scope": {
            "controls": "latin_hypercube_over_full_model_bounds",
            "external_context": config["external_context"],
        },
        "supported_sample_counts": sample_counts, "tuning_comparisons": comparisons,
        "selection_rule": "maximum true positives among zero-false-positive tuning candidates of the explicitly configured method",
        "bundle_signatures": signatures,
        "offsets": offsets,
        "data_use": "calibration and tuning splits select the interval; final boundary validation uses a separate deterministic grid",
        "limitations": [
            "Marginal output intervals do not provide a joint probability guarantee.",
            "Offsets are valid only for this exact bundle combination and supported calibration domain.",
        ],
    }
    OUTPUT.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


if __name__ == "__main__":
    value = run()
    print(json.dumps({key: value[key] for key in value if key != "offsets"}, ensure_ascii=True, indent=2))
