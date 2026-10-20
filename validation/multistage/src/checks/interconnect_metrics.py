"""探索結果を学習に使っていない物理oracleで採点します。"""

from __future__ import annotations

import copy

import numpy as np

from .interconnect_data import local_margin, physical_contract, sample_controls, stage_inputs


class OraclePredictor:
    """配線・探索方式の基準専用。実運用bundleとして保存しません。"""

    def __init__(self, stage: str, evaluate):
        self.stage, self.evaluate = stage, evaluate

    def predict_arrays(self, inputs: dict) -> dict:
        outputs = self.evaluate(self.stage, inputs)
        first = np.asarray(next(iter(outputs.values())))
        return {"outputs": {name: {"mean": np.asarray(values), "std": np.zeros_like(values)}
                            for name, values in outputs.items()},
                "support": np.ones(first.shape), "valid": np.ones(first.shape, dtype=bool)}


def oracle_predictors() -> dict:
    """v005と同じ契約に包み、予測器と正解評価の経路を分離します。"""
    _, manifests, evaluate = physical_contract()
    result = {}
    for stage, original in manifests.items():
        manifest = copy.deepcopy(original)
        manifest.update(bundle_schema_version="2.0", capabilities={"mean_prediction": True,
                        "deterministic_oracle": True, "joint_sampling": False,
                        "calibrated_process_probability": False})
        result[stage] = (manifest, OraclePredictor(stage, evaluate))
    return result


def score_controls(config: dict, controls: dict, predictors: dict | None = None) -> dict:
    """予測が合格でも正解評価が不合格なら、そのままfalse passとして記録します。"""
    from r002.src.pipeline import evaluate_pipeline
    result = evaluate_pipeline(config, oracle_predictors() if predictors is None else predictors, controls)
    return {"oracle_feasible": bool(result["trusted_feasible"][0]),
            "oracle_margin": float(np.nan_to_num(result["process_margin"][0], nan=-1e6)),
            "oracle_bottleneck": str(result["bottleneck"][0]),
            "valid_connection": bool(result["valid_connection"][0])}


def random_reference(config: dict, budget: int, seed: int, predictors: dict | None = None) -> dict:
    """同じ予測器・ライン評価件数で選択し、別経路の物理正解で採点します。"""
    from r002.src.pipeline import evaluate_pipeline
    controls = sample_controls(config["candidate_axes"], budget, np.random.default_rng(seed))
    result = evaluate_pipeline(config, oracle_predictors() if predictors is None else predictors, controls)
    margins = np.asarray(result["trusted_margin"])
    index = int(np.argmax(margins))
    selected = {name: float(value[index]) for name, value in controls.items()}
    return {"evaluation_count": budget, "model_prediction_rows": result.get("model_prediction_rows"),
            "selection_model": "oracle" if predictors is None else "learned_v005",
            "best": {"controls": selected},
            **score_controls(config, selected)}


def individual_greedy(config: dict, budget: int, seed: int) -> dict:
    """各工程で局所Bestを一つ選び、完成したワークだけ次工程へ送ります。

    最終品質の正解を見ながら選ばないので、局所最適が最終最適と異なることを
    比較できます。上流がNGなら下流を架空の条件で評価せず停止します。
    """
    _, manifests, evaluate = physical_contract()
    rng = np.random.default_rng(seed)
    outputs, selected = {}, {}
    evaluations = 0
    per_stage = max(1, budget // len(manifests))
    for stage, manifest in manifests.items():
        axes = {name: axis for name, axis in config["candidate_axes"].items() if name.startswith(stage + ".")}
        controls = sample_controls(axes, per_stage, rng)
        inputs = stage_inputs(stage, manifest, config, controls, outputs)
        inputs = {name: np.broadcast_to(np.asarray(value, dtype=float), (per_stage,)) for name, value in inputs.items()}
        for name, values in inputs.items():
            lo, hi = manifest["input_bounds"][name]
            if not np.all(np.isfinite(values) & (values >= lo) & (values <= hi)):
                return {"status": "UPSTREAM_STATE_OUTSIDE_DOMAIN", "evaluation_count": evaluations,
                        "oracle_feasible": False, "best": None}
        measured = evaluate(stage, inputs)
        evaluations += per_stage
        margins = local_margin(measured, manifest)
        objective = manifest["local_objective"]
        utility = np.asarray(measured[objective["name"]], dtype=float)
        if objective["direction"] in ("minimize", "less_equal"):
            utility = -utility
        good = margins >= 0
        index = int(np.argmax(np.where(good, utility, -np.inf))) if good.any() else int(np.argmax(margins))
        selected.update({name: float(values[index]) for name, values in controls.items()})
        outputs[stage] = {name: float(np.asarray(value)[index]) for name, value in measured.items()}
        if not good.any():
            return {"status": "LOCAL_CONSTRAINT_FAILED", "evaluation_count": evaluations,
                    "oracle_feasible": False, "best": None, "stopped_stage": stage}
    return {"status": "COMPLETED", "evaluation_count": evaluations, "single_stage_evaluation_rows": evaluations,
            "budget_comparability": "local_oracle_baseline_not_equal_full_line_evaluation_budget",
            "best": {"controls": selected}, **score_controls(config, selected)}


def prediction_errors(holdout: dict, predictors: dict) -> dict:
    """系譜を保った別ロット上で、工程ごとの予測誤差と支持度を測ります。"""
    metrics = {}
    for stage, (manifest, predictor) in predictors.items():
        block = holdout["blocks"][stage]
        if not block["admitted_count"]:
            metrics[stage] = {"count": 0, "status": "NO_ACHIEVABLE_HOLDOUT"}
            continue
        prediction = predictor.predict_arrays(block["inputs"])
        errors = {}
        for name, expected in block["outputs"].items():
            mean = np.asarray(prediction["outputs"][name]["mean"])
            errors[name] = {"rmse": float(np.sqrt(np.mean((mean - expected) ** 2))),
                            "unit": manifest.get("output_units", {}).get(name, "")}
        metrics[stage] = {"count": block["admitted_count"], "outputs": errors,
                          "mean_support": float(np.mean(prediction["support"])),
                          "observation_regime": "PRIVILEGED_SYNTHETIC_LABELS"}
    return metrics


def compact_search(result: dict, config: dict) -> dict:
    """大きな候補履歴をレポートへ重複保存せず、根拠と実際の品質を残します。"""
    summary = {name: result[name] for name in ("status", "evaluation_count", "elapsed_seconds",
               "stop_reason", "full_grid_candidate_count", "free_control_count")}
    summary["best"] = result["best"]
    summary["model_prediction_rows"] = result.get("model_prediction_rows")
    summary["uncertainty_policy"] = result.get("uncertainty_policy", "mean_prediction_only")
    if result["best"]:
        summary.update(score_controls(config, result["best"]["controls"]))
        summary["false_pass"] = bool(result["best"]["predicted_feasible"] and not summary["oracle_feasible"])
    return summary
