"""r001の実行手順。入力検査→予測器読込→候補評価→順位付け→結果保存を制御します。

数値計算はpipeline・windows・process_variationへ渡し、指定された評価だけを実行します。
"""

from __future__ import annotations

import itertools
import numpy as np
from .data_loader import write_csv, write_json
from .validation import UserInputError, load_pipeline_config, validate_pipeline
from .trials import trial_path
from .bundle_runtime import load_bundle_runtime
from .pipeline import individual_best_chain, margin_and_feasible
from .windows.evaluation import evaluate_connected_window
from .windows.profiles import connected_window_profiles
from .windows.selection import compare_window_centers
from .windows.simultaneous import evaluate_simultaneous_window
from .process_variation import process_variation_rows


def run_trial(name: str, recommendation_count: int) -> None:
    """入力を検査して全候補を評価し、有効化された追加評価と結果保存を順に行います。"""

    # 1. 設定と接続契約を検査してから、候補と予測器を用意します。
    trial = trial_path(name)
    config = load_pipeline_config(trial)
    stages = validate_pipeline(trial, config)
    axis_names = list(config["candidate_axes"])
    points = np.asarray(list(itertools.product(*(config["candidate_axes"][key] for key in axis_names))), dtype=float)
    if len(points) > 200_000:
        raise UserInputError("全体候補が200,000件を超えます。")
    values = {name: points[:, index] for index, name in enumerate(axis_names)}
    load_bundle = load_bundle_runtime()
    predictors = [(item, manifest, load_bundle(path)[1]) for item, path, manifest in stages]
    connections = {str(row["to"]): str(row["from"]) for row in config.get("connections", [])}
    # 2. 個別Best連結を比較基準とし、同じ候補を全体の制約で採点します。
    individual_chain = individual_best_chain(predictors, config, connections)
    window_result = evaluate_connected_window(predictors, config, connections, values)
    stage_outputs = window_result["stage_outputs"]
    support = window_result["minimum_support"]
    final_outputs = stage_outputs[str(stages[-1][0]["id"])]
    final_margin, final_feasible = margin_and_feasible(final_outputs, config.get("final_specifications", []))
    use_window = bool((config.get("connected_window") or {}).get("use_for_selection", False))
    ranking_margin = window_result["trusted_margin"] if use_window else final_margin
    feasible = window_result["trusted_feasible"] if use_window else final_feasible
    # 3. 余裕を優先し、同率なら支持度で順位付けします。窓中心探索は任意です。
    order = np.lexsort((-support, -ranking_margin))
    baseline_selected_index = int(order[0])
    center_comparisons = None
    center_setting = config.get("window_center_selection") or {}
    if center_setting.get("enabled"):
        center_comparisons = compare_window_centers(
            predictors, config, connections, points, axis_names,
            np.asarray(window_result["trusted_feasible"], dtype=bool), order,
        )
        if center_comparisons:
            selected_index = int(center_comparisons[0]["candidate_index"])
            order = np.concatenate(([selected_index], order[order != selected_index]))
    rows = []
    for rank, index in enumerate(order[:min(recommendation_count, len(order))], start=1):
        row: dict[str, object] = {
            "rank": rank, "predicted_feasible": bool(feasible[index]),
            "predicted_quality_margin": float(final_margin[index]),
            "connected_window_margin": float(window_result["trusted_margin"][index]),
            "connected_process_margin": float(window_result["process_margin"][index]),
            "connected_window_bottleneck": str(window_result["bottleneck"][index]),
            "minimum_stage_support": float(support[index]),
            "good_product_probability": "NOT_EVALUATED",
        }
        row.update({key: float(values[key][index]) for key in axis_names})
        row.update({f"final_{key}_mean": float(value[index]) for key, value in final_outputs.items()})
        rows.append(row)
    # 4. 順位と追加評価を保存します。未設定の良品確率は未評価のまま残します。
    output = trial / "output"
    write_csv(output / "system_recommendations.csv", rows)
    if center_comparisons is not None:
        flattened = []
        for rank, item in enumerate(center_comparisons, start=1):
            row = {key: value for key, value in item.items() if key != "normalized_headrooms"}
            row["window_rank"] = rank
            row.update({f"headroom_ratio_{name}": value for name, value in item["normalized_headrooms"].items()})
            flattened.append(row)
        write_csv(output / "window_center_candidates.csv", flattened)
    robust_rows = None
    if config.get("process_variation") is not None:
        robust_rows = process_variation_rows(predictors, config, connections, axis_names, values, order, final_margin)
        write_csv(output / "robust_recommendations.csv", robust_rows)
    margin_improvement = (
        float(final_margin[order[0]] - individual_chain["predicted_quality_margin"])
        if individual_chain["status"] == "AVAILABLE" else None
    )
    summary = {
        "schema_version": "1.0", "candidate_count": len(points),
        "predicted_feasible_count": int(np.sum(feasible)),
        "best_predicted_quality_margin": float(final_margin[order[0]]),
        "probability_status": "UNCALIBRATED_PROCESS_VARIATION_ESTIMATE" if robust_rows is not None else "NOT_EVALUATED",
        "probability_reason": (
            "設定した独立正規工程ばらつきを平均予測モデルへ伝播。Wilson区間はMonte Carlo誤差だけで、モデル誤差は含まない。"
            if robust_rows is not None else "stage bundleが同時サンプリングと校正済み工程確率に未対応"
        ),
        "individual_best_chain": individual_chain,
        "quality_margin_improvement_over_individual_chain": margin_improvement,
        "connected_window": {
            "enabled_for_selection": use_window,
            "interval_method": str((config.get("connected_window") or {}).get("interval_method", "std_multiplier")),
            "support_threshold": float((config.get("connected_window") or {}).get("support_threshold", 0.0)),
            "confidence_z": float((config.get("connected_window") or {}).get("confidence_z", 0.0)),
            "enforce_local_constraints": bool((config.get("connected_window") or {}).get("enforce_local_constraints", False)),
            "feasible_count": int(np.sum(window_result["trusted_feasible"])),
            "selected_margin": float(window_result["trusted_margin"][order[0]]),
            "selected_process_margin": float(window_result["process_margin"][order[0]]),
            "selected_bottleneck": str(window_result["bottleneck"][order[0]]),
        },
        "selected": rows[0],
    }
    if robust_rows is not None:
        summary["process_variation"] = {
            "distribution": "independent_clipped_normal",
            "samples": int(config["process_variation"]["samples"]),
            "seed": int(config["process_variation"]["seed"]),
            "candidate_limit": int(config["process_variation"]["candidate_limit"]),
            "selection_metric": "mc95_low_then_probability_then_deterministic_margin",
            "selected": robust_rows[0],
        }
    if center_comparisons is not None:
        baseline_comparison = next(
            (item for item in center_comparisons if item["candidate_index"] == baseline_selected_index), None
        )
        summary["window_center_selection"] = {
            "status": "EVALUATED" if center_comparisons else "NO_TRUSTED_CANDIDATES",
            "selection_metric": "symmetric_headroom_rho_then_connected_margin_then_support",
            "evaluated_candidates": len(center_comparisons),
            "eligible_candidates": int(np.sum(window_result["trusted_feasible"])),
            "required_variations": center_setting["required_variations"],
            "selected": center_comparisons[0] if center_comparisons else None,
            "baseline_connected_margin_selection": baseline_comparison,
            "rho_improvement_over_baseline": (
                center_comparisons[0]["symmetric_headroom_rho"] - baseline_comparison["symmetric_headroom_rho"]
                if center_comparisons and baseline_comparison is not None else None
            ),
            "claim": "budgeted_diverse_candidate_comparison_not_global_optimum",
        }
    profiles = None
    if config.get("connected_window") is not None:
        center = {name: float(values[name][order[0]]) for name in axis_names}
        profiles = connected_window_profiles(predictors, config, connections, center)
        write_json(output / "connected_window.json", profiles)
    simultaneous_setting = config.get("simultaneous_window") or {}
    if simultaneous_setting.get("enabled"):
        if not center_comparisons:
            raise UserInputError("simultaneous_windowには有効なwindow_center_selection結果が必要です。")
        selected_window = center_comparisons[0]
        simultaneous = evaluate_simultaneous_window(
            predictors, config, connections, center,
            float(selected_window["symmetric_headroom_rho"]), selected_window["normalized_headrooms"],
        )
        write_json(output / "simultaneous_window.json", simultaneous)
        summary["simultaneous_window"] = {
            "status": "EVALUATED", "rho": simultaneous["rho"],
            "random_uniform_boxes": simultaneous["random_uniform_boxes"],
            "rho_box_vertices": simultaneous["rho_box_vertices"],
            "pairwise_sections": simultaneous["pairwise_sections"],
            "claim": "finite_sample_joint_evaluation_not_exact_volume",
        }
    write_json(output / "run_summary.json", summary)
    print(f"{name}の全体条件を{len(points):,}候補から評価しました。")
    print(f"予測適合候補: {summary['predicted_feasible_count']:,}件")
    if margin_improvement is None:
        print(f"個別Best連結: 未評価（{individual_chain['reason']}）")
    else:
        print(f"個別Best連結からの仕様余裕改善: {margin_improvement:.6g}")
    print(f"出力: {output / 'system_recommendations.csv'}")
    if robust_rows is None:
        print("良品確率: 未評価（平均予測bundleのため）")
    else:
        print(f"工程ばらつき込み未校正予測良品率: {robust_rows[0]['predicted_good_probability']:.3%}")
        print(f"確率最適化出力: {output / 'robust_recommendations.csv'}")

