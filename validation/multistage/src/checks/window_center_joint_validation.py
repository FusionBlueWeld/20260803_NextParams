"""窓中心の選択と複数変数の同時変更を、独立した物理oracleで検証します。

一変数断面の幅と、同時変更した箱内の有限サンプルの合否は分けて評価します。"""

from __future__ import annotations

import csv
import importlib
import itertools
import json
from pathlib import Path

import numpy as np

from validation.multistage.src.checks.calibrate_connected_window import _truth_feasible
from validation.multistage.src.checks.connected_window_validation import _oracle_process_feasible
from validation.multistage.functional_coating.pipeline import LINE_INPUT_BOUNDS, evaluate_line
from r001.src.cli import (
    _contiguous_true_interval, _load_bundle_runtime, _read_config, _run_connected_window, _validate,
)


from ..settings import PROJECT_ROOT as ROOT
TRIAL = ROOT / "r001/trials/trial_validated_individual_chain_r001"
REPORT = ROOT / "validation/multistage/results/window_center_joint_validation.json"


def _load():
    config = _read_config(TRIAL)
    stages = _validate(TRIAL, config)
    load_bundle = _load_bundle_runtime()
    predictors = [(item, manifest, load_bundle(path)[1]) for item, path, manifest in stages]
    connections = {row["to"]: row["from"] for row in config["connections"]}
    return config, predictors, connections


def _values(config: dict, controls: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    size = len(next(iter(controls.values())))
    return {**controls, **{name: np.full(size, value) for name, value in config["external_context"].items()}}


def _oracle_truth(config: dict, controls: dict[str, np.ndarray]) -> tuple[dict, np.ndarray]:
    conditions = dict(controls)
    size = len(next(iter(controls.values())))
    conditions.update({
        "material_viscosity_pa_s": np.full(size, config["external_context"]["coating.incoming_viscosity_pa_s"]),
        "material_solids_fraction": np.full(size, config["external_context"]["coating.incoming_solids_fraction"]),
        "material_bubble_fraction": np.full(size, config["external_context"]["coating.incoming_bubble_fraction"]),
    })
    truth = evaluate_line(conditions)
    return truth, conditions


def _compare(config, predictors, connections, controls) -> dict[str, object]:
    predicted = np.asarray(_run_connected_window(predictors, config, connections, _values(config, controls))["trusted_feasible"], dtype=bool)
    truth, _ = _oracle_truth(config, controls)
    actual = _truth_feasible(predictors, config, truth)
    return {
        "count": len(actual), "predicted_pass_count": int(predicted.sum()), "oracle_pass_count": int(actual.sum()),
        "true_positive": int(np.sum(predicted & actual)), "false_positive": int(np.sum(predicted & ~actual)),
        "false_negative": int(np.sum(~predicted & actual)), "true_negative": int(np.sum(~predicted & ~actual)),
    }


def _oracle_rho(center: dict[str, float], required: dict[str, float], scan_points: int = 241) -> dict[str, object]:
    ratios = {}
    windows = {}
    for name, (lower, upper) in ((name, LINE_INPUT_BOUNDS[name]) for name in required):
        axis = np.unique(np.append(np.linspace(lower, upper, scan_points), center[name]))
        mask = np.asarray([_oracle_process_feasible({**center, name: float(value)})[0] for value in axis])
        low, high = _contiguous_true_interval(axis, mask, center[name])
        windows[name] = [low, high]
        ratios[name] = None if low is None else min(center[name] - low, high - center[name]) / required[name]
    return {"rho": min(value for value in ratios.values() if value is not None), "ratios": ratios, "windows": windows}


def run() -> dict[str, object]:
    """固定trialの中心選択・同時変更結果を読み、oracle比較をREPORTへ保存します。"""

    config, predictors, connections = _load()
    summary = json.loads((TRIAL / "output/run_summary.json").read_text(encoding="utf-8"))
    simultaneous = json.loads((TRIAL / "output/simultaneous_window.json").read_text(encoding="utf-8"))
    selection = summary["window_center_selection"]
    names = list(config["candidate_axes"])
    selected = {name: float(selection["selected"][name]) for name in names}
    baseline = {name: float(selection["baseline_connected_margin_selection"][name]) for name in names}
    required = {name: float(value) for name, value in selection["required_variations"].items()}
    selected_oracle = _oracle_rho({**{
        "material_viscosity_pa_s": config["external_context"]["coating.incoming_viscosity_pa_s"],
        "material_solids_fraction": config["external_context"]["coating.incoming_solids_fraction"],
        "material_bubble_fraction": config["external_context"]["coating.incoming_bubble_fraction"],
    }, **selected}, required)
    baseline_oracle = _oracle_rho({**{
        "material_viscosity_pa_s": config["external_context"]["coating.incoming_viscosity_pa_s"],
        "material_solids_fraction": config["external_context"]["coating.incoming_solids_fraction"],
        "material_bubble_fraction": config["external_context"]["coating.incoming_bubble_fraction"],
    }, **baseline}, required)

    multi_center = []
    with (TRIAL / "output/window_center_candidates.csv").open(encoding="utf-8-sig", newline="") as file:
        candidate_rows = list(csv.DictReader(file))[:8]
    external = {
        "material_viscosity_pa_s": config["external_context"]["coating.incoming_viscosity_pa_s"],
        "material_solids_fraction": config["external_context"]["coating.incoming_solids_fraction"],
        "material_bubble_fraction": config["external_context"]["coating.incoming_bubble_fraction"],
    }
    for row in candidate_rows:
        controls = {name: float(row[name]) for name in names}
        oracle = _oracle_rho({**external, **controls}, required)
        multi_center.append({
            "window_rank": int(row["window_rank"]), "predicted_rho": float(row["symmetric_headroom_rho"]),
            "oracle_rho": oracle["rho"], "controls": controls,
        })

    rng = np.random.default_rng(int(simultaneous["random_seed"]))
    random_rows = []
    samples = int(config["simultaneous_window"]["samples"])
    rho = float(simultaneous["rho"])
    for scale in map(float, config["simultaneous_window"]["radius_scales"]):
        controls = {name: np.clip(selected[name] + rng.uniform(-1, 1, samples) * rho * scale * required[name], *LINE_INPUT_BOUNDS[name]) for name in names}
        random_rows.append({"radius_scale": scale, **_compare(config, predictors, connections, controls)})
    signs = np.asarray(list(itertools.product((-1.0, 1.0), repeat=len(names))))
    vertices = _compare(config, predictors, connections, {
        name: np.clip(selected[name] + signs[:, column] * rho * required[name], *LINE_INPUT_BOUNDS[name])
        for column, name in enumerate(names)
    })
    pair_rows = []
    unit = np.linspace(-1, 1, int(config["simultaneous_window"]["pair_grid_points"]))
    a, b = np.meshgrid(unit, unit, indexing="ij")
    for row in simultaneous["pairwise_sections"]:
        first, second = row["controls"]
        controls = {name: np.full(a.size, value) for name, value in selected.items()}
        controls[first] = np.clip(selected[first] + a.reshape(-1) * rho * required[first], *LINE_INPUT_BOUNDS[first])
        controls[second] = np.clip(selected[second] + b.reshape(-1) * rho * required[second], *LINE_INPUT_BOUNDS[second])
        pair_rows.append({"controls": [first, second], **_compare(config, predictors, connections, controls)})
    all_comparisons = [*random_rows, vertices, *pair_rows]
    report = {
        "status": "pass" if selected_oracle["rho"] > baseline_oracle["rho"] and sum(row["false_positive"] for row in all_comparisons) == 0 else "fail",
        "scope": "Independent deterministic oracle evaluation; no real-world calibration claim.",
        "selected_predicted_rho": selection["selected"]["symmetric_headroom_rho"],
        "baseline_predicted_rho": selection["baseline_connected_margin_selection"]["symmetric_headroom_rho"],
        "selected_oracle": selected_oracle, "baseline_oracle": baseline_oracle,
        "oracle_rho_improvement": selected_oracle["rho"] - baseline_oracle["rho"],
        "top_center_oracle_comparison": multi_center,
        "joint_random_boxes": random_rows, "joint_vertices": vertices, "pairwise_sections": pair_rows,
        "limitations": ["One-axis oracle rho is a cross-sectional metric.", "Joint results cover only the recorded finite samples."],
    }
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    value = run()
    print(json.dumps({key: value[key] for key in value if key not in {"selected_oracle", "baseline_oracle"}}, ensure_ascii=True, indent=2))
