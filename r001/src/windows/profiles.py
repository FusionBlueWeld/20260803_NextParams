"""中心条件の周囲を1変数ずつ走査し、連続した許容区間と境界理由を返します。

離れた合格区間と中心を含む窓は区別して記録します。
"""

from __future__ import annotations

import copy
import numpy as np
from ..pipeline import input_bounds
from ..windows.evaluation import evaluate_connected_window


def contiguous_true_interval(axis: np.ndarray, mask: np.ndarray, center: float) -> tuple[float | None, float | None]:
    """中心を含む連続したTrue区間の両端を返します。中心が不合格なら空窓です。"""

    nearest = int(np.argmin(np.abs(axis - center)))
    if not mask[nearest]:
        return None, None
    lower = upper = nearest
    while lower > 0 and mask[lower - 1]:
        lower -= 1
    while upper + 1 < len(mask) and mask[upper + 1]:
        upper += 1
    return float(axis[lower]), float(axis[upper])


def all_true_intervals(axis: np.ndarray, mask: np.ndarray) -> list[list[float]]:
    """走査軸上の全True区間を列挙し、中心から離れた許容区間も記録します。"""

    intervals: list[list[float]] = []
    start = None
    for index, passed in enumerate(np.asarray(mask, dtype=bool)):
        if passed and start is None:
            start = index
        if start is not None and (not passed or index == len(mask) - 1):
            end = index if passed and index == len(mask) - 1 else index - 1
            intervals.append([float(axis[start]), float(axis[end])])
            start = None
    return intervals


def boundary_diagnostic(evaluated: dict[str, object], index: int) -> dict[str, object]:
    """窓境界で制約となった工程・出力・支持度の情報を保存用にまとめます。"""

    bottleneck = str(np.asarray(evaluated["bottleneck"])[index])
    row: dict[str, object] = {
        "bottleneck": bottleneck,
        "margin": float(np.asarray(evaluated["process_margin"])[index]),
        "minimum_support": float(np.asarray(evaluated["minimum_support"])[index]),
        "stage_support": {stage: float(np.asarray(value)[index]) for stage, value in evaluated["stage_support"].items()},
    }
    head = bottleneck.split(":", 1)[0]
    parts = head.split(".")
    if parts[0] == "local" and len(parts) >= 3:
        stage, output = parts[1], ".".join(parts[2:])
    elif parts[0] == "final" and len(parts) >= 2:
        stage, output = next(reversed(evaluated["stage_outputs"])), ".".join(parts[1:])
    else:
        stage = output = None
    if stage is not None and output in evaluated["stage_outputs"][stage]:
        mean = float(np.asarray(evaluated["stage_outputs"][stage][output])[index])
        std = float(np.asarray(evaluated["stage_stds"][stage][output])[index])
        row.update({"output_mean": mean, "output_std": std})
    return row


def connected_window_profiles(
    predictors, config: dict[str, object], connections: dict[str, str], center: dict[str, float],
) -> dict[str, object]:
    """中心以外の操作を固定して各軸を走査し、許容窓と境界理由を返します。"""

    scan_points = int((config.get("connected_window") or {}).get("scan_points", 121))
    external = dict(config["external_context"])
    bounds = {}
    for _, manifest, predictor in predictors:
        stage_bounds = input_bounds(predictor)
        bounds.update({name: stage_bounds[name] for name in manifest["controls"]})
    profiles = {}
    mean_config = copy.deepcopy(config)
    mean_config["connected_window"] = {**(mean_config.get("connected_window") or {}),
                                       "interval_method": "std_multiplier", "confidence_z": 0.0,
                                       "support_threshold": 0.0}
    mean_config.pop("_connected_window_calibration", None)
    for name, (lower, upper) in bounds.items():
        axis = np.unique(np.append(np.linspace(lower, upper, scan_points), center[name]))
        values = {control: np.full(len(axis), value) for control, value in center.items()}
        values[name] = axis
        values.update({key: np.full(len(axis), value) for key, value in external.items()})
        mean_evaluated = evaluate_connected_window(predictors, mean_config, connections, values)
        evaluated = evaluate_connected_window(predictors, config, connections, values)
        mean_low, mean_high = contiguous_true_interval(axis, mean_evaluated["process_feasible"], center[name])
        process_low, process_high = contiguous_true_interval(axis, evaluated["process_feasible"], center[name])
        trust_low, trust_high = contiguous_true_interval(axis, evaluated["trusted_feasible"], center[name])
        span = upper - lower
        def boundary_rows(result, low, high):
            rows = {}
            for label, boundary in (("lower", low), ("upper", high)):
                if boundary is not None:
                    index = int(np.argmin(np.abs(axis - boundary)))
                    rows[label] = {"control_value": boundary, **boundary_diagnostic(result, index)}
            return rows
        profiles[name] = {
            "model_input_bounds": [lower, upper],
            "mean_only_window": [mean_low, mean_high],
            "buffered_process_window": [process_low, process_high],
            "process_window": [process_low, process_high],
            "trusted_window": [trust_low, trust_high],
            "mean_only_all_intervals": all_true_intervals(axis, mean_evaluated["process_feasible"]),
            "buffered_all_intervals": all_true_intervals(axis, evaluated["process_feasible"]),
            "trusted_all_intervals": all_true_intervals(axis, evaluated["trusted_feasible"]),
            "boundary_diagnostics": {
                "mean_only": boundary_rows(mean_evaluated, mean_low, mean_high),
                "buffered": boundary_rows(evaluated, process_low, process_high),
                "trusted": boundary_rows(evaluated, trust_low, trust_high),
            },
            "scan_evaluation_points": len(axis),
            "evaluation_unavailable_count": int(np.sum(~np.asarray(evaluated["valid_connection"], dtype=bool))),
            "lower_headroom": None if trust_low is None else center[name] - trust_low,
            "upper_headroom": None if trust_high is None else trust_high - center[name],
            "nearest_normalized_headroom": None if trust_low is None else min(center[name] - trust_low, trust_high - center[name]) / span,
        }
    scalar_values = {**center, **external}
    at_center = evaluate_connected_window(predictors, config, connections, scalar_values)
    return {
        "definition": "One-control-at-a-time connected feasible interval; interactions require joint scans.",
        "center": center, "center_process_margin": float(at_center["process_margin"]),
        "center_trusted_margin": float(at_center["trusted_margin"]),
        "center_minimum_support": float(at_center["minimum_support"]),
        "center_bottleneck": str(at_center["bottleneck"]), "controls": profiles,
    }

