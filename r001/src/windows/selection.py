"""有限の評価予算で多様な候補を選び、必要変動幅に対する対称余裕を比較します。

rhoは各操作軸の余裕比の最小値です。全空間の大域最適性は主張しません。
"""

from __future__ import annotations

import copy
import numpy as np
from ..pipeline import input_bounds
from ..windows.evaluation import evaluate_connected_window
from ..windows.profiles import contiguous_true_interval


def diverse_candidate_indices(points: np.ndarray, eligible: np.ndarray, order: np.ndarray, limit: int) -> np.ndarray:
    """順位を尊重しながら候補の偏りを抑え、評価予算内の候補を選びます。"""

    available = np.flatnonzero(eligible)
    if len(available) <= limit:
        return available
    top_count = max(1, limit // 2)
    chosen = list(np.asarray([index for index in order if eligible[index]][:top_count], dtype=int))
    normalized = (points - points.min(axis=0)) / np.maximum(np.ptp(points, axis=0), 1e-12)
    distance = np.min([np.sum((normalized - normalized[index]) ** 2, axis=1) for index in chosen], axis=0)
    distance[~eligible] = -1
    distance[chosen] = -1
    while len(chosen) < limit:
        index = int(np.argmax(distance))
        chosen.append(index)
        distance = np.minimum(distance, np.sum((normalized - normalized[index]) ** 2, axis=1))
        distance[~eligible] = -1
        distance[chosen] = -1
    return np.asarray(chosen, dtype=int)


def compare_window_centers(
    predictors, config: dict[str, object], connections: dict[str, str], points: np.ndarray,
    axis_names: list[str], eligible: np.ndarray, base_order: np.ndarray,
) -> list[dict[str, object]]:
    """選んだ候補の対称余裕rhoを比較し、余裕・支持度を使って順位を決めます。"""

    setting = config["window_center_selection"]
    indices = diverse_candidate_indices(points, eligible, base_order, int(setting["candidate_limit"]))
    scan_config = copy.deepcopy(config)
    scan_config["connected_window"]["scan_points"] = int(setting["coarse_scan_points"])
    required = {name: float(value) for name, value in setting["required_variations"].items()}
    rows = []
    for index in indices:
        center = {name: float(points[index, column]) for column, name in enumerate(axis_names)}
        profile = batched_window_headrooms(predictors, scan_config, connections, center)
        normalized_headrooms = {}
        valid = True
        for name, item in profile["controls"].items():
            if item["lower_headroom"] is None:
                valid = False
                normalized_headrooms[name] = None
            else:
                normalized_headrooms[name] = min(item["lower_headroom"], item["upper_headroom"]) / required[name]
        rho = min(value for value in normalized_headrooms.values() if value is not None) if valid else -np.inf
        row = {
            "candidate_index": int(index), "symmetric_headroom_rho": float(rho),
            "connected_window_margin": float(profile["center_trusted_margin"]),
            "minimum_support": float(profile["center_minimum_support"]),
            "bottleneck": profile["center_bottleneck"], "scan_points_per_axis": int(setting["coarse_scan_points"]),
            "normalized_headrooms": normalized_headrooms,
        }
        row.update(center)
        rows.append(row)
    rows.sort(key=lambda row: (row["symmetric_headroom_rho"], row["connected_window_margin"], row["minimum_support"]), reverse=True)
    return rows


def batched_window_headrooms(
    predictors, config: dict[str, object], connections: dict[str, str], center: dict[str, float],
) -> dict[str, object]:
    """複数中心の各軸をまとめて評価し、必要変動幅に対する左右の余裕を計算します。"""

    scan_points = int(config["connected_window"]["scan_points"])
    bounds = {}
    for _, manifest, predictor in predictors:
        stage_bounds = input_bounds(predictor)
        bounds.update({name: stage_bounds[name] for name in manifest["controls"]})
    axes = {name: np.unique(np.append(np.linspace(*bounds[name], scan_points), center[name])) for name in center}
    offsets = {}
    total = 0
    for name, axis in axes.items():
        offsets[name] = (total, total + len(axis))
        total += len(axis)
    values = {name: np.full(total, value) for name, value in center.items()}
    for name, axis in axes.items():
        start, end = offsets[name]
        values[name][start:end] = axis
    values.update({name: np.full(total, value) for name, value in config["external_context"].items()})
    evaluated = evaluate_connected_window(predictors, config, connections, values)
    controls = {}
    for name, axis in axes.items():
        start, end = offsets[name]
        low, high = contiguous_true_interval(axis, evaluated["trusted_feasible"][start:end], center[name])
        controls[name] = {
            "lower_headroom": None if low is None else center[name] - low,
            "upper_headroom": None if high is None else high - center[name],
        }
    first_name = next(iter(axes))
    start, end = offsets[first_name]
    center_index = start + int(np.argmin(np.abs(axes[first_name] - center[first_name])))
    return {
        "controls": controls,
        "center_trusted_margin": float(evaluated["trusted_margin"][center_index]),
        "center_minimum_support": float(evaluated["minimum_support"][center_index]),
        "center_bottleneck": str(evaluated["bottleneck"][center_index]),
    }

