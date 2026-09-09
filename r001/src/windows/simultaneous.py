"""複数操作条件を同時に変え、箱内標本・頂点・二変数断面の成立を評価します。

一変数窓を組み合わせた箱の全点が合格するとは限らないため、別に評価します。
"""

from __future__ import annotations

import itertools
import numpy as np
from ..pipeline import input_bounds
from ..windows.evaluation import evaluate_connected_window


def evaluate_simultaneous_window(
    predictors, config: dict[str, object], connections: dict[str, str], center: dict[str, float],
    rho: float, normalized_headrooms: dict[str, float],
) -> dict[str, object]:
    """同時変更の箱内標本・頂点・二変数断面を採点します。連続領域の保証は行いません。"""

    setting = config["simultaneous_window"]
    required = {name: float(value) for name, value in config["window_center_selection"]["required_variations"].items()}
    bounds = {}
    for _, manifest, predictor in predictors:
        stage_bounds = input_bounds(predictor)
        bounds.update({name: stage_bounds[name] for name in manifest["controls"]})
    rng = np.random.default_rng(int(setting["seed"]))

    def evaluate_controls(control_values: dict[str, np.ndarray]) -> dict[str, object]:
        size = len(next(iter(control_values.values())))
        values = dict(control_values)
        values.update({name: np.full(size, value) for name, value in config["external_context"].items()})
        result = evaluate_connected_window(predictors, config, connections, values)
        good = np.asarray(result["trusted_feasible"], dtype=bool)
        return {"count": size, "predicted_pass_count": int(good.sum()), "predicted_pass_fraction": float(good.mean())}

    random_rows = []
    samples = int(setting["samples"])
    for scale in map(float, setting["radius_scales"]):
        controls = {}
        clipped = np.zeros(samples, dtype=bool)
        for name in center:
            raw = center[name] + rng.uniform(-1, 1, samples) * rho * scale * required[name]
            controls[name] = np.clip(raw, *bounds[name])
            clipped |= controls[name] != raw
        row = {"radius_scale": scale, "clipped_fraction": float(clipped.mean()), **evaluate_controls(controls)}
        random_rows.append(row)

    names = list(center)
    signs = np.asarray(list(itertools.product((-1.0, 1.0), repeat=len(names))))
    vertex_controls = {
        name: np.clip(center[name] + signs[:, column] * rho * required[name], *bounds[name])
        for column, name in enumerate(names)
    }
    vertices = evaluate_controls(vertex_controls)

    limiting = sorted(names, key=lambda name: normalized_headrooms[name])[:max(2, int(setting["pair_count"]) + 1)]
    pairs = list(itertools.combinations(limiting, 2))[:int(setting["pair_count"])]
    pair_rows = []
    grid_points = int(setting["pair_grid_points"])
    unit = np.linspace(-1, 1, grid_points)
    mesh_a, mesh_b = np.meshgrid(unit, unit, indexing="ij")
    for first, second in pairs:
        size = mesh_a.size
        controls = {name: np.full(size, value) for name, value in center.items()}
        controls[first] = np.clip(center[first] + mesh_a.reshape(-1) * rho * required[first], *bounds[first])
        controls[second] = np.clip(center[second] + mesh_b.reshape(-1) * rho * required[second], *bounds[second])
        pair_rows.append({"controls": [first, second], **evaluate_controls(controls)})
    return {
        "definition": "Finite samples inside the joint +/-rho*required_variation box; not exact 9-D volume.",
        "rho": rho, "required_variations": required, "random_seed": int(setting["seed"]),
        "random_uniform_boxes": random_rows, "rho_box_vertices": vertices,
        "pairwise_sections": pair_rows,
    }

