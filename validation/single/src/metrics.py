"""検証結果を採点する機能。予測誤差、最良値との差、制約合否を計算します。

全候補の正解は採点専用です。推薦するoptimizerへ渡す学習データには混ぜません。
解空間がない版では、推薦点の誤差を全候補の誤差として代用しません。
"""

from __future__ import annotations

import re
from pathlib import Path
import numpy as np

from .data_loader import read_csv
from .settings import PROJECT_ROOT
from .simulators import Simulator


def regression_metrics(prediction, truth, span: float) -> dict:
    """予測と真値の差からMAE・RMSEと出力幅で正規化したRMSEを計算します。"""

    error = np.asarray(prediction, dtype=float) - np.asarray(truth, dtype=float)
    if not np.all(np.isfinite(error)):
        raise ValueError("Nonfinite prediction error")
    if not error.size:
        return {"count": 0, "mae": None, "rmse": None, "normalized_rmse": None}
    rmse = float(np.sqrt(np.mean(error**2)))
    return {"count": int(error.size), "mae": float(np.mean(np.abs(error))), "rmse": rmse,
            "normalized_rmse": rmse / span if span > 0 else None}


def regret(sim: Simulator, value: float | None, optimum: float, span: float) -> float | None:
    """候補グリッド最良値との差を出力幅で割ります。最良観測なしはNoneです。"""

    if value is None:
        return None
    gap = optimum - value if sim.objective.direction == "maximize" else value - optimum
    return max(0.0, gap) / span if span > 0 else 0.0


def summarize_best(sim: Simulator, points, outputs) -> dict | None:
    """制約を満たす最良点の条件と出力を、保存用の辞書にまとめます。"""

    idx = sim.best_index(outputs)
    if idx is None:
        return None
    return {"parameters": {n: float(points[idx, j]) for j, n in enumerate(sim.parameter_columns)},
            "outputs": {n: float(outputs[n][idx]) for n in sim.output_columns}}


def response_metrics(sim: Simulator, trial: Path, seen_points: np.ndarray, spans: dict) -> dict:
    # v000 has no full response-space export. Never substitute recommendation-only errors.
    """最新の全候補予測を採点し、未測定点の誤差も分けて返します。"""

    files = sorted((trial / "output").rglob("*response_space_run_*.csv"))
    if not files:
        return {"status": "not_available", "reason": "This version did not export a response space"}
    path = max(files, key=lambda p: int(re.search(r"response_space_run_(\d+)\.csv$", p.name).group(1)))
    rows = read_csv(path)
    if not rows:
        raise ValueError("Empty response space")
    if "data_rows" in rows[0] and any(int(row["data_rows"]) != len(seen_points) for row in rows):
        return {"status": "not_available", "reason": "Response space predates the final observations"}
    points = np.array([[float(row[n]) for n in sim.parameter_columns] for row in rows])
    truth = sim.evaluate_points(points)
    seen = {tuple(np.round(p, 8)) for p in seen_points}
    unobserved = np.array([tuple(np.round(p, 8)) not in seen for p in points])
    metrics = {}
    selected_prediction = {}
    for n in sim.output_columns:
        keys = [f"{n}_hybrid_mean", f"{n}_mean", f"{n}_nn_pred"]
        key = next((key for key in keys if key in rows[0]), None)
        if key is None:
            raise ValueError(f"Missing prediction for {n} in {path.name}")
        predicted = np.array([float(row[key]) for row in rows])
        selected_prediction[n] = predicted
        metrics[n] = {"prediction_column": key,
                      "all_grid": regression_metrics(predicted, truth[n], spans[n]),
                      "unobserved": regression_metrics(predicted[unobserved], truth[n][unobserved], spans[n])}
    # Check the export actually covers the declared candidate grid, not a partial file.
    keys = {tuple(np.round(p, 8)) for p in points}
    expected = {tuple(np.round(p, 8)) for p in sim.grid()}
    if len(keys) != len(points) or keys != expected:
        raise ValueError("Response-space export is not the complete unique candidate grid")
    return {"status": "ok", "file": str(path.relative_to(PROJECT_ROOT)), "row_count": len(rows),
            "unobserved_count": int(unobserved.sum()), "outputs": metrics,
            "feasibility_classification_accuracy": float(np.mean(sim.feasible(selected_prediction) == sim.feasible(truth)))}

