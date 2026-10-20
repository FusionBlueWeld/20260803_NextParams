"""既存の完了ワーク系譜を、再現可能なレシピ起点として保持します。

独立工程ごとの間引きで同じ系譜が全モデルから消えることを防ぎます。
使うのは既に検査した実験履歴であり、未来の未測定条件のoracle値ではありません。
"""

from __future__ import annotations

import copy
from pathlib import Path

import numpy as np

from .interconnect_data import train_bundles
from .interconnect_metrics import compact_search


def historical_warm_start(config: dict, dataset: dict, destination: Path, sample_cap: int, limit: int = 4) -> dict:
    """全工程を通った既知の良品レシピから、同じ予算で新しい条件を探索します。"""
    from r002.src.api import run_search
    from r002.src.pipeline import specification_margin
    outputs = dataset["stage_outputs"]["welding"]
    margins = np.minimum.reduce([specification_margin(outputs[spec["name"]], spec)
                                 for spec in config["final_specifications"]])
    eligible = np.flatnonzero(dataset["passing"]["welding"] & np.isfinite(margins) & (margins >= 0))
    selected = eligible[np.argsort(margins[eligible])[::-1]][:limit]
    if not len(selected):
        return {"status": "NO_OBSERVED_FEASIBLE_HISTORY", "anchor_count": 0}
    anchors = [{name: float(value[row]) for name, value in dataset["controls"].items()} for row in selected]
    linked = copy.deepcopy(dataset)
    for block in linked["blocks"].values():
        preferred = np.isin(block["row_indices"], selected)
        order = np.concatenate((np.flatnonzero(~preferred), np.flatnonzero(preferred)))
        for group in ("inputs", "outputs"):
            block[group] = {name: value[order] for name, value in block[group].items()}
        block["row_indices"] = block["row_indices"][order]
        block["workpiece_ids"] = [block["workpiece_ids"][index] for index in order]
        block["retention_policy"] = "coherent_completed_lineage_priority"
    _, predictors = train_bundles(linked, destination / "bundles", sample_cap)
    local = copy.deepcopy(config)
    local["warm_start"] = anchors
    result = run_search(local, predictors, recommendation_count=4)
    return {"status": "SEARCHED_FROM_OBSERVED_HISTORY", "anchor_count": len(anchors),
            "anchors": [{"controls": controls, "observed_margin": float(margins[row])}
                        for controls, row in zip(anchors, selected)],
            "learned_search": compact_search(result, local),
            "interpretation": "reproducing_known_feasible_recipes_is_not_discovery_of_a_new_optimum",
            "training_regime": "PRIVILEGED_SYNTHETIC_COMPLETED_WORK_HISTORY"}

