"""r002 trialの作成、検査、JSON/CSV結果保存。"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from .api import load_pipeline, run_search
from .settings import DEFAULT_SEARCH, TRIALS_ROOT


def trial_path(name: str) -> Path:
    """trial名が専用フォルダを逸脱しないことを確認します。"""
    if not name.startswith("trial_") or not all(char.isascii() and (char.isalnum() or char in "_-") for char in name):
        raise ValueError("trial名はtrial_で始まる半角英数字、_、-です。")
    return TRIALS_ROOT / name


def _json_default(value):
    """NumPy値をJSONに保存できる型へ変換します。"""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def write_json(path: Path, value: dict) -> None:
    """出力先を作成し、人が読めるUTF-8 JSONとして保存します。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=_json_default, allow_nan=False), encoding="utf-8")


def create_trial(name: str) -> None:
    """既存trialを上書きせず、工程接続の記入用設定を作ります。"""
    path = trial_path(name)
    path.mkdir(parents=True, exist_ok=False)
    template = {
        "schema_version": "2.0",
        "stages": [{"id": "stage_1", "bundle": "../../../v005/trials/trial_example/output/stage_bundle"}],
        "connections": [], "external_context": {}, "candidate_axes": {},
        "final_specifications": [], "search": dict(DEFAULT_SEARCH),
        "fixed_controls": {}, "observed_outputs": {}, "inventory": [],
    }
    write_json(path / "pipeline.json", template)
    print(f"{name}を作成しました。設定: {path / 'pipeline.json'}")


def prepare_trial(name: str) -> None:
    """bundleと接続・操作・支持域・予算を予測前に検査します。"""
    config, models = load_pipeline(trial_path(name))
    print(f"pipeline.jsonを確認しました。工程数={len(models)}、全直積は生成しません。")


def run_trial(name: str, recommendation_count: int) -> dict:
    """探索結果と必要な部分実験・入力ばらつき評価を保存します。"""
    path = trial_path(name)
    result = run_search(path, recommendation_count=recommendation_count)
    output = path / "output"
    write_json(output / "run_summary.json", result)
    rows = []
    for item in result["candidates"]:
        row = {key: value for key, value in item.items() if key != "controls"}
        row.update(item["controls"])
        rows.append(row)
    # 候補ゼロでも空のCSVを保存し、前runの推薦が今回の結果として残るのを防ぎます。
    fields = list(rows[0]) if rows else ["rank", "predicted_feasible", "trusted_margin"]
    with (output / "system_recommendations.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"{name}: {result['status']}、評価{result['evaluation_count']:,}件、{result['elapsed_seconds']:.2f}秒")
    print(f"出力: {output / 'run_summary.json'}")
    return result
