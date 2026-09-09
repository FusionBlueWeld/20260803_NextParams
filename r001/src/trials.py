"""trial名の確認、新規設定の作成、prepareの実行を担当します。"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from .settings import CONFIG_NAME, TRIALS_ROOT
from .data_loader import write_json
from .validation import UserInputError, load_pipeline_config, validate_pipeline


def trial_path(name: str) -> Path:
    """trial名を検査し、r001のtrials配下のパスを返します。存在確認は行いません。"""

    if not name.startswith("trial_") or not all(c.isalnum() or c in "_-" for c in name):
        raise UserInputError("trial名はtrial_で始め、半角英数字、_、-だけで指定してください。")
    return TRIALS_ROOT / name


def create_trial(name: str) -> None:
    """既存trialを上書きせず、接続設定の記入例を持つ新規trialを作成します。"""

    path = trial_path(name)
    if path.exists():
        raise UserInputError(f"{name}は既に存在します。")
    TRIALS_ROOT.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{name}_", dir=TRIALS_ROOT))
    try:
        (temporary / "output").mkdir()
        template = {
            "schema_version": "1.0",
            "stages": [{"id": "stage_1", "bundle": "../../../v004/trials/trial_example/output/stage_bundles/stage_bundle_run_001"}],
            "connections": [],
            "external_context": {},
            "candidate_axes": {},
            "final_specifications": [],
            "connected_window": {
                "support_threshold": 0.5,
                "interval_method": "std_multiplier",
                "confidence_z": 1.645,
                "enforce_local_constraints": True,
                "use_for_selection": True,
                "scan_points": 121,
            },
            "window_center_selection": {
                "enabled": False,
                "candidate_limit": 64,
                "coarse_scan_points": 41,
                "required_variations": {},
            },
            "simultaneous_window": {
                "enabled": False,
                "samples": 4096,
                "seed": 20260923,
                "radius_scales": [0.5, 1.0, 1.5],
                "pair_grid_points": 31,
                "pair_count": 3,
            },
        }
        write_json(temporary / CONFIG_NAME, template)
        temporary.rename(path)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    print(f"{name}を作成しました。\n設定: {path / CONFIG_NAME}")


def prepare_trial(name: str) -> None:
    """保存済み予測器のmanifestとpipeline設定を照合し、学習前に不整合を検出します。"""

    trial = trial_path(name)
    if not trial.is_dir():
        raise UserInputError(f"{name}が見つかりません。")
    stages = validate_pipeline(trial, load_pipeline_config(trial))
    print(f"pipeline.jsonを確認しました。工程数={len(stages)}")

