"""v005の安定した名前空間を使うbundle読込境界。"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

from .settings import PROJECT_ROOT


def load_predictors(config: dict) -> dict:
    """工程IDごとにmanifestと予測器を読み込みます。物理oracleは参照しません。"""
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    loader = importlib.import_module("v005.src.stage_bundle").load_stage_bundle
    base = Path(config.get("_base_path", PROJECT_ROOT))
    predictors = {}
    for item in config["stages"]:
        path = Path(item["bundle"])
        path = path if path.is_absolute() else base / path
        predictors[item["id"]] = loader(path.resolve())
    return predictors
