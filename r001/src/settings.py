"""r001内のフォルダと設定ファイル名。計算処理は持ちません。"""

from __future__ import annotations

from pathlib import Path


VERSION_ROOT = Path(__file__).resolve().parents[1]


PROJECT_ROOT = VERSION_ROOT.parent


TRIALS_ROOT = VERSION_ROOT / "trials"


CONFIG_NAME = "pipeline.json"

