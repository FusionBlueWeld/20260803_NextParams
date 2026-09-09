"""v004が保存した予測器の型を復元するための読込境界。

既存pickleはsrc.*という型名を持つため、v004の名前空間へ切り替えます。
この処理はCLIの実行プロセスで使い、v004側の予測器クラスの配置を維持します。
"""

from __future__ import annotations

import importlib
import sys
from .settings import PROJECT_ROOT


def load_bundle_runtime():
    """r001のsrc名前空間を退避し、v004が保存した型でbundleを復元します。"""
    for name in list(sys.modules):
        if name == "src" or name.startswith("src."):
            del sys.modules[name]
    sys.path.insert(0, str(PROJECT_ROOT / "v004"))
    sys.path.insert(1, str(PROJECT_ROOT))
    return importlib.import_module("src.stage_bundle").load_stage_bundle

