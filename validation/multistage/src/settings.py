"""検証用のフォルダ設定。実行場所に依存せず、同じモデルと結果先を参照します。"""

from pathlib import Path


# srcの1階層上が検証領域、validationの1階層上がプロジェクト直下です。
VALIDATION_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = VALIDATION_ROOT.parents[1]
RESULTS_ROOT = VALIDATION_ROOT / "results"
