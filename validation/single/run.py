"""単工程検証の起動入口。引数の解釈と実行はsrc/cli.pyが担当します。"""

from pathlib import Path
import sys

# ファイルを直接起動した場合も、作業フォルダを基準にパッケージを読み込みます。
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from validation.single.src.cli import main


if __name__ == "__main__":
    raise SystemExit(main())
