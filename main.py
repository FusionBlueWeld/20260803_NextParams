"""ParamOptimizerの全バージョン共通CLI入口。

利用者は最初に--versionで実装を選び、その後へ各バージョン共通の
--new、--prepare、--runなどの操作を続けます。
"""

from __future__ import annotations

import subprocess
import sys
import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
VERSION_ROOTS = {
    "v000": PROJECT_ROOT / "v000",
    "v001": PROJECT_ROOT / "v001",
    "v002": PROJECT_ROOT / "v002",
}


def print_help(*, stream: object = sys.stdout) -> None:
    """共通入口の使い方と利用可能バージョンを表示します。"""

    print(
        "ParamOptimizer 共通実行入口\n"
        "\n"
        "使い方:\n"
        "  python main.py --version VERSION --new TRIAL\n"
        "  python main.py --version VERSION --prepare TRIAL\n"
        "  python main.py --version VERSION --run TRIAL [--n COUNT]\n"
        "  python main.py --version VERSION --help\n"
        "\n"
        "利用可能なVERSION:\n"
        "  v000  Gaussian Processによる次実験条件探索\n"
        "  v001  GP探索 + NN予測空間のrun別出力\n"
        "  v002  GP–NN統計Hybridによる探索とrun別解空間\n"
        "\n"
        "例:\n"
        "  python main.py --version v000 --run trial_000\n"
        "  python main.py --version v001 --run trial_001 --n 9\n"
        "  python main.py --version v002 --run trial_002 --n 9",
        file=stream,
    )


def main(arguments: list[str] | None = None) -> int:
    """バージョン指定を検査し、選択した実装へ残りの引数を渡します。"""

    arguments = list(sys.argv[1:] if arguments is None else arguments)
    if not arguments or arguments[0] in {"-h", "--help"}:
        print_help()
        return 0

    if arguments[0] != "--version":
        print(
            "[エラー]\n最初のオプションには --version を指定してください。\n",
            file=sys.stderr,
        )
        print_help(stream=sys.stderr)
        return 2
    if len(arguments) < 2:
        print("[エラー]\n--versionの後にv000、v001、v002のいずれかを指定してください。", file=sys.stderr)
        return 2

    version = arguments[1]
    version_root = VERSION_ROOTS.get(version)
    if version_root is None:
        choices = ", ".join(VERSION_ROOTS)
        print(
            f"[エラー]\n未対応のバージョンです: {version}\n"
            f"利用可能: {choices}",
            file=sys.stderr,
        )
        return 2

    remaining = arguments[2:]
    if not remaining:
        print(
            "[エラー]\nバージョンの後に--new、--prepare、--run、--helpの"
            "いずれかを指定してください。",
            file=sys.stderr,
        )
        return 2

    implementation = version_root / "src" / "cli.py"
    if not implementation.is_file():
        print(
            f"[エラー]\n{version}の実行ファイルが見つかりません。\n対象: {implementation}",
            file=sys.stderr,
        )
        return 1

    child_environment = os.environ.copy()
    child_environment["PYTHONIOENCODING"] = "utf-8"
    completed = subprocess.run(
        [sys.executable, "-m", "src.cli", *remaining],
        cwd=version_root,
        env=child_environment,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    command_prefix = f"python main.py --version {version} --"
    def rewrite_child_text(text: str) -> str:
        return (
            text.replace("python main.py --", command_prefix)
            .replace("usage: cli.py", f"usage: main.py --version {version}")
            .replace("cli.py: error:", f"main.py --version {version}: error:")
        )

    if completed.stdout:
        sys.stdout.write(rewrite_child_text(completed.stdout))
    if completed.stderr:
        sys.stderr.write(rewrite_child_text(completed.stderr))
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
