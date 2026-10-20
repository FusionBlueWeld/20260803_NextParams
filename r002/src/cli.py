"""r002のCLI入口。工程接続の準備と予算付き探索を実行します。"""

from __future__ import annotations

import argparse
import sys

from .trials import create_trial, prepare_trial, run_trial


def positive_integer(value: str) -> int:
    """正の推薦件数だけを許可します。"""
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("1以上の整数を指定してください。")
    return number


def main() -> int:
    """共通main.pyから渡された操作を実行します。"""
    parser = argparse.ArgumentParser(description="工程bundleをDAGで接続し、PC評価予算内で全体条件を探索します。")
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--new", metavar="TRIAL")
    action.add_argument("--prepare", metavar="TRIAL")
    action.add_argument("--run", metavar="TRIAL")
    parser.add_argument("--n", type=positive_integer, default=3)
    arguments = parser.parse_args()
    try:
        if arguments.new:
            create_trial(arguments.new)
        elif arguments.prepare:
            prepare_trial(arguments.prepare)
        else:
            run_trial(arguments.run, arguments.n)
    except (ValueError, OSError, KeyError) as error:
        print(f"[入力エラー]\n{error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
