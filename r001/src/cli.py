"""r001の操作入口。引数を解釈し、trial準備または工程接続の実行へ渡します。

計算は各担当モジュールに置き、ここでは終了コードと利用者向けエラーを扱います。
"""

from __future__ import annotations

import argparse
import sys
from .validation import UserInputError
from .trials import create_trial, prepare_trial
from .workflow import run_trial


def positive_integer(value: str) -> int:
    """推薦件数として1以上の整数だけを受け付けます。"""

    result = int(value)
    if result < 1:
        raise argparse.ArgumentTypeError("1以上の整数を指定してください")
    return result


def main() -> int:
    """指定された操作を1つ実行し、入力エラーを終了コード2で通知します。"""

    parser = argparse.ArgumentParser(description="v004 stage bundleを接続して全体条件を調整します。")
    actions = parser.add_mutually_exclusive_group(required=True)
    actions.add_argument("--new", metavar="TRIAL")
    actions.add_argument("--prepare", metavar="TRIAL")
    actions.add_argument("--run", metavar="TRIAL")
    parser.add_argument("--n", type=positive_integer, default=3)
    args = parser.parse_args()
    try:
        if args.new:
            create_trial(args.new)
        elif args.prepare:
            prepare_trial(args.prepare)
        else:
            run_trial(args.run, args.n)
        return 0
    except (UserInputError, ValueError, OSError) as error:
        print(f"[入力エラー]\n{error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
