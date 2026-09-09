"""v004の操作入口。引数の解釈と利用者向けエラー表示を担当します。

trialの作成・準備はtrials.py、探索の実行手順はworkflow.pyへ渡します。
"""

from __future__ import annotations

import argparse
import sys
from .settings import DEFAULT_RECOMMENDATION_COUNT
from .validation import UserInputError
from .trials import create_trial, prepare_trial
from .workflow import run_trial


def parse_arguments() -> argparse.Namespace:
    """--new、--prepare、--runのいずれか1つを受け取ります。"""

    parser = argparse.ArgumentParser(
        description="実験データから、次に試すパラメータ条件を推薦します。"
    )
    actions = parser.add_mutually_exclusive_group(required=True)
    actions.add_argument("--new", metavar="TRIAL", help="新しいtrialを作成します")
    actions.add_argument(
        "--prepare",
        metavar="TRIAL",
        help="problem.csvを確認し、experiments.csvを準備します",
    )
    actions.add_argument("--run", metavar="TRIAL", help="指定trialの探索を実行します")
    parser.add_argument(
        "--n",
        metavar="COUNT",
        type=positive_integer,
        help=(
            "--runで出力する次実験候補の件数を指定します"
            f"（省略時: {DEFAULT_RECOMMENDATION_COUNT}）"
        ),
    )
    arguments = parser.parse_args()
    if arguments.n is not None and arguments.run is None:
        parser.error("--nは--runと一緒に指定してください")
    return arguments


def positive_integer(value: str) -> int:
    """CLIで1以上の整数だけを受け付けます。"""

    try:
        parsed = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("1以上の整数を指定してください") from error
    if parsed < 1:
        raise argparse.ArgumentTypeError("1以上の整数を指定してください")
    return parsed


def main() -> int:
    """選択されたオプションに対応する処理を1つだけ実行します。"""

    arguments = parse_arguments()
    action_name = ""

    try:
        if arguments.new:
            action_name = "--new"
            create_trial(arguments.new)
        elif arguments.prepare:
            action_name = "--prepare"
            prepare_trial(arguments.prepare)
        else:
            action_name = "--run"
            recommendation_count = arguments.n or DEFAULT_RECOMMENDATION_COUNT
            run_trial(arguments.run, recommendation_count)
        return 0
    except UserInputError as error:
        print(f"[エラー]\n{error}", file=sys.stderr)
        if action_name == "--run":
            print(
                "\n今回の推薦は公開されていません。\n"
                "前回の推薦はoutput/recommendations_history/で確認できます。",
                file=sys.stderr,
            )
        return 1
    except PermissionError as error:
        print(
            "[エラー]\nファイルを開けませんでした。Excelで開いている場合は閉じてください。\n"
            f"対象: {error.filename or '不明'}",
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
