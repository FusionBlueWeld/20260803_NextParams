"""複合工程のコマンドを受け取り、単工程評価・ライン評価・監査へ振り分けます。

工程の一覧はstages.py、物理的な接続はfunctional_coating/pipeline.pyが担当します。
学習済みモデルを使う検証はchecks/から個別に実行します。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

from ..functional_coating.benchmark import audit
from ..functional_coating.pipeline import DEFAULT_MATERIAL_STATE, LINE_INPUT_BOUNDS, evaluate_line, evaluate_continuous_line
from .stages import available, load


# ---------------------------------------------------------------------------
# コマンド入力とJSON出力の変換
# ---------------------------------------------------------------------------


def _assignments(items: list[str]) -> dict[str, float]:
    """NAME=VALUEを数値辞書へ変換し、重複名や不正な値を拒否します。"""

    result: dict[str, float] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Expected NAME=VALUE: {item}")
        name, raw = item.split("=", 1)
        if not name or name in result:
            raise ValueError(f"Invalid or duplicate input: {name!r}")
        try:
            result[name] = float(raw)
        except ValueError as error:
            raise ValueError(f"Invalid numeric value: {item}") from error
    return result


def _jsonable(value):
    """NumPy配列・スカラーをJSONで保存できる値へ変換します。"""

    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, np.ndarray):
        return value.item() if value.ndim == 0 else value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _write_or_print(value: object, output: str | None) -> None:
    """結果を標準出力または新規JSONファイルへ出力します。既存ファイルは拒否します。"""

    text = json.dumps(_jsonable(value), ensure_ascii=False, indent=2, allow_nan=False)
    if output is None:
        print(text)
        return
    path = Path(output)
    if path.exists():
        raise ValueError(f"Output already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)
    print(path.resolve())


# ---------------------------------------------------------------------------
# コマンドライン引数の定義
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    """工程評価とライン監査で指定できる引数を定義します。"""

    parser = argparse.ArgumentParser(description="Multi-stage physics validation")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("list", help="list standalone stages")

    stage = subparsers.add_parser("evaluate-stage", help="evaluate one standalone stage")
    stage.add_argument("stage", choices=available())
    stage.add_argument("--set", nargs="+", required=True, metavar="NAME=VALUE")

    line = subparsers.add_parser("evaluate-line", help="evaluate the connected three-stage line")
    line.add_argument("--drying-length-m", type=float, help="continuous-line dryer length; derive dwell times")
    line.add_argument("--curing-length-m", type=float, help="continuous-line curing oven length")
    line.add_argument("--web-width-m", type=float, default=1.0)
    line.add_argument("--set", nargs="+", required=True, metavar="NAME=VALUE")
    line.add_argument(
        "--require-material",
        action="store_true",
        help="require raw-material inputs instead of applying nominal defaults",
    )

    check = subparsers.add_parser("audit", help="run the 3^9 reference-space audit")
    check.add_argument("--samples", type=int, default=512)
    check.add_argument("--seed", type=int, default=0)
    check.add_argument("--output")
    subparsers.add_parser("interconnect", help="Cu/Alの分岐合流ライン検証（interconnect --helpで詳細）")
    return parser


# ---------------------------------------------------------------------------
# コマンドごとの実行
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    """引数を解釈して担当処理を実行し、終了コードを返します。"""

    arguments = list(sys.argv[1:] if argv is None else argv)
    if arguments and arguments[0] == "interconnect":
        # 既存の塗工系CLIを保ち、追加ラインは専用の検証器へ渡します。
        from .checks.electrical_interconnect_validation import main as interconnect_main
        return interconnect_main(arguments[1:])
    args = build_parser().parse_args(arguments)
    try:
        if args.command == "list":
            payload = {
                stage_id: {
                    "name": load(stage_id).name,
                    "controls": load(stage_id).control_names,
                    "incoming_states": load(stage_id).incoming_state_names,
                }
                for stage_id in available()
            }
            _write_or_print(payload, None)
        elif args.command == "evaluate-stage":
            stage = load(args.stage)
            _write_or_print(stage.evaluate(_assignments(args.set)), None)
        elif args.command == "evaluate-line":
            values = _assignments(args.set)
            if not args.require_material:
                values = {**DEFAULT_MATERIAL_STATE, **values}
            if args.drying_length_m is not None or args.curing_length_m is not None:
                if args.drying_length_m is None or args.curing_length_m is None:
                    raise ValueError("Supply both --drying-length-m and --curing-length-m")
                payload = evaluate_continuous_line(
                    values, drying_length_m=args.drying_length_m,
                    curing_length_m=args.curing_length_m, web_width_m=args.web_width_m,
                )
            else:
                if args.web_width_m != 1.0:
                    raise ValueError("--web-width-m requires continuous-line oven lengths")
                payload = evaluate_line(values)
            _write_or_print(payload, None)
        else:
            _write_or_print(audit(args.samples, args.seed), args.output)
    except ValueError as error:
        build_parser().error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
