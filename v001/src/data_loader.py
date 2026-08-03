"""CSVファイルの読込と書込を担当する機能。

CSVの内容が正しいかどうかはvalidation.pyで判断します。このファイルでは、
文字コードの違いを吸収し、CSVを行の集まりとして読み込むことに集中します。
"""

from __future__ import annotations

import csv
import os
from pathlib import Path
from typing import Iterable


# ---------------------------------------------------------------------------
# CSVを読み込む機能
# ---------------------------------------------------------------------------


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    """CSVを読み込み、ヘッダーと各行を返します。

    Excelから保存されたCSVも扱いやすいように、UTF-8、UTF-8 BOM付き、
    WindowsのCP932の順で読込を試します。
    """

    last_error: UnicodeDecodeError | None = None

    for encoding in ("utf-8-sig", "cp932"):
        try:
            with path.open("r", encoding=encoding, newline="") as file:
                reader = csv.DictReader(file)
                header = list(reader.fieldnames or [])
                rows = [
                    {key: (value or "").strip() for key, value in row.items() if key is not None}
                    for row in reader
                ]
                return header, rows
        except UnicodeDecodeError as error:
            last_error = error

    if last_error is not None:
        raise last_error
    return [], []


# ---------------------------------------------------------------------------
# CSVを安全に書き込む機能
# ---------------------------------------------------------------------------


def write_csv_atomic(
    path: Path,
    header: list[str],
    rows: Iterable[dict[str, object]],
) -> None:
    """CSVを一時ファイルへ書いた後、完成時だけ本番ファイルへ置き換えます。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")

    try:
        with temporary_path.open("w", encoding="utf-8-sig", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=header, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary_path, path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def write_problem_template(path: Path) -> None:
    """新しいtrialへ、記入例付きのproblem.csvを作成します。"""

    header = [
        "column",
        "display_name",
        "unit",
        "role",
        "direction",
        "lower",
        "upper",
        "step",
        "target",
    ]
    rows = [
        {
            "column": "input_1",
            "display_name": "入力パラメータ1",
            "unit": "",
            "role": "parameter",
            "direction": "",
            "lower": 0,
            "upper": 100,
            "step": 10,
            "target": "",
        },
        {
            "column": "result_1",
            "display_name": "評価結果1",
            "unit": "",
            "role": "objective",
            "direction": "minimize",
            "lower": "",
            "upper": "",
            "step": "",
            "target": "",
        },
    ]
    write_csv_atomic(path, header, rows)


