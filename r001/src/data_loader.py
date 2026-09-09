"""r001のJSON・CSV保存。完成した一時ファイルを公開先へ置き換えます。

列順と既存の出力形式を維持し、計算や合否判定は呼出し側で行います。
"""

from __future__ import annotations

import csv
import json
from pathlib import Path


def write_json(path: Path, value: object) -> None:
    """日本語を保持したJSONを一時ファイルへ書き、完成後に置き換えます。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    """行辞書のキーから列順を作り、結果をCSVへ保存します。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    header = list(rows[0]) if rows else []
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=header)
        if header:
            writer.writeheader()
            writer.writerows(rows)
    temporary.replace(path)

