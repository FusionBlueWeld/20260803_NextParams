"""検証用CSV・JSONの読込と保存を担当します。

問題定義の意味やモデルの出力はsimulators.pyが検査します。
書込みは一時ファイルを完成させてから置き換え、途中の結果が残ることを防ぎます。
"""

from __future__ import annotations

import csv
import json
from pathlib import Path


def read_csv(path: Path) -> list[dict[str, str]]:
    """UTF-8 BOM付きCSVを行辞書の一覧として読み込みます。"""

    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, header: list[str], rows) -> None:
    """列順を維持してCSVを保存し、完成後に一時ファイルから置き換えます。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=header)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def write_json(path: Path, value: object) -> None:
    """NaN・Infを拒否し、日本語を保持したJSONとして保存します。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)

