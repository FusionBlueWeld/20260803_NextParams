"""v003追加検証のCSV契約と、実CLI・予測空間への接続を担当します。

レーザー検証と合成データ検証が同じ列名の解釈・CLI起動方法を使うための共通部です。
6工程共通のdata_loader.pyとは戻り値が異なる既存契約を維持しています。
"""

from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable

from ..settings import PROJECT_ROOT


PROBLEM_HEADER = [
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


KNOWLEDGE_HEADER = [
    "rule_id",
    "type",
    "target",
    "wrt",
    "value",
    "tolerance",
    "strength",
    "enabled",
    "note",
]


SEARCH_REGIONS_HEADER = [
    "region_id",
    "kind",
    "parameter",
    "lower",
    "upper",
    "lower_inclusive",
    "upper_inclusive",
    "strength",
    "enabled",
    "note",
]


def write_csv(path: Path, header: list[str], rows: Iterable[dict[str, Any]]) -> None:
    """検証用 CSV を UTF-8 BOM 付きで安全に作成します。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=header, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    """BOM の有無を吸収して CSV を読む小さなヘルパーです。"""

    with path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        return list(reader.fieldnames or []), list(reader)


def number(row: dict[str, str], names: Iterable[str], default: float = float("nan")) -> float:
    """候補名を順に探して数値化します。"""

    for name in names:
        value = row.get(name, "")
        if value not in (None, ""):
            try:
                return float(value)
            except ValueError:
                continue
    return default


def detect_prediction(row: dict[str, str], column: str) -> float:
    """v000〜v003 の命名差を吸収して予測平均を取得します。"""

    return number(
        row,
        (
            f"{column}_mean",
            f"{column}_hybrid_mean",
            f"{column}_predicted",
            f"{column}_prediction",
            f"{column}_nn_pred",
            f"{column}_gp_mean",
            column,
        ),
    )


def run_optimizer(optimizer_root: Path, *arguments: str) -> str:
    """共通 main.py 経由で v003 CLI を実行します。"""

    command = [
        sys.executable,
        str(PROJECT_ROOT / "main.py"),
        "--version",
        optimizer_root.resolve().name,
        *arguments,
    ]
    completed = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if completed.returncode:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise RuntimeError(f"CLI failed ({' '.join(arguments)}):\n{detail}")
    return completed.stdout


def latest_response_space(trial_path: Path) -> Path | None:
    """v002/v003 の解空間 CSV のうち最新ファイルを返します。"""

    directories = [
        trial_path / "output" / "response_spaces",
        trial_path / "output" / "response_space",
        trial_path / "output",
    ]
    candidates: list[Path] = []
    for directory in directories:
        if directory.is_dir():
            candidates.extend(
                path
                for path in directory.glob("*.csv")
                if "recommend" not in path.name.lower()
            )
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime_ns)

