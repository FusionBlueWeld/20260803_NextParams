"""終了判定設定CSVの読込とテンプレート生成。"""

from __future__ import annotations

import csv
import math
from pathlib import Path

from .criteria import StopConfig


STOP_SETTINGS_HEADER = ["setting", "value", "description"]
DEFAULT_SETTINGS: tuple[tuple[str, str, str], ...] = (
    ("max_additional_experiments", "", "空欄なら実験回数による強制終了なし"),
    ("min_history", "5", "収束判定を始める異なる実験データの最小状態数"),
    ("patience", "3", "直近の異なる実験データ何状態で安定を確認するか"),
    ("support_threshold", "0.8", "1候補をデータ支持ありとみなすGP支持度"),
    ("coverage_threshold", "0.8", "十分とみなすデータ支持候補の割合"),
    ("recent_improvement_tolerance", "0.001", "目的値停滞の許容幅"),
    ("top_score_threshold", "0.05", "十分小さいとみなす正規化推薦スコア"),
    ("top_score_tolerance", "0.001", "直近スコア変動の許容幅"),
    ("condition_tolerance", "0.05", "予測最適条件の正規化移動距離"),
)


def write_stop_settings_template(path: str | Path) -> None:
    """新規trialへ既定値と説明を保存します。"""

    csv_path = Path(path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=STOP_SETTINGS_HEADER)
        writer.writeheader()
        for setting, value, description in DEFAULT_SETTINGS:
            writer.writerow(
                {"setting": setting, "value": value, "description": description}
            )


def load_stop_config(path: str | Path, *, objective_direction: str) -> StopConfig:
    """CSVを読み、未指定値には``StopConfig``の既定値を使います。"""

    csv_path = Path(path)
    if not csv_path.exists():
        return StopConfig(objective_direction=objective_direction)
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if list(reader.fieldnames or ()) != STOP_SETTINGS_HEADER:
            raise ValueError(
                "stop_settings.csvのヘッダーが不正です。必要: "
                + ",".join(STOP_SETTINGS_HEADER)
            )
        rows = list(reader)

    known = {name for name, _, _ in DEFAULT_SETTINGS}
    values: dict[str, str] = {}
    for row in rows:
        name = (row.get("setting") or "").strip()
        if name not in known:
            raise ValueError(f"stop_settings.csvに未定義のsettingがあります: {name}")
        if name in values:
            raise ValueError(f"stop_settings.csvでsettingが重複しています: {name}")
        values[name] = (row.get("value") or "").strip()

    def integer(name: str, default: int | None) -> int | None:
        raw = values.get(name, "")
        return default if raw == "" else int(raw)

    def number(name: str, default: float | None) -> float | None:
        raw = values.get(name, "")
        result = default if raw == "" else float(raw)
        if result is not None and not math.isfinite(result):
            raise ValueError(f"{name}は有限値で指定してください")
        return result

    defaults = StopConfig(objective_direction=objective_direction)
    return StopConfig(
        max_additional_experiments=integer("max_additional_experiments", None),
        min_history=integer("min_history", defaults.min_history) or 0,
        patience=integer("patience", defaults.patience) or 0,
        support_threshold=number("support_threshold", defaults.support_threshold) or 0.0,
        coverage_threshold=number("coverage_threshold", defaults.coverage_threshold) or 0.0,
        recent_improvement_tolerance=number(
            "recent_improvement_tolerance", defaults.recent_improvement_tolerance
        )
        or 0.0,
        top_score_threshold=number("top_score_threshold", defaults.top_score_threshold),
        top_score_tolerance=number("top_score_tolerance", defaults.top_score_tolerance)
        or 0.0,
        condition_tolerance=number("condition_tolerance", defaults.condition_tolerance)
        or 0.0,
        objective_direction=objective_direction,
    )
