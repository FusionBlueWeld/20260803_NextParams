"""初期実験計画とパラメータ集中の任意設定。"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import math
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class FocusConfig:
    """初期設計・集中探索の設定。3機能は既定で有効です。"""

    initial_design_enabled: bool = True
    parameter_focus_enabled: bool = True
    temporary_fix_enabled: bool = True
    initial_min_conditions: int = 12
    initial_max_conditions: int = 48
    max_total_experiments: int | None = None
    confirmation_slots: int = 1
    stability_updates: int = 3
    effect_tolerances: dict[str, float] = field(default_factory=dict)
    never_fix: tuple[str, ...] = ()
    resamples: int = 12
    max_screening_samples: int = 256
    min_background_checks: int = 2
    global_check_interval: int = 3


_KNOWN = set(FocusConfig.__dataclass_fields__)
_INTEGER_LIMITS = {
    "initial_min_conditions": 10_000,
    "initial_max_conditions": 10_000,
    "max_total_experiments": 1_000_000,
    "confirmation_slots": 1_000,
    "stability_updates": 1_000,
    "resamples": 64,
    "max_screening_samples": 512,
    "min_background_checks": 8,
    "global_check_interval": 10_000,
}


def _positive_int(value: Any, name: str, *, allow_zero: bool = False) -> int:
    lower = 0 if allow_zero else 1
    upper = _INTEGER_LIMITS[name]
    if type(value) is not int or not lower <= value <= upper:
        raise ValueError(f"{name}は{lower}～{upper}の整数にしてください。")
    return value


def load_focus_config(path: str | Path, problem=None, connection=None) -> FocusConfig:
    """JSONを厳密に読み、問題・接続で有効な名前を検査します。"""

    config_path = Path(path)
    if not config_path.exists():
        return FocusConfig()
    try:
        raw = json.loads(config_path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"focus_settings.jsonを読み取れません: {error}") from error
    if not isinstance(raw, dict):
        raise ValueError("focus_settings.jsonはJSONオブジェクトにしてください。")
    unknown = set(raw) - _KNOWN
    if unknown:
        raise ValueError("未定義の設定があります: " + ", ".join(sorted(unknown)))

    values = dict(raw)
    for key in ("initial_design_enabled", "parameter_focus_enabled", "temporary_fix_enabled"):
        if key in values and type(values[key]) is not bool:
            raise ValueError(f"{key}はtrue/falseにしてください。")
    integer_fields = ("initial_min_conditions", "initial_max_conditions", "confirmation_slots",
                     "stability_updates", "resamples", "max_screening_samples",
                     "min_background_checks", "global_check_interval")
    for key in integer_fields:
        if key in values:
            _positive_int(values[key], key)
    if "max_total_experiments" in values and values["max_total_experiments"] is not None:
        _positive_int(values["max_total_experiments"], "max_total_experiments")
    if values.get("initial_min_conditions", 12) > values.get("initial_max_conditions", 48):
        raise ValueError("initial_min_conditionsはinitial_max_conditions以下にしてください。")
    if values.get('initial_min_conditions', 12) < 2:
        raise ValueError('初期実験は異なる条件を2件以上用意してください。')
    if values.get("temporary_fix_enabled", True) and not values.get("parameter_focus_enabled", True):
        raise ValueError("temporary_fix_enabledにはparameter_focus_enabledが必要です。")

    tolerances = values.get("effect_tolerances", {})
    if not isinstance(tolerances, dict):
        raise ValueError("effect_tolerancesは出力列名と正の有限許容幅の辞書にしてください。")
    for name, tolerance in tolerances.items():
        if not isinstance(name, str) or not name.strip() or isinstance(tolerance, bool) or not isinstance(tolerance, (int, float)) or not math.isfinite(tolerance) or tolerance <= 0:
            raise ValueError("effect_tolerancesは出力列名と正の有限許容幅の辞書にしてください。")
    if problem is not None:
        allowed_outputs = {problem.objective.column, *(item.column for item in problem.constraints)}
        extra = set(tolerances) - allowed_outputs
        if extra:
            raise ValueError("effect_tolerancesに目的・制約以外の列があります: " + ", ".join(sorted(extra)))
    values["effect_tolerances"] = {str(name): float(value) for name, value in tolerances.items()}

    never_fix = values.get("never_fix", ())
    if not isinstance(never_fix, (list, tuple)) or any(not isinstance(name, str) or not name.strip() for name in never_fix) or len(set(never_fix)) != len(never_fix):
        raise ValueError("never_fixは重複のない制御列名の配列にしてください。")
    if problem is not None:
        parameters = {item.column for item in problem.parameters}
        controls = parameters if connection is None else set(connection.controls)
        unknown_names = set(never_fix) - controls
        if unknown_names:
            raise ValueError("never_fixに制御列以外の名前があります: " + ", ".join(sorted(unknown_names)))
    values["never_fix"] = tuple(never_fix)
    return FocusConfig(**values)
