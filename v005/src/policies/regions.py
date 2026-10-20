"""``search_regions.csv`` の読込と候補方針の評価。

方針はNNの損失ではなく、実験候補の除外と推薦スコアの調整に使います。
同じ ``region_id`` の行はANDで結合されるため、将来「P1>2000かつP2<10」
のような複合領域へ自然に拡張できます。未指定のパラメータにはルールを
適用しません。
"""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np

KIND_FORBIDDEN = "forbidden"
KIND_PREFERRED = "preferred"
STRENGTH_FACTORS = (1.05, 1.10, 1.25, 1.50, 2.00)
DEFAULT_MAX_MULTIPLIER = 2.0
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


def _number(value: str | float | int | None) -> float | None:
    if value is None or str(value).strip() == "":
        return None
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"範囲値は有限値で指定してください: {value!r}")
    return number


def _boolean(value: str | bool | None, *, default: bool = True) -> bool:
    if value is None or str(value).strip() == "":
        return default
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "on", "有効"}:
        return True
    if normalized in {"0", "false", "no", "off", "無効"}:
        return False
    raise ValueError(f"真偽値を解釈できません: {value!r}")


@dataclass(frozen=True)
class RegionCondition:
    """1つのパラメータに対する区間条件。"""

    parameter: str
    lower: float | None = None
    upper: float | None = None
    lower_inclusive: bool = True
    upper_inclusive: bool = True

    def __post_init__(self) -> None:
        if not self.parameter.strip():
            raise ValueError("parameterは空にできません")
        if self.lower is None and self.upper is None:
            raise ValueError("lowerまたはupperの少なくとも一方が必要です")
        if self.lower is not None and self.upper is not None and self.lower > self.upper:
            raise ValueError("lowerはupper以下にしてください")

    def matches(self, values: np.ndarray) -> np.ndarray:
        """``values``（1次元）の各値が条件に入るか返します。"""

        values = np.asarray(values, dtype=float)
        mask = np.ones(values.shape, dtype=bool)
        if self.lower is not None:
            mask &= values >= self.lower if self.lower_inclusive else values > self.lower
        if self.upper is not None:
            mask &= values <= self.upper if self.upper_inclusive else values < self.upper
        return mask


@dataclass(frozen=True)
class RegionRule:
    """同一region_idの条件をAND結合した方針領域。"""

    region_id: str
    kind: str
    conditions: tuple[RegionCondition, ...]
    strength: int = 3
    enabled: bool = True
    note: str = ""

    def __post_init__(self) -> None:
        if not self.region_id.strip():
            raise ValueError("region_idは空にできません")
        if self.kind not in {KIND_FORBIDDEN, KIND_PREFERRED}:
            raise ValueError(f"kindは{KIND_FORBIDDEN}/{KIND_PREFERRED}のいずれかです")
        if not self.conditions:
            raise ValueError("条件がありません")
        if int(self.strength) != self.strength or not 1 <= int(self.strength) <= 5:
            raise ValueError("strengthは1〜5で指定してください")

    def matches(self, candidates: np.ndarray, parameter_names: Sequence[str]) -> np.ndarray:
        matrix = _as_matrix(candidates, parameter_names)
        positions = {name: index for index, name in enumerate(parameter_names)}
        mask = np.ones(matrix.shape[0], dtype=bool)
        for condition in self.conditions:
            if condition.parameter not in positions:
                # 未定義の入力に対して、推測で制約を適用しない。
                return np.zeros(matrix.shape[0], dtype=bool)
            mask &= condition.matches(matrix[:, positions[condition.parameter]])
        return mask if self.enabled else np.zeros(matrix.shape[0], dtype=bool)


@dataclass(frozen=True)
class PolicyResult:
    """方針適用結果。元配列との対応を保つためmaskも返します。"""

    allowed_mask: np.ndarray
    preferred_multiplier: np.ndarray
    scores: np.ndarray | None = None

    @property
    def forbidden_mask(self) -> np.ndarray:
        return ~self.allowed_mask


@dataclass(frozen=True)
class RegionPolicy:
    rules: tuple[RegionRule, ...] = field(default_factory=tuple)
    max_multiplier: float = DEFAULT_MAX_MULTIPLIER

    def __post_init__(self) -> None:
        if not math.isfinite(self.max_multiplier) or self.max_multiplier < 1:
            raise ValueError("max_multiplierは1以上の有限値にしてください")

    def evaluate(
        self,
        candidates: np.ndarray | Sequence[Mapping[str, float]],
        parameter_names: Sequence[str] | None = None,
        scores: np.ndarray | Sequence[float] | None = None,
    ) -> PolicyResult:
        matrix, names = _normalise_candidates(candidates, parameter_names)
        allowed = np.ones(matrix.shape[0], dtype=bool)
        multiplier = np.ones(matrix.shape[0], dtype=float)
        forbidden = [
            rule
            for rule in self.rules
            if rule.kind == KIND_FORBIDDEN and rule.enabled
        ]
        preferred = [
            rule
            for rule in self.rules
            if rule.kind == KIND_PREFERRED and rule.enabled
        ]
        for rule in forbidden:
            allowed &= ~rule.matches(matrix, names)
        for rule in preferred:
            multiplier *= np.where(
                rule.matches(matrix, names),
                STRENGTH_FACTORS[rule.strength - 1],
                1.0,
            )
        # 禁止領域は優先範囲の加点より常に優先される。
        multiplier = np.where(
            allowed,
            np.minimum(multiplier, self.max_multiplier),
            1.0,
        )
        adjusted = None if scores is None else np.asarray(scores, dtype=float) * multiplier
        return PolicyResult(allowed, multiplier, adjusted)


def _as_matrix(candidates: np.ndarray, parameter_names: Sequence[str]) -> np.ndarray:
    matrix = np.asarray(candidates, dtype=float)
    if matrix.ndim == 1:
        matrix = matrix.reshape(1, -1)
    if matrix.ndim != 2 or matrix.shape[1] != len(parameter_names):
        raise ValueError("candidatesは(候補数, パラメータ数)の配列にしてください")
    return matrix


def _normalise_candidates(
    candidates: np.ndarray | Sequence[Mapping[str, float]],
    parameter_names: Sequence[str] | None,
) -> tuple[np.ndarray, tuple[str, ...]]:
    if isinstance(candidates, np.ndarray) or (
        candidates and not isinstance(candidates[0], Mapping)  # type: ignore[index]
    ):
        if parameter_names is None:
            raise ValueError("配列候補にはparameter_namesが必要です")
        names = tuple(parameter_names)
        return _as_matrix(np.asarray(candidates, dtype=float), names), names
    rows = list(candidates)
    names = tuple(parameter_names or (tuple(rows[0].keys()) if rows else ()))
    matrix = np.array([[float(row[name]) for name in names] for row in rows], dtype=float)
    return matrix.reshape((-1, len(names))) if names else matrix.reshape((0, 0)), names


def _read_rows(path: Path) -> list[dict[str, str]]:
    for encoding in ("utf-8-sig", "cp932"):
        try:
            with path.open("r", encoding=encoding, newline="") as handle:
                reader = csv.DictReader(handle)
                if list(reader.fieldnames or ()) != SEARCH_REGIONS_HEADER:
                    raise ValueError(
                        "search_regions.csvのヘッダーが不正です。必要: "
                        + ",".join(SEARCH_REGIONS_HEADER)
                    )
                return [
                    {key: (value or "").strip() for key, value in row.items() if key}
                    for row in reader
                ]
        except UnicodeDecodeError:
            continue
    raise ValueError(f"CSVを読み込めません: {path}")


def load_search_regions(
    path: str | Path,
    parameter_names: Sequence[str] | None = None,
) -> RegionPolicy:
    """search_regions.csvを読み、region_idごとに条件をAND結合します。"""

    csv_path = Path(path)
    if not csv_path.exists():
        return RegionPolicy()
    grouped: dict[str, list[dict[str, str]]] = {}
    for row in _read_rows(csv_path):
        region_id = row.get("region_id", "")
        if not region_id:
            raise ValueError("search_regions.csv: region_idが空です")
        grouped.setdefault(region_id, []).append(row)
    rules: list[RegionRule] = []
    for region_id, rows in grouped.items():
        kinds = {row.get("kind", "").strip().lower() for row in rows}
        if len(kinds) != 1 or not kinds <= {KIND_FORBIDDEN, KIND_PREFERRED}:
            raise ValueError(f"region_id={region_id}: kindが不正、または混在しています")
        strengths = {int(row.get("strength") or 3) for row in rows}
        if len(strengths) != 1:
            raise ValueError(f"region_id={region_id}: strengthは同じ値にしてください")
        conditions = tuple(
            RegionCondition(
                parameter=row.get("parameter", ""),
                lower=_number(row.get("lower")),
                upper=_number(row.get("upper")),
                lower_inclusive=_boolean(row.get("lower_inclusive"), default=True),
                upper_inclusive=_boolean(row.get("upper_inclusive"), default=True),
            )
            for row in rows
        )
        rules.append(
            RegionRule(
                region_id=region_id,
                kind=next(iter(kinds)),
                conditions=conditions,
                strength=next(iter(strengths)),
                enabled=all(_boolean(row.get("enabled"), default=True) for row in rows),
                note="; ".join(row.get("note", "") for row in rows if row.get("note")),
            )
        )
    policy = RegionPolicy(tuple(rules))
    if parameter_names is not None:
        known = set(parameter_names)
        unknown = sorted(
            {
                condition.parameter
                for rule in policy.rules
                if rule.enabled
                for condition in rule.conditions
                if condition.parameter not in known
            }
        )
        if unknown:
            raise ValueError(
                "search_regions.csvにproblem.csv未定義のparameterがあります: "
                + ", ".join(unknown)
            )
    return policy


def write_search_regions_template(path: str | Path) -> None:
    """新規trialへ、無効状態の記入例を作成します。"""

    csv_path = Path(path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SEARCH_REGIONS_HEADER)
        writer.writeheader()
        writer.writerow(
            {
                "region_id": "R001",
                "kind": KIND_FORBIDDEN,
                "parameter": "input_1",
                "lower": "2000",
                "lower_inclusive": "false",
                "strength": "5",
                "enabled": "false",
                "note": "input_1 > 2000を実験候補から除外",
            }
        )


def filter_experiment_candidates(
    candidates: np.ndarray | Sequence[Mapping[str, float]],
    parameter_names: Sequence[str] | None,
    policy: RegionPolicy,
) -> tuple[np.ndarray, np.ndarray]:
    """禁止領域を除外し、filtered_candidatesと元行のmaskを返します。"""

    matrix, names = _normalise_candidates(candidates, parameter_names)
    result = policy.evaluate(matrix, names)
    return matrix[result.allowed_mask], result.allowed_mask


def apply_preferred_range_bonus(
    scores: np.ndarray | Sequence[float],
    candidates: np.ndarray | Sequence[Mapping[str, float]],
    parameter_names: Sequence[str] | None,
    policy: RegionPolicy,
) -> np.ndarray:
    """推薦スコアへ有界な好ましい範囲倍率を適用する。禁止候補は0点。"""

    matrix, names = _normalise_candidates(candidates, parameter_names)
    result = policy.evaluate(matrix, names, scores)
    assert result.scores is not None
    return np.where(result.allowed_mask, result.scores, 0.0)


__all__ = [
    "KIND_FORBIDDEN", "KIND_PREFERRED", "PolicyResult", "RegionCondition",
    "RegionPolicy", "RegionRule", "apply_preferred_range_bonus",
    "filter_experiment_candidates", "load_search_regions",
    "write_search_regions_template",
]
