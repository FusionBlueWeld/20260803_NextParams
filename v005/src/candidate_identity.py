"""離散軸の番号で条件を識別し、旧CSVの10桁丸めだけを復元します。

距離や小数桁の一律丸めで比較すると、細かい別水準や異なる流入状態まで
同一条件になります。この実装は全格子を作らず、各軸の近傍だけを調べます。
10桁表現が隣の水準と衝突する場合は復元しません。既にCSVで情報が失われた
衝突値から元の水準を推定することはできないため、新規CSVは17桁保存が必要です。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import math

import numpy as np

from .settings import ProblemDefinition, VariableDefinition
from .validation import axis_count


@dataclass(frozen=True)
class _AxisIdentity:
    lower: float
    upper: float
    step: float
    count: int

    @classmethod
    def from_variable(cls, variable: VariableDefinition) -> "_AxisIdentity":
        if variable.lower is None or variable.upper is None or variable.step is None:
            raise ValueError("条件の同一性には入力の上下限と刻みが必要です。")
        return cls(variable.lower, variable.upper, variable.step,
                   axis_count(variable.lower, variable.upper, variable.step))

    def value(self, index: int) -> float:
        return self.upper if index == self.count - 1 else self.lower + self.step * index

    def grid_index(self, value: float) -> int | None:
        """正確な格子値または一意な旧 .10g 表現だけを軸番号へ変換します。"""
        position = (value - self.lower) / self.step
        if not math.isfinite(position):
            return None
        nearest = int(round(position))
        possible = {max(0, min(self.count - 1, nearest + offset))
                    for offset in (-1, 0, 1)}
        possible.add(self.count - 1)  # 上限だけ短い最後の区間にも対応。
        for index in sorted(possible):
            if self.value(index) == value:
                return index

        for index in sorted(possible):
            serialized = float(format(self.value(index), ".10g"))
            if serialized != value:
                continue
            # .10gで潰れる隣接水準は、範囲距離の許容差を使って統合しない。
            if any(float(format(self.value(other), ".10g")) == serialized
                   for other in (index - 1, index + 1) if 0 <= other < self.count):
                continue
            return index
        return None


class CandidateIdentity:
    """候補除外と反復集約で共通の識別を使います。

    ``fixed_context`` の列は格子への復元対象にせず、実際の流入値を厳密に
    識別します。非格子条件も厳密なfloat値のまま保持します。
    """

    def __init__(self, problem: ProblemDefinition,
                 fixed_context: Mapping[str, float] | None = None):
        context = {} if fixed_context is None else fixed_context
        self._axes = [None if parameter.column in context else
                      _AxisIdentity.from_variable(parameter)
                      for parameter in problem.parameters]

    def _array(self, rows: Sequence[Sequence[float]] | np.ndarray) -> np.ndarray:
        result = np.asarray(rows, dtype=float)
        if result.size == 0:
            result = result.reshape(0, len(self._axes))
        if result.ndim != 2 or result.shape[1] != len(self._axes):
            raise ValueError("条件配列の列数がproblemの入力数と一致しません。")
        if not np.all(np.isfinite(result)):
            raise ValueError("条件の同一性には有限の入力値が必要です。")
        return result

    def _row(self, row: np.ndarray) -> tuple[tuple[object, ...], np.ndarray]:
        key: list[object] = []
        canonical = row.copy()
        for column, (value, axis) in enumerate(zip(row, self._axes)):
            index = None if axis is None else axis.grid_index(float(value))
            if index is None:
                # float.hexは丸めを追加せず、-0と0は同じ数値として扱います。
                key.append(("exact", float(0.0 if value == 0.0 else value).hex()))
                if value == 0.0:
                    canonical[column] = 0.0
            else:
                key.append(("grid", index))
                canonical[column] = axis.value(index)
        return tuple(key), canonical

    def _canonical_chunk(self, source: np.ndarray,
                         caches: list[dict[float, float]]) -> np.ndarray:
        result = source.copy(order="C")
        for column, axis in enumerate(self._axes):
            if axis is None:
                continue
            # 4水準を65,536行ごとに再判定せず、そのチャンク内の固有値だけ判定。
            values, inverse = np.unique(source[:, column], return_inverse=True)
            replacements = np.empty(len(values), dtype=float)
            cache = caches[column]
            for offset, value in enumerate(values):
                scalar = float(value)
                if scalar not in cache:
                    grid_index = axis.grid_index(scalar)
                    replacement = scalar if grid_index is None else axis.value(grid_index)
                    # 連続値のデータでもキャッシュが無制限に増えないようにする。
                    if len(cache) >= 256:
                        cache.clear()
                    cache[scalar] = replacement
                replacements[offset] = cache[scalar]
            result[:, column] = replacements[inverse]
        result[result == 0.0] = 0.0  # byte識別でも-0と0を同じ条件として扱う。
        return result

    def _chunk_size(self) -> int:
        # キー用コピー＋np.isinの作業配列を小さく保つ。候補全件のキーは作らない。
        return min(4096, max(1, (4 * 1024 * 1024) // (8 * max(len(self._axes), 1))))

    def keys(self, rows: Sequence[Sequence[float]] | np.ndarray) -> list[tuple[object, ...]]:
        return [self._row(row)[0] for row in self._array(rows)]

    def canonicalize(self, rows: Sequence[Sequence[float]] | np.ndarray) -> np.ndarray:
        """旧CSVの一意な格子値を復元したコピー。元の実測CSVは変更しません。"""
        source = self._array(rows)
        result = np.empty_like(source)
        caches: list[dict[float, float]] = [{} for _ in self._axes]
        chunk_size = self._chunk_size()
        for start in range(0, len(source), chunk_size):
            stop = min(start + chunk_size, len(source))
            result[start:stop] = self._canonical_chunk(source[start:stop], caches)
        return result

    def group_indices(self, rows: Sequence[Sequence[float]] | np.ndarray) -> dict[tuple[object, ...], list[int]]:
        groups: dict[tuple[object, ...], list[int]] = {}
        for index, key in enumerate(self.keys(rows)):
            groups.setdefault(key, []).append(index)
        return groups

    def measured_mask(self, candidates: np.ndarray,
                      observed: Sequence[Sequence[float]] | np.ndarray) -> np.ndarray:
        source = self._array(candidates)
        measured = np.ascontiguousarray(self.canonicalize(observed))
        result = np.zeros(len(source), dtype=bool)
        if not len(measured) or not len(source):
            return result
        # 復元済みfloatを行の固定長byte列で比較するため、確率的なhash衝突がない。
        row_dtype = np.dtype((np.void, source.shape[1] * source.dtype.itemsize))
        measured_keys = measured.view(row_dtype).ravel()
        caches: list[dict[float, float]] = [{} for _ in self._axes]
        chunk_size = self._chunk_size()
        for start in range(0, len(source), chunk_size):
            stop = min(start + chunk_size, len(source))
            canonical = self._canonical_chunk(source[start:stop], caches)
            result[start:stop] = np.isin(canonical.view(row_dtype).ravel(), measured_keys)
        return result

    def exclude_observed(self, candidates: np.ndarray,
                         observed: Sequence[Sequence[float]] | np.ndarray) -> np.ndarray:
        source = self._array(candidates)
        return source[~self.measured_mask(source, observed)]
