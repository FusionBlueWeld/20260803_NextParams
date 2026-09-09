"""検証用のフォルダ設定。実行場所に依存せず、同じモデルと結果先を参照します。"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path

import numpy as np


# srcの1階層上が検証領域、validationの1階層上がプロジェクト直下です。
VALIDATION_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = VALIDATION_ROOT.parents[1]
RESULTS_ROOT = VALIDATION_ROOT / "results"


# 候補の直積を作る前に件数を検査し、過大な配列確保を防ぎます。
MAX_CANDIDATES = 200_000

PROBLEM_HEADER = [
    "column", "display_name", "unit", "role", "direction",
    "lower", "upper", "step", "target",
]


@dataclass(frozen=True)
class Variable:
    """problem.csvの1行。入力軸と目的・制約・監視出力の定義を保持します。"""

    column: str
    display_name: str
    unit: str
    role: str
    direction: str
    lower: float | None
    upper: float | None
    step: float | None
    target: float | None

    @classmethod
    def from_row(cls, row: dict[str, str]) -> Variable:
        """セル数と有限値を検査し、空の数値欄をNoneへ変換します。"""

        if set(row) != set(PROBLEM_HEADER) or any(v is None for v in row.values()):
            raise ValueError(f"Wrong cell count in problem.csv: {row}")
        values = dict(row)
        for field in ("lower", "upper", "step", "target"):
            values[field] = float(row[field]) if row[field] else None
            if values[field] is not None and not math.isfinite(values[field]):
                raise ValueError(f"Nonfinite {field}: {row['column']}")
        return cls(**values)

    def axis(self) -> np.ndarray:
        """下限から上限までの候補を作ります。刻みで割り切れない上限も含めます。"""

        if self.role != "parameter":
            raise ValueError("Only parameters have candidate axes")
        count = int(math.floor((self.upper - self.lower) / self.step + 1e-12)) + 1
        values = self.lower + self.step * np.arange(count)
        if self.upper - values[-1] > max(1e-10, abs(self.upper) * 1e-12):
            values = np.append(values, self.upper)
        else:
            values[-1] = self.upper
        return values

    def axis_count(self) -> int:
        """配列を確保する前に、端点を含めた候補件数を求めます。"""

        intervals = (self.upper - self.lower) / self.step
        if not math.isfinite(intervals) or intervals > MAX_CANDIDATES:
            raise ValueError("Candidate axis exceeds benchmark limit")
        count = int(math.floor(intervals + 1e-12)) + 1
        last = self.lower + self.step * (count - 1)
        return count + int(self.upper - last > max(1e-10, abs(self.upper) * 1e-12))

