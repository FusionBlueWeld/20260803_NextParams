"""塗工・乾燥・硬化の物理モデルを共通の入口から利用する機能。

Stageは操作条件と流入状態の名前・範囲を公開します。
工程同士の接続や最適化は行わず、1工程の入力確認と評価を担当します。"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import ModuleType

import numpy as np
from numpy.typing import ArrayLike

from ..functional_coating.coating import physics_model as coating
from ..functional_coating.curing import physics_model as curing
from ..functional_coating.drying import physics_model as drying


@dataclass(frozen=True)
class Stage:
    """1工程の操作条件・流入状態と物理モデルへの参照を保持します。"""

    id: str
    name: str
    module: ModuleType

    @property
    def input_bounds(self) -> dict[str, tuple[float, float]]:
        return dict(self.module.INPUT_BOUNDS)

    @property
    def control_names(self) -> tuple[str, ...]:
        return tuple(self.module.CONTROL_BOUNDS)

    @property
    def incoming_state_names(self) -> tuple[str, ...]:
        return tuple(self.module.INCOMING_STATE_BOUNDS)

    def evaluate(self, conditions: Mapping[str, ArrayLike]) -> dict[str, np.ndarray]:
        """入力名が工程の契約と一致することを確認して物理モデルを呼びます。"""

        if set(conditions) != set(self.input_bounds):
            raise ValueError(f"Expected inputs: {', '.join(self.input_bounds)}")
        return self.module.evaluate_model(**conditions)


STAGES = {
    "coating": Stage("coating", "塗工", coating),
    "drying": Stage("drying", "乾燥", drying),
    "curing": Stage("curing", "熱硬化", curing),
}


def available() -> list[str]:
    """登録済みの工程IDを、表示順が安定するようソートして返します。"""

    return sorted(STAGES)


def load(stage_id: str) -> Stage:
    """工程IDから物理モデルを取得し、未知のIDなら利用可能な一覧を示します。"""

    try:
        return STAGES[stage_id]
    except KeyError as error:
        raise ValueError(
            f"Unknown stage {stage_id!r}; available: {', '.join(available())}"
        ) from error
