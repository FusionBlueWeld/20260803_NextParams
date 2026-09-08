"""Registry for standalone access to each stage oracle."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import ModuleType

import numpy as np
from numpy.typing import ArrayLike

from .functional_coating.coating import physics_model as coating
from .functional_coating.curing import physics_model as curing
from .functional_coating.drying import physics_model as drying


@dataclass(frozen=True)
class Stage:
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
        if set(conditions) != set(self.input_bounds):
            raise ValueError(f"Expected inputs: {', '.join(self.input_bounds)}")
        return self.module.evaluate_model(**conditions)


STAGES = {
    "coating": Stage("coating", "塗工", coating),
    "drying": Stage("drying", "乾燥", drying),
    "curing": Stage("curing", "熱硬化", curing),
}


def available() -> list[str]:
    return sorted(STAGES)


def load(stage_id: str) -> Stage:
    try:
        return STAGES[stage_id]
    except KeyError as error:
        raise ValueError(
            f"Unknown stage {stage_id!r}; available: {', '.join(available())}"
        ) from error
