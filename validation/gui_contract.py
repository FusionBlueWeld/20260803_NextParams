"""物理シミュレータGUIの画面定義。物理式は各工程のモジュールが担当します。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping


@dataclass(frozen=True)
class FieldSpec:
    name: str
    label: str
    unit: str
    lower: float | None = None
    upper: float | None = None
    default: float | None = None
    group: str = ""
    readonly: bool = False

    def initial_value(self) -> float:
        if self.default is not None:
            return self.default
        if self.lower is not None and self.upper is not None:
            return (self.lower + self.upper) / 2.0
        return 0.0


@dataclass(frozen=True)
class SimulatorSpec:
    id: str
    label: str
    inputs: tuple[FieldSpec, ...]
    outputs: tuple[FieldSpec, ...]
    evaluate: Callable[[Mapping[str, float]], Mapping[str, object]]
    noise_scales: Mapping[str, float]
    description: str = ""
    image_id: str | None = None
    # node: (ID, 表示名, 横位置0..1, 縦位置0..1)。辺は工程IDの組です。
    diagram_nodes: tuple[tuple[str, str, float, float], ...] = ()
    diagram_edges: tuple[tuple[str, str], ...] = ()
    observe: Callable[[Mapping[str, object], object], Mapping[str, object]] | None = None
    noise_label: str = "測定ノイズあり（出力レンジの2%）"
    allow_noise: bool = True
