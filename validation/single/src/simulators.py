"""工程定義と物理モデルを、共通の検証APIとして扱います。

manifest.jsonとproblem.csvを検査し、候補条件の生成・正解評価・合否判定を行います。
ファイルの読書きはdata_loader.py、定数と変数定義はsettings.pyが担当します。"""

from __future__ import annotations

import hashlib
import importlib
import inspect
import itertools
import json
import math
import re
from pathlib import Path
from typing import Mapping

import numpy as np

from .settings import VALIDATION_ROOT as ROOT, PROBLEM_HEADER, MAX_CANDIDATES, Variable
from .data_loader import read_csv


# ---------------------------------------------------------------------------
# 工程定義の読込と物理モデルの共通API
# ---------------------------------------------------------------------------


class Simulator:
    """1工程の定義・物理モデル・候補生成と合否判定をまとめた窓口です。"""

    def __init__(self, folder: Path):
        """工程定義を検査してから物理モデルを読み込み、入力名の対応を確認します。"""

        self.folder = folder
        self.manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
        self.id = self.manifest["id"]
        if self.id != folder.name or not re.fullmatch(r"[a-z][a-z0-9_]*", self.id):
            raise ValueError(f"Invalid simulator id: {self.id}")
        self.rows = read_csv(folder / "problem.csv")
        if not self.rows or list(self.rows[0]) != PROBLEM_HEADER:
            raise ValueError(f"Invalid problem.csv header: {self.id}")
        self.variables = [Variable.from_row(row) for row in self.rows]
        columns = [v.column for v in self.variables]
        if len(set(columns)) != len(columns) or "experiment_id" in columns:
            raise ValueError("Duplicate or reserved column")
        self.parameters = [v for v in self.variables if v.role == "parameter"]
        self.outputs = [v for v in self.variables if v.role != "parameter"]
        objectives = [v for v in self.outputs if v.role == "objective"]
        self.constraints = [v for v in self.outputs if v.role == "constraint"]
        if not self.parameters or len(objectives) != 1:
            raise ValueError("Need parameters and exactly one objective")
        self.objective = objectives[0]
        for v in self.variables:
            if v.role == "parameter":
                if (v.lower is None or v.upper is None or v.step is None
                        or v.lower >= v.upper or v.step <= 0 or v.direction or v.target is not None):
                    raise ValueError(f"Invalid parameter: {v.column}")
            elif v.role == "objective":
                if v.direction not in ("minimize", "maximize"):
                    raise ValueError(f"Invalid objective: {v.column}")
            elif v.role == "constraint":
                if v.direction not in ("less_equal", "greater_equal") or v.target is None:
                    raise ValueError(f"Invalid constraint: {v.column}")
            elif v.role != "monitor" or v.direction:
                raise ValueError(f"Invalid role: {v.column}")
        # Bound allocation before building any full Cartesian array.
        counts = [v.axis_count() for v in self.parameters]
        if math.prod(counts) > MAX_CANDIDATES:
            raise ValueError("Candidate grid exceeds optimizer limit")
        self.axes = [v.axis() for v in self.parameters]
        if math.prod(len(a) for a in self.axes) > MAX_CANDIDATES:
            raise ValueError("Candidate grid exceeds optimizer limit")
        initial_size = self.manifest.get("initial_design_size")
        if initial_size is not None and (not isinstance(initial_size, int) or initial_size < 2):
            raise ValueError("initial_design_size must be an integer >= 2")
        self.module = importlib.import_module(f"validation.single.{self.id}.physics_model")
        self.input_arguments = self.manifest.get("input_arguments", {})
        if set(self.input_arguments) - set(self.parameter_columns):
            raise ValueError("Unknown column in input_arguments")
        arguments = [self.input_arguments.get(n, n) for n in self.parameter_columns]
        if len(set(arguments)) != len(arguments):
            raise ValueError("Duplicate model input argument")
        inspect.signature(self.module.evaluate_model).bind(**dict.fromkeys(arguments, 0.0))

    @property
    def parameter_columns(self) -> list[str]:
        return [v.column for v in self.parameters]

    @property
    def output_columns(self) -> list[str]:
        return [v.column for v in self.outputs]

    @property
    def experiment_header(self) -> list[str]:
        return ["experiment_id", *self.parameter_columns, *self.output_columns]

    def evaluate(self, conditions: Mapping[str, object]) -> dict[str, np.ndarray]:
        """入力を同じ形の配列にそろえ、範囲と出力形状を検査して真値を返します。"""

        if set(conditions) != set(self.parameter_columns):
            raise ValueError(f"Expected inputs: {', '.join(self.parameter_columns)}")
        arrays = np.broadcast_arrays(*[np.asarray(conditions[n], dtype=float) for n in self.parameter_columns])
        for v, values in zip(self.parameters, arrays):
            if not np.all(np.isfinite(values)) or np.any(values < v.lower) or np.any(values > v.upper):
                raise ValueError(f"{v.column} must be finite and within [{v.lower}, {v.upper}] {v.unit}")
        result = self.module.evaluate_model(**{
            self.input_arguments.get(n, n): value for n, value in zip(self.parameter_columns, arrays)})
        if not set(self.output_columns).issubset(result):
            raise ValueError(f"Missing model outputs: {self.id}")
        checked = {}
        for name, raw in result.items():
            values = np.asarray(raw)
            if values.shape != arrays[0].shape or not np.all(np.isfinite(values)):
                raise ValueError(f"Invalid shape or nonfinite model output: {self.id}/{name}")
            checked[name] = values
        return checked

    def evaluate_points(self, points: np.ndarray) -> dict[str, np.ndarray]:
        """行が条件、列が操作変数の2次元配列をまとめて評価します。"""

        points = np.asarray(points, dtype=float)
        if points.ndim != 2 or points.shape[1] != len(self.parameters):
            raise ValueError("Expected a matrix with one column per parameter")
        return self.evaluate(dict(zip(self.parameter_columns, points.T)))

    def grid(self) -> np.ndarray:
        """各入力軸の直積を、1行1条件の候補配列へ展開します。"""

        grids = np.meshgrid(*self.axes, indexing="ij")
        return np.column_stack([grid.ravel() for grid in grids])

    def initial_design(self, seed: int = 0) -> np.ndarray:
        # Seed 0 preserves the laser's original 27 corner/midpoint conditions.
        """seedと工程定義から初期測定点を選びます。同一seedでは同じ順序を保ちます。"""

        requested = self.manifest.get("initial_design_size")
        if requested is not None:
            grid = self.grid()
            count = min(requested, len(grid))
            normalized = (grid - np.array([v.lower for v in self.parameters])) / np.array([
                v.upper - v.lower for v in self.parameters])
            rng = np.random.default_rng(seed)
            order = np.arange(len(grid)) if seed == 0 else rng.permutation(len(grid))
            candidates = normalized[order]
            first = int(np.argmin(np.sum((candidates - 0.5) ** 2, axis=1))) if seed == 0 else 0
            selected = [first]
            minimum_distance = np.sum((candidates - candidates[first]) ** 2, axis=1)
            minimum_distance[first] = -1.0
            for _ in range(1, count):
                next_index = int(np.argmax(minimum_distance))
                selected.append(next_index)
                distance = np.sum((candidates - candidates[next_index]) ** 2, axis=1)
                minimum_distance = np.minimum(minimum_distance, distance)
                minimum_distance[selected] = -1.0
            return grid[order[np.asarray(selected)]]
        if seed == 0:
            levels = [list(dict.fromkeys([a[0], a[len(a)//2], a[-1]])) for a in self.axes]
            return np.array(list(itertools.product(*levels)))
        count = min(3 ** len(self.axes), math.prod(len(a) for a in self.axes))
        rng = np.random.default_rng(seed)
        return self.grid()[rng.choice(math.prod(len(a) for a in self.axes), count, replace=False)]

    def initial_design_method(self, seed: int = 0) -> str:
        """結果の再現に必要な初期点の選択方式を返します。"""

        if self.manifest.get("initial_design_size") is not None:
            return "deterministic_maximin_grid" if seed == 0 else "seeded_maximin_grid"
        return "corners_midpoints" if seed == 0 else "random_without_replacement"

    def feasible(self, outputs: Mapping[str, np.ndarray]) -> np.ndarray:
        """すべての制約を満たす条件だけをTrueにします。"""

        mask = np.ones(np.asarray(outputs[self.objective.column]).shape, dtype=bool)
        for v in self.constraints:
            value = np.asarray(outputs[v.column])
            mask &= (value <= v.target) if v.direction == "less_equal" else (value >= v.target)
        return mask

    def best_index(self, outputs: Mapping[str, np.ndarray]) -> int | None:
        """制約適合点のうち目的値が最良の行番号を返します。適合点なしはNoneです。"""

        indices = np.flatnonzero(self.feasible(outputs))
        if not len(indices):
            return None
        sign = 1 if self.objective.direction == "maximize" else -1
        return int(indices[np.argmax(sign * np.asarray(outputs[self.objective.column]).ravel()[indices])])

    def experiment_rows(self, points, outputs, prefix: str) -> list[dict]:
        """測定ID、操作条件、出力を実験CSVの列に合わせて並べます。"""

        return [{"experiment_id": f"{prefix}_{i+1:04d}",
                 **{n: float(points[i, j]) for j, n in enumerate(self.parameter_columns)},
                 **{n: float(outputs[n][i]) for n in self.output_columns}}
                for i in range(len(points))]

    def provenance(self) -> dict:
        # Normalize CRLF: same source after a Windows Git checkout has the same identity.
        """物理モデルと定義ファイルの版・ハッシュを再現用に記録します。"""

        files = [p for p in sorted(self.folder.rglob("*.py"))
                 if not p.name.startswith("test") and "tests" not in p.relative_to(self.folder).parts]
        files += [self.folder / "problem.csv", self.folder / "manifest.json"]
        return {"id": self.id, "model_version": self.manifest["model_version"],
                "sha256_lf": {p.relative_to(self.folder).as_posix(): hashlib.sha256(
                    p.read_bytes().replace(b"\r\n", b"\n")).hexdigest() for p in files}}


# ---------------------------------------------------------------------------
# 工程の登録・取得と測定ノイズ
# ---------------------------------------------------------------------------


def available() -> list[str]:
    """定義JSON、問題CSV、物理モデルがそろった工程IDを列挙します。"""

    return sorted(p.parent.name for p in ROOT.glob("*/manifest.json")
                  if (p.parent / "problem.csv").is_file() and (p.parent / "physics_model.py").is_file())


def load(simulator_id: str) -> Simulator:
    """登録済みの工程IDを確認し、検証用の共通窓口を生成します。"""

    if simulator_id not in available():
        raise ValueError(f"Unknown simulator {simulator_id!r}; available: {', '.join(available())}")
    return Simulator(ROOT / simulator_id)


def measure(truth: Mapping[str, np.ndarray], scales: Mapping[str, float], noise: float,
            rng: np.random.Generator) -> dict[str, np.ndarray]:
    """真値とは別の配列に、出力幅×noiseを標準偏差とする独立正規ノイズを加えます。
    
    測定器の読み値を表すため、物理範囲への丸めやクリップは行いません。"""
    if not math.isfinite(noise) or noise < 0:
        raise ValueError("Noise fraction must be finite and nonnegative")
    return {key: np.asarray(values, dtype=float).copy() + (
        rng.normal(0, noise * scales[key], size=np.shape(values)) if noise else 0)
        for key, values in truth.items() if key in scales}
