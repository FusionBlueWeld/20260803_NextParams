"""Registry, common input/output contract and deterministic benchmark designs.

Adding a process only requires a subfolder with manifest.json, problem.csv and
physics_model.py. Optimizers receive measured CSV rows, never the oracle itself.
"""

from __future__ import annotations

import csv
import hashlib
import importlib
import inspect
import itertools
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np

ROOT = Path(__file__).resolve().parent
PROBLEM_HEADER = [
    "column", "display_name", "unit", "role", "direction",
    "lower", "upper", "step", "target",
]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, header: list[str], rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=header)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


@dataclass(frozen=True)
class Variable:
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
        if set(row) != set(PROBLEM_HEADER) or any(v is None for v in row.values()):
            raise ValueError(f"Wrong cell count in problem.csv: {row}")
        values = dict(row)
        for field in ("lower", "upper", "step", "target"):
            values[field] = float(row[field]) if row[field] else None
            if values[field] is not None and not math.isfinite(values[field]):
                raise ValueError(f"Nonfinite {field}: {row['column']}")
        return cls(**values)

    def axis(self) -> np.ndarray:
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
        intervals = (self.upper - self.lower) / self.step
        if not math.isfinite(intervals) or intervals > 200_000:
            raise ValueError("Candidate axis exceeds benchmark limit")
        count = int(math.floor(intervals + 1e-12)) + 1
        last = self.lower + self.step * (count - 1)
        return count + int(self.upper - last > max(1e-10, abs(self.upper) * 1e-12))


class Simulator:
    def __init__(self, folder: Path):
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
        if math.prod(counts) > 200_000:
            raise ValueError("Candidate grid exceeds optimizer limit")
        self.axes = [v.axis() for v in self.parameters]
        if math.prod(len(a) for a in self.axes) > 200_000:
            raise ValueError("Candidate grid exceeds optimizer limit")
        initial_size = self.manifest.get("initial_design_size")
        if initial_size is not None and (not isinstance(initial_size, int) or initial_size < 2):
            raise ValueError("initial_design_size must be an integer >= 2")
        self.module = importlib.import_module(f"validation.{self.id}.physics_model")
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
        points = np.asarray(points, dtype=float)
        if points.ndim != 2 or points.shape[1] != len(self.parameters):
            raise ValueError("Expected a matrix with one column per parameter")
        return self.evaluate(dict(zip(self.parameter_columns, points.T)))

    def grid(self) -> np.ndarray:
        grids = np.meshgrid(*self.axes, indexing="ij")
        return np.column_stack([grid.ravel() for grid in grids])

    def initial_design(self, seed: int = 0) -> np.ndarray:
        # Seed 0 preserves the laser's original 27 corner/midpoint conditions.
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
        if self.manifest.get("initial_design_size") is not None:
            return "deterministic_maximin_grid" if seed == 0 else "seeded_maximin_grid"
        return "corners_midpoints" if seed == 0 else "random_without_replacement"

    def feasible(self, outputs: Mapping[str, np.ndarray]) -> np.ndarray:
        mask = np.ones(np.asarray(outputs[self.objective.column]).shape, dtype=bool)
        for v in self.constraints:
            value = np.asarray(outputs[v.column])
            mask &= (value <= v.target) if v.direction == "less_equal" else (value >= v.target)
        return mask

    def best_index(self, outputs: Mapping[str, np.ndarray]) -> int | None:
        indices = np.flatnonzero(self.feasible(outputs))
        if not len(indices):
            return None
        sign = 1 if self.objective.direction == "maximize" else -1
        return int(indices[np.argmax(sign * np.asarray(outputs[self.objective.column]).ravel()[indices])])

    def experiment_rows(self, points, outputs, prefix: str) -> list[dict]:
        return [{"experiment_id": f"{prefix}_{i+1:04d}",
                 **{n: float(points[i, j]) for j, n in enumerate(self.parameter_columns)},
                 **{n: float(outputs[n][i]) for n in self.output_columns}}
                for i in range(len(points))]

    def provenance(self) -> dict:
        # Normalize CRLF: same source after a Windows Git checkout has the same identity.
        files = [p for p in sorted(self.folder.rglob("*.py"))
                 if not p.name.startswith("test") and "tests" not in p.relative_to(self.folder).parts]
        files += [self.folder / "problem.csv", self.folder / "manifest.json"]
        return {"id": self.id, "model_version": self.manifest["model_version"],
                "sha256_lf": {p.relative_to(self.folder).as_posix(): hashlib.sha256(
                    p.read_bytes().replace(b"\r\n", b"\n")).hexdigest() for p in files}}


def available() -> list[str]:
    return sorted(p.parent.name for p in ROOT.glob("*/manifest.json")
                  if (p.parent / "problem.csv").is_file() and (p.parent / "physics_model.py").is_file())


def load(simulator_id: str) -> Simulator:
    if simulator_id not in available():
        raise ValueError(f"Unknown simulator {simulator_id!r}; available: {', '.join(available())}")
    return Simulator(ROOT / simulator_id)


def measure(truth: Mapping[str, np.ndarray], scales: Mapping[str, float], noise: float,
            rng: np.random.Generator) -> dict[str, np.ndarray]:
    """Independent additive Gaussian inspection error, sigma=noise*grid output span.

    No clipping or rounding: noisy instrument readings can exceed physical bounds.
    Truth and noisy observations are recorded separately by the benchmark runner.
    """
    if not math.isfinite(noise) or noise < 0:
        raise ValueError("Noise fraction must be finite and nonnegative")
    return {key: np.asarray(values, dtype=float).copy() + (
        rng.normal(0, noise * scales[key], size=np.shape(values)) if noise else 0)
        for key, values in truth.items() if key in scales}
