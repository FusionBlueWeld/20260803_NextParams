"""NN予測空間をrunごとのCSVとして保存します。"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from ..data_loader import write_csv_atomic
from ..settings import NN_RESPONSE_FILE_PREFIX, NN_SEED, ProblemDefinition
from ..validation import ExperimentData
from ..preprocessing import PreparedData


def next_run_id(output_directory: Path) -> str:
    """既存のNN予測空間を調べ、次の連番を返します。"""

    pattern = re.compile(rf"^{re.escape(NN_RESPONSE_FILE_PREFIX)}(\d{{4}})\.csv$")
    numbers: list[int] = []
    if output_directory.is_dir():
        for path in output_directory.iterdir():
            match = pattern.fullmatch(path.name)
            if match:
                numbers.append(int(match.group(1)))
    return f"run_{max(numbers, default=0) + 1:04d}"


def response_space_path(output_directory: Path, run_id: str) -> Path:
    """run番号からNN予測空間の保存先を作ります。"""

    number = run_id.removeprefix("run_")
    return output_directory / f"{NN_RESPONSE_FILE_PREFIX}{number}.csv"


def write_response_space(
    path: Path,
    run_id: str,
    experiments: ExperimentData,
    prepared: PreparedData,
    problem: ProblemDefinition,
    raw_grid: np.ndarray,
    predictions: np.ndarray,
) -> None:
    """全グリッド点と全結果変数のNN予測を1つのCSVへ保存します。"""

    header = ["run_id", "data_rows", "unique_conditions", "nn_seed"]
    header.extend(item.column for item in problem.parameters)
    header.extend(f"{item.column}_nn_pred" for item in problem.result_variables)

    rows: list[dict[str, object]] = []
    for row_index in range(len(raw_grid)):
        row: dict[str, object] = {
            "run_id": run_id,
            "data_rows": experiments.row_count,
            "unique_conditions": prepared.unique_condition_count,
            "nn_seed": NN_SEED,
        }
        for column_index, parameter in enumerate(problem.parameters):
            row[parameter.column] = f"{raw_grid[row_index, column_index]:.10g}"
        for column_index, result in enumerate(problem.result_variables):
            row[f"{result.column}_nn_pred"] = f"{predictions[row_index, column_index]:.10g}"
        rows.append(row)
    write_csv_atomic(path, header, rows)
