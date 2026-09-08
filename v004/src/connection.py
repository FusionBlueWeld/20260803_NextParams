"""v004の任意接続設定を検査し、流入状態を探索軸から固定します。"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .settings import ProblemDefinition
from .validation import UserInputError


CONNECTION_FILE_NAME = "stage_connection.json"
SCHEMA_VERSION = "1.0"


@dataclass(frozen=True)
class StageConnection:
    stage_id: str
    controls: tuple[str, ...]
    incoming_context: tuple[str, ...]
    connector_outputs: tuple[str, ...]
    current_incoming_context: dict[str, float]


def load_stage_connection(path: Path, problem: ProblemDefinition) -> StageConnection | None:
    """設定がなければ互換経路、あれば役割と必須流入値を厳密に検査します。"""

    if not path.exists():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as error:
        raise UserInputError(f"{CONNECTION_FILE_NAME}を読み取れません: {error}") from error
    if value.get("schema_version") != SCHEMA_VERSION:
        raise UserInputError(f"{CONNECTION_FILE_NAME}のschema_versionは{SCHEMA_VERSION}にしてください。")
    stage_id = str(value.get("stage_id", "")).strip()
    if not stage_id:
        raise UserInputError(f"{CONNECTION_FILE_NAME}にstage_idが必要です。")
    controls = tuple(value.get("controls", []))
    incoming = tuple(value.get("incoming_context", []))
    outputs = tuple(value.get("connector_outputs", []))
    parameter_names = [item.column for item in problem.parameters]
    result_names = [item.column for item in problem.result_variables]
    if len(set((*controls, *incoming))) != len(parameter_names) or set((*controls, *incoming)) != set(parameter_names):
        raise UserInputError("controlsとincoming_contextはproblem.csvのparameterを重複なく全て分類してください。")
    if not controls:
        raise UserInputError("controlsを1個以上指定してください。")
    unknown_outputs = sorted(set(outputs) - set(result_names))
    if unknown_outputs:
        raise UserInputError("connector_outputsに未定義の出力があります: " + ", ".join(unknown_outputs))
    raw_context = value.get("current_incoming_context", {})
    missing = sorted(set(incoming) - set(raw_context))
    extra = sorted(set(raw_context) - set(incoming))
    if missing or extra:
        details = []
        if missing:
            details.append("不足=" + ", ".join(missing))
        if extra:
            details.append("未定義=" + ", ".join(extra))
        raise UserInputError("current_incoming_contextが一致しません（" + "; ".join(details) + "）。")
    context: dict[str, float] = {}
    definitions = {item.column: item for item in problem.parameters}
    for name in incoming:
        try:
            number = float(raw_context[name])
        except (TypeError, ValueError) as error:
            raise UserInputError(f"流入状態{name}は有限の数値にしてください。") from error
        definition = definitions[name]
        if not np.isfinite(number) or number < definition.lower or number > definition.upper:
            raise UserInputError(f"流入状態{name}={number}は学習適用範囲[{definition.lower}, {definition.upper}]外です。")
        context[name] = number
    return StageConnection(stage_id, controls, incoming, outputs, context)


def fixed_context_grid(grid: np.ndarray, problem: ProblemDefinition, connection: StageConnection | None) -> np.ndarray:
    """接続案件では流入状態を現在値へ固定し、操作条件だけを探索可能にします。"""

    if connection is None:
        return grid
    names = [item.column for item in problem.parameters]
    result = np.array(grid, copy=True)
    for name, value in connection.current_incoming_context.items():
        result[:, names.index(name)] = value
    return np.unique(result, axis=0)


def current_context_observation_mask(
    raw_parameters: np.ndarray,
    problem: ProblemDefinition,
    connection: StageConnection | None,
) -> np.ndarray:
    """現在の流入状態と一致する実測だけを、局所改善の比較基準にします。"""

    mask = np.ones(len(raw_parameters), dtype=bool)
    if connection is None:
        return mask
    names = [item.column for item in problem.parameters]
    for name, value in connection.current_incoming_context.items():
        column = names.index(name)
        mask &= np.isclose(raw_parameters[:, column], value, rtol=1e-9, atol=1e-12)
    return mask
