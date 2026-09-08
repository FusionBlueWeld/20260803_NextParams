
"""problem.csvとexperiments.csvを検査する機能。

入力に問題がある場合は、計算途中で難しい例外を出すのではなく、利用者が
修正できる日本語メッセージへ変換します。
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path

from .data_loader import read_csv
from .settings import MAX_CANDIDATES, ProblemDefinition, VariableDefinition


PROBLEM_HEADER = [
    "column",
    "display_name",
    "unit",
    "role",
    "direction",
    "lower",
    "upper",
    "step",
    "target",
]


class UserInputError(Exception):
    """利用者がCSVやオプションを修正すれば解決できるエラーです。"""


@dataclass(frozen=True)
class ExperimentData:
    """検査済みのexperiments.csvを保持します。"""

    experiment_ids: list[str]
    parameter_rows: list[list[float]]
    result_rows: dict[str, list[float]]

    @property
    def row_count(self) -> int:
        return len(self.experiment_ids)


# ---------------------------------------------------------------------------
# 共通の小さな検査機能
# ---------------------------------------------------------------------------


def validate_trial_name(trial_name: str) -> None:
    """trial名にパス記号などが混ざることを防ぎます。"""

    if not re.fullmatch(r"trial_[A-Za-z0-9_-]+", trial_name):
        raise UserInputError(
            "trial名は 'trial_' で始め、半角英数字、_、- だけで指定してください。\n"
            "例: trial_000"
        )


def parse_optional_float(value: str, label: str, row_number: int) -> float | None:
    """空欄を許す数値列を読み込みます。"""

    if value == "":
        return None
    try:
        number = float(value)
    except ValueError as error:
        raise UserInputError(
            f"problem.csvの{row_number}行目に問題があります。\n"
            f"{label}には数値を入力してください。現在の値: {value}"
        ) from error
    if not math.isfinite(number):
        raise UserInputError(
            f"problem.csvの{row_number}行目に問題があります。\n"
            f"{label}には有限の数値を入力してください。"
        )
    return number


def axis_count(lower: float, upper: float, step: float) -> int:
    """上下限と刻みから、探索軸に含まれる候補数を数えます。"""

    regular_count = int(math.floor((upper - lower) / step)) + 1
    last_regular_value = lower + step * (regular_count - 1)
    if upper - last_regular_value > max(1e-10, abs(upper) * 1e-12):
        return regular_count + 1
    return regular_count


# ---------------------------------------------------------------------------
# problem.csvの検査
# ---------------------------------------------------------------------------


def load_and_validate_problem(path: Path) -> ProblemDefinition:
    """problem.csvを読み込み、探索に使える定義へ変換します。"""

    if not path.exists():
        raise UserInputError(f"problem.csvが見つかりません。\n確認場所: {path}")

    try:
        header, rows = read_csv(path)
    except UnicodeDecodeError as error:
        raise UserInputError(
            "problem.csvの文字コードを読み取れませんでした。\n"
            "UTF-8またはWindowsのCSV形式で保存してください。"
        ) from error

    if header != PROBLEM_HEADER:
        raise UserInputError(
            "problem.csvのヘッダーが想定と一致しません。\n"
            f"必要: {','.join(PROBLEM_HEADER)}\n"
            f"現在: {','.join(header)}"
        )
    if not rows:
        raise UserInputError("problem.csvに設定行がありません。")

    definitions: list[VariableDefinition] = []
    used_columns: set[str] = set()

    for index, row in enumerate(rows, start=2):
        column = row["column"].strip()
        role = row["role"].strip().lower()
        direction = row["direction"].strip().lower()

        if not column:
            raise UserInputError(f"problem.csvの{index}行目でcolumnが空欄です。")
        if column == "experiment_id":
            raise UserInputError("experiment_idはシステム予約列のため、columnには指定できません。")
        if column in used_columns:
            raise UserInputError(f"problem.csvでcolumn '{column}' が重複しています。")
        used_columns.add(column)

        lower = parse_optional_float(row["lower"], "lower", index)
        upper = parse_optional_float(row["upper"], "upper", index)
        step = parse_optional_float(row["step"], "step", index)
        target = parse_optional_float(row["target"], "target", index)

        if role == "parameter":
            if lower is None or upper is None or step is None:
                raise UserInputError(
                    f"problem.csvの{index}行目に問題があります。\n"
                    "parameterにはlower、upper、stepが必要です。"
                )
            if lower >= upper:
                raise UserInputError(
                    f"problem.csvの{index}行目に問題があります。\n"
                    f"{column}のlowerはupperより小さくしてください。"
                )
            if step <= 0:
                raise UserInputError(
                    f"problem.csvの{index}行目に問題があります。\n"
                    f"{column}のstepは0より大きくしてください。"
                )
            if direction or target is not None:
                raise UserInputError(
                    f"problem.csvの{index}行目に問題があります。\n"
                    "parameterのdirectionとtargetは空欄にしてください。"
                )
        elif role == "objective":
            if direction not in {"minimize", "maximize"}:
                raise UserInputError(
                    f"problem.csvの{index}行目に問題があります。\n"
                    "objectiveのdirectionはminimizeまたはmaximizeにしてください。"
                )
        elif role == "constraint":
            if direction not in {"greater_equal", "less_equal"} or target is None:
                raise UserInputError(
                    f"problem.csvの{index}行目に問題があります。\n"
                    "constraintにはgreater_equalまたはless_equalとtargetが必要です。"
                )
        elif role == "monitor":
            if direction or target is not None:
                raise UserInputError(
                    f"problem.csvの{index}行目に問題があります。\n"
                    "monitorのdirectionとtargetは空欄にしてください。"
                )
        else:
            raise UserInputError(
                f"problem.csvの{index}行目に問題があります。\n"
                "roleはparameter、objective、constraint、monitorのいずれかです。"
            )

        definitions.append(
            VariableDefinition(
                column=column,
                display_name=row["display_name"].strip() or column,
                unit=row["unit"].strip(),
                role=role,
                direction=direction,
                lower=lower,
                upper=upper,
                step=step,
                target=target,
            )
        )

    parameters = [item for item in definitions if item.role == "parameter"]
    objectives = [item for item in definitions if item.role == "objective"]
    constraints = [item for item in definitions if item.role == "constraint"]
    monitors = [item for item in definitions if item.role == "monitor"]

    if not parameters:
        raise UserInputError("problem.csvにはparameterを1個以上指定してください。")
    if len(objectives) != 1:
        raise UserInputError(
            "v004ではobjectiveを1個だけ指定してください。\n"
            f"現在の個数: {len(objectives)}"
        )

    candidate_count = 1
    for parameter in parameters:
        assert parameter.lower is not None
        assert parameter.upper is not None
        assert parameter.step is not None
        candidate_count *= axis_count(parameter.lower, parameter.upper, parameter.step)

    if candidate_count > MAX_CANDIDATES:
        raise UserInputError(
            f"探索候補が{candidate_count:,}件になります。v004の上限は{MAX_CANDIDATES:,}件です。\n"
            "探索範囲を狭くするか、stepを大きくしてください。"
        )

    return ProblemDefinition(
        parameters=parameters,
        objective=objectives[0],
        constraints=constraints,
        monitors=monitors,
        result_order=[item for item in definitions if item.role != "parameter"],
        candidate_count=candidate_count,
    )


# ---------------------------------------------------------------------------
# experiments.csvの検査
# ---------------------------------------------------------------------------


def load_and_validate_experiments(path: Path, problem: ProblemDefinition) -> ExperimentData:
    """experiments.csvを読み込み、全数値を検査します。"""

    if not path.exists():
        raise UserInputError(
            f"experiments.csvが見つかりません。\n"
            f"先に python main.py --prepare <trial名> を実行してください。\n"
            f"確認場所: {path}"
        )

    try:
        header, rows = read_csv(path)
    except UnicodeDecodeError as error:
        raise UserInputError(
            "experiments.csvの文字コードを読み取れませんでした。\n"
            "UTF-8またはWindowsのCSV形式で保存してください。"
        ) from error

    expected_header = problem.experiment_header
    if header != expected_header:
        raise UserInputError(
            "experiments.csvのヘッダーがproblem.csvと一致しません。\n"
            f"必要: {','.join(expected_header)}\n"
            f"現在: {','.join(header)}"
        )
    if not rows:
        raise UserInputError(
            "experiments.csvに実験結果がありません。\n"
            "過去の実験結果を1行ずつ入力してください。"
        )

    experiment_ids: list[str] = []
    parameter_rows: list[list[float]] = []
    result_rows = {item.column: [] for item in problem.result_variables}
    used_ids: set[str] = set()

    for row_number, row in enumerate(rows, start=2):
        experiment_id = row["experiment_id"].strip()
        if not experiment_id:
            raise UserInputError(
                f"experiments.csvの{row_number}行目でexperiment_idが空欄です。"
            )
        if experiment_id in used_ids:
            raise UserInputError(
                f"experiments.csvでexperiment_id '{experiment_id}' が重複しています。"
            )
        used_ids.add(experiment_id)
        experiment_ids.append(experiment_id)

        numeric_values: dict[str, float] = {}
        for variable in [*problem.parameters, *problem.result_variables]:
            raw_value = row[variable.column].strip()
            if not raw_value:
                raise UserInputError(
                    f"experiments.csvの{row_number}行目に問題があります。\n"
                    f"列「{variable.column}」が空欄です。"
                )
            try:
                number = float(raw_value)
            except ValueError as error:
                raise UserInputError(
                    f"experiments.csvの{row_number}行目に問題があります。\n"
                    f"列「{variable.column}」には数値を入力してください。現在の値: {raw_value}"
                ) from error
            if not math.isfinite(number):
                raise UserInputError(
                    f"experiments.csvの{row_number}行目に問題があります。\n"
                    f"列「{variable.column}」には有限の数値を入力してください。"
                )
            numeric_values[variable.column] = number

        parameter_rows.append([numeric_values[item.column] for item in problem.parameters])
        for result in problem.result_variables:
            result_rows[result.column].append(numeric_values[result.column])

    return ExperimentData(
        experiment_ids=experiment_ids,
        parameter_rows=parameter_rows,
        result_rows=result_rows,
    )
