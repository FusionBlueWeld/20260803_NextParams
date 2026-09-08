"""技能者の知見ルールを読み込み、決定論的な制約点を作る機能。

知見は「出力」と「変化させる入力」の組で表現します。未指定の組み合わせに
制約を自動適用しないことが、v003の重要な設計方針です。
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .data_loader import read_csv
from .parameter_space import make_axis, make_parameter_grid
from .settings import KNOWLEDGE_TRAINING_MAX_POINTS, ProblemDefinition
from .validation import UserInputError


KNOWLEDGE_HEADER = [
    "rule_id", "type", "target", "wrt", "value", "tolerance",
    "strength", "enabled", "note",
]
RULE_TYPES = {
    "lower_bound",
    "monotonic_increasing",
    "monotonic_decreasing",
    "low_sensitivity",
}
STRENGTH_WEIGHTS = {1: 0.1, 2: 0.3, 3: 1.0, 4: 3.0, 5: 10.0}


@dataclass(frozen=True)
class KnowledgeRule:
    """1つの知見。targetは出力列、wrtは入力列を指します。"""

    rule_id: str
    type: str
    target: str
    wrt: str | None = None
    value: float | None = None
    tolerance: float | None = None
    strength: int = 3
    enabled: bool = True
    note: str = ""

    @property
    def weight(self) -> float:
        return STRENGTH_WEIGHTS[self.strength]

    @property
    def is_gate(self) -> bool:
        return self.enabled and self.strength == 5


@dataclass(frozen=True)
class KnowledgePointSet:
    """ルールの学習点。単調性はbefore/afterのペアを持ちます。"""

    points: np.ndarray
    pairs: tuple[tuple[np.ndarray, np.ndarray], ...] = ()


def _float(value: str, label: str, row: int, *, optional: bool = True) -> float | None:
    if optional and not value.strip():
        return None
    try:
        number = float(value)
    except ValueError as error:
        raise UserInputError(f"knowledge_constraints.csv {row}行目: {label}は数値で指定してください。") from error
    if not np.isfinite(number):
        raise UserInputError(f"knowledge_constraints.csv {row}行目: {label}は有限値にしてください。")
    return number


def load_knowledge_constraints(path: Path, problem: ProblemDefinition) -> list[KnowledgeRule]:
    """knowledge_constraints.csvを読み込みます。未作成なら空ルールです。"""

    if not path.exists():
        return []
    header, rows = read_csv(path)
    if header != KNOWLEDGE_HEADER:
        raise UserInputError(
            "knowledge_constraints.csvのヘッダーが不正です。必要: " + ",".join(KNOWLEDGE_HEADER)
        )
    outputs = {v.column for v in problem.result_variables}
    inputs = {p.column for p in problem.parameters}
    rules: list[KnowledgeRule] = []
    ids: set[str] = set()
    for row_number, row in enumerate(rows, start=2):
        rule_id = row["rule_id"].strip()
        if not rule_id or rule_id in ids:
            raise UserInputError(
                f"knowledge_constraints.csv {row_number}行目: "
                "rule_idが空欄または重複しています。"
            )
        ids.add(rule_id)
        enabled = row["enabled"].strip().lower() not in {
            "0",
            "false",
            "no",
            "off",
        }
        # 新規trialの記入例は無効状態。problem.csv変更後も検査を妨げない。
        if not enabled:
            continue

        kind = row["type"].strip().lower()
        target = row["target"].strip()
        wrt = row["wrt"].strip() or None
        if kind not in RULE_TYPES:
            raise UserInputError(
                f"knowledge_constraints.csv {row_number}行目: "
                f"typeは{sorted(RULE_TYPES)}のいずれかです。"
            )
        if target not in outputs:
            raise UserInputError(
                f"knowledge_constraints.csv {row_number}行目: "
                f"target '{target}' は出力列にありません。"
            )
        needs_input = kind in {
            "monotonic_increasing",
            "monotonic_decreasing",
            "low_sensitivity",
        }
        if needs_input and wrt not in inputs:
            raise UserInputError(
                f"knowledge_constraints.csv {row_number}行目: "
                "このtypeには有効なwrt入力列が必要です。"
            )
        if kind == "lower_bound" and wrt is not None:
            raise UserInputError(
                f"knowledge_constraints.csv {row_number}行目: "
                "lower_boundのwrtは空欄にしてください。"
            )
        value = _float(row["value"], "value", row_number)
        tolerance = _float(row["tolerance"], "tolerance", row_number)
        if kind == "lower_bound" and value is None:
            raise UserInputError(
                f"knowledge_constraints.csv {row_number}行目: {kind}にはvalueが必要です。"
            )
        if kind == "low_sensitivity" and tolerance is None:
            raise UserInputError(
                f"knowledge_constraints.csv {row_number}行目: "
                "low_sensitivityにはtoleranceが必要です。"
            )
        if tolerance is not None and tolerance < 0:
            raise UserInputError(
                f"knowledge_constraints.csv {row_number}行目: toleranceは0以上です。"
            )
        raw_strength = row["strength"].strip() or "3"
        try:
            strength = int(raw_strength)
        except ValueError as error:
            raise UserInputError(
                f"knowledge_constraints.csv {row_number}行目: strengthは1〜5です。"
            ) from error
        if strength not in STRENGTH_WEIGHTS:
            raise UserInputError(
                f"knowledge_constraints.csv {row_number}行目: strengthは1〜5です。"
            )
        rules.append(
            KnowledgeRule(
                rule_id,
                kind,
                target,
                wrt,
                value,
                tolerance,
                strength,
                enabled,
                row["note"].strip(),
            )
        )
    return rules


def write_knowledge_template(path: Path) -> None:
    """新規trialへ、記入例をコメント代わりのnote付きで作ります。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=KNOWLEDGE_HEADER)
        writer.writeheader()
        writer.writerow(
            {
                "rule_id": "K001",
                "type": "lower_bound",
                "target": "result_1",
                "wrt": "",
                "value": "0",
                "tolerance": "",
                "strength": "3",
                "enabled": "false",
                "note": "result_1は0以上（列名はproblem.csvに合わせる）",
            }
        )


def build_knowledge_points(
    problem: ProblemDefinition,
    rules: list[KnowledgeRule],
    *,
    max_points: int = KNOWLEDGE_TRAINING_MAX_POINTS,
) -> dict[str, KnowledgePointSet]:
    """全候補グリッドからルールごとの点を決定論的に抽出します。"""

    if not rules:
        return {}
    if max_points < 1:
        raise ValueError("max_pointsは1以上です。")
    grid = make_parameter_grid(problem)
    # 順序はmake_parameter_gridの安定した辞書順。間引きも等間隔で再現可能。
    if len(grid) > max_points:
        indices = np.linspace(0, len(grid) - 1, max_points, dtype=int)
        grid = grid[np.unique(indices)]
    normalized = _normalize(grid, problem)
    parameter_index = {p.column: i for i, p in enumerate(problem.parameters)}
    result: dict[str, KnowledgePointSet] = {}
    for rule in rules:
        if not rule.enabled:
            continue
        if rule.wrt is None:
            result[rule.rule_id] = KnowledgePointSet(normalized)
            continue
        index = parameter_index[rule.wrt]
        parameter = problem.parameters[index]
        axis = np.asarray(make_axis(parameter.lower, parameter.upper, parameter.step))
        # 上端だけ刻みが短い場合も、実際の候補軸の隣接点を検査する。
        next_indices = np.searchsorted(axis, grid[:, index], side="right")
        valid = next_indices < len(axis)
        before = grid[valid]
        after = before.copy()
        after[:, index] = axis[next_indices[valid]]
        pairs = list(zip(_normalize(before, problem), _normalize(after, problem)))
        if len(pairs) > max_points:
            indices = np.linspace(0, len(pairs) - 1, max_points, dtype=int)
            pairs = [pairs[i] for i in indices]
        result[rule.rule_id] = KnowledgePointSet(
            points=(
                np.array([pair[0] for pair in pairs], dtype=float)
                if pairs
                else np.empty((0, len(problem.parameters)))
            ),
            pairs=tuple(pairs),
        )
    return result


def _normalize(raw: np.ndarray, problem: ProblemDefinition) -> np.ndarray:
    lower = np.array([float(p.lower) for p in problem.parameters])
    upper = np.array([float(p.upper) for p in problem.parameters])
    return (raw - lower) / (upper - lower)


# 後から読む人・外部検証コード向けの短い別名。
load_rules = load_knowledge_constraints
make_constraint_points = build_knowledge_points


def evaluate_rules_on_model(
    model,
    problem: ProblemDefinition,
    rules: list[KnowledgeRule],
    *,
    points: dict[str, KnowledgePointSet] | None = None,
):
    """学習後のモデルを点群で検査し、ルール別の違反診断を返します。"""
    from .knowledge_loss import calculate_rule_loss, gate_report

    point_sets = build_knowledge_points(problem, rules) if points is None else points
    columns = {item.column: i for i, item in enumerate(problem.result_variables)}
    output_scale = np.asarray(
        getattr(model, "output_scale", np.ones(len(columns))),
        dtype=float,
    )
    summaries = []
    for rule in rules:
        point_set = point_sets.get(rule.rule_id)
        if point_set is None:
            continue
        index = columns[rule.target]
        if rule.wrt is None:
            prediction = model.predict(point_set.points)[:, index]
            summary, _, _ = calculate_rule_loss(
                rule,
                point_set,
                prediction,
                loss_scale=output_scale[index],
            )
        else:
            before = np.asarray([pair[0] for pair in point_set.pairs], dtype=float)
            after = np.asarray([pair[1] for pair in point_set.pairs], dtype=float)
            if len(before):
                before_prediction = model.predict(before)[:, index]
                after_prediction = model.predict(after)[:, index]
                pairs = list(zip(before_prediction.tolist(), after_prediction.tolist()))
            else:
                pairs = []
            summary, _, _ = calculate_rule_loss(
                rule,
                point_set,
                np.empty((0,)),
                pairs,
                loss_scale=output_scale[index],
            )
        summaries.append(summary)
    return {"rules": summaries, "gate": gate_report(rules, summaries)}


def evaluate_rules_on_hybrid(
    model,
    problem: ProblemDefinition,
    rules: list[KnowledgeRule],
    *,
    points: dict[str, KnowledgePointSet] | None = None,
):
    """HybridModelの最終平均も同じルール点で確認します。"""
    # evaluate_rules_on_modelは列番号をresult_variables全体で解釈するため、
    # 各出力列をまとめた軽いアダプタを使う。
    class _AllOutputs:
        output_scale = model.nn_model.output_scale

        def predict(self, x):
            prediction = model.predict(x)
            return np.column_stack(
                [
                    prediction.results[item.column]["hybrid_mean"]
                    for item in problem.result_variables
                ]
            )

    return evaluate_rules_on_model(_AllOutputs(), problem, rules, points=points)


def evaluate_rules_on_observations(
    prepared,
    problem: ProblemDefinition,
    rules: list[KnowledgeRule],
):
    """実測平均と知見の明白な矛盾を、予測診断と同じ形式で返します。"""

    from .knowledge_loss import calculate_rule_loss, gate_report

    parameter_index = {
        item.column: index for index, item in enumerate(problem.parameters)
    }
    result_scale = {
        item.column: max(float(np.std(prepared.result_means[item.column])), 1e-12)
        for item in problem.result_variables
    }
    point_lookup = {
        tuple(np.round(row, 12)): index
        for index, row in enumerate(prepared.raw_parameters)
    }
    summaries = []
    for rule in rules:
        if not rule.enabled:
            continue
        values = prepared.result_means[rule.target]
        if rule.wrt is None:
            point_set = KnowledgePointSet(prepared.normalized_parameters)
            summary, _, _ = calculate_rule_loss(
                rule,
                point_set,
                values,
                loss_scale=result_scale[rule.target],
            )
        else:
            axis = parameter_index[rule.wrt]
            parameter = problem.parameters[axis]
            step = float(parameter.step)
            axis_values = make_axis(parameter.lower, parameter.upper, step)
            successors = {
                round(value, 12): following
                for value, following in zip(axis_values, [*axis_values[1:], None])
            }
            pairs: list[tuple[float, float]] = []
            for before_index, before in enumerate(prepared.raw_parameters):
                next_value = successors.get(round(before[axis], 12), before[axis] + step)
                if next_value is None:
                    continue
                after = before.copy()
                after[axis] = next_value
                after_index = point_lookup.get(tuple(np.round(after, 12)))
                if after_index is not None:
                    pairs.append((float(values[before_index]), float(values[after_index])))
            point_set = KnowledgePointSet(
                np.empty((0, len(problem.parameters))),
                pairs=tuple(),
            )
            summary, _, _ = calculate_rule_loss(
                rule,
                point_set,
                np.empty((0,)),
                pairs,
                loss_scale=result_scale[rule.target],
            )
        summaries.append(summary)
    # 実測に比較可能なペアがなければ未検証と記録する。全候補のモデル検査は必須。
    return {
        "rules": summaries,
        "gate": gate_report(rules, summaries, require_points=False),
    }
