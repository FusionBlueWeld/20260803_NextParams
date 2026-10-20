"""知見ルールの損失と、予測値に対する勾配。

ここは将来PyTorchへ移す際の境界です。現在はNumPy NNの手書き逆伝播へ
``d_loss/d_prediction``を返し、NNの重み更新へ直接加算します。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .knowledge import KnowledgePointSet, KnowledgeRule


@dataclass(frozen=True)
class RuleLoss:
    rule_id: str
    loss: float
    violation_rate: float
    mean_violation: float
    max_violation: float
    points: int


def _empty_summary(rule: KnowledgeRule) -> RuleLoss:
    return RuleLoss(rule.rule_id, 0.0, 0.0, 0.0, 0.0, 0)


def calculate_rule_loss(
    rule: KnowledgeRule,
    point_set: KnowledgePointSet,
    predictions: np.ndarray,
    pair_predictions: list[tuple[float, float]] | None = None,
    *,
    loss_scale: float = 1.0,
) -> tuple[RuleLoss, np.ndarray, list[tuple[float, float]]]:
    """1ルールを評価し、lossと予測値勾配を返します。

    ``predictions``は単一点ルール用です。単調性・影響小はペア予測を使います。
    ``loss_scale``で違反量を無次元化します。診断値は元の出力単位、返す勾配は
    元の予測値に対する勾配です。これにより単位が違う出力でもstrengthを比較
    でき、後段のNN逆伝播へそのまま接続できます。
    """
    scale = max(float(loss_scale), 1e-12)
    if not rule.enabled:
        return _empty_summary(rule), np.zeros_like(predictions), []
    if rule.type.startswith("monotonic") or rule.type == "low_sensitivity":
        pairs = pair_predictions or []
        grad = np.zeros(2 * len(pairs), dtype=float)
        violations: list[float] = []
        for i, (before, after) in enumerate(pairs):
            delta = after - before
            if rule.type == "monotonic_increasing":
                amount = max(0.0, -delta)
                # v=before-after: dL/dbefore=+2v, dL/dafter=-2v
                sign = 1.0
            elif rule.type == "monotonic_decreasing":
                amount = max(0.0, delta)
                # v=after-before: dL/dbefore=-2v, dL/dafter=+2v
                sign = -1.0
            else:
                tolerance = float(rule.tolerance or 0.0)
                amount = max(0.0, abs(delta) - tolerance)
                sign = float(np.sign(delta))
            violations.append(amount)
            if amount > 0:
                # loss=(weight/n)*violation^2。before/afterの勾配を記録。
                factor = (
                    2.0 * rule.weight * amount
                    / (max(len(pairs), 1) * scale**2)
                )
                if rule.type == "low_sensitivity":
                    grad[2 * i] -= factor * sign
                    grad[2 * i + 1] += factor * sign
                else:
                    grad[2 * i] += factor * sign
                    grad[2 * i + 1] -= factor * sign
        v = np.asarray(violations, dtype=float)
        loss = (
            rule.weight * float(np.mean(np.square(v / scale)))
            if len(v)
            else 0.0
        )
        summary = RuleLoss(
            rule_id=rule.rule_id,
            loss=loss,
            violation_rate=float(np.mean(v > 0)) if len(v) else 0.0,
            mean_violation=float(np.mean(v)) if len(v) else 0.0,
            max_violation=float(np.max(v)) if len(v) else 0.0,
            points=len(v),
        )
        return summary, grad, pairs

    values = np.asarray(predictions, dtype=float)
    if len(values) == 0:
        return _empty_summary(rule), np.zeros_like(values), []
    if rule.type == "lower_bound":
        violation = np.maximum(float(rule.value) - values, 0.0)
        gradient = -2.0 * rule.weight * violation / (len(values) * scale**2)
    else:
        raise ValueError(f"unknown knowledge rule: {rule.type}")
    summary = RuleLoss(
        rule.rule_id,
        rule.weight * float(np.mean(np.square(violation / scale))),
        float(np.mean(violation > 0)),
        float(np.mean(violation)),
        float(np.max(violation)),
        len(values),
    )
    return summary, gradient, []


def gate_report(
    rules: list[KnowledgeRule],
    summaries: list[RuleLoss],
    *,
    tolerance: float = 1e-8,
    require_points: bool = True,
) -> dict[str, object]:
    """strength=5の制約を推薦ゲートとして判定します。"""
    summary_map = {item.rule_id: item for item in summaries}
    unchecked = [
        rule.rule_id
        for rule in rules
        if rule.is_gate
        and summary_map.get(rule.rule_id, _empty_summary(rule)).points == 0
    ]
    failures = list(unchecked) if require_points else []
    for rule in rules:
        summary = summary_map.get(rule.rule_id)
        if not rule.is_gate or summary is None or summary.points == 0:
            continue
        if not np.isfinite(summary.max_violation) or summary.max_violation > tolerance:
            failures.append(rule.rule_id)
    return {
        "passed": not failures,
        "failed_rule_ids": failures,
        "unchecked_rule_ids": unchecked,
        "checked": sum(rule.is_gate for rule in rules) - len(unchecked),
    }
