"""NN・Hybrid・実測の知識ルール診断を同じ形式で保存します。

学習時のNN損失だけで判断せず、最終予測と観測でも必須ルールを確認します。
"""

from __future__ import annotations

from pathlib import Path
from .data_loader import write_csv_atomic
from .knowledge import (
    build_knowledge_points,
    evaluate_rules_on_model,
    evaluate_rules_on_hybrid,
    evaluate_rules_on_observations,
)


def _knowledge_diagnostic_rows(
    model_type: str,
    report: dict[str, object],
) -> list[dict[str, object]]:
    """知識診断の共通形式へ、モデル種別を付けて変換します。"""

    return [
        {
            "model_type": model_type,
            "rule_id": item.rule_id,
            "loss": item.loss,
            "violation_rate": item.violation_rate,
            "mean_violation": item.mean_violation,
            "max_violation": item.max_violation,
            "points": item.points,
        }
        for item in report["rules"]
    ]


def write_knowledge_diagnostics(current_trial: Path, model, prepared, problem, knowledge_rules) -> list[str]:
    """3種類の診断結果を保存し、必須ルールで違反・未検証となったIDを返します。"""

    # 知見損失へ直接入ったNNと、最終Hybrid平均を分けて診断する。
    diagnostic_points = build_knowledge_points(
        problem,
        knowledge_rules,
        max_points=problem.candidate_count,
    )
    diagnostics = evaluate_rules_on_model(
        model.nn_model,
        problem,
        knowledge_rules,
        points=diagnostic_points,
    )
    hybrid_diagnostics = evaluate_rules_on_hybrid(
        model,
        problem,
        knowledge_rules,
        points=diagnostic_points,
    )
    observed_diagnostics = evaluate_rules_on_observations(
        prepared,
        problem,
        knowledge_rules,
    )
    diagnostic_path = current_trial / "output" / "knowledge_diagnostics.csv"
    diagnostic_rows = [
        *_knowledge_diagnostic_rows("nn", diagnostics),
        *_knowledge_diagnostic_rows("hybrid", hybrid_diagnostics),
        *_knowledge_diagnostic_rows("observed", observed_diagnostics),
    ]
    diagnostic_header = [
        "model_type",
        "rule_id",
        "loss",
        "violation_rate",
        "mean_violation",
        "max_violation",
        "points",
    ]
    write_csv_atomic(diagnostic_path, diagnostic_header, diagnostic_rows)
    print(f"  知見診断: {diagnostic_path}")
    failed_gates = sorted(
        set(diagnostics["gate"]["failed_rule_ids"])
        | set(hybrid_diagnostics["gate"]["failed_rule_ids"])
        | set(observed_diagnostics["gate"]["failed_rule_ids"])
    )
    return failed_gates
