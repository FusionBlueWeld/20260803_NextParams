"""終了判定のrun履歴を安全に保存します。"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

from .criteria import StopDecision


def _fingerprint(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def build_stop_context(experiments, *, problem, knowledge_rules, region_policy, stop_config):
    """実験IDごとの内容と設定を記録し、追記・再実行・訂正を区別します。"""
    observations = {
        experiment_id: _fingerprint({
            "parameters": experiments.parameter_rows[index],
            "results": {
                name: values[index] for name, values in experiments.result_rows.items()
            },
        })
        for index, experiment_id in enumerate(experiments.experiment_ids)
    }
    return {
        "settings_fingerprint": _fingerprint({
            "problem": asdict(problem),
            "knowledge": [asdict(rule) for rule in knowledge_rules],
            "regions": asdict(region_policy),
            "stopping": asdict(stop_config),
        }),
        "data_fingerprint": _fingerprint(observations),
        "observations": observations,
    }


def load_stop_history(
    path: str | Path,
    current_data_rows: int,
    *,
    input_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """既存履歴を読む。初回runでは現在行数を追加実験数の起点にします。"""

    json_path = Path(path)
    if not json_path.exists():
        return {
            "initial_data_rows": current_data_rows,
            "runs": [],
            "convergence_start_index": 0,
        }
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    runs = list(payload.get("runs", []))
    start = int(payload.get("convergence_start_index", 0))
    if input_context is not None:
        previous = payload.get("input_context") or {}
        current_observations = input_context["observations"]
        unchanged_or_appended = (
            previous.get("settings_fingerprint") == input_context["settings_fingerprint"]
            and "observations" in previous
            and all(
                current_observations.get(identifier) == fingerprint
                for identifier, fingerprint in previous["observations"].items()
            )
        )
        if not unchanged_or_appended:
            # 設定変更・既存実験の訂正/削除・旧形式の履歴は収束根拠に混ぜない。
            # 履歴本体と実験予算の起点は保持する。
            start = len(runs)
    return {
        "initial_data_rows": int(payload.get("initial_data_rows", current_data_rows)),
        "runs": runs,
        "convergence_start_index": start,
    }


def convergence_history(runs: list[dict[str, Any]], start_index: int) -> list[dict[str, Any]]:
    """現在の設定下の異なる実験データだけを、収束の時系列に使います。"""
    distinct: dict[str, dict[str, Any]] = {}
    for run in runs[start_index:]:
        fingerprint = run.get("data_fingerprint")
        if (
            fingerprint is None
            or run.get("best_feasible_objective") is None
            or run.get("predicted_optimum") is None
        ):
            # 判定不能なrunを飛び越えて、古い安定履歴をつなげない。
            distinct.clear()
            continue
        distinct[fingerprint] = run
    return list(distinct.values())


def write_stop_status(
    path: str | Path,
    *,
    decision: StopDecision,
    initial_data_rows: int,
    runs: list[dict[str, Any]],
    input_context: dict[str, Any] | None = None,
    convergence_start_index: int = 0,
) -> None:
    """最新状態と判定根拠を、一時ファイル完成後に置換します。"""

    json_path = Path(path)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "status": decision.status.value,
        "reasons": list(decision.reasons),
        "metrics": decision.metrics,
        "initial_data_rows": initial_data_rows,
        "runs": runs,
        "input_context": input_context,
        "convergence_start_index": convergence_start_index,
    }
    temporary = json_path.with_suffix(json_path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(json_path)
