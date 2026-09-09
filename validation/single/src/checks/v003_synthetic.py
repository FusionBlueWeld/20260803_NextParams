"""既知の知識ルールを持つ合成データで、v003の予測と探索領域を検証します。

非負・単調増加・低感度の真の関係を作り、予測空間CSVの違反率を採点します。
レーザーの物理モデルを使う試験とは独立したシナリオです。
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from ..settings import VALIDATION_ROOT
from .v003_runtime import (
    PROBLEM_HEADER, KNOWLEDGE_HEADER, SEARCH_REGIONS_HEADER,
    write_csv, read_csv, number, detect_prediction, run_optimizer, latest_response_space,
)


def synthetic_validation(
    optimizer_root: Path,
    trial_name: str,
) -> dict[str, Any]:
    """真の知識ルールを含む小さな trial を実行し、出力空間を検査します。"""

    trial_path = optimizer_root / "trials" / trial_name
    if trial_path.exists():
        raise FileExistsError(f"既存 trial を上書きしません: {trial_path}")
    run_optimizer(optimizer_root, "--new", trial_name)
    problem = [
        {
            "column": "p1", "display_name": "p1", "unit": "", "role": "parameter",
            "direction": "", "lower": 0, "upper": 4, "step": 1, "target": "",
        },
        {
            "column": "p2", "display_name": "p2", "unit": "", "role": "parameter",
            "direction": "", "lower": 0, "upper": 4, "step": 1, "target": "",
        },
        {
            "column": "p3", "display_name": "p3", "unit": "", "role": "parameter",
            "direction": "", "lower": 0, "upper": 4, "step": 1, "target": "",
        },
        {
            "column": "y_mono", "display_name": "y_mono", "unit": "", "role": "objective",
            "direction": "maximize", "lower": "", "upper": "", "step": "", "target": "",
        },
        {
            "column": "y_nonnegative",
            "display_name": "y_nonnegative",
            "unit": "",
            "role": "monitor",
            "direction": "", "lower": "", "upper": "", "step": "", "target": "",
        },
        {
            "column": "y_quiet", "display_name": "y_quiet", "unit": "", "role": "monitor",
            "direction": "", "lower": "", "upper": "", "step": "", "target": "",
        },
    ]
    write_csv(trial_path / "problem.csv", PROBLEM_HEADER, problem)
    write_csv(
        trial_path / "knowledge_constraints.csv",
        KNOWLEDGE_HEADER,
        [
            {
                "rule_id": "nonnegative", "type": "lower_bound", "target": "y_nonnegative",
                "wrt": "", "value": 0, "tolerance": "", "strength": 3,
                "enabled": "true", "note": "常に正",
            },
            {
                "rule_id": "mono_p1", "type": "monotonic_increasing", "target": "y_mono",
                "wrt": "p1", "value": "", "tolerance": "", "strength": 3,
                "enabled": "true", "note": "p1に対して単調増加",
            },
            {
                "rule_id": "quiet_p2", "type": "low_sensitivity", "target": "y_quiet",
                "wrt": "p2", "value": "", "tolerance": 0.05, "strength": 3,
                "enabled": "true", "note": "p2の影響は小さい",
            },
        ],
    )
    write_csv(
        trial_path / "search_regions.csv",
        SEARCH_REGIONS_HEADER,
        [
            {
                "region_id": "ban_p3_4",
                "kind": "forbidden",
                "parameter": "p3",
                "lower": 4,
                "upper": "",
                "lower_inclusive": "true",
                "upper_inclusive": "true",
                "strength": 5,
                "enabled": "true",
                "note": "p3>=4は実験しない",
            },
            {
                "region_id": "prefer_p2_0_2",
                "kind": "preferred",
                "parameter": "p2",
                "lower": 0,
                "upper": 2,
                "lower_inclusive": "true",
                "upper_inclusive": "true",
                "strength": 3,
                "enabled": "true",
                "note": "p2=0～2が好ましい",
            },
        ],
    )
    run_optimizer(optimizer_root, "--prepare", trial_name)

    rows: list[dict[str, Any]] = []
    # 学習データは疎にし、残りの候補を v003 の response-space へ残します。
    # 全125点を実験済みにすると未測定候補がなく、推薦経路を検査できません。
    points = np.array(
        [
            [p1, p2, p3]
            for p1 in (0, 4)
            for p2 in (0, 4)
            for p3 in (0, 4)
        ]
        + [[2, 2, 2], [1, 3, 2], [3, 1, 1]],
        dtype=float,
    )
    for index, (p1, p2, p3) in enumerate(points):
        rows.append(
            {
                "experiment_id": f"synthetic_{index:03d}",
                "p1": p1,
                "p2": p2,
                "p3": p3,
                "y_mono": 1.0 + 2.0 * p1 - 0.1 * p2 + 0.2 * p3,
                "y_nonnegative": 0.5 + 0.1 * p1 + 0.2 * p2 + 0.1 * p3,
                "y_quiet": 3.0 + 0.01 * p2 + 0.4 * p1 + 0.1 * p3,
            }
        )
    write_csv(
        trial_path / "data" / "experiments.csv",
        ["experiment_id", "p1", "p2", "p3", "y_mono", "y_nonnegative", "y_quiet"],
        rows,
    )
    run_optimizer(optimizer_root, "--run", trial_name, "--n", "1")
    response = latest_response_space(trial_path)
    checks = (
        _check_synthetic_response(response)
        if response
        else {"status": "missing_response_space"}
    )
    result = {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "optimizer_root": str(optimizer_root.resolve()),
        "trial": trial_name,
        "truth": {
            "nonnegative": True,
            "y_mono_vs_p1": "increasing",
            "y_quiet_vs_p2": "low_sensitivity",
            "p3_ge_4": "forbidden",
            "p2_0_to_2": "preferred_strength_3",
        },
        "response_space": str(response or ""),
        "checks": checks,
    }
    result_root = VALIDATION_ROOT / "results" / "v003"
    result_root.mkdir(parents=True, exist_ok=True)
    path = result_root / f"{trial_name}.json"
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def _check_synthetic_response(response: Path | None) -> dict[str, Any]:
    """予測空間 CSV で3種類のルール違反率を算出します。"""

    if response is None:
        return {"status": "missing_response_space"}
    _, rows = read_csv(response)
    if not rows:
        return {"status": "empty_response_space"}
    arrays = {
        key: np.array([number(row, (key,)) for row in rows])
        for key in ("p1", "p2", "p3")
    }
    predictions = {
        key: np.array([detect_prediction(row, key) for row in rows])
        for key in ("y_mono", "y_nonnegative", "y_quiet")
    }
    finite = all(np.isfinite(value).all() for value in (*arrays.values(), *predictions.values()))
    if not finite:
        return {"status": "non_finite_prediction"}

    def paired_violation(output: str, axis: str, increasing: bool) -> tuple[int, int]:
        grouped: dict[tuple[float, ...], dict[float, float]] = {}
        for index in range(len(rows)):
            key = tuple(arrays[name][index] for name in ("p1", "p2", "p3") if name != axis)
            grouped.setdefault(key, {})[arrays[axis][index]] = predictions[output][index]
        total = violations = 0
        for values in grouped.values():
            ordered = [values[key] for key in sorted(values)]
            differences = np.diff(ordered)
            total += len(differences)
            violations += int(
                np.sum(differences < -1e-8 if increasing else differences > 1e-8)
            )
        return violations, total

    mono_bad, mono_total = paired_violation("y_mono", "p1", True)
    quiet_bad, quiet_total = paired_violation("y_quiet", "p2", True)
    quiet_deltas: list[float] = []
    for index, row in enumerate(rows):
        match = np.flatnonzero(
            (arrays["p1"] == arrays["p1"][index])
            & (arrays["p3"] == arrays["p3"][index])
            & (arrays["p2"] == arrays["p2"][index] + 1)
        )
        if len(match):
            quiet_deltas.append(
                abs(predictions["y_quiet"][match[0]] - predictions["y_quiet"][index])
            )
    policy_columns = {
        "experiment_allowed",
        "preferred_multiplier",
    }
    has_policy_columns = policy_columns <= set(rows[0])
    forbidden_flag_errors = preferred_multiplier_errors = None
    if has_policy_columns:
        allowed = np.array(
            [
                str(row["experiment_allowed"]).strip().lower()
                in {"1", "true", "yes"}
                for row in rows
            ]
        )
        multiplier = np.array(
            [number(row, ("preferred_multiplier",)) for row in rows]
        )
        expected_forbidden = arrays["p3"] >= 4
        forbidden_flag_errors = int(np.sum(allowed != ~expected_forbidden))
        expected_multiplier = np.where(arrays["p2"] <= 2, 1.25, 1.0)
        # forbidden行は倍率を1へ戻す仕様なので、許可行だけを比較する。
        preferred_multiplier_errors = int(
            np.sum(~np.isclose(multiplier[allowed], expected_multiplier[allowed]))
        )

    return {
        "status": "ok",
        "rows": len(rows),
        "nonnegative_min": float(np.min(predictions["y_nonnegative"])),
        "monotonic_p1_violation_rate": (
            float(mono_bad / mono_total) if mono_total else float("nan")
        ),
        "low_sensitivity_p2_max_step_change": float(max(quiet_deltas, default=float("nan"))),
        "low_sensitivity_tolerance": 0.05,
        "low_sensitivity_violation_rate": (
            float(np.mean(np.array(quiet_deltas) > 0.05))
            if quiet_deltas
            else float("nan")
        ),
        "forbidden_policy_flag_errors": forbidden_flag_errors,
        "preferred_policy_multiplier_errors": preferred_multiplier_errors,
    }

