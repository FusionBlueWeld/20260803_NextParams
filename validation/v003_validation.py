"""v003 の動作検証用ランナー。

このファイルは製品コードを呼び出すだけの検証入口です。次の3つを同じ
形式の JSON レポートへまとめます。

* ``laser``: validation/oracle のレーザー溶接モデルを用いた反復実験
* ``synthetic``: 真である非負・入力別単調性・low_sensitivity の検査
* ``plot``: 反復履歴と最後の予測空間の静的 PNG（matplotlib が任意依存）

v003 の CSV 列は今後増える可能性があるため、推薦値や予測値は固定列名に
依存せず、候補名の集合から検出します。既存 v000〜v002 の出力にも使えます。
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from laser_welding_pseudo_experiment import (
    CONSTRAINT_COLUMN,
    OBJECTIVE_COLUMN,
    PARAMETER_COLUMNS,
    best_feasible,
    candidate_axes,
    evaluate_points,
    initial_experiments,
    load_oracle,
    problem_rows,
)


VALIDATION_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = VALIDATION_ROOT.parent
DEFAULT_OPTIMIZER_ROOT = PROJECT_ROOT / "v003"

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
KNOWLEDGE_HEADER = [
    "rule_id",
    "type",
    "target",
    "wrt",
    "value",
    "tolerance",
    "strength",
    "enabled",
    "note",
]
SEARCH_REGIONS_HEADER = [
    "region_id",
    "kind",
    "parameter",
    "lower",
    "upper",
    "lower_inclusive",
    "upper_inclusive",
    "strength",
    "enabled",
    "note",
]


def write_csv(path: Path, header: list[str], rows: Iterable[dict[str, Any]]) -> None:
    """検証用 CSV を UTF-8 BOM 付きで安全に作成します。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=header, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    """BOM の有無を吸収して CSV を読む小さなヘルパーです。"""

    with path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        return list(reader.fieldnames or []), list(reader)


def number(row: dict[str, str], names: Iterable[str], default: float = float("nan")) -> float:
    """候補名を順に探して数値化します。"""

    for name in names:
        value = row.get(name, "")
        if value not in (None, ""):
            try:
                return float(value)
            except ValueError:
                continue
    return default


def detect_prediction(row: dict[str, str], column: str) -> float:
    """v000〜v003 の命名差を吸収して予測平均を取得します。"""

    return number(
        row,
        (
            f"{column}_mean",
            f"{column}_hybrid_mean",
            f"{column}_predicted",
            f"{column}_prediction",
            f"{column}_nn_pred",
            f"{column}_gp_mean",
            column,
        ),
    )


def run_optimizer(optimizer_root: Path, *arguments: str) -> str:
    """共通 main.py 経由で v003 CLI を実行します。"""

    command = [
        sys.executable,
        str(PROJECT_ROOT / "main.py"),
        "--version",
        optimizer_root.resolve().name,
        *arguments,
    ]
    completed = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if completed.returncode:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise RuntimeError(f"CLI failed ({' '.join(arguments)}):\n{detail}")
    return completed.stdout


def latest_response_space(trial_path: Path) -> Path | None:
    """v002/v003 の解空間 CSV のうち最新ファイルを返します。"""

    directories = [
        trial_path / "output" / "response_spaces",
        trial_path / "output" / "response_space",
        trial_path / "output",
    ]
    candidates: list[Path] = []
    for directory in directories:
        if directory.is_dir():
            candidates.extend(
                path
                for path in directory.glob("*.csv")
                if "recommend" not in path.name.lower()
            )
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime_ns)


def write_problem(path: Path) -> None:
    """レーザー溶接検証の固定問題を trial へ書き込みます。"""

    write_csv(path, PROBLEM_HEADER, problem_rows())


def write_laser_knowledge(path: Path) -> None:
    """レーザー用に、明らかに真である最小限の知識だけを記録します。

    実際の物理モデルの全域単調性は仮定しません。深さの非負だけを v003 の
    知識損失へ渡し、単調性の検査は後述の合成問題で分離して行います。
    """

    write_csv(
        path,
        KNOWLEDGE_HEADER,
        [
            {
                "rule_id": "laser_nonnegative",
                "type": "lower_bound",
                "target": OBJECTIVE_COLUMN,
                "wrt": "",
                "value": "0",
                "tolerance": "",
                "strength": "2",
                "enabled": "true",
                "note": "溶込み深さは負にならない",
            }
        ],
    )


def laser_validation(
    optimizer_root: Path,
    trial_name: str,
    iterations: int,
    recommendations: int,
    with_knowledge: bool,
) -> dict[str, Any]:
    """レーザー溶接の反復推薦を実行し、収束指標を保存します。"""

    if iterations < 1 or recommendations < 1:
        raise ValueError("iterations と recommendations は1以上にしてください")
    oracle = load_oracle(VALIDATION_ROOT / "oracle")
    true_best = _true_best(oracle)
    trial_path = optimizer_root / "trials" / trial_name
    if trial_path.exists():
        raise FileExistsError(f"既存 trial を上書きしません: {trial_path}")

    run_optimizer(optimizer_root, "--new", trial_name)
    write_problem(trial_path / "problem.csv")
    if with_knowledge:
        write_laser_knowledge(trial_path / "knowledge_constraints.csv")
    run_optimizer(optimizer_root, "--prepare", trial_name)

    experiments = initial_experiments(oracle)
    experiment_path = trial_path / "data" / "experiments.csv"
    write_csv(
        experiment_path,
        ["experiment_id", *PARAMETER_COLUMNS, OBJECTIVE_COLUMN, CONSTRAINT_COLUMN],
        experiments,
    )
    initial_best = best_feasible(experiments)
    history: list[dict[str, Any]] = []
    recommendation_path = trial_path / "output" / "recommendations.csv"

    for iteration in range(1, iterations + 1):
        run_optimizer(optimizer_root, "--run", trial_name, "--n", str(recommendations))
        _, rows = read_csv(recommendation_path)
        rows.sort(key=lambda row: number(row, ("rank",), float("inf")))
        if len(rows) < recommendations:
            raise RuntimeError(f"推薦行数が不足しています: {len(rows)} < {recommendations}")
        points = np.array(
            [
                [number(row, (column,)) for column in PARAMETER_COLUMNS]
                for row in rows[:recommendations]
            ],
            dtype=float,
        )
        if not np.isfinite(points).all():
            raise RuntimeError("recommendations.csv に入力パラメータがありません")
        depths, spatters = evaluate_points(oracle, points)
        new_rows: list[dict[str, Any]] = []
        for slot, (recommendation, point) in enumerate(
            zip(rows[:recommendations], points, strict=True), start=1
        ):
            item = {
                "experiment_id": f"bo_{iteration:03d}_{slot:02d}",
                **{column: float(point[index]) for index, column in enumerate(PARAMETER_COLUMNS)},
                OBJECTIVE_COLUMN: float(depths[slot - 1]),
                CONSTRAINT_COLUMN: int(spatters[slot - 1]),
            }
            new_rows.append(item)
            history.append(
                {
                    "iteration": iteration,
                    "slot": slot,
                    "parameters": {column: item[column] for column in PARAMETER_COLUMNS},
                    "predicted_objective": detect_prediction(recommendation, OBJECTIVE_COLUMN),
                    "recommendation_score": number(
                        recommendation,
                        ("recommendation_score", "score"),
                    ),
                    "actual_objective": item[OBJECTIVE_COLUMN],
                    "actual_constraint": item[CONSTRAINT_COLUMN],
                    "actual_feasible": bool(item[CONSTRAINT_COLUMN] <= 4),
                }
            )
        experiments.extend(new_rows)
        write_csv(
            experiment_path,
            ["experiment_id", *PARAMETER_COLUMNS, OBJECTIVE_COLUMN, CONSTRAINT_COLUMN],
            experiments,
        )

    # 最後のモデルで応答空間を出力させ、可視化の入力を残します。
    run_optimizer(optimizer_root, "--run", trial_name, "--n", str(recommendations))
    final_best = best_feasible(experiments)
    if initial_best is None or final_best is None:
        raise RuntimeError("実験結果に適合条件がありません")

    result_root = VALIDATION_ROOT / "results" / "v003"
    result_root.mkdir(parents=True, exist_ok=True)
    diagnostic_path = trial_path / "output" / "knowledge_diagnostics.csv"
    response_path = latest_response_space(trial_path)
    stop_path = trial_path / "output" / "stopping_status.json"
    _, diagnostics = read_csv(diagnostic_path) if diagnostic_path.is_file() else ([], [])
    stopping = (
        json.loads(stop_path.read_text(encoding="utf-8"))
        if stop_path.is_file()
        else {"status": "missing"}
    )
    optimum_found_iteration = next(
        (
            int(item["iteration"])
            for item in history
            if item["actual_feasible"]
            and all(
                float(item["parameters"][column]) == float(true_best[column])
                for column in PARAMETER_COLUMNS
            )
        ),
        None,
    )
    summary = {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "optimizer_root": str(optimizer_root.resolve()),
        "trial": trial_name,
        "iterations": iterations,
        "recommendations_per_iteration": recommendations,
        "knowledge_constraints_enabled": with_knowledge,
        "true_best": true_best,
        "initial_best": initial_best,
        "final_best": final_best,
        "initial_optimum_ratio": (
            float(initial_best[OBJECTIVE_COLUMN])
            / true_best[OBJECTIVE_COLUMN]
        ),
        "final_optimum_ratio": float(final_best[OBJECTIVE_COLUMN]) / true_best[OBJECTIVE_COLUMN],
        "optimum_found": all(
            float(final_best[column]) == float(true_best[column]) for column in PARAMETER_COLUMNS
        ),
        "optimum_found_iteration": optimum_found_iteration,
        "feasible_recommendation_count": sum(
            bool(item["actual_feasible"]) for item in history
        ),
        "history": history,
        "response_space": str(response_path or ""),
        "response_space_metrics": _response_space_metrics(oracle, response_path),
        "knowledge_diagnostics": diagnostics,
        "stopping": stopping,
    }
    summary_path = result_root / f"{trial_name}.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    trajectory_path = result_root / f"{trial_name}_trajectory.csv"
    write_csv(
        trajectory_path,
        [
            "iteration",
            "slot",
            *PARAMETER_COLUMNS,
            "actual_objective",
            "actual_constraint",
            "recommendation_score",
        ],
        (
            {
                "iteration": item["iteration"],
                "slot": item["slot"],
                **item["parameters"],
                "actual_objective": item["actual_objective"],
                "actual_constraint": item["actual_constraint"],
                "recommendation_score": item["recommendation_score"],
            }
            for item in history
        ),
    )
    _write_laser_report(summary, summary_path)
    return summary


def _true_best(oracle: Any) -> dict[str, Any]:
    """oracle 全候補から比較用の真の最良点を計算します。"""

    axes = candidate_axes()
    grids = np.meshgrid(*axes, indexing="ij")
    points = np.column_stack([grid.ravel() for grid in grids])
    depths, spatters = evaluate_points(oracle, points)
    feasible = np.flatnonzero(spatters <= 4)
    index = int(feasible[np.argmax(depths[feasible])])
    return {
        "candidate_count": int(len(points)),
        **{column: float(points[index, offset]) for offset, column in enumerate(PARAMETER_COLUMNS)},
        OBJECTIVE_COLUMN: float(depths[index]),
        CONSTRAINT_COLUMN: int(spatters[index]),
    }


def _response_space_metrics(oracle: Any, response: Path | None) -> dict[str, Any]:
    """最後の全候補予測を oracle と突き合わせ、空間全体の誤差を測ります。"""

    if response is None:
        return {"status": "missing_response_space"}
    _, rows = read_csv(response)
    if not rows:
        return {"status": "empty_response_space"}

    points = np.array(
        [[number(row, (column,)) for column in PARAMETER_COLUMNS] for row in rows],
        dtype=float,
    )
    predicted_objective = np.array(
        [detect_prediction(row, OBJECTIVE_COLUMN) for row in rows], dtype=float
    )
    predicted_constraint = np.array(
        [detect_prediction(row, CONSTRAINT_COLUMN) for row in rows], dtype=float
    )
    support = np.array(
        [number(row, ("gp_support", "data_support", "support")) for row in rows],
        dtype=float,
    )
    finite = (
        np.isfinite(points).all(axis=1)
        & np.isfinite(predicted_objective)
        & np.isfinite(predicted_constraint)
        & np.isfinite(support)
    )
    nonfinite = int(np.sum(~finite))
    if not np.any(finite):
        return {"status": "non_finite_prediction", "rows": len(rows)}

    true_objective, true_constraint = evaluate_points(oracle, points[finite])
    objective_error = predicted_objective[finite] - true_objective
    constraint_error = predicted_constraint[finite] - true_constraint
    predicted_feasible = predicted_constraint[finite] <= 4
    true_feasible = true_constraint <= 4
    return {
        "status": "ok",
        "rows": len(rows),
        "nonfinite_rows": nonfinite,
        "objective_mae": float(np.mean(np.abs(objective_error))),
        "objective_rmse": float(np.sqrt(np.mean(objective_error**2))),
        "constraint_mae": float(np.mean(np.abs(constraint_error))),
        "constraint_rmse": float(np.sqrt(np.mean(constraint_error**2))),
        "feasible_classification_accuracy": float(
            np.mean(predicted_feasible == true_feasible)
        ),
        "support_min": float(np.min(support[finite])),
        "support_median": float(np.median(support[finite])),
        "support_max": float(np.max(support[finite])),
    }


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


def _write_laser_report(summary: dict[str, Any], path: Path) -> None:
    """LLM と人が要点を拾いやすい短い Markdown 概要を出力します。"""

    space = summary["response_space_metrics"]
    stopping = summary["stopping"]
    lines = [
        "# v003 レーザー溶接検証レポート",
        "",
        f"- Trial: `{summary['trial']}`",
        f"- 最終最良 / 真の最良: `{summary['final_optimum_ratio']:.2%}`",
        f"- 真の最良条件を発見: `{summary['optimum_found']}`",
        f"- 発見反復: `{summary['optimum_found_iteration']}`",
        f"- 適合推薦: `{summary['feasible_recommendation_count']} / {len(summary['history'])}`",
        f"- 終了判定: `{stopping.get('status', 'missing')}`",
        f"- 予測空間CSV: `{summary['response_space'] or '未生成'}`",
        f"- 予測空間行数: `{space.get('rows', 0):,}`",
        f"- 目的値MAE: `{space.get('objective_mae', float('nan')):.6g}`",
        f"- 制約値MAE: `{space.get('constraint_mae', float('nan')):.6g}`",
        f"- 適合/不適合の分類正解率: `{space.get('feasible_classification_accuracy', float('nan')):.2%}`",
        "",
        "## 主要結論",
        "",
        "`history` は各反復の推薦条件・推薦スコア・oracle実測値を保持する。",
        "予測空間は最後の `response_space` を可視化スクリプトへ渡して確認する。",
        "",
        f"機械可読な詳細: `{path.name}`",
    ]
    path.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="v003 validation runner")
    parser.add_argument("--mode", choices=("laser", "synthetic", "all"), default="all")
    parser.add_argument("--optimizer-root", type=Path, default=DEFAULT_OPTIMIZER_ROOT)
    parser.add_argument("--trial", default="trial_laser_welding_v003_validation")
    parser.add_argument("--synthetic-trial", default="trial_synthetic_constraints_v003")
    parser.add_argument("--iterations", type=int, default=15)
    parser.add_argument("--recommendations", type=int, default=1)
    parser.add_argument("--without-knowledge", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.mode in {"laser", "all"}:
        summary = laser_validation(
            args.optimizer_root,
            args.trial,
            args.iterations,
            args.recommendations,
            not args.without_knowledge,
        )
        print(
            f"laser: {summary['final_optimum_ratio']:.2%} "
            f"optimum_found={summary['optimum_found']}"
        )
        # グラフは任意依存のため、失敗しても数値検証は成功扱いにします。
        try:
            from plot_v003_validation import plot

            plot(
                VALIDATION_ROOT / "results" / "v003" / f"{args.trial}.json",
                VALIDATION_ROOT / "results" / "v003" / f"{args.trial}.png",
            )
        except (ImportError, OSError, ValueError) as error:
            print(f"plot: skipped ({error})")
    if args.mode in {"synthetic", "all"}:
        result = synthetic_validation(args.optimizer_root, args.synthetic_trial)
        print(f"synthetic: {result['checks'].get('status', 'unknown')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
