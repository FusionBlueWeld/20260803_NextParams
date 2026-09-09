"""v003 の動作検証用ランナー。

このファイルは製品コードを呼び出すだけの検証入口です。次の3つを同じ
形式の JSON レポートへまとめます。

* ``laser``: validation/single/oracle のレーザー溶接モデルを用いた反復実験
* ``synthetic``: 真である非負・入力別単調性・low_sensitivity の検査
* ``plot``: 反復履歴と最後の予測空間の静的 PNG（matplotlib が任意依存）

v003 の CSV 列は今後増える可能性があるため、推薦値や予測値は固定列名に
依存せず、候補名の集合から検出します。既存 v000〜v002 の出力にも使えます。
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from .laser_welding_pseudo_experiment import (
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


from ..settings import VALIDATION_ROOT, PROJECT_ROOT
from .v003_runtime import (
    PROBLEM_HEADER, KNOWLEDGE_HEADER,
    write_csv, read_csv, number, detect_prediction, run_optimizer, latest_response_space,
)
from .v003_synthetic import synthetic_validation


DEFAULT_OPTIMIZER_ROOT = PROJECT_ROOT / "v003"


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
            from .plot_v003_validation import plot

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
