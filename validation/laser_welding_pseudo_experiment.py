"""レーザー溶接の論理モデルを使って、各バージョンの反復探索を検証します。

人間が行う次の作業を自動で再現する、検証専用のスクリプトです。

1. trialを新規作成する
2. problem.csvを設定する
3. 初期実験結果をexperiments.csvへ入力する
4. main.py --run --nで次の条件を複数推薦する
5. 同じ学習状態で推薦された全条件を論理モデルで擬似実験する
6. バッチの結果をexperiments.csvへまとめて追加し、再び探索する

通常運用のコードと混ざらないよう、このファイルはvalidation配下に置きます。
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import itertools
import json
import os
import subprocess
import sys
import types
from datetime import datetime
from pathlib import Path
from typing import Callable

import numpy as np


VALIDATION_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = VALIDATION_ROOT.parent
DEFAULT_OPTIMIZER_ROOT = PROJECT_ROOT / "v000"
DEFAULT_ORACLE_ROOT = VALIDATION_ROOT / "oracle"

PARAMETER_COLUMNS = [
    "laser_power_w",
    "spot_diameter_um",
    "scan_speed_mm_s",
]
OBJECTIVE_COLUMN = "penetration_depth_mm"
CONSTRAINT_COLUMN = "spatter_level_0_9"
EXPERIMENT_HEADER = [
    "experiment_id",
    *PARAMETER_COLUMNS,
    OBJECTIVE_COLUMN,
    CONSTRAINT_COLUMN,
]


def parse_arguments() -> argparse.Namespace:
    """検証対象と反復回数をCLIから受け取ります。"""

    parser = argparse.ArgumentParser(
        description="レーザー溶接モデルで指定バージョンの擬似実験を反復します。"
    )
    parser.add_argument(
        "--oracle-root",
        type=Path,
        default=DEFAULT_ORACLE_ROOT,
        help="論理モデルのフォルダ。省略時はリポジトリ内のvalidation/oracle",
    )
    parser.add_argument(
        "--optimizer-root",
        type=Path,
        default=DEFAULT_OPTIMIZER_ROOT,
        help="検証するバージョンフォルダ。省略時はv000",
    )
    parser.add_argument(
        "--trial",
        default="trial_laser_welding_validation",
        help="新規作成する検証用trial名",
    )
    parser.add_argument("--iterations", type=int, default=15)
    parser.add_argument(
        "--recommendations",
        type=int,
        default=1,
        help="1回の探索で同時取得する推薦条件数。省略時は従来どおり1件",
    )
    return parser.parse_args()


def load_oracle(oracle_root: Path) -> Callable[..., dict[str, np.ndarray]]:
    """リポジトリ内または指定された場所のphysics_model.pyを読み込みます。"""

    resolved_root = oracle_root.resolve()
    model_path = resolved_root / "physics_model.py"
    if not model_path.is_file():
        model_path = resolved_root / "src" / "physics_model.py"
    if not model_path.is_file():
        raise FileNotFoundError(f"論理モデルが見つかりません: {model_path}")

    # 通常の「src.physics_model」として読むとsrc/__init__.pyが実行され、論理モデルと
    # 無関係な探索UI（SciPy、scikit-learn等）まで必要になります。専用の空パッケージを
    # 用意し、physics_model.pyとその直接の相対importだけを読み込みます。
    package_name = "_paramoptimizer_validation_oracle"
    package = types.ModuleType(package_name)
    package.__path__ = [str(model_path.parent)]  # type: ignore[attr-defined]
    sys.modules[package_name] = package

    module_name = f"{package_name}.physics_model"
    spec = importlib.util.spec_from_file_location(module_name, model_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"論理モデルの読込設定を作成できません: {model_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module.evaluate_model


def run_optimizer(optimizer_root: Path, *arguments: str) -> str:
    """利用者と同じように共通main.pyから指定バージョンを実行します。"""

    command = [
        sys.executable,
        str(PROJECT_ROOT / "main.py"),
        "--version",
        optimizer_root.resolve().name,
        *arguments,
    ]
    child_environment = os.environ.copy()
    child_environment["PYTHONIOENCODING"] = "utf-8"
    completed = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        env=child_environment,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    if completed.returncode != 0:
        message = completed.stderr.strip() or completed.stdout.strip()
        raise RuntimeError(
            f"{optimizer_root.name}の実行に失敗しました: "
            f"{' '.join(arguments)}\n{message}"
        )
    return completed.stdout


def write_csv(path: Path, header: list[str], rows: list[dict[str, object]]) -> None:
    """Excelでも開きやすいUTF-8 BOM付きCSVとして保存します。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=header)
        writer.writeheader()
        writer.writerows(rows)
    temporary_path.replace(path)


def read_csv(path: Path) -> list[dict[str, str]]:
    """推薦CSVを列名付きで読み込みます。"""

    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def problem_rows() -> list[dict[str, object]]:
    """今回の検証で使用する最適化問題を定義します。"""

    return [
        {
            "column": "laser_power_w",
            "display_name": "レーザー出力",
            "unit": "W",
            "role": "parameter",
            "direction": "",
            "lower": 100,
            "upper": 6000,
            "step": 200,
            "target": "",
        },
        {
            "column": "spot_diameter_um",
            "display_name": "スポット径",
            "unit": "um",
            "role": "parameter",
            "direction": "",
            "lower": 50,
            "upper": 300,
            "step": 25,
            "target": "",
        },
        {
            "column": "scan_speed_mm_s",
            "display_name": "走査速度",
            "unit": "mm/s",
            "role": "parameter",
            "direction": "",
            "lower": 10,
            "upper": 1000,
            "step": 25,
            "target": "",
        },
        {
            "column": OBJECTIVE_COLUMN,
            "display_name": "溶込み深さ",
            "unit": "mm",
            "role": "objective",
            "direction": "maximize",
            "lower": "",
            "upper": "",
            "step": "",
            "target": "",
        },
        {
            "column": CONSTRAINT_COLUMN,
            "display_name": "スパッタレベル",
            "unit": "level",
            "role": "constraint",
            "direction": "less_equal",
            "lower": "",
            "upper": "",
            "step": "",
            "target": 4,
        },
    ]


def candidate_axes() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """v000のproblem.csvと同じ規則で候補軸を作ります。"""

    power = np.append(np.arange(100.0, 6000.0, 200.0), 6000.0)
    spot = np.arange(50.0, 300.0 + 1.0, 25.0)
    speed = np.append(np.arange(10.0, 1000.0, 25.0), 1000.0)
    return power, spot, speed


def evaluate_points(
    oracle: Callable[..., dict[str, np.ndarray]],
    points: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """入力条件を論理モデルへ渡し、深さとスパッタを返します。"""

    outputs = oracle(points[:, 0], points[:, 1], points[:, 2])
    depths = np.asarray(outputs[OBJECTIVE_COLUMN], dtype=float)
    spatters = np.asarray(outputs[CONSTRAINT_COLUMN], dtype=int)
    return depths, spatters


def find_true_best(
    oracle: Callable[..., dict[str, np.ndarray]],
) -> dict[str, float | int]:
    """候補全点を評価し、比較基準となる真の最良条件を求めます。"""

    axes = candidate_axes()
    grids = np.meshgrid(*axes, indexing="ij")
    points = np.column_stack([grid.ravel() for grid in grids])
    depths, spatters = evaluate_points(oracle, points)
    feasible_indices = np.flatnonzero(spatters <= 4)
    best_index = feasible_indices[np.argmax(depths[feasible_indices])]
    return {
        "candidate_count": int(len(points)),
        "laser_power_w": float(points[best_index, 0]),
        "spot_diameter_um": float(points[best_index, 1]),
        "scan_speed_mm_s": float(points[best_index, 2]),
        "penetration_depth_mm": float(depths[best_index]),
        "spatter_level_0_9": int(spatters[best_index]),
    }


def initial_experiments(
    oracle: Callable[..., dict[str, np.ndarray]],
) -> list[dict[str, object]]:
    """各軸の下端・中央・上端を組み合わせた27点を用意します。"""

    axes = candidate_axes()
    levels = [[axis[0], axis[len(axis) // 2], axis[-1]] for axis in axes]
    points = np.array(list(itertools.product(*levels)), dtype=float)
    depths, spatters = evaluate_points(oracle, points)

    rows: list[dict[str, object]] = []
    for index, point in enumerate(points):
        rows.append(
            {
                "experiment_id": f"initial_{index + 1:03d}",
                "laser_power_w": float(point[0]),
                "spot_diameter_um": float(point[1]),
                "scan_speed_mm_s": float(point[2]),
                "penetration_depth_mm": float(depths[index]),
                "spatter_level_0_9": int(spatters[index]),
            }
        )
    return rows


def best_feasible(rows: list[dict[str, object]]) -> dict[str, object] | None:
    """現在までの擬似実験から、制約を満たす最良行を返します。"""

    feasible = [row for row in rows if float(row[CONSTRAINT_COLUMN]) <= 4.0]
    if not feasible:
        return None
    return max(feasible, key=lambda row: float(row[OBJECTIVE_COLUMN]))


def recommendation_point(row: dict[str, str]) -> np.ndarray:
    """推薦CSVの1行を論理モデルへ渡せる配列へ変換します。"""

    return np.array([[float(row[column]) for column in PARAMETER_COLUMNS]])


def batch_diversity(points: np.ndarray) -> dict[str, float]:
    """探索範囲で正規化したバッチ内の点間距離を集計します。"""

    if len(points) < 2:
        return {"minimum_pairwise_distance": 0.0, "mean_pairwise_distance": 0.0}
    lower = np.array([100.0, 50.0, 10.0])
    upper = np.array([6000.0, 300.0, 1000.0])
    normalized = (points - lower) / (upper - lower)
    differences = normalized[:, None, :] - normalized[None, :, :]
    distances = np.sqrt(np.sum(differences * differences, axis=2))
    upper_triangle = distances[np.triu_indices(len(points), k=1)]
    return {
        "minimum_pairwise_distance": float(np.min(upper_triangle)),
        "mean_pairwise_distance": float(np.mean(upper_triangle)),
    }


def main() -> int:
    """検証trialを作り、推薦と擬似実験の反復を最後まで実行します。"""

    arguments = parse_arguments()
    if arguments.iterations < 1:
        raise ValueError("--iterationsは1以上にしてください。")
    if arguments.recommendations < 1:
        raise ValueError("--recommendationsは1以上にしてください。")

    oracle = load_oracle(arguments.oracle_root)
    true_best = find_true_best(oracle)
    optimizer_root = arguments.optimizer_root.resolve()
    if not (PROJECT_ROOT / "main.py").is_file() or not (
        optimizer_root / "src" / "cli.py"
    ).is_file():
        raise FileNotFoundError(
            f"検証対象のCLI実装が見つかりません: {optimizer_root / 'src' / 'cli.py'}"
        )

    trial_path = optimizer_root / "trials" / arguments.trial
    if trial_path.exists():
        raise FileExistsError(
            f"検証trialは既に存在します。上書きしません: {trial_path}"
        )

    print(f"[準備] trialを作成: {arguments.trial}")
    run_optimizer(optimizer_root, "--new", arguments.trial)

    problem_header = [
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
    write_csv(trial_path / "problem.csv", problem_header, problem_rows())
    run_optimizer(optimizer_root, "--prepare", arguments.trial)

    experiments = initial_experiments(oracle)
    experiment_path = trial_path / "data" / "experiments.csv"
    write_csv(experiment_path, EXPERIMENT_HEADER, experiments)
    initial_best = best_feasible(experiments)

    history: list[dict[str, object]] = []
    batch_history: list[dict[str, object]] = []
    recommendation_path = trial_path / "output" / "recommendations.csv"

    for iteration in range(1, arguments.iterations + 1):
        run_optimizer(
            optimizer_root,
            "--run",
            arguments.trial,
            "--n",
            str(arguments.recommendations),
        )
        recommendations = sorted(
            read_csv(recommendation_path), key=lambda row: int(row["rank"])
        )
        if len(recommendations) != arguments.recommendations:
            raise RuntimeError(
                f"推薦件数が不一致です。要求={arguments.recommendations}, "
                f"出力={len(recommendations)}"
            )
        points = np.vstack([recommendation_point(row) for row in recommendations])
        depths, spatters = evaluate_points(oracle, points)

        new_rows: list[dict[str, object]] = []
        for slot, (recommendation, point) in enumerate(
            zip(recommendations, points, strict=True), start=1
        ):
            new_row: dict[str, object] = {
                "experiment_id": f"bo_{iteration:03d}_{slot:02d}",
                "laser_power_w": float(point[0]),
                "spot_diameter_um": float(point[1]),
                "scan_speed_mm_s": float(point[2]),
                "penetration_depth_mm": float(depths[slot - 1]),
                "spatter_level_0_9": int(spatters[slot - 1]),
            }
            new_rows.append(new_row)
            history.append(
                {
                    "iteration": iteration,
                    "slot": slot,
                    "rank": int(recommendation["rank"]),
                    "selection_role": recommendation.get("selection_role", ""),
                    "experiment_id": new_row["experiment_id"],
                    "recommended_parameters": {
                        column: new_row[column] for column in PARAMETER_COLUMNS
                    },
                    "predicted_depth_mean": float(
                        recommendation[f"{OBJECTIVE_COLUMN}_mean"]
                    ),
                    "predicted_depth_std": float(
                        recommendation[f"{OBJECTIVE_COLUMN}_std"]
                    ),
                    "predicted_feasibility_probability": float(
                        recommendation["feasibility_probability"]
                    ),
                    "actual_depth": new_row[OBJECTIVE_COLUMN],
                    "actual_spatter": new_row[CONSTRAINT_COLUMN],
                    "actual_feasible": int(new_row[CONSTRAINT_COLUMN]) <= 4,
                }
            )

        # 全条件を同じ学習状態で評価した後に、一括して実験CSVへ反映します。
        experiments.extend(new_rows)
        write_csv(experiment_path, EXPERIMENT_HEADER, experiments)
        current_best = best_feasible(experiments)
        assert current_best is not None
        diversity = batch_diversity(points)
        batch_history.append(
            {
                "iteration": iteration,
                "recommendation_count": len(recommendations),
                "auto_diversity_weight": float(
                    recommendations[0].get("auto_diversity_weight", 0.0)
                ),
                "auto_uncertainty_signal": float(
                    recommendations[0].get("auto_uncertainty_signal", 0.0)
                ),
                "auto_coverage_signal": float(
                    recommendations[0].get("auto_coverage_signal", 0.0)
                ),
                "feasibility_guard_active": str(
                    recommendations[0].get("feasibility_guard_active", "False")
                ).lower()
                == "true",
                **diversity,
                "feasible_count": int(np.sum(spatters <= 4)),
                "best_feasible_depth_so_far": current_best[OBJECTIVE_COLUMN],
                "best_feasible_parameters_so_far": {
                    column: current_best[column] for column in PARAMETER_COLUMNS
                },
            }
        )
        print(
            f"[反復 {iteration:02d}] 推薦={len(points)}件 "
            f"バッチ内最小距離={diversity['minimum_pairwise_distance']:.4f} "
            f"制約適合={int(np.sum(spatters <= 4))}/{len(points)} "
            f"最良深さ={float(current_best[OBJECTIVE_COLUMN]):.6g} mm"
        )

    # 最後に追加した擬似実験まで反映した推薦CSVを残します。
    run_optimizer(
        optimizer_root,
        "--run",
        arguments.trial,
        "--n",
        str(arguments.recommendations),
    )
    final_best = best_feasible(experiments)
    assert initial_best is not None and final_best is not None
    optimum_depth = float(true_best[OBJECTIVE_COLUMN])
    final_depth = float(final_best[OBJECTIVE_COLUMN])

    summary = {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "oracle_root": str(arguments.oracle_root.resolve()),
        "optimizer_root": str(optimizer_root),
        "trial": arguments.trial,
        "test_problem": {
            "objective": "penetration_depth_mmを最大化",
            "constraint": "spatter_level_0_9 <= 4",
        },
        "initial_experiment_count": 27,
        "bayesian_iterations": arguments.iterations,
        "recommendations_per_iteration": arguments.recommendations,
        "added_experiment_count": arguments.iterations * arguments.recommendations,
        "final_experiment_count": len(experiments),
        "true_best": true_best,
        "initial_best": initial_best,
        "final_best": final_best,
        "initial_optimum_ratio": float(initial_best[OBJECTIVE_COLUMN]) / optimum_depth,
        "final_optimum_ratio": final_depth / optimum_depth,
        "optimum_found": all(
            float(final_best[column]) == float(true_best[column])
            for column in PARAMETER_COLUMNS
        ),
        "feasible_recommendation_count": sum(
            1 for item in history if item["actual_feasible"]
        ),
        "duplicate_recommendation_count": len(history)
        - len(
            {
                tuple(item["recommended_parameters"][column] for column in PARAMETER_COLUMNS)
                for item in history
            }
        ),
        "batch_history": batch_history,
        "history": history,
    }

    # 比較結果は各バージョン内へ散らさず、共通validation配下へ集約します。
    results_root = VALIDATION_ROOT / "results" / optimizer_root.name
    results_root.mkdir(parents=True, exist_ok=True)
    summary_path = results_root / f"{arguments.trial}.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\n[完了]")
    print(f"初期最良 / 真の最良: {summary['initial_optimum_ratio']:.2%}")
    print(f"最終最良 / 真の最良: {summary['final_optimum_ratio']:.2%}")
    print(f"真の最良条件を発見: {summary['optimum_found']}")
    print(f"検証結果: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
