"""v003 検証結果の静的可視化。

matplotlib は必須依存にしません。未導入環境では PNG を作らず、インストール
方法と読み込めなかった入力を標準出力へ表示して正常終了します。これにより、
CLI の動作検証をグラフ環境の有無から分離できます。
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

PARAMETERS = ("laser_power_w", "spot_diameter_um", "scan_speed_mm_s")
OBJECTIVE = "penetration_depth_mm"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def numeric(row: dict[str, str], names: tuple[str, ...]) -> float:
    for name in names:
        value = row.get(name, "")
        if value not in (None, ""):
            try:
                return float(value)
            except ValueError:
                pass
    return float("nan")


def prediction(row: dict[str, str], column: str) -> float:
    return numeric(
        row,
        (
            f"{column}_mean",
            f"{column}_hybrid_mean",
            f"{column}_predicted",
            f"{column}_nn_pred",
            column,
        ),
    )


def newest_response(summary: dict[str, object]) -> Path | None:
    value = str(summary.get("response_space", ""))
    if value and Path(value).is_file():
        return Path(value)
    trial = Path(str(summary.get("optimizer_root", ""))) / "trials" / str(summary.get("trial", ""))
    candidates = list((trial / "output" / "response_spaces").glob("*.csv"))
    return max(candidates, key=lambda path: path.stat().st_mtime_ns) if candidates else None


def plot(summary_path: Path, output_path: Path) -> bool:
    """収束履歴、予測空間、データ支持度を1枚へまとめます。"""

    try:
        import matplotlib.pyplot as plt
    except (ImportError, OSError) as error:
        print(f"SKIP: matplotlib is unavailable ({error}); no PNG was created.")
        return False

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    history = list(summary.get("history", []))
    response_path = newest_response(summary)
    response = read_csv(response_path) if response_path else []
    figure, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)

    # 反復ごとの実測最良値。batch の場合にも rank/slot を潰して扱います。
    initial_best = float(
        summary.get("initial_best", {}).get(OBJECTIVE, float("nan"))
    )
    best_values: list[float] = [initial_best]
    iterations: list[int] = [0]
    running = initial_best
    for item in history:
        value = float(item.get("actual_objective", float("nan")))
        if item.get("actual_feasible", False) and np.isfinite(value):
            running = max(running, value)
        best_values.append(running)
        iterations.append(int(item.get("iteration", len(iterations))))
    axes[0, 0].plot(
        iterations, best_values, marker="o", label="best feasible experiment"
    )
    true_best = float(summary.get("true_best", {}).get(OBJECTIVE, float("nan")))
    if np.isfinite(true_best):
        axes[0, 0].axhline(true_best, color="tab:red", linestyle="--", label="oracle global best")
    axes[0, 0].set(title="Convergence", xlabel="iteration", ylabel=OBJECTIVE)
    axes[0, 0].grid(alpha=0.25)
    axes[0, 0].legend()

    # 推薦スコアの変化。v003 で列が無い場合は空のグラフに理由を表示します。
    scores = [float(item.get("recommendation_score", float("nan"))) for item in history]
    score_iterations = [int(item.get("iteration", index + 1)) for index, item in enumerate(history)]
    axes[0, 1].plot(score_iterations, scores, marker=".", color="tab:purple")
    axes[0, 1].set(title="Recommendation score", xlabel="iteration", ylabel="score")
    if scores and all(np.isfinite(scores)) and min(scores) > 0:
        axes[0, 1].set_yscale("log")
    axes[0, 1].grid(alpha=0.25)

    if response:
        x = np.array([numeric(row, (PARAMETERS[0],)) for row in response])
        y = np.array([numeric(row, (PARAMETERS[1],)) for row in response])
        speed = np.array([numeric(row, (PARAMETERS[2],)) for row in response])
        z = np.array([prediction(row, OBJECTIVE) for row in response])
        support = np.array(
            [
                numeric(row, ("gp_support", "data_support", "support"))
                for row in response
            ]
        )
        # 3入力を2次元へ投影するときは、真の最良点に近い速度の断面を使う。
        # これで同じ(p1,p2)へ異なる速度の値が重なり、景色が潰れるのを防ぐ。
        target_speed = float(summary.get("true_best", {}).get(PARAMETERS[2], float("nan")))
        if np.isfinite(target_speed) and np.isfinite(speed).any():
            available_speed = speed[np.isfinite(speed)]
            nearest_speed = available_speed[np.argmin(np.abs(available_speed - target_speed))]
            section = np.isclose(speed, nearest_speed)
        else:
            nearest_speed = float("nan")
            section = np.ones(len(response), dtype=bool)
        x, y, z, support = x[section], y[section], z[section], support[section]
        finite = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
        if np.any(finite):
            scatter = axes[1, 0].scatter(x[finite], y[finite], c=z[finite], s=12, cmap="viridis")
            figure.colorbar(scatter, ax=axes[1, 0], label=f"predicted {OBJECTIVE}")
        section_title = (
            f"Predicted parameter space ({PARAMETERS[2]}={nearest_speed:g})"
            if np.isfinite(nearest_speed)
            else "Predicted parameter space"
        )
        axes[1, 0].set(title=section_title, xlabel=PARAMETERS[0], ylabel=PARAMETERS[1])
        axes[1, 0].grid(alpha=0.15)
        finite_support = np.isfinite(x) & np.isfinite(y) & np.isfinite(support)
        if np.any(finite_support):
            support_plot = axes[1, 1].scatter(
                x[finite_support],
                y[finite_support],
                c=support[finite_support],
                s=12,
                cmap="magma",
                vmin=0,
                vmax=1,
            )
            figure.colorbar(support_plot, ax=axes[1, 1], label="data support")
        axes[1, 1].set(
            title="Data support / confidence",
            xlabel=PARAMETERS[0],
            ylabel=PARAMETERS[1],
        )
        axes[1, 1].grid(alpha=0.15)
    else:
        axes[1, 0].text(0.5, 0.5, "response-space CSV not found", ha="center", va="center")
        axes[1, 1].text(0.5, 0.5, "response-space CSV not found", ha="center", va="center")

    figure.suptitle(f"v003 validation: {summary.get('trial', summary_path.stem)}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=150)
    plt.close(figure)
    print(f"PNG: {output_path}")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="Plot v003 validation summary")
    parser.add_argument("summary", type=Path, help="validation/single/results/v003/*.json")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or args.summary.with_suffix(".png")
    plot(args.summary, output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
