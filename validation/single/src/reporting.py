"""工程・版・seedごとの比較結果をCSV、JSON、Markdownに保存します。

採点済みの値を表示用に整えるだけで、合否や評価指標の再計算はしません。
"""

from pathlib import Path

from .data_loader import write_csv, write_json


def suite_report(directory: Path, results: list[dict]) -> None:
    """同じ条件で比較した結果を表にし、CSV・JSON・Markdownを保存します。"""

    columns = ["simulator", "optimizer", "seed", "noise_fraction", "status", "parameter_count",
               "initial_normalized_regret", "normalized_regret", "normalized_regret_reduction",
               "true_feasible_recommendation_rate", "feasibility_brier_score", "optimum_found", "elapsed_seconds", "error"]
    write_csv(directory / "comparison.csv", columns, [{n: r.get(n, "") for n in columns} for r in results])
    write_json(directory / "comparison.json", results)
    lines = ["# 多工程ベンチマーク", "", "同一工程・seed・ノイズ条件でバージョンを比較します。",
             "regretは候補グリッド上の最良値との差を目的値の全グリッド幅で正規化した値です。小さいほど良好。",
             "解空間の誤差はsummary.jsonのresponse_space_metricsを参照してください。v000は解空間出力なし。", "",
             "|工程|入力数|version|seed|状態|初期regret|最終regret|改善量|制約適合率|",
             "|---|---:|---|---:|---|---:|---:|---:|---:|"]
    def fmt(value):
        return "—" if value is None else f"{value:.4f}"
    for r in results:
        lines.append(f"|{r['simulator']}|{r.get('parameter_count', '—')}|{r['optimizer']}|{r['seed']}|{r['status']}|"
                     f"{fmt(r.get('initial_normalized_regret'))}|{fmt(r.get('normalized_regret'))}|"
                     f"{fmt(r.get('normalized_regret_reduction'))}|{fmt(r.get('true_feasible_recommendation_rate'))}|")
    (directory / "comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

