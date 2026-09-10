"""Combine batch=1 and batch=3 v002/v003 evolution results into a 2x2 matrix."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np


def _load(path: Path, expected_batch: int) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    actual = int(payload["config"].get("batch_size", 1))
    if actual != expected_batch:
        raise ValueError(f"{path}: expected batch={expected_batch}, got {actual}")
    return payload


def _mean(rows: list[dict[str, Any]], *keys: str) -> float:
    values: list[float] = []
    for row in rows:
        value: Any = row
        for key in keys:
            value = value[key]
        values.append(float(value))
    return float(np.mean(values))


def _cell(rows: list[dict[str, Any]], budget: int, batch: int, group: str, version: str) -> dict[str, Any]:
    selected = [
        row for row in rows
        if row["version"] == version
        and (group == "overall" or row["complexity_group"] == group)
    ]
    arrivals = [row["arrival_step"] if row["arrival_step"] is not None else budget + 1 for row in selected]
    consumed = [
        (row.get("arrival_consumed_conditions") or row["arrival_step"])
        if row["arrival_step"] is not None else budget + 1
        for row in selected
    ]
    censored_round = int(np.ceil(budget / batch)) + 1
    rounds = [
        (row.get("arrival_round") or int(np.ceil(row["arrival_step"] / batch)))
        if row["arrival_step"] is not None else censored_round
        for row in selected
    ]
    return {
        "batch_size": batch,
        "group": group,
        "version": version,
        "trials": len(selected),
        "arrival_rate": float(np.mean([row["found"] for row in selected])),
        "median_censored_arrival_conditions": float(np.median(arrivals)),
        "median_censored_consumed_conditions": float(np.median(consumed)),
        "median_censored_arrival_round": float(np.median(rounds)),
        "mean_final_regret": _mean(selected, "final_regret"),
        "global_nrmse": _mean(selected, "metrics", "global", "macro_nrmse"),
        "global_nlpd": _mean(selected, "metrics", "global", "macro_nlpd"),
        "global_coverage95": _mean(selected, "metrics", "global", "macro_coverage95"),
        "global_feasible_accuracy": _mean(selected, "metrics", "global", "feasible_accuracy"),
        "near_optimum_k64_nrmse": _mean(selected, "metrics", "near_optimum_k64", "macro_nrmse"),
        "near_optimum_k64_nlpd": _mean(selected, "metrics", "near_optimum_k64", "macro_nlpd"),
        "near_optimum_k64_coverage95": _mean(selected, "metrics", "near_optimum_k64", "macro_coverage95"),
    }


def build_matrix(batch1: dict[str, Any], batch3: dict[str, Any]) -> dict[str, Any]:
    for key in ("simulators", "seeds", "iterations"):
        if batch1["config"][key] != batch3["config"][key]:
            raise ValueError(f"comparison config differs: {key}")
    budget = int(batch1["config"]["iterations"])
    cells = []
    for batch, payload in ((1, batch1), (3, batch3)):
        for group in ("simple", "high_dim", "overall"):
            for version in ("v002", "v003"):
                cells.append(_cell(payload["results"], budget, batch, group, version))
    return {
        "config": {
            "simulators": batch1["config"]["simulators"],
            "seeds": batch1["config"]["seeds"],
            "total_condition_budget": budget,
            "batch_sizes": [1, 3],
        },
        "cells": cells,
    }


def _write_report(path: Path, matrix: dict[str, Any]) -> None:
    lines = [
        "# batch=1/3 × v002/v003 4セル比較",
        "",
        f"総取得条件数={matrix['config']['total_condition_budget']}、seed={matrix['config']['seeds']}。",
        "到達中央値は未到達をbudget+1として含む。",
        "",
    ]
    for group in ("overall", "simple", "high_dim"):
        lines += [f"## {group}", "", "| batch | version | 到達率 | 推薦順位中央値 | 消費条件中央値 | round中央値 | final regret | global NRMSE | k64 NRMSE | NLPD | coverage95 |", "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for cell in matrix["cells"]:
            if cell["group"] != group:
                continue
            lines.append(
                f"| {cell['batch_size']} | {cell['version']} | {cell['arrival_rate']:.3f} | "
                f"{cell['median_censored_arrival_conditions']:.1f} | "
                f"{cell['median_censored_consumed_conditions']:.1f} | "
                f"{cell['median_censored_arrival_round']:.1f} | {cell['mean_final_regret']:.6f} | "
                f"{cell['global_nrmse']:.6f} | {cell['near_optimum_k64_nrmse']:.6f} | "
                f"{cell['global_nlpd']:.3f} | {cell['global_coverage95']:.3f} |"
            )
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def run(batch1_path: Path, batch3_path: Path, output: Path) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing output directory: {output}")
    matrix = build_matrix(_load(batch1_path, 1), _load(batch3_path, 3))
    output.mkdir(parents=True)
    (output / "matrix.json").write_text(json.dumps(matrix, ensure_ascii=False, indent=2), encoding="utf-8")
    with (output / "matrix.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(matrix["cells"][0]))
        writer.writeheader()
        writer.writerows(matrix["cells"])
    _write_report(output / "report.md", matrix)
    return matrix


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="batch=1/3 × v002/v003 4セル比較")
    parser.add_argument("--batch1", type=Path, required=True)
    parser.add_argument("--batch3", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    run(args.batch1, args.batch3, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
