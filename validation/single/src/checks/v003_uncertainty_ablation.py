"""v003のlegacy不確実性と5-member ensemble+倍率校正を比較する。"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from ..simulators import load
from .v002_v003_evolution import DEFAULT_LOCAL_RADIUS, run_arm


VARIANTS = ("baseline", "ensemble_calibrated")
GROUPS = ("simple", "high_dim", "overall")


def _load_baseline(path: Path, expected_batch: int) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    batch = int(payload["config"].get("batch_size", 1))
    if batch != expected_batch:
        raise ValueError(f"{path}: expected batch={expected_batch}, got {batch}")
    rows = []
    for source in payload["results"]:
        if source["version"] != "v003":
            continue
        row = dict(source)
        row["variant"] = "baseline"
        row["batch_size"] = batch
        rows.append(row)
    return payload, rows


def _mean(rows: list[dict[str, Any]], *keys: str) -> float:
    values = []
    for row in rows:
        value: Any = row
        for key in keys:
            value = value[key]
        values.append(float(value))
    return float(np.mean(values))


def _cell(rows: list[dict[str, Any]], variant: str, batch: int, group: str, budget: int) -> dict[str, Any]:
    selected = [
        row for row in rows
        if row["variant"] == variant
        and row["batch_size"] == batch
        and (group == "overall" or row["complexity_group"] == group)
    ]
    arrivals = [row["arrival_step"] if row["arrival_step"] is not None else budget + 1 for row in selected]
    consumed = [
        (row.get("arrival_consumed_conditions") or row["arrival_step"])
        if row["arrival_step"] is not None else budget + 1
        for row in selected
    ]
    return {
        "variant": variant,
        "batch_size": batch,
        "group": group,
        "trials": len(selected),
        "arrival_rate": float(np.mean([row["found"] for row in selected])),
        "median_censored_arrival_rank": float(np.median(arrivals)),
        "median_censored_consumed_conditions": float(np.median(consumed)),
        "mean_final_regret": _mean(selected, "final_regret"),
        "global_nrmse": _mean(selected, "metrics", "global", "macro_nrmse"),
        "global_nlpd": _mean(selected, "metrics", "global", "macro_nlpd"),
        "global_coverage95": _mean(selected, "metrics", "global", "macro_coverage95"),
        "global_feasible_accuracy": _mean(selected, "metrics", "global", "feasible_accuracy"),
        "near_optimum_k64_nrmse": _mean(selected, "metrics", "near_optimum_k64", "macro_nrmse"),
        "near_optimum_k64_nlpd": _mean(selected, "metrics", "near_optimum_k64", "macro_nlpd"),
        "near_optimum_k64_coverage95": _mean(selected, "metrics", "near_optimum_k64", "macro_coverage95"),
    }


def _paired(rows: list[dict[str, Any]], budget: int) -> list[dict[str, Any]]:
    pairs = []
    keys = sorted({(row["batch_size"], row["simulator"], row["seed"]) for row in rows})
    for batch, simulator, seed in keys:
        arms = {
            row["variant"]: row for row in rows
            if row["batch_size"] == batch and row["simulator"] == simulator and row["seed"] == seed
        }
        if set(arms) != set(VARIANTS):
            continue
        old, new = arms["baseline"], arms["ensemble_calibrated"]
        old_arrival = old["arrival_step"] if old["arrival_step"] is not None else budget + 1
        new_arrival = new["arrival_step"] if new["arrival_step"] is not None else budget + 1
        pairs.append({
            "batch_size": batch,
            "simulator": simulator,
            "complexity_group": old["complexity_group"],
            "seed": seed,
            "baseline_arrival": old_arrival,
            "enhanced_arrival": new_arrival,
            "arrival_delta": new_arrival - old_arrival,
            "baseline_final_regret": old["final_regret"],
            "enhanced_final_regret": new["final_regret"],
            "final_regret_delta": new["final_regret"] - old["final_regret"],
            "global_nrmse_delta": new["metrics"]["global"]["macro_nrmse"] - old["metrics"]["global"]["macro_nrmse"],
            "global_nlpd_delta": new["metrics"]["global"]["macro_nlpd"] - old["metrics"]["global"]["macro_nlpd"],
            "global_coverage95_delta": new["metrics"]["global"]["macro_coverage95"] - old["metrics"]["global"]["macro_coverage95"],
            "near_optimum_k64_nrmse_delta": new["metrics"]["near_optimum_k64"]["macro_nrmse"] - old["metrics"]["near_optimum_k64"]["macro_nrmse"],
            "near_optimum_k64_nlpd_delta": new["metrics"]["near_optimum_k64"]["macro_nlpd"] - old["metrics"]["near_optimum_k64"]["macro_nlpd"],
            "near_optimum_k64_coverage95_delta": new["metrics"]["near_optimum_k64"]["macro_coverage95"] - old["metrics"]["near_optimum_k64"]["macro_coverage95"],
        })
    return pairs


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_report(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# v003 uncertainty upgrade ablation",
        "",
        "baselineは旧v003（NN 1、scale 1）。enhancedはNN 5-member ensemble、scale 2。",
        f"総追加条件数={payload['config']['iterations']}、seed={payload['config']['seeds']}。",
        "",
    ]
    for group in GROUPS:
        lines += [
            f"## {group}", "",
            "| batch | variant | arrival | consumed median | regret | NRMSE | NLPD | coverage | k64 NRMSE | k64 NLPD | k64 coverage |",
            "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for cell in payload["cells"]:
            if cell["group"] != group:
                continue
            lines.append(
                f"| {cell['batch_size']} | {cell['variant']} | {cell['arrival_rate']:.3f} | "
                f"{cell['median_censored_consumed_conditions']:.1f} | {cell['mean_final_regret']:.6f} | "
                f"{cell['global_nrmse']:.6f} | {cell['global_nlpd']:.3f} | {cell['global_coverage95']:.3f} | "
                f"{cell['near_optimum_k64_nrmse']:.6f} | {cell['near_optimum_k64_nlpd']:.3f} | "
                f"{cell['near_optimum_k64_coverage95']:.3f} |"
            )
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def _source_hash(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted((root / "src").rglob("*.py")):
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(path.read_bytes().replace(b"\r\n", b"\n"))
    return digest.hexdigest()


def run(baseline_batch1: Path, baseline_batch3: Path, output: Path) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing output directory: {output}")
    old1, baseline1 = _load_baseline(baseline_batch1, 1)
    old3, baseline3 = _load_baseline(baseline_batch3, 3)
    for key in ("simulators", "seeds", "iterations"):
        if old1["config"][key] != old3["config"][key]:
            raise ValueError(f"baseline configs differ: {key}")
    config = old1["config"]
    enhanced = []
    for batch in (1, 3):
        for simulator_id in config["simulators"]:
            simulator = load(simulator_id)
            for seed in config["seeds"]:
                row = run_arm(
                    simulator, "v003", int(seed), int(config["iterations"]),
                    float(config.get("local_radius", DEFAULT_LOCAL_RADIUS)), batch,
                )
                row["variant"] = "ensemble_calibrated"
                enhanced.append(row)
    rows = baseline1 + baseline3 + enhanced
    budget = int(config["iterations"])
    cells = [
        _cell(rows, variant, batch, group, budget)
        for group in GROUPS for batch in (1, 3) for variant in VARIANTS
    ]
    repo_root = Path(__file__).resolve().parents[4]
    payload = {
        "config": {
            "simulators": config["simulators"], "seeds": config["seeds"],
            "iterations": budget, "batch_sizes": [1, 3],
            "baseline": {"ensemble_size": 1, "calibration_scale": 1.0},
            "ensemble_calibrated": {"ensemble_size": 5, "calibration_scale": 2.0},
            "baseline_source_hash": old1["version_provenance"]["v003"]["source_hash"],
            "enhanced_source_hash": _source_hash(repo_root / "v003"),
        },
        "cells": cells,
        "paired": _paired(rows, budget),
        "results": rows,
    }
    output.mkdir(parents=True)
    (output / "raw_results.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    _write_csv(output / "cells.csv", cells)
    _write_csv(output / "paired.csv", payload["paired"])
    _write_report(output / "report.md", payload)
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="v003 ensemble uncertainty ablation")
    parser.add_argument("--baseline-batch1", type=Path, required=True)
    parser.add_argument("--baseline-batch3", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    run(args.baseline_batch1, args.baseline_batch3, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
