"""CLI: python -m validation.run {list,evaluate,export,audit,run,suite}."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

# Also support python validation/run.py from any working directory.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from validation import benchmark
from validation.simulators import ROOT, PROBLEM_HEADER, available, load, write_csv, write_json


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="6工程共通の仮想実験・最適化検証")
    commands = result.add_subparsers(dest="command", required=True)
    commands.add_parser("list", help="工程・入力・検査値の一覧")
    evaluate = commands.add_parser("evaluate", help="入力条件1件を評価してJSON出力")
    evaluate.add_argument("simulator", choices=available())
    evaluate.add_argument("--set", nargs="+", required=True, metavar="NAME=VALUE")
    export = commands.add_parser("export", help="初期条件または全グリッドの仮想実験をCSVへ保存")
    export.add_argument("simulator", choices=available())
    export.add_argument("--design", choices=["initial", "grid"], default="initial")
    export.add_argument("--output", type=Path, required=True, help="新規の出力フォルダ")
    audit = commands.add_parser("audit", help="各工程の候補・制約付き最良値を確認")
    audit.add_argument("--simulators", nargs="+", choices=available(), default=available())
    audit.add_argument("--output", type=Path, help="新規JSONファイル（省略時は標準出力のみ）")
    run = commands.add_parser("run", help="1工程・1バージョンの閉ループ仮想実験")
    run.add_argument("simulator", choices=available())
    run.add_argument("--version", choices=benchmark.versions(), default="v000")
    run.add_argument("--trial", help="省略時は一意なtrial名")
    run.add_argument("--seed", type=int, default=0)
    suite = commands.add_parser("suite", help="工程×バージョン×seedの比較を一括実行")
    suite.add_argument("--simulators", nargs="+", choices=available(), default=available())
    suite.add_argument("--versions", nargs="+", choices=benchmark.versions(), default=["v000"])
    suite.add_argument("--seeds", nargs="+", type=int, default=[0])
    for cmd in (run, suite):
        cmd.add_argument("--iterations", type=int, default=15)
        cmd.add_argument("--recommendations", type=int, default=1)
        cmd.add_argument("--noise", type=float, default=0, help="検査ノイズ標準偏差 / 各出力の全グリッド幅")
        cmd.add_argument("--output", type=Path, help="新規の結果フォルダ（省略時はvalidation/results/multiphysics配下）")
    return result


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    try:
        if args.command == "list":
            for name in available():
                sim = load(name)
                print(f"{name}: {sim.manifest['name']}\n  入力: {', '.join(sim.parameter_columns)}"
                      f"\n  出力: {', '.join(sim.output_columns)}\n  {sim.objective.direction}: {sim.objective.column}")
            return 0
        if args.command == "evaluate":
            conditions = {}
            for item in args.set:
                name, value = item.split("=", 1)
                if name in conditions:
                    raise ValueError(f"Duplicate input: {name}")
                conditions[name] = float(value)
            sim = load(args.simulator)
            outputs = sim.evaluate(conditions)
            print(json.dumps({"simulator": sim.id, "inputs": conditions,
                              "outputs": {n: v.item() for n, v in outputs.items()},
                              "feasible": bool(sim.feasible(outputs))}, ensure_ascii=False, indent=2, allow_nan=False))
            return 0
        if args.command == "export":
            sim = load(args.simulator)
            points = sim.grid() if args.design == "grid" else sim.initial_design()
            outputs = sim.evaluate_points(points)
            args.output.mkdir(parents=True, exist_ok=False)
            write_csv(args.output / "problem.csv", PROBLEM_HEADER, sim.rows)
            write_csv(args.output / "experiments.csv", sim.experiment_header, sim.experiment_rows(points, outputs, args.design))
            write_json(args.output / "provenance.json", sim.provenance())
            print(args.output.resolve())
            return 0
        if args.command == "audit":
            results = [benchmark.audit(load(name)) for name in args.simulators]
            if args.output:
                if args.output.exists():
                    raise FileExistsError(args.output)
                write_json(args.output, results)
            print(json.dumps(results, ensure_ascii=False, indent=2, allow_nan=False))
            return 0
        seeds = [args.seed] if args.command == "run" else list(dict.fromkeys(args.seeds))
        if (min(seeds) < 0 or args.iterations < 1 or args.recommendations < 1
                or not math.isfinite(args.noise) or args.noise < 0):
            raise ValueError("iterations/recommendations >= 1、seed/noise >= 0 の有限値が必要です")
        token = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S") + "_" + uuid4().hex[:8]
        output = args.output or ROOT / "results" / "multiphysics" / token
        if args.command == "run":
            trial = args.trial or f"trial_{args.simulator}_{token}"
            benchmark.run(load(args.simulator), args.version, trial, output,
                          args.iterations, args.recommendations, args.seed, args.noise)
            print(f"結果: {output.resolve()}")
            return 0
        output.mkdir(parents=True, exist_ok=False)
        results = []
        for name in dict.fromkeys(args.simulators):
            for version in dict.fromkeys(args.versions):
                for seed in seeds:
                    label = f"{name}_{version}_s{seed}"
                    try:
                        summary = benchmark.run(load(name), version, f"trial_{label}_{token}", output / label,
                                                args.iterations, args.recommendations, seed, args.noise)
                    except Exception as error:
                        summary = {"simulator": name, "optimizer": version, "seed": seed,
                                   "noise_fraction": args.noise, "status": "failed", "error": str(error)}
                        print(f"[{label}] FAILED: {error}", file=sys.stderr, flush=True)
                    results.append(summary)
                    benchmark.suite_report(output, results)
        print(f"比較レポート: {output.resolve() / 'comparison.md'}")
        return 1 if any(r["status"] == "failed" for r in results) else 0
    except (ValueError, OSError, RuntimeError, KeyError, subprocess.TimeoutExpired) as error:
        print(f"[エラー] {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
