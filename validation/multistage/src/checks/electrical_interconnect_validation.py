"""Cu/Alラインの物理監査と、v005→r002→実測追加の閉ループ検証。

入口は `python -m validation.multistage interconnect --help`。
本体アルゴリズムとoracleを分離し、正解値の流出を検証側に限定します。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import platform
import sys
import time
import tracemalloc

import numpy as np

from ..settings import PROJECT_ROOT
from .interconnect_data import generate_workpieces, physical_contract, sample_controls, train_bundles
from .interconnect_lifecycle import candidate_confirmation_loop, experiment_loop, recovery_checks
from .interconnect_history import historical_warm_start
from .interconnect_metrics import compact_search, individual_greedy, oracle_predictors, prediction_errors, random_reference


def jsonable(value):
    """数値配列を有限なJSONへ変換します。未評価のNaNはnullとして保存します。"""
    if isinstance(value, dict):
        return {str(name): jsonable(item) for name, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return jsonable(value.tolist())
    if isinstance(value, np.generic):
        return jsonable(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def source_hashes() -> dict:
    """改行コードを揃えたsource hashを保存し、検証した実装を識別します。"""
    paths = [*PROJECT_ROOT.joinpath("r002/src").glob("*.py"),
             *PROJECT_ROOT.joinpath("v005/src").glob("*.py"),
             *PROJECT_ROOT.joinpath("validation/multistage/electrical_interconnect").glob("*.py"),
             *Path(__file__).parent.glob("interconnect_*.py"), Path(__file__)]
    return {path.relative_to(PROJECT_ROOT).as_posix(): hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
            for path in sorted(set(paths))}


def audit(samples: int = 1024, seed: int = 0, levels: int = 4) -> dict:
    """各工程の条件数・実施可能な払い出し数と、異材/同材の違いを確認します。"""
    from ...electrical_interconnect.pipeline import axes, evaluate_line
    config, manifests, _ = physical_contract()
    dataset = generate_workpieces(samples, seed, levels)
    candidate_axes = axes(levels)
    nominal = {f"{stage}.{name}": manifest["default_inputs"][name]
               for stage, manifest in manifests.items() for name in manifest["controls"]}
    variants = {variant: evaluate_line(nominal, variant=variant)["final"]
                for variant in ("cu_al", "cu_cu", "al_al")}
    return {"kind": "electrical_interconnect_physics_audit", "control_count": len(candidate_axes),
            "levels": levels, "full_grid_count": math.prod(map(len, candidate_axes.values())),
            "stages": {stage: {"controls": len(manifest["controls"]),
                               "incoming_context": len(manifest["incoming_context"]),
                               "standalone_control_grid_count": levels ** len(manifest["controls"]),
                               "admitted_count": dataset["blocks"][stage]["admitted_count"],
                               "passed_count": dataset["blocks"][stage]["passed_count"]}
                       for stage, manifest in manifests.items()},
            "samples": samples, "nominal_variants": variants, "sources": source_hashes(),
            "claim": "synthetic_physics_trends_not_industrial_calibration"}


def benchmark(destination: Path, *, samples: int = 4096, budget: int = 1024,
              seeds: list[int] | None = None, levels: int = 4, sample_cap: int = 256,
              max_seconds: float = 120.0) -> dict:
    """実施可能データから学習し、独立ロット・正解評価・追加実験まで実行します。"""
    from r002.src.api import run_search
    from ...electrical_interconnect.pipeline import axes
    if samples < 32 or budget < 8 or not 8 <= sample_cap <= 512 or levels < 2 or max_seconds <= 0:
        raise ValueError("samples>=32, budget>=8, sample_cap=8..512, levels>=2, max_seconds>0が必要です。")
    started = time.perf_counter()
    sources = source_hashes()
    tracemalloc.start()
    trials = []
    for seed in ([0, 1, 2] if seeds is None else seeds):
        dataset = generate_workpieces(samples, seed, levels)
        config, predictors = train_bundles(dataset, destination / f"seed_{seed}" / "bundles", sample_cap)
        config["candidate_axes"] = axes(levels)
        config["search"] = {"max_evaluations": budget, "population_size": min(64, budget),
                            "batch_size": 64, "max_seconds": max_seconds,
                            "seed": seed, "support_threshold": 0.5}
        holdout = generate_workpieces(max(512, samples // 2), seed + 1000, levels)
        learned = run_search(config, predictors, recommendation_count=4)
        conservative_config = {**config, "search": {**config["search"], "conservative_std_multiplier": 1.0}}
        conservative = run_search(conservative_config, predictors, recommendation_count=4)
        # 正解関数を直接使う探索は上限の比較基準。学習モデルへ情報を戻しません。
        oracle_config = {**config, "search": {**config["search"], "max_evaluations": budget * 4}}
        reference = run_search(oracle_config, oracle_predictors(), recommendation_count=1)
        trial = {"seed": seed,
                 "training": {stage: {"admitted_count": block["admitted_count"],
                                        "passed_count": block["passed_count"],
                                        "retained_samples": predictors[stage][0]["training"]["retained_samples"]}
                              for stage, block in dataset["blocks"].items()},
                 "holdout_errors": prediction_errors(holdout, predictors),
                 "learned_r002": compact_search(learned, config),
                 "conservative_r002": {**compact_search(conservative, conservative_config),
                                       "interpretation": "std_penalty_is_uncalibrated_risk_heuristic"},
                 "random_equal_budget": random_reference(config, budget, seed + 2000, predictors),
                 "oracle_random_reference": random_reference(config, budget, seed + 2000),
                 "individual_greedy": individual_greedy(config, budget, seed + 3000),
                 "oracle_budget_reference": compact_search(reference, config)}
        selected = reference["best"] or learned["best"]
        if selected:
            trial["recovery"] = recovery_checks(config, predictors, selected["controls"], min(512, budget))
            trial["recovery"]["anchor_source"] = "oracle_budget_reference" if reference["best"] else "learned_r002"
        trial["partial_experiment_loop"] = experiment_loop(config, predictors, dataset,
                                                           destination / f"seed_{seed}", sample_cap, seed)
        trial["candidate_confirmation_loop"] = candidate_confirmation_loop(
            config, predictors, dataset, learned, destination / f"seed_{seed}" / "confirmation", sample_cap)
        trial["history_warm_start"] = historical_warm_start(config, dataset,
            destination / f"seed_{seed}" / "history", sample_cap)
        pipeline_path = destination / f"seed_{seed}" / "pipeline.json"
        pipeline_path.parent.mkdir(parents=True, exist_ok=True)
        pipeline_path.write_text(json.dumps(jsonable(config), ensure_ascii=False, indent=2), encoding="utf-8")
        if not trials:
            # 先頭seedは通常のr002 trial入口にも渡せるよう、親にも設定を用意します。
            (destination / "pipeline.json").write_text(json.dumps(jsonable(config), ensure_ascii=False, indent=2), encoding="utf-8")
        trial["pipeline_config"] = str(pipeline_path.resolve())
        trials.append(trial)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return {"kind": "v005_r002_electrical_interconnect_closed_loop", "settings": {
                "requested_samples": samples, "budget_per_search": budget, "levels": levels,
                "sample_cap_per_stage": sample_cap, "seeds": [row["seed"] for row in trials]},
            "runtime": {"python": platform.python_version(), "numpy": np.__version__,
                        "elapsed_seconds": time.perf_counter() - started,
                        "python_tracemalloc_peak_bytes": peak,
                        "memory_scope": "Python_and_NumPy_traced_allocations_not_total_process_RSS"},
            "trials": trials, "sources": sources, "source_files_unchanged_during_run": sources == source_hashes(),
            "limitations": ["合成内部状態ラベルを使った完全情報の模型検証。現場部分観測は品質未確定として別扱い。",
                            "出力標準偏差は未校正。モデル誤差・加工ばらつきの同時伝播による良品確率は未評価。",
                            "大予算oracleも全域の最適解ではない。探索未発見は不可能の証明ではない。",
                            "同材ケースは物理傾向監査。閉ループ模型はCu/Alの主ケースで検証。"]}


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    check = sub.add_parser("audit", help="物理契約・条件数・払い出し・3材質組合せを監査")
    check.add_argument("--samples", type=int, default=1024)
    check.add_argument("--seed", type=int, default=0)
    check.add_argument("--levels", type=int, default=4)
    check.add_argument("--output", type=Path)
    run = sub.add_parser("benchmark", help="学習→全体探索→上流変更→部分実験→更新")
    run.add_argument("--workdir", type=Path, required=True, help="生成bundleと接続設定の新しい保存先")
    run.add_argument("--samples", type=int, default=4096)
    run.add_argument("--budget", type=int, default=1024)
    run.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    run.add_argument("--levels", type=int, default=4)
    run.add_argument("--sample-cap", type=int, default=256)
    run.add_argument("--max-seconds", type=float, default=120.0)
    run.add_argument("--output", type=Path)
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.output and args.output.exists():
            raise ValueError(f"結果ファイルが既に存在します: {args.output}")
        if args.command == "audit":
            payload = audit(args.samples, args.seed, args.levels)
        else:
            if args.workdir.exists():
                raise ValueError(f"workdirは新しい保存先を指定してください: {args.workdir}")
            payload = benchmark(args.workdir, samples=args.samples, budget=args.budget, seeds=args.seeds,
                                levels=args.levels, sample_cap=args.sample_cap, max_seconds=args.max_seconds)
        rendered = json.dumps(jsonable(payload), ensure_ascii=False, indent=2, allow_nan=False)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(rendered, encoding="utf-8")
            print(args.output.resolve())
        else:
            print(rendered)
    except (ValueError, KeyError) as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
