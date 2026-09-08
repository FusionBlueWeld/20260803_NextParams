"""個別閉ループ検証に合格したstage bundleをr001へ再接続する。"""

from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

from multistage_validation.functional_coating.benchmark import CONTROL_AXES, individual_best_chain
from multistage_validation.functional_coating.pipeline import DEFAULT_MATERIAL_STATE, evaluate_line
from multistage_validation.r001_validation import trial_config


ROOT = Path(__file__).resolve().parents[1]
R001 = ROOT / "r001"


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def _cli(*args: str) -> str:
    environment = os.environ.copy()
    environment["PYTHONIOENCODING"] = "utf-8"
    result = subprocess.run(
        [sys.executable, str(ROOT / "main.py"), "--version", "r001", *args], cwd=ROOT,
        env=environment, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=600,
    )
    if result.returncode:
        raise RuntimeError(result.stderr or result.stdout)
    return result.stdout


def run(trial_name: str, report_path: Path) -> dict[str, object]:
    bundles = {
        "coating": ROOT / "v004/trials/trial_individual_stage_production2_coating_s0/output/stage_bundles/stage_bundle_run_0016",
        "drying": ROOT / "v004/trials/trial_individual_stage_production2_drying_s0/output/stage_bundles/stage_bundle_run_0016",
        "curing": ROOT / "v004/trials/trial_curing_dense512_curing_s0/output/stage_bundles/stage_bundle_run_0016",
    }
    if any(not path.is_dir() for path in bundles.values()):
        raise FileNotFoundError(bundles)
    trial = R001 / "trials" / trial_name
    if trial.exists() or report_path.exists():
        raise FileExistsError(trial if trial.exists() else report_path)
    _cli("--new", trial_name)
    config = trial_config(ROOT / "unused")
    config["stages"] = [{"id": stage, "bundle": str(path.resolve())} for stage, path in bundles.items()]
    config["process_variation"] = {
        "distribution": "independent_clipped_normal", "samples": 1024,
        "seed": 20260910, "candidate_limit": 128,
        "standard_deviations": {
            "coating_gap_um": 4.0, "line_speed_m_min": 0.35, "web_tension_n": 3.0,
            "air_temperature_c": 2.0, "air_speed_m_s": 0.12, "residence_time_min": 0.25,
            "oven_temperature_c": 2.0, "hold_time_min": 0.5, "nip_pressure_mpa": 0.015,
            "coating.incoming_viscosity_pa_s": 0.08,
            "coating.incoming_solids_fraction": 0.01,
            "coating.incoming_bubble_fraction": 0.0015,
        },
    }
    _write_json(trial / "pipeline.json", config)
    prepare_stdout = _cli("--prepare", trial_name)
    run_stdout = _cli("--run", trial_name, "--n", "10")
    summary = json.loads((trial / "output/run_summary.json").read_text(encoding="utf-8"))
    selected = summary["selected"]
    robust = summary["process_variation"]["selected"]

    def oracle(row: dict[str, object]) -> dict[str, object]:
        conditions = {**DEFAULT_MATERIAL_STATE, **{name: float(row[name]) for name in CONTROL_AXES}}
        result = evaluate_line(conditions)["final"]
        return {
            "feasible": bool(result["feasible"]),
            "quality_margin": float(result["quality_margin"]),
            "conditions": {name: conditions[name] for name in CONTROL_AXES},
        }

    deterministic_oracle = oracle(selected)
    robust_oracle = oracle(robust)
    local = individual_best_chain()
    report = {
        "status": "pass" if deterministic_oracle["feasible"] and deterministic_oracle["quality_margin"] > local["final"]["quality_margin"] else "fail",
        "trial": trial_name,
        "bundles": {stage: str(path.relative_to(ROOT)) for stage, path in bundles.items()},
        "candidate_count": summary["candidate_count"],
        "predicted_feasible_count": summary["predicted_feasible_count"],
        "deterministic_selected": selected,
        "deterministic_oracle": deterministic_oracle,
        "robust_selected": robust,
        "robust_center_oracle": robust_oracle,
        "individual_best_chain_oracle_margin": float(local["final"]["quality_margin"]),
        "deterministic_margin_improvement_over_individual": deterministic_oracle["quality_margin"] - float(local["final"]["quality_margin"]),
        "probability_status": summary["probability_status"],
        "commands": {"prepare_stdout": prepare_stdout, "run_stdout": run_stdout},
    }
    _write_json(report_path, report)
    return report


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit("usage: python -m multistage_validation.validated_chain_retest TRIAL REPORT.json")
    value = run(sys.argv[1], Path(sys.argv[2]))
    print(json.dumps({key: value[key] for key in value if key not in {"deterministic_selected", "commands"}}, ensure_ascii=True, indent=2))
