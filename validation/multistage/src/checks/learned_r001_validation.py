"""3工程を物理oracleデータで学習したv004 bundleに置換し、r001を検証する。"""

from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

from validation.multistage.functional_coating.benchmark import CONTROL_AXES, individual_best_chain
from validation.multistage.functional_coating.pipeline import DEFAULT_MATERIAL_STATE, evaluate_line
from validation.multistage.src.checks.r001_validation import trial_config


from ..settings import PROJECT_ROOT as ROOT
V004 = ROOT / "v004"
R001 = ROOT / "r001"
BASE = ROOT / "validation" / "multistage" / "functional_coating"
STAGES = ("coating", "drying", "curing")


def _json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_csv(path: Path, header: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=header)
        writer.writeheader()
        writer.writerows(rows)


def _cli(version: str, *args: str) -> dict[str, object]:
    environment = os.environ.copy()
    environment["PYTHONIOENCODING"] = "utf-8"
    result = subprocess.run([sys.executable, str(ROOT / "main.py"), "--version", version, *args], cwd=ROOT, env=environment, capture_output=True, text=True, encoding="utf-8", timeout=600)
    if result.returncode:
        raise RuntimeError(result.stderr or result.stdout)
    return {"version": version, "args": list(args), "returncode": result.returncode}


def _lhs(bounds: list[tuple[float, float]], samples: int, seed: int) -> np.ndarray:
    """各入力範囲を等分し、区間内乱数と並べ替えで再現可能な標本を作ります。"""

    rng = np.random.default_rng(seed)
    unit = (np.arange(samples)[:, None] + rng.random((samples, len(bounds)))) / samples
    for column in range(unit.shape[1]):
        rng.shuffle(unit[:, column])
    return np.column_stack([lower + unit[:, i] * (upper - lower) for i, (lower, upper) in enumerate(bounds)])


def _current_contexts() -> dict[str, dict[str, float]]:
    base = {
        **DEFAULT_MATERIAL_STATE,
        "coating_gap_um": 210.0, "line_speed_m_min": 18.0, "web_tension_n": 100.0,
        "air_temperature_c": 90.0, "air_speed_m_s": 5.5, "residence_time_min": 8.0,
        "oven_temperature_c": 140.0, "hold_time_min": 60.0, "nip_pressure_mpa": 0.325,
    }
    result = evaluate_line(base)
    return {
        "coating": {
            "incoming_viscosity_pa_s": base["material_viscosity_pa_s"],
            "incoming_solids_fraction": base["material_solids_fraction"],
            "incoming_bubble_fraction": base["material_bubble_fraction"],
        },
        "drying": {
            "incoming_wet_thickness_um": float(result["coating"]["wet_thickness_um"]),
            "incoming_solids_fraction": float(result["coating"]["solids_fraction"]),
            "incoming_thickness_cv_fraction": float(result["coating"]["thickness_cv_fraction"]),
            "incoming_coating_defect_index": float(result["coating"]["coating_defect_index"]),
        },
        "curing": {
            "incoming_dry_thickness_um": float(result["drying"]["dry_thickness_um"]),
            "incoming_residual_solvent_pct": float(result["drying"]["residual_solvent_pct"]),
            "incoming_internal_stress_mpa": float(result["drying"]["internal_stress_mpa"]),
            "incoming_drying_defect_index": float(result["drying"]["drying_defect_index"]),
            "incoming_thickness_cv_fraction": float(result["drying"]["dry_thickness_cv_fraction"]),
        },
    }


def train_stage(stage: str, trial_name: str, samples: int, seed: int, current: dict[str, float]) -> tuple[Path, list[dict[str, object]]]:
    manifest = _json(BASE / stage / "manifest.json")
    trial = V004 / "trials" / trial_name
    commands = [_cli("v004", "--new", trial_name)]
    inputs = [*manifest["controls"], *manifest["incoming_state"]]
    constraints = {item["name"]: item for item in manifest["local_constraints"]}
    objective = manifest["local_objective"]
    problem_rows = []
    for item in inputs:
        lower, upper = item["range"]
        problem_rows.append({"column": item["name"], "display_name": item["name"], "unit": item["unit"], "role": "parameter", "direction": "", "lower": lower, "upper": upper, "step": (upper - lower) / 2, "target": ""})
    for item in manifest["outputs"]:
        name = item["name"]
        if name == objective["name"]:
            role, direction, target = "objective", objective["direction"], ""
        elif name in constraints:
            role, direction, target = "constraint", constraints[name]["direction"], constraints[name]["target"]
        else:
            role, direction, target = "monitor", "", ""
        problem_rows.append({"column": name, "display_name": name, "unit": item["unit"], "role": role, "direction": direction, "lower": "", "upper": "", "step": "", "target": target})
    problem_header = ["column", "display_name", "unit", "role", "direction", "lower", "upper", "step", "target"]
    _write_csv(trial / "problem.csv", problem_header, problem_rows)
    connection = {
        "schema_version": "1.0", "stage_id": stage,
        "controls": [item["name"] for item in manifest["controls"]],
        "incoming_context": [item["name"] for item in manifest["incoming_state"]],
        "connector_outputs": [item["name"] for item in manifest["outputs"]],
        "current_incoming_context": current,
    }
    _write_json(trial / "stage_connection.json", connection)
    commands.append(_cli("v004", "--prepare", trial_name))
    points = _lhs([tuple(item["range"]) for item in inputs], samples, seed)
    raw = {item["name"]: points[:, index] for index, item in enumerate(inputs)}
    module = __import__(f"validation.multistage.functional_coating.{stage}.physics_model", fromlist=["evaluate_model"])
    outputs = module.evaluate_model(**raw)
    rows = []
    for index in range(samples):
        row: dict[str, object] = {"experiment_id": f"oracle_{index + 1:04d}"}
        row.update({name: float(value[index]) for name, value in raw.items()})
        row.update({name: float(np.asarray(value)[index]) for name, value in outputs.items()})
        rows.append(row)
    experiment_header = ["experiment_id", *raw, *[item["name"] for item in manifest["outputs"]]]
    _write_csv(trial / "data" / "experiments.csv", experiment_header, rows)
    commands.append(_cli("v004", "--run", trial_name, "--n", "3"))
    bundles = sorted((trial / "output" / "stage_bundles").glob("stage_bundle_run_*"))
    if len(bundles) != 1:
        raise AssertionError(f"{stage}: bundle count={len(bundles)}")
    return bundles[0], commands


def run(prefix: str, report_path: Path, samples: int = 128, seed: int = 31) -> dict[str, object]:
    """3工程を個別に学習し、r001で接続して指定先に比較レポートを保存します。"""

    trial_names = {stage: f"trial_{prefix}_{stage}" for stage in STAGES}
    r_trial_name = f"trial_{prefix}_r001"
    if report_path.exists():
        raise FileExistsError("report already exists")
    bundles: dict[str, Path] = {}
    commands: list[dict[str, object]] = []
    contexts = _current_contexts()
    for index, stage in enumerate(STAGES):
        existing = sorted((V004 / "trials" / trial_names[stage] / "output" / "stage_bundles").glob("stage_bundle_run_*"))
        if existing:
            bundles[stage] = existing[-1]
        else:
            bundles[stage], stage_commands = train_stage(stage, trial_names[stage], samples, seed + index, contexts[stage])
            commands.extend(stage_commands)
    r_trial = R001 / "trials" / r_trial_name
    if not r_trial.exists():
        r_trial.mkdir(parents=True)
        (r_trial / "output").mkdir()
        config = trial_config(V004 / "unused")
        config["stages"] = [{"id": stage, "bundle": str(bundles[stage].resolve())} for stage in STAGES]
        _write_json(r_trial / "pipeline.json", config)
        commands.append(_cli("r001", "--prepare", r_trial_name))
    commands.append(_cli("r001", "--run", r_trial_name, "--n", "10"))
    summary = _json(r_trial / "output" / "run_summary.json")
    selected = summary["selected"]
    conditions = {**DEFAULT_MATERIAL_STATE, **{name: selected[name] for name in CONTROL_AXES}}
    oracle = evaluate_line(conditions)
    true_final = {name: float(np.asarray(value)) if np.asarray(value).dtype != bool else bool(np.asarray(value)) for name, value in oracle["final"].items()}
    predicted_errors = {}
    for name, value in oracle["curing"].items():
        column = f"final_{name}_mean"
        predicted_errors[name] = abs(float(selected[column]) - float(np.asarray(value)))
    local = individual_best_chain()
    report = {
        "status": "complete", "samples_per_stage": samples, "seed": seed,
        "bundles": {stage: str(path.relative_to(ROOT)) for stage, path in bundles.items()},
        "candidate_count": summary["candidate_count"],
        "predicted_feasible_count": summary["predicted_feasible_count"],
        "selected_conditions": {name: selected[name] for name in CONTROL_AXES},
        "selected_predicted_feasible": selected["predicted_feasible"],
        "selected_true_feasible": true_final["feasible"],
        "selected_predicted_margin": selected["predicted_quality_margin"],
        "selected_true_margin": true_final["quality_margin"],
        "oracle_individual_best_chain_margin": local["final"]["quality_margin"],
        "true_margin_improvement_over_individual_chain": true_final["quality_margin"] - local["final"]["quality_margin"],
        "maximum_selected_final_output_absolute_error": max(predicted_errors.values()),
        "selected_final_output_absolute_errors": predicted_errors,
        "probability_status": summary["probability_status"],
        "commands": commands,
    }
    _write_json(report_path, report)
    return report


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit("usage: python -m validation.multistage.src.checks.learned_r001_validation PREFIX REPORT.json")
    print(json.dumps(run(sys.argv[1], Path(sys.argv[2])), ensure_ascii=True, indent=2))
