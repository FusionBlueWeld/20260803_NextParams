"""流入状態を変えた塗工oracle実験でv004 trialとbundle再読込を検証する。"""

from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

from validation.multistage.functional_coating.coating.physics_model import evaluate_model


from ..settings import PROJECT_ROOT as ROOT
V004 = ROOT / "v004"

PARAMETERS = [
    ("coating_gap_um", "um", 120.0, 300.0, 90.0),
    ("line_speed_m_min", "m/min", 5.0, 30.0, 12.5),
    ("web_tension_n", "N", 40.0, 160.0, 60.0),
    ("incoming_viscosity_pa_s", "Pa·s", 0.8, 3.0, 1.1),
    ("incoming_solids_fraction", "fraction", 0.35, 0.65, 0.15),
    ("incoming_bubble_fraction", "fraction", 0.0, 0.03, 0.015),
]
OUTPUTS = [
    ("wet_thickness_um", "um", "monitor", "", ""),
    ("wet_solvent_g_m2", "g/m2", "monitor", "", ""),
    ("wet_solids_g_m2", "g/m2", "monitor", "", ""),
    ("solids_fraction", "fraction", "monitor", "", ""),
    ("thickness_cv_fraction", "fraction", "constraint", "less_equal", 0.08),
    ("coating_defect_index", "0-1", "constraint", "less_equal", 0.25),
    ("capillary_number", "dimensionless", "monitor", "", ""),
    ("coated_solids_rate_kg_m_h", "kg/(m·h)", "objective", "maximize", ""),
]


def _write_csv(path: Path, header: list[str], rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=header)
        writer.writeheader()
        writer.writerows(rows)


def _command(*args: str) -> dict[str, object]:
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    result = subprocess.run([sys.executable, str(ROOT / "main.py"), "--version", "v004", *args], cwd=ROOT, env=env, capture_output=True, text=True, encoding="utf-8", timeout=600)
    if result.returncode:
        raise RuntimeError(result.stderr or result.stdout)
    return {"args": list(args), "returncode": result.returncode, "stdout": result.stdout, "stderr": result.stderr}


def run(trial_name: str, report_path: Path, samples: int = 96, seed: int = 7) -> dict[str, object]:
    """新規v004 trialを学習させ、保存したbundleの予測・状態入力を検証します。"""

    trial = V004 / "trials" / trial_name
    if trial.exists() or report_path.exists():
        raise FileExistsError("trial/report already exists")
    commands = [_command("--new", trial_name)]
    problem_rows = [
        {"column": name, "display_name": name, "unit": unit, "role": "parameter", "direction": "", "lower": lower, "upper": upper, "step": step, "target": ""}
        for name, unit, lower, upper, step in PARAMETERS
    ] + [
        {"column": name, "display_name": name, "unit": unit, "role": role, "direction": direction, "lower": "", "upper": "", "step": "", "target": target}
        for name, unit, role, direction, target in OUTPUTS
    ]
    _write_csv(trial / "problem.csv", ["column", "display_name", "unit", "role", "direction", "lower", "upper", "step", "target"], problem_rows)
    connection = {
        "schema_version": "1.0", "stage_id": "coating",
        "controls": [item[0] for item in PARAMETERS[:3]],
        "incoming_context": [item[0] for item in PARAMETERS[3:]],
        "connector_outputs": [item[0] for item in OUTPUTS],
        "current_incoming_context": {
            "incoming_viscosity_pa_s": 1.6,
            "incoming_solids_fraction": 0.50,
            "incoming_bubble_fraction": 0.005,
        },
    }
    (trial / "stage_connection.json").write_text(json.dumps(connection, ensure_ascii=False, indent=2), encoding="utf-8")
    commands.append(_command("--prepare", trial_name))
    rng = np.random.default_rng(seed)
    unit = (np.arange(samples)[:, None] + rng.random((samples, len(PARAMETERS)))) / samples
    for column in range(unit.shape[1]):
        rng.shuffle(unit[:, column])
    raw = {name: lower + unit[:, index] * (upper - lower) for index, (name, _, lower, upper, _) in enumerate(PARAMETERS)}
    truth = evaluate_model(**raw)
    rows = []
    for index in range(samples):
        row = {"experiment_id": f"oracle_{index + 1:03d}"}
        row.update({name: float(value[index]) for name, value in raw.items()})
        row.update({name: float(np.asarray(value)[index]) for name, value in truth.items()})
        rows.append(row)
    header = ["experiment_id", *[item[0] for item in PARAMETERS], *[item[0] for item in OUTPUTS]]
    _write_csv(trial / "data" / "experiments.csv", header, rows)
    commands.append(_command("--run", trial_name, "--n", "3"))
    bundles = sorted((trial / "output" / "stage_bundles").glob("stage_bundle_run_*"))
    if len(bundles) != 1:
        raise AssertionError("run対応stage bundleが1件ではありません")
    sys.path.insert(0, str(V004))
    from src.stage_bundle import load_stage_bundle
    manifest, predictor = load_stage_bundle(bundles[0])
    axes = [np.asarray([140.0, 210.0, 280.0]), np.asarray([8.0, 18.0, 28.0]), np.asarray([60.0, 100.0, 140.0])]
    points = np.asarray(np.meshgrid(*axes, indexing="ij")).reshape(3, -1).T
    inputs = {name: points[:, index] for index, name in enumerate(connection["controls"])}
    inputs.update({name: value for name, value in connection["current_incoming_context"].items()})
    predicted = predictor.predict_arrays(inputs)
    expected = evaluate_model(**inputs)
    errors = {}
    for name, value in expected.items():
        span = max(float(np.ptp(value)), float(np.mean(np.abs(value))), 1e-12)
        rmse = float(np.sqrt(np.mean((predicted["outputs"][name]["mean"] - value) ** 2)))
        errors[name] = {"rmse": rmse, "normalized_rmse": rmse / span}
    _, predictor_again = load_stage_bundle(bundles[0])
    repeated = predictor_again.predict_arrays(inputs)
    reload_max_error = max(float(np.max(np.abs(predicted["outputs"][name]["mean"] - repeated["outputs"][name]["mean"]))) for name in expected)
    with (trial / "output" / "recommendations.csv").open(encoding="utf-8-sig", newline="") as file:
        recommendations = list(csv.DictReader(file))
    incoming_fixed = all(all(abs(float(row[name]) - value) < 1e-12 for name, value in connection["current_incoming_context"].items()) for row in recommendations)
    report = {
        "status": "complete", "samples": samples, "seed": seed,
        "bundle": str(bundles[0].relative_to(ROOT)), "bundle_stage_id": manifest["stage_id"],
        "recommendation_count": len(recommendations), "incoming_context_fixed": incoming_fixed,
        "reload_maximum_absolute_error": reload_max_error,
        "grid_prediction_errors": errors,
        "maximum_normalized_rmse": max(item["normalized_rmse"] for item in errors.values()),
        "commands": commands,
    }
    if not incoming_fixed or reload_max_error != 0 or not all(np.isfinite(item["normalized_rmse"]) for item in errors.values()):
        raise AssertionError(report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit("usage: python -m validation.multistage.src.checks.v004_validation TRIAL REPORT.json")
    print(json.dumps(run(sys.argv[1], Path(sys.argv[2])), ensure_ascii=True, indent=2))
