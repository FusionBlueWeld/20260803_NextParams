"""v004 oracle bundleを作り、r001の実trialを物理pipelineと厳密照合する。"""

from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

from validation.multistage.functional_coating.benchmark import CONTROL_AXES, full_line_grid, individual_best_chain


from ..settings import PROJECT_ROOT as ROOT
V004_ROOT = ROOT / "v004"
R001_ROOT = ROOT / "r001"
STAGES = ("coating", "drying", "curing")


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def create_oracle_bundles(bundle_root: Path) -> None:
    sys.path.insert(0, str(V004_ROOT))
    from src.stage_bundle import export_oracle_bundle

    base = "validation.multistage.functional_coating"
    for stage in STAGES:
        manifest = _read_json(ROOT / "validation" / "multistage" / "functional_coating" / stage / "manifest.json")
        export_oracle_bundle(bundle_root / stage, manifest, f"{base}.{stage}.physics_model")


def trial_config(bundle_root: Path) -> dict[str, object]:
    process = _read_json(ROOT / "validation" / "multistage" / "functional_coating" / "process_manifest.json")
    margin_scales = {
        "bond_strength_mpa": 32.0,
        "cure_fraction": 0.20,
        "degradation_fraction": 0.20,
        "final_residual_solvent_pct": 2.0,
        "final_defect_index": 0.25,
        "dimensional_change_um": 6.0,
        "final_thickness_um": 15.0,
    }
    specifications = [
        {**item, "margin_scale": margin_scales[item["name"]]}
        for item in process["final_specifications"]
    ]
    return {
        "schema_version": "1.0",
        "stages": [{"id": stage, "bundle": str((bundle_root / stage).resolve())} for stage in STAGES],
        "connections": process["connections"],
        "external_context": {
            "coating.incoming_viscosity_pa_s": 1.6,
            "coating.incoming_solids_fraction": 0.50,
            "coating.incoming_bubble_fraction": 0.005,
        },
        "candidate_axes": {name: list(values) for name, values in CONTROL_AXES.items()},
        "final_specifications": specifications,
    }


def run(trial_name: str, output: Path) -> dict[str, object]:
    """新規oracle bundleとr001 trialを作り、実CLIの接続結果を物理モデルと比較します。"""

    trial = R001_ROOT / "trials" / trial_name
    bundle_trial = V004_ROOT / "trials" / f"{trial_name}_oracle_stages"
    bundle_root = bundle_trial / "output" / "stage_bundles"
    if trial.exists() or bundle_trial.exists() or output.exists():
        raise FileExistsError("trial/bundle/output already exists")
    create_oracle_bundles(bundle_root)
    trial.mkdir(parents=True)
    (trial / "output").mkdir()
    _write_json(trial / "pipeline.json", trial_config(bundle_root))
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    commands = []
    for args in (("--prepare", trial_name), ("--run", trial_name, "--n", "3")):
        completed = subprocess.run([sys.executable, str(ROOT / "main.py"), "--version", "r001", *args], cwd=ROOT, env=env, capture_output=True, text=True, encoding="utf-8", timeout=600)
        commands.append({"args": list(args), "returncode": completed.returncode, "stdout": completed.stdout, "stderr": completed.stderr})
        if completed.returncode:
            raise RuntimeError(completed.stderr or completed.stdout)
    with (trial / "output" / "system_recommendations.csv").open(encoding="utf-8-sig", newline="") as file:
        selected = next(csv.DictReader(file))
    points, _, oracle = full_line_grid()
    oracle_margin = np.asarray(oracle["final"]["quality_margin"])
    oracle_index = int(np.argmax(oracle_margin))
    selected_controls = np.array([float(selected[name]) for name in CONTROL_AXES])
    selected_index = next((i for i, point in enumerate(points) if np.allclose(point, selected_controls)), None)
    if selected_index is None:
        raise AssertionError("r001 selected an off-grid condition")
    final_errors = {
        name: abs(float(selected[f"final_{name}_mean"]) - float(np.asarray(value)[selected_index]))
        for name, value in oracle["curing"].items()
    }
    local_chain = individual_best_chain()
    report = {
        "status": "complete",
        "candidate_count": len(points),
        "selected_index": selected_index,
        "oracle_best_index": oracle_index,
        "selected_is_oracle_margin_best": selected_index == oracle_index,
        "selected_margin": float(oracle_margin[selected_index]),
        "oracle_best_margin": float(oracle_margin[oracle_index]),
        "maximum_final_output_absolute_error": max(final_errors.values()),
        "final_output_absolute_errors": final_errors,
        "individual_best_chain_feasible": bool(local_chain["final"]["feasible"]),
        "individual_best_chain_margin": float(local_chain["final"]["quality_margin"]),
        "quality_margin_improvement_over_individual_chain": float(oracle_margin[selected_index] - local_chain["final"]["quality_margin"]),
        "commands": commands,
    }
    if (not report["selected_is_oracle_margin_best"] or report["maximum_final_output_absolute_error"] > 1e-9
            or report["quality_margin_improvement_over_individual_chain"] <= 0):
        raise AssertionError(report)
    _write_json(output, report)
    return report


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit("usage: python -m validation.multistage.src.checks.r001_validation TRIAL OUTPUT.json")
    result = run(sys.argv[1], Path(sys.argv[2]))
    print(json.dumps(result, ensure_ascii=False, indent=2))
