"""学習済みv004工程をrシリーズから安全に再利用するstage bundle。"""

from __future__ import annotations

import hashlib
import importlib
import json
import pickle
import shutil
import time
from dataclasses import asdict
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from .connection import SCHEMA_VERSION, StageConnection
from .preprocessing import normalize_parameters
from .settings import ProblemDefinition


BUNDLE_SCHEMA_VERSION = "1.0"


def _publish_directory(temporary: Path, destination: Path) -> None:
    """Windowsの短時間ファイルロックを吸収してbundleを公開します。"""
    if destination.exists():
        shutil.rmtree(destination)
    last_error: OSError | None = None
    for attempt in range(5):
        try:
            temporary.rename(destination)
            return
        except OSError as error:
            last_error = error
            if attempt < 4:
                time.sleep(0.1 * (attempt + 1))
    assert last_error is not None
    raise last_error


class StagePredictor:
    """変数順・正規化範囲とHybridモデルを一体で保持する予測口。"""

    def __init__(self, problem: ProblemDefinition, model, connection: StageConnection):
        self.problem = problem
        self.model = model
        self.connection = connection

    @property
    def parameter_names(self) -> list[str]:
        return [item.column for item in self.problem.parameters]

    def predict_arrays(self, values: Mapping[str, object]) -> dict[str, object]:
        expected = set(self.parameter_names)
        supplied = set(values)
        if supplied != expected:
            missing = sorted(expected - supplied)
            extra = sorted(supplied - expected)
            raise ValueError(f"予測入力が一致しません: missing={missing}, extra={extra}")
        arrays = np.broadcast_arrays(*[np.asarray(values[name], dtype=float) for name in self.parameter_names])
        shape = arrays[0].shape
        raw = np.column_stack([array.reshape(-1) for array in arrays])
        for index, definition in enumerate(self.problem.parameters):
            if not np.all(np.isfinite(raw[:, index])):
                raise ValueError(f"{definition.column}にNaNまたはInfがあります。")
            if np.any(raw[:, index] < definition.lower) or np.any(raw[:, index] > definition.upper):
                raise ValueError(f"{definition.column}が適用範囲外です。")
        prediction = self.model.predict(normalize_parameters(raw, self.problem))
        outputs = {
            name: {
                "mean": item["hybrid_mean"].reshape(shape),
                "std": item["hybrid_std"].reshape(shape),
            }
            for name, item in prediction.results.items()
        }
        support = prediction.support.reshape(shape)
        return {"outputs": outputs, "support": support, "support_class": np.where(support >= 0.5, "SUPPORTED", "EXTRAPOLATION")}

    def predict(self, *, controls: Mapping[str, float], incoming_context: Mapping[str, float]) -> dict[str, object]:
        if set(controls) != set(self.connection.controls):
            raise ValueError("controlsの項目がstage bundle定義と一致しません。")
        if set(incoming_context) != set(self.connection.incoming_context):
            raise ValueError("incoming_contextの項目がstage bundle定義と一致しません。")
        result = self.predict_arrays({**controls, **incoming_context})
        return {
            "outputs": {name: {key: float(np.asarray(value)) for key, value in item.items()} for name, item in result["outputs"].items()},
            "support": float(np.asarray(result["support"])),
            "support_class": str(np.asarray(result["support_class"])),
            "uncertainty_kind": "model_predictive_unseparated",
        }


class OracleStagePredictor:
    """接続配線の正解照合に限って使う決定論的物理oracleアダプター。"""

    def __init__(self, module_name: str, controls: Sequence[str], incoming_context: Sequence[str], bounds: Mapping[str, Sequence[float]]):
        self.module_name = module_name
        self.controls = tuple(controls)
        self.incoming_context = tuple(incoming_context)
        self.bounds = {name: tuple(map(float, value)) for name, value in bounds.items()}

    def predict_arrays(self, values: Mapping[str, object]) -> dict[str, object]:
        expected = set((*self.controls, *self.incoming_context))
        if set(values) != expected:
            raise ValueError(f"oracle入力が一致しません: required={sorted(expected)}")
        for name, value in values.items():
            array = np.asarray(value, dtype=float)
            lower, upper = self.bounds[name]
            if not np.all(np.isfinite(array)) or np.any(array < lower) or np.any(array > upper):
                raise ValueError(f"{name}がoracle適用範囲外です。")
        evaluate = importlib.import_module(self.module_name).evaluate_model
        outputs = evaluate(**values)
        first = np.asarray(next(iter(outputs.values())))
        return {
            "outputs": {name: {"mean": np.asarray(value), "std": np.zeros_like(np.asarray(value), dtype=float)} for name, value in outputs.items()},
            "support": np.ones(first.shape, dtype=float),
            "support_class": np.full(first.shape, "ORACLE", dtype=object),
        }


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def export_stage_bundle(trial: Path, run_id: int | str, problem: ProblemDefinition, model, connection: StageConnection) -> Path:
    """runと対応した自己完結bundleを一時フォルダ経由で公開します。"""

    run_label = f"{run_id:03d}" if isinstance(run_id, int) else str(run_id).removeprefix("run_")
    destination = trial / "output" / "stage_bundles" / f"stage_bundle_run_{run_label}"
    temporary = destination.with_name(destination.name + ".tmp")
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)
    predictor_path = temporary / "predictor.pkl"
    with predictor_path.open("wb") as file:
        pickle.dump(StagePredictor(problem, model, connection), file, protocol=pickle.HIGHEST_PROTOCOL)
    variables = []
    for item in [*problem.parameters, *problem.result_variables]:
        roles = []
        if item.column in connection.controls:
            roles.append("control")
        if item.column in connection.incoming_context:
            roles.append("incoming_context")
        if item in problem.result_variables:
            roles.append("predicted_output")
        if item.column == problem.objective.column:
            roles.append("local_objective")
        if item in problem.constraints:
            roles.append("local_constraint")
        if item in problem.monitors:
            roles.append("monitor")
        row = asdict(item)
        row["roles"] = roles
        variables.append(row)
    (temporary / "variables.json").write_text(json.dumps(variables, ensure_ascii=False, indent=2), encoding="utf-8")
    manifest = {
        "bundle_schema_version": BUNDLE_SCHEMA_VERSION,
        "connection_schema_version": SCHEMA_VERSION,
        "stage_id": connection.stage_id,
        "run_id": run_id,
        "controls": list(connection.controls),
        "incoming_context": list(connection.incoming_context),
        "predicted_outputs": [item.column for item in problem.result_variables],
        "connector_outputs": list(connection.connector_outputs),
        "input_units": {item.column: item.unit for item in problem.parameters},
        "output_units": {item.column: item.unit for item in problem.result_variables},
        "local_objective": {"name": problem.objective.column, "direction": problem.objective.direction},
        "local_constraints": [
            {"name": item.column, "direction": item.direction, "target": item.target}
            for item in problem.constraints
        ],
        "capabilities": {"mean_prediction": True, "model_predictive_std": True, "joint_sampling": False, "calibrated_process_probability": False},
        "predictor": "predictor.pkl",
    }
    (temporary / "stage_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    (temporary / "uncertainty_schema.json").write_text(json.dumps({"std_kind": "model_predictive_unseparated", "joint_sampling": False, "warning": "工程ばらつき・測定誤差・出力間相関は未分離"}, ensure_ascii=False, indent=2), encoding="utf-8")
    provenance = {"files": {name: _sha256(trial / name) for name in ("problem.csv", "data/experiments.csv", "stage_connection.json")}}
    (temporary / "provenance.json").write_text(json.dumps(provenance, ensure_ascii=False, indent=2), encoding="utf-8")
    _publish_directory(temporary, destination)
    return destination


def load_stage_bundle(path: Path) -> tuple[dict[str, object], StagePredictor | OracleStagePredictor]:
    manifest = json.loads((path / "stage_manifest.json").read_text(encoding="utf-8"))
    if manifest.get("bundle_schema_version") != BUNDLE_SCHEMA_VERSION:
        raise ValueError("未対応のstage bundle schemaです。")
    with (path / str(manifest["predictor"])).open("rb") as file:
        predictor = pickle.load(file)
    if not isinstance(predictor, (StagePredictor, OracleStagePredictor)):
        raise ValueError("stage bundleの予測器形式が不正です。")
    return manifest, predictor


def export_oracle_bundle(destination: Path, stage_manifest: Mapping[str, object], module_name: str) -> Path:
    """multistage_validation用oracleを同じbundle契約へ包みます。"""
    if destination.exists():
        raise FileExistsError(destination)
    destination.mkdir(parents=True)
    controls = [item["name"] for item in stage_manifest["controls"]]
    incoming = [item["name"] for item in stage_manifest["incoming_state"]]
    outputs = [item["name"] for item in stage_manifest["outputs"]]
    bounds = {item["name"]: item["range"] for item in [*stage_manifest["controls"], *stage_manifest["incoming_state"]]}
    with (destination / "predictor.pkl").open("wb") as file:
        pickle.dump(OracleStagePredictor(module_name, controls, incoming, bounds), file, protocol=pickle.HIGHEST_PROTOCOL)
    manifest = {
        "bundle_schema_version": BUNDLE_SCHEMA_VERSION,
        "connection_schema_version": SCHEMA_VERSION,
        "stage_id": stage_manifest["id"],
        "model_version": stage_manifest["model_version"],
        "controls": controls,
        "incoming_context": incoming,
        "predicted_outputs": outputs,
        "connector_outputs": outputs,
        "input_units": {item["name"]: item["unit"] for item in [*stage_manifest["controls"], *stage_manifest["incoming_state"]]},
        "output_units": {item["name"]: item["unit"] for item in stage_manifest["outputs"]},
        "local_objective": stage_manifest["local_objective"],
        "local_constraints": stage_manifest["local_constraints"],
        "capabilities": {"mean_prediction": True, "deterministic_oracle": True, "joint_sampling": False, "calibrated_process_probability": False},
        "predictor": "predictor.pkl",
    }
    (destination / "stage_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    (destination / "uncertainty_schema.json").write_text(json.dumps({"std_kind": "deterministic_oracle", "joint_sampling": False}, indent=2), encoding="utf-8")
    (destination / "provenance.json").write_text(json.dumps({"source": module_name, "model_version": stage_manifest["model_version"]}, ensure_ascii=False, indent=2), encoding="utf-8")
    return destination
