"""r002へ渡すbundle2.0。数値NPZと検査済みJSONだけを復元する。"""

from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
from typing import Mapping
import uuid
import zipfile
import numpy as np
from .bundle_schema import validate_manifest
from .fast_stage_model import FastStageModel
from .derived_outputs import apply_derived_outputs

BUNDLE_SCHEMA_VERSION = "2.0"
MAX_TRAINING_SAMPLES = 512


def _input_matrix(values, names):
    if set(values) != set(names):
        raise ValueError(f"予測入力が一致しません: missing={sorted(set(names) - set(values))}, extra={sorted(set(values) - set(names))}")
    arrays = np.broadcast_arrays(*[np.asarray(values[name], dtype=float) for name in names])
    shape = arrays[0].shape
    matrix = np.column_stack([array.reshape(-1) for array in arrays])
    if not np.all(np.isfinite(matrix)):
        raise ValueError("予測入力にNaNまたはInfがあります。")
    return matrix, shape


class StagePredictor:
    """適用範囲と多変量支持度を分離する共通の予測口。"""
    def __init__(self, manifest, model):
        self.manifest, self.model = manifest, model
        self.parameter_names = manifest["controls"] + manifest["incoming_context"]
        self.lower = np.array([manifest["input_bounds"][name][0] for name in self.parameter_names])
        self.upper = np.array([manifest["input_bounds"][name][1] for name in self.parameter_names])

    def predict_arrays(self, values: Mapping[str, object]) -> dict:
        raw, shape = _input_matrix(values, self.parameter_names)
        valid = np.all((raw >= self.lower) & (raw <= self.upper), axis=1)
        # 範囲外行は利用不可。極端な外挿による数値overflowも避ける。
        normalized = np.clip((raw - self.lower) / (self.upper - self.lower), -1.0, 2.0)
        if isinstance(self.model, FastStageModel):
            mean, std, support = self.model.predict(normalized)
        else:
            prediction = self.model.predict(normalized)
            mean = np.column_stack([prediction.results[name]["hybrid_mean"] for name in self.manifest["predicted_outputs"]])
            std = np.column_stack([prediction.results[name]["hybrid_std"] for name in self.manifest["predicted_outputs"]])
            support = prediction.support
        mean, std = apply_derived_outputs(mean, std, self.manifest["predicted_outputs"],
                                          self.manifest.get("derived_outputs", {}))
        valid &= np.all(np.isfinite(mean) & np.isfinite(std), axis=1)
        support = np.where(valid, support, 0.0)
        classes = np.where(~valid, "OUT_OF_BOUNDS", np.where(support >= 0.5, "SUPPORTED", "EXTRAPOLATION"))
        return {"outputs": {name: {"mean": mean[:, index].reshape(shape), "std": std[:, index].reshape(shape)}
                            for index, name in enumerate(self.manifest["predicted_outputs"])},
                "support": support.reshape(shape), "support_class": classes.reshape(shape),
                "valid": valid.reshape(shape), "uncertainty_kind": "uncalibrated_model_predictive",
                "derived_output_policy": "point_estimate_identity_uncalibrated_std_sum"
                if self.manifest.get("derived_outputs") else "none"}

    def predict(self, *, controls: Mapping[str, float], incoming_context: Mapping[str, float]) -> dict:
        if set(controls) != set(self.manifest["controls"]) or set(incoming_context) != set(self.manifest["incoming_context"]):
            raise ValueError("操作条件と流入状態の役割が一致しません。")
        return self.predict_arrays({**controls, **incoming_context})


def _save(destination: Path, manifest: dict, arrays: dict, backend: str) -> Path:
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    label = uuid.uuid4().hex
    # 新しいNPZを完成してからmanifestを交換。失敗時に既存契約を壊さない。
    filename = f"predictor_{label}.npz"
    model_path = destination / filename
    with model_path.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    value = validate_manifest(manifest)
    value.update(predictor=filename, backend=backend,
                 predictor_sha256=hashlib.sha256(model_path.read_bytes()).hexdigest())
    value["capabilities"] = {**value.get("capabilities", {}), "mean_prediction": True,
                             "model_predictive_std": True, "joint_sampling": False,
                             "calibrated_process_probability": False,
                             "calibrated_model_uncertainty": False,
                             "knowledge_loss": backend == "snapshot_hybrid"}
    if value.get("derived_outputs"):
        value["capabilities"]["derived_outputs"] = "point_estimate_identity_uncalibrated_std_sum"
    temporary = destination / f".stage_manifest_{label}.tmp"
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(temporary, destination / "stage_manifest.json")
    return destination


def train_stage_bundle(destination: Path, manifest: dict, inputs: Mapping[str, object],
                       outputs: Mapping[str, object], seed: int = 0,
                       max_training_samples: int = 256) -> Path:
    """2～512点に費用を制限して高速模型を学習し、安全なbundleを保存。"""
    value = validate_manifest(manifest)
    names = value["controls"] + value["incoming_context"]
    if len(names) > 64 or len(value["predicted_outputs"]) > 64:
        raise ValueError("高速stage模型は入出力各64軸以下にしてください。")
    if not isinstance(max_training_samples, int) or not 2 <= max_training_samples <= MAX_TRAINING_SAMPLES:
        raise ValueError("max_training_samplesは2～512にしてください。")
    raw, shape = _input_matrix(inputs, names)
    if set(outputs) != set(value["predicted_outputs"]):
        raise ValueError("学習出力がmanifestと一致しません。")
    try:
        y = np.column_stack([np.broadcast_to(np.asarray(outputs[name], dtype=float), shape).reshape(-1)
                             for name in value["predicted_outputs"]])
    except ValueError as error:
        raise ValueError("学習入出力の行数が一致しません。") from error
    if len(raw) < 2 or not np.all(np.isfinite(y)):
        raise ValueError("学習には2行以上の有限な出力が必要です。")
    lower = np.array([value["input_bounds"][name][0] for name in names])
    upper = np.array([value["input_bounds"][name][1] for name in names])
    if np.any(raw < lower) or np.any(raw > upper):
        raise ValueError("学習点がinput_boundsの外にあります。")
    source_count = len(raw)
    fingerprint = hashlib.sha256(raw.tobytes() + y.tobytes()).hexdigest()
    if len(raw) > max_training_samples:
        # 最近の確認実験を保持し、追加測定が間引きで全て消えないようにする。
        recent = min(32, max_training_samples // 4)
        rng = np.random.default_rng(seed)
        indices = np.sort(np.concatenate((rng.choice(len(raw) - recent, max_training_samples - recent, replace=False),
                                          np.arange(len(raw) - recent, len(raw)))))
        raw, y = raw[indices], y[indices]
    model = FastStageModel.fit((raw - lower) / (upper - lower), y, seed=seed)
    value["training"] = {"source_samples": source_count, "retained_samples": len(raw),
                         "sample_cap": max_training_samples, "seed": seed,
                         "data_sha256": fingerprint, "fit_seconds": model.fit_seconds,
                         "backend": "random_feature_nn_residual_kernel",
                         "noise_separated": False, "ensemble_members": 3,
                         "input_feature_normalization": "observed_std_floor_0.02_global_bounds_units"}
    return _save(destination, value, model.arrays(), "fast_stage")


def load_stage_bundle(path: Path) -> tuple[dict, StagePredictor]:
    """外部コードを実行せず、hashと数値配列を確認して復元。"""
    path = Path(path)
    manifest = validate_manifest(json.loads((path / "stage_manifest.json").read_text(encoding="utf-8-sig")))
    filename = manifest.get("predictor")
    if not isinstance(filename, str) or Path(filename).name != filename or not filename.endswith(".npz"):
        raise ValueError("bundle2.0は同一directoryのNPZのみ対応します。")
    model_path = path / filename
    if model_path.resolve().parent != path.resolve():
        raise ValueError("NPZがbundle directoryの外を参照しています。")
    with zipfile.ZipFile(model_path) as archive:
        if sum(item.file_size for item in archive.infolist()) > 256 * 1024**2:
            raise ValueError("stage bundleが復元メモリ上限を超えています。")
    if hashlib.sha256(model_path.read_bytes()).hexdigest() != manifest.get("predictor_sha256"):
        raise ValueError("stage bundleのNPZ hashが一致しません。")
    with np.load(model_path, allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    if any(array.dtype.kind not in "fiu" or not np.all(np.isfinite(array)) for array in arrays.values()):
        raise ValueError("stage bundleに不正な数値配列があります。")
    backend = manifest.get("backend")
    if backend == "fast_stage":
        model = FastStageModel.from_arrays(arrays)
        inputs = len(manifest["controls"]) + len(manifest["incoming_context"])
        outputs = len(manifest["predicted_outputs"])
        if model.train_x.ndim != 2 or model.train_x.shape[1] != inputs or not 2 <= len(model.train_x) <= MAX_TRAINING_SAMPLES:
            raise ValueError("stage bundleの学習配列shapeが不正です。")
        if model.output_mean.shape != (outputs,) or model.output_scale.shape != (outputs,) or np.any(model.output_scale <= 0):
            raise ValueError("stage bundleの出力shape/scaleが不正です。")
        if model.support_radius <= 0 or model.length_scale <= 0:
            raise ValueError("stage bundleの距離scaleが不正です。")
        n = len(model.train_x)
        expected = {"hidden_w": (3, inputs, 32), "hidden_b": (3, 32),
                    "cholesky": (n, n), "residual_alpha": (n, outputs),
                    "residual_scale": (outputs,), "error_floor": (outputs,), "precision": (inputs, inputs),
                    "feature_center": (inputs,), "feature_scale": (inputs,)}
        if any(getattr(model, name).shape != shape for name, shape in expected.items()):
            raise ValueError("stage bundleの模型shapeが不正です。")
        if np.any(model.feature_scale <= 0):
            raise ValueError("stage bundleの入力特徴scaleが不正です。")
        if arrays["pairs"].dtype.kind not in "iu" or model.pairs.ndim != 2 or model.pairs.shape[1] != 2 or np.any(model.pairs < 0) or np.any(model.pairs >= inputs):
            raise ValueError("stage bundleの交互作用indexが不正です。")
        if model.readouts.shape != (3, 1 + 2 * inputs + len(model.pairs) + 32, outputs) or np.any(np.diag(model.cholesky) <= 0):
            raise ValueError("stage bundleの読出し/カーネル形状が不正です。")
    elif backend == "snapshot_hybrid":
        from .safe_snapshot import restore_hybrid
        try:
            model = restore_hybrid(arrays, manifest["predicted_outputs"],
                                   len(manifest["controls"]) + len(manifest["incoming_context"]))
        except (KeyError, TypeError, IndexError) as error:
            raise ValueError("Hybrid bundleの必須配列が不正です。") from error
    else:
        raise ValueError("未対応のbundle2.0 backendです。")
    return manifest, StagePredictor(manifest, model)


def export_stage_bundle(trial, run_id, problem, model, connection) -> Path:
    """従来CLIで学習したNN＋GPを、数値重みとして損失なく保存。"""
    from .safe_snapshot import serialize_hybrid
    value = {"bundle_schema_version": "2.0", "stage_id": connection.stage_id, "run_id": run_id,
             "controls": list(connection.controls), "incoming_context": list(connection.incoming_context),
             "predicted_outputs": [item.column for item in problem.result_variables],
             "connector_outputs": list(connection.connector_outputs),
             "input_bounds": {item.column: [item.lower, item.upper] for item in problem.parameters},
             "input_units": {item.column: item.unit for item in problem.parameters},
             "output_units": {item.column: item.unit for item in problem.result_variables},
             "local_objective": {"name": problem.objective.column, "direction": problem.objective.direction},
             "local_constraints": [{"name": item.column, "direction": item.direction, "target": item.target,
                                    "margin_scale": max(abs(item.target) * 0.1, 1e-6)} for item in problem.constraints],
             "capabilities": {"knowledge_loss": True}}
    value["parameter_order"] = [item.column for item in problem.parameters]
    arrays = serialize_hybrid(model)
    role_order = value["controls"] + value["incoming_context"]
    if role_order != value["parameter_order"]:
        arrays["input_permutation"] = np.asarray([role_order.index(name) for name in value["parameter_order"]], dtype=int)
    return _save(Path(trial) / "output" / "stage_bundles" / f"stage_bundle_run_{run_id}", value, arrays, "snapshot_hybrid")
