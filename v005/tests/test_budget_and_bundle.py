"""候補爆発・支持域・安全な保存・実CLIをまとめて検証する。"""
from __future__ import annotations
import json
import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch
import uuid
import numpy as np

from v005.src.stage_bundle import train_stage_bundle, load_stage_bundle, export_stage_bundle
from v005.src.settings import ProblemDefinition, VariableDefinition
from v005.src.connection import StageConnection
from v005.src.parameter_space import make_parameter_grid, candidate_metadata
from v005.src.validation import ExperimentData
from v005.src.preprocessing import prepare_experiments
from v005.src.hybrid.model import train_hybrid_model
from v005.src.bundle_schema import validate_manifest
from v005.src.fast_stage_model import FastStageModel

ROOT = Path(__file__).resolve().parents[2]


def manifest():
    return {"bundle_schema_version": "2.0", "stage_id": "test", "controls": ["x"],
            "incoming_context": ["c"], "predicted_outputs": ["y"],
            "input_bounds": {"x": [0, 1], "c": [0, 1]}, "input_units": {"x": "", "c": ""},
            "output_units": {"y": ""}, "local_objective": {"name": "y", "direction": "maximize"},
            "local_constraints": []}


def problem(parameters):
    result = VariableDefinition("y", "y", "", "objective", "maximize", None, None, None, None)
    return ProblemDefinition(parameters, result, [], [], [result], 10**50)


def variable(name, step=0.5):
    return VariableDefinition(name, name, "", "parameter", "", 0, 1, step, None)


class BundleTests(unittest.TestCase):
    def setUp(self):
        # Windows sandboxではOS側Tempのrenameが制限される場合があるためworkspaceを使用。
        self.temporary = tempfile.TemporaryDirectory(prefix=".v005_test_", dir=ROOT / "v005" / "tests")
        self.addCleanup(self.temporary.cleanup)
        self.path = Path(self.temporary.name)
        rng = np.random.default_rng(6)
        self.x = rng.random(128)
        self.c = 0.5 + 0.012 * rng.normal(size=128)
        self.y = np.sin(self.x * 2) + self.x * self.c

    def train(self, y=None):
        return train_stage_bundle(self.path, manifest(), {"x": self.x, "c": self.c},
                                  {"y": self.y if y is None else y})

    def test_roundtrip_prediction_and_hash(self):
        self.train()
        m, predictor = load_stage_bundle(self.path)
        raw = predictor.predict_arrays({"x": self.x, "c": self.c})
        self.assertLess(float(np.mean(np.abs(raw["outputs"]["y"]["mean"] - self.y))), 0.03)
        _, again = load_stage_bundle(self.path)
        np.testing.assert_array_equal(raw["outputs"]["y"]["mean"], again.predict_arrays({"x": self.x, "c": self.c})["outputs"]["y"]["mean"])
        self.assertFalse(m["capabilities"]["joint_sampling"])
        self.assertFalse(m["capabilities"]["calibrated_model_uncertainty"])
        model_path = self.path / m["predictor"]
        with model_path.open("ab") as stream:
            stream.write(b"tampered")
        with self.assertRaisesRegex(ValueError, "hash"):
            load_stage_bundle(self.path)

    def test_support_declines_inside_bounds_outside_observed_context(self):
        self.train()
        _, predictor = load_stage_bundle(self.path)
        r = predictor.predict_arrays({"x": [0.5, 0.5, 0.5], "c": [0.5, 0.95, 2]})
        self.assertGreater(r["support"][0], 0.5)
        self.assertLess(r["support"][1], 0.05)
        np.testing.assert_array_equal(r["valid"], [True, True, False])
        self.assertEqual(r["support"][2], 0)
        with self.assertRaises(ValueError):
            predictor.predict_arrays({"x": 0.5, "c": np.nan})
        with self.assertRaises(ValueError):
            predictor.predict_arrays({"x": 0.5})

    def test_retraining_updates_predictions_and_contract(self):
        self.train()
        first, predictor = load_stage_bundle(self.path)
        before = predictor.predict_arrays({"x": 0.5, "c": 0.5})["outputs"]["y"]["mean"]
        self.train(self.y + 0.7)
        second, predictor = load_stage_bundle(self.path)
        after = predictor.predict_arrays({"x": 0.5, "c": 0.5})["outputs"]["y"]["mean"]
        self.assertAlmostEqual(float(after - before), 0.7, places=8)
        self.assertNotEqual(first["predictor_sha256"], second["predictor_sha256"])

    def test_training_cap_retains_new_experiments(self):
        rng = np.random.default_rng(4)
        x, c = rng.random(540), rng.random(540)
        train_stage_bundle(self.path, manifest(), {"x": x, "c": c}, {"y": x + c}, max_training_samples=128)
        m, predictor = load_stage_bundle(self.path)
        self.assertEqual(m["training"]["retained_samples"], 128)
        self.assertEqual(m["training"]["source_samples"], 540)
        self.assertTrue(np.any(np.all(predictor.model.train_x == [x[-1], c[-1]], axis=1)))

    def test_input_output_names_can_overlap(self):
        m = manifest()
        m["predicted_outputs"], m["output_units"] = ["c"], {"c": ""}
        m["local_objective"]["name"] = "c"
        train_stage_bundle(self.path, m, {"x": self.x, "c": self.c}, {"c": self.c})
        load_stage_bundle(self.path)

    def test_blank_units_normalize_and_connect_to_r002(self):
        from r002.src.config import validate_config
        self.train()
        loaded = load_stage_bundle(self.path)
        self.assertEqual(loaded[0]["input_units"], {"x": "1", "c": "1"})
        self.assertEqual(loaded[0]["output_units"], {"y": "1"})
        config = {"schema_version": "2.0", "stages": [{"id": "test"}],
                  "connections": [], "external_context": {"test.c": 0.5},
                  "candidate_axes": {"test.x": [0, 0.5, 1]},
                  "final_specifications": [{"name": "y", "direction": "greater_equal", "target": 0}]}
        self.assertEqual(validate_config(config, {"test": loaded})["order"], ["test"])

    def test_derived_outputs_override_inconsistent_predictions_and_roundtrip(self):
        m = manifest()
        names = ["a", "b", "peak", "maximum", "sum", "nested"]
        m["predicted_outputs"], m["output_units"] = names, {name: "1" for name in names}
        m["local_objective"]["name"] = "peak"
        # 単独回帰なら矛盾する小さなaggregateを学習させ、恒等式による補正を確認。
        values = {"a": 2.5 + 0.1 * self.x, "b": -2.0 - 0.1 * self.c,
                  **{name: np.full(128, 0.01) for name in names[2:]}}
        train_stage_bundle(self.path, m, {"x": self.x, "c": self.c}, values)
        original, plain = load_stage_bundle(self.path)
        self.assertLess(float(plain.predict_arrays({"x": 0.5, "c": 0.5})["outputs"]["peak"]["mean"]), 0.1)
        # 既存NPZへmetadataだけを追加でき、NPZの保存形式・hashを変更しない。
        original["derived_outputs"] = {
            "nested": {"operation": "max", "sources": ["maximum", "sum"]},
            "peak": {"operation": "max_abs", "sources": ["a", "b"]},
            "maximum": {"operation": "max", "sources": ["a", "b"]},
            "sum": {"operation": "sum", "sources": ["a", "b"]}}
        (self.path / "stage_manifest.json").write_text(json.dumps(original), encoding="utf-8")
        persisted, predictor = load_stage_bundle(self.path)
        result = predictor.predict_arrays({"x": [0.2, 0.8], "c": 0.5})
        o = result["outputs"]
        np.testing.assert_allclose(o["peak"]["mean"], np.maximum(np.abs(o["a"]["mean"]), np.abs(o["b"]["mean"])))
        np.testing.assert_allclose(o["maximum"]["mean"], np.maximum(o["a"]["mean"], o["b"]["mean"]))
        np.testing.assert_allclose(o["sum"]["mean"], o["a"]["mean"] + o["b"]["mean"])
        np.testing.assert_allclose(o["nested"]["mean"], np.maximum(o["maximum"]["mean"], o["sum"]["mean"]))
        np.testing.assert_allclose(o["peak"]["std"], o["a"]["std"] + o["b"]["std"])
        np.testing.assert_allclose(o["nested"]["std"], o["maximum"]["std"] + o["sum"]["std"])
        self.assertGreater(float(o["peak"]["mean"][0]), 2.0)
        self.assertEqual(result["derived_output_policy"], "point_estimate_identity_uncalibrated_std_sum")
        self.assertEqual(original["predictor_sha256"], persisted["predictor_sha256"])
        _, again = load_stage_bundle(self.path)
        np.testing.assert_array_equal(o["peak"]["mean"], again.predict_arrays({"x": [0.2, 0.8], "c": 0.5})["outputs"]["peak"]["mean"])

    def test_invalid_derived_metadata_rejected_before_loading_npz(self):
        invalid = [[], {"y": {"operation": "eval", "sources": ["y"]}},
                   {"y": {"operation": "max", "sources": []}},
                   {"y": {"operation": "sum", "sources": ["unknown"]}},
                   {"unknown": {"operation": "max", "sources": ["y"]}},
                   {"y": {"operation": "sum", "sources": ["y", "y"]}},
                   {"y": {"operation": "sum", "sources": ["y"], "code": "print(1)"}},
                   {"y": {"operation": "max_abs", "sources": ["y"]}}]
        for definition in invalid:
            with self.subTest(definition=definition):
                m = manifest()
                m["derived_outputs"] = definition
                (self.path / "stage_manifest.json").write_text(json.dumps(m), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "derived_outputs"):
                    load_stage_bundle(self.path)
        m = manifest()
        m["predicted_outputs"], m["output_units"] = ["y", "z"], {"y": "mm", "z": "deg"}
        m["derived_outputs"] = {"y": {"operation": "max", "sources": ["z"]}}
        with self.assertRaisesRegex(ValueError, "単位"):
            validate_manifest(m)
        m["output_units"]["z"] = "mm"
        m["derived_outputs"]["z"] = {"operation": "sum", "sources": ["y"]}
        with self.assertRaisesRegex(ValueError, "循環"):
            validate_manifest(m)

    def test_observed_input_scaling_preserves_small_variation_sensitivity(self):
        rng = np.random.default_rng(43)
        x = np.column_stack((rng.random(128), 0.01 + 0.025 * rng.random(128)))
        query = np.column_stack((rng.random(400), 0.01 + 0.025 * rng.random(400)))
        def quality(values):
            return (0.3 * values[:, 0] + np.sin((values[:, 1] - 0.01) * 80)
                    + 0.5 * values[:, 0] * (values[:, 1] - 0.01) * 40)[:, None]
        truth = quality(query)
        learned = FastStageModel.fit(x, quality(x), seed=2)
        # 同じNN/RBFをglobal範囲の固定尺度で学習する対照とheldoutで比較する。
        with patch("v005.src.fast_stage_model._input_transform",
                   return_value=(np.full(2, 0.5), np.full(2, 0.5))):
            baseline = FastStageModel.fit(x, quality(x), seed=2)
        error = float(np.mean(np.abs(learned.predict(query)[0] - truth)))
        baseline_error = float(np.mean(np.abs(baseline.predict(query)[0] - truth)))
        self.assertLess(error, 0.01)
        self.assertLess(error, baseline_error * 0.25)
        train_stage_bundle(self.path, manifest(), {"x": x[:, 0], "c": x[:, 1]},
                           {"y": quality(x)[:, 0]}, seed=2, max_training_samples=128)
        _, loaded = load_stage_bundle(self.path)
        np.testing.assert_array_equal(loaded.model.train_x, x)
        np.testing.assert_array_equal(loaded.model.feature_scale, learned.feature_scale)
        expected = learned.predict(query)
        actual = loaded.predict_arrays({"x": query[:, 0], "c": query[:, 1]})
        np.testing.assert_array_equal(actual["outputs"]["y"]["mean"], expected[0][:, 0])
        np.testing.assert_array_equal(actual["outputs"]["y"]["std"], expected[1][:, 0])
        # 観測された狭い状態から離れた流入を、bounds内だけで支持しない。
        self.assertLess(float(loaded.predict_arrays({"x": 0.5, "c": 0.8})["support"]), 0.01)

    def test_legacy_fast_npz_restores_equivalent_coordinates(self):
        with patch("v005.src.fast_stage_model._input_transform",
                   return_value=(np.full(2, 0.5), np.full(2, 0.5))):
            self.train()
        m, reference = load_stage_bundle(self.path)
        query = {"x": np.linspace(0, 1, 20), "c": 0.5}
        expected = reference.predict_arrays(query)
        model_path = self.path / m["predictor"]
        with np.load(model_path, allow_pickle=False) as archive:
            arrays = {name: archive[name] for name in archive.files}
        # 旧NPZには変換情報がなく、kernel/supportは元のglobal座標を用いた。
        arrays.pop("feature_center")
        arrays.pop("feature_scale")
        arrays["precision"] *= 4
        arrays["length_scale"] /= 2
        np.savez_compressed(model_path, **arrays)
        m["predictor_sha256"] = hashlib.sha256(model_path.read_bytes()).hexdigest()
        (self.path / "stage_manifest.json").write_text(json.dumps(m), encoding="utf-8")
        _, legacy = load_stage_bundle(self.path)
        actual = legacy.predict_arrays(query)
        np.testing.assert_allclose(actual["outputs"]["y"]["mean"], expected["outputs"]["y"]["mean"], atol=1e-12)
        np.testing.assert_allclose(actual["outputs"]["y"]["std"], expected["outputs"]["y"]["std"], atol=1e-12)
        np.testing.assert_allclose(actual["support"], expected["support"], atol=1e-12)

    def test_invalid_saved_feature_scale_rejected(self):
        self.train()
        m, _ = load_stage_bundle(self.path)
        model_path = self.path / m["predictor"]
        with np.load(model_path, allow_pickle=False) as archive:
            arrays = {name: archive[name] for name in archive.files}
        arrays["feature_scale"][0] = 0
        np.savez_compressed(model_path, **arrays)
        m["predictor_sha256"] = hashlib.sha256(model_path.read_bytes()).hexdigest()
        (self.path / "stage_manifest.json").write_text(json.dumps(m), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "入力特徴scale"):
            load_stage_bundle(self.path)

    def test_cli_hybrid_roundtrip_with_different_role_order(self):
        p = problem([variable("c"), variable("x")])
        observations = ExperimentData([f"e{i}" for i in range(12)],
                                      np.column_stack((self.c[:12], self.x[:12])).tolist(),
                                      {"y": self.y[:12].tolist()})
        prepared = prepare_experiments(observations, p)
        trained = train_hybrid_model(prepared, p, ensemble_size=2, max_nn_epochs=30)
        connection = StageConnection("ordered", ("x",), ("c",), ("y",), {"c": 0.5})
        destination = export_stage_bundle(self.path, 1, p, trained.model, connection)
        _, loaded = load_stage_bundle(destination)
        expected = trained.model.predict(prepared.normalized_parameters).results["y"]
        actual = loaded.predict_arrays({"x": self.x[:12], "c": self.c[:12]})["outputs"]["y"]
        np.testing.assert_allclose(actual["mean"], expected["hybrid_mean"], rtol=1e-12)
        np.testing.assert_allclose(actual["std"], expected["hybrid_std"], rtol=1e-12)


class CandidateBudgetTests(unittest.TestCase):
    def test_huge_incoming_axes_fixed_before_generation(self):
        parameters = [variable(f"u{i}") for i in range(3)] + [variable(f"c{i}", 1e-9) for i in range(12)]
        p = problem(parameters)
        connection = StageConnection("s", tuple(f"u{i}" for i in range(3)), tuple(f"c{i}" for i in range(12)),
                                     ("y",), {f"c{i}": 0.123 for i in range(12)})
        with patch("v005.src.parameter_space.make_axis", side_effect=AssertionError("軸を列挙してはいけない")):
            result = make_parameter_grid(p, connection, max_candidates=64)
        self.assertEqual(result.shape, (27, 15))
        np.testing.assert_allclose(result[:, 3:], 0.123)
        self.assertTrue(candidate_metadata(p, connection, max_candidates=64)["exhaustive"])

    def test_high_dimension_budget_and_preallocation_refusal(self):
        p = problem([variable(f"x{i}", 0.25) for i in range(24)])
        started = time.perf_counter()
        result = make_parameter_grid(p, max_candidates=256)
        self.assertLessEqual(len(result), 256)
        self.assertLess(time.perf_counter() - started, 2)
        self.assertFalse(candidate_metadata(p, max_candidates=256)["exhaustive"])
        with self.assertRaisesRegex(ValueError, "列挙前"):
            make_parameter_grid(p, max_candidates=256, mode="grid")


class RealCliTests(unittest.TestCase):
    def test_new_prepare_run_exports_safe_snapshot(self):
        name = "trial_v005_test_" + uuid.uuid4().hex[:10]
        trial = ROOT / "v005" / "trials" / name
        self.assertEqual(trial.parent.resolve(), (ROOT / "v005" / "trials").resolve())
        self.addCleanup(lambda: shutil.rmtree(trial) if trial.exists() else None)
        def cli(*args):
            r = subprocess.run([sys.executable, str(ROOT / "main.py"), "--version", "v005", *args],
                               cwd=ROOT, text=True, encoding="utf-8", capture_output=True, timeout=45,
                               env={**os.environ, "PYTHONIOENCODING": "utf-8"})
            self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        cli("--new", name)
        focus = json.loads((trial / "focus_settings.json").read_text(encoding="utf-8"))
        for flag in ("initial_design_enabled", "parameter_focus_enabled", "temporary_fix_enabled"):
            self.assertTrue(focus[flag])
            focus[flag] = False  # This test verifies the original snapshot-export path.
        (trial / "focus_settings.json").write_text(json.dumps(focus), encoding="utf-8")
        (trial / "stage_connection.json").write_text(json.dumps({"schema_version": "1.0", "stage_id": "cli_stage",
                                                                "controls": ["input_1"], "incoming_context": [],
                                                                "connector_outputs": ["result_1"], "current_incoming_context": {}}), encoding="utf-8")
        cli("--prepare", name)
        (trial / "data" / "experiments.csv").write_text("experiment_id,input_1,result_1\na,0,100\nb,25,56.25\nc,50,25\nd,75,6.25\ne,100,0\n", encoding="utf-8")
        config = json.loads((trial / "search_settings.json").read_text(encoding="utf-8"))
        config["max_nn_epochs"] = 100
        (trial / "search_settings.json").write_text(json.dumps(config), encoding="utf-8")
        cli("--run", name, "--n", "2")
        bundle = next((trial / "output" / "stage_bundles").iterdir())
        m, predictor = load_stage_bundle(bundle)
        self.assertEqual(m["backend"], "snapshot_hybrid")
        self.assertEqual(m["input_units"], {"input_1": "1"})
        self.assertEqual(m["output_units"], {"result_1": "1"})
        self.assertFalse(list(bundle.glob("*.pkl")))
        self.assertTrue(predictor.predict_arrays({"input_1": 30})["valid"])
        self.assertTrue((trial / "output" / "search_budget.json").exists())
        # 必須知見に実測が反する場合は新しい推薦・bundleを公開しない。
        (trial / "knowledge_constraints.csv").write_text(
            "rule_id,type,target,wrt,value,tolerance,strength,enabled,note\n"
            "K001,upper_bound,result_1,,0,0,5,true,required\n", encoding="utf-8")
        failed = subprocess.run([sys.executable, str(ROOT / "main.py"), "--version", "v005", "--run", name],
                                cwd=ROOT, text=True, encoding="utf-8", capture_output=True, timeout=45,
                                env={**os.environ, "PYTHONIOENCODING": "utf-8"})
        self.assertEqual(failed.returncode, 1)
        self.assertFalse((trial / "output" / "recommendations.csv").exists())
        self.assertEqual(len(list((trial / "output" / "stage_bundles").iterdir())), 1)


if __name__ == "__main__":
    unittest.main()
