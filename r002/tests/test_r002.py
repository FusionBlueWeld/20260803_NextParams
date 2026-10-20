"""工程合流・計算予算・現場制約の意味を独立した小モデルで検証します。"""

from __future__ import annotations

import copy
import csv
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from r002.src.api import evaluate_acceptance, propose_additional_experiments, run_search
from r002.src.config import ConfigurationError, validate_config
from r002.src.pipeline import evaluate_pipeline
from r002.src.variation import reevaluate_variation


class SumPredictor:
    """入力和を返す接続照合用の予測器。"""

    def __init__(self, support=1.0):
        self.support = support
        self.calls = 0

    def predict_arrays(self, values):
        self.calls += 1
        arrays = np.broadcast_arrays(*[np.asarray(value, dtype=float) for value in values.values()])
        mean = np.sum(np.stack(arrays), axis=0)
        return {"outputs": {"y": {"mean": mean, "std": np.full(mean.shape, 0.02)}},
                "support": np.full(mean.shape, self.support)}


def manifest(stage, incoming=()):
    """単位と変数役割を明示した最小bundle契約を作ります。"""
    return {"bundle_schema_version": "2.0", "stage_id": stage,
            "controls": ["speed"], "incoming_context": list(incoming), "predicted_outputs": ["y"],
            "input_bounds": {"speed": [0, 1], **{name: [-10, 10] for name in incoming}},
            "input_units": {"speed": "mm", **{name: "mm" for name in incoming}},
            "output_units": {"y": "mm"}, "local_objective": {"name": "y", "direction": "maximize"},
            "local_constraints": [], "capabilities": {"mean_prediction": True}}


def chain():
    """同名操作を持つ2工程の条件付き接続設定。"""
    predictors = {"cut": (manifest("cut"), SumPredictor()), "weld": (manifest("weld", ["incoming"]), SumPredictor())}
    config = {"schema_version": "2.0", "stages": [{"id": stage} for stage in predictors],
              "connections": [{"from": "cut.y", "to": "weld.incoming"}], "external_context": {},
              "candidate_axes": {"cut.speed": [0, 0.5, 1], "weld.speed": [0, 0.5, 1]},
              "final_specifications": [{"name": "y", "direction": "between", "target": [0.9, 1.1]}],
              "search": {"max_evaluations": 64, "population_size": 16, "max_seconds": 5, "seed": 7, "batch_size": 8}}
    return config, predictors


class PipelineTests(unittest.TestCase):
    def test_invalid_std_cannot_improve_conservative_margin(self):
        class InvalidStdPredictor(SumPredictor):
            def __init__(self, std):
                super().__init__()
                self.std = std

            def predict_arrays(self, values):
                prediction = super().predict_arrays(values)
                prediction["outputs"]["y"]["std"][:] = self.std
                return prediction

        for std in (-0.5, np.nan, np.inf):
            with self.subTest(std=std):
                config, models = chain()
                config["search"]["conservative_std_multiplier"] = 1.0
                models["weld"] = (models["weld"][0], InvalidStdPredictor(std))
                result = evaluate_pipeline(config, models, {"cut.speed": [0.5], "weld.speed": [0.5]})
                self.assertFalse(result["trusted_feasible"][0])
                self.assertFalse(result["valid_connection"][0])

    def test_predictor_declared_invalid_cannot_become_feasible(self):
        class InvalidPredictor(SumPredictor):
            def predict_arrays(self, values):
                prediction = super().predict_arrays(values)
                prediction["valid"] = np.array([False])
                return prediction

        config, models = chain()
        models["weld"] = (models["weld"][0], InvalidPredictor())
        result = evaluate_pipeline(config, models, {"cut.speed": [0.5], "weld.speed": [0.5]})
        self.assertGreater(result["process_margin"][0], 0)
        self.assertFalse(result["valid_connection"][0])
        self.assertFalse(result["trusted_feasible"][0])

    def test_optional_std_margin_rejects_mean_pass_without_probability_claim(self):
        config, models = chain()
        config["search"]["conservative_std_multiplier"] = 8.0
        result = evaluate_pipeline(config, models, {"cut.speed": [0.5], "weld.speed": [0.5]})
        self.assertGreater(result["process_margin"][0], 0)
        self.assertLess(result["conservative_process_margin"][0], 0)
        self.assertFalse(result["trusted_feasible"][0])
        self.assertEqual(result["uncertainty_policy"], "uncalibrated_heuristic")
        self.assertEqual(result["probability_status"], "NOT_EVALUATED")

    def test_measured_outputs_do_not_receive_model_std_margin(self):
        config, models = chain()
        config["search"]["conservative_std_multiplier"] = 8.0
        config["observed_outputs"] = {"cut.y": 0.5, "weld.y": 1.0}
        config["fixed_controls"] = {"cut.speed": 0.5, "weld.speed": 0.5}
        result = evaluate_pipeline(config, models, config["fixed_controls"])
        self.assertTrue(result["trusted_feasible"][0])
        self.assertEqual(result["process_margin"][0], result["conservative_process_margin"][0])

    def test_same_named_controls_remain_independent_and_upstream_propagates(self):
        config, models = chain()
        result = evaluate_pipeline(config, models, {"cut.speed": np.array([0, 0.5]), "weld.speed": np.array([0.5, 0.5])})
        np.testing.assert_allclose(result["stage_outputs"]["weld"]["y"], [0.5, 1])
        np.testing.assert_allclose(result["stage_inputs"]["weld"]["incoming"], [0, 0.5])

    def test_branch_merge_topological_order(self):
        config, models = chain()
        models = {"merge": (manifest("merge", ["cu", "al"]), SumPredictor()),
                  "cu": (manifest("cu"), SumPredictor()), "al": (manifest("al"), SumPredictor())}
        config["stages"] = [{"id": name} for name in models]
        config["connections"] = [{"from": "cu.y", "to": "merge.cu"}, {"from": "al.y", "to": "merge.al"}]
        config["candidate_axes"] = {f"{name}.speed": [0, 0.5, 1] for name in models}
        config["final_specifications"][0]["stage"] = "merge"
        result = evaluate_pipeline(config, models, {"cu.speed": [0.2], "al.speed": [0.3], "merge.speed": [0.5]})
        np.testing.assert_allclose(result["stage_outputs"]["merge"]["y"], [1])

    def test_unit_mismatch_and_missing_unit_rejected(self):
        config, models = chain()
        models["weld"][0]["input_units"]["incoming"] = "s"
        with self.assertRaises(ConfigurationError):
            validate_config(config, models)
        models["weld"][0]["input_units"].pop("incoming")
        with self.assertRaises(ConfigurationError):
            validate_config(config, models)

    def test_cycle_and_wrong_stage_id_rejected(self):
        config, models = chain()
        models["cut"][0]["stage_id"] = "wrong"
        with self.assertRaises(ConfigurationError):
            validate_config(config, models)
        models["cut"] = (manifest("cut", ["incoming"]), SumPredictor())
        config["connections"].append({"from": "weld.y", "to": "cut.incoming"})
        with self.assertRaises(ConfigurationError):
            validate_config(config, models)

    def test_observation_freezes_completed_controls_and_bypasses_prediction(self):
        config, models = chain()
        config["observed_outputs"] = {"cut.y": 0.3}
        with self.assertRaises(ConfigurationError):
            run_search(config, models)
        config["fixed_controls"] = {"cut.speed": 0.5}
        result = run_search(config, models)
        self.assertTrue(all(item["controls"]["cut.speed"] == 0.5 for item in result["candidates"]))
        self.assertEqual(models["cut"][1].calls, 0)
        evaluation = evaluate_pipeline(config, models, {"cut.speed": [0.5], "weld.speed": [0.5]})
        self.assertEqual(evaluation["observation_status"]["cut"], "FULLY_MEASURED_MEANS")
        self.assertEqual(evaluation["observation_noise_status"], "NOT_PROPAGATED_AS_PHYSICAL_VARIATION")


class SearchTests(unittest.TestCase):
    def test_budget_and_small_grid_solution(self):
        config, models = chain()
        result = run_search(config, models)
        self.assertLessEqual(result["evaluation_count"], 9)
        self.assertTrue(result["best"]["predicted_feasible"])
        self.assertAlmostEqual(result["best"]["controls"]["cut.speed"] + result["best"]["controls"]["weld.speed"], 1)

    def test_local_constraints_are_always_required(self):
        config, models = chain()
        models["cut"][0]["local_constraints"] = [{"name": "y", "direction": "less_equal", "target": 0.6}]
        result = run_search(config, models)
        self.assertTrue(result["best"]["predicted_feasible"])
        self.assertLessEqual(result["best"]["controls"]["cut.speed"], 0.6)

    def test_unsupported_result_is_not_presented_as_feasible(self):
        config, models = chain()
        models["weld"][1].support = 0.1
        result = run_search(config, models)
        self.assertFalse(result["best"]["predicted_feasible"])
        self.assertEqual(result["status"], "NO_TRUSTED_CANDIDATE_FOUND_WITHIN_BUDGET")

    def test_explosive_space_obeys_prediction_budget(self):
        config, models = chain()
        controls = [f"x{index}" for index in range(24)]
        model = manifest("line")
        model["controls"] = controls
        model["input_bounds"] = {name: [0, 1] for name in controls}
        model["input_units"] = {name: "mm" for name in controls}
        config["stages"] = [{"id": "line"}]
        config["connections"] = []
        config["candidate_axes"] = {f"line.{name}": [0, 0.25, 0.5, 0.75, 1] for name in controls}
        config["search"]["max_evaluations"] = 37
        models = {"line": (model, SumPredictor())}
        result = run_search(config, models)
        self.assertEqual(result["full_grid_candidate_count"], 5 ** 24)
        self.assertEqual(result["evaluation_count"], 37)

    def test_continuous_search_can_find_non_grid_condition(self):
        config, models = chain()
        config["candidate_axes"] = {"cut.speed": [0, 1], "weld.speed": [0, 1]}
        config["continuous_controls"] = ["weld.speed"]
        config["fixed_controls"] = {"cut.speed": 0}
        config["final_specifications"] = [{"name": "y", "direction": "between", "target": [0.31, 0.35]}]
        config["search"]["max_evaluations"] = 256
        result = run_search(config, models)
        self.assertTrue(result["best"]["predicted_feasible"])
        self.assertTrue(0.31 <= result["best"]["controls"]["weld.speed"] <= 0.35)

    def test_local_acceptance_reoptimizes_downstream_only(self):
        config, models = chain()
        config["candidate_axes"]["weld.speed"] = [0, 0.25, 0.5, 0.75, 1]
        result = evaluate_acceptance(config, [{"outputs": {"cut.y": 0.25}, "fixed_controls": {"cut.speed": 0.25}, "reachable": True}], models, budget=16)
        self.assertTrue(result["states"][0]["best"]["predicted_feasible"])
        self.assertEqual(result["states"][0]["best"]["controls"]["weld.speed"], 0.75)
        self.assertLessEqual(result["evaluation_count"], 16)

    def test_acceptance_full_reference_does_not_freeze_future_controls(self):
        config, models = chain()
        config["reference_controls"] = {"cut.speed": 0.5, "weld.speed": 0.0}
        result = evaluate_acceptance(config, {"outputs": {"cut.y": 0.5}}, models, budget=16)
        self.assertTrue(result["states"][0]["best"]["predicted_feasible"])
        self.assertEqual(result["states"][0]["best"]["controls"]["weld.speed"], 0.5)


class ExperimentAndVariationTests(unittest.TestCase):
    def test_empty_rerun_does_not_leave_prior_recommendations_csv(self):
        from r002.src.trials import run_trial
        with tempfile.TemporaryDirectory() as directory:
            trial = Path(directory)
            output = trial / "output"
            output.mkdir()
            csv_path = output / "system_recommendations.csv"
            csv_path.write_text("rank,old_value\n1,0.9\n", encoding="utf-8-sig")
            empty = {"candidates": [], "status": "NO_TRUSTED_CANDIDATE_FOUND_WITHIN_BUDGET",
                     "evaluation_count": 0, "elapsed_seconds": 0.0}
            with patch("r002.src.trials.trial_path", return_value=trial), patch("r002.src.trials.run_search", return_value=empty), patch("builtins.print"):
                run_trial("trial_example", 3)
            with csv_path.open(encoding="utf-8-sig", newline="") as stream:
                self.assertEqual(list(csv.DictReader(stream)), [])
            self.assertNotIn("old_value", csv_path.read_text(encoding="utf-8-sig"))

    def test_processing_reserves_workpiece_even_without_destructive_inspection(self):
        config, models = chain()
        config["inventory"] = [{"workpiece_id": "C001", "stage_id": "cut", "outputs": {"y": 0.4}}]
        config["additional_experiments"] = {"recommendation_count": 8, "conditions_per_workpiece": 8,
                                            "costs": {"weld": {"destructive": False}}}
        suggestions = propose_additional_experiments(config, models)
        self.assertEqual(len(suggestions), 1)
        self.assertFalse(suggestions[0]["destructive"])
        self.assertTrue(suggestions[0]["consumes_workpiece"])

    def test_destructive_workpiece_is_not_reused_in_shortlist(self):
        config, models = chain()
        config["inventory"] = [{"workpiece_id": "C001", "stage_id": "cut", "outputs": {"y": 0.4}, "available": True}]
        config["additional_experiments"] = {"recommendation_count": 8, "conditions_per_workpiece": 8}
        suggestions = propose_additional_experiments(config, models)
        self.assertEqual(len(suggestions), 1)
        self.assertEqual(suggestions[0]["incoming_context"], {"incoming": 0.4})
        self.assertEqual(suggestions[0]["status"], "PROPOSED_NOT_EXECUTED")

    def test_consumed_inventory_cannot_be_suggested(self):
        config, models = chain()
        config["inventory"] = [{"workpiece_id": "C001", "stage_id": "cut", "outputs": {"y": 0.4}, "available": True, "consumed": True}]
        self.assertEqual(propose_additional_experiments(config, models), [])

    def test_merge_requires_explicit_pair_and_both_real_workpieces(self):
        config, models = chain()
        models = {"cu": (manifest("cu"), SumPredictor()), "al": (manifest("al"), SumPredictor()), "merge": (manifest("merge", ["cu", "al"]), SumPredictor())}
        config["stages"] = [{"id": name} for name in models]
        config["candidate_axes"] = {f"{name}.speed": [0, 0.5, 1] for name in models}
        config["connections"] = [{"from": "cu.y", "to": "merge.cu"}, {"from": "al.y", "to": "merge.al"}]
        config["inventory"] = [{"workpiece_id": "C", "stage_id": "cu", "outputs": {"y": 0.2}}, {"workpiece_id": "A", "stage_id": "al", "outputs": {"y": 0.4}}]
        self.assertEqual(propose_additional_experiments(config, models), [])
        for item in config["inventory"]:
            item["assembly_pair_id"] = "pair1"
        suggestions = propose_additional_experiments(config, models)
        self.assertEqual(len(suggestions), 1)
        self.assertEqual(set(suggestions[0]["workpiece_ids"]), {"C", "A"})

    def test_input_variation_is_uncalibrated_and_does_not_clip_bounds(self):
        config, models = chain()
        config["process_variation"] = {"samples": 128, "max_evaluations": 80, "standard_deviations": {"cut.speed": 0.5}, "seed": 9}
        result = reevaluate_variation(config, models, [{"controls": {"cut.speed": 0.0, "weld.speed": 1.0}}])
        self.assertEqual(result["evaluation_count"], 80)
        self.assertLess(result["candidates"][0]["valid_connection_fraction"], 1)
        self.assertEqual(result["status"], "UNCALIBRATED_INPUT_VARIATION_ESTIMATE")
        self.assertIn("measurement_noise", result["excluded_uncertainty"])


if __name__ == "__main__":
    unittest.main()
