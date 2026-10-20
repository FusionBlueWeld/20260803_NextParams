"""実bundleを使い、学習・配線・在庫更新と部分観測の境界を検査します。"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from r002.src.api import run_search
from r002.src.pipeline import evaluate_pipeline
from validation.multistage.electrical_interconnect.pipeline import DEFAULT_CONTROLS, evaluate_line, line_config
from validation.multistage.src.checks.interconnect_data import generate_workpieces, train_bundles
from validation.multistage.src.checks.interconnect_lifecycle import candidate_confirmation_loop, experiment_loop, recovery_checks
from validation.multistage.src.checks.interconnect_metrics import oracle_predictors
from validation.multistage.src.checks.interconnect_history import historical_warm_start


class InterconnectIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.directory = Path(cls.temporary.name)
        cls.dataset = generate_workpieces(1024, seed=0)
        cls.config, cls.predictors = train_bundles(cls.dataset, cls.directory / "bundles", 64)
        cls.config["search"] = {"max_evaluations": 64, "population_size": 16,
                                "max_seconds": 30, "seed": 0, "batch_size": 16,
                                "support_threshold": 0.5}

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def test_oracle_wiring_matches_independent_line_evaluator(self):
        expected = evaluate_line(DEFAULT_CONTROLS)
        actual = evaluate_pipeline(line_config(), oracle_predictors(), DEFAULT_CONTROLS)
        for stage, outputs in expected["stage_outputs"].items():
            for name, value in outputs.items():
                np.testing.assert_allclose(actual["stage_outputs"][stage][name][0], value)
        self.assertTrue(actual["trusted_feasible"][0])

    def test_training_only_uses_admitted_lineages(self):
        edges = {edge["to"]: edge["from"] for edge in self.config["connections"]}
        for stage, block in self.dataset["blocks"].items():
            for name, values in block["inputs"].items():
                key = f"{stage}.{name}"
                if key in edges:
                    parent, output = edges[key].split(".")
                    rows = block["row_indices"]
                    self.assertTrue(np.all(self.dataset["passing"][parent][rows]))
                    np.testing.assert_array_equal(values, self.dataset["stage_outputs"][parent][output][rows])

    def test_manufactured_confirmation_uses_the_same_material_context(self):
        context = dict(self.config["external_context"])
        context["cut_cu.stock_yield_mpa"] = 110.0
        work = generate_workpieces(1, selected_controls=DEFAULT_CONTROLS, external_context=context)
        self.assertEqual(work["blocks"]["cut_cu"]["inputs"]["stock_yield_mpa"][0], 110.0)
        expected = evaluate_line(DEFAULT_CONTROLS, material_context=context)
        for stage, block in work["blocks"].items():
            if block["admitted_count"]:
                for name, values in block["outputs"].items():
                    np.testing.assert_allclose(values[0], expected["stage_outputs"][stage][name])

    def test_real_bundle_search_stays_bounded_for_44_controls(self):
        result = run_search(self.config, self.predictors)
        self.assertEqual(result["free_control_count"], 44)
        self.assertEqual(result["full_grid_candidate_count"], 4 ** 44)
        self.assertLessEqual(result["evaluation_count"], 64)
        self.assertIsNotNone(result["best"])
        self.assertEqual(result["probability_status"], "NOT_EVALUATED")

    def test_failed_upstream_confirmation_does_not_label_unmade_downstream_work(self):
        controls = dict(DEFAULT_CONTROLS)
        controls["form_cu.first_overbend_deg"] = 5.0
        initial = {"candidates": [{"controls": controls, "predicted_feasible": True}], "evaluation_count": 1}
        result = candidate_confirmation_loop(self.config, self.predictors, self.dataset, initial,
                                             self.directory / "confirmation", sample_cap=64, rounds=1)
        self.assertEqual(result["status"], "NO_CONFIRMED_FEASIBLE_WITHIN_BUDGET")
        self.assertFalse(result["measurements"][0]["oracle_feasible"])
        self.assertNotIn("assembly", result["measurements"][0]["measured_stages"])
        self.assertNotIn("welding", result["measurements"][0]["measured_stages"])

    def test_partial_observation_does_not_claim_verified_hidden_state(self):
        result = recovery_checks(self.config, self.predictors, DEFAULT_CONTROLS, 32)
        self.assertEqual(result["partial_observation_recovery"]["field_quality_status"], "UNKNOWN_WITH_UNOBSERVED_STATE")
        self.assertTrue(result["partial_observation_recovery"]["unobserved_outputs"])
        self.assertEqual(result["complete_observation_recovery"]["completed_stages"],
                         ["assembly", "cut_al", "cut_cu", "form_al", "form_cu"])

    def test_inventory_measurements_are_consumed_and_retained_after_retraining(self):
        result = experiment_loop(self.config, self.predictors, self.dataset,
                                 self.directory / "update", sample_cap=64, seed=0)
        self.assertEqual(result["status"], "MEASURED_APPENDED_RETRAINED")
        self.assertTrue(result["consumed_workpieces_excluded"])
        self.assertTrue(result["new_measurements_retained_in_model"])
        self.assertNotEqual(result["original_data_sha256"], result["updated_data_sha256"])

    def test_historical_recipe_anchors_remain_in_each_stage_model(self):
        from v005.src.stage_bundle import load_stage_bundle
        destination = self.directory / "history"
        result = historical_warm_start(self.config, self.dataset, destination, sample_cap=64)
        self.assertGreater(result["anchor_count"], 0)
        self.assertLessEqual(result["learned_search"]["evaluation_count"], 64)
        for anchor in result["anchors"]:
            actual = evaluate_pipeline(self.config, oracle_predictors(), anchor["controls"])
            self.assertTrue(actual["trusted_feasible"][0])
            for stage, inputs in actual["stage_inputs"].items():
                manifest, predictor = load_stage_bundle(destination / "bundles" / stage)
                names = manifest["controls"] + manifest["incoming_context"]
                point = np.array([inputs[name][0] for name in names])
                lower = np.array([manifest["input_bounds"][name][0] for name in names])
                upper = np.array([manifest["input_bounds"][name][1] for name in names])
                self.assertTrue(np.any(np.all(np.isclose(predictor.model.train_x, (point - lower) / (upper - lower)), axis=1)))


if __name__ == "__main__":
    unittest.main()
