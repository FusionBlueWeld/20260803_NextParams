"""Integration checks for the opt-in initial design workflow."""

import csv
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from v005.src.hybrid.model import HybridPrediction
from v005.src.validation import UserInputError
from v005.src.workflow import run_trial


class AdaptiveWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.workspace = Path(__file__).resolve().parents[1] / "benchmarks" / "results"
        self.workspace.mkdir(parents=True, exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(prefix="v005_adaptive_", dir=self.workspace)
        self.trial = Path(self.temporary.name)
        (self.trial / "data").mkdir()
        (self.trial / "output").mkdir()
        (self.trial / "problem.csv").write_text(
            "column,display_name,unit,role,direction,lower,upper,step,target\n"
            "x,x,1,parameter,,0,1,0.25,\n"
            "y,y,unit,objective,minimize,,,,\n",
            encoding="utf-8",
        )
        (self.trial / "data" / "experiments.csv").write_text(
            "experiment_id,x,y\n", encoding="utf-8"
        )
        (self.trial / "focus_settings.json").write_text(
            json.dumps({
                "initial_design_enabled": True,
                "initial_min_conditions": 4,
                "initial_max_conditions": 4,
                "max_total_experiments": 3,
            }),
            encoding="utf-8",
        )

    def tearDown(self):
        self.temporary.cleanup()

    def _run(self):
        with patch("v005.src.workflow.require_trial", return_value=self.trial), \
             patch("v005.src.hybrid.model.train_hybrid_model", side_effect=AssertionError("initial design must not train")):
            run_trial("trial_adaptive_initial", 10)

    def _recommendations(self):
        path = self.trial / "output" / "recommendations.csv"
        with path.open(encoding="utf-8-sig", newline="") as handle:
            return list(csv.DictReader(handle))

    def test_empty_trial_publishes_capped_design_without_training_or_convergence(self):
        self._run()

        rows = self._recommendations()
        self.assertGreater(len(rows), 0)
        self.assertLessEqual(len(rows), 3)  # total experiment budget includes initial design
        self.assertTrue(all(row["selection_role"] == "INITIAL_DESIGN" for row in rows))
        for row in rows:
            self.assertEqual(row["y_nn_pred"], "")
            self.assertEqual(row["y_mean"], "")

        status = json.loads((self.trial / "output" / "initial_design_status.json").read_text(encoding="utf-8"))
        self.assertEqual(status["phase"], "INITIAL_DESIGN")
        self.assertEqual(status["target_conditions"], 4)
        self.assertEqual(status["actual_experiments"], 0)
        self.assertFalse(status["converged"])
        budget = json.loads((self.trial / "output" / "search_budget.json").read_text(encoding="utf-8"))
        self.assertTrue(budget["learning_skipped"])
        self.assertEqual(budget["prediction_grid_passes"], 0)
        self.assertFalse((self.trial / "output" / "parameter_focus.json").exists())
        stopping_path = self.trial / "output" / "stopping_status.json"
        if stopping_path.exists():
            stopping = json.loads(stopping_path.read_text(encoding="utf-8"))
            self.assertFalse(stopping.get("metrics", {}).get("converged", False))

    def test_same_empty_data_republishes_the_same_pending_design(self):
        self._run()
        first = [(row["x"], row["selection_role"]) for row in self._recommendations()]

        self._run()
        second = [(row["x"], row["selection_role"]) for row in self._recommendations()]

        self.assertEqual(second, first)

    def test_existing_measurements_consume_the_shared_total_budget(self):
        (self.trial / "data" / "experiments.csv").write_text(
            "experiment_id,x,y\n"
            "e1,0,1\n"
            "e2,0.25,1\n"
            "e3,0.5,1\n",
            encoding="utf-8",
        )

        self._run()

        status = json.loads((self.trial / "output" / "stopping_status.json").read_text(encoding="utf-8"))
        self.assertEqual(status["status"], "STOP_REQUIRED")
        self.assertFalse(status["metrics"]["converged"])
        self.assertFalse((self.trial / "output" / "recommendations.csv").exists())

    def test_empty_experiments_still_require_opt_in(self):
        (self.trial / "focus_settings.json").write_text(json.dumps({
            "initial_design_enabled": False, "parameter_focus_enabled": False,
            "temporary_fix_enabled": False,
        }), encoding="utf-8")

        with patch("v005.src.workflow.require_trial", return_value=self.trial):
            with self.assertRaises(UserInputError):
                run_trial("trial_adaptive_initial", 10)

    def _complete_small_grid(self):
        import time
        from v005.src.trial_inputs import load_trial_inputs
        from v005.src.validation import load_and_validate_experiments
        from v005.src.workflow import _run_initial_design

        (self.trial / "focus_settings.json").write_text(json.dumps({
            "initial_design_enabled": True, "initial_min_conditions": 12,
            "initial_max_conditions": 48,
        }), encoding="utf-8")
        (self.trial / "data" / "experiments.csv").write_text(
            "experiment_id,x,y\n" + "".join(
                f"e{i},{x},{1+x*x}\n" for i, x in enumerate([0, .25, .5, .75, 1])
            ), encoding="utf-8")
        inputs = load_trial_inputs(self.trial)
        experiments = load_and_validate_experiments(
            self.trial / "data" / "experiments.csv", inputs.problem)
        self.assertFalse(_run_initial_design(
            self.trial, experiments, inputs, 2, {}, time.perf_counter()))
        status = json.loads((self.trial / "output" / "initial_design_status.json").read_text(encoding="utf-8"))
        self.assertEqual(status["phase"], "SEARCH")
        self.assertTrue(status["finite_space_measured"])
        self.assertFalse(status["converged"])

    def test_fully_measured_small_grid_leaves_initial_stage(self):
        self._complete_small_grid()

    def test_completed_initial_stage_does_not_restart_for_same_settings(self):
        # Complete the small grid first, then ensure subsequent calls bypass design diagnosis.
        self._complete_small_grid()
        import time
        from v005.src.trial_inputs import load_trial_inputs
        from v005.src.validation import load_and_validate_experiments
        from v005.src.workflow import _run_initial_design

        inputs = load_trial_inputs(self.trial)
        experiments = load_and_validate_experiments(
            self.trial / "data" / "experiments.csv", inputs.problem)
        with patch("v005.src.experimental_design.design_diagnostics",
                   side_effect=AssertionError("completed initial stage must stay completed")):
            self.assertFalse(_run_initial_design(
                self.trial, experiments, inputs, 2, {}, time.perf_counter()))

    def test_parameter_focus_runs_through_one_shared_full_prediction(self):
        (self.trial / "focus_settings.json").write_text(
            json.dumps({"initial_design_enabled": False, "parameter_focus_enabled": True,
                        "temporary_fix_enabled": False, "confirmation_slots": 1}),
            encoding="utf-8",
        )
        (self.trial / "data" / "experiments.csv").write_text(
            "experiment_id,x,y\n"
            "e1,0,4\n"
            "e2,0.25,3\n"
            "e3,0.5,2\n"
            "e4,0.75,1\n",
            encoding="utf-8",
        )
        prediction_calls = []

        def predict(normalized):
            prediction_calls.append(len(normalized))
            count = len(normalized)
            values = {"y": {
                "nn_pred": np.full(count, 1.5),
                "nn_std": np.full(count, 0.1),
                "hybrid_mean": np.full(count, 1.5),
                "hybrid_std": np.full(count, 0.1),
                "raw_hybrid_std": np.full(count, 0.1),
                "acquisition_std": np.full(count, 0.1),
                "uncertainty_calibration_scale": np.ones(count),
            }}
            return HybridPrediction(np.ones(count), values)

        model = SimpleNamespace(
            predict=predict,
            nn_model=SimpleNamespace(members=[None]),
            gp_models={"y": SimpleNamespace(y_scale=1.0)},
            uncertainty_calibration_scale=1.0,
            acquisition_uncertainty_scale=1.0,
        )
        training = SimpleNamespace(
            model=model, source_samples=4, retained_samples=4, learning_seconds=0.001,
            nn_epochs=1, nn_normalized_mse=0.0,
        )

        with patch("v005.src.workflow.require_trial", return_value=self.trial), \
             patch("v005.src.hybrid.model.train_hybrid_model", return_value=training) as fit, \
             patch("v005.src.workflow.write_knowledge_diagnostics", return_value=[]):
            run_trial("trial_adaptive_focus", 2)

        self.assertEqual(fit.call_count, 1)
        self.assertEqual(len(prediction_calls), 1)
        focus_state = json.loads((self.trial / "output" / "parameter_focus.json").read_text(encoding="utf-8"))
        self.assertEqual(focus_state["data_rows"], 4)
        self.assertIn(focus_state["phase"], {"SCREENING", "ADAPTIVE_SEARCH", "FOCUSED_SEARCH"})
        self.assertEqual(focus_state["fixed_values"], {})
        budget = json.loads((self.trial / "output" / "search_budget.json").read_text(encoding="utf-8"))
        self.assertEqual(budget["prediction_grid_passes"], 1)
        self.assertEqual(budget["parameter_focus"]["confirmation_slots"], 1)
        self.assertTrue(budget["parameter_focus"]["full_model_input_columns_retained"])
        recommendations = self._recommendations()
        self.assertTrue(any(row["selection_role"] == "SCREENING" for row in recommendations))


if __name__ == "__main__":
    unittest.main()
