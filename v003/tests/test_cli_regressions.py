"""実際のCLI処理で知識・停止履歴・推薦ファイルの整合を検査します。"""

from __future__ import annotations

import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

VERSION_ROOT = Path(__file__).resolve().parents[1]
if str(VERSION_ROOT) not in sys.path:
    sys.path.insert(0, str(VERSION_ROOT))

from src import cli  # noqa: E402
from src.data_loader import read_csv, write_csv_atomic  # noqa: E402
from src.knowledge import KNOWLEDGE_HEADER  # noqa: E402
from src.policies.regions import SEARCH_REGIONS_HEADER  # noqa: E402
from src.stopping.config import STOP_SETTINGS_HEADER  # noqa: E402
from src.validation import PROBLEM_HEADER, UserInputError  # noqa: E402


class CliRegressionTest(unittest.TestCase):
    def setUp(self) -> None:
        workspace = VERSION_ROOT.parent.resolve()
        temporary = tempfile.TemporaryDirectory(prefix=".v003_test_", dir=workspace)
        self.root = Path(temporary.name).resolve()
        self.assertEqual(self.root.parent, workspace)
        self.addCleanup(temporary.cleanup)
        self.trial = self.root / "trial_review"
        (self.trial / "data").mkdir(parents=True)
        (self.trial / "output").mkdir()
        self.problem_rows = [
            dict(column="x", display_name="x", unit="", role="parameter", direction="",
                 lower=0, upper=1, step=0.05, target=""),
            dict(column="y", display_name="y", unit="", role="objective", direction="maximize",
                 lower="", upper="", step="", target=""),
        ]
        self.observations = [
            dict(experiment_id=f"e{i}", x=x, y=x)
            for i, x in enumerate([0, 0.25, 0.5, 0.75, 1])
        ]
        self.write_problem()
        self.write_observations()
        patcher = patch.object(cli, "trial_path", return_value=self.trial)
        patcher.start()
        self.addCleanup(patcher.stop)

    def write_problem(self):
        write_csv_atomic(self.trial / "problem.csv", PROBLEM_HEADER, self.problem_rows)

    def write_observations(self):
        write_csv_atomic(
            self.trial / "data" / "experiments.csv", ["experiment_id", "x", "y"],
            self.observations,
        )

    def settings(self, **values):
        write_csv_atomic(
            self.trial / "stop_settings.csv", STOP_SETTINGS_HEADER,
            [dict(setting=key, value=value, description="") for key, value in values.items()],
        )

    def run_trial(self, count=3):
        with contextlib.redirect_stdout(io.StringIO()):
            cli.run_trial("trial_review", count)
        return self.status()

    def status(self):
        return json.loads((self.trial / "output" / "stopping_status.json").read_text(encoding="utf-8"))

    def recommendations(self):
        return read_csv(self.trial / "output" / "recommendations.csv")[1]

    def test_rerunning_and_reordering_same_data_does_not_establish_convergence(self):
        for _ in range(5):
            status = self.run_trial()
        self.observations.reverse()
        self.write_observations()
        status = self.run_trial()
        self.assertEqual(status["status"], "CONTINUE")
        self.assertFalse(status["metrics"]["minimum_history_reached"])
        self.assertEqual(len(status["runs"]), 6)  # 実行記録自体は残す。

    def test_new_observations_can_establish_convergence(self):
        self.settings(min_history=3, patience=3, coverage_threshold=0,
                      recent_improvement_tolerance=10, top_score_threshold=10,
                      top_score_tolerance=10, condition_tolerance=1)
        for i, x in enumerate([0.1, 0.15]):
            self.run_trial()
            self.observations.append(dict(experiment_id=f"new{i}", x=x, y=x))
            self.write_observations()
        self.assertEqual(self.run_trial()["status"], "STOP_RECOMMENDED")

    def test_setting_change_and_data_correction_restart_convergence_not_budget(self):
        self.run_trial()
        self.observations.append(dict(experiment_id="new", x=0.1, y=0.1))
        self.write_observations()
        self.run_trial()
        self.settings(max_additional_experiments=10)
        status = self.run_trial()
        self.assertEqual(status["initial_data_rows"], 5)
        self.assertEqual(status["convergence_start_index"], 2)
        self.observations[0]["y"] = 0.01
        self.write_observations()
        status = self.run_trial()
        self.assertEqual(status["convergence_start_index"], 3)
        self.assertFalse(status["metrics"]["minimum_history_reached"])

    def test_budget_caps_batch_and_required_stop_archives_previous_recommendations(self):
        self.settings(max_additional_experiments=1)
        self.run_trial(count=3)
        recommendations = self.recommendations()
        self.assertEqual(len(recommendations), 1)
        path = self.trial / "output" / "recommendations.csv"
        previous = path.read_bytes()
        x = float(recommendations[0]["x"])
        self.observations.append(dict(experiment_id="new", x=x, y=x))
        self.write_observations()
        with patch("src.hybrid.model.train_hybrid_model", side_effect=AssertionError("must not train")):
            status = self.run_trial()
        self.assertEqual(status["status"], "STOP_REQUIRED")
        self.assertFalse(path.exists())
        archives = list((self.trial / "output" / "recommendations_history").glob("*.csv"))
        self.assertEqual(len(archives), 1)
        self.assertEqual(archives[0].read_bytes(), previous)

    def test_forbidden_best_does_not_satisfy_target_or_set_improvement_baseline(self):
        self.problem_rows[1]["target"] = 0.8
        self.write_problem()
        write_csv_atomic(self.trial / "search_regions.csv", SEARCH_REGIONS_HEADER, [{
            "region_id": "ban", "kind": "forbidden", "parameter": "x", "lower": 1,
            "lower_inclusive": True, "strength": 5, "enabled": True,
        }])
        from src.hybrid.optimizer import run_optimization

        with patch("src.hybrid.optimizer.run_optimization", wraps=run_optimization) as optimizer:
            status = self.run_trial()
        self.assertEqual(status["status"], "CONTINUE")
        self.assertEqual(status["runs"][-1]["best_feasible_objective"], 0.75)
        self.assertEqual(optimizer.call_args.args[1].unique_condition_count, 5)
        from src.hybrid.gp_component import expected_improvement
        import numpy as np

        row = self.recommendations()[0]
        expected = expected_improvement(
            np.array([float(row["y_mean"])]), np.array([float(row["y_std"])]), 0.75, "maximize",
        )[0]
        self.assertAlmostEqual(float(row["expected_improvement"]), expected, places=7)

    def test_gate_failure_archives_recommendations_and_sets_required_stop(self):
        self.run_trial()
        write_csv_atomic(self.trial / "knowledge_constraints.csv", KNOWLEDGE_HEADER, [{
            "rule_id": "bound", "type": "lower_bound", "target": "y", "value": 2,
            "strength": 5, "enabled": True,
        }])
        with self.assertRaisesRegex(UserInputError, "必須知見"):
            self.run_trial()
        self.assertFalse((self.trial / "output" / "recommendations.csv").exists())
        self.assertEqual(self.status()["status"], "STOP_REQUIRED")
        self.assertTrue((self.trial / "output" / "knowledge_diagnostics.csv").exists())

    def test_input_error_cannot_leave_previous_recommendations_as_current(self):
        self.run_trial()
        write_csv_atomic(self.trial / "knowledge_constraints.csv", KNOWLEDGE_HEADER, [{
            "rule_id": "bound", "type": "lower_bound", "target": "y", "wrt": "x",
            "value": 0, "strength": 5, "enabled": True,
        }])
        with self.assertRaisesRegex(UserInputError, "wrtは空欄"):
            self.run_trial()
        self.assertFalse((self.trial / "output" / "recommendations.csv").exists())

    def test_legacy_history_is_preserved_but_not_used_for_convergence(self):
        payload = {
            "initial_data_rows": 3,
            "runs": [dict(data_rows=5, best_feasible_objective=1,
                          normalized_top_score=0, predicted_optimum=[1])] * 5,
        }
        (self.trial / "output" / "stopping_status.json").write_text(
            json.dumps(payload), encoding="utf-8",
        )
        status = self.run_trial()
        self.assertEqual(status["initial_data_rows"], 3)
        self.assertEqual(status["convergence_start_index"], 5)
        self.assertEqual(status["status"], "CONTINUE")


if __name__ == "__main__":
    unittest.main()
