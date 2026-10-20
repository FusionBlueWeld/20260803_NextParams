"""有望領域の停止、同値解、履歴安全性を実験状態の単位で検証します。"""

from __future__ import annotations

import csv
from dataclasses import dataclass, replace
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from v005.src.stopping import (
    StopConfig, StopStatus, build_stop_context, convergence_history,
    evaluate_stop, load_stop_config, load_stop_history, write_stop_settings_template,
)
from v005.src.stopping.reporting import STOP_ALGORITHM_REVISION


ROOT = Path(__file__).resolve().parent


class EfficiencyStoppingTests(unittest.TestCase):
    def setUp(self):
        self.config = StopConfig(min_history=5, patience=3)
        self.base = dict(
            config=self.config,
            best_feasible_objective_history=[1.0] * 5,
            support=[0.9] * 5,
            top_scores=[0.001] * 5,
            predicted_optimum_conditions=[[0.2, 0.3]] * 5,
        )

    def decision(self, **changes):
        arguments = {**self.base, **changes}
        return evaluate_stop(**arguments)

    def promising(self, **changes):
        return self.decision(**{
            "support": [0.05] * 200,
            "promising_support": [0.91] * 8,
            "promising_region_complete": True,
            "predicted_optimum_objective_history": [1.0] * 5,
            **changes,
        })

    def test_legacy_call_keeps_global_coverage_and_condition_check(self):
        decision = self.decision()
        self.assertEqual(decision.status, StopStatus.STOP_RECOMMENDED)
        self.assertEqual(decision.metrics["coverage_scope"], "global")
        self.assertEqual(self.decision(support=[0.05] * 5).status, StopStatus.CONTINUE)
        self.assertEqual(self.decision(
            predicted_optimum_conditions=[[0.0], [1.0], [0.0], [1.0], [0.0]],
        ).status, StopStatus.CONTINUE)

    def test_supported_promising_region_allows_equivalent_optimum_movement(self):
        decision = self.promising(
            predicted_optimum_conditions=[[0.0], [1.0], [0.0], [1.0], [0.0]],
        )
        self.assertEqual(decision.status, StopStatus.STOP_RECOMMENDED)
        self.assertFalse(decision.metrics["predicted_optimum_condition_stable"])
        self.assertTrue(decision.metrics["predicted_optimum_objective_stable"])
        self.assertEqual(decision.metrics["global_support_coverage"], 0.0)
        self.assertEqual(decision.metrics["promising_support_coverage"], 1.0)
        self.assertEqual(decision.metrics["promising_candidate_count"], 8)

    def test_explicit_global_scope_keeps_full_space_requirement(self):
        decision = self.promising(config=replace(self.config, coverage_scope="global"))
        self.assertEqual(decision.status, StopStatus.CONTINUE)
        self.assertEqual(decision.metrics["coverage_scope"], "global")

    def test_incomplete_or_unproven_promising_region_cannot_stop(self):
        for complete in [False, None]:
            with self.subTest(complete=complete):
                decision = self.promising(promising_region_complete=complete)
                self.assertEqual(decision.status, StopStatus.CONTINUE)
                self.assertIsNone(decision.metrics["support_coverage"])

    def test_global_scope_cannot_claim_coverage_from_sampled_grid(self):
        decision = self.promising(
            config=replace(self.config, coverage_scope="global"),
            support=[0.95] * 20,
            promising_region_complete=False,
        )
        self.assertEqual(decision.status, StopStatus.CONTINUE)
        self.assertEqual(decision.metrics["global_support_coverage"], 1.0)
        self.assertFalse(decision.metrics["coverage_sufficient"])
        self.assertIsNone(decision.metrics["support_coverage"])

    def test_empty_or_unsupported_promising_region_cannot_stop(self):
        for support in [[], [0.3] * 8, [float("inf")] * 8]:
            with self.subTest(support=support):
                self.assertEqual(
                    self.promising(promising_support=support).status,
                    StopStatus.CONTINUE,
                )

    def test_changing_prediction_cannot_hide_behind_stable_coordinates(self):
        decision = self.promising(predicted_optimum_objective_history=[1, 1, 1, 2, 3])
        self.assertEqual(decision.status, StopStatus.CONTINUE)
        self.assertTrue(decision.metrics["predicted_optimum_condition_stable"])
        self.assertFalse(decision.metrics["predicted_optimum_objective_stable"])

    def test_new_value_history_needs_distinct_minimum_states(self):
        decision = self.promising(predicted_optimum_objective_history=[1.0] * 3)
        self.assertEqual(decision.status, StopStatus.CONTINUE)
        self.assertFalse(decision.metrics["minimum_history_reached"])

    def test_value_stability_needs_no_coordinate_history(self):
        decision = self.promising(predicted_optimum_conditions=None)
        self.assertEqual(decision.status, StopStatus.STOP_RECOMMENDED)

    def test_stagnation_and_prediction_alone_are_insufficient(self):
        self.assertEqual(self.promising(
            top_scores=[0.2] * 5,
        ).status, StopStatus.CONTINUE)
        self.assertEqual(self.promising(
            best_feasible_objective_history=[1, 1, 1, 2, 3],
        ).status, StopStatus.CONTINUE)

    def test_budget_is_required_even_without_coverage_evidence(self):
        decision = self.promising(
            config=replace(self.config, max_additional_experiments=10),
            additional_experiments=10,
            promising_region_complete=False,
        )
        self.assertEqual(decision.status, StopStatus.STOP_REQUIRED)
        self.assertIn("最大追加実験回数に到達", decision.reasons)

    def test_tiny_distinct_levels_are_not_all_explored(self):
        decision = evaluate_stop(
            config=self.config,
            allowed_candidates=np.asarray([[0.0], [1e-10], [2e-10]]),
            explored_candidates=np.asarray([[0.0]]),
        )
        self.assertEqual(decision.status, StopStatus.CONTINUE)
        self.assertFalse(decision.metrics["all_candidates_explored"])

    def test_explicit_identity_evidence_controls_exhaustion(self):
        exhausted = evaluate_stop(config=self.config, all_allowed_explored=True)
        self.assertEqual(exhausted.status, StopStatus.STOP_REQUIRED)
        remaining = evaluate_stop(
            config=self.config, all_allowed_explored=False,
            allowed_candidates=np.asarray([[0.0]]), explored_candidates=np.asarray([[0.0]]),
        )
        self.assertEqual(remaining.status, StopStatus.CONTINUE)
        self.assertFalse(remaining.metrics["all_candidates_explored"])

    def test_invalid_configuration_rejected(self):
        for changes in [
            {"coverage_scope": "local"}, {"predicted_objective_tolerance": -1},
            {"predicted_objective_tolerance": float("nan")},
            {"top_score_threshold": float("inf")},
        ]:
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                replace(self.config, **changes)


@dataclass
class _Settings:
    value: str = "same"


class StopHistorySafetyTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix=".stop_test_", dir=ROOT)
        self.addCleanup(self.temporary.cleanup)
        self.path = Path(self.temporary.name)

    def context(self):
        experiments = SimpleNamespace(
            experiment_ids=["one"], parameter_rows=[{"x": 0.5}], result_rows={"y": [1.0]},
        )
        return build_stop_context(
            experiments, problem=_Settings(), knowledge_rules=[],
            region_policy=_Settings(), stop_config=StopConfig(),
        )

    def test_algorithm_revision_invalidates_convergence_not_budget_origin(self):
        old = self.context()
        runs = [{"data_fingerprint": "old"}]
        path = self.path / "stopping_status.json"
        path.write_text(json.dumps({
            "initial_data_rows": 24, "runs": runs, "input_context": old,
        }), encoding="utf-8")
        unchanged = load_stop_history(path, 29, input_context=self.context())
        self.assertEqual(unchanged["convergence_start_index"], 0)
        with patch("v005.src.stopping.reporting.STOP_ALGORITHM_REVISION", "next-revision"):
            changed = self.context()
        self.assertNotEqual(old["settings_fingerprint"], changed["settings_fingerprint"])
        result = load_stop_history(path, 29, input_context=changed)
        self.assertEqual(result["convergence_start_index"], 1)
        self.assertEqual(result["initial_data_rows"], 24)
        self.assertEqual(old["algorithm_revision"], STOP_ALGORITHM_REVISION)

    def test_repeated_same_data_does_not_increase_history(self):
        states = [{
            "data_fingerprint": "same", "best_feasible_objective": 1.0,
            "predicted_optimum": [0.5], "run_id": index,
        } for index in range(10)]
        history = convergence_history(states, 0)
        self.assertEqual(len(history), 1)
        self.assertEqual(history[0]["run_id"], 9)

    def test_configuration_and_template_include_new_fields_and_accept_old_csv(self):
        path = self.path / "stop_settings.csv"
        write_stop_settings_template(path)
        loaded = load_stop_config(path, objective_direction="minimize")
        self.assertEqual(loaded.coverage_scope, "promising")
        self.assertEqual(loaded.objective_direction, "minimize")
        with path.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["setting", "value", "description"])
            writer.writerow(["patience", "4", "old settings file"])
        legacy = load_stop_config(path, objective_direction="maximize")
        self.assertEqual(legacy.patience, 4)
        self.assertEqual(legacy.coverage_scope, "promising")


if __name__ == "__main__":
    unittest.main()
