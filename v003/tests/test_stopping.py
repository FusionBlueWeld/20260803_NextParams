"""強制終了と収束による終了提案を検証します。"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

VERSION_ROOT = Path(__file__).resolve().parents[1]
if str(VERSION_ROOT) not in sys.path:
    sys.path.insert(0, str(VERSION_ROOT))

from src.stopping import (  # noqa: E402
    StopConfig,
    StopStatus,
    evaluate_stop,
    load_stop_config,
    normalized_condition_distance,
    support_coverage,
    write_stop_settings_template,
)


class StopCriteriaTest(unittest.TestCase):
    def test_required_when_budget_reached(self) -> None:
        decision = evaluate_stop(
            config=StopConfig(max_additional_experiments=3),
            additional_experiments=3,
        )
        self.assertIs(decision.status, StopStatus.STOP_REQUIRED)
        self.assertIn("最大追加実験回数", decision.reasons[0])

    def test_required_when_all_allowed_candidates_explored(self) -> None:
        decision = evaluate_stop(
            config=StopConfig(),
            allowed_candidates=np.array([[0], [1]], dtype=float),
            explored_candidates=np.array([[1], [0], [1]], dtype=float),
        )
        self.assertIs(decision.status, StopStatus.STOP_REQUIRED)

    def test_recommended_after_all_convergence_metrics_hold(self) -> None:
        config = StopConfig(
            min_history=5,
            patience=3,
            coverage_threshold=0.8,
            support_threshold=0.8,
            recent_improvement_tolerance=0.01,
            top_score_threshold=0.1,
            top_score_tolerance=0.01,
            condition_tolerance=0.1,
        )
        decision = evaluate_stop(
            config=config,
            best_feasible_objective_history=[1.0, 1.01, 1.01, 1.01, 1.01],
            support=[0.9, 0.95, 0.81],
            top_scores=[0.04, 0.041, 0.039, 0.04, 0.041],
            predicted_optimum_conditions=[[0.5, 0.5]] * 5,
        )
        self.assertIs(decision.status, StopStatus.STOP_RECOMMENDED)
        self.assertTrue(decision.metrics["converged"])

    def test_continue_when_one_metric_is_unstable(self) -> None:
        decision = evaluate_stop(
            config=StopConfig(
                min_history=3,
                patience=3,
                coverage_threshold=0.5,
                top_score_threshold=0.1,
            ),
            best_feasible_objective_history=[1, 1, 1],
            support=[1, 1],
            top_scores=[0.01, 0.2, 0.01],
            predicted_optimum_conditions=[[0], [0.5], [0]],
        )
        self.assertIs(decision.status, StopStatus.CONTINUE)

    def test_all_convergence_histories_must_reach_minimum_length(self) -> None:
        decision = evaluate_stop(
            config=StopConfig(min_history=5, patience=3),
            best_feasible_objective_history=[1.0] * 5,
            support=[1.0],
            top_scores=[0.0] * 3,
            predicted_optimum_conditions=[[0.5]] * 3,
        )
        self.assertIs(decision.status, StopStatus.CONTINUE)
        self.assertFalse(decision.metrics["minimum_history_reached"])

    def test_target_is_required_even_without_history(self) -> None:
        decision = evaluate_stop(config=StopConfig(), target_reached=True)
        self.assertIs(decision.status, StopStatus.STOP_REQUIRED)

    def test_coverage_and_distance_helpers(self) -> None:
        self.assertAlmostEqual(support_coverage([0.8, 0.79, 0.9], threshold=0.8), 2 / 3)
        self.assertEqual(
            support_coverage([0, 1], threshold=0.5, allowed_mask=[False, True]),
            1,
        )
        self.assertAlmostEqual(normalized_condition_distance([0, 0], [0.1, 0.1]), 0.1)

    def test_default_stop_settings_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "stop_settings.csv"
            write_stop_settings_template(path)
            config = load_stop_config(path, objective_direction="minimize")
        self.assertEqual(config.objective_direction, "minimize")
        self.assertEqual(config.min_history, 5)
        self.assertEqual(config.coverage_threshold, 0.8)


if __name__ == "__main__":
    unittest.main()
