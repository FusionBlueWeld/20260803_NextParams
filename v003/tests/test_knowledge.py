"""知見損失の値と予測値に対する勾配を検証します。"""

from __future__ import annotations

import sys
import tempfile
import unittest
from types import SimpleNamespace
from pathlib import Path

import numpy as np

VERSION_ROOT = Path(__file__).resolve().parents[1]
if str(VERSION_ROOT) not in sys.path:
    sys.path.insert(0, str(VERSION_ROOT))

from src.knowledge import (  # noqa: E402
    KnowledgePointSet,
    KnowledgeRule,
    KNOWLEDGE_HEADER,
    build_knowledge_points,
    evaluate_rules_on_model,
    evaluate_rules_on_observations,
    load_knowledge_constraints,
    write_knowledge_template,
)
from src.knowledge_loss import calculate_rule_loss  # noqa: E402
from src.settings import ProblemDefinition, VariableDefinition  # noqa: E402
from src.data_loader import write_csv_atomic  # noqa: E402
from src.validation import UserInputError  # noqa: E402


class KnowledgeLossTest(unittest.TestCase):
    def setUp(self) -> None:
        self.empty_points = KnowledgePointSet(np.empty((0, 1)))

    def test_monotonic_increasing_gradient_has_correct_direction(self) -> None:
        rule = KnowledgeRule("r", "monotonic_increasing", "y", "x", strength=3)
        summary, gradient, _ = calculate_rule_loss(
            rule, self.empty_points, np.empty(0), [(3.0, 1.0)]
        )
        self.assertEqual(summary.loss, 4.0)
        np.testing.assert_allclose(gradient, [4.0, -4.0])

    def test_monotonic_decreasing_gradient_has_correct_direction(self) -> None:
        rule = KnowledgeRule("r", "monotonic_decreasing", "y", "x", strength=3)
        _, gradient, _ = calculate_rule_loss(
            rule, self.empty_points, np.empty(0), [(1.0, 3.0)]
        )
        np.testing.assert_allclose(gradient, [-4.0, 4.0])

    def test_low_sensitivity_only_penalizes_excess(self) -> None:
        rule = KnowledgeRule(
            "r", "low_sensitivity", "y", "x", tolerance=0.5, strength=3
        )
        summary, gradient, _ = calculate_rule_loss(
            rule, self.empty_points, np.empty(0), [(1.0, 2.0)]
        )
        self.assertEqual(summary.loss, 0.25)
        np.testing.assert_allclose(gradient, [-1.0, 1.0])

    def test_lower_bound_gradient_pushes_negative_prediction_up(self) -> None:
        rule = KnowledgeRule("r", "lower_bound", "y", value=0, strength=3)
        points = KnowledgePointSet(np.array([[0.0], [1.0]]))
        summary, gradient, _ = calculate_rule_loss(
            rule, points, np.array([-2.0, 1.0])
        )
        self.assertEqual(summary.violation_rate, 0.5)
        np.testing.assert_allclose(gradient, [-2.0, 0.0])

    def test_disabled_template_does_not_depend_on_problem_column_names(self) -> None:
        parameter = VariableDefinition(
            "x", "x", "", "parameter", "", 0, 1, 1, None
        )
        objective = VariableDefinition(
            "custom_y", "custom_y", "", "objective", "maximize", None, None, None, None
        )
        problem = ProblemDefinition(
            [parameter], objective, [], [], [objective], candidate_count=2
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "knowledge_constraints.csv"
            write_knowledge_template(path)
            self.assertEqual(load_knowledge_constraints(path, problem), [])


class KnowledgeGateRegressionTest(unittest.TestCase):
    def setUp(self) -> None:
        parameter = VariableDefinition("x", "x", "", "parameter", "", 0, 1, 0.6, None)
        objective = VariableDefinition(
            "y", "y", "", "objective", "maximize", None, None, None, None,
        )
        self.problem = ProblemDefinition([parameter], objective, [], [], [objective], 3)

    def test_nonuniform_final_interval_is_checked_for_all_pair_rules(self) -> None:
        for kind in ("monotonic_increasing", "monotonic_decreasing", "low_sensitivity"):
            with self.subTest(kind=kind):
                rule = KnowledgeRule("r", kind, "y", "x", tolerance=0.5, strength=5)
                sign = -1 if kind == "monotonic_decreasing" else 1
                model = SimpleNamespace(
                    output_scale=np.ones(1),
                    predict=lambda x: sign * np.where(x[:, :1] > 0.9, -10.0, 0.0),
                )
                points = build_knowledge_points(self.problem, [rule], max_points=3)
                np.testing.assert_allclose(
                    np.asarray(points["r"].pairs).reshape(-1, 2), [[0, 0.6], [0.6, 1]],
                )
                report = evaluate_rules_on_model(model, self.problem, [rule], points=points)
                self.assertEqual(report["gate"]["failed_rule_ids"], ["r"])

    def test_observed_final_interval_is_checked(self) -> None:
        rule = KnowledgeRule("r", "monotonic_increasing", "y", "x", strength=5)
        prepared = SimpleNamespace(
            raw_parameters=np.array([[0.0], [0.6], [1.0]]),
            normalized_parameters=np.array([[0.0], [0.6], [1.0]]),
            result_means={"y": np.array([0.0, 0.6, -10.0])},
        )
        report = evaluate_rules_on_observations(prepared, self.problem, [rule])
        self.assertEqual(report["rules"][0].points, 2)
        self.assertEqual(report["gate"]["failed_rule_ids"], ["r"])

    def test_lower_bound_with_wrt_is_rejected_during_csv_loading(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "knowledge_constraints.csv"
            write_csv_atomic(path, KNOWLEDGE_HEADER, [{
                "rule_id": "r", "type": "lower_bound", "target": "y", "wrt": "x",
                "value": 0, "strength": 5, "enabled": "true",
            }])
            with self.assertRaisesRegex(UserInputError, "wrtは空欄"):
                load_knowledge_constraints(path, self.problem)

    def test_missing_or_empty_model_checks_cannot_pass_mandatory_gate(self) -> None:
        rule = KnowledgeRule("r", "monotonic_increasing", "y", "x", strength=5)
        model = SimpleNamespace(output_scale=np.ones(1))
        for points in ({}, {"r": KnowledgePointSet(np.empty((0, 1)))}):
            with self.subTest(points=points):
                report = evaluate_rules_on_model(model, self.problem, [rule], points=points)
                self.assertFalse(report["gate"]["passed"])
                self.assertEqual(report["gate"]["unchecked_rule_ids"], ["r"])
                self.assertEqual(report["gate"]["checked"], 0)

    def test_missing_observed_pairs_are_reported_as_unchecked(self) -> None:
        rule = KnowledgeRule("r", "monotonic_increasing", "y", "x", strength=5)
        prepared = SimpleNamespace(
            raw_parameters=np.array([[0.0], [1.0]]),
            normalized_parameters=np.array([[0.0], [1.0]]),
            result_means={"y": np.array([0.0, 1.0])},
        )
        report = evaluate_rules_on_observations(prepared, self.problem, [rule])
        self.assertEqual(report["gate"]["unchecked_rule_ids"], ["r"])
        self.assertEqual(report["gate"]["failed_rule_ids"], [])


if __name__ == "__main__":
    unittest.main()
