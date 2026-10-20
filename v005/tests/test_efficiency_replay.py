"""Decision-boundary tests for useful experiment allocation, without training."""
from __future__ import annotations

import unittest
from dataclasses import replace
from types import SimpleNamespace

import numpy as np

from v005.benchmarks.efficiency_validation import SavedPredictionModel, optimization_options
from v005.src.hybrid.optimizer import run_optimization
from v005.src.preprocessing import normalize_parameters, prepare_experiments
from v005.src.settings import ProblemDefinition, VariableDefinition
from v005.src.validation import ExperimentData


def definition(column, role, *, direction="", target=None):
    return VariableDefinition(column, column, "1", role, direction,
                              0.0 if role == "parameter" else None,
                              1.0 if role == "parameter" else None,
                              0.1 if role == "parameter" else None, target)


class ExperimentAllocationTests(unittest.TestCase):
    def setUp(self):
        x = definition("x", "parameter")
        y = definition("y", "objective", direction="minimize")
        g = definition("g", "constraint", direction="less_equal", target=0.0)
        self.problem = ProblemDefinition([x], y, [g], [], [y, g], 11)
        self.grid = np.arange(2, 11, dtype=float).reshape(-1, 1) / 10

    def optimize(self, mean_y, mean_g, *, std=1e-5, feasible_observed=True, grid=None, observed_g=None):
        grid = self.grid if grid is None else grid
        experiments = ExperimentData(["a", "b"], [[0], [0.1]],
            {"y": [1.0, 1.1], "g": observed_g if observed_g is not None
             else ([-1.0, -1.0] if feasible_observed else [1.0, 1.0])})
        prepared = prepare_experiments(experiments, self.problem)
        count = len(grid)
        outputs = {}
        for column, mean in (("y", mean_y), ("g", mean_g)):
            value = np.broadcast_to(np.asarray(mean, dtype=float), (count,))
            spread = np.broadcast_to(np.asarray(std, dtype=float), (count,))
            outputs[column] = {"hybrid_mean": value, "hybrid_std": spread,
                "acquisition_std": spread, "nn_pred": value, "nn_std": spread,
                "raw_hybrid_std": spread, "uncertainty_calibration_scale": np.ones(count)}
        model = SavedPredictionModel(self.problem, normalize_parameters(grid, self.problem),
                                     outputs, np.ones(count), {"y": 1.0, "g": 1.0})
        return run_optimization(experiments, prepared, self.problem, 5, model,
                                candidate_grid=grid, **optimization_options())

    def test_only_useful_feasible_point_is_returned_without_filling_five(self):
        means = np.full(len(self.grid), 1.2)
        means[0] = 0.5
        constraint = np.ones(len(self.grid))
        constraint[0] = -1.0
        result = self.optimize(means, constraint)
        self.assertEqual(len(result.recommendations), 1)
        self.assertGreater(result.recommendations[0]["feasibility_probability"], 0.99)
        self.assertEqual(result.recommendations[0]["x"], 0.2)

    def test_certainly_inferior_candidates_do_not_trigger_filler_experiments(self):
        result = self.optimize(1.2, -1.0)
        self.assertEqual(result.recommendations, [])
        self.assertTrue(result.had_feasible_observation)

    def test_first_feasible_search_remains_open_even_when_probabilities_are_tiny(self):
        result = self.optimize(1.2, 1.0, feasible_observed=False)
        self.assertFalse(result.had_feasible_observation)
        self.assertGreaterEqual(len(result.recommendations), 1)
        self.assertLessEqual(len(result.recommendations), 5)
        self.assertTrue(all(r.get("selection_role") != "IMPROVEMENT" for r in result.recommendations))

    def test_requested_number_is_upper_bound_when_only_one_candidate_remains(self):
        result = self.optimize(0.5, -1.0, grid=np.array([[0.2]]))
        self.assertEqual(len(result.recommendations), 1)
        self.assertEqual(result.requested_recommendation_count, 5)

    def test_saved_prediction_adapter_matches_reordered_subsets_without_inventing_values(self):
        model = SavedPredictionModel(self.problem, np.array([[0.2], [0.3], [0.4]]),
            {"y": {"hybrid_mean": np.array([2, 3, 4], dtype=float)}},
            np.array([0.2, 0.3, 0.4]), {"y": 1, "g": 1})
        result = model.predict(np.array([[0.4], [0.2]]))
        np.testing.assert_array_equal(result.results["y"]["hybrid_mean"], [4, 2])
        np.testing.assert_array_equal(result.support, [0.4, 0.2])

    def test_ordinal_boundary_keeps_promising_candidate_near_an_acceptable_integer(self):
        constraint = replace(self.problem.constraints[0], unit="level", target=4.0)
        self.problem = replace(self.problem, constraints=[constraint], result_order=[self.problem.objective, constraint])
        result = self.optimize(0.5, 4.4, std=0.1, observed_g=[3.0, 4.0])
        self.assertGreaterEqual(len(result.recommendations), 1)
        self.assertGreater(result.recommendations[0]["feasibility_probability"], 0.8)

    def test_continuous_boundary_keeps_original_threshold(self):
        constraint = replace(self.problem.constraints[0], unit="mm", target=4.0)
        self.problem = replace(self.problem, constraints=[constraint], result_order=[self.problem.objective, constraint])
        result = self.optimize(0.5, 4.4, std=0.1, observed_g=[3.0, 4.0])
        self.assertEqual(result.recommendations, [])


if __name__ == "__main__":
    unittest.main()
