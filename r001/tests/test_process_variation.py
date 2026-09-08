"""r001 process-variation optimizer regression tests."""

from __future__ import annotations

import unittest

import numpy as np

from r001.src.cli import (
    _all_true_intervals,
    _diverse_candidate_indices,
    _process_variation_rows,
    _run_connected_window,
    _wilson_interval,
)


class FakePredictor:
    bounds = {"control": (0.0, 1.0), "incoming": (0.0, 1.0)}

    def predict_arrays(self, inputs):
        value = np.asarray(inputs["control"]) + np.asarray(inputs["incoming"])
        return {
            "outputs": {"quality": {"mean": value, "std": np.zeros_like(value)}},
            "support": np.ones_like(value),
        }


class ProcessVariationTests(unittest.TestCase):
    def test_wilson_interval_contains_observed_fraction(self):
        successes = np.asarray([0, 50, 100])
        low, high = _wilson_interval(successes, 100)
        observed = successes / 100
        self.assertTrue(np.all(low <= observed))
        self.assertTrue(np.all(observed <= high))
        self.assertGreater(low[2], 0.95)
        self.assertLess(high[0], 0.05)

    def test_zero_variation_reorders_by_predicted_good_probability(self):
        manifest = {
            "controls": ["control"], "incoming_context": ["incoming"],
            "predicted_outputs": ["quality"],
        }
        predictors = [({"id": "stage"}, manifest, FakePredictor())]
        config = {
            "external_context": {"stage.incoming": 0.1},
            "final_specifications": [{"name": "quality", "direction": "greater_equal", "target": 0.5}],
            "process_variation": {
                "distribution": "independent_clipped_normal",
                "samples": 100, "seed": 7, "candidate_limit": 2,
                "standard_deviations": {"control": 0.0, "stage.incoming": 0.0},
            },
        }
        values = {"control": np.asarray([0.2, 0.8])}
        rows = _process_variation_rows(
            predictors, config, {}, ["control"], values,
            np.asarray([0, 1]), np.asarray([-0.2, 0.4]),
        )
        self.assertEqual(rows[0]["control"], 0.8)
        self.assertEqual(rows[0]["predicted_good_probability"], 1.0)
        self.assertEqual(rows[1]["predicted_good_probability"], 0.0)

    def test_connected_window_intersects_local_final_and_support_constraints(self):
        manifest = {
            "controls": ["control"], "incoming_context": ["incoming"],
            "predicted_outputs": ["quality"],
            "local_constraints": [{"name": "quality", "direction": "less_equal", "target": 1.0}],
        }
        predictors = [({"id": "stage"}, manifest, FakePredictor())]
        config = {
            "external_context": {"stage.incoming": 0.1},
            "final_specifications": [{"name": "quality", "direction": "greater_equal", "target": 0.5}],
            "connected_window": {"support_threshold": 0.5, "enforce_local_constraints": True},
        }
        result = _run_connected_window(
            predictors, config, {}, {"control": np.asarray([0.2, 0.5, 1.0])},
        )
        np.testing.assert_array_equal(result["process_feasible"], [False, True, False])
        np.testing.assert_array_equal(result["trusted_feasible"], [False, True, False])
        self.assertEqual(result["bottleneck"][0], "final.quality:lower")
        self.assertEqual(result["bottleneck"][2], "local.stage.quality:upper")

    def test_connected_window_accepts_scalar_center(self):
        manifest = {
            "controls": ["control"], "incoming_context": ["incoming"],
            "predicted_outputs": ["quality"], "local_constraints": [],
        }
        result = _run_connected_window(
            [({"id": "stage"}, manifest, FakePredictor())],
            {"external_context": {"stage.incoming": 0.1},
             "final_specifications": [{"name": "quality", "direction": "greater_equal", "target": 0.5}],
             "connected_window": {"support_threshold": 0.5}},
            {}, {"control": 0.8},
        )
        self.assertTrue(bool(result["trusted_feasible"]))
        self.assertIsInstance(str(result["bottleneck"]), str)

    def test_diverse_candidate_budget_is_deterministic_and_eligible(self):
        points = np.asarray([[0., 0.], [0., 1.], [1., 0.], [1., 1.], [.5, .5]])
        eligible = np.asarray([True, False, True, True, True])
        order = np.asarray([4, 3, 2, 0, 1])
        first = _diverse_candidate_indices(points, eligible, order, 3)
        second = _diverse_candidate_indices(points, eligible, order, 3)
        np.testing.assert_array_equal(first, second)
        self.assertEqual(len(first), 3)
        self.assertTrue(np.all(eligible[first]))

    def test_disconnected_true_intervals_are_not_joined(self):
        grid = np.asarray([0.0, 1.0, 2.0, 3.0, 4.0])
        feasible = np.asarray([True, True, False, True, True])
        self.assertEqual(_all_true_intervals(grid, feasible), [[0.0, 1.0], [3.0, 4.0]])


if __name__ == "__main__":
    unittest.main()
