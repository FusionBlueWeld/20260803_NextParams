import copy
import unittest

import numpy as np

from validation.multistage.src.checks.connected_window_validation import _acceptance
from validation.multistage.src.checks.calibrate_connected_window import _fit_offsets


class WindowAcceptanceTests(unittest.TestCase):
    def test_residual_fit_keeps_input_mask_and_population_across_alphas(self):
        predictors = [({"id": name}, {"predicted_outputs": ["y"]}, None) for name in ("a", "b")]
        truth = {name: {"y": np.linspace(0., 1., 130)} for name in ("a", "b")}
        predicted = {
            "valid_connection": np.ones(130, dtype=bool),
            "stage_support": {"a": np.r_[np.zeros(10), np.ones(120)],
                              "b": np.r_[np.ones(120), np.zeros(10)]},
            "stage_outputs": {name: {"y": np.zeros(130)} for name in ("a", "b")},
            "stage_stds": {name: {"y": np.ones(130)} for name in ("a", "b")},
        }
        for alpha in (.05, .15):
            _, counts = _fit_offsets(predictors, truth, predicted, alpha)
            self.assertEqual(counts, {"a": 110, "b": 110})
            self.assertTrue(predicted["valid_connection"].all())

    def test_false_allowance_even_outside_center_segment_fails(self):
        rows = {"x": {"true_window_recovery_grid_fraction": .8}}
        result = _acceptance(True, rows, {"false_positive": 1}, 0, [0., 0.], 241, ["x"])
        self.assertEqual(result["status"], "fail")
        self.assertFalse(result["checks"]["no_false_allowances"])

    def test_empty_low_recovery_missing_axes_and_center_ng_fail(self):
        args = [True, {"x": {"true_window_recovery_grid_fraction": .8}},
                {"false_positive": 0}, 0, [0., 0.], 241, ["x"]]
        self.assertEqual(_acceptance(*args)["status"], "pass")
        self.assertEqual(_acceptance(*args, center_predicted_feasible=False)["status"], "fail")
        for index, value in [(0, False), (1, {"x": {"true_window_recovery_grid_fraction": .1}}),
                             (1, {"x": {"true_window_recovery_grid_fraction": None}}),
                             (3, 1), (4, [.1]), (6, ["x", "y"])]:
            bad = copy.deepcopy(args); bad[index] = value
            with self.subTest(index=index, value=value):
                self.assertEqual(_acceptance(*bad)["status"], "fail")


if __name__ == "__main__":
    unittest.main()
