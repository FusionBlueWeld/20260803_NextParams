import unittest

import numpy as np

from validation.single.src.checks.gp_residual_ablation import (
    NoGpMeanAdapter,
    aggregate,
    calculate_metrics,
    contains_point,
)


class _Prediction:
    def __init__(self):
        self.support = np.array([0.2, 0.8])
        self.results = {
            "y": {
                "gp_mean": np.array([1.0, 2.0]),
                "gp_std": np.array([0.3, 0.4]),
                "nn_pred": np.array([10.0, 20.0]),
                "hybrid_mean": np.array([11.0, 22.0]),
                "hybrid_std": np.array([0.3, 0.4]),
            }
        }


class _Model:
    gp_models = {"y": object()}

    def predict(self, x):
        return _Prediction()


class GpResidualAblationTests(unittest.TestCase):
    def test_initial_design_can_be_arrival_step_zero(self):
        self.assertTrue(contains_point(np.array([[0.0, 1.0], [2.0, 3.0]]), np.array([2.0, 3.0])))
        self.assertFalse(contains_point(np.array([[0.0, 1.0]]), np.array([2.0, 3.0])))

    def test_adapter_changes_mean_only(self):
        base = _Model()
        adapted = NoGpMeanAdapter(base)
        prediction = adapted.predict(np.zeros((2, 1)))
        self.assertTrue(np.array_equal(prediction.support, [0.2, 0.8]))
        self.assertTrue(np.array_equal(prediction.results["y"]["hybrid_std"], [0.3, 0.4]))
        self.assertTrue(np.array_equal(prediction.results["y"]["hybrid_mean"], [10.0, 20.0]))
        self.assertTrue(np.array_equal(prediction.results["y"]["gp_mean"], [1.0, 2.0]))

    def test_metrics_include_local_window_and_nlpd(self):
        metrics = calculate_metrics(
            {"y": np.array([0.0, 1.0, 2.0])},
            {"y": np.array([0.0, 1.0, 2.0])},
            {"y": np.array([0.1, 0.1, 0.1])},
            np.array([True, True, False]),
            np.array([True, False, False]),
            np.array([[0.0], [0.1], [1.0]]),
            np.array([0.0]),
            local_radius=0.2,
        )
        self.assertEqual(metrics["local_count"], 2)
        self.assertEqual(metrics["global"]["outputs"]["y"]["rmse"], 0.0)
        self.assertTrue(np.isfinite(metrics["local"]["outputs"]["y"]["nlpd"]))
        self.assertEqual(metrics["global"]["feasible_accuracy"], 2 / 3)

    def test_aggregate_pairs_and_censored_arrival(self):
        def row(arm, found, arrival, nrmse):
            scope = {"macro_nrmse": nrmse, "macro_nlpd": nrmse, "macro_coverage95": 1.0 - nrmse, "feasible_accuracy": 1.0 - nrmse}
            return {
                "simulator": "toy", "arm": arm, "seed": 0,
                "found": found, "arrival_step": arrival,
                "metrics": {"global": scope, "local": scope},
                "history": [],
            }
        summary = aggregate([row("gp", True, 2, 0.2), row("no_gp", False, None, 0.3)], 3)
        paired = summary["paired"][0]
        self.assertEqual(paired["no_gp_arrival"], 4)
        self.assertEqual(summary["by_simulator"]["toy"]["win_tie_loss"]["global_macro_nrmse"]["gp"], 1)


if __name__ == "__main__":
    unittest.main()
