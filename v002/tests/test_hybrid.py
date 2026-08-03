"""Hybridの支持度、混合境界、学習結果を確認します。"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


VERSION_ROOT = Path(__file__).resolve().parents[1]
if str(VERSION_ROOT) not in sys.path:
    sys.path.insert(0, str(VERSION_ROOT))

from src.hybrid.model import blend_predictions, train_hybrid_model  # noqa: E402
from src.hybrid.support import SupportModel  # noqa: E402
from src.preprocessing import PreparedData  # noqa: E402
from src.settings import ProblemDefinition, VariableDefinition  # noqa: E402


def make_problem() -> ProblemDefinition:
    parameters = [
        VariableDefinition("x1", "入力1", "", "parameter", "", 0.0, 1.0, 0.5, None),
        VariableDefinition("x2", "入力2", "", "parameter", "", 0.0, 1.0, 0.5, None),
    ]
    objective = VariableDefinition(
        "y", "結果", "", "objective", "maximize", None, None, None, None
    )
    return ProblemDefinition(parameters, objective, [], [], [objective], 9)


def make_prepared() -> PreparedData:
    inputs = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
    return PreparedData(
        raw_parameters=inputs.copy(),
        normalized_parameters=inputs,
        result_means={"y": np.array([0.0, 1.0, 1.0, 2.0])},
        result_noise_std={"y": 0.01},
        repeat_counts=np.ones(4, dtype=int),
        warnings=[],
    )


class HybridTest(unittest.TestCase):
    def test_blend_boundaries(self) -> None:
        gp = np.array([1.0, 1.0, 1.0])
        nn = np.array([3.0, 3.0, 3.0])
        support = np.array([0.0, 0.5, 1.0])
        np.testing.assert_allclose(blend_predictions(gp, nn, support), [1.0, 2.0, 3.0])

    def test_support_is_bounded_and_higher_near_measurements(self) -> None:
        model = SupportModel.fit(np.array([[0.0], [0.1]]))
        values = model.predict(np.array([[0.05], [1.0]]))
        self.assertTrue(np.all((0.0 <= values) & (values <= 1.0)))
        self.assertGreater(values[0], values[1])

    def test_training_produces_finite_hybrid_prediction(self) -> None:
        training = train_hybrid_model(make_prepared(), make_problem())
        prediction = training.model.predict(np.array([[0.25, 0.5], [0.75, 0.5]]))
        values = prediction.results["y"]
        self.assertEqual(values["hybrid_mean"].shape, (2,))
        self.assertTrue(np.all(np.isfinite(values["hybrid_mean"])))
        lower = np.minimum(values["gp_mean"], values["nn_pred"])
        upper = np.maximum(values["gp_mean"], values["nn_pred"])
        self.assertTrue(np.all(values["hybrid_mean"] >= lower - 1e-12))
        self.assertTrue(np.all(values["hybrid_mean"] <= upper + 1e-12))

    def test_fixed_seed_reproduces_predictions(self) -> None:
        query = np.array([[0.4, 0.6]])
        first = train_hybrid_model(make_prepared(), make_problem()).model.predict(query)
        second = train_hybrid_model(make_prepared(), make_problem()).model.predict(query)
        np.testing.assert_allclose(
            first.results["y"]["hybrid_mean"],
            second.results["y"]["hybrid_mean"],
            rtol=0.0,
            atol=0.0,
        )


if __name__ == "__main__":
    unittest.main()
