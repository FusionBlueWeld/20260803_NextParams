"""NN予測空間の学習、再現性、run採番を確認します。"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


VERSION_ROOT = Path(__file__).resolve().parents[1]
if str(VERSION_ROOT) not in sys.path:
    sys.path.insert(0, str(VERSION_ROOT))

from src.nn.reporting import next_run_id, response_space_path  # noqa: E402
from src.nn.trainer import train_network  # noqa: E402
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


class NeuralNetworkTest(unittest.TestCase):
    def test_prediction_shape_and_finite_values(self) -> None:
        result = train_network(make_prepared(), make_problem())
        prediction = result.model.predict(np.array([[0.25, 0.5], [0.75, 0.5]]))
        self.assertEqual(prediction.shape, (2, 1))
        self.assertTrue(np.all(np.isfinite(prediction)))

    def test_fixed_seed_reproduces_predictions(self) -> None:
        first = train_network(make_prepared(), make_problem()).model.predict(
            np.array([[0.4, 0.6]])
        )
        second = train_network(make_prepared(), make_problem()).model.predict(
            np.array([[0.4, 0.6]])
        )
        np.testing.assert_allclose(first, second, rtol=0.0, atol=0.0)

    def test_run_number_uses_existing_response_files(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "nn_response_space_run_0001.csv").touch()
            (root / "nn_response_space_run_0003.csv").touch()
            self.assertEqual(next_run_id(root), "run_0004")
            self.assertEqual(
                response_space_path(root, "run_0004").name,
                "nn_response_space_run_0004.csv",
            )


if __name__ == "__main__":
    unittest.main()
