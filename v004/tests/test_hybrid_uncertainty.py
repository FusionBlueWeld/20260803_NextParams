"""v004がv003と同じNN ensemble・Hybrid不確実性を使うことを確認します。"""

from __future__ import annotations

import unittest

import numpy as np

from v004.src.hybrid.model import train_hybrid_model
from v004.src.preprocessing import PreparedData
from v004.src.settings import ProblemDefinition, VariableDefinition


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


class HybridUncertaintyTests(unittest.TestCase):
    def test_single_member_scale_one_reproduces_legacy_std(self) -> None:
        model = train_hybrid_model(
            make_prepared(), make_problem(), ensemble_size=1,
            uncertainty_calibration_scale=1.0,
        ).model
        values = model.predict(np.array([[0.25, 0.5], [0.75, 0.5]])).results["y"]
        np.testing.assert_allclose(values["nn_std"], 0.0)
        np.testing.assert_allclose(values["raw_hybrid_std"], values["gp_std"])
        np.testing.assert_allclose(values["hybrid_std"], values["gp_std"])

    def test_five_members_add_uncertainty_and_keep_acquisition_unscaled(self) -> None:
        model = train_hybrid_model(
            make_prepared(), make_problem(), ensemble_size=5,
            uncertainty_calibration_scale=2.0,
        ).model
        self.assertEqual(len(model.nn_model.members), 5)
        values = model.predict(np.array([[0.25, 0.5], [0.75, 0.5]])).results["y"]
        expected = np.sqrt(np.square(values["gp_std"]) + np.square(values["nn_std"]))
        self.assertTrue(np.any(values["nn_std"] > 0.0))
        np.testing.assert_allclose(values["raw_hybrid_std"], expected)
        np.testing.assert_allclose(values["hybrid_std"], 2.0 * expected)
        np.testing.assert_allclose(values["acquisition_std"], expected)

    def test_configuration_is_validated(self) -> None:
        with self.assertRaises(ValueError):
            train_hybrid_model(make_prepared(), make_problem(), ensemble_size=0)
        with self.assertRaises(ValueError):
            train_hybrid_model(
                make_prepared(), make_problem(), uncertainty_calibration_scale=0.0
            )
        with self.assertRaises(ValueError):
            train_hybrid_model(
                make_prepared(), make_problem(), acquisition_uncertainty_scale=0.0
            )


if __name__ == "__main__":
    unittest.main()
