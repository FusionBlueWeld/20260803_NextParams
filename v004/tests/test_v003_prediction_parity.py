"""接続なしのv004予測コアが現行v003と数値一致することを確認します。"""

from __future__ import annotations

import unittest

import numpy as np

from v003.src.hybrid.model import train_hybrid_model as train_v003
from v003.src.preprocessing import PreparedData as PreparedDataV003
from v003.src.settings import ProblemDefinition as ProblemV003, VariableDefinition as VariableV003
from v004.src.hybrid.model import train_hybrid_model as train_v004
from v004.src.preprocessing import PreparedData as PreparedDataV004
from v004.src.settings import ProblemDefinition as ProblemV004, VariableDefinition as VariableV004


def make_inputs(version: str):
    variable = VariableV003 if version == "v003" else VariableV004
    problem_type = ProblemV003 if version == "v003" else ProblemV004
    prepared_type = PreparedDataV003 if version == "v003" else PreparedDataV004
    parameters = [
        variable("x1", "入力1", "", "parameter", "", 0.0, 1.0, 0.5, None),
        variable("x2", "入力2", "", "parameter", "", 0.0, 1.0, 0.5, None),
    ]
    objective = variable("y", "結果", "", "objective", "maximize", None, None, None, None)
    problem = problem_type(parameters, objective, [], [], [objective], 9)
    inputs = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
    prepared = prepared_type(
        raw_parameters=inputs.copy(), normalized_parameters=inputs,
        result_means={"y": np.array([0.0, 1.0, 1.0, 2.0])},
        result_noise_std={"y": 0.01}, repeat_counts=np.ones(4, dtype=int), warnings=[],
    )
    return prepared, problem


class V003PredictionParityTest(unittest.TestCase):
    def test_prediction_core_matches_v003(self) -> None:
        prepared3, problem3 = make_inputs("v003")
        prepared4, problem4 = make_inputs("v004")
        model3 = train_v003(prepared3, problem3).model
        model4 = train_v004(prepared4, problem4).model
        points = np.array([[0.1, 0.2], [0.5, 0.5], [0.9, 0.8]])
        prediction3 = model3.predict(points)
        prediction4 = model4.predict(points)
        np.testing.assert_allclose(prediction4.support, prediction3.support)
        for key in prediction3.results["y"]:
            np.testing.assert_allclose(
                prediction4.results["y"][key], prediction3.results["y"][key]
            )


if __name__ == "__main__":
    unittest.main()
