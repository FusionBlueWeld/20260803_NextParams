"""探索器へ使用禁止範囲と好ましい範囲が接続されていることを確認します。"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

VERSION_ROOT = Path(__file__).resolve().parents[1]
if str(VERSION_ROOT) not in sys.path:
    sys.path.insert(0, str(VERSION_ROOT))

from src.hybrid.model import HybridPrediction  # noqa: E402
from src.hybrid.optimizer import run_optimization  # noqa: E402
from src.policies import RegionCondition, RegionPolicy, RegionRule  # noqa: E402
from src.preprocessing import PreparedData  # noqa: E402
from src.settings import ProblemDefinition, VariableDefinition  # noqa: E402
from src.validation import ExperimentData  # noqa: E402


class _FlatModel:
    gp_models = {"y": SimpleNamespace(y_scale=1.0)}

    def predict(self, inputs: np.ndarray) -> HybridPrediction:
        size = len(inputs)
        values = np.zeros(size)
        return HybridPrediction(
            support=values,
            results={
                "y": {
                    "hybrid_mean": values,
                    "hybrid_std": np.ones(size),
                    "nn_pred": values,
                }
            },
        )


class OptimizerPolicyTest(unittest.TestCase):
    def test_forbidden_is_removed_and_preferred_wins_equal_base_score(self) -> None:
        parameter = VariableDefinition(
            "p", "入力", "", "parameter", "", 0.0, 3.0, 1.0, None
        )
        objective = VariableDefinition(
            "y", "結果", "", "objective", "maximize", None, None, None, None
        )
        problem = ProblemDefinition([parameter], objective, [], [], [objective], 4)
        prepared = PreparedData(
            raw_parameters=np.array([[0.0]]),
            normalized_parameters=np.array([[0.0]]),
            result_means={"y": np.array([0.0])},
            result_noise_std={"y": 0.1},
            repeat_counts=np.ones(1, dtype=int),
            warnings=[],
        )
        experiments = ExperimentData(["e1"], [[0.0]], {"y": [0.0]})
        policy = RegionPolicy(
            (
                RegionRule("ban", "forbidden", (RegionCondition("p", 3, 3),)),
                RegionRule("good", "preferred", (RegionCondition("p", 2, 2),), strength=5),
            )
        )

        result = run_optimization(
            experiments,
            prepared,
            problem,
            recommendation_count=1,
            model=_FlatModel(),
            region_policy=policy,
        )

        self.assertEqual(result.recommendations[0]["p"], 2.0)
        self.assertEqual(result.recommendations[0]["preferred_multiplier"], 2.0)
        self.assertEqual(result.forbidden_candidate_count, 1)
        self.assertEqual(result.allowed_candidate_count, 3)


if __name__ == "__main__":
    unittest.main()
