"""v003 検証ヘルパー自身の回帰テスト。"""

from __future__ import annotations

import csv
import sys
import tempfile
import unittest
from pathlib import Path

from validation.single.src.checks.v003_synthetic import _check_synthetic_response
from validation.single.src.checks.v003_runtime import detect_prediction


class V003ValidationHelpersTest(unittest.TestCase):
    def test_prediction_column_fallbacks(self) -> None:
        self.assertEqual(detect_prediction({"y_mean": "1.5"}, "y"), 1.5)
        self.assertEqual(detect_prediction({"y_hybrid_mean": "2.5"}, "y"), 2.5)
        self.assertEqual(detect_prediction({"y_nn_pred": "3.5"}, "y"), 3.5)

    def test_synthetic_response_checks_true_rules(self) -> None:
        rows = []
        for p1 in range(3):
            for p2 in range(5):
                for p3 in (0, 4):
                    forbidden = p3 >= 4
                    rows.append(
                        {
                            "p1": p1,
                            "p2": p2,
                            "p3": p3,
                            "y_mono_mean": 1 + p1,
                            "y_nonnegative_mean": 0.5 + p1,
                            "y_quiet_mean": 2 + 0.01 * p2,
                            "experiment_allowed": str(not forbidden),
                            "preferred_multiplier": (
                                1.0 if forbidden else (1.25 if p2 <= 2 else 1.0)
                            ),
                        }
                    )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "response.csv"
            with path.open("w", encoding="utf-8", newline="") as file:
                writer = csv.DictWriter(file, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            result = _check_synthetic_response(path)
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["monotonic_p1_violation_rate"], 0.0)
        self.assertEqual(result["low_sensitivity_violation_rate"], 0.0)
        self.assertEqual(result["forbidden_policy_flag_errors"], 0)
        self.assertEqual(result["preferred_policy_multiplier_errors"], 0)
        self.assertGreater(result["nonnegative_min"], 0.0)


if __name__ == "__main__":
    unittest.main()
