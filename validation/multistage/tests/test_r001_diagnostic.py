from __future__ import annotations

import unittest

from validation.multistage.functional_coating.fault_scenarios import REFERENCE_NOMINAL
from validation.multistage.functional_coating.pipeline import evaluate_line
from validation.multistage.src.checks.r001_diagnostic_validation import (
    _candidate_values,
    _constraint_probability,
    _failed_specs,
)


class R001DiagnosticValidationTests(unittest.TestCase):
    def test_constraint_probability_has_correct_direction(self):
        self.assertGreater(_constraint_probability(0.9, 0.02, "greater_equal", 0.8), 0.999)
        self.assertLess(_constraint_probability(0.7, 0.02, "greater_equal", 0.8), 0.001)
        self.assertGreater(_constraint_probability(1.0, 0.05, "less_equal", 2.0), 0.999)

    def test_between_probability_is_bounded(self):
        probability = _constraint_probability(70.0, 2.0, "between", (55.0, 85.0))
        self.assertGreaterEqual(probability, 0.0)
        self.assertLessEqual(probability, 1.0)

    def test_candidate_generation_keeps_current_off_grid_value(self):
        conditions = {**REFERENCE_NOMINAL, "coating_gap_um": 230.0}
        candidates = _candidate_values(conditions, {"coating"})
        self.assertEqual(len(candidates["coating_gap_um"]), 4 * 3 * 3)
        self.assertIn(230.0, candidates["coating_gap_um"])
        self.assertTrue((candidates["air_temperature_c"] == conditions["air_temperature_c"]).all())

    def test_failed_specs_reports_oracle_ng_outputs(self):
        conditions = {**REFERENCE_NOMINAL, "oven_temperature_c": 105.0}
        failed = _failed_specs(evaluate_line(conditions)["final"])
        self.assertIn("bond_strength_mpa", failed)
        self.assertIn("cure_fraction", failed)


if __name__ == "__main__":
    unittest.main()
