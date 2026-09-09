"""Unit tests for the convection-drying validation model."""

from __future__ import annotations

import unittest

import numpy as np

try:
    from .physics_model import evaluate_model
except ImportError:  # Supports direct unittest discovery from this folder.
    from physics_model import evaluate_model


class ConvectionDryingModelTests(unittest.TestCase):
    def test_outputs_are_physical_and_deterministic(self) -> None:
        first = evaluate_model(85.0, 2.0, 8.0)
        second = evaluate_model(85.0, 2.0, 8.0)
        self.assertEqual(set(first), {"energy_kj_m2", "residual_moisture_pct", "defect_index"})
        for key in first:
            np.testing.assert_array_equal(first[key], second[key])
            self.assertTrue(np.all(np.isfinite(first[key])))
        self.assertGreater(float(first["energy_kj_m2"]), 0.0)
        self.assertGreaterEqual(float(first["residual_moisture_pct"]), 0.0)
        self.assertGreaterEqual(float(first["defect_index"]), 0.0)
        self.assertLessEqual(float(first["defect_index"]), 1.0)

    def test_temperature_and_time_promote_drying(self) -> None:
        cool = evaluate_model(55.0, 2.0, 4.0)
        hot = evaluate_model(95.0, 2.0, 4.0)
        long = evaluate_model(75.0, 2.0, 12.0)
        self.assertLess(float(hot["residual_moisture_pct"]), float(cool["residual_moisture_pct"]))
        self.assertLess(float(long["residual_moisture_pct"]), float(evaluate_model(75.0, 2.0, 4.0)["residual_moisture_pct"]))

    def test_aggressive_short_drying_increases_defect_surrogate(self) -> None:
        gentle = evaluate_model(65.0, 0.5, 12.0)
        aggressive = evaluate_model(105.0, 6.0, 2.0)
        self.assertGreater(
            float(aggressive["defect_index"]),
            float(gentle["defect_index"]),
        )

    def test_air_speed_has_diminishing_mass_transfer_benefit(self) -> None:
        low_gain = (
            float(evaluate_model(75.0, 2.5, 8.0)["residual_moisture_pct"])
            - float(evaluate_model(75.0, 1.5, 8.0)["residual_moisture_pct"])
        )
        high_gain = (
            float(evaluate_model(75.0, 6.0, 8.0)["residual_moisture_pct"])
            - float(evaluate_model(75.0, 5.0, 8.0)["residual_moisture_pct"])
        )
        self.assertLess(abs(high_gain), abs(low_gain))

    def test_broadcasting_and_scalar_outputs(self) -> None:
        result = evaluate_model(
            np.array([[55.0], [85.0]]),
            np.array([1.0, 3.0, 5.0]),
            8.0,
        )
        for value in result.values():
            self.assertEqual(value.shape, (2, 3))
        scalar = evaluate_model(75.0, 2.0, 6.0)
        for value in scalar.values():
            self.assertEqual(value.shape, ())

    def test_invalid_nonfinite_and_domain_values(self) -> None:
        invalid_cases = (
            (np.nan, 2.0, 6.0),
            (np.inf, 2.0, 6.0),
            (-np.inf, 2.0, 6.0),
            (44.999, 2.0, 6.0),
            (105.001, 2.0, 6.0),
            (75.0, 0.499, 6.0),
            (75.0, 6.001, 6.0),
            (75.0, 2.0, 0.999),
            (75.0, 2.0, 12.001),
        )
        for args in invalid_cases:
            with self.subTest(args=args):
                with self.assertRaises(ValueError):
                    evaluate_model(*args)

    def test_full_grid_is_finite_and_respects_bounds(self) -> None:
        temperatures = np.arange(45.0, 105.0 + 5.0, 5.0)
        speeds = np.arange(0.5, 6.0 + 0.5, 0.5)
        times = np.arange(1.0, 12.0 + 1.0, 1.0)
        result = evaluate_model(temperatures[:, None, None], speeds[None, :, None], times[None, None, :])
        for value in result.values():
            self.assertEqual(value.shape, (13, 12, 12))
            self.assertTrue(np.all(np.isfinite(value)))
        self.assertTrue(np.all(result["energy_kj_m2"] > 0.0))
        self.assertTrue(np.all(result["residual_moisture_pct"] >= 0.0))
        self.assertTrue(np.all(result["defect_index"] >= 0.0))
        self.assertTrue(np.all(result["defect_index"] <= 1.0))


if __name__ == "__main__":
    unittest.main()
