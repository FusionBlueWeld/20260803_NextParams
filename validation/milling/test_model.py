"""Unit tests for the eight-input deterministic milling oracle."""

from __future__ import annotations

import unittest

import numpy as np

try:
    from .physics_model import evaluate_model
except ImportError:  # pragma: no cover - supports direct script execution
    from physics_model import evaluate_model


_MID = (4500.0, 0.10, 2.0, 12.5, 45.0, 5.0, 1.0, 0.15)


class MillingModelTests(unittest.TestCase):
    """Check physical trends, domains, array behavior, and determinism."""

    def test_mrr_scales_with_depth_engagement_speed_and_teeth(self) -> None:
        cases = (
            (2, 0.50, 1.00),
            (3, 5.00, 10.00),
            (0, 1800.0, 3600.0),
            (5, 2.00, 4.00),
        )
        for index, base_value, changed_value in cases:
            base_case = list(_MID)
            changed = list(_MID)
            base_case[index] = base_value
            changed[index] = changed_value
            base = evaluate_model(*base_case)
            value = evaluate_model(*changed)["material_removal_rate_cm3_min"]
            self.assertAlmostEqual(
                float(value) / float(base["material_removal_rate_cm3_min"]),
                changed_value / base_value,
            )

    def test_power_tracks_mrr_and_cutting_speed_wear_effects(self) -> None:
        base = evaluate_model(*_MID)
        changed_feed = list(_MID)
        changed_feed[1] = 0.12
        feed_result = evaluate_model(*changed_feed)
        base_ratio = float(base["spindle_power_kw"]) / float(
            base["material_removal_rate_cm3_min"]
        )
        feed_ratio = float(feed_result["spindle_power_kw"]) / float(
            feed_result["material_removal_rate_cm3_min"]
        )
        self.assertAlmostEqual(feed_ratio, base_ratio, delta=base_ratio * 0.01)

        worn = list(_MID)
        worn[7] = 0.30
        larger_cutter = list(_MID)
        larger_cutter[4] = 60.0
        self.assertGreater(
            float(evaluate_model(*worn)["spindle_power_kw"]),
            float(base["spindle_power_kw"]),
        )
        self.assertGreater(
            float(evaluate_model(*larger_cutter)["spindle_power_kw"]),
            float(base["spindle_power_kw"]),
        )

    def test_feed_geometry_nose_radius_and_wear_trends(self) -> None:
        fine = list(_MID)
        fine[1] = 0.04
        large_radius = list(_MID)
        large_radius[6] = 1.60
        worn = list(_MID)
        worn[7] = 0.30
        base = evaluate_model(*_MID)
        self.assertLess(
            float(evaluate_model(*fine)["roughness_ra_um"]),
            float(base["roughness_ra_um"]),
        )
        self.assertLess(
            float(evaluate_model(*large_radius)["roughness_ra_um"]),
            float(base["roughness_ra_um"]),
        )
        self.assertGreater(
            float(evaluate_model(*worn)["roughness_ra_um"]),
            float(base["roughness_ra_um"]),
        )

    def test_multiple_tooth_passing_resonances_are_nonmonotonic(self) -> None:
        # For z=4, 2400/4500/6600 rpm correspond to 160/300/440 Hz.
        speeds = np.array(
            [2100.0, 2400.0, 2700.0, 4200.0, 4800.0, 6300.0, 6600.0, 6900.0]
        )
        values = list(_MID)
        values[0] = speeds
        values[5] = 4.0
        roughness = evaluate_model(*values)["roughness_ra_um"]
        self.assertGreater(float(roughness[1]), float(roughness[0]))
        self.assertGreater(float(roughness[1]), float(roughness[2]))
        self.assertGreater(float(roughness[4]), float(roughness[3]))
        self.assertGreater(float(roughness[4]), float(roughness[5]))
        self.assertGreater(float(roughness[6]), float(roughness[5]))
        self.assertGreater(float(roughness[6]), float(roughness[7]))

    def test_invalid_values_raise_for_all_eight_inputs(self) -> None:
        bounds = (
            (1800.0, 7200.0),
            (0.04, 0.16),
            (0.50, 3.50),
            (5.0, 20.0),
            (30.0, 60.0),
            (2.0, 8.0),
            (0.40, 1.60),
            (0.0, 0.30),
        )
        for index, (lower, upper) in enumerate(bounds):
            for bad in (lower - 1.0e-3, upper + 1.0e-3, np.nan, np.inf, -np.inf):
                case = list(_MID)
                case[index] = bad
                with self.subTest(index=index, bad=bad):
                    with self.assertRaises(ValueError):
                        evaluate_model(*case)

    def test_broadcast_outputs_have_identical_shape(self) -> None:
        values = list(_MID)
        values[0] = np.array([[1800.0], [5400.0]])
        values[1] = np.array([0.04, 0.10, 0.16])
        outputs = evaluate_model(*values)
        self.assertEqual(set(outputs), {
            "material_removal_rate_cm3_min",
            "roughness_ra_um",
            "spindle_power_kw",
        })
        for output in outputs.values():
            self.assertEqual(output.shape, (2, 3))
            self.assertTrue(np.all(np.isfinite(output)))

    def test_full_4_to_the_8_grid_is_finite_and_deterministic(self) -> None:
        axes = (
            np.array([1800.0, 3600.0, 5400.0, 7200.0]),
            np.array([0.04, 0.08, 0.12, 0.16]),
            np.array([0.50, 1.50, 2.50, 3.50]),
            np.array([5.0, 10.0, 15.0, 20.0]),
            np.array([30.0, 40.0, 50.0, 60.0]),
            np.array([2.0, 4.0, 6.0, 8.0]),
            np.array([0.40, 0.80, 1.20, 1.60]),
            np.array([0.0, 0.10, 0.20, 0.30]),
        )
        grid = np.meshgrid(*axes, indexing="ij")
        first = evaluate_model(*grid)
        second = evaluate_model(*grid)
        self.assertEqual(first["material_removal_rate_cm3_min"].size, 4**8)
        for name in first:
            self.assertTrue(np.all(np.isfinite(first[name])))
            np.testing.assert_array_equal(first[name], second[name])


if __name__ == "__main__":
    unittest.main()
