"""Unit tests for the press-forming synthetic validation oracle."""

from __future__ import annotations

import csv
import sys
import unittest
from pathlib import Path

import numpy as np

try:
    from .physics_model import evaluate_model
except ImportError:  # Support ``python validation/single/press_forming/test_model.py``.
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from validation.single.press_forming.physics_model import evaluate_model


HERE = Path(__file__).resolve().parent


class PressFormingModelTest(unittest.TestCase):
    """Check physical structure, input contract, and deterministic behavior."""

    def test_clearance_burr_minimum_and_force_trend(self) -> None:
        low = evaluate_model(3.0, 200.0, 60.0)
        nominal = evaluate_model(8.0, 200.0, 60.0)
        high = evaluate_model(15.0, 200.0, 60.0)
        self.assertLess(nominal["burr_height_mm"], low["burr_height_mm"])
        self.assertLess(nominal["burr_height_mm"], high["burr_height_mm"])
        self.assertGreater(low["peak_force_kn"], nominal["peak_force_kn"])
        self.assertGreater(high["peak_force_kn"], nominal["peak_force_kn"])

    def test_speed_and_holder_limiting_trends(self) -> None:
        slow = evaluate_model(12.0, 80.0, 60.0)
        fast = evaluate_model(12.0, 320.0, 60.0)
        weak_holder = evaluate_model(12.0, 200.0, 20.0)
        strong_holder = evaluate_model(12.0, 200.0, 100.0)
        self.assertGreater(fast["peak_force_kn"], slow["peak_force_kn"])
        self.assertGreater(fast["flatness_error_mm"], slow["flatness_error_mm"])
        self.assertGreater(
            weak_holder["flatness_error_mm"], strong_holder["flatness_error_mm"]
        )
        self.assertGreater(
            strong_holder["peak_force_kn"], weak_holder["peak_force_kn"]
        )

    def test_invalid_nonfinite_and_domain_values_raise_value_error(self) -> None:
        for bad in (np.nan, np.inf, -np.inf):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    evaluate_model(bad, 200.0, 60.0)
                with self.assertRaises(ValueError):
                    evaluate_model(8.0, bad, 60.0)
                with self.assertRaises(ValueError):
                    evaluate_model(8.0, 200.0, bad)
        for bad, good in ((2.999, 8.0), (15.001, 8.0)):
            with self.subTest(clearance=bad):
                with self.assertRaises(ValueError):
                    evaluate_model(bad, 200.0, 60.0)
            with self.subTest(speed=bad):
                with self.assertRaises(ValueError):
                    evaluate_model(8.0, 79.999 if bad < 8 else 320.001, 60.0)
            with self.subTest(holder=bad):
                with self.assertRaises(ValueError):
                    evaluate_model(8.0, 200.0, 19.999 if bad < 8 else 100.001)
            self.assertTrue(np.isfinite(evaluate_model(good, 200.0, 60.0)["burr_height_mm"]))

    def test_broadcast_shapes_and_scalar_arrays(self) -> None:
        result = evaluate_model(
            np.array([[3.0], [8.0]]),
            np.array([80.0, 200.0, 320.0]),
            60.0,
        )
        for value in result.values():
            self.assertIsInstance(value, np.ndarray)
            self.assertEqual(value.shape, (2, 3))
        scalar_result = evaluate_model(8.0, 200.0, 60.0)
        self.assertEqual(scalar_result["burr_height_mm"].shape, ())

    def test_problem_grid_is_finite_bounded_and_has_feasible_region(self) -> None:
        with (HERE / "problem.csv").open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(
            list(rows[0].keys()),
            [
                "column",
                "display_name",
                "unit",
                "role",
                "direction",
                "lower",
                "upper",
                "step",
                "target",
            ],
        )
        parameters = [row for row in rows if row["role"] == "parameter"]
        self.assertEqual(len(parameters), 3)
        self.assertFalse(any(row["role"] == "input" for row in rows))
        self.assertEqual(
            {row["column"] for row in parameters},
            {"clearance_pct", "stroke_speed_mm_s", "blank_holder_force_kn"},
        )
        levels = []
        for row in parameters:
            lower = {"clearance_pct": 3.0, "stroke_speed_mm_s": 80.0, "blank_holder_force_kn": 20.0}[row["column"]]
            upper = {"clearance_pct": 15.0, "stroke_speed_mm_s": 320.0, "blank_holder_force_kn": 100.0}[row["column"]]
            step = {"clearance_pct": 1.0, "stroke_speed_mm_s": 20.0, "blank_holder_force_kn": 10.0}[row["column"]]
            grid = np.arange(lower, upper + step * 0.5, step)
            levels.append(grid)
            self.assertGreaterEqual(grid.size, 9)
            self.assertLessEqual(grid.size, 15)
        cc, ss, hh = np.meshgrid(*levels, indexing="ij")
        result = evaluate_model(cc, ss, hh)
        self.assertEqual(cc.size, 1521)
        for value in result.values():
            self.assertTrue(np.all(np.isfinite(value)))
            self.assertEqual(value.shape, cc.shape)
        feasible = (result["peak_force_kn"] <= 19.5) & (result["flatness_error_mm"] <= 0.20)
        self.assertGreater(np.count_nonzero(feasible), 0)
        self.assertGreater(np.count_nonzero(~feasible), 0)

        initial = np.meshgrid(
            np.array([3.0, 9.0, 15.0]),
            np.array([80.0, 200.0, 320.0]),
            np.array([20.0, 60.0, 100.0]),
            indexing="ij",
        )
        initial_result = evaluate_model(*initial)
        initial_feasible = (initial_result["peak_force_kn"] <= 19.5) & (
            initial_result["flatness_error_mm"] <= 0.20
        )
        self.assertGreater(np.count_nonzero(initial_feasible), 0)

    def test_real_registry_load_audit_and_individual_constraint_violations(self) -> None:
        """Exercise the parent registry and prove each declared constraint bites."""

        from validation.single.src.benchmark import audit
        from validation.single.src.simulators import load

        simulator = load("press_forming")
        self.assertEqual(simulator.parameter_columns, [
            "clearance_pct",
            "stroke_speed_mm_s",
            "blank_holder_force_kn",
        ])
        self.assertEqual(simulator.output_columns, [
            "burr_height_mm",
            "peak_force_kn",
            "flatness_error_mm",
        ])
        summary = audit(simulator)
        self.assertEqual(summary["candidate_count"], 1521)
        self.assertGreater(summary["initial_feasible_count"], 0)
        self.assertGreater(summary["feasible_count"], 0)
        self.assertLess(summary["feasible_count"], summary["candidate_count"])

        grid_outputs = simulator.evaluate_points(simulator.grid())
        for constraint in simulator.constraints:
            values = grid_outputs[constraint.column]
            if constraint.direction == "less_equal":
                violations = values > constraint.target
            else:
                violations = values < constraint.target
            with self.subTest(constraint=constraint.column):
                self.assertGreater(int(np.count_nonzero(violations)), 0)

    def test_deterministic(self) -> None:
        args = (
            np.array([3.0, 8.0, 15.0]),
            np.array([80.0, 200.0, 320.0]),
            np.array([20.0, 60.0, 100.0]),
        )
        first = evaluate_model(*args)
        second = evaluate_model(*args)
        for name in first:
            np.testing.assert_array_equal(first[name], second[name])


if __name__ == "__main__":
    unittest.main()
