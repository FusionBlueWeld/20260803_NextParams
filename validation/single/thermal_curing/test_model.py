"""Unit tests for the deterministic thermal-curing benchmark."""

from __future__ import annotations

import csv
import json
import sys
import unittest
from pathlib import Path

import numpy as np

try:
    from .physics_model import (
        _AMBIENT_TEMPERATURE_C,
        _CURE_ACTIVATION_ENERGY_J_PER_MOL,
        _CURE_REFERENCE_RATE_PER_MIN,
        _CURE_REFERENCE_TEMPERATURE_K,
        _DEGRADATION_ACTIVATION_ENERGY_J_PER_MOL,
        _DEGRADATION_REFERENCE_RATE_PER_MIN,
        _DEGRADATION_REFERENCE_TEMPERATURE_K,
        _GAS_CONSTANT_J_PER_MOL_K,
        _THERMAL_THICKNESS_COEFFICIENT,
        _THERMAL_TIME_CONSTANT_MIN_AT_1MM,
        evaluate_model,
    )
except ImportError:  # Support ``python validation/single/thermal_curing/test_model.py``.
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from validation.single.thermal_curing.physics_model import (
        _AMBIENT_TEMPERATURE_C,
        _CURE_ACTIVATION_ENERGY_J_PER_MOL,
        _CURE_REFERENCE_RATE_PER_MIN,
        _CURE_REFERENCE_TEMPERATURE_K,
        _DEGRADATION_ACTIVATION_ENERGY_J_PER_MOL,
        _DEGRADATION_REFERENCE_RATE_PER_MIN,
        _DEGRADATION_REFERENCE_TEMPERATURE_K,
        _GAS_CONSTANT_J_PER_MOL_K,
        _THERMAL_THICKNESS_COEFFICIENT,
        _THERMAL_TIME_CONSTANT_MIN_AT_1MM,
        evaluate_model,
    )


ROOT = Path(__file__).resolve().parent


def _reference_integrals(oven_c: float, hold_min: float, thickness_mm: float, steps: int) -> tuple[float, float]:
    """Compute a refined midpoint reference for the two exposure integrals."""

    tau = _THERMAL_TIME_CONSTANT_MIN_AT_1MM + _THERMAL_THICKNESS_COEFFICIENT * (
        thickness_mm / 0.50
    ) ** 2
    fraction = (np.arange(steps, dtype=float) + 0.5) / steps
    t = hold_min * fraction
    temp_k = _AMBIENT_TEMPERATURE_C + (oven_c - _AMBIENT_TEMPERATURE_C) * (
        1.0 - np.exp(-t / tau)
    ) + 273.15
    cure_rate = _CURE_REFERENCE_RATE_PER_MIN * np.exp(
        _CURE_ACTIVATION_ENERGY_J_PER_MOL / _GAS_CONSTANT_J_PER_MOL_K
        * (1.0 / _CURE_REFERENCE_TEMPERATURE_K - 1.0 / temp_k)
    )
    degradation_rate = _DEGRADATION_REFERENCE_RATE_PER_MIN * np.exp(
        _DEGRADATION_ACTIVATION_ENERGY_J_PER_MOL / _GAS_CONSTANT_J_PER_MOL_K
        * (1.0 / _DEGRADATION_REFERENCE_TEMPERATURE_K - 1.0 / temp_k)
    )
    return (
        float(np.sum(cure_rate) * hold_min / steps),
        float(np.sum(degradation_rate) * hold_min / steps),
    )


class ThermalCuringModelTests(unittest.TestCase):
    """Behavioral, interface, and domain checks."""

    def test_physical_trends_and_nonmonotonic_strength(self) -> None:
        thin = evaluate_model(140.0, 35.0, 0.10)
        thick = evaluate_model(140.0, 35.0, 0.50)
        self.assertLess(thick["peak_temperature_c"], thin["peak_temperature_c"])
        self.assertLess(thick["cure_fraction"], thin["cure_fraction"])

        short = evaluate_model(180.0, 15.0, 0.30)
        long = evaluate_model(180.0, 65.0, 0.30)
        self.assertGreater(long["cure_fraction"], short["cure_fraction"])
        self.assertGreater(long["degradation_fraction"], short["degradation_fraction"])
        middle = evaluate_model(180.0, 25.0, 0.30)
        self.assertGreater(middle["bond_strength_mpa"], long["bond_strength_mpa"])

    def test_refined_midpoint_integration_is_close(self) -> None:
        out = evaluate_model(165.0, 42.0, 0.37)
        cure_exposure, degradation_exposure = _reference_integrals(165.0, 42.0, 0.37, 384)
        expected_cure = -np.expm1(-cure_exposure)
        expected_degradation = -np.expm1(-degradation_exposure)
        self.assertAlmostEqual(out["cure_fraction"].item(), expected_cure, places=5)
        self.assertAlmostEqual(out["degradation_fraction"].item(), expected_degradation, places=5)

    def test_invalid_nan_inf_and_domains(self) -> None:
        valid = (140.0, 30.0, 0.25)
        for index, bad in enumerate((np.nan, np.inf, -np.inf, 99.999, 180.001)):
            values = list(valid)
            values[index % 3] = bad
            with self.subTest(index=index, bad=bad):
                with self.assertRaises(ValueError):
                    evaluate_model(*values)
        with self.assertRaises(ValueError):
            evaluate_model("not-a-number", 30.0, 0.25)

    def test_broadcast_and_scalar_array_outputs(self) -> None:
        out = evaluate_model(
            np.array([[120.0], [160.0]]),
            np.array([10.0, 30.0, 50.0]),
            0.25,
        )
        self.assertEqual({value.shape for value in out.values()}, {(2, 3)})
        scalar = evaluate_model(140.0, 30.0, 0.25)
        self.assertTrue(all(isinstance(value, np.ndarray) for value in scalar.values()))
        self.assertTrue(all(value.shape == () for value in scalar.values()))

    def test_full_grid_finite_and_has_feasible_and_infeasible_points(self) -> None:
        temperatures = np.linspace(100.0, 180.0, 9)
        holds = np.linspace(5.0, 65.0, 13)
        thicknesses = np.linspace(0.10, 0.50, 9)
        grid = np.meshgrid(temperatures, holds, thicknesses, indexing="ij")
        out = evaluate_model(*grid)
        self.assertEqual(out["bond_strength_mpa"].shape, (9, 13, 9))
        self.assertTrue(all(np.all(np.isfinite(value)) for value in out.values()))
        feasible = (out["cure_fraction"] >= 0.80) & (out["degradation_fraction"] <= 0.20)
        self.assertGreater(int(np.count_nonzero(feasible)), 0)
        self.assertGreater(int(np.count_nonzero(~feasible)), 0)

    def test_problem_metadata_and_grid_definition(self) -> None:
        with ROOT.joinpath("problem.csv").open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(
            rows[0].keys(),
            {"column", "display_name", "unit", "role", "direction", "lower", "upper", "step", "target"},
        )
        self.assertEqual(len({row["column"] for row in rows}), len(rows))
        parameters = [row for row in rows if row["role"] == "parameter"]
        self.assertEqual(len(parameters), 3)
        count = 1
        for row in parameters:
            lower, upper, step = (float(row[key]) for key in ("lower", "upper", "step"))
            self.assertTrue(np.isfinite([lower, upper, step]).all())
            self.assertLess(lower, upper)
            self.assertGreater(step, 0.0)
            count *= int(np.floor((upper - lower) / step + 1.0e-12)) + 1
        self.assertLessEqual(count, 6000)
        self.assertEqual(sum(row["role"] == "objective" for row in rows), 1)
        self.assertEqual(sum(row["role"] == "constraint" for row in rows), 2)
        for row in rows:
            if row["role"] in {"objective", "constraint", "monitor"}:
                self.assertEqual(row["lower"], "")
                self.assertEqual(row["upper"], "")
                self.assertEqual(row["step"], "")
        manifest = json.loads(ROOT.joinpath("manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["id"], "thermal_curing")
        self.assertIsInstance(manifest["reference_datasets"], list)

    def test_deterministic(self) -> None:
        args = (np.array([110.0, 150.0]), np.array([10.0, 40.0]), np.array([0.15, 0.35]))
        first = evaluate_model(*args)
        second = evaluate_model(*args)
        for key in first:
            np.testing.assert_array_equal(first[key], second[key])


if __name__ == "__main__":
    unittest.main()
