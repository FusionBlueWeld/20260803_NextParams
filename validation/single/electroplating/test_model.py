"""Unit tests for the high-dimensional deterministic electroplating model."""

from __future__ import annotations

import csv
import json
import sys
import unittest
from pathlib import Path

import numpy as np

try:
    from .physics_model import evaluate_model
except ImportError:  # Support ``python validation/single/electroplating/test_model.py``.
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from validation.single.electroplating.physics_model import evaluate_model


PARAMETER_COLUMNS = [
    "current_density_a_dm2",
    "bath_temperature_c",
    "plating_time_min",
    "copper_concentration_mol_l",
    "agitation_speed_m_s",
    "electrode_gap_cm",
    "duty_cycle",
    "bath_ph",
]


def _baseline() -> tuple[float, ...]:
    """Return a central, valid eight-input condition."""

    return (5.0, 35.0, 15.0, 1.0, 0.15, 1.0, 0.8, 2.0)


class ElectroplatingModelTests(unittest.TestCase):
    """Check scaling, physical trends, contract behavior, and grid structure."""

    def test_manifest_and_problem_define_high_dimension_contract(self) -> None:
        folder = Path(__file__).resolve().parent
        manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["model_version"], "2.0.0")
        self.assertTrue(manifest["high_dimension"])
        self.assertEqual(manifest["initial_design_size"], 64)

        with (folder / "problem.csv").open(encoding="utf-8-sig", newline="") as handle:
            raw_rows = list(csv.reader(handle))
        header = raw_rows[0]
        self.assertEqual(
            header,
            [
                "column", "display_name", "unit", "role", "direction",
                "lower", "upper", "step", "target",
            ],
        )
        for row_number, row in enumerate(raw_rows[1:], start=2):
            with self.subTest(row_number=row_number):
                self.assertEqual(len(row), len(header))

        with (folder / "problem.csv").open(encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
        for row_number, row in enumerate(rows, start=2):
            with self.subTest(parsed_row_number=row_number):
                self.assertNotIn(None, row)
        parameters = [row for row in rows if row["role"] == "parameter"]
        self.assertEqual([row["column"] for row in parameters], PARAMETER_COLUMNS)
        for row in parameters:
            lower = float(row["lower"])
            upper = float(row["upper"])
            step = float(row["step"])
            levels = np.arange(lower, upper + step * 0.01, step)
            self.assertEqual(len(levels), 4, row["column"])
            self.assertAlmostEqual(float(levels[-1]), upper)

    def test_faraday_time_and_duty_scaling(self) -> None:
        values = _baseline()
        short = evaluate_model(*values[:2], 5.0, *values[3:])
        long = evaluate_model(*values[:2], 15.0, *values[3:])
        np.testing.assert_allclose(
            long["deposit_thickness_um"],
            3.0 * short["deposit_thickness_um"],
            rtol=1e-12,
        )

        low_duty = evaluate_model(*values[:6], 0.4, values[7])
        high_duty = evaluate_model(*values[:6], 1.0, values[7])
        self.assertGreater(
            high_duty["deposit_thickness_um"], low_duty["deposit_thickness_um"]
        )

    def test_transport_concentration_and_agitation_trends(self) -> None:
        low_transport = evaluate_model(7.0, 35.0, 15.0, 0.4, 0.05, 1.0, 1.0, 2.0)
        high_transport = evaluate_model(7.0, 35.0, 15.0, 1.6, 0.35, 1.0, 1.0, 2.0)
        self.assertGreater(
            high_transport["current_efficiency"],
            low_transport["current_efficiency"],
        )
        self.assertGreater(
            high_transport["deposit_thickness_um"],
            low_transport["deposit_thickness_um"],
        )

    def test_gap_ph_temperature_and_current_voltage_trends(self) -> None:
        narrow_gap = evaluate_model(5.0, 35.0, 15.0, 1.0, 0.15, 0.5, 0.8, 1.5)
        wide_gap = evaluate_model(5.0, 35.0, 15.0, 1.0, 0.15, 2.0, 0.8, 3.0)
        self.assertGreater(wide_gap["cell_voltage_v"], narrow_gap["cell_voltage_v"])

        hot = evaluate_model(5.0, 50.0, 15.0, 1.0, 0.15, 1.0, 0.8, 2.0)
        cold = evaluate_model(5.0, 20.0, 15.0, 1.0, 0.15, 1.0, 0.8, 2.0)
        self.assertLess(hot["cell_voltage_v"], cold["cell_voltage_v"])

    def test_high_current_efficiency_and_roughness_penalty(self) -> None:
        low = evaluate_model(1.0, 35.0, 15.0, 1.0, 0.15, 1.0, 0.8, 2.0)
        high = evaluate_model(7.0, 35.0, 15.0, 0.4, 0.05, 1.0, 1.0, 2.0)
        self.assertLess(high["current_efficiency"], low["current_efficiency"])
        self.assertGreater(high["roughness_ra_um"], low["roughness_ra_um"])

    def test_broadcast_shape_and_determinism(self) -> None:
        first = evaluate_model(
            np.array([[1.0], [5.0]]),
            np.array([20.0, 35.0, 50.0]),
            np.array([[5.0, 15.0, 25.0]]),
            1.0,
            0.15,
            1.0,
            0.8,
            2.0,
        )
        second = evaluate_model(
            np.array([[1.0], [5.0]]),
            np.array([20.0, 35.0, 50.0]),
            np.array([[5.0, 15.0, 25.0]]),
            1.0,
            0.15,
            1.0,
            0.8,
            2.0,
        )
        for key, value in first.items():
            self.assertIsInstance(value, np.ndarray)
            self.assertEqual(value.shape, (2, 3))
            np.testing.assert_array_equal(value, second[key])

    def test_full_4_power_8_grid_is_finite_and_mixed_feasibility(self) -> None:
        axes = [
            np.linspace(1.0, 7.0, 4),
            np.linspace(20.0, 50.0, 4),
            np.linspace(5.0, 35.0, 4),
            np.linspace(0.4, 1.6, 4),
            np.linspace(0.05, 0.35, 4),
            np.linspace(0.5, 2.0, 4),
            np.linspace(0.4, 1.0, 4),
            np.linspace(1.5, 3.0, 4),
        ]
        grid = np.meshgrid(*axes, indexing="ij")
        outputs = evaluate_model(*grid)
        for values in outputs.values():
            self.assertEqual(values.shape, (4,) * 8)
            self.assertTrue(np.all(np.isfinite(values)))
        feasible = (
            (outputs["roughness_ra_um"] <= 0.65)
            & (outputs["current_efficiency"] >= 0.80)
        )
        self.assertEqual(feasible.size, 4**8)
        feasible_count = int(np.count_nonzero(feasible))
        self.assertGreater(feasible_count, 0)
        self.assertLess(feasible_count, feasible.size)
        self.assertGreater(feasible_count, feasible.size * 0.01)
        self.assertLess(feasible_count, feasible.size * 0.99)

    def test_invalid_nonfinite_and_out_of_range_inputs(self) -> None:
        lower = (1.0, 20.0, 5.0, 0.4, 0.05, 0.5, 0.4, 1.5)
        upper = (7.0, 50.0, 35.0, 1.6, 0.35, 2.0, 1.0, 3.0)
        for index in range(8):
            for label, source, delta in (
                ("below", lower, -1.0e-6),
                ("above", upper, 1.0e-6),
            ):
                values = list(source)
                values[index] += delta
                with self.subTest(index=index, label=label):
                    with self.assertRaises(ValueError):
                        evaluate_model(*values)
            for invalid in (np.nan, np.inf, -np.inf):
                values = list(_baseline())
                values[index] = invalid
                with self.subTest(index=index, invalid=invalid):
                    with self.assertRaises(ValueError):
                        evaluate_model(*values)


if __name__ == "__main__":
    unittest.main()
