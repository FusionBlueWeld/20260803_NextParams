from __future__ import annotations

import itertools
import unittest

import numpy as np

from validation.multistage.functional_coating.coating import physics_model as coating
from validation.multistage.functional_coating.curing import physics_model as curing
from validation.multistage.functional_coating.drying import physics_model as drying


class CoatingModelTests(unittest.TestCase):
    def nominal(self, **overrides):
        values = {
            "coating_gap_um": 210.0,
            "line_speed_m_min": 18.0,
            "web_tension_n": 100.0,
            "incoming_viscosity_pa_s": 1.6,
            "incoming_solids_fraction": 0.50,
            "incoming_bubble_fraction": 0.005,
        }
        values.update(overrides)
        return coating.evaluate_model(**values)

    def test_gap_controls_deposited_thickness_and_solids(self):
        result = self.nominal(coating_gap_um=np.array([140.0, 210.0, 280.0]))
        self.assertTrue(np.all(np.diff(result["wet_thickness_um"]) > 0))
        self.assertTrue(np.all(np.diff(result["wet_solids_g_m2"]) > 0))

    def test_bubbles_raise_defects_and_reduce_wet_thickness(self):
        result = self.nominal(incoming_bubble_fraction=np.array([0.0, 0.03]))
        self.assertLess(result["coating_defect_index"][0], result["coating_defect_index"][1])
        self.assertGreater(result["wet_thickness_um"][0], result["wet_thickness_um"][1])

    def test_broadcast_determinism_and_invalid_input(self):
        values = self.nominal(
            coating_gap_um=np.array([[180.0], [220.0]]),
            line_speed_m_min=np.array([10.0, 20.0, 25.0]),
        )
        repeated = self.nominal(
            coating_gap_um=np.array([[180.0], [220.0]]),
            line_speed_m_min=np.array([10.0, 20.0, 25.0]),
        )
        for name in values:
            self.assertEqual(values[name].shape, (2, 3))
            np.testing.assert_array_equal(values[name], repeated[name])
        with self.assertRaises(ValueError):
            self.nominal(coating_gap_um=np.nan)
        with self.assertRaises(ValueError):
            self.nominal(line_speed_m_min=31.0)

    def test_all_domain_corners_are_finite_and_physical(self):
        names = list(coating.INPUT_BOUNDS)
        points = np.asarray(list(itertools.product(*(coating.INPUT_BOUNDS[n] for n in names))))
        result = coating.evaluate_model(**dict(zip(names, points.T)))
        for values in result.values():
            self.assertTrue(np.all(np.isfinite(values)))
        self.assertTrue(np.all((result["coating_defect_index"] >= 0) & (result["coating_defect_index"] <= 1)))


class DryingModelTests(unittest.TestCase):
    def nominal(self, **overrides):
        values = {
            "air_temperature_c": 90.0,
            "air_speed_m_s": 3.0,
            "residence_time_min": 8.0,
            "incoming_wet_thickness_um": 155.0,
            "incoming_solids_fraction": 0.50,
            "incoming_thickness_cv_fraction": 0.025,
            "incoming_coating_defect_index": 0.10,
        }
        values.update(overrides)
        return drying.evaluate_model(**values)

    def test_temperature_time_and_thickness_trends(self):
        thermal = self.nominal(
            air_temperature_c=np.array([65.0, 90.0, 115.0]),
            residence_time_min=np.array([4.0, 8.0, 12.0]),
        )
        self.assertTrue(np.all(np.diff(thermal["residual_solvent_pct"]) < 0))
        thick = self.nominal(incoming_wet_thickness_um=np.array([90.0, 220.0]))
        self.assertLess(thick["residual_solvent_pct"][0], thick["residual_solvent_pct"][1])

    def test_mass_and_energy_are_positive(self):
        result = self.nominal()
        self.assertGreater(float(result["dry_thickness_um"]), 0)
        self.assertLessEqual(float(result["dry_thickness_um"]), 155.0)
        self.assertGreater(float(result["drying_energy_kj_m2"]), 0)
        self.assertGreaterEqual(float(result["residual_solvent_pct"]), 0)

    def test_broadcast_determinism_and_domain_validation(self):
        first = self.nominal(air_speed_m_s=np.array([1.0, 3.0, 5.0]))
        second = self.nominal(air_speed_m_s=np.array([1.0, 3.0, 5.0]))
        for name in first:
            np.testing.assert_array_equal(first[name], second[name])
        with self.assertRaises(ValueError):
            self.nominal(incoming_wet_thickness_um=55.0)
        with self.assertRaises(ValueError):
            self.nominal(incoming_solids_fraction=np.inf)

    def test_all_domain_corners_are_finite_and_bounded(self):
        names = list(drying.INPUT_BOUNDS)
        points = np.asarray(list(itertools.product(*(drying.INPUT_BOUNDS[n] for n in names))))
        result = drying.evaluate_model(**dict(zip(names, points.T)))
        for values in result.values():
            self.assertTrue(np.all(np.isfinite(values)))
        self.assertTrue(np.all(result["dry_thickness_um"] <= points[:, names.index("incoming_wet_thickness_um")]))
        self.assertTrue(np.all((result["drying_defect_index"] >= 0) & (result["drying_defect_index"] <= 1)))


class CuringModelTests(unittest.TestCase):
    def nominal(self, **overrides):
        values = {
            "oven_temperature_c": 140.0,
            "hold_time_min": 45.0,
            "nip_pressure_mpa": 0.30,
            "incoming_dry_thickness_um": 68.0,
            "incoming_residual_solvent_pct": 2.5,
            "incoming_internal_stress_mpa": 0.6,
            "incoming_drying_defect_index": 0.10,
            "incoming_thickness_cv_fraction": 0.035,
        }
        values.update(overrides)
        return curing.evaluate_model(**values)

    def test_cure_and_degradation_follow_thermal_exposure(self):
        result = self.nominal(oven_temperature_c=np.array([110.0, 140.0, 175.0]))
        self.assertTrue(np.all(np.diff(result["cure_fraction"]) > 0))
        self.assertTrue(np.all(np.diff(result["degradation_fraction"]) > 0))

    def test_upstream_solvent_and_defect_propagate(self):
        solvent = self.nominal(incoming_residual_solvent_pct=np.array([0.5, 15.0]))
        self.assertLess(solvent["blister_index"][0], solvent["blister_index"][1])
        self.assertGreater(solvent["bond_strength_mpa"][0], solvent["bond_strength_mpa"][1])
        defect = self.nominal(incoming_drying_defect_index=np.array([0.0, 0.7]))
        self.assertLess(defect["final_defect_index"][0], defect["final_defect_index"][1])
        self.assertTrue(np.all(defect["final_thickness_um"] <= 68.0))

    def test_broadcast_determinism_and_domain_validation(self):
        first = self.nominal(hold_time_min=np.array([[20.0], [40.0]]), nip_pressure_mpa=np.array([0.1, 0.3]))
        second = self.nominal(hold_time_min=np.array([[20.0], [40.0]]), nip_pressure_mpa=np.array([0.1, 0.3]))
        for name in first:
            self.assertEqual(first[name].shape, (2, 2))
            np.testing.assert_array_equal(first[name], second[name])
        with self.assertRaises(ValueError):
            self.nominal(oven_temperature_c=181.0)
        with self.assertRaises(ValueError):
            self.nominal(incoming_internal_stress_mpa=np.nan)

    def test_all_domain_corners_are_finite_and_bounded(self):
        names = list(curing.INPUT_BOUNDS)
        points = np.asarray(list(itertools.product(*(curing.INPUT_BOUNDS[n] for n in names))))
        result = curing.evaluate_model(**dict(zip(names, points.T)))
        for values in result.values():
            self.assertTrue(np.all(np.isfinite(values)))
        for name in ("cure_fraction", "degradation_fraction", "final_defect_index", "blister_index"):
            self.assertTrue(np.all((result[name] >= 0) & (result[name] <= 1)))
        self.assertTrue(np.all(result["final_thickness_um"] > 0))


if __name__ == "__main__":
    unittest.main()
