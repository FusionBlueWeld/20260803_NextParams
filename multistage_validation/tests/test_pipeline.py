from __future__ import annotations

import json
import unittest
from pathlib import Path

import numpy as np

from multistage_validation.functional_coating.benchmark import audit, full_line_grid
from multistage_validation.functional_coating.fault_scenarios import evaluate_fault_scenarios
from multistage_validation.functional_coating.pipeline import (
    DEFAULT_MATERIAL_STATE,
    FINAL_SPECIFICATIONS,
    LINE_INPUT_BOUNDS,
    evaluate_line,
)
from multistage_validation.functional_coating.robustness import (
    estimate_robustness,
    sample_conditions,
)
from multistage_validation.stages import available, load


def nominal_line(**overrides):
    values = {
        **DEFAULT_MATERIAL_STATE,
        "coating_gap_um": 210.0,
        "line_speed_m_min": 18.0,
        "web_tension_n": 100.0,
        "air_temperature_c": 90.0,
        "air_speed_m_s": 5.5,
        "residence_time_min": 13.0,
        "oven_temperature_c": 140.0,
        "hold_time_min": 60.0,
        "nip_pressure_mpa": 0.325,
    }
    values.update(overrides)
    return values


class PipelineTests(unittest.TestCase):
    def test_stages_are_independently_registered(self):
        self.assertEqual(available(), ["coating", "curing", "drying"])
        for stage_id in available():
            stage = load(stage_id)
            self.assertTrue(stage.control_names)
            self.assertTrue(stage.incoming_state_names)
            self.assertFalse(set(stage.control_names) & set(stage.incoming_state_names))

    def test_pipeline_handoffs_are_exact(self):
        result = evaluate_line(nominal_line())
        coating = result["coating"]
        drying_stage = load("drying")
        direct_drying = drying_stage.evaluate({
            "air_temperature_c": 90.0,
            "air_speed_m_s": 5.5,
            "residence_time_min": 13.0,
            "incoming_wet_thickness_um": coating["wet_thickness_um"],
            "incoming_solids_fraction": coating["solids_fraction"],
            "incoming_thickness_cv_fraction": coating["thickness_cv_fraction"],
            "incoming_coating_defect_index": coating["coating_defect_index"],
        })
        for name in direct_drying:
            np.testing.assert_array_equal(result["drying"][name], direct_drying[name])

    def test_line_is_deterministic_and_broadcasts_globally(self):
        values = nominal_line(line_speed_m_min=np.array([12.0, 18.0, 24.0]))
        first = evaluate_line(values)
        second = evaluate_line(values)
        for stage in ("coating", "drying", "curing", "final"):
            for name in first[stage]:
                self.assertEqual(np.asarray(first[stage][name]).shape, (3,))
                np.testing.assert_array_equal(first[stage][name], second[stage][name])

    def test_exact_line_contract_and_bounds(self):
        missing = nominal_line()
        del missing["air_speed_m_s"]
        with self.assertRaises(ValueError):
            evaluate_line(missing)
        extra = {**nominal_line(), "unknown": 1.0}
        with self.assertRaises(ValueError):
            evaluate_line(extra)
        self.assertEqual(set(nominal_line()), set(LINE_INPUT_BOUNDS))

    def test_reference_grid_contains_both_final_outcomes(self):
        points, _, result = full_line_grid()
        feasible = result["final"]["feasible"]
        self.assertEqual(len(points), 3**9)
        self.assertTrue(np.any(feasible))
        self.assertTrue(np.any(~feasible))
        self.assertGreater(float(np.max(result["final"]["quality_margin"])), 0)
        self.assertLess(float(np.min(result["final"]["quality_margin"])), 0)

    def test_local_bests_can_fail_while_global_coordination_is_robust(self):
        report = audit(robustness_samples=128, seed=7)
        local = report["individual_best_chain"]
        coordinated = report["global_max_quality_margin"]
        self.assertFalse(local["final"]["feasible"])
        self.assertTrue(coordinated["final"]["feasible"])
        self.assertGreater(
            coordinated["robustness"]["yield_probability"],
            local["robustness"]["yield_probability"],
        )

    def test_variation_is_seeded_and_does_not_change_oracle(self):
        nominal = nominal_line()
        first = sample_conditions(nominal, samples=64, seed=11)
        second = sample_conditions(nominal, samples=64, seed=11)
        for name in first:
            np.testing.assert_array_equal(first[name], second[name])
        r1 = estimate_robustness(nominal, samples=64, seed=11)
        r2 = estimate_robustness(nominal, samples=64, seed=11)
        self.assertEqual(r1, r2)
        exact1 = evaluate_line(nominal)
        exact2 = evaluate_line(nominal)
        for name in exact1["final"]:
            np.testing.assert_array_equal(exact1["final"][name], exact2["final"][name])

    def test_known_source_drifts_create_root_cause_benchmarks(self):
        scenarios = evaluate_fault_scenarios()
        self.assertTrue(scenarios["reference"]["final"]["feasible"])
        for name, case in scenarios.items():
            if name == "reference":
                continue
            self.assertTrue(case["source_stages"])
            self.assertFalse(case["final"]["feasible"], name)

    def test_manifests_match_code_input_contracts(self):
        root = Path(__file__).resolve().parents[1] / "functional_coating"
        line_result = evaluate_line(nominal_line())
        for stage_id in available():
            manifest = json.loads((root / stage_id / "manifest.json").read_text(encoding="utf-8"))
            declared = {
                item["name"]: tuple(item["range"])
                for role in ("controls", "incoming_state")
                for item in manifest[role]
            }
            self.assertEqual(declared, load(stage_id).input_bounds)
            output_ranges = {item["name"]: item["range"] for item in manifest["outputs"]}
            self.assertEqual(set(output_ranges), set(line_result[stage_id]))
            for name, (lower, upper) in output_ranges.items():
                value = float(line_result[stage_id][name])
                self.assertGreaterEqual(value, lower)
                self.assertLessEqual(value, upper)

    def test_process_manifest_matches_connections_and_final_specs(self):
        root = Path(__file__).resolve().parents[1] / "functional_coating"
        manifest = json.loads((root / "process_manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["stages"], ["coating", "drying", "curing"])
        self.assertEqual(manifest["reference_grid"]["candidate_count"], 3**9)
        declared_specs = {
            item["name"]: (item["direction"], item["target"])
            for item in manifest["final_specifications"]
        }
        code_specs = {
            name: (direction, list(target) if isinstance(target, tuple) else target)
            for name, (direction, target) in FINAL_SPECIFICATIONS.items()
        }
        self.assertEqual(declared_specs, code_specs)
        outputs = evaluate_line(nominal_line())
        for connection in manifest["connections"]:
            source_stage, source_name = connection["from"].split(".")
            target_stage, target_name = connection["to"].split(".")
            self.assertIn(source_name, outputs[source_stage])
            self.assertIn(target_name, load(target_stage).incoming_state_names)


if __name__ == "__main__":
    unittest.main()
