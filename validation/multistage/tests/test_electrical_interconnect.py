"""合成oracleの物理契約と工程間因果を検証します。実設備の校正試験ではありません。"""

from __future__ import annotations

import itertools
import math
import unittest

import numpy as np

from validation.multistage.electrical_interconnect.inspection import observe_stage, inspect_final
from validation.multistage.electrical_interconnect.pipeline import (
    DEFAULT_CONTROLS, FINAL_SPECIFICATIONS, STAGE_ORDER, VARIANTS, axes, evaluate_line, line_config,
)
from validation.multistage.electrical_interconnect.stages import (
    STAGE_MANIFESTS, default_inputs, evaluate_stage, incoming_proxy,
)


class ElectricalInterconnectTests(unittest.TestCase):
    def stage(self, sid, **updates):
        values = default_inputs(sid)
        values.update(updates)
        return evaluate_stage(sid, values)

    def test_manifest_is_six_node_dag_with_explosive_unexpanded_axes(self):
        self.assertEqual(set(STAGE_ORDER), set(STAGE_MANIFESTS))
        self.assertEqual(len(axes()), 44)
        self.assertEqual(math.prod(len(v) for v in axes().values()), 4 ** 44)
        config = line_config()
        rank = {sid: i for i, sid in enumerate(STAGE_ORDER)}
        for connection in config["connections"]:
            source, source_output = connection["from"].split(".", 1)
            target, target_input = connection["to"].split(".", 1)
            self.assertLess(rank[source], rank[target])
            self.assertIn(source_output, STAGE_MANIFESTS[source]["predicted_outputs"])
            self.assertIn(target_input, STAGE_MANIFESTS[target]["incoming_context"])
        for manifest in STAGE_MANIFESTS.values():
            self.assertIn(len(manifest["controls"]), range(6, 9))
            self.assertEqual(set(manifest["predicted_outputs"]), set(manifest["output_units"]))

    def test_nominal_complete_line_is_feasible(self):
        result = evaluate_line({})
        self.assertTrue(result["final"]["line_feasible"])
        self.assertGreater(float(result["final"]["quality_margin"]), 0.0)

    def test_broadcast_over_whole_line_and_determinism(self):
        controls = {"cut_cu.cut_speed_m_min": np.array([[60.0], [130.0]]),
                    "welding.laser_power_kw": np.array([1.7, 2.4, 3.1])}
        first, second = evaluate_line(controls), evaluate_line(controls)
        for sid, outputs in first["stage_outputs"].items():
            for name, values in outputs.items():
                self.assertEqual(values.shape, (2, 3))
                np.testing.assert_array_equal(values, second["stage_outputs"][sid][name])

    def test_all_control_corners_per_stage_are_finite(self):
        for sid, manifest in STAGE_MANIFESTS.items():
            names = manifest["controls"]
            # 最大2^8点だけ。全ライン44次元の直積は作りません。
            corners = np.asarray(list(itertools.product(*(manifest["input_bounds"][n] for n in names))))
            inputs = default_inputs(sid)
            inputs.update(dict(zip(names, corners.T)))
            outputs = evaluate_stage(sid, inputs)
            for name, values in outputs.items():
                self.assertTrue(np.all(np.isfinite(values)), (sid, name))
                if manifest["output_units"][name] == "fraction":
                    self.assertTrue(np.all((values >= 0.0) & (values <= 1.0)), (sid, name))

    def test_random_lines_include_good_and_bad_and_all_variants(self):
        rng = np.random.default_rng(71)
        controls = {name: rng.uniform(v[0], v[-1], 2048) for name, v in axes().items()}
        for variant in VARIANTS:
            result = evaluate_line(controls, variant=variant)
            self.assertTrue(np.any(result["final"]["feasible"]))
            self.assertTrue(np.any(~result["final"]["feasible"]))
            for outputs in result["stage_outputs"].values():
                self.assertTrue(all(np.all(np.isfinite(v)) for v in outputs.values()))

    def test_material_batch_states_propagate_to_geometry_and_electrics(self):
        result = evaluate_line({}, material_context={"cut_al.stock_thickness_mm": np.array([0.9, 1.1]),
            "cut_al.stock_camber_mm": np.array([-0.2, 0.2])})
        self.assertNotEqual(*result["stage_outputs"]["form_al"]["thickness_mm"])
        self.assertNotEqual(*result["final"]["base_path_resistance_uohm"])

    def test_validation_rejects_invalid_inputs_and_material_category(self):
        for bad in (np.nan, np.inf, 39.9):
            with self.assertRaises(ValueError):
                self.stage("cut_cu", cut_speed_m_min=bad)
        with self.assertRaises(ValueError):
            self.stage("cut_cu", material_is_cu=0.5)
        with self.assertRaises(ValueError):
            evaluate_stage("cut_cu", {})
        with self.assertRaises(ValueError):
            self.stage("cut_cu", unexpected=3.0)
        with self.assertRaises(ValueError):
            evaluate_line({"gap": 0.0})
        with self.assertRaises(ValueError):
            evaluate_line({}, material_context={"cut_al.material_is_cu": 1.0})
        with self.assertRaises(ValueError):
            axes(1)

    def test_finish_removes_cut_damage_and_burr(self):
        output = self.stage("cut_al", finish_allowance_mm=np.array([0.05, 0.6]), deburr_depth_mm=0.0)
        self.assertGreater(*output["edge_damage_fraction"])
        self.assertGreater(*output["burr_mm"])

    def test_edge_damage_is_not_erased_by_forming(self):
        output = self.stage("form_al", edge_damage_fraction=np.array([0.02, 0.3]))
        self.assertLess(*output["form_damage_fraction"])
        self.assertTrue(np.all(output["form_damage_fraction"] >= np.array([0.02, 0.3])))

    def test_elastic_recovery_and_radius_control_springback_and_thinning(self):
        output = self.stage("form_al", elastic_modulus_gpa=np.array([69.0, 118.0]))
        self.assertLess(*output["first_angle_error_deg"])
        output = self.stage("form_al", punch_radius_mm=np.array([1.0, 4.0]), die_opening_mm=8.0)
        self.assertGreater(*output["thickness_loss_fraction"])
        self.assertGreater(*output["first_angle_error_deg"])

    def test_gap_is_derived_and_clamps_have_irreversible_tradeoff(self):
        output = self.stage("assembly", top_pad_height_error_mm=0.5,
            bottom_clamp_force_n=np.array([200.0, 1600.0]), top_clamp_force_n=np.array([200.0, 1600.0]))
        self.assertGreater(*output["gap_center_mm"])
        self.assertLess(*output["assembly_plastic_set_mm"])
        self.assertNotIn("gap_center_mm", STAGE_MANIFESTS["assembly"]["controls"])

    def test_upstream_pose_change_reaches_weld_and_final_quality(self):
        result = evaluate_line({"form_al.first_overbend_deg": np.array([0.65, 5.0])})
        self.assertLess(*result["stage_outputs"]["assembly"]["gap_max_mm"])
        self.assertGreater(*result["final"]["fusion_completion_fraction"])

    def test_gap_reduces_bonding_and_increases_electrical_resistance(self):
        output = self.stage("welding", gap_center_mm=np.array([0.0, 0.25]),
                            gap_start_mm=0.0, gap_end_mm=0.0)
        self.assertGreater(*output["bonded_area_mm2"])
        self.assertLess(*output["cold_unclamped_resistance_uohm"])
        self.assertGreater(*output["shear_force_n"])

    def test_laser_intensity_matters_beyond_power_over_speed(self):
        output = self.stage("welding", focus_offset_mm=np.array([0.0, 2.0]))
        self.assertEqual(*output["line_energy_j_mm"])
        self.assertGreater(*output["peak_intensity_kw_mm2"])
        self.assertNotEqual(*output["absorbed_energy_j_mm"])

    def test_constant_gap_field_does_not_change_with_beam_offset(self):
        output = self.stage("welding", seam_offset_mm=np.array([-3.0, 0.0, 3.0]),
            gap_start_mm=0.06, gap_center_mm=0.06, gap_end_mm=0.06)
        np.testing.assert_allclose(output["bonded_area_mm2"], output["bonded_area_mm2"][0])

    def test_imc_only_exists_for_dissimilar_metal_and_r_includes_base_path(self):
        output = self.stage("welding", top_material_is_cu=0.0, bottom_material_is_cu=np.array([0.0, 1.0]))
        self.assertEqual(output["imc_proxy_um"][0], 0.0)
        self.assertGreater(output["imc_proxy_um"][1], 0.0)
        np.testing.assert_allclose(output["cold_unclamped_resistance_uohm"],
            output["base_path_resistance_uohm"] + output["interface_resistance_uohm"])
        self.assertTrue(np.all(output["cold_unclamped_resistance_uohm"] >= output["base_path_resistance_uohm"]))

    def test_strength_cannot_exceed_damaged_parent_net_section(self):
        inputs = default_inputs("welding")
        inputs["top_damage_fraction"] = np.array([0.0, 0.95])
        output = evaluate_stage("welding", inputs)
        limit = inputs["top_yield_strength_mpa"] * 40.0 * inputs["top_thickness_mm"] * (1.0 - inputs["top_damage_fraction"])
        self.assertTrue(np.all(output["shear_force_n"] <= limit))

    def test_observation_whitelist_noise_and_destructive_consumption(self):
        truth = evaluate_line({})["final"]
        normal = inspect_final(truth, noise=False)
        self.assertNotIn("imc_proxy_um", normal["measurements"])
        self.assertNotIn("shear_force_n", normal["measurements"])
        self.assertNotIn("mixing_fraction", normal["measurements"])
        self.assertFalse(normal["consumed"])
        self.assertEqual(normal["measurements"]["cold_unclamped_resistance_uohm"], truth["cold_unclamped_resistance_uohm"])
        destructive = inspect_final(truth, destructive=True, noise=False)
        self.assertTrue(destructive["consumed"])
        self.assertFalse(destructive["available_for_next_stage"])
        self.assertIn("shear_force_n", destructive["measurements"])
        self.assertNotIn("mixing_fraction", destructive["measurements"])
        a = inspect_final(truth, rng=np.random.default_rng(6))
        b = inspect_final(truth, rng=np.random.default_rng(6))
        for name in a["measurements"]:
            np.testing.assert_array_equal(a["measurements"][name], b["measurements"][name])

    def test_proxy_regime_removes_prefixed_latent_truth(self):
        original = default_inputs("assembly")
        original["bottom_residual_stress_mpa"] = np.array([5.0, 250.0])
        original["top_oxide_nm"] = np.array([1.0, 70.0])
        proxy = incoming_proxy("assembly", original)
        np.testing.assert_array_equal(proxy["bottom_residual_stress_mpa"], [35.0, 35.0])
        np.testing.assert_array_equal(proxy["top_oxide_nm"], [8.0, 8.0])
        np.testing.assert_array_equal(incoming_proxy("assembly", original, "complete")["top_oxide_nm"], [1.0, 70.0])


if __name__ == "__main__":
    unittest.main()
