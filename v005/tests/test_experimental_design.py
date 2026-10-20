import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from v005.src.candidate_identity import CandidateIdentity
from v005.src.connection import StageConnection
from v005.src.experimental_design import design_diagnostics, design_target, recommend_design
from v005.src.focus_config import FocusConfig, load_focus_config
from v005.src.policies.regions import RegionCondition, RegionPolicy, RegionRule
from v005.src.search_config import SearchConfig
from v005.src.settings import ProblemDefinition, VariableDefinition


def variable(name, role, lower=None, upper=None, step=None, direction="minimize", unit=""):
    return VariableDefinition(name, name, unit, role, direction, lower, upper, step, None)


def problem(two_controls=False, *, levels=4, unit=""):
    parameters = [variable("a", "parameter", 0, levels - 1, 1, unit=unit)]
    if two_controls:
        parameters.append(variable("b", "parameter", 0, 3, 1))
    parameters.append(variable("context", "parameter", 0, 2, 1))
    objective = variable("score", "objective")
    return ProblemDefinition(parameters, objective, [], [], [objective], 1)


def connection(two_controls=False):
    controls = ("a", "b") if two_controls else ("a",)
    return StageConnection("stage", controls, ("context",), (), {"context": 1.25})


class ExperimentalDesignTests(unittest.TestCase):
    def test_missing_config_is_off_and_target_uses_controls_only(self):
        with tempfile.TemporaryDirectory() as directory:
            config = load_focus_config(Path(directory) / "missing.json")
        self.assertEqual(config, FocusConfig())
        self.assertEqual(design_target(problem(two_controls=True), connection(two_controls=True), config), 12)

    def test_diagnostics_deduplicate_and_scope_to_current_context(self):
        p = problem()
        rows = np.array([[0, 1.25], [0, 1.25], [1, 1.25], [2, 1.25], [3, 1.25], [3, 1.0]])
        report = design_diagnostics(rows, p, connection())
        self.assertEqual(report["current_context_unique_conditions"], 4)
        self.assertEqual(report["level_coverage"]["a"]["observed_levels"], 4)
        self.assertEqual(report["design_rank"], 3)
        self.assertFalse(report["design_ready"])

    def test_two_level_numeric_variable_has_linear_rank_only(self):
        p = problem(levels=2)
        report = design_diagnostics([[0, 1.25], [1, 1.25]] * 6, p, connection())
        self.assertEqual(report["design_rank"], 2)
        self.assertEqual(report["target_rank"], 2)
        self.assertFalse(report["design_ready"])

    def test_categorical_variable_uses_level_indicators(self):
        p = problem(unit="category")
        report = design_diagnostics([[0, 1.25], [1, 1.25], [2, 1.25], [3, 1.25]] * 3,
                                    p, connection())
        self.assertEqual(report["design_rank"], 4)
        self.assertEqual(report["target_rank"], 4)
        self.assertFalse(report["design_ready"])

    def test_canonicalization_precedes_unique_condition_count(self):
        p = ProblemDefinition([variable("a", "parameter", 0.1, 0.5, 0.2),
                               variable("context", "parameter", 0, 2, 1)],
                              variable("score", "objective"), [], [], [variable("score", "objective")], 1)
        c = connection()
        report = design_diagnostics([[0.3, 1.25], [0.30000000000000004, 1.25]], p, c)
        self.assertEqual(report["current_context_unique_conditions"], 1)

    def test_recommendations_exclude_measured_and_forbidden_levels(self):
        p = problem()
        c = connection()
        measured = np.array([[0, 1.25], [1, 1.25]])
        forbidden = RegionPolicy((RegionRule("avoid-high", "forbidden", (RegionCondition("a", lower=3),)),))
        chosen = recommend_design(p, c, forbidden, measured, 4, FocusConfig(max_screening_samples=16),
                                  SearchConfig(max_candidates=64, seed=11))
        self.assertEqual(chosen.shape, (1, 2))
        np.testing.assert_array_equal(chosen, [[2, 1.25]])
        identity = CandidateIdentity(p, c.current_incoming_context)
        self.assertFalse(identity.measured_mask(chosen, measured).any())

    def test_recommendations_repeat_deterministically_and_respect_screening_cap(self):
        p = problem(two_controls=True)
        c = connection(two_controls=True)
        config = FocusConfig(max_screening_samples=7)
        original = __import__("v005.src.experimental_design", fromlist=["make_parameter_grid"]).make_parameter_grid
        seen = {}

        def bounded(*args, **kwargs):
            seen.update(kwargs)
            return original(*args, **kwargs)

        with patch("v005.src.experimental_design.make_parameter_grid", bounded):
            first = recommend_design(p, c, RegionPolicy(), np.empty((0, 3)), 5, config, SearchConfig(seed=4))
            second = recommend_design(p, c, RegionPolicy(), np.empty((0, 3)), 5, config, SearchConfig(seed=4))
        self.assertEqual(seen["max_candidates"], 7)
        np.testing.assert_array_equal(first, second)
        self.assertEqual(first.shape, (5, 3))
        self.assertTrue(np.all(first[:, 2] == 1.25))

    def test_invalid_settings_are_rejected(self):
        invalid = [
            {"unknown": 1},
            {"initial_design_enabled": 1},
            {"initial_min_conditions": True},
            {"initial_min_conditions": 49, "initial_max_conditions": 48},
            {"temporary_fix_enabled": True, "parameter_focus_enabled": False},
            {"effect_tolerances": {"context": 0.1}},
            {"never_fix": ["context"]},
            {"effect_tolerances": {"score": "NaN"}},
            {"resamples": 1001},
            {"max_screening_samples": 10001},
            {"min_background_checks": 1001},
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "focus_settings.json"
            for payload in invalid:
                with self.subTest(payload=payload):
                    path.write_text(json.dumps(payload), encoding="utf-8")
                    with self.assertRaises(ValueError):
                        load_focus_config(path, problem(), connection())

    def test_valid_output_tolerance_and_control_protection(self):
        payload = {"parameter_focus_enabled": True, "temporary_fix_enabled": True,
                   "effect_tolerances": {"score": 0.02}, "never_fix": ["a"]}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "focus_settings.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            loaded = load_focus_config(path, problem(), connection())
        self.assertEqual(loaded.effect_tolerances, {"score": 0.02})
        self.assertEqual(loaded.never_fix, ("a",))

    def test_default_flags_are_on_and_explicit_off_is_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "focus_settings.json"
            for payload in (None, {}):
                if payload is not None:
                    path.write_text(json.dumps(payload), encoding="utf-8")
                config = load_focus_config(path)
                self.assertTrue(config.initial_design_enabled)
                self.assertTrue(config.parameter_focus_enabled)
                self.assertTrue(config.temporary_fix_enabled)
            path.write_text(json.dumps({
                "initial_design_enabled": False, "parameter_focus_enabled": False,
                "temporary_fix_enabled": False,
            }), encoding="utf-8")
            config = load_focus_config(path)
            self.assertFalse(config.initial_design_enabled)
            self.assertFalse(config.parameter_focus_enabled)
            self.assertFalse(config.temporary_fix_enabled)


if __name__ == "__main__":
    unittest.main()
