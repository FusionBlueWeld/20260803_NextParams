"""Regression tests for data-supported parameter effect and temporary fixing."""

from types import SimpleNamespace
import unittest

import numpy as np

from v005.src.connection import StageConnection
from v005.src.focus_config import FocusConfig
from v005.src.parameter_focus import analyze_focus
from v005.src.settings import ProblemDefinition, VariableDefinition


def variable(name, role, lower=None, upper=None, step=None, direction="minimize", target=None):
    return VariableDefinition(name, name, "unit", role, direction, lower, upper, step, target)


def make_problem(parameter_names, constraints=()):
    parameters = [variable(name, "parameter", 0, 1, 0.5) for name in parameter_names]
    objective = variable("score", "objective", direction="minimize")
    result_variables = [objective, *constraints]
    problem = ProblemDefinition(parameters, objective, list(constraints), [], result_variables, 3 ** len(parameters))
    connection = StageConnection("focus-test", tuple(parameter_names), (), (), {})
    return problem, connection


def prepared(rows, **outputs):
    return SimpleNamespace(
        raw_parameters=np.asarray(rows, dtype=float),
        result_means={name: np.asarray(values, dtype=float) for name, values in outputs.items()},
        result_noise_std={},
        unique_condition_count=len(rows),
    )


def item_for(state, name):
    return next(item for item in state["parameters"] if item["column"] == name)


def test_constraint_only_effect_is_reported_even_when_objective_is_flat():
    constraint = variable("safe", "constraint", direction="less_equal", target=0.5)
    problem, connection = make_problem(["x", "bg"], [constraint])
    rows = [[x, bg] for bg in (0, 0.5, 1) for x in (0, 0.5, 1)]
    data = prepared(rows, score=[4] * len(rows), safe=[0.1 + 0.8 * x for x, bg in rows])

    state = analyze_focus(data, problem, connection, FocusConfig(parameter_focus_enabled=True))

    x = item_for(state, "x")
    assert set(x["effects"]) == {"score", "safe"}
    assert x["effects"]["score"] == 0
    assert x["effects"]["safe"] > 0.5
    assert state["fixed_values"] == {}


def test_quadratic_middle_optimum_is_not_mistaken_for_a_flat_parameter():
    problem, connection = make_problem(["x", "bg"])
    rows = [[x, bg] for bg in (0, 0.5, 1) for x in (0, 0.5, 1)]
    data = prepared(rows, score=[1 - 4 * (x - 0.5) ** 2 + 0.01 * bg for x, bg in rows])

    state = analyze_focus(data, problem, connection, FocusConfig(parameter_focus_enabled=True))

    x = item_for(state, "x")
    assert x["effects"]["score"] > 0.5
    assert x["fixed_value"] is None
    assert state["fixed_values"] == {}


def test_product_interaction_is_counted_for_each_parameter():
    problem, connection = make_problem(["x", "z"])
    rows = [[x, z] for x in (0, 0.5, 1) for z in (0, 0.5, 1)]
    data = prepared(rows, score=[(x - 0.5) * (z - 0.5) for x, z in rows])

    config = FocusConfig(parameter_focus_enabled=True, temporary_fix_enabled=True,
                         effect_tolerances={"score": 0.01}, stability_updates=1,
                         min_background_checks=1)
    state = analyze_focus(data, problem, connection, config)

    assert item_for(state, "x")["background_checks"] < 3
    assert item_for(state, "z")["background_checks"] < 3
    assert state["fixed_values"] == {}


def test_effects_use_only_the_current_incoming_context():
    context = variable("context", "parameter", 0, 1, 0.5)
    objective = variable("score", "objective", direction="minimize")
    problem = ProblemDefinition(
        [variable("x", "parameter", 0, 1, 0.5), variable("bg", "parameter", 0, 1, 0.5), context],
        objective, [], [], [objective], 9
    )
    connection = StageConnection("focus-test", ("x", "bg"), ("context",), (), {"context": 0.5})
    rows = [[x, bg, c] for c in (0, 0.5, 1) for bg in (0, 0.5, 1) for x in (0, 0.5, 1)]
    # The other incoming contexts contain a large x effect. At the active context,
    # score is flat, so using all upstream rows would produce a false effect.
    values = [100 * x if c != 0.5 else 5 for x, bg, c in rows]
    data = prepared(rows, score=values)

    state = analyze_focus(data, problem, connection, FocusConfig(parameter_focus_enabled=True))

    assert state["data_rows"] == 9
    assert item_for(state, "x")["effects"]["score"] == 0


def test_insufficient_data_keeps_effect_unknown_and_cannot_freeze():
    problem, connection = make_problem(["x", "z"])
    data = prepared([[0, 0], [1, 1]], score=[2, 2])
    config = FocusConfig(parameter_focus_enabled=True, temporary_fix_enabled=True,
                         effect_tolerances={"score": 0.01}, stability_updates=1)

    state = analyze_focus(data, problem, connection, config)

    for parameter in state["parameters"]:
        assert parameter["fixed_value"] is None
        assert parameter["state"] in {"UNTESTED", "UNDECIDED"}
    assert state["fixed_values"] == {}


def test_identical_dataset_reruns_do_not_advance_freeze_stability():
    problem, connection = make_problem(["x", "bg"])
    # Two backgrounds, each with x toggled. The output is practically invariant.
    rows = [[x, bg] for bg in (0, 0.5, 1) for x in (0, 0.5, 1)]
    data = prepared(rows, score=[10 + bg for x, bg in rows])
    config = FocusConfig(parameter_focus_enabled=True, temporary_fix_enabled=True,
                         effect_tolerances={"score": 2.0}, stability_updates=3,
                         min_background_checks=2)

    first = analyze_focus(data, problem, connection, config)
    second = analyze_focus(data, problem, connection, config, previous=first)

    assert second["data_fingerprint"] == first["data_fingerprint"]
    assert second["settings_fingerprint"] == first["settings_fingerprint"]
    assert item_for(second, "x")["stable_updates"] == item_for(first, "x")["stable_updates"]
    assert second["fixed_values"] == {}


def test_settings_change_resets_stability_history():
    problem, connection = make_problem(["x", "bg"])
    rows = [[x, bg] for bg in (0, 0.5, 1) for x in (0, 0.5, 1)]
    values = [10 + bg for x, bg in rows]
    data = prepared(rows, score=values)
    config = FocusConfig(parameter_focus_enabled=True, temporary_fix_enabled=True,
                         effect_tolerances={"score": 2.0}, stability_updates=3,
                         min_background_checks=2)
    prior = analyze_focus(data, problem, connection, config)

    changed_config = FocusConfig(parameter_focus_enabled=True, temporary_fix_enabled=True,
                                 effect_tolerances={"score": 1.9}, stability_updates=3,
                                 min_background_checks=2)
    after_settings_change = analyze_focus(data, problem, connection, changed_config, previous=prior)

    assert after_settings_change["settings_fingerprint"] != prior["settings_fingerprint"]
    assert item_for(after_settings_change, "x")["stable_updates"] == 1


def test_freeze_needs_every_output_tolerance_and_enough_paired_backgrounds():
    constraint = variable("safe", "constraint", direction="less_equal", target=2)
    problem, connection = make_problem(["x", "bg"], [constraint])
    rows = [[x, bg] for bg in (0, 0.5, 1) for x in (0, 0.5, 1)]
    data = prepared(rows, score=[10 + bg for x, bg in rows], safe=[1 + 0.1 * bg for x, bg in rows])
    missing_output_tolerance = FocusConfig(parameter_focus_enabled=True, temporary_fix_enabled=True,
        effect_tolerances={"score": 2.0}, stability_updates=1, min_background_checks=2)
    too_many_backgrounds_required = FocusConfig(parameter_focus_enabled=True, temporary_fix_enabled=True,
        effect_tolerances={"score": 2.0, "safe": 1.0}, stability_updates=1, min_background_checks=4)

    missing_state = analyze_focus(data, problem, connection, missing_output_tolerance)
    checked_state = analyze_focus(data, problem, connection, too_many_backgrounds_required)

    assert missing_state["fixed_values"] == {}
    assert checked_state["fixed_values"] == {}
    assert item_for(checked_state, "x")["background_checks"] == 3
    assert item_for(checked_state, "x")["state"] == "FIX_CANDIDATE"


def test_infeasible_data_cannot_provide_a_freeze_anchor():
    constraint = variable("safe", "constraint", direction="less_equal", target=0)
    problem, connection = make_problem(["x", "bg"], [constraint])
    rows = [[x, bg] for bg in (0, 0.5, 1) for x in (0, 0.5, 1)]
    data = prepared(rows, score=[10 + bg for x, bg in rows], safe=[1] * len(rows))
    config = FocusConfig(parameter_focus_enabled=True, temporary_fix_enabled=True,
        effect_tolerances={"score": 2.0, "safe": 2.0}, stability_updates=1, min_background_checks=2)

    state = analyze_focus(data, problem, connection, config)

    assert state["anchor"] is None
    assert state["fixed_values"] == {}


class ParameterFocusTests(unittest.TestCase):
    def test_constraint_only_effect(self):
        test_constraint_only_effect_is_reported_even_when_objective_is_flat()

    def test_quadratic_middle_optimum(self):
        test_quadratic_middle_optimum_is_not_mistaken_for_a_flat_parameter()

    def test_product_interaction(self):
        test_product_interaction_is_counted_for_each_parameter()

    def test_current_context_scope(self):
        test_effects_use_only_the_current_incoming_context()

    def test_small_data_is_undecided(self):
        test_insufficient_data_keeps_effect_unknown_and_cannot_freeze()

    def test_same_data_does_not_advance_stability(self):
        test_identical_dataset_reruns_do_not_advance_freeze_stability()

    def test_settings_change_resets_stability(self):
        test_settings_change_resets_stability_history()

    def test_freeze_requires_every_output_and_enough_backgrounds(self):
        test_freeze_needs_every_output_tolerance_and_enough_paired_backgrounds()

    def test_freeze_requires_feasible_anchor(self):
        test_infeasible_data_cannot_provide_a_freeze_anchor()

