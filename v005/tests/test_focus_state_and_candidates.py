"""固定の成立・解除、候補予算、全空間との区別を検証する。"""
from dataclasses import replace
from types import SimpleNamespace
import unittest

import numpy as np

from v005.tests.test_parameter_focus import make_problem, prepared, item_for
from v005.src.focus_config import FocusConfig
from v005.src.parameter_focus import analyze_focus, focus_candidate_plan, check_hybrid_effects
from v005.src.search_config import SearchConfig
from v005.src.parameter_space import make_parameter_grid
from v005.src.policies import RegionPolicy


class FocusStateAndCandidateTests(unittest.TestCase):
    def setUp(self):
        self.problem, self.connection = make_problem(['x', 'bg'])
        self.rows = np.array([[x, bg] for bg in [0, .5, 1] for x in [0, .5, 1]])
        self.data = prepared(self.rows, score=10 + self.rows[:, 1])
        self.config = FocusConfig(parameter_focus_enabled=True, temporary_fix_enabled=True,
            effect_tolerances={'score': .1}, never_fix=('bg',), stability_updates=1,
            min_background_checks=2)

    def test_verified_flat_variable_is_fixed_and_important_variable_stays_active(self):
        state = analyze_focus(self.data, self.problem, self.connection, self.config)
        self.assertEqual(item_for(state, 'x')['state'], 'TEMPORARILY_FIXED')
        self.assertEqual(state['fixed_values'], {'x': 0.})
        self.assertNotIn('bg', state['fixed_values'])

    def test_weights_only_switch_never_fixes_inputs(self):
        config = replace(self.config, temporary_fix_enabled=False)
        state = analyze_focus(self.data, self.problem, self.connection, config)
        self.assertEqual(item_for(state, 'x')['state'], 'FIX_CANDIDATE')
        self.assertEqual(state['fixed_values'], {})

    def test_newly_observed_effect_releases_fixed_variable(self):
        before = analyze_focus(self.data, self.problem, self.connection, self.config)
        changed = prepared(self.rows, score=10 + self.rows[:, 1] + 3 * self.rows[:, 0])
        after = analyze_focus(changed, self.problem, self.connection, self.config, before)
        self.assertNotIn('x', after['fixed_values'])
        self.assertEqual(item_for(after, 'x')['state'], 'ACTIVE')

    def test_measurement_correction_resets_stability_instead_of_adding_evidence(self):
        config = replace(self.config, stability_updates=3)
        before = analyze_focus(self.data, self.problem, self.connection, config)
        item_for(before, 'x')['stable_updates'] = 2
        values = 10 + self.rows[:, 1]
        values[0] += .001
        corrected = prepared(self.rows, score=values)
        after = analyze_focus(corrected, self.problem, self.connection, config, before)
        self.assertEqual(item_for(after, 'x')['stable_updates'], 1)
        self.assertEqual(after['fixed_values'], {})

    def test_focused_pool_preserves_full_input_shape_and_combined_budget(self):
        state = analyze_focus(self.data, self.problem, self.connection, self.config)
        grid = make_parameter_grid(self.problem, self.connection, max_candidates=16)
        pool, plan = focus_candidate_plan(grid, self.data, self.problem, self.connection,
            state, self.config, SearchConfig(max_candidates=2), RegionPolicy())
        self.assertLessEqual(len(pool), 2)
        self.assertEqual(pool.shape[1], len(self.problem.parameters))
        self.assertEqual(plan['candidate_space_scope'], 'focused_with_checks')
        self.assertTrue(plan['pool_changed'])

    def test_hybrid_disagreement_requires_rechecking_instead_of_continuing_fixed(self):
        state = analyze_focus(self.data, self.problem, self.connection, self.config)
        grid = np.array([[0., .5], [1., .5]])
        prediction = SimpleNamespace(results={'score': {'hybrid_mean': np.array([10., 13.])}})
        check_hybrid_effects(state, grid, prediction,
            {'confirmation_roles': ['FROZEN_CHECK', 'FROZEN_CHECK']}, self.problem, self.config)
        self.assertTrue(item_for(state, 'x')['hybrid_recheck'])
        after = analyze_focus(self.data, self.problem, self.connection, self.config, state)
        self.assertNotIn('x', after['fixed_values'])

    def test_categorical_input_does_not_receive_a_false_numeric_freeze(self):
        variable = replace(self.problem.parameters[0], unit='category')
        problem = replace(self.problem, parameters=[variable, self.problem.parameters[1]])
        state = analyze_focus(self.data, problem, self.connection, self.config)
        self.assertEqual(state['fixed_values'], {})
        self.assertFalse(state['analysis']['identifiable'])


if __name__ == '__main__':
    unittest.main()
