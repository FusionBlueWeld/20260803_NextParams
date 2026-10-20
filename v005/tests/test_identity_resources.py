"""旧CSV丸めの回帰と、GPの有界メモリ予測を検証します。"""

from __future__ import annotations

from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np

from v005.src.candidate_identity import CandidateIdentity
from v005.src.connection import load_stage_connection
from v005.src.hybrid.gp_component import GaussianProcess, predict_in_chunks, rbf_kernel
from v005.src.parameter_space import make_axis, make_parameter_grid
from v005.src.preprocessing import prepare_experiments
from v005.src.settings import ProblemDefinition, VariableDefinition
from v005.src.validation import ExperimentData, load_and_validate_experiments, load_and_validate_problem


def variable(name="x", lower=0.0, upper=1.0, step=1 / 3):
    return VariableDefinition(name, name, "", "parameter", "", lower, upper, step, None)


def problem(parameters):
    objective = VariableDefinition("y", "y", "", "objective", "minimize", None, None, None, None)
    return ProblemDefinition(parameters, objective, [], [], [objective], 4)


class IdentityTests(unittest.TestCase):
    def test_legacy_serialized_grid_excluded(self):
        definition = problem([variable()])
        candidates = np.asarray(make_axis(0.0, 1.0, 1 / 3))[:, None]
        measured = np.asarray([[float(format(value, ".10g"))] for value in candidates[:, 0]])
        identity = CandidateIdentity(definition)
        np.testing.assert_array_equal(identity.measured_mask(candidates, measured), np.ones(4, dtype=bool))
        np.testing.assert_array_equal(identity.canonicalize(measured), candidates)
        self.assertEqual(len(identity.exclude_observed(candidates, measured)), 0)

    def test_repeat_aggregation_reuses_identity(self):
        definition = problem([variable()])
        data = ExperimentData(["a", "b", "c"], [[1 / 3], [0.3333333333], [1.0]], {"y": [2, 4, 8]})
        prepared = prepare_experiments(data, definition)
        np.testing.assert_array_equal(prepared.repeat_counts, [2, 1])
        np.testing.assert_array_equal(prepared.result_means["y"], [3, 8])
        np.testing.assert_array_equal(prepared.raw_parameters[:, 0], [1 / 3, 1])

    def test_fine_distinct_levels_do_not_merge(self):
        definition = problem([variable(lower=1.0, upper=1.0 + 1e-11, step=1e-12)])
        candidates = make_parameter_grid(definition, max_candidates=100)
        identity = CandidateIdentity(definition)
        self.assertEqual(len(set(identity.keys(candidates))), len(candidates))
        measured = candidates[4:5]
        self.assertEqual(int(np.sum(identity.measured_mask(candidates, measured))), 1)

    def test_ambiguous_legacy_serialization_is_not_snapped(self):
        definition = problem([variable(lower=1.000000000012, upper=1.000000000052, step=1e-11)])
        identity = CandidateIdentity(definition)
        candidates = make_parameter_grid(definition, max_candidates=100)
        self.assertEqual(int(np.sum(identity.measured_mask(candidates, [[1.0]]))), 0)
        np.testing.assert_array_equal(identity.canonicalize([[1.0]]), [[1.0]])

    def test_actual_off_grid_measurements_are_kept_distinct(self):
        identity = CandidateIdentity(problem([variable()]))
        rows = np.asarray([[0.1234567890123], [0.1234567890124]])
        self.assertEqual(len(set(identity.keys(rows))), 2)
        np.testing.assert_array_equal(identity.canonicalize(rows), rows)

    def test_incoming_context_never_snapped(self):
        definition = problem([variable("control"), variable("incoming")])
        identity = CandidateIdentity(definition, {"incoming": 1 / 3})
        rows = np.asarray([[1 / 3, 1 / 3], [0.3333333333, 0.3333333333]])
        self.assertEqual(len(set(identity.keys(rows))), 2)
        self.assertEqual(len(identity.exclude_observed(rows[:1], rows[1:])), 1)
        data = ExperimentData(["a", "b"], rows.tolist(), {"y": [2, 4]})
        self.assertEqual(prepare_experiments(data, definition, {"incoming": 1 / 3}).unique_condition_count, 2)

    def test_upper_endpoint_with_short_last_interval(self):
        identity = CandidateIdentity(problem([variable(lower=-0.4, upper=0.45, step=0.2)]))
        candidates = np.asarray(make_axis(-0.4, 0.45, 0.2))[:, None]
        self.assertEqual(len(set(identity.keys(candidates))), len(candidates))
        self.assertTrue(identity.measured_mask(candidates, [[0.45]])[-1])

    def test_empty_and_invalid_rows(self):
        identity = CandidateIdentity(problem([variable()]))
        self.assertEqual(identity.exclude_observed(np.empty((0, 1)), []).shape, (0, 1))
        with self.assertRaises(ValueError):
            identity.keys([[np.nan]])
        with self.assertRaises(ValueError):
            identity.keys([[1, 2]])

    def test_existing_copper_trial_rounding_regression(self):
        trial = Path(__file__).resolve().parents[1] / "trials" / "trial_cut_cu_20261009"
        if not trial.exists():
            self.skipTest("既存trialがない環境")
        definition = load_and_validate_problem(trial / "problem.csv")
        connection = load_stage_connection(trial / "stage_connection.json", definition)
        data = load_and_validate_experiments(trial / "data" / "experiments.csv", definition)
        identity = CandidateIdentity(definition, connection.current_incoming_context)
        grid = make_parameter_grid(definition, connection, max_candidates=20000)
        canonical = identity.canonicalize(data.parameter_rows)
        self.assertEqual(len(identity.keys(canonical)), 124)
        self.assertEqual(len(set(identity.keys(canonical))), 110)
        self.assertEqual(int(np.sum(identity.measured_mask(grid, data.parameter_rows))), 110)


class GPResourceTests(unittest.TestCase):
    def test_rbf_matches_broadcast_reference(self):
        rng = np.random.default_rng(28)
        left, right = rng.random((131, 48)), rng.random((23, 48))
        expected = np.exp(-0.5 * np.sum((left[:, None] - right[None]) ** 2, axis=2) / 0.35**2)
        np.testing.assert_allclose(rbf_kernel(left, right, 0.35), expected, rtol=1e-12, atol=1e-15)
        np.testing.assert_allclose(np.diag(rbf_kernel(left, left, 0.35)), 1.0, atol=1e-12)

    def test_chunked_prediction_matches_unbounded_reference(self):
        rng = np.random.default_rng(18)
        train = rng.random((43, 8))
        model = GaussianProcess.fit(train, np.sin(train[:, 0] * 3) + train[:, 1], 0.01)
        queries = rng.random((2500, 8))
        reference = model._predict_chunk(queries)
        sizes = []
        original = model._predict_chunk

        def record(chunk):
            sizes.append(len(chunk))
            return original(chunk)

        with patch.object(model, "_predict_chunk", side_effect=record):
            bounded = model.predict(queries, max_working_bytes=256 * 1024)
        self.assertGreater(len(sizes), 1)
        self.assertLessEqual(max(sizes), 256 * 1024 // (8 * (4 * 43 + 2 * 8 + 4)))
        for actual, expected in zip(bounded, reference):
            np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)

    def test_empty_and_invalid_prediction_budgets(self):
        model = GaussianProcess.fit(np.asarray([[0.0], [1.0]]), np.asarray([0.0, 1.0]), 0.01)
        self.assertEqual(model.predict(np.empty((0, 1)))[0].shape, (0,))
        self.assertEqual(predict_in_chunks(model, np.empty((0, 1)))[0].shape, (0,))
        with self.assertRaises(ValueError):
            model.predict([[0.0]], max_working_bytes=0)
        with self.assertRaises(ValueError):
            model.predict([[0.0]], max_working_bytes=1)
        with self.assertRaises(ValueError):
            predict_in_chunks(model, [[0.0]], chunk_size=0)
        with self.assertRaises(ValueError):
            rbf_kernel(np.zeros((1, 1)), np.zeros((1, 1)), 0)


if __name__ == "__main__":
    unittest.main()
