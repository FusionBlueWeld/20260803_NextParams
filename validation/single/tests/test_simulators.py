from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from validation.single.src import benchmark, metrics
from validation.single.run import main
from validation.single.src.simulators import Simulator, available, load, measure
from validation.single.src.settings import Variable
from validation.single.src.data_loader import read_csv, write_json


class SimulatorContracts(unittest.TestCase):
    def test_parameter_row_order_does_not_change_physics(self):
        for name in available():
            sim = load(name)
            inputs = {v.column: (v.lower + v.upper) / 2 for v in sim.parameters}
            expected = sim.evaluate(inputs)
            sim.parameters.reverse()
            actual = sim.evaluate(inputs)
            for column in expected:
                np.testing.assert_array_equal(expected[column], actual[column])

    def test_multidimensional_best_index_is_flat(self):
        sim = load("laser_welding")
        self.assertEqual(sim.best_index({"penetration_depth_mm": np.array([[2., 10.], [5., 4.]]),
                                         "spatter_level_0_9": np.array([[1., 7.], [3., 3.]])}), 2)

    def test_provenance_includes_nested_helpers(self):
        sim = load("milling")
        with tempfile.TemporaryDirectory() as temp:
            sim.folder = Path(temp)
            (sim.folder / "helpers").mkdir()
            (sim.folder / "helpers" / "core.py").write_text("# helper\n", encoding="utf-8")
            (sim.folder / "problem.csv").write_text("", encoding="utf-8")
            (sim.folder / "manifest.json").write_text("{}", encoding="utf-8")
            self.assertIn("helpers/core.py", sim.provenance()["sha256_lf"])

    def test_six_processes_and_nontrivial_grids(self):
        self.assertEqual(set(available()), {"laser_welding", "milling", "press_forming", "thermal_curing", "convection_drying", "electroplating"})
        expected_dimensions = {"milling": 8, "electroplating": 8}
        for name in available():
            with self.subTest(name=name):
                sim = load(name)
                grid = sim.grid()
                truth = sim.evaluate_points(grid)
                self.assertGreater(sim.feasible(truth).sum(), 0)
                self.assertLess(sim.feasible(truth).sum(), len(grid))
                for constraint in sim.constraints:
                    values = truth[constraint.column]
                    good = values <= constraint.target if constraint.direction == "less_equal" else values >= constraint.target
                    self.assertTrue(good.any(), constraint.column)
                    self.assertTrue((~good).any(), f"Vacuous constraint: {constraint.column}")
                initial = sim.initial_design()
                dimension = expected_dimensions.get(name, 3)
                size = sim.manifest.get("initial_design_size", 27)
                self.assertEqual(initial.shape, (size, dimension))
                self.assertEqual(len({tuple(p) for p in initial}), len(initial))
                self.assertGreater(sim.feasible(sim.evaluate_points(initial)).sum(), 0)
                best = sim.best_index(truth)
                self.assertIsNotNone(best)
                self.assertFalse(any(np.allclose(grid[best], p) for p in initial), "Optimum already in initial design")
                for v in truth.values():
                    self.assertEqual(v.shape, (len(grid),))

    def test_high_dimensional_designs_are_reproducible_and_cover_every_level(self):
        for name in ("milling", "electroplating"):
            sim = load(name)
            self.assertEqual(len(sim.parameters), 8)
            self.assertEqual(len(sim.grid()), 4**8)
            first = sim.initial_design()
            np.testing.assert_array_equal(first, sim.initial_design())
            self.assertFalse(np.array_equal(first, sim.initial_design(1)))
            for column, axis in enumerate(sim.axes):
                self.assertEqual(set(first[:, column]), set(axis))
            self.assertEqual(sim.initial_design_method(), "deterministic_maximin_grid")

    def test_broadcast_scalar_equivalence_and_validation(self):
        for name in available():
            sim = load(name)
            values = [np.array([[a[0]], [a[-1]]]) if i == 0 else
                      np.array([a[0], a[len(a)//2], a[-1]]) if i == 1 else a[len(a)//2]
                      for i, a in enumerate(sim.axes)]
            conditions = dict(zip(sim.parameter_columns, values))
            batch = sim.evaluate(conditions)
            for v in batch.values():
                self.assertEqual(v.shape, (2, 3))
            point = {n: np.broadcast_arrays(*values)[i][1, 2] for i, n in enumerate(sim.parameter_columns)}
            scalar = sim.evaluate(point)
            for n in scalar:
                self.assertAlmostEqual(float(scalar[n]), float(batch[n][1, 2]))
            for p in sim.parameters:
                for bad in [np.nan, np.inf, -np.inf, p.lower - 1, p.upper + 1]:
                    with self.subTest(name=name, parameter=p.column, bad=bad), self.assertRaises(ValueError):
                        sim.evaluate({**point, p.column: bad})
            with self.assertRaises(ValueError):
                sim.evaluate({**point, "unknown_input": 1})

    def test_old_laser_grid_and_import_remain_equivalent(self):
        from validation.single.src.checks.laser_welding_pseudo_experiment import (
            DEFAULT_ORACLE_ROOT, candidate_axes, load_oracle, initial_experiments,
        )
        sim = load("laser_welding")
        for a, b in zip(sim.axes, candidate_axes()):
            np.testing.assert_array_equal(a, b)
        grid = sim.grid()
        old = load_oracle(DEFAULT_ORACLE_ROOT)(*grid.T)
        new = sim.evaluate_points(grid)
        for key in old:
            np.testing.assert_array_equal(old[key], new[key])
        initial = initial_experiments(load_oracle(DEFAULT_ORACLE_ROOT))
        np.testing.assert_array_equal(sim.initial_design(), [[r[n] for n in sim.parameter_columns] for r in initial])
        self.assertEqual(int(sim.feasible(new).sum()), 9223)
        self.assertAlmostEqual(new["penetration_depth_mm"][sim.best_index(new)], 8.386370654563162, places=12)

    def test_seeded_design_and_separate_measurements(self):
        sim = load("electroplating")
        np.testing.assert_array_equal(sim.initial_design(7), sim.initial_design(7))
        self.assertFalse(np.array_equal(sim.initial_design(7), sim.initial_design(8)))
        truth = sim.evaluate_points(sim.initial_design())
        before = {n: v.copy() for n, v in truth.items()}
        spans = {n: float(np.ptp(v)) for n, v in truth.items()}
        first = measure(truth, spans, .02, np.random.default_rng(9))
        second = measure(truth, spans, .02, np.random.default_rng(9))
        zero = measure(truth, spans, 0, np.random.default_rng(9))
        for n in truth:
            np.testing.assert_array_equal(first[n], second[n])
            np.testing.assert_array_equal(truth[n], before[n])
            np.testing.assert_array_equal(zero[n], truth[n])
        self.assertFalse(np.array_equal(first[sim.objective.column], truth[sim.objective.column]))


class HarnessTests(unittest.TestCase):
    def test_response_space_uses_numeric_latest_and_rejects_stale(self):
        with tempfile.TemporaryDirectory() as temp:
            trial = Path(temp)
            (trial / "output").mkdir()
            for number in (9999, 10000):
                (trial / "output" / f"response_space_run_{number}.csv").touch()
            with patch.object(metrics, "read_csv", return_value=[{"data_rows": "26"}]) as reader:
                result = metrics.response_metrics(load("milling"), trial, np.zeros((27, 3)), {})
            self.assertEqual(result["status"], "not_available")
            self.assertEqual(reader.call_args.args[0].name, "response_space_run_10000.csv")

    def test_setup_failure_records_failed_status(self):
        with tempfile.TemporaryDirectory() as temp, patch.object(benchmark, "write_csv", side_effect=OSError("setup failure")):
            output = Path(temp) / "failed_setup"
            with self.assertRaisesRegex(OSError, "setup failure"):
                benchmark.run(load("milling"), "v000", "trial_setup_failure", output, iterations=1)
            self.assertEqual(json.loads((output / "run.json").read_text(encoding="utf-8"))["status"], "failed")

    def test_required_stop_does_not_reuse_recommendations(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "main.py").write_text("# test CLI\n", encoding="utf-8")
            trial = root / "v003" / "trials" / "trial_stop"
            def fake_cli(version, *args):
                if args[0] == "--new":
                    trial.mkdir(parents=True)
                elif args[0] == "--run":
                    write_json(trial / "output" / "stopping_status.json", {"status": "STOP_REQUIRED", "reasons": ["test limit"]})
                    (trial / "output" / "recommendations.csv").write_text("stale invalid content", encoding="utf-8")
                return ""
            with patch.object(benchmark, "PROJECT_ROOT", root), patch.object(benchmark, "versions", return_value=["v003"]), patch.object(benchmark, "cli", side_effect=fake_cli):
                result = benchmark.run(load("milling"), "v003", "trial_stop", root / "result", iterations=2)
            self.assertEqual(result["status"], "complete")
            self.assertEqual(result["completed_iterations"], 0)
            self.assertEqual(result["final_count"], len(load("milling").initial_design()))
            self.assertEqual(result["termination_reason"], "optimizer_stop_required")
            self.assertIsNone(result["true_feasible_recommendation_rate"])

    def test_small_axis_initial_design_has_no_duplicates(self):
        sim = load("laser_welding")
        sim.axes = [np.array([0., 1.]), np.array([2., 3.]), np.array([4., 5.])]
        points = sim.initial_design()
        self.assertEqual(len(points), 8)
        self.assertEqual(len({tuple(p) for p in points}), 8)

    def test_minimize_maximize_and_empty_feasible(self):
        minimum = load("electroplating")
        outputs = {"thickness_error_um": np.array([5., 0., 2.]), "roughness_ra_um": np.array([.2, 2., .3]),
                   "current_efficiency": np.array([.9, .9, .9])}
        self.assertEqual(minimum.best_index(outputs), 2)
        self.assertAlmostEqual(metrics.regret(minimum, 3, 1, 10), .2)
        outputs["current_efficiency"][:] = 0
        self.assertIsNone(minimum.best_index(outputs))
        maximum = load("laser_welding")
        self.assertEqual(maximum.best_index({"penetration_depth_mm": np.array([2., 10., 5.]), "spatter_level_0_9": np.array([1., 7., 3.])}), 2)
        self.assertAlmostEqual(metrics.regret(maximum, 8, 10, 10), .2)
        self.assertIsNone(metrics.regret(maximum, None, 10, 10))

    def test_duplicate_and_off_grid_recommendations(self):
        sim = load("milling")
        initial = sim.initial_design()
        seen = {tuple(np.round(p, 8)) for p in initial}
        with self.assertRaises(ValueError):
            benchmark.validate_recommendations(sim, initial[:1], seen)
        point = initial[:1].copy()
        point[0, 0] = sim.axes[0][0] + 0.1
        with self.assertRaises(ValueError):
            benchmark.validate_recommendations(sim, point, set())

    def test_cli_evaluate_export_and_existing_destination(self):
        with contextlib.redirect_stdout(io.StringIO()) as output:
            status = main(["evaluate", "laser_welding", "--set", "laser_power_w=5900", "spot_diameter_um=300", "scan_speed_mm_s=60"])
        self.assertEqual(status, 0)
        self.assertTrue(json.loads(output.getvalue())["feasible"])
        with tempfile.TemporaryDirectory() as temp:
            destination = Path(temp) / "export"
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(main(["export", "milling", "--output", str(destination)]), 0)
            original = (destination / "experiments.csv").read_bytes()
            self.assertEqual(len(read_csv(destination / "experiments.csv")), len(load("milling").initial_design()))
            with contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(main(["export", "milling", "--output", str(destination)]), 1)
            self.assertEqual(original, (destination / "experiments.csv").read_bytes())

    def test_invalid_cli_and_existing_trial_do_not_call_optimizer(self):
        with contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(main(["evaluate", "milling", "--set", "a=1"]), 1)
            self.assertEqual(main(["run", "milling", "--iterations", "0"]), 1)
        with tempfile.TemporaryDirectory() as temp, patch.object(benchmark, "cli") as cli:
            with self.assertRaises(FileExistsError):
                benchmark.run(load("milling"), "v000", "trial_test", Path(temp))
            cli.assert_not_called()

    def test_optimizer_failure_preserves_initial_data_and_status(self):
        with tempfile.TemporaryDirectory() as temp, patch.object(benchmark, "cli", side_effect=RuntimeError("test failure")):
            output = Path(temp) / "failed_run"
            with self.assertRaisesRegex(RuntimeError, "test failure"):
                benchmark.run(load("milling"), "v000", "trial_test_failure", output, iterations=1)
            status = json.loads((output / "run.json").read_text(encoding="utf-8"))
            self.assertEqual(status["status"], "failed")
            expected = len(load("milling").initial_design())
            self.assertEqual(len(read_csv(output / "observations.csv")), expected)
            self.assertEqual(len(read_csv(output / "truth.csv")), expected)


if __name__ == "__main__":
    unittest.main()
