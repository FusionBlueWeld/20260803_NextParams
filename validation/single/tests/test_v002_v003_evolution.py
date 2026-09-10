import unittest

import numpy as np

from validation.single.src.checks.v002_v003_evolution import _optimal_mask, aggregate, build_parser, main, make_problem
from validation.single.src.simulators import load


def _arm(simulator, version, seed, complexity, arrival, final_regret, reduction):
    scopes = ("global", "local", "near_optimum_k64")
    metrics = {
        scope: {
            "macro_nrmse": 0.20 if version == "v002" else 0.10,
            "macro_nlpd": 1.0 if version == "v002" else 0.5,
            "macro_coverage95": 0.70 if version == "v002" else 0.80,
            "feasible_accuracy": 0.75 if version == "v002" else 0.85,
            "outputs": {},
        }
        for scope in scopes
    }
    metrics["near_optimum_k64"]["count"] = 64
    metrics["near_optimum_k64"]["effective_radius"] = 0.47
    return {
        "simulator": simulator,
        "complexity_group": complexity,
        "version": version,
        "seed": seed,
        "found": arrival is not None,
        "arrival_step": arrival,
        "initial_regret": 0.8,
        "final_regret": final_regret,
        "regret_reduction": reduction,
        "metrics": metrics,
    }


class V002V003EvolutionTests(unittest.TestCase):
    def test_version_adapter_uses_same_problem_shape(self):
        sim = load("thermal_curing")
        old = make_problem(sim, "v002")
        new = make_problem(sim, "v003")
        self.assertEqual(len(old.parameters), len(new.parameters))
        self.assertEqual([x.column for x in old.result_variables], [x.column for x in new.result_variables])
        self.assertEqual(old.candidate_count, len(sim.grid()))

    def test_optimum_detection_includes_equivalent_optima(self):
        sim = load("milling")
        truth = sim.evaluate_points(sim.grid())
        best_index = sim.best_index(truth)
        self.assertIsNotNone(best_index)
        global_best = float(np.asarray(truth[sim.objective.column])[best_index])
        mask = _optimal_mask(sim, truth, global_best)
        self.assertGreater(np.count_nonzero(mask), 1)
        self.assertTrue(mask[best_index])

    def test_aggregate_direction_complexity_and_censoring(self):
        results = [
            _arm("thermal_curing", "v002", 0, "simple", 5, 0.30, 0.50),
            _arm("thermal_curing", "v003", 0, "simple", 3, 0.20, 0.60),
            _arm("milling", "v002", 0, "high_dim", None, 0.70, 0.10),
            _arm("milling", "v003", 0, "high_dim", 15, 0.50, 0.30),
        ]
        summary = aggregate(results, iterations=15)
        self.assertEqual(len(summary["paired"]), 2)
        high = next(row for row in summary["paired"] if row["simulator"] == "milling")
        self.assertEqual(high["v002_arrival"], 16)  # censored = budget + 1
        self.assertEqual(high["arrival_winner"], "v003")
        self.assertEqual(summary["by_complexity"]["simple"]["pairs"], 1)
        self.assertEqual(summary["by_complexity"]["high_dim"]["pairs"], 1)
        self.assertEqual(summary["overall"]["pairs"], 2)
        self.assertLess(summary["overall"]["final_regret_mean_delta_v003_minus_v002"], 0)
        self.assertEqual(summary["by_simulator"]["thermal_curing"]["pairs"], 1)

    def test_parser_rejects_invalid_values_in_main_validation_layer(self):
        parser = build_parser()
        args = parser.parse_args(["--output", "x", "--iterations", "0", "--seeds", "0"])
        self.assertEqual(args.iterations, 0)
        with self.assertRaises(SystemExit):
            main(["--output", "x", "--iterations", "0", "--seeds", "0"])
        with self.assertRaises(SystemExit):
            main(["--output", "x", "--iterations", "3", "--batch-size", "0"])

    def test_parser_accepts_batch_size(self):
        args = build_parser().parse_args(["--output", "x", "--batch-size", "3"])
        self.assertEqual(args.batch_size, 3)


if __name__ == "__main__":
    unittest.main()
