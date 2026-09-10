import unittest

from validation.single.src.checks.batch_matrix import build_matrix


def _result(version, found, arrival, value, complexity="simple"):
    metric = {
        "macro_nrmse": value,
        "macro_nlpd": value,
        "macro_coverage95": value,
        "feasible_accuracy": value,
    }
    return {
        "version": version,
        "complexity_group": complexity,
        "found": found,
        "arrival_step": arrival,
        "final_regret": value,
        "metrics": {"global": metric, "near_optimum_k64": metric},
    }


class BatchMatrixTests(unittest.TestCase):
    def test_builds_four_overall_cells_and_censors_arrival(self):
        config = {"simulators": ["x"], "seeds": [0], "iterations": 15}
        batch1 = {"config": {**config, "batch_size": 1}, "results": [_result("v002", False, None, 0.2), _result("v003", True, 3, 0.1), _result("v002", True, 8, 0.4, "high_dim"), _result("v003", True, 5, 0.2, "high_dim")]}
        batch3 = {"config": {**config, "batch_size": 3}, "results": [_result("v002", True, 6, 0.3), _result("v003", True, 2, 0.15), _result("v002", False, None, 0.5, "high_dim"), _result("v003", True, 6, 0.25, "high_dim")]}
        matrix = build_matrix(batch1, batch3)
        overall = [cell for cell in matrix["cells"] if cell["group"] == "overall"]
        self.assertEqual(len(overall), 4)
        old = next(cell for cell in overall if cell["batch_size"] == 1 and cell["version"] == "v002")
        self.assertEqual(old["median_censored_arrival_conditions"], 12.0)
        self.assertEqual(old["median_censored_consumed_conditions"], 12.0)
        self.assertEqual(old["arrival_rate"], 0.5)

    def test_rejects_mismatched_budget(self):
        a = {"config": {"simulators": ["x"], "seeds": [0], "iterations": 15}, "results": []}
        b = {"config": {"simulators": ["x"], "seeds": [0], "iterations": 12}, "results": []}
        with self.assertRaises(ValueError):
            build_matrix(a, b)


if __name__ == "__main__":
    unittest.main()
