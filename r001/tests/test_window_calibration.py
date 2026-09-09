import copy
import tempfile
import unittest
from pathlib import Path

import numpy as np

from r001.src.windows.evaluation import evaluate_connected_window
from r001.src.validation import UserInputError
from r001.src.window_calibration import calibration_contract, validate_calibration


class Predictor:
    bounds = {"x": (0., 2.), "incoming": (0., 2.)}

    def predict_arrays(self, inputs):
        value = np.asarray(inputs["x"]) + np.asarray(inputs["incoming"])
        return {"outputs": {"y": {"mean": value, "std": np.ones_like(value) * .2}},
                "support": np.ones_like(value)}


class CalibrationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        path = Path(self.tmp.name)
        (path / "predictor.pkl").write_bytes(b"model-v1")
        self.manifest = {"controls": ["x"], "incoming_context": ["incoming"],
                         "predicted_outputs": ["y"], "local_constraints": []}
        self.stages = [({"id": "s"}, path, self.manifest)]
        self.config = {"external_context": {"s.incoming": .1}, "connections": [],
                       "final_specifications": [{"name": "y", "direction": "less_equal", "target": 1.}],
                       "connected_window": {"interval_method": "residual_quantile", "support_threshold": .5}}
        self.offset = {"lower": -.1, "upper": .1, "lower_z": 2., "upper_z": 2., "std_floor": .01}
        self.cal = {"schema_version": "1.1", "contract": calibration_contract(self.config, self.stages),
                    "offsets": {"s": {"y": self.offset}}}

    def test_contract_rejects_changed_context_wiring_policy_manifest_and_model(self):
        validate_calibration(self.cal, self.config, self.stages)
        for key, value in [("external_context", {"s.incoming": .2}),
                           ("connections", [{"from": "a.y", "to": "s.incoming"}]),
                           ("final_specifications", []),
                           ("connected_window", {"interval_method": "residual_quantile", "support_threshold": .1})]:
            changed = copy.deepcopy(self.config); changed[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_calibration(self.cal, changed, self.stages)
        changed_manifest = copy.deepcopy(self.manifest); changed_manifest["input_units"] = {"x": "new"}
        with self.assertRaises(ValueError):
            validate_calibration(self.cal, self.config, [(self.stages[0][0], self.stages[0][1], changed_manifest)])
        (self.stages[0][1] / "predictor.pkl").write_bytes(b"model-v2")
        with self.assertRaises(ValueError):
            validate_calibration(self.cal, self.config, self.stages)

    def test_missing_nonfinite_reversed_offsets_and_legacy_artifacts_rejected(self):
        for offset in [{}, {"lower": float("nan"), "upper": .1}, {"lower": 2., "upper": 1.}]:
            cal = copy.deepcopy(self.cal); cal["offsets"]["s"]["y"] = offset
            with self.subTest(offset=offset), self.assertRaises(ValueError):
                validate_calibration(cal, self.config, self.stages)
        cal = copy.deepcopy(self.cal); cal["schema_version"] = "1.0"
        with self.assertRaises(ValueError):
            validate_calibration(cal, self.config, self.stages)

    def test_method_changes_decision_for_local_final_and_connection_bounds(self):
        for connection in (False, True):
            config = copy.deepcopy(self.config)
            ps = [({"id": "s"}, self.manifest, Predictor())]
            links = {}
            if connection:
                # The downstream incoming upper bound is 1, so the source interval matters.
                downstream = Predictor(); downstream.bounds = {"x": (0., 2.), "incoming": (0., 1.)}
                ps.append(({"id": "t"}, self.manifest, downstream))
                links = {"t.incoming": "s.y"}
                config["final_specifications"] = [{"name": "y", "direction": "less_equal", "target": 10.}]
            config["_connected_window_calibration"] = {"offsets": {"s": {"y": self.offset}, "t": {"y": self.offset}}}
            outcomes = []
            for method in ("residual_quantile", "standardized_residual_quantile"):
                config["connected_window"]["interval_method"] = method
                result = evaluate_connected_window(ps, config, links, {"x": .7})
                outcomes.append(bool(result["trusted_feasible"]))
            self.assertEqual(outcomes, [True, False])

    def test_runtime_rejects_varying_external_context(self):
        config = copy.deepcopy(self.config); config["_connected_window_calibration"] = self.cal
        with self.assertRaises(UserInputError):
            evaluate_connected_window([({"id": "s"}, self.manifest, Predictor())], config, {},
                                  {"x": .7, "s.incoming": np.asarray([.1, .2])})


if __name__ == "__main__":
    unittest.main()
