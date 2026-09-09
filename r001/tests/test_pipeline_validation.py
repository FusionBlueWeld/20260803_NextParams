"""入力検査の分離後も、接続順・単位・流入状態の契約を維持する回帰テスト。"""

import copy
from pathlib import Path
import unittest
from unittest.mock import patch

from r001.src.validation import UserInputError, validate_pipeline


class PipelineValidationTests(unittest.TestCase):
    def setUp(self):
        self.manifests = {
            "up": {
                "controls": ["power"], "incoming_context": [],
                "connector_outputs": ["temperature"], "predicted_outputs": ["temperature"],
                "output_units": {"temperature": "C"}, "input_units": {},
            },
            "down": {
                "controls": ["time"], "incoming_context": ["temperature"],
                "connector_outputs": ["quality"], "predicted_outputs": ["quality"],
                "output_units": {"quality": "MPa"}, "input_units": {"temperature": "C"},
            },
        }
        self.config = {
            "stages": [{"id": "up", "bundle": "up"}, {"id": "down", "bundle": "down"}],
            "connections": [{"from": "up.temperature", "to": "down.temperature"}],
            "candidate_axes": {"power": [1, 2], "time": [3, 4]},
            "external_context": {},
            "final_specifications": [{"name": "quality", "direction": "greater_equal", "target": 1}],
        }

    def validate(self, config=None):
        with patch("r001.src.validation.load_bundle_manifest", side_effect=lambda path: self.manifests[path.name]):
            return validate_pipeline(Path("trial_contract"), config or self.config)

    def test_valid_forward_chain_preserves_order(self):
        stages = self.validate()
        self.assertEqual([item[0]["id"] for item in stages], ["up", "down"])

    def test_reverse_connection_is_rejected(self):
        self.config["stages"].reverse()
        with self.assertRaisesRegex(UserInputError, "順方向"):
            self.validate()

    def test_unit_mismatch_is_rejected(self):
        self.manifests["down"]["input_units"]["temperature"] = "K"
        with self.assertRaisesRegex(UserInputError, "単位"):
            self.validate()

    def test_missing_external_state_is_rejected(self):
        self.config["connections"] = []
        with self.assertRaisesRegex(UserInputError, "external_context"):
            self.validate()

    def test_duplicate_supply_is_rejected(self):
        self.config["connections"].append(copy.deepcopy(self.config["connections"][0]))
        with self.assertRaisesRegex(UserInputError, "重複"):
            self.validate()


if __name__ == "__main__":
    unittest.main()
