"""使用禁止範囲と好ましい範囲の方針を検証します。"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

VERSION_ROOT = Path(__file__).resolve().parents[1]
if str(VERSION_ROOT) not in sys.path:
    sys.path.insert(0, str(VERSION_ROOT))

from src.policies import (  # noqa: E402
    RegionCondition,
    RegionPolicy,
    RegionRule,
    apply_preferred_range_bonus,
    load_search_regions,
)


class RegionPolicyTest(unittest.TestCase):
    def test_forbidden_is_hard_filter_and_preferred_is_bounded(self) -> None:
        policy = RegionPolicy(
            (
                RegionRule("ban", "forbidden", (RegionCondition("p1", lower=2),)),
                RegionRule("good", "preferred", (RegionCondition("p1", 0, 1),), strength=5),
            )
        )
        result = policy.evaluate(
            np.array([[0.0], [1.0], [2.0], [3.0]]),
            ["p1"],
            scores=np.ones(4),
        )
        self.assertEqual(result.allowed_mask.tolist(), [True, True, False, False])
        self.assertEqual(result.preferred_multiplier.tolist(), [2.0, 2.0, 1.0, 1.0])
        self.assertEqual(result.scores.tolist(), [2.0, 2.0, 1.0, 1.0])

    def test_same_region_id_is_and(self) -> None:
        policy = RegionPolicy(
            (
                RegionRule(
                    "r",
                    "forbidden",
                    (RegionCondition("p1", lower=1), RegionCondition("p2", upper=2)),
                ),
            )
        )
        candidates = np.array([[1, 2], [1, 3], [0, 1]], dtype=float)
        self.assertEqual(
            policy.evaluate(candidates, ["p1", "p2"]).allowed_mask.tolist(),
            [False, True, True],
        )

    def test_forbidden_has_priority_over_preferred(self) -> None:
        policy = RegionPolicy(
            (
                RegionRule("ban", "forbidden", (RegionCondition("p1", 0, 1),)),
                RegionRule("good", "preferred", (RegionCondition("p1", 0, 1),), strength=5),
            )
        )
        result = policy.evaluate(np.array([[0.5]]), ["p1"], scores=[3])
        self.assertFalse(bool(result.allowed_mask[0]))
        self.assertEqual(result.preferred_multiplier[0], 1)
        self.assertEqual(
            apply_preferred_range_bonus([3], np.array([[0.5]]), ["p1"], policy)[0],
            0,
        )

    def test_csv_loader_supports_multicondition(self) -> None:
        contents = (
            "region_id,kind,parameter,lower,upper,lower_inclusive,"
            "upper_inclusive,strength,enabled,note\n"
            "r1,forbidden,p1,2,,,false,3,true,\n"
            "r1,forbidden,p2,,5,true,true,3,true,\n"
            "r2,preferred,p1,0,1,true,true,1,true,好ましい\n"
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "search_regions.csv"
            path.write_text(contents, encoding="utf-8")
            policy = load_search_regions(path)

        self.assertEqual(len(policy.rules), 2)
        mask = policy.evaluate(
            np.array([[2, 5], [2, 6], [0.5, 8]]), ["p1", "p2"]
        ).allowed_mask
        self.assertEqual(mask.tolist(), [False, True, True])


if __name__ == "__main__":
    unittest.main()
