
"""複数推薦の選択方針を確認するテスト。"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


VERSION_ROOT = Path(__file__).resolve().parents[1]
if str(VERSION_ROOT) not in sys.path:
    sys.path.insert(0, str(VERSION_ROOT))

from src.gp.optimizer import (  # noqa: E402
    calculate_diversity_state,
    select_diverse_top_candidates,
)


class DiverseRecommendationTest(unittest.TestCase):
    def test_first_recommendation_is_always_highest_score(self) -> None:
        candidates = np.array([[0.1], [0.2], [0.6], [0.9]], dtype=float)
        measured = np.array([[0.0]], dtype=float)
        score = np.array([0.8, 1.0, 0.7, 0.6], dtype=float)

        selected = select_diverse_top_candidates(candidates, score, 3, measured, 0.35)

        self.assertEqual(selected[0], 1)

    def test_later_recommendations_cover_unobserved_space(self) -> None:
        candidates = np.array([[0.1], [0.2], [0.6], [0.9]], dtype=float)
        measured = np.array([[0.0]], dtype=float)
        score = np.array([1.0, 0.99, 0.8, 0.7], dtype=float)

        selected = select_diverse_top_candidates(candidates, score, 2, measured, 0.35)

        self.assertEqual(selected[0], 0)
        self.assertIn(selected[1], {2, 3})

    def test_requested_count_is_returned_without_duplicates(self) -> None:
        candidates = np.array(
            [[0.1, 0.1], [0.2, 0.2], [0.5, 0.5], [0.8, 0.8], [0.9, 0.1]],
            dtype=float,
        )
        measured = np.array([[0.0, 0.0]], dtype=float)
        score = np.array([1.0, 0.9, 0.8, 0.7, 0.6], dtype=float)

        selected = select_diverse_top_candidates(candidates, score, 4, measured, 0.35)

        self.assertEqual(len(selected), 4)
        self.assertEqual(len(set(selected)), 4)

    def test_feasibility_preference_applies_only_after_first_candidate(self) -> None:
        candidates = np.array([[0.1], [0.3], [0.6], [0.9]], dtype=float)
        measured = np.array([[0.0]], dtype=float)
        score = np.array([100.0, 0.8, 0.7, 0.6], dtype=float)
        preferred = np.array([False, True, True, True])

        selected = select_diverse_top_candidates(
            candidates,
            score,
            3,
            measured,
            0.35,
            preferred_candidates=preferred,
        )

        self.assertEqual(selected[0], 0)
        self.assertTrue(all(preferred[index] for index in selected[1:]))

    def test_auto_diversity_is_higher_for_uncertain_uncovered_candidates(self) -> None:
        score = np.linspace(1.0, 0.1, 200)
        broad = calculate_diversity_state(
            score=score,
            objective_std=np.full(200, 0.8),
            objective_scale=1.0,
            nearest_distance=np.full(200, 0.30),
        )
        narrow = calculate_diversity_state(
            score=score,
            objective_std=np.full(200, 0.1),
            objective_scale=1.0,
            nearest_distance=np.full(200, 0.03),
        )

        self.assertGreater(broad.weight, narrow.weight)
        self.assertGreaterEqual(narrow.weight, 0.15)
        self.assertLessEqual(broad.weight, 0.50)

    def test_degenerate_state_forces_minimum_diversity(self) -> None:
        state = calculate_diversity_state(
            score=np.ones(200),
            objective_std=np.ones(200),
            objective_scale=1.0,
            nearest_distance=np.ones(200),
            force_narrow=True,
        )

        self.assertAlmostEqual(state.weight, 0.15)


if __name__ == "__main__":
    unittest.main()


