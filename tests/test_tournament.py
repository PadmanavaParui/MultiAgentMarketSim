"""
Unit tests for the round-robin tournament payoff matrix and Nash equilibrium pipeline.

Tests:
  1. Matrix shape: NxN payoff and standard error matrices matching strategy roster.
  2. Opponent dependence: Row player's payoff genuinely depends on the opponent's strategy
     (verifies the fix for the bug where A[i,j]=means[a1] made rows constant).
  3. Determinism: Exact bit-for-bit reproducibility across runs with the same random seed.
  4. End-to-end tournament execution and serialization with standard errors.
"""

import os
import shutil
import unittest
import numpy as np

from src.gametheory.payoff_builder import PayoffBuilder
from src.gametheory.nash_solver import NashSolver
from scripts.run_tournament import run_tournament, get_default_strategies


class TestTournamentMatrix(unittest.TestCase):
    def setUp(self):
        self.strategies = [
            {"label": "ZI", "type": "ZI", "kwargs": {"aggression": 0.8, "max_quantity": 10}},
            {"label": "Tight MM", "type": "MM", "kwargs": {"spread": 0.5, "quantity": 10}},
            {"label": "Wide MM", "type": "MM", "kwargs": {"spread": 2.0, "quantity": 10}},
        ]
        self.background = [
            {"label": "BG_ZI", "type": "ZI", "kwargs": {"aggression": 0.5, "max_quantity": 10}}
        ]

    def tearDown(self):
        test_dir = os.path.join("data", "runs", "test_tournament_eval")
        if os.path.exists(test_dir):
            shutil.rmtree(test_dir, ignore_errors=True)

    def test_tournament_matrix_shape(self):
        """Test that build_tournament_matrix produces NxN matrices with matching labels."""
        n = len(self.strategies)
        A, B, SE_A, SE_B, labels = PayoffBuilder.build_tournament_matrix(
            strategies=self.strategies,
            background_agents=self.background,
            n_episodes=2,
            cycles_per_episode=15,
            seed=42,
        )

        self.assertEqual(A.shape, (n, n))
        self.assertEqual(B.shape, (n, n))
        self.assertEqual(SE_A.shape, (n, n))
        self.assertEqual(SE_B.shape, (n, n))
        self.assertEqual(len(labels), n)
        self.assertEqual(labels, ["ZI", "Tight MM", "Wide MM"])

    def test_tournament_matrix_opponent_dependence(self):
        """
        Test that Player X's payoff depends on the opponent strategy.
        In the buggy implementation, A[i,j] = means[a1], so every row had identical values.
        In the tournament matrix, the outcome depends on the opponent's behavior.
        """
        A, B, SE_A, SE_B, labels = PayoffBuilder.build_tournament_matrix(
            strategies=self.strategies,
            background_agents=self.background,
            n_episodes=3,
            cycles_per_episode=20,
            seed=123,
        )

        # Check that A is NOT row-constant (at least one row has distinct payoffs across columns)
        row_variations = [np.ptp(A[i, :]) for i in range(len(labels))]
        has_opponent_dependence = any(var > 1e-4 for var in row_variations)
        self.assertTrue(
            has_opponent_dependence,
            f"Payoff matrix rows should depend on opponent strategy, but row spreads are: {row_variations}",
        )

        # Check that B is NOT column-constant
        col_variations = [np.ptp(B[:, j]) for j in range(len(labels))]
        has_col_dependence = any(var > 1e-4 for var in col_variations)
        self.assertTrue(
            has_col_dependence,
            f"Payoff matrix columns should depend on row player strategy, but column spreads are: {col_variations}",
        )

    def test_tournament_matrix_determinism_by_seed(self):
        """Test that identical seeds yield identical payoff matrices and standard errors."""
        A1, B1, SE_A1, SE_B1, _ = PayoffBuilder.build_tournament_matrix(
            strategies=self.strategies,
            background_agents=self.background,
            n_episodes=2,
            cycles_per_episode=15,
            seed=777,
        )

        A2, B2, SE_A2, SE_B2, _ = PayoffBuilder.build_tournament_matrix(
            strategies=self.strategies,
            background_agents=self.background,
            n_episodes=2,
            cycles_per_episode=15,
            seed=777,
        )

        np.testing.assert_allclose(A1, A2, atol=1e-6)
        np.testing.assert_allclose(B1, B2, atol=1e-6)
        np.testing.assert_allclose(SE_A1, SE_A2, atol=1e-6)
        np.testing.assert_allclose(SE_B1, SE_B2, atol=1e-6)

        # Different seed should yield different numbers
        A3, _, _, _, _ = PayoffBuilder.build_tournament_matrix(
            strategies=self.strategies,
            background_agents=self.background,
            n_episodes=2,
            cycles_per_episode=15,
            seed=999,
        )
        self.assertFalse(np.allclose(A1, A3, atol=1e-4))

    def test_tournament_run_script_and_standard_errors_persistence(self):
        """Test run_tournament end-to-end and check standard_errors field in json."""
        test_run_id = "test_tournament_eval"
        logger = run_tournament(
            episodes=2,
            cycles_per_episode=10,
            seed=42,
            run_id=test_run_id,
            output_dir="data/runs",
            verbose=False,
        )

        eq_analysis = logger.get_equilibrium_analysis(test_run_id)
        self.assertIsNotNone(eq_analysis)
        self.assertIn("standard_errors", eq_analysis)
        self.assertIsNotNone(eq_analysis["standard_errors"])
        self.assertIn("payoff_matrix", eq_analysis)
        self.assertIn("equilibrium_candidates", eq_analysis)
        self.assertIn("strategies", eq_analysis)
        self.assertEqual(len(eq_analysis["strategies"]), 4)


if __name__ == "__main__":
    unittest.main()
