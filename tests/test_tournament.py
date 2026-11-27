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
            pretrain_rl=False,
        )

        A2, B2, SE_A2, SE_B2, _ = PayoffBuilder.build_tournament_matrix(
            strategies=self.strategies,
            background_agents=self.background,
            n_episodes=2,
            cycles_per_episode=15,
            seed=777,
            pretrain_rl=False,
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
            pretrain_rl=False,
        )
        self.assertFalse(np.allclose(A1, A3, atol=1e-4))

    def test_q_table_unchanged_during_cells(self):
        """
        Verify that a pretrained, frozen RL agent's Q-table remains bit-for-bit
        identical before and after being used across tournament cells.
        """
        pretrained_rl = PayoffBuilder.pretrain_rl_agent(
            episodes=3,
            cycles_per_episode=15,
            seed=42,
        )
        self.assertTrue(pretrained_rl.frozen)
        self.assertEqual(pretrained_rl.epsilon, 0.0)
        self.assertEqual(pretrained_rl.alpha, 0.0)

        # Take deep copy snapshot of Q-table
        q_table_before = {k: np.copy(v) for k, v in pretrained_rl.q_table.items()}
        self.assertGreater(len(q_table_before), 0, "Pretrained agent should have visited states")

        strategies = [
            {"label": "ZI", "type": "ZI", "kwargs": {"aggression": 0.7, "max_quantity": 10}},
            {"label": "RL (pretrained)", "type": "RL", "kwargs": {}},
        ]

        A, B, SE_A, SE_B, labels = PayoffBuilder.build_tournament_matrix(
            strategies=strategies,
            background_agents=self.background,
            n_episodes=2,
            cycles_per_episode=15,
            seed=99,
            pretrain_rl=False,
            pretrained_rl_agent=pretrained_rl,
        )

        # Verify Q-table after tournament is 100% unchanged
        self.assertEqual(set(pretrained_rl.q_table.keys()), set(q_table_before.keys()))
        for k in q_table_before:
            np.testing.assert_array_equal(pretrained_rl.q_table[k], q_table_before[k])

    def test_weak_equilibrium_flag(self):
        """
        Verify that equilibria are flagged as 'weak' when deviation margins do not beat
        1 combined standard error, and 'strong' when they strictly do.
        """
        # Coordination Game with 2 pure Nash equilibria at (0, 0) and (1, 1)
        A = np.array([[10.0, 0.0], [0.0, 2.0]])
        B = np.array([[10.0, 0.0], [0.0, 2.0]])

        # Small standard errors -> (0, 0) margin is 10 - 0 = 10, combined SE = sqrt(1^2 + 1^2) = 1.41 -> Strong
        # Large standard errors for (1, 1) -> margin is 2 - 0 = 2, combined SE = sqrt(2^2 + 2^2) = 2.83 -> Weak
        SE_A = np.array([[1.0, 1.0], [2.0, 2.0]])
        SE_B = np.array([[1.0, 1.0], [2.0, 2.0]])

        solver = NashSolver(A, B)
        candidates = solver.get_equilibrium_candidates(
            method="support_enumeration",
            SE_A=SE_A,
            SE_B=SE_B,
        )

        self.assertGreaterEqual(len(candidates), 2)
        # Find (0, 0) and (1, 1) equilibria
        eq_00 = next(c for c in candidates if c["player_1_strategy"][0] == 1.0 and c["player_2_strategy"][0] == 1.0)
        eq_11 = next(c for c in candidates if c["player_1_strategy"][1] == 1.0 and c["player_2_strategy"][1] == 1.0)

        # eq_00: margin 10.0 > combined SE 1.414 -> is_weak == False
        self.assertFalse(eq_00["is_weak"])
        self.assertTrue(eq_00["p1_beats_deviation"])
        self.assertTrue(eq_00["p2_beats_deviation"])
        self.assertAlmostEqual(eq_00["p1_deviation_margin"], 10.0)

        # eq_11: margin 2.0 <= combined SE 2.828 -> is_weak == True
        self.assertTrue(eq_11["is_weak"])
        self.assertFalse(eq_11["p1_beats_deviation"])
        self.assertFalse(eq_11["p2_beats_deviation"])
        self.assertAlmostEqual(eq_11["p1_deviation_margin"], 2.0)

        # Mixed equilibrium in Matching Pennies -> always flagged as is_weak == True
        A_mp = np.array([[1.0, -1.0], [-1.0, 1.0]])
        solver_mp = NashSolver(A_mp)
        candidates_mp = solver_mp.get_equilibrium_candidates(
            method="support_enumeration",
            SE_A=np.ones((2, 2)),
        )
        self.assertEqual(len(candidates_mp), 1)
        self.assertTrue(candidates_mp[0]["is_weak"])
        self.assertFalse(candidates_mp[0]["is_pure"])

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
            pretrain_episodes=2,
            pretrain_cycles=10,
        )

        eq_analysis = logger.get_equilibrium_analysis(test_run_id)
        self.assertIsNotNone(eq_analysis)
        self.assertIn("standard_errors", eq_analysis)
        self.assertIsNotNone(eq_analysis["standard_errors"])
        self.assertIn("payoff_matrix", eq_analysis)
        self.assertIn("equilibrium_candidates", eq_analysis)
        self.assertIn("strategies", eq_analysis)
        self.assertEqual(len(eq_analysis["strategies"]), 4)
        self.assertIn("RL (pretrained)", eq_analysis["strategies"])
        self.assertIn("episodes_per_cell", eq_analysis)
        self.assertIn("pretrain_settings", eq_analysis)


if __name__ == "__main__":
    unittest.main()
