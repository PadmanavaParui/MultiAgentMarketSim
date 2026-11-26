"""
Tests for Phase 3: Game Theory (PayoffBuilder, NashSolver, BestResponseDynamics).
Covers TEST-008: Nashpy solver on a known toy 2x2 payoff matrix returns the analytically known equilibrium.
"""

import unittest
import numpy as np

from src.gametheory.payoff_builder import PayoffBuilder
from src.gametheory.nash_solver import NashSolver
from src.gametheory.best_response import BestResponseDynamics


class TestGameTheory(unittest.TestCase):
    def test_test_008_nash_solver_prisoners_dilemma(self):
        """
        TEST-008: Nashpy solver on a known toy 2x2 payoff matrix returns
        the analytically known equilibrium.
        Game: Prisoner's Dilemma
          Actions: Cooperate (0), Defect (1)
          Payoffs:
            P1: Cooperate/Cooperate -> 3, Cooperate/Defect -> 0, Defect/Cooperate -> 5, Defect/Defect -> 1
            P2: Cooperate/Cooperate -> 3, Defect/Cooperate -> 0, Cooperate/Defect -> 5, Defect/Defect -> 1
          Dominant strategy for both players is Defect (row 1, col 1).
          Known unique Nash Equilibrium is ( [0, 1], [0, 1] ).
        """
        A = np.array([[3.0, 0.0], [5.0, 1.0]])
        B = np.array([[3.0, 5.0], [0.0, 1.0]])

        solver = NashSolver(A, B)
        equilibria = solver.solve(method="support_enumeration")

        self.assertGreater(len(equilibria), 0)
        p1, p2 = equilibria[0]

        # Check that both players play Defect (index 1) with probability 1.0
        np.testing.assert_allclose(p1, [0.0, 1.0], atol=1e-3)
        np.testing.assert_allclose(p2, [0.0, 1.0], atol=1e-3)

    def test_test_008_nash_solver_matching_pennies(self):
        """
        TEST-008 (Mixed Equilibrium): Matching Pennies (Zero-sum game).
          A = [[ 1, -1],
               [-1,  1]]
          B = -A
          Unique mixed Nash equilibrium: ([0.5, 0.5], [0.5, 0.5]).
        """
        A = np.array([[1.0, -1.0], [-1.0, 1.0]])
        solver = NashSolver(A)  # zero-sum
        equilibria = solver.solve(method="support_enumeration")

        self.assertEqual(len(equilibria), 1)
        p1, p2 = equilibria[0]
        np.testing.assert_allclose(p1, [0.5, 0.5], atol=1e-3)
        np.testing.assert_allclose(p2, [0.5, 0.5], atol=1e-3)

    def test_payoff_builder_from_metrics(self):
        """Test building empirical payoff matrix from agent metrics."""
        mock_metrics = [
            {"agent_id": "agent_0", "pnl": 10.0},
            {"agent_id": "agent_1", "pnl": -10.0},
            {"agent_id": "agent_0", "pnl": 25.0},
            {"agent_id": "agent_1", "pnl": -25.0},
            {"agent_id": "agent_0", "pnl": -5.0},
            {"agent_id": "agent_1", "pnl": 5.0},
        ]
        A, B, strategies = PayoffBuilder.build_from_metrics(mock_metrics, "agent_0", "agent_1")
        self.assertEqual(A.shape, (2, 2))
        self.assertEqual(B.shape, (2, 2))
        self.assertEqual(len(strategies), 2)

    def test_payoff_builder_mean_pnl_ranking(self):
        """Test building mean PnL ranking matrix."""
        archetype_pnls = {
            "RL": [12.0, 15.0, 20.0],
            "MM": [5.0, 6.0, 7.0],
            "ZI": [-10.0, -8.0, -12.0],
        }
        A, B, archetypes = PayoffBuilder.mean_pnl_ranking(archetype_pnls)
        self.assertEqual(A.shape, (3, 3))
        self.assertEqual(B.shape, (3, 3))
        self.assertIn("RL", archetypes)

    def test_best_response_dynamics(self):
        """Test best-response calculation and tracing."""
        # Coordination game:
        # Both choose 0 -> (2, 2)
        # Both choose 1 -> (1, 1)
        A = np.array([[2.0, 0.0], [0.0, 1.0]])
        B = np.array([[2.0, 0.0], [0.0, 1.0]])

        brd = BestResponseDynamics(A, B)
        # Against pure action 0, best response should be 0
        br1 = brd.best_response_p1(np.array([1.0, 0.0]))
        np.testing.assert_allclose(br1, [1.0, 0.0])

        # Trace dynamics starting biased towards action 0
        trace = brd.trace(initial_p=[0.8, 0.2], initial_q=[0.8, 0.2], max_iterations=40, learning_rate=0.3)
        self.assertEqual(len(trace), 40)
        # Should converge towards pure (0, 0)
        converged = brd.check_convergence(trace, tolerance=1e-2)
        self.assertTrue(converged)


if __name__ == "__main__":
    unittest.main()
