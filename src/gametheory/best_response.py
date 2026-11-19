"""
Best-Response Dynamics for MultiAgentMarketSim.

Simulates and traces iterative best-response dynamics and fictitious play
over normal-form payoff matrices (LLD § gametheory/best_response.py).
Tracks strategy trajectories and evaluates empirical convergence or limit cycles.
"""

from __future__ import annotations

from typing import List, Tuple, Dict, Any, Optional
import numpy as np


class BestResponseDynamics:
    """
    Computes best responses and traces iterative best-response paths.
    """

    def __init__(self, A: np.ndarray, B: Optional[np.ndarray] = None):
        self.A = np.asarray(A, dtype=np.float64)
        if B is None:
            self.B = -self.A
        else:
            self.B = np.asarray(B, dtype=np.float64)

        self.nrows, self.ncols = self.A.shape

    def best_response_p1(self, q: np.ndarray) -> np.ndarray:
        """
        Computes Player 1's best response (pure strategy vector)
        against Player 2's mixed strategy q.
        """
        expected_payoffs = np.dot(self.A, q)
        best_action = int(np.argmax(expected_payoffs))
        br = np.zeros(self.nrows, dtype=np.float64)
        br[best_action] = 1.0
        return br

    def best_response_p2(self, p: np.ndarray) -> np.ndarray:
        """
        Computes Player 2's best response (pure strategy vector)
        against Player 1's mixed strategy p.
        """
        expected_payoffs = np.dot(p, self.B)
        best_action = int(np.argmax(expected_payoffs))
        br = np.zeros(self.ncols, dtype=np.float64)
        br[best_action] = 1.0
        return br

    def trace(
        self,
        initial_p: Optional[np.ndarray] = None,
        initial_q: Optional[np.ndarray] = None,
        max_iterations: int = 50,
        learning_rate: float = 0.2,
    ) -> List[Dict[str, Any]]:
        """
        Traces continuous/damped best response dynamics (Fictitious Play style).
        p_{t+1} = (1 - lr) * p_t + lr * BR_1(q_t)
        q_{t+1} = (1 - lr) * q_t + lr * BR_2(p_t)
        """
        if initial_p is None:
            p = np.ones(self.nrows) / self.nrows
        else:
            p = np.asarray(initial_p, dtype=np.float64)
            p = p / np.sum(p)

        if initial_q is None:
            q = np.ones(self.ncols) / self.ncols
        else:
            q = np.asarray(initial_q, dtype=np.float64)
            q = q / np.sum(q)

        history: List[Dict[str, Any]] = []

        for t in range(max_iterations):
            # Compute current expected payoffs
            p1_payoff = float(np.dot(p, np.dot(self.A, q)))
            p2_payoff = float(np.dot(p, np.dot(self.B, q)))

            history.append({
                "step": t,
                "p": [round(float(x), 4) for x in p],
                "q": [round(float(y), 4) for y in q],
                "p1_payoff": round(p1_payoff, 4),
                "p2_payoff": round(p2_payoff, 4),
            })

            # Calculate best responses
            br1 = self.best_response_p1(q)
            br2 = self.best_response_p2(p)

            # Update strategies
            p_next = (1.0 - learning_rate) * p + learning_rate * br1
            q_next = (1.0 - learning_rate) * q + learning_rate * br2

            p = p_next / np.sum(p_next)
            q = q_next / np.sum(q_next)

        return history

    @staticmethod
    def check_convergence(trace_history: List[Dict[str, Any]], tolerance: float = 1e-3) -> bool:
        """
        Checks whether the last several steps of a trace have stabilized (converged).
        """
        if len(trace_history) < 5:
            return False
        recent_p = [step["p"] for step in trace_history[-5:]]
        diffs = [
            np.max(np.abs(np.array(recent_p[i]) - np.array(recent_p[i - 1])))
            for i in range(1, len(recent_p))
        ]
        return bool(max(diffs) < tolerance)
