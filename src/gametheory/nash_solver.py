"""
Nash Equilibrium Solver for MultiAgentMarketSim using Nashpy.

Computes pure and mixed-strategy Nash equilibria for 2-player normal form games
using algorithms such as Lemke-Howson and Support Enumeration (LLD, TRD §TR-006).
"""

from __future__ import annotations

from typing import List, Tuple, Dict, Any, Optional
import numpy as np

try:
    import nashpy as nash
    HAS_NASHPY = True
except ImportError:
    HAS_NASHPY = False


class NashSolver:
    """
    Solves 2-player normal-form games represented by payoff matrices A and B.
    """

    def __init__(self, A: np.ndarray, B: Optional[np.ndarray] = None):
        """
        Initialize the game with payoff matrices.
        If B is None, game is treated as zero-sum (B = -A).
        """
        self.A = np.asarray(A, dtype=np.float64)
        if B is None:
            self.B = -self.A
        else:
            self.B = np.asarray(B, dtype=np.float64)

        if HAS_NASHPY:
            self.game = nash.Game(self.A, self.B)
        else:
            self.game = None

    def solve(self, method: str = "support_enumeration") -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Solves the game for Nash equilibria using the specified method.
        Supported methods:
          - "support_enumeration" (default, finds all equilibria for small games)
          - "lemke_howson" (pivoting algorithm)
        """
        if not HAS_NASHPY:
            return self._solve_pure_fallback()

        method = method.lower()
        equilibria = []

        if method == "lemke_howson":
            try:
                # Run Lemke-Howson algorithm
                eq = self.game.lemke_howson(initial_dropped_label=0)
                equilibria.append((np.asarray(eq[0]), np.asarray(eq[1])))
            except Exception:
                # Fallback to support enumeration if Lemke-Howson fails (e.g. degenerate)
                for eq in self.game.support_enumeration():
                    equilibria.append((np.asarray(eq[0]), np.asarray(eq[1])))
        else:
            # support_enumeration (default)
            try:
                for eq in self.game.support_enumeration():
                    equilibria.append((np.asarray(eq[0]), np.asarray(eq[1])))
            except Exception:
                # Fallback
                equilibria = self._solve_pure_fallback()

        return equilibria

    def get_equilibrium_candidates(
        self,
        method: str = "support_enumeration",
        strategy_labels: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Returns structured equilibrium candidates with expected payoffs and formatted strings.
        """
        eqs = self.solve(method=method)
        results = []

        for idx, (p1, p2) in enumerate(eqs):
            # Calculate expected payoffs: p1 @ A @ p2, p1 @ B @ p2
            payoff_1 = float(np.dot(p1, np.dot(self.A, p2)))
            payoff_2 = float(np.dot(p1, np.dot(self.B, p2)))

            p1_probs = [round(float(x), 4) for x in p1]
            p2_probs = [round(float(x), 4) for x in p2]

            results.append({
                "equilibrium_id": idx + 1,
                "player_1_strategy": p1_probs,
                "player_2_strategy": p2_probs,
                "player_1_payoff": round(payoff_1, 4),
                "player_2_payoff": round(payoff_2, 4),
                "is_pure": bool(all(x in (0.0, 1.0) for x in p1_probs) and all(y in (0.0, 1.0) for y in p2_probs)),
            })

        return results

    def _solve_pure_fallback(self) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Lightweight fallback to check for pure-strategy Nash equilibria
        without needing external packages.
        """
        nrows, ncols = self.A.shape
        equilibria = []

        for i in range(nrows):
            for j in range(ncols):
                # Check if i is a best response to j for Player 1: A[i, j] >= A[k, j] for all k
                p1_best = bool(self.A[i, j] >= np.max(self.A[:, j]))
                # Check if j is a best response to i for Player 2: B[i, j] >= B[i, k] for all k
                p2_best = bool(self.B[i, j] >= np.max(self.B[i, :]))

                if p1_best and p2_best:
                    p1 = np.zeros(nrows)
                    p1[i] = 1.0
                    p2 = np.zeros(ncols)
                    p2[j] = 1.0
                    equilibria.append((p1, p2))

        return equilibria
