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
        SE_A: Optional[np.ndarray] = None,
        SE_B: Optional[np.ndarray] = None,
    ) -> List[Dict[str, Any]]:
        """
        Returns structured equilibrium candidates with expected payoffs, formatted strings,
        and standard-error deviation robustness checks (flagging weak equilibria).
        """
        eqs = self.solve(method=method)
        results = []

        se_a_arr = np.asarray(SE_A, dtype=np.float64) if SE_A is not None else None
        se_b_arr = np.asarray(SE_B, dtype=np.float64) if SE_B is not None else (
            se_a_arr.T if (se_a_arr is not None and np.allclose(self.B, self.A.T, atol=1e-3)) else None
        )

        for idx, (p1, p2) in enumerate(eqs):
            # Calculate expected payoffs: p1 @ A @ p2, p1 @ B @ p2
            payoff_1 = float(np.dot(p1, np.dot(self.A, p2)))
            payoff_2 = float(np.dot(p1, np.dot(self.B, p2)))

            p1_probs = [round(float(x), 4) for x in p1]
            p2_probs = [round(float(x), 4) for x in p2]

            is_pure = bool(all(x in (0.0, 1.0) for x in p1_probs) and all(y in (0.0, 1.0) for y in p2_probs))

            # Robustness check: check whether each player's payoff beats their best deviation by > 1 combined SE
            p1_beats_deviation = False
            p2_beats_deviation = False
            delta_1 = 0.0
            delta_2 = 0.0
            comb_se_1 = 0.0
            comb_se_2 = 0.0

            if is_pure:
                i_star = int(np.argmax(p1))
                j_star = int(np.argmax(p2))
                nrows, ncols = self.A.shape

                # Player 1 deviations: i != i_star against j_star
                if nrows > 1:
                    other_i = [i for i in range(nrows) if i != i_star]
                    dev_payoffs_1 = [self.A[i, j_star] for i in other_i]
                    best_dev_1_idx = other_i[int(np.argmax(dev_payoffs_1))]
                    best_dev_1_val = self.A[best_dev_1_idx, j_star]
                    delta_1 = float(payoff_1 - best_dev_1_val)

                    if se_a_arr is not None:
                        se_cur = float(se_a_arr[i_star, j_star])
                        se_dev = float(se_a_arr[best_dev_1_idx, j_star])
                        comb_se_1 = float(np.sqrt(se_cur**2 + se_dev**2))
                    else:
                        comb_se_1 = 0.0

                    p1_beats_deviation = bool(delta_1 > comb_se_1)
                else:
                    delta_1 = 0.0
                    comb_se_1 = 0.0
                    p1_beats_deviation = True

                # Player 2 deviations: j != j_star against i_star
                if ncols > 1:
                    other_j = [j for j in range(ncols) if j != j_star]
                    dev_payoffs_2 = [self.B[i_star, j] for j in other_j]
                    best_dev_2_idx = other_j[int(np.argmax(dev_payoffs_2))]
                    best_dev_2_val = self.B[i_star, best_dev_2_idx]
                    delta_2 = float(payoff_2 - best_dev_2_val)

                    if se_b_arr is not None:
                        se_cur_2 = float(se_b_arr[i_star, j_star])
                        se_dev_2 = float(se_b_arr[i_star, best_dev_2_idx])
                        comb_se_2 = float(np.sqrt(se_cur_2**2 + se_dev_2**2))
                    else:
                        comb_se_2 = 0.0

                    p2_beats_deviation = bool(delta_2 > comb_se_2)
                else:
                    delta_2 = 0.0
                    comb_se_2 = 0.0
                    p2_beats_deviation = True

                is_weak = not (p1_beats_deviation and p2_beats_deviation)
            else:
                # Mixed strategy equilibria have zero margin against support actions
                is_weak = True
                p1_beats_deviation = False
                p2_beats_deviation = False

            results.append({
                "equilibrium_id": idx + 1,
                "player_1_strategy": p1_probs,
                "player_2_strategy": p2_probs,
                "player_1_payoff": round(payoff_1, 4),
                "player_2_payoff": round(payoff_2, 4),
                "is_pure": is_pure,
                "is_weak": is_weak,
                "p1_deviation_margin": round(delta_1, 4),
                "p1_combined_se": round(comb_se_1, 4),
                "p2_deviation_margin": round(delta_2, 4),
                "p2_combined_se": round(comb_se_2, 4),
                "p1_beats_deviation": p1_beats_deviation,
                "p2_beats_deviation": p2_beats_deviation,
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
