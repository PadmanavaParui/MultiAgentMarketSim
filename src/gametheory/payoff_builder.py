"""
Payoff Matrix Builder for MultiAgentMarketSim.

Constructs 2-player normal-form payoff matrices (A, B) from logged simulation
records (trades, agent metrics, or head-to-head empirical evaluations).
Compatible with Nashpy games and game-theoretic equilibrium analysis.
"""

from __future__ import annotations

from typing import List, Dict, Any, Tuple, Optional
import numpy as np


class PayoffBuilder:
    """
    Constructs normal-form payoff matrices from simulation run logs.
    """

    def __init__(self):
        pass

    @staticmethod
    def build_from_metrics(
        agent_metrics: List[Dict[str, Any]],
        row_agent: str,
        col_agent: str,
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Constructs a 2x2 empirical payoff matrix comparing two agents across episodes.
        Strategies represent relative market regimes (e.g., positive return vs negative return episodes).
        """
        row_data = [m for m in agent_metrics if m.get("agent_id") == row_agent]
        col_data = [m for m in agent_metrics if m.get("agent_id") == col_agent]

        n = min(len(row_data), len(col_data))
        if n == 0:
            # Fallback zero matrix
            return np.zeros((2, 2)), np.zeros((2, 2)), ["Strategy 1", "Strategy 2"]

        row_pnls = [m.get("pnl", 0.0) for m in row_data[:n]]
        col_pnls = [m.get("pnl", 0.0) for m in col_data[:n]]

        mean_row = np.mean(row_pnls)
        mean_col = np.mean(col_pnls)

        # 2x2 matrix:
        # State 0: Below mean, State 1: Above mean
        A = np.zeros((2, 2))
        B = np.zeros((2, 2))

        counts = np.zeros((2, 2))

        for r_pnl, c_pnl in zip(row_pnls, col_pnls):
            r_idx = 1 if r_pnl >= mean_row else 0
            c_idx = 1 if c_pnl >= mean_col else 0
            A[r_idx, c_idx] += r_pnl
            B[r_idx, c_idx] += c_pnl
            counts[r_idx, c_idx] += 1

        # Average where counts > 0
        for i in range(2):
            for j in range(2):
                if counts[i, j] > 0:
                    A[i, j] /= counts[i, j]
                    B[i, j] /= counts[i, j]
                else:
                    A[i, j] = mean_row
                    B[i, j] = mean_col

        strategies = ["Conservative / Below-Avg", "Aggressive / Above-Avg"]
        return A, B, strategies

    @staticmethod
    def build_archetype_matrix(
        archetype_pnls: Dict[str, List[float]]
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Constructs a payoff matrix between archetypes (e.g., ZI, MM, RL).
        If 2 archetypes: 2x2 matrix.
        If N archetypes: NxN zero-sum or general-sum matrix of relative advantages.
        """
        archetypes = sorted(list(archetype_pnls.keys()))
        n = len(archetypes)
        if n < 2:
            return np.zeros((2, 2)), np.zeros((2, 2)), ["Default 1", "Default 2"]

        means = {arch: float(np.mean(pnls)) if pnls else 0.0 for arch, pnls in archetype_pnls.items()}

        # Build pairwise payoff matrix
        A = np.zeros((n, n))
        B = np.zeros((n, n))

        for i, a1 in enumerate(archetypes):
            for j, a2 in enumerate(archetypes):
                # Relative payoff difference
                diff = means[a1] - means[a2]
                A[i, j] = means[a1]
                B[i, j] = means[a2]

        return A, B, archetypes

    @staticmethod
    def build_symmetric_matrix(values_2x2: List[List[float]]) -> Tuple[np.ndarray, np.ndarray]:
        """Utility to convert a 2x2 list of payoffs into (A, B) numpy arrays."""
        A = np.array(values_2x2, dtype=np.float64)
        B = A.T  # Symmetric game
        return A, B
