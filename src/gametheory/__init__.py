"""
Game-Theoretic Analysis module for MultiAgentMarketSim.

Provides:
- PayoffBuilder: constructs payoff matrices from logged simulation records
- NashSolver: solves for Nash equilibria (Lemke-Howson / Support Enumeration) via Nashpy
- BestResponseDynamics: traces best-response trajectories and stability across strategy profiles
"""

from src.gametheory.payoff_builder import PayoffBuilder
from src.gametheory.nash_solver import NashSolver
from src.gametheory.best_response import BestResponseDynamics

__all__ = [
    "PayoffBuilder",
    "NashSolver",
    "BestResponseDynamics",
]
