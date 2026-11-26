"""
Head-to-head empirical tournament runner for MultiAgentMarketSim.

Runs pairwise round-robin matchups for game-theoretic analysis:
  Strategies:
    - Zero-Intelligence (ZI)
    - Tight Market Maker (Tight MM, spread=0.5)
    - Wide Market Maker (Wide MM, spread=2.0)
    - RL Directional Trader (RL)
  Opponents:
    Every ordered pair (i, j) of strategies is evaluated with fixed background agents.
  Outputs:
    Saves payoff matrices (A, B), per-cell standard errors, strategy names,
    and Nash equilibrium candidates to data/runs/<run_id>/equilibrium_analysis.json.

Usage:
    python scripts/run_tournament.py --episodes 5 --cycles 50 --seed 42 --run-id tourney_001
"""

from __future__ import annotations

import os
import sys
import argparse
from typing import Dict, Any, List

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.gametheory.payoff_builder import PayoffBuilder
from src.gametheory.nash_solver import NashSolver
from src.gametheory.best_response import BestResponseDynamics
from src.metrics.logger import MetricsLogger


def get_default_strategies() -> List[Dict[str, Any]]:
    """Returns the four core tournament strategies: ZI, tight MM, wide MM, RL."""
    return [
        {
            "label": "ZI",
            "type": "ZI",
            "kwargs": {"aggression": 0.7, "max_quantity": 10},
        },
        {
            "label": "Tight MM",
            "type": "MM",
            "kwargs": {"spread": 0.5, "quantity": 10},
        },
        {
            "label": "Wide MM",
            "type": "MM",
            "kwargs": {"spread": 2.0, "quantity": 10},
        },
        {
            "label": "RL",
            "type": "RL",
            "kwargs": {"alpha": 0.1, "epsilon": 0.3, "order_quantity": 10},
        },
    ]


def get_default_background() -> List[Dict[str, Any]]:
    """Returns the fixed identical background agents."""
    return [
        {
            "label": "BG_ZI",
            "type": "ZI",
            "kwargs": {"aggression": 0.6, "max_quantity": 10},
        }
    ]


def run_tournament(
    episodes: int = 5,
    cycles_per_episode: int = 50,
    seed: int = 42,
    run_id: str = "tournament_run_001",
    output_dir: str = "data/runs",
    verbose: bool = True,
) -> MetricsLogger:
    """
    Executes the round-robin tournament and records the game-theoretic analysis.
    """
    strategies = get_default_strategies()
    background = get_default_background()
    strategy_labels = [s["label"] for s in strategies]

    logger = MetricsLogger(base_dir=output_dir)
    config_snapshot = {
        "mode": "tournament",
        "episodes_per_matchup": episodes,
        "cycles_per_episode": cycles_per_episode,
        "seed": seed,
        "run_id": run_id,
        "strategies": strategy_labels,
    }
    logger.start_run(run_id=run_id, seed=seed, config=config_snapshot)

    if verbose:
        print(f"=== Starting Head-to-Head Tournament [{run_id}] ===")
        print(f"Strategies ({len(strategies)}): {strategy_labels}")
        print(f"Episodes/Matchup: {episodes} | Cycles/Episode: {cycles_per_episode} | Seed: {seed}\n")

    try:
        # Build empirical payoff matrices via round-robin head-to-head simulations
        A, B, SE_A, SE_B, labels = PayoffBuilder.build_tournament_matrix(
            strategies=strategies,
            background_agents=background,
            n_episodes=episodes,
            cycles_per_episode=cycles_per_episode,
            seed=seed,
        )

        # Solve for Nash equilibria via support enumeration
        solver = NashSolver(A, B)
        eq_candidates = solver.get_equilibrium_candidates(
            method="support_enumeration",
            strategy_labels=labels,
        )

        # Trace best response dynamics
        brd = BestResponseDynamics(A, B)
        trace = brd.trace(max_iterations=20)

        # Log equilibrium analysis with per-cell standard errors
        logger.log_equilibrium_analysis(
            run_id=run_id,
            payoff_matrix=A,
            equilibrium_candidates=eq_candidates,
            method="support_enumeration",
            strategies=labels,
            best_response_trace=trace,
            standard_errors=SE_A,
        )

        logger.finish_run(status="COMPLETE")

        if verbose:
            print("--- Payoff Matrix A (Player X Mean PnL) ---")
            header = "          " + "  ".join(f"{lbl:>10}" for lbl in labels)
            print(header)
            for i, row_lbl in enumerate(labels):
                row_str = f"{row_lbl:>10}" + "  ".join(f"{A[i, j]:>10.2f}" for j in range(len(labels)))
                print(row_str)

            print("\n--- Standard Errors (Player X) ---")
            print(header)
            for i, row_lbl in enumerate(labels):
                row_str = f"{row_lbl:>10}" + "  ".join(f"{SE_A[i, j]:>10.2f}" for j in range(len(labels)))
                print(row_str)

            print(f"\nNash Equilibrium Candidates Found: {len(eq_candidates)}")
            for cand in eq_candidates:
                p1_s = ", ".join(f"{labels[k]}: {p:.2f}" for k, p in enumerate(cand["player_1_strategy"]) if p > 0.01)
                p2_s = ", ".join(f"{labels[k]}: {p:.2f}" for k, p in enumerate(cand["player_2_strategy"]) if p > 0.01)
                pure_tag = "Pure" if cand.get("is_pure") else "Mixed"
                print(f"  #{cand['equilibrium_id']} [{pure_tag}]: P1({p1_s}) vs P2({p2_s}) | Payoffs: ({cand['player_1_payoff']:.2f}, {cand['player_2_payoff']:.2f})")

            print(f"\nResults successfully saved to {output_dir}/{run_id}/equilibrium_analysis.json")

    except Exception as e:
        logger.finish_run(status="SIMULATION_INVALID", error_message=str(e))
        if verbose:
            print(f"\nTournament failed: {e}")
        raise e

    return logger


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MultiAgentMarketSim Tournament Runner")
    parser.add_argument("--episodes", type=int, default=5, help="Episodes per pairwise cell")
    parser.add_argument("--cycles", type=int, default=50, help="Cycles per episode")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed")
    parser.add_argument("--run-id", type=str, default="tournament_demo_001", help="Run identifier")
    parser.add_argument("--output-dir", type=str, default="data/runs", help="Output directory")
    args = parser.parse_args()

    run_tournament(
        episodes=args.episodes,
        cycles_per_episode=args.cycles,
        seed=args.seed,
        run_id=args.run_id,
        output_dir=args.output_dir,
    )
