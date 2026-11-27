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
    """Returns the four core tournament strategies: ZI, tight MM, wide MM, RL (pretrained)."""
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
            "label": "RL (pretrained)",
            "type": "RL",
            "kwargs": {"alpha": 0.0, "epsilon": 0.0, "order_quantity": 10},
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
    episodes: int = 30,
    cycles_per_episode: int = 200,
    seed: int = 42,
    run_id: str = "tournament_demo_001",
    output_dir: str = "data/runs",
    verbose: bool = True,
    pretrain_episodes: int = 100,
    pretrain_cycles: int = 200,
    only_equilibrium: bool = False,
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
        "episodes_per_cell": episodes,
        "cycles_per_episode": cycles_per_episode,
        "seed": seed,
        "run_id": run_id,
        "strategies": strategy_labels,
    }
    if not only_equilibrium:
        logger.start_run(run_id=run_id, seed=seed, config=config_snapshot)

    pretrain_settings = {
        "episodes": pretrain_episodes,
        "cycles": pretrain_cycles,
        "alpha": 0.1,
        "order_quantity": 10,
        "max_position": 100,
        "price_band": 0.10,
        "order_ttl": 20,
    }

    if verbose:
        print(f"=== Starting Head-to-Head Tournament [{run_id}] ===")
        print(f"Strategies ({len(strategies)}): {strategy_labels}")
        print(f"Episodes/Cell: {episodes} | Cycles/Episode: {cycles_per_episode} | Seed: {seed}")
        print(f"RL Pretrain Settings: {pretrain_settings}\n")

    try:
        # Build empirical payoff matrices via round-robin head-to-head simulations
        A, B, SE_A, SE_B, labels = PayoffBuilder.build_tournament_matrix(
            strategies=strategies,
            background_agents=background,
            n_episodes=episodes,
            cycles_per_episode=cycles_per_episode,
            seed=seed,
            pretrain_rl=True,
            pretrain_episodes=pretrain_episodes,
            pretrain_cycles=pretrain_cycles,
        )

        # Solve for Nash equilibria via support enumeration with combined SE deviation robustness checks
        solver = NashSolver(A, B)
        eq_candidates = solver.get_equilibrium_candidates(
            method="support_enumeration",
            strategy_labels=labels,
            SE_A=SE_A,
            SE_B=SE_B,
        )

        # Trace best response dynamics
        brd = BestResponseDynamics(A, B)
        trace = brd.trace(max_iterations=20)

        # Log equilibrium analysis with per-cell standard errors and metadata
        logger.log_equilibrium_analysis(
            run_id=run_id,
            payoff_matrix=A,
            equilibrium_candidates=eq_candidates,
            method="support_enumeration",
            strategies=labels,
            best_response_trace=trace,
            standard_errors=SE_A,
            payoff_matrix_b=B,
            standard_errors_b=SE_B,
            episodes_per_cell=episodes,
            cycles=cycles_per_episode,
            pretrain_settings=pretrain_settings,
            seed=seed,
        )

        if only_equilibrium:
            logger.save_equilibrium_only(run_id=run_id)
        else:
            logger.finish_run(status="COMPLETE")

        if verbose:
            print("--- Payoff Matrix A (Player X Mean PnL) ---")
            header = "          " + "  ".join(f"{lbl:>16}" for lbl in labels)
            print(header)
            for i, row_lbl in enumerate(labels):
                row_str = f"{row_lbl:>16}" + "  ".join(f"{A[i, j]:>16.2f}" for j in range(len(labels)))
                print(row_str)

            print("\n--- Standard Errors (Player X) ---")
            print(header)
            for i, row_lbl in enumerate(labels):
                row_str = f"{row_lbl:>16}" + "  ".join(f"{SE_A[i, j]:>16.2f}" for j in range(len(labels)))
                print(row_str)

            print(f"\nNash Equilibrium Candidates Found: {len(eq_candidates)}")
            for cand in eq_candidates:
                p1_s = ", ".join(f"{labels[k]}: {p:.2f}" for k, p in enumerate(cand["player_1_strategy"]) if p > 0.01)
                p2_s = ", ".join(f"{labels[k]}: {p:.2f}" for k, p in enumerate(cand["player_2_strategy"]) if p > 0.01)
                pure_tag = "Pure" if cand.get("is_pure") else "Mixed"
                weak_tag = "Weak" if cand.get("is_weak") else "Strong"
                m1 = cand.get("p1_deviation_margin", 0.0)
                se1 = cand.get("p1_combined_se", 0.0)
                m2 = cand.get("p2_deviation_margin", 0.0)
                se2 = cand.get("p2_combined_se", 0.0)
                print(
                    f"  #{cand['equilibrium_id']} [{pure_tag}, {weak_tag}]: "
                    f"P1({p1_s}) vs P2({p2_s}) | "
                    f"Payoffs: ({cand['player_1_payoff']:.2f}, {cand['player_2_payoff']:.2f}) | "
                    f"Margins vs 1 SE: P1(${m1:.2f} vs ${se1:.2f}), P2(${m2:.2f} vs ${se2:.2f})"
                )

            print(f"\nResults successfully saved to {output_dir}/{run_id}/equilibrium_analysis.json")

    except Exception as e:
        if not only_equilibrium:
            logger.finish_run(status="SIMULATION_INVALID", error_message=str(e))
        if verbose:
            print(f"\nTournament failed: {e}")
        raise e

    return logger


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MultiAgentMarketSim Tournament Runner")
    parser.add_argument("--episodes", type=int, default=30, help="Episodes per pairwise cell (default: 30)")
    parser.add_argument("--cycles", type=int, default=200, help="Cycles per episode (default: 200)")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed (default: 42)")
    parser.add_argument("--pretrain-episodes", type=int, default=100, help="RL pretraining episodes (default: 100)")
    parser.add_argument("--pretrain-cycles", type=int, default=200, help="RL pretraining cycles (default: 200)")
    parser.add_argument("--only-equilibrium", action="store_true", help="Only write equilibrium_analysis.json without modifying other run files")
    parser.add_argument("--run-id", type=str, default="tournament_demo_001", help="Run identifier")
    parser.add_argument("--output-dir", type=str, default="data/runs", help="Output directory")
    args = parser.parse_args()

    run_tournament(
        episodes=args.episodes,
        cycles_per_episode=args.cycles,
        seed=args.seed,
        run_id=args.run_id,
        output_dir=args.output_dir,
        pretrain_episodes=args.pretrain_episodes,
        pretrain_cycles=args.pretrain_cycles,
        only_equilibrium=args.only_equilibrium,
    )
