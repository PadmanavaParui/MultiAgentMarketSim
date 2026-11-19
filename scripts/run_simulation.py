"""
Main simulation and RL training runner for MultiAgentMarketSim.

Supports the multi-stage training curriculum outlined in the OJT documentation:
  Stage 1: 'baselines' — Non-learning Zero-Intelligence + Market-Maker validation.
  Stage 2: 'rl_vs_baseline' — Single learning RL agent against frozen baseline agents (TEST-006).
  Stage 3: 'multi_rl' — Multiple concurrent RL directional agents interacting with baselines.
  Stage 4: 'full' — Executes curriculum stages and persists logs for post-hoc analysis.

Usage:
    python scripts/run_simulation.py --mode rl_vs_baseline --episodes 10 --cycles 50 --seed 42
"""

from __future__ import annotations

import os
import sys
import argparse
from typing import Dict, Any, List

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.env.market_env import MarketEnvironment
from src.agents.zero_intelligence import ZeroIntelligenceAgent
from src.agents.market_maker import MarketMakerAgent
from src.agents.rl_directional import RLDirectionalAgent
from src.metrics.logger import MetricsLogger
from src.metrics.pnl import compute_agent_metrics


def run_simulation(
    mode: str = "rl_vs_baseline",
    episodes: int = 10,
    cycles_per_episode: int = 50,
    seed: int = 42,
    run_id: str = "sim_run",
    output_dir: str = "data/runs",
    verbose: bool = True,
) -> MetricsLogger:
    """
    Executes a multi-episode market simulation run and logs all records.
    """
    logger = MetricsLogger(base_dir=output_dir)
    config_snapshot = {
        "mode": mode,
        "episodes": episodes,
        "cycles_per_episode": cycles_per_episode,
        "seed": seed,
        "run_id": run_id,
    }
    logger.start_run(run_id=run_id, seed=seed, config=config_snapshot)

    # Agent setup based on mode
    if mode == "baselines":
        agent_configs = {
            "mm_0": {"type": "MM", "spread": 0.5, "quantity": 10},
            "zi_0": {"type": "ZI", "aggression": 0.8, "max_quantity": 15},
            "zi_1": {"type": "ZI", "aggression": 0.6, "max_quantity": 10},
        }
    elif mode == "multi_rl":
        agent_configs = {
            "rl_0": {"type": "RL", "alpha": 0.1, "epsilon": 0.5},
            "rl_1": {"type": "RL", "alpha": 0.1, "epsilon": 0.5},
            "mm_0": {"type": "MM", "spread": 0.5, "quantity": 10},
            "zi_0": {"type": "ZI", "aggression": 0.7, "max_quantity": 10},
        }
    else:  # "rl_vs_baseline" or default
        agent_configs = {
            "rl_0": {"type": "RL", "alpha": 0.1, "epsilon": 0.5},
            "mm_0": {"type": "MM", "spread": 0.5, "quantity": 10},
            "zi_0": {"type": "ZI", "aggression": 0.7, "max_quantity": 10},
        }

    agent_ids = list(agent_configs.keys())
    env = MarketEnvironment(
        agents=agent_ids,
        max_cycles=cycles_per_episode,
        dict_spaces=True,
        run_id=run_id,
    )

    # Instantiate agent policy instances
    agents = {}
    for i, (a_id, cfg) in enumerate(agent_configs.items()):
        agent_seed = (seed + i * 100) if seed is not None else None
        if cfg["type"] == "MM":
            agents[a_id] = MarketMakerAgent(
                agent_id=a_id,
                seed=agent_seed,
                spread=cfg.get("spread", 1.0),
                quantity=cfg.get("quantity", 10),
            )
        elif cfg["type"] == "ZI":
            agents[a_id] = ZeroIntelligenceAgent(
                agent_id=a_id,
                seed=agent_seed,
                aggression=cfg.get("aggression", 0.7),
                max_quantity=cfg.get("max_quantity", 15),
            )
        elif cfg["type"] == "RL":
            agents[a_id] = RLDirectionalAgent(
                agent_id=a_id,
                seed=agent_seed,
                alpha=cfg.get("alpha", 0.1),
                epsilon=cfg.get("epsilon", 0.5),
            )

    if verbose:
        print(f"=== Starting Simulation Run [{run_id}] ===")
        print(f"Mode: {mode} | Episodes: {episodes} | Cycles/Ep: {cycles_per_episode} | Seed: {seed}")
        print(f"Agents: {list(agents.keys())}\n")

    try:
        for ep in range(episodes):
            ep_seed = (seed + ep * 1000) if seed is not None else None
            obs, infos = env.reset(seed=ep_seed)

            # Reset episode tracking for agents
            for a in agents.values():
                a.reset()

            ep_rewards: Dict[str, List[float]] = {a: [] for a in agent_ids}
            ep_portfolios: Dict[str, List[float]] = {a: [] for a in agent_ids}
            ep_inventories: Dict[str, List[int]] = {a: [] for a in agent_ids}

            step = 0
            while env.agents:
                step += 1
                actions = {a_id: agents[a_id].act(obs[a_id]) for a_id in env.agents if a_id in obs}

                # Step the market environment
                next_obs, rewards, terminations, truncations, step_infos = env.step(actions)

                # Process rewards and agent learning updates
                for a_id in agent_ids:
                    if a_id in rewards:
                        r = rewards[a_id]
                        ep_rewards[a_id].append(r)
                        if a_id in next_obs:
                            agents[a_id].update(r, next_obs[a_id])

                    if a_id in step_infos:
                        ep_portfolios[a_id].append(step_infos[a_id]["portfolio_value"])
                        ep_inventories[a_id].append(step_infos[a_id]["inventory"])

                obs = next_obs

            # Log trades produced in this episode
            for trade in env.matching_engine.trades_history:
                logger.log_trade(
                    trade_id=trade.trade_id,
                    run_id=trade.run_id,
                    buy_order_id=trade.buy_order_id,
                    sell_order_id=trade.sell_order_id,
                    price=trade.price,
                    quantity=trade.quantity,
                    timestamp_step=trade.timestamp_step,
                )

            # Log end-of-episode agent metrics
            ref_price = env._get_reference_price()
            for a_id in agent_ids:
                m = compute_agent_metrics(
                    rewards=ep_rewards[a_id],
                    portfolio_values=ep_portfolios[a_id],
                    inventory_history=ep_inventories[a_id],
                    initial_cash=env._initial_cash[a_id],
                    final_cash=env.cash[a_id],
                    final_inventory=env.inventory[a_id],
                    reference_price=ref_price,
                )
                logger.log_agent_metric(
                    run_id=run_id,
                    agent_id=a_id,
                    episode=ep + 1,
                    pnl=m["pnl"],
                    sharpe_ratio=m["sharpe"],
                    inventory=env.inventory[a_id],
                    reward=sum(ep_rewards[a_id]),
                )

            if verbose and (ep + 1) % max(1, episodes // 5) == 0:
                print(f"Episode {ep + 1}/{episodes} finished. Reference Price: ${ref_price:.2f}")
                for a_id in agent_ids:
                    print(f"  {a_id}: Cash=${env.cash[a_id]:.2f}, Inv={env.inventory[a_id]}, Total PnL=${(env.cash[a_id] + env.inventory[a_id] * ref_price - env._initial_cash[a_id]):.2f}")

        # Post-hoc game-theoretic analysis (OJT doc Step 8 / Phase 3)
        try:
            from src.gametheory.payoff_builder import PayoffBuilder
            from src.gametheory.nash_solver import NashSolver
            from src.gametheory.best_response import BestResponseDynamics

            archetype_pnls: Dict[str, List[float]] = {}
            for m in logger.agent_metrics:
                a_id = m["agent_id"]
                arch = agent_configs.get(a_id, {}).get("type", a_id)
                archetype_pnls.setdefault(arch, []).append(m["pnl"])

            if len(archetype_pnls) >= 2:
                A, B, strats = PayoffBuilder.build_archetype_matrix(archetype_pnls)
                solver = NashSolver(A, B)
                eq_candidates = solver.get_equilibrium_candidates(method="support_enumeration")
                brd = BestResponseDynamics(A, B)
                trace = brd.trace(max_iterations=20)
                logger.log_equilibrium_analysis(
                    run_id=run_id,
                    payoff_matrix=A,
                    equilibrium_candidates=eq_candidates,
                    strategies=strats,
                    best_response_trace=trace,
                )
                if verbose:
                    print(f"\n[Game Theory] Computed {len(eq_candidates)} Nash equilibrium candidate(s).")
        except Exception as gt_err:
            if verbose:
                print(f"[Game Theory] Post-hoc analysis notice: {gt_err}")

        logger.finish_run(status="COMPLETE")
        if verbose:
            print(f"\nRun [{run_id}] completed successfully. Saved to {output_dir}/{run_id}")

    except Exception as e:
        logger.finish_run(status="SIMULATION_INVALID", error_message=str(e))
        if verbose:
            print(f"\nRun [{run_id}] failed: {e}")
        raise e

    return logger


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MultiAgentMarketSim Simulation Runner")
    parser.add_argument("--mode", type=str, default="rl_vs_baseline", choices=["baselines", "rl_vs_baseline", "multi_rl"])
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--cycles", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run-id", type=str, default="demo_run_001")
    parser.add_argument("--output-dir", type=str, default="data/runs")
    args = parser.parse_args()

    run_simulation(
        mode=args.mode,
        episodes=args.episodes,
        cycles_per_episode=args.cycles,
        seed=args.seed,
        run_id=args.run_id,
        output_dir=args.output_dir,
    )
