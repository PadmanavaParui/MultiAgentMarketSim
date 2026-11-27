"""
Payoff Matrix Builder for MultiAgentMarketSim.

Constructs 2-player normal-form payoff matrices (A, B) from logged simulation
records (trades, agent metrics, or head-to-head empirical evaluations).
Compatible with Nashpy games and game-theoretic equilibrium analysis.
"""

from __future__ import annotations

from typing import List, Dict, Any, Tuple, Optional
import numpy as np

from src.env.market_env import MarketEnvironment
from src.agents.zero_intelligence import ZeroIntelligenceAgent
from src.agents.market_maker import MarketMakerAgent
from src.agents.rl_directional import RLDirectionalAgent
from src.agents.avellaneda_stoikov import AvellanedaStoikovMM
from src.agents.informed_trader import InformedTrader
from src.agents.shock_agent import ShockAgent
from src.metrics.pnl import compute_agent_metrics


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
    def mean_pnl_ranking(
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
    def _make_agent(agent_id: str, strategy: Dict[str, Any], seed: int):
        """Instantiate an agent from a strategy descriptor dict."""
        stype = strategy["type"]
        kw = dict(strategy.get("kwargs", {}))
        if stype in ("ZI", "ZeroIntelligenceAgent"):
            return ZeroIntelligenceAgent(agent_id=agent_id, seed=seed, **kw)
        elif stype in ("MM", "MarketMakerAgent"):
            return MarketMakerAgent(agent_id=agent_id, seed=seed, **kw)
        elif stype in ("RL", "RLDirectionalAgent"):
            return RLDirectionalAgent(agent_id=agent_id, seed=seed, **kw)
        elif stype in ("AvellanedaStoikovMM", "AS_MM", "AS"):
            return AvellanedaStoikovMM(agent_id=agent_id, seed=seed, **kw)
        elif stype in ("InformedTrader", "Informed"):
            return InformedTrader(agent_id=agent_id, seed=seed, **kw)
        elif stype in ("ShockAgent", "Shock"):
            return ShockAgent(agent_id=agent_id, seed=seed, **kw)
        else:
            raise ValueError(f"Unknown strategy type: {stype}")

    @staticmethod
    def pretrain_rl_agent(
        episodes: int = 100,
        cycles_per_episode: int = 200,
        seed: int = 42,
        alpha: float = 0.1,
        epsilon: float = 1.0,
        epsilon_min: float = 0.05,
        epsilon_decay: float = 0.995,
        order_quantity: int = 10,
        max_position: int = 100,
        price_band: float = 0.10,
        order_ttl: int = 20,
    ) -> RLDirectionalAgent:
        """
        Pretrains an RLDirectionalAgent against standard baseline agents (MM + ZI)
        with risk guardrails active (max_position, price_band, order_ttl).
        Returns a frozen copy with epsilon=0 and updates disabled.
        """
        rl_agent = RLDirectionalAgent(
            agent_id="rl_pretrain",
            seed=seed,
            alpha=alpha,
            epsilon=epsilon,
            epsilon_min=epsilon_min,
            epsilon_decay=epsilon_decay,
            order_quantity=order_quantity,
        )

        agent_ids = ["rl_pretrain", "mm_0", "zi_0"]
        env = MarketEnvironment(
            agents=agent_ids,
            max_cycles=cycles_per_episode,
            dict_spaces=True,
            run_id="pretrain_rl",
            max_position=max_position,
            price_band=price_band,
            order_ttl=order_ttl,
        )

        for ep in range(episodes):
            ep_seed = seed + ep * 1000
            obs, _ = env.reset(seed=ep_seed)
            rl_agent.reset()

            mm_agent = MarketMakerAgent(
                agent_id="mm_0",
                seed=ep_seed + 101,
                spread=0.5,
                quantity=10,
            )
            zi_agent = ZeroIntelligenceAgent(
                agent_id="zi_0",
                seed=ep_seed + 202,
                aggression=0.7,
                max_quantity=10,
            )
            agents = {
                "rl_pretrain": rl_agent,
                "mm_0": mm_agent,
                "zi_0": zi_agent,
            }

            while env.agents:
                actions = {
                    a_id: agents[a_id].act(obs[a_id])
                    for a_id in env.agents
                    if a_id in obs
                }
                next_obs, rewards, _, _, _ = env.step(actions)
                if "rl_pretrain" in rewards and "rl_pretrain" in next_obs:
                    rl_agent.update(rewards["rl_pretrain"], next_obs["rl_pretrain"])
                obs = next_obs

        # Return frozen clone with epsilon=0 and updates disabled
        return rl_agent.clone(seed=seed, freeze=True)

    @staticmethod
    def build_tournament_matrix(
        strategies: List[Dict[str, Any]],
        background_agents: Optional[List[Dict[str, Any]]] = None,
        n_episodes: int = 30,
        cycles_per_episode: int = 200,
        seed: int = 42,
        pretrain_rl: bool = True,
        pretrain_episodes: int = 100,
        pretrain_cycles: int = 200,
        pretrained_rl_agent: Optional[RLDirectionalAgent] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
        """
        Construct payoff matrices via round-robin head-to-head tournament.

        For every ordered pair (i, j) of strategies, runs ``n_episodes``
        seeded episodes where Player X plays strategy *i* and Player Y
        plays strategy *j*, with a fixed identical background of other agents.
        """
        n = len(strategies)
        if n < 2:
            labels = [s.get("label", f"S{i}") for i, s in enumerate(strategies)]
            z = np.zeros((max(n, 2), max(n, 2)))
            return z, z.copy(), z.copy(), z.copy(), labels

        bg = background_agents or []

        # Pretrain RL agent once if RL is included in the tournament roster
        has_rl = any(
            s.get("type") in ("RL", "RLDirectionalAgent") or "RL" in s.get("label", "")
            for s in strategies
        )
        if has_rl and pretrained_rl_agent is None and pretrain_rl:
            pretrained_rl_agent = PayoffBuilder.pretrain_rl_agent(
                episodes=pretrain_episodes,
                cycles_per_episode=pretrain_cycles,
                seed=seed,
            )

        labels = []
        for i, s in enumerate(strategies):
            lbl = s.get("label", f"S{i}")
            if pretrained_rl_agent is not None and (
                s.get("type") in ("RL", "RLDirectionalAgent") or lbl in ("RL", "RL (pretrained)")
            ):
                lbl = "RL (pretrained)"
            labels.append(lbl)

        A = np.zeros((n, n))
        B = np.zeros((n, n))
        SE_A = np.zeros((n, n))
        SE_B = np.zeros((n, n))

        def _get_player_agent(player_id: str, strat_idx: int, a_seed: int):
            strat = strategies[strat_idx]
            stype = strat.get("type")
            is_rl = (stype in ("RL", "RLDirectionalAgent")) or ("RL" in strat.get("label", ""))
            if is_rl and pretrained_rl_agent is not None:
                clone_agent = pretrained_rl_agent.clone(seed=a_seed, freeze=True)
                clone_agent.agent_id = player_id
                return clone_agent
            return PayoffBuilder._make_agent(player_id, strat, seed=a_seed)

        for i in range(n):
            for j in range(n):
                x_pnls: List[float] = []
                y_pnls: List[float] = []

                for ep in range(n_episodes):
                    # Deterministic per-cell, per-episode seed
                    cell_seed = seed + i * 10000 + j * 1000 + ep * 100

                    # Build agent roster: player_x, player_y, plus background
                    all_ids = ["player_x", "player_y"]
                    bg_ids = [f"bg_{k}" for k in range(len(bg))]
                    all_ids.extend(bg_ids)

                    env = MarketEnvironment(
                        agents=all_ids,
                        max_cycles=cycles_per_episode,
                        dict_spaces=True,
                        run_id=f"tourney_{i}_{j}_{ep}",
                    )

                    agents_map = {}
                    agents_map["player_x"] = _get_player_agent("player_x", i, cell_seed)
                    agents_map["player_y"] = _get_player_agent("player_y", j, cell_seed + 1)
                    for k_idx, bg_cfg in enumerate(bg):
                        bg_id = f"bg_{k_idx}"
                        agents_map[bg_id] = PayoffBuilder._make_agent(
                            bg_id, bg_cfg, seed=cell_seed + 2 + k_idx,
                        )

                    # Run one episode
                    obs, _ = env.reset(seed=cell_seed)
                    for a in agents_map.values():
                        a.reset()

                    while env.agents:
                        actions = {
                            a_id: agents_map[a_id].act(obs[a_id])
                            for a_id in env.agents
                            if a_id in obs
                        }
                        next_obs, rewards, _, _, _ = env.step(actions)
                        for a_id in env.agents:
                            if a_id in rewards and a_id in next_obs:
                                agents_map[a_id].update(rewards[a_id], next_obs[a_id])
                        obs = next_obs

                    # Compute PnL for the two focal players
                    ref = env._get_reference_price()
                    for focal_id, pnl_list in [("player_x", x_pnls), ("player_y", y_pnls)]:
                        pv = env.cash[focal_id] + env.inventory[focal_id] * ref
                        pnl_list.append(pv - env._initial_cash[focal_id])

                x_arr = np.array(x_pnls)
                y_arr = np.array(y_pnls)
                A[i, j] = float(np.mean(x_arr))
                B[i, j] = float(np.mean(y_arr))
                SE_A[i, j] = float(np.std(x_arr, ddof=1) / np.sqrt(n_episodes)) if n_episodes > 1 else 0.0
                SE_B[i, j] = float(np.std(y_arr, ddof=1) / np.sqrt(n_episodes)) if n_episodes > 1 else 0.0

        return A, B, SE_A, SE_B, labels

    @staticmethod
    def build_symmetric_matrix(values_2x2: List[List[float]]) -> Tuple[np.ndarray, np.ndarray]:
        """Utility to convert a 2x2 list of payoffs into (A, B) numpy arrays."""
        A = np.array(values_2x2, dtype=np.float64)
        B = A.T  # Symmetric game
        return A, B
