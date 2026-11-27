"""
RL Directional Trader Agent — Tabular Q-Learning.

Implements a lightweight Q-learning agent that learns a directional trading
policy from PnL-based rewards.  Follows the shared AgentPolicy interface
(``act`` / ``update``).

Design notes (ADR-003, TRD):
  • The doc specifies "Ray RLlib (PPO/Q-learning) policy wrapper", but RLlib is
    a very heavy dependency and GPU is optional.  We implement a standalone
    tabular Q-learning agent that captures the same Bellman-style update
    logic described in the RL architecture section.
  • The observation space is discretized into coarse state buckets so that a
    Q-table is feasible without function approximation.
  • Epsilon-greedy exploration with linear decay.
  • Full reward/loss history is stored for convergence diagnostics.

Bellman update (doc §10, Core Algorithms):
    Q(s,a) ← Q(s,a) + α [ R + γ · max_a' Q(s',a') − Q(s,a) ]
"""

from __future__ import annotations

import numpy as np
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple


# ------------------------------------------------------------------
# Action encoding
# ------------------------------------------------------------------
# 0 = HOLD
# 1 = BUY  LIMIT  (small offset)
# 2 = SELL LIMIT  (small offset)
# 3 = BUY  MARKET
# 4 = SELL MARKET
NUM_ACTIONS = 5

# Maps discrete action index → environment action dict
_ACTION_MAP: List[Dict[str, Any]] = [
    {"order_type": 0, "side": 0, "price_offset": 0.0,  "quantity": 0},    # HOLD
    {"order_type": 1, "side": 0, "price_offset": -0.5, "quantity": 10},   # BUY LIMIT
    {"order_type": 1, "side": 1, "price_offset": 0.5,  "quantity": 10},   # SELL LIMIT
    {"order_type": 2, "side": 0, "price_offset": 0.0,  "quantity": 10},   # BUY MARKET
    {"order_type": 2, "side": 1, "price_offset": 0.0,  "quantity": 10},   # SELL MARKET
]


class RLDirectionalAgent:
    """
    Tabular Q-learning RL directional trader.

    Parameters
    ----------
    agent_id : str
        Unique identifier.
    seed : int | None
        Random seed for reproducibility.
    alpha : float
        Learning rate (step size).
    gamma : float
        Discount factor.
    epsilon : float
        Initial exploration probability (ε-greedy).
    epsilon_min : float
        Minimum exploration probability.
    epsilon_decay : float
        Multiplicative decay applied to epsilon after each ``update()``.
    order_quantity : int
        Fixed quantity for every order the agent places.
    """

    def __init__(
        self,
        agent_id: str,
        seed: Optional[int] = None,
        alpha: float = 0.1,
        gamma: float = 0.99,
        epsilon: float = 1.0,
        epsilon_min: float = 0.05,
        epsilon_decay: float = 0.995,
        order_quantity: int = 10,
        decay_per_episode: bool = False,
        epsilon_decay_per_episode: Optional[bool] = None,
        frozen: bool = False,
    ):
        self.agent_id = agent_id
        self.alpha = 0.0 if frozen else alpha
        self.gamma = gamma
        self.epsilon = 0.0 if frozen else epsilon
        self.epsilon_min = 0.0 if frozen else epsilon_min
        self.epsilon_decay = 1.0 if frozen else epsilon_decay
        self.order_quantity = order_quantity
        self.decay_per_episode = (
            bool(epsilon_decay_per_episode)
            if epsilon_decay_per_episode is not None
            else bool(decay_per_episode)
        ) and not frozen
        self.frozen = bool(frozen)
        self.rng = np.random.default_rng(seed)

        # Track steps taken within the current episode
        self._episode_steps: int = 0

        # Q-table: state_key → np.array of shape (NUM_ACTIONS,)
        self.q_table: Dict[Tuple, np.ndarray] = defaultdict(
            lambda: np.zeros(NUM_ACTIONS, dtype=np.float64)
        )

        # Bookkeeping for convergence diagnostics
        self.reward_history: List[float] = []
        self.td_error_history: List[float] = []
        self.epsilon_history: List[float] = []

        # Previous state-action for SARSA-style delayed update
        self._prev_state: Optional[Tuple] = None
        self._prev_action: Optional[int] = None

    # ------------------------------------------------------------------
    # AgentPolicy interface
    # ------------------------------------------------------------------

    def act(self, observation: Any) -> Dict[str, Any]:
        """Select an action via ε-greedy over the Q-table."""
        self._episode_steps += 1
        state = self._discretize(observation)
        self._prev_state = state
        # Ensure state entry exists in Q-table
        q_values = self.q_table[state]

        if not self.frozen and self.rng.random() < self.epsilon:
            action_idx = int(self.rng.integers(0, NUM_ACTIONS))
        else:
            action_idx = int(np.argmax(q_values))

        self._prev_action = action_idx

        # Build env action dict, overriding quantity with the agent's setting
        action = dict(_ACTION_MAP[action_idx])
        if action["order_type"] != 0:
            action["quantity"] = self.order_quantity
        return action

    def update(self, reward: float, next_observation: Any) -> None:
        """
        Bellman-style Q-learning update.

        Q(s,a) ← Q(s,a) + α [ R + γ · max_a' Q(s',a') − Q(s,a) ]
        """
        if self.frozen or self.alpha == 0.0:
            return

        if self._prev_state is None or self._prev_action is None:
            return

        s = self._prev_state
        a = self._prev_action
        s_next = self._discretize(next_observation)

        old_q = self.q_table[s][a]
        best_next_q = float(np.max(self.q_table[s_next]))
        td_target = reward + self.gamma * best_next_q
        td_error = td_target - old_q

        self.q_table[s][a] = old_q + self.alpha * td_error

        # Bookkeeping
        self.reward_history.append(reward)
        self.td_error_history.append(abs(td_error))
        self.epsilon_history.append(self.epsilon)

        # Decay epsilon only if step-wise decay is active
        if not self.decay_per_episode:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def decay_epsilon(self) -> float:
        """
        Decay exploration rate epsilon by factor self.epsilon_decay down to self.epsilon_min.
        Returns the updated epsilon value.
        """
        if self.frozen or self.epsilon <= 0.0:
            return 0.0
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        return self.epsilon

    def reset(self, seed: Optional[int] = None, decay_epsilon: Optional[bool] = None) -> None:
        """
        Reset per-episode state (keeps the learned Q-table).

        Parameters
        ----------
        seed : int | None
            Optional new random seed for the episode.
        decay_epsilon : bool | None
            If True, decay epsilon immediately. If None and decay_per_episode is True,
            epsilon decays if the agent took steps in the preceding episode.
        """
        should_decay = (
            not self.frozen
            and (
                decay_epsilon
                if decay_epsilon is not None
                else (self.decay_per_episode and self._episode_steps > 0)
            )
        )
        if should_decay:
            self.decay_epsilon()

        self._prev_state = None
        self._prev_action = None
        self._episode_steps = 0
        if seed is not None:
            self.rng = np.random.default_rng(seed)

    def clone(self, seed: Optional[int] = None, freeze: bool = True) -> RLDirectionalAgent:
        """
        Creates a copy of this RL agent with its learned Q-table.
        If freeze is True, sets epsilon=0.0, alpha=0.0, and frozen=True.
        """
        new_agent = RLDirectionalAgent(
            agent_id=self.agent_id,
            seed=seed,
            alpha=0.0 if freeze else self.alpha,
            gamma=self.gamma,
            epsilon=0.0 if freeze else self.epsilon,
            epsilon_min=0.0 if freeze else self.epsilon_min,
            epsilon_decay=1.0 if freeze else self.epsilon_decay,
            order_quantity=self.order_quantity,
            decay_per_episode=False if freeze else self.decay_per_episode,
            frozen=freeze,
        )
        new_agent.q_table = defaultdict(
            lambda: np.zeros(NUM_ACTIONS, dtype=np.float64),
            {k: np.copy(v) for k, v in self.q_table.items()}
        )
        return new_agent

    # ------------------------------------------------------------------
    # State discretization
    # ------------------------------------------------------------------

    def _discretize(self, observation: Any) -> Tuple:
        """
        Convert a continuous observation into a coarse discrete state tuple.

        State features (all bucketed):
            1. price_trend  — sign of (best_bid − initial_price approximation)
            2. spread_bucket — narrow / medium / wide / no-spread
            3. inventory_bucket — large-short / short / flat / long / large-long
            4. cash_bucket — low / medium / high
        """
        if isinstance(observation, dict):
            bids = np.asarray(observation.get("bids", np.zeros((1, 2))))
            asks = np.asarray(observation.get("asks", np.zeros((1, 2))))
            cash = float(np.asarray(observation.get("cash", [0.0])).flatten()[0])
            inventory = float(np.asarray(observation.get("inventory", [0.0])).flatten()[0])
        elif isinstance(observation, np.ndarray):
            obs = observation.flatten()
            # Flat layout: [bid_p, bid_v, ..., ask_p, ask_v, ..., cash, inv]
            k = max(1, (len(obs) - 2) // 4)
            bids = obs[:2 * k].reshape(k, 2)
            asks = obs[2 * k:4 * k].reshape(k, 2)
            cash = float(obs[-2])
            inventory = float(obs[-1])
        else:
            return (0, 0, 0, 0)

        # 1. Price trend: sign of best_bid price vs 100 (approximate reference)
        best_bid_price = float(bids[0, 0]) if bids[0, 0] > 0 else 0.0
        best_ask_price = float(asks[0, 0]) if asks[0, 0] > 0 else 0.0

        if best_bid_price > 0 and best_ask_price > 0:
            mid = (best_bid_price + best_ask_price) / 2.0
        elif best_bid_price > 0:
            mid = best_bid_price
        elif best_ask_price > 0:
            mid = best_ask_price
        else:
            mid = 100.0

        price_trend = 0 if mid < 99.5 else (2 if mid > 100.5 else 1)

        # 2. Spread bucket
        if best_bid_price > 0 and best_ask_price > 0:
            spread = best_ask_price - best_bid_price
            if spread < 0.5:
                spread_bucket = 0  # narrow
            elif spread < 2.0:
                spread_bucket = 1  # medium
            else:
                spread_bucket = 2  # wide
        else:
            spread_bucket = 3  # no spread (one side empty)

        # 3. Inventory bucket
        if inventory < -20:
            inv_bucket = 0   # large short
        elif inventory < -5:
            inv_bucket = 1   # short
        elif inventory <= 5:
            inv_bucket = 2   # flat
        elif inventory <= 20:
            inv_bucket = 3   # long
        else:
            inv_bucket = 4   # large long

        # 4. Cash bucket
        if cash < 8000:
            cash_bucket = 0  # low
        elif cash < 12000:
            cash_bucket = 1  # medium
        else:
            cash_bucket = 2  # high

        return (price_trend, spread_bucket, inv_bucket, cash_bucket)

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def get_diagnostics(self) -> Dict[str, Any]:
        """Return training diagnostics for convergence analysis."""
        return {
            "q_table_size": len(self.q_table),
            "epsilon": self.epsilon,
            "decay_per_episode": self.decay_per_episode,
            "total_updates": len(self.reward_history),
            "mean_reward_last_100": (
                float(np.mean(self.reward_history[-100:]))
                if self.reward_history else 0.0
            ),
            "mean_td_error_last_100": (
                float(np.mean(self.td_error_history[-100:]))
                if self.td_error_history else 0.0
            ),
        }

    def __repr__(self) -> str:
        return (
            f"RLDirectionalAgent(id={self.agent_id!r}, "
            f"α={self.alpha}, γ={self.gamma}, ε={self.epsilon:.3f})"
        )
