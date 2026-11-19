"""
Zero-Intelligence (ZI) baseline agent.

Generates random order flow — randomly chooses BID/ASK, random price offset
around the mid-price, and random quantity.  Follows the shared AgentPolicy
interface so it can be used interchangeably with the Market-Maker and RL agents.

This agent performs *no learning*; its ``update()`` method is a no-op.
It exists purely to validate the environment before any RL training is trusted
(BRD §BR-004, Training Curriculum step 1).
"""

from __future__ import annotations

import numpy as np
from typing import Any, Dict, Optional, Tuple, Union


class ZeroIntelligenceAgent:
    """
    Zero-Intelligence (ZI) agent — simplest correctness baseline.

    Parameters
    ----------
    agent_id : str
        Unique identifier for this agent instance.
    seed : int | None
        Random seed for reproducibility.
    max_quantity : int
        Maximum order quantity per action.
    price_range : float
        Maximum absolute price offset from the reference price.
    aggression : float
        Probability of submitting an order vs. holding (0.0–1.0).
    """

    def __init__(
        self,
        agent_id: str,
        seed: Optional[int] = None,
        max_quantity: int = 20,
        price_range: float = 5.0,
        aggression: float = 0.7,
    ):
        self.agent_id = agent_id
        self.max_quantity = max_quantity
        self.price_range = price_range
        self.aggression = aggression
        self.rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------
    # AgentPolicy interface
    # ------------------------------------------------------------------

    def act(self, observation: Any) -> Dict[str, Any]:
        """
        Select an action given the current observation.

        Returns a dict action compatible with MarketEnvironment's Dict action
        space:
            order_type: 0 (HOLD), 1 (LIMIT), 2 (MARKET)
            side:       0 (BID), 1 (ASK)
            price_offset: float
            quantity:   int
        """
        # Decide whether to place an order at all
        if self.rng.random() > self.aggression:
            return {"order_type": 0, "side": 0, "price_offset": 0.0, "quantity": 0}

        # Random order type: ~80 % limit, ~20 % market
        order_type = 1 if self.rng.random() < 0.8 else 2

        # Random side
        side = int(self.rng.integers(0, 2))

        # Random price offset within [-price_range, price_range]
        price_offset = float(self.rng.uniform(-self.price_range, self.price_range))

        # Random quantity in [1, max_quantity]
        quantity = int(self.rng.integers(1, self.max_quantity + 1))

        return {
            "order_type": order_type,
            "side": side,
            "price_offset": price_offset,
            "quantity": quantity,
        }

    def update(self, reward: float, next_observation: Any) -> None:
        """No-op — ZI agents do not learn."""
        pass

    def reset(self, seed: Optional[int] = None) -> None:
        """Reset the agent's RNG (optionally with a new seed)."""
        if seed is not None:
            self.rng = np.random.default_rng(seed)

    def __repr__(self) -> str:
        return (
            f"ZeroIntelligenceAgent(id={self.agent_id!r}, "
            f"aggression={self.aggression}, max_qty={self.max_quantity})"
        )
