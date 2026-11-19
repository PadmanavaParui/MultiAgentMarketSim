"""
Market-Maker (MM) baseline agent.

Quotes both sides of the book around the mid-price, earning the bid-ask
spread.  Applies inventory-based skewing — when the agent accumulates a
large long (or short) position the spread is shifted to attract
mean-reverting flow.

Like the ZI agent this is a *non-learning* baseline whose ``update()``
method is a no-op (BRD §BR-004).
"""

from __future__ import annotations

import numpy as np
from typing import Any, Dict, List, Optional, Tuple, Union


class MarketMakerAgent:
    """
    Market-Maker (MM) agent — quotes both sides of the book.

    Parameters
    ----------
    agent_id : str
        Unique identifier for this agent instance.
    seed : int | None
        Random seed for reproducibility.
    spread : float
        Half-spread around the mid-price (each side is offset by ±spread).
    quantity : int
        Default quote quantity on each side.
    max_inventory : int
        Soft inventory limit — beyond this the agent skews its quotes.
    skew_factor : float
        How aggressively quotes are skewed per unit of inventory beyond
        the soft limit (price units per excess share).
    """

    def __init__(
        self,
        agent_id: str,
        seed: Optional[int] = None,
        spread: float = 1.0,
        quantity: int = 10,
        max_inventory: int = 50,
        skew_factor: float = 0.05,
    ):
        self.agent_id = agent_id
        self.spread = spread
        self.quantity = quantity
        self.max_inventory = max_inventory
        self.skew_factor = skew_factor
        self.rng = np.random.default_rng(seed)
        self._last_inventory: int = 0

    # ------------------------------------------------------------------
    # AgentPolicy interface
    # ------------------------------------------------------------------

    def act(self, observation: Any) -> Dict[str, Any]:
        """
        Return *one* order per step.

        The MM alternates between placing a bid and an ask each step,
        achieving two-sided quoting across consecutive steps.  Inventory
        skewing shifts the offset toward reducing the position.

        Returns a dict action compatible with MarketEnvironment's Dict
        action space.
        """
        # Extract inventory from observation
        inventory = self._extract_inventory(observation)
        self._last_inventory = inventory

        # Compute inventory skew
        skew = 0.0
        if abs(inventory) > self.max_inventory:
            excess = abs(inventory) - self.max_inventory
            # If long → skew asks down (more eager to sell)
            # If short → skew bids up (more eager to buy)
            skew = excess * self.skew_factor * (-1 if inventory > 0 else 1)

        # Alternate sides: even RNG → BID, odd → ASK  (simple two-sided quoting)
        if self.rng.random() < 0.5:
            # Place a bid
            offset = -self.spread + skew
            side = 0  # BID
        else:
            # Place an ask
            offset = self.spread + skew
            side = 1  # ASK

        return {
            "order_type": 1,  # LIMIT
            "side": side,
            "price_offset": float(offset),
            "quantity": self.quantity,
        }

    def update(self, reward: float, next_observation: Any) -> None:
        """No-op — MM agents do not learn."""
        pass

    def reset(self, seed: Optional[int] = None) -> None:
        """Reset the agent's RNG (optionally with a new seed)."""
        self._last_inventory = 0
        if seed is not None:
            self.rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_inventory(observation: Any) -> int:
        """Pull inventory from either dict or flat-array observation."""
        if isinstance(observation, dict):
            inv = observation.get("inventory", np.array([0.0]))
            return int(np.asarray(inv).flatten()[0])
        elif isinstance(observation, np.ndarray):
            # In the flat layout the last element is inventory
            return int(observation[-1])
        return 0

    def __repr__(self) -> str:
        return (
            f"MarketMakerAgent(id={self.agent_id!r}, "
            f"spread={self.spread}, qty={self.quantity})"
        )
