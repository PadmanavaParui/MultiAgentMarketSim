"""
Avellaneda-Stoikov Market-Making Agent (High-Frequency Trading in a Limit Order Book, 2008).

Formulation:
  Reservation price:
    r(s, q, t) = s - q * gamma * sigma^2 * (T - t)
  Optimal bid/ask spread:
    delta(t) = gamma * sigma^2 * (T - t) + (2 / gamma) * ln(1 + gamma / k)
  Quotes:
    bid = r - delta / 2
    ask = r + delta / 2

Parameters:
  gamma: inventory risk aversion parameter (> 0)
  k: order book liquidity parameter / arrival intensity parameter (> 0)
  window: rolling window size of recent mid returns to estimate sigma
"""

from __future__ import annotations

import collections
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np


class AvellanedaStoikovMM:
    """
    Avellaneda-Stoikov optimal inventory-risk market maker.

    Parameters
    ----------
    agent_id : str
        Unique identifier for this agent.
    gamma : float
        Inventory risk aversion parameter (default: 0.1).
    k : float
        Order arrival intensity parameter (default: 1.5).
    window : int
        Rolling window size for computing return volatility sigma (default: 10).
    T : float
        Terminal trading horizon (default: 1.0).
    max_steps : int
        Estimated total steps in episode for normalizing time (default: 100).
    quantity : int
        Order quantity per quote (default: 10).
    seed : int | None
        Random seed for reproducibility.
    default_sigma : float
        Fallback return volatility when fewer than 2 return observations exist (default: 0.01).
    """

    def __init__(
        self,
        agent_id: str,
        gamma: float = 0.1,
        k: float = 1.5,
        window: int = 10,
        T: float = 1.0,
        max_steps: int = 100,
        quantity: int = 10,
        seed: Optional[int] = None,
        default_sigma: float = 0.01,
        **kwargs: Any,
    ):
        self.agent_id = agent_id
        self.gamma = max(1e-6, float(gamma))
        self.k = max(1e-6, float(k))
        self.window = max(2, int(window))
        self.T = float(T)
        self.max_steps = max(1, int(max_steps))
        self.quantity = int(quantity)
        self.default_sigma = float(default_sigma)

        self.rng = np.random.default_rng(seed)
        self.current_step = 0
        self._last_mid: Optional[float] = None
        self.returns_history: collections.deque[float] = collections.deque(maxlen=self.window)
        self.current_inventory: int = 0

    # ------------------------------------------------------------------
    # Analytical Avellaneda-Stoikov Formulas
    # ------------------------------------------------------------------

    def compute_reservation_price(
        self,
        mid: float,
        inventory: int,
        sigma: float,
        time_left: float,
    ) -> float:
        """
        r(s, q, t) = s - q * gamma * sigma^2 * (T - t)
        """
        tau = max(0.0, float(time_left))
        return float(mid - inventory * self.gamma * (sigma ** 2) * tau)

    def compute_spread(
        self,
        sigma: float,
        time_left: float,
    ) -> float:
        """
        delta(t) = gamma * sigma^2 * (T - t) + (2 / gamma) * ln(1 + gamma / k)
        """
        tau = max(0.0, float(time_left))
        return float(
            self.gamma * (sigma ** 2) * tau
            + (2.0 / self.gamma) * np.log(1.0 + self.gamma / self.k)
        )

    def compute_quotes(
        self,
        mid: float,
        inventory: int,
        sigma: float,
        time_left: float,
    ) -> Tuple[float, float]:
        """
        Returns (bid_price, ask_price) = (r - delta/2, r + delta/2).
        """
        r = self.compute_reservation_price(mid, inventory, sigma, time_left)
        delta = self.compute_spread(sigma, time_left)
        return r - delta / 2.0, r + delta / 2.0

    def compute_sigma(self) -> float:
        """
        Computes sample standard deviation of rolling mid-price returns.
        Falls back to default_sigma if fewer than 2 return samples are recorded.
        """
        if len(self.returns_history) >= 2:
            std = float(np.std(self.returns_history, ddof=1))
            return max(1e-6, std)
        return self.default_sigma

    # ------------------------------------------------------------------
    # AgentPolicy Interface (act / update / reset)
    # ------------------------------------------------------------------

    def act(self, observation: Any) -> Dict[str, Any]:
        """
        Produces optimal Avellaneda-Stoikov quotes.
        Alternates between BID (even step) and ASK (odd step) to maintain two-sided quoting.
        """
        mid = self._extract_mid(observation)
        q = self._extract_inventory(observation)
        self.current_inventory = q

        # Update mid returns history
        if self._last_mid is not None and self._last_mid > 0:
            ret = (mid - self._last_mid) / self._last_mid
            self.returns_history.append(ret)
        self._last_mid = mid

        sigma = self.compute_sigma()
        time_left = max(0.0, self.T - (self.current_step / self.max_steps) * self.T)

        bid_price, ask_price = self.compute_quotes(mid, q, sigma, time_left)

        # Alternate side: step % 2 == 0 -> BID (r - delta/2), step % 2 == 1 -> ASK (r + delta/2)
        if self.current_step % 2 == 0:
            side = 0  # BID
            price_offset = float(bid_price - mid)
        else:
            side = 1  # ASK
            price_offset = float(ask_price - mid)

        self.current_step += 1

        return {
            "order_type": 1,  # LIMIT
            "side": side,
            "price_offset": price_offset,
            "quantity": self.quantity,
        }

    def update(self, reward: float, next_observation: Any) -> None:
        """Non-learning market maker; updates tracking if needed."""
        pass

    def reset(self, seed: Optional[int] = None) -> None:
        """Reset internal step counter, mid history, and RNG."""
        self.current_step = 0
        self._last_mid = None
        self.returns_history.clear()
        self.current_inventory = 0
        if seed is not None:
            self.rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------
    # Observation Parsers
    # ------------------------------------------------------------------

    def _extract_mid(self, observation: Any) -> float:
        """Extract current mid-price from dict or array observation."""
        if isinstance(observation, dict):
            bids = observation.get("bids")
            asks = observation.get("asks")
            best_bid = bids[0, 0] if (bids is not None and len(bids) > 0 and bids[0, 1] > 0) else None
            best_ask = asks[0, 0] if (asks is not None and len(asks) > 0 and asks[0, 1] > 0) else None
            if best_bid is not None and best_ask is not None:
                return float((best_bid + best_ask) / 2.0)
            elif best_bid is not None:
                return float(best_bid)
            elif best_ask is not None:
                return float(best_ask)
        elif isinstance(observation, np.ndarray) and len(observation) >= 2:
            # Flat space: first two values might be best_bid, vol
            if observation[0] > 0:
                return float(observation[0])
        return self._last_mid if self._last_mid is not None else 100.0

    @staticmethod
    def _extract_inventory(observation: Any) -> int:
        """Extract agent inventory from observation."""
        if isinstance(observation, dict):
            inv = observation.get("inventory", np.array([0.0]))
            return int(round(float(np.asarray(inv).flatten()[0])))
        elif isinstance(observation, np.ndarray) and len(observation) > 0:
            return int(round(float(observation[-1])))
        return 0

    def __repr__(self) -> str:
        return (
            f"AvellanedaStoikovMM(id={self.agent_id!r}, "
            f"gamma={self.gamma}, k={self.k}, window={self.window})"
        )
