"""
Informed Trader Agent with Oracle Knowledge.

Equipped with advance knowledge of a future price jump (jump_step, jump_size).
Before jump_step:
  Aggressively front-runs the jump by submitting market orders in the direction
  of the jump (BUY if jump_size > 0, SELL if jump_size < 0).
At and after jump_step:
  Stops trading (submits HOLD / NOOP).
"""

from __future__ import annotations

from typing import Any, Dict, Optional
import numpy as np


class InformedTrader:
    """
    Informed trader with oracle foresight of a price jump.

    Parameters
    ----------
    agent_id : str
        Unique identifier for this agent.
    jump_step : int
        Simulation step at which the price jump is scheduled to occur.
    jump_size : float
        Magnitude and direction of the future price jump (> 0 implies upward jump, < 0 downward).
    quantity : int
        Size of aggressive market orders submitted prior to jump_step (default: 10).
    seed : int | None
        Random seed for reproducibility.
    """

    def __init__(
        self,
        agent_id: str,
        jump_step: int,
        jump_size: float,
        quantity: int = 10,
        seed: Optional[int] = None,
        **kwargs: Any,
    ):
        self.agent_id = agent_id
        self.jump_step = int(jump_step)
        self.jump_size = float(jump_size)
        self.quantity = int(quantity)
        self.rng = np.random.default_rng(seed)
        self.current_step = 0

    def act(self, observation: Any) -> Dict[str, Any]:
        """
        Submits market order in direction of jump before jump_step, then holds.
        """
        # Determine whether we are still prior to the price jump
        if self.current_step < self.jump_step and self.jump_size != 0.0:
            # 0: BID (BUY) if positive jump; 1: ASK (SELL) if negative jump
            side = 0 if self.jump_size > 0 else 1
            action = {
                "order_type": 2,  # MARKET
                "side": side,
                "price_offset": 0.0,
                "quantity": self.quantity,
            }
        else:
            action = {
                "order_type": 0,  # HOLD / NOOP
                "side": 0,
                "price_offset": 0.0,
                "quantity": 0,
            }

        self.current_step += 1
        return action

    def update(self, reward: float, next_observation: Any) -> None:
        """No-op — informed trader uses fixed oracle schedule."""
        pass

    def reset(self, seed: Optional[int] = None) -> None:
        """Reset step counter and RNG."""
        self.current_step = 0
        if seed is not None:
            self.rng = np.random.default_rng(seed)

    def __repr__(self) -> str:
        return (
            f"InformedTrader(id={self.agent_id!r}, "
            f"jump_step={self.jump_step}, jump_size={self.jump_size}, qty={self.quantity})"
        )
