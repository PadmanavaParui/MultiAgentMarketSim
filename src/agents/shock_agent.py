"""
Shock Agent — Scripted liquidity shock injector (flash crash or short squeeze).

At a designated step (shock_step), fires large aggressive market orders in a single
direction for n steps:
  - SELL / CRASH: Sweeps the bid ladder to trigger a price collapse.
  - BUY / SQUEEZE: Sweeps the ask ladder to trigger a rapid price surge.
Outside the active shock window [shock_step, shock_step + n_steps), submits HOLD (noop).
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Union
import numpy as np


class ShockAgent:
    """
    Scripted market shock agent.

    Parameters
    ----------
    agent_id : str
        Unique identifier for this agent.
    shock_step : int
        Step at which the liquidity shock sequence begins.
    n_steps : int
        Duration in steps for which large market orders are continuously submitted (default: 5).
    direction : str | int
        Direction of the shock:
          'sell' / 'crash' / 'down' / 1 -> flash crash (large market sells).
          'buy' / 'squeeze' / 'up' / 0 -> short squeeze (large market buys).
        Default: 'sell'.
    quantity : int
        Size of the aggressive market orders (default: 50).
    seed : int | None
        Random seed for reproducibility.
    """

    def __init__(
        self,
        agent_id: str,
        shock_step: int,
        n_steps: int = 5,
        direction: Union[str, int] = "sell",
        quantity: int = 50,
        seed: Optional[int] = None,
        **kwargs: Any,
    ):
        self.agent_id = agent_id
        self.shock_step = int(shock_step)
        # Support duration as alias if passed in kwargs
        if "duration" in kwargs:
            n_steps = kwargs["duration"]
        self.n_steps = max(1, int(n_steps))
        self.quantity = int(quantity)

        # Parse direction: 0 = BID (buy/squeeze), 1 = ASK (sell/crash)
        if isinstance(direction, str):
            d_lower = direction.strip().lower()
            if d_lower in ("buy", "up", "squeeze", "b"):
                self.side = 0
                self.direction_label = "buy"
            else:
                self.side = 1
                self.direction_label = "sell"
        else:
            self.side = 0 if int(direction) == 0 else 1
            self.direction_label = "buy" if self.side == 0 else "sell"

        self.rng = np.random.default_rng(seed)
        self.current_step = 0

    def act(self, observation: Any) -> Dict[str, Any]:
        """
        Executes large market orders during [shock_step, shock_step + n_steps), otherwise holds.
        """
        in_shock_window = (
            self.shock_step <= self.current_step < (self.shock_step + self.n_steps)
        )

        if in_shock_window:
            action = {
                "order_type": 2,  # MARKET
                "side": self.side,
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
        """No-op — shock agent follows scripted shock schedule."""
        pass

    def reset(self, seed: Optional[int] = None) -> None:
        """Reset step counter and RNG."""
        self.current_step = 0
        if seed is not None:
            self.rng = np.random.default_rng(seed)

    def __repr__(self) -> str:
        return (
            f"ShockAgent(id={self.agent_id!r}, shock_step={self.shock_step}, "
            f"n_steps={self.n_steps}, direction={self.direction_label!r}, qty={self.quantity})"
        )
