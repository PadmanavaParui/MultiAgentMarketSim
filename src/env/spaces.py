"""
Observation and action space definitions for the MarketEnvironment.

Extracted from market_env.py per the LLD module layout (src/env/spaces.py).
Provides factory functions used by MarketEnvironment to build Gymnasium spaces.
"""

import numpy as np
import gymnasium as gym
from typing import Dict, List


def build_observation_space_dict(k: int) -> gym.spaces.Dict:
    """Builds a Dict observation space with top-K bid/ask depth + agent state."""
    return gym.spaces.Dict({
        "bids": gym.spaces.Box(low=0.0, high=np.inf, shape=(k, 2), dtype=np.float32),
        "asks": gym.spaces.Box(low=0.0, high=np.inf, shape=(k, 2), dtype=np.float32),
        "cash": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32),
        "inventory": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32),
    })


def build_observation_space_flat(k: int) -> gym.spaces.Box:
    """Builds a flat Box observation space: [bid_prices, bid_vols, ask_prices, ask_vols, cash, inv]."""
    return gym.spaces.Box(
        low=-np.inf,
        high=np.inf,
        shape=(4 * k + 2,),
        dtype=np.float32,
    )


def build_action_space_dict(
    max_price_offset: float = 10.0,
    max_order_quantity: int = 100,
) -> gym.spaces.Dict:
    """
    Builds a Dict action space.
    Keys:
        order_type: Discrete(3)  — 0: HOLD, 1: LIMIT, 2: MARKET
        side:       Discrete(2)  — 0: BID, 1: ASK
        price_offset: Box        — offset relative to reference/mid price
        quantity:     Box        — order quantity
    """
    return gym.spaces.Dict({
        "order_type": gym.spaces.Discrete(3),
        "side": gym.spaces.Discrete(2),
        "price_offset": gym.spaces.Box(
            low=-max_price_offset, high=max_price_offset,
            shape=(1,), dtype=np.float32,
        ),
        "quantity": gym.spaces.Box(
            low=0.0, high=float(max_order_quantity),
            shape=(1,), dtype=np.float32,
        ),
    })


def build_action_space_flat(
    max_price_offset: float = 10.0,
    max_order_quantity: int = 100,
) -> gym.spaces.Box:
    """Builds a flat Box action space: [order_type, side, price_offset, quantity]."""
    return gym.spaces.Box(
        low=np.array([0.0, 0.0, -max_price_offset, 0.0], dtype=np.float32),
        high=np.array([2.0, 1.0, max_price_offset, float(max_order_quantity)], dtype=np.float32),
        dtype=np.float32,
    )
