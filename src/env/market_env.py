import functools
from collections import deque
from typing import List, Dict, Optional, Tuple, Any, Union
import numpy as np
import gymnasium as gym
from pettingzoo import ParallelEnv

from src.engine.order_book import OrderBook, Order, Side, OrderType, Status
from src.engine.matching_engine import MatchingEngine, Trade
from src.engine.validators import validate_no_cross, validate_zero_sum, SimulationInvalidError


class MarketEnvironment(ParallelEnv):
    """
    PettingZoo ParallelEnv wrapper around the Central Limit Order Book (OrderBook)
    and MatchingEngine.
    """
    metadata = {
        "render_modes": ["ansi"],
        "name": "market_env_v0"
    }

    def __init__(
        self,
        agents: Optional[Union[List[str], int, Dict[str, Any]]] = None,
        tick_size: float = 0.01,
        k: int = 5,
        max_cycles: int = 100,
        initial_cash: float = 10000.0,
        initial_inventory: int = 0,
        initial_price: float = 100.0,
        max_order_quantity: int = 100,
        max_price_offset: float = 10.0,
        fee_rate: float = 0.0,
        dict_spaces: bool = True,
        run_id: str = "default_run",
        render_mode: Optional[str] = None,
        max_position: Optional[int] = 100,
        price_band: Optional[Union[float, Tuple[float, float]]] = 0.10,
        order_ttl: Optional[int] = 20,
    ):
        super().__init__()
        # Determine agent IDs
        if agents is None:
            self.possible_agents = ["agent_0", "agent_1"]
        elif isinstance(agents, int):
            self.possible_agents = [f"agent_{i}" for i in range(agents)]
        elif isinstance(agents, dict):
            self.possible_agents = [str(k) for k in agents.keys()]
        elif isinstance(agents, (list, tuple)):
            self.possible_agents = [str(a) for a in agents]
        else:
            self.possible_agents = ["agent_0", "agent_1"]

        self.agents = self.possible_agents[:]
        self.tick_size = tick_size
        self.k = k
        self.max_cycles = max_cycles
        self.initial_price = initial_price
        self.max_order_quantity = max_order_quantity
        self.max_price_offset = max_price_offset
        self.fee_rate = fee_rate
        self.dict_spaces = dict_spaces
        self.run_id = run_id
        self.render_mode = render_mode
        self.max_position = max_position
        self.price_band = price_band
        self.order_ttl = order_ttl

        self.min_price: Optional[float] = None
        self.max_price: Optional[float] = None
        self.episode_reference_price: Optional[float] = None
        self.rejected_orders: Dict[str, int] = {a: 0 for a in self.possible_agents}
        self.position_limit_rejections: Dict[str, int] = {a: 0 for a in self.possible_agents}
        self.price_band_rejections: Dict[str, int] = {a: 0 for a in self.possible_agents}
        self.expired_orders: Dict[str, int] = {a: 0 for a in self.possible_agents}
        self.open_bid_qty: Dict[str, int] = {a: 0 for a in self.possible_agents}
        self.open_ask_qty: Dict[str, int] = {a: 0 for a in self.possible_agents}
        self._resting_order_queue: deque[Tuple[int, str]] = deque()

        # Configure initial cash & inventory per agent
        self._initial_cash: Dict[str, float] = {}
        self._initial_inventory: Dict[str, int] = {}
        for a in self.possible_agents:
            if isinstance(agents, dict) and isinstance(agents[a], dict):
                self._initial_cash[a] = float(agents[a].get("cash", agents[a].get("capital", initial_cash)))
                self._initial_inventory[a] = int(agents[a].get("inventory", initial_inventory))
            else:
                self._initial_cash[a] = float(initial_cash)
                self._initial_inventory[a] = int(initial_inventory)

        # Build observation and action spaces
        self._observation_spaces: Dict[str, gym.spaces.Space] = {}
        self._action_spaces: Dict[str, gym.spaces.Space] = {}

        for a in self.possible_agents:
            if self.dict_spaces:
                self._observation_spaces[a] = gym.spaces.Dict({
                    "bids": gym.spaces.Box(low=0.0, high=np.inf, shape=(self.k, 2), dtype=np.float32),
                    "asks": gym.spaces.Box(low=0.0, high=np.inf, shape=(self.k, 2), dtype=np.float32),
                    "cash": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32),
                    "inventory": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32),
                })
                self._action_spaces[a] = gym.spaces.Dict({
                    "order_type": gym.spaces.Discrete(3),  # 0: HOLD / NOOP, 1: LIMIT, 2: MARKET
                    "side": gym.spaces.Discrete(2),        # 0: BID, 1: ASK
                    "price_offset": gym.spaces.Box(low=-self.max_price_offset, high=self.max_price_offset, shape=(1,), dtype=np.float32),
                    "quantity": gym.spaces.Box(low=0.0, high=float(self.max_order_quantity), shape=(1,), dtype=np.float32),
                })
            else:
                self._observation_spaces[a] = gym.spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(4 * self.k + 2,),
                    dtype=np.float32
                )
                self._action_spaces[a] = gym.spaces.Box(
                    low=np.array([0.0, 0.0, -self.max_price_offset, 0.0], dtype=np.float32),
                    high=np.array([2.0, 1.0, self.max_price_offset, float(self.max_order_quantity)], dtype=np.float32),
                    dtype=np.float32
                )

        self.observation_spaces = self._observation_spaces
        self.action_spaces = self._action_spaces

        # Internal simulation states
        self.np_random, self._seed = gym.utils.seeding.np_random(None)
        self.order_book = OrderBook(tick_size=self.tick_size)
        self.matching_engine = MatchingEngine(self.order_book)
        self.current_step = 0
        self.fees_collected = 0.0
        self._order_counter = 0
        self.last_trade_price: Optional[float] = None
        self.cash: Dict[str, float] = {}
        self.inventory: Dict[str, int] = {}
        self.initial_total_value = 0.0
        self.prev_portfolio_value: Dict[str, float] = {}
        self._all_orders: Dict[str, Order] = {}

    def observation_space(self, agent: str) -> gym.spaces.Space:
        return self._observation_spaces[agent]

    def action_space(self, agent: str) -> gym.spaces.Space:
        return self._action_spaces[agent]

    def _get_reference_price(self) -> float:
        """Returns mid-price if spread exists, otherwise touch price, last trade or initial price."""
        best_bid = self.order_book.best_bid()
        best_ask = self.order_book.best_ask()
        if best_bid is not None and best_ask is not None:
            ref = (best_bid.price + best_ask.price) / 2.0
        elif best_bid is not None:
            ref = best_bid.price
        elif best_ask is not None:
            ref = best_ask.price
        elif self.last_trade_price is not None:
            ref = self.last_trade_price
        else:
            ref = self.initial_price

        if hasattr(self, "min_price") and self.min_price is not None:
            ref = max(ref, self.min_price)
        if hasattr(self, "max_price") and self.max_price is not None:
            ref = min(ref, self.max_price)
        return ref

    def _get_obs(self, agent: str) -> Union[Dict[str, np.ndarray], np.ndarray]:
        """Returns top-K bids/asks and agent's own cash and inventory."""
        depth = self.order_book.depth(levels=self.k)
        bids_arr = np.zeros((self.k, 2), dtype=np.float32)
        for i, (price, vol) in enumerate(depth.get("bids", [])[:self.k]):
            bids_arr[i, 0] = np.float32(price)
            bids_arr[i, 1] = np.float32(vol)

        asks_arr = np.zeros((self.k, 2), dtype=np.float32)
        for i, (price, vol) in enumerate(depth.get("asks", [])[:self.k]):
            asks_arr[i, 0] = np.float32(price)
            asks_arr[i, 1] = np.float32(vol)

        cash_val = np.float32(self.cash[agent])
        inv_val = np.float32(self.inventory[agent])

        if self.dict_spaces:
            return {
                "bids": bids_arr,
                "asks": asks_arr,
                "cash": np.array([cash_val], dtype=np.float32),
                "inventory": np.array([inv_val], dtype=np.float32),
            }
        else:
            return np.concatenate([
                bids_arr.flatten(),
                asks_arr.flatten(),
                np.array([cash_val, inv_val], dtype=np.float32)
            ])

    def _parse_action(self, action: Any) -> Tuple[Optional[OrderType], Side, float, int]:
        """
        Parses action input into (order_type, side, price_offset, quantity).
        Supports dicts, lists, tuples, numpy arrays, and enums.
        """
        if isinstance(action, dict):
            raw_ot = action.get("order_type", 0)
            raw_side = action.get("side", 0)
            raw_offset = action.get("price_offset", 0.0)
            raw_qty = action.get("quantity", 0)
        elif isinstance(action, (list, tuple, np.ndarray)):
            raw_ot = action[0] if len(action) > 0 else 0
            raw_side = action[1] if len(action) > 1 else 0
            raw_offset = action[2] if len(action) > 2 else 0.0
            raw_qty = action[3] if len(action) > 3 else 0
        else:
            return None, Side.BID, 0.0, 0

        # Parse order type: 0 = HOLD, 1 = LIMIT, 2 = MARKET
        if isinstance(raw_ot, OrderType):
            order_type = raw_ot
        elif isinstance(raw_ot, str):
            if raw_ot.upper() in ("LIMIT", "L"):
                order_type = OrderType.LIMIT
            elif raw_ot.upper() in ("MARKET", "M"):
                order_type = OrderType.MARKET
            else:
                order_type = None
        else:
            val = int(round(float(np.asarray(raw_ot).item() if np.asarray(raw_ot).ndim > 0 else raw_ot)))
            if val == 1:
                order_type = OrderType.LIMIT
            elif val == 2:
                order_type = OrderType.MARKET
            else:
                order_type = None

        # Parse side: 0 = BID, 1 = ASK
        if isinstance(raw_side, Side):
            side = raw_side
        elif isinstance(raw_side, str):
            side = Side.BID if raw_side.upper() in ("BID", "BUY", "B") else Side.ASK
        else:
            val = int(round(float(np.asarray(raw_side).item() if np.asarray(raw_side).ndim > 0 else raw_side)))
            side = Side.BID if val == 0 else Side.ASK

        # Parse price offset
        if isinstance(raw_offset, (int, float, np.floating, np.integer)):
            price_offset = float(raw_offset)
        else:
            arr = np.asarray(raw_offset)
            price_offset = float(arr.item()) if arr.size > 0 else 0.0

        # Parse quantity
        if isinstance(raw_qty, (int, float, np.floating, np.integer)):
            qty = int(round(float(raw_qty)))
        else:
            arr = np.asarray(raw_qty)
            qty = int(round(float(arr.item()))) if arr.size > 0 else 0

        qty = max(0, min(qty, self.max_order_quantity))

        return order_type, side, price_offset, qty

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict] = None
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Resets the environment for a new episode.
        """
        if options and "max_position" in options:
            self.max_position = options["max_position"]
        if options and "price_band" in options:
            self.price_band = options["price_band"]
        if options and "order_ttl" in options:
            self.order_ttl = options["order_ttl"]

        if seed is not None:
            self._seed = seed
            self.np_random, _ = gym.utils.seeding.np_random(seed)
            for a in self.possible_agents:
                self._action_spaces[a].seed(seed)
                self._observation_spaces[a].seed(seed)
        elif not hasattr(self, "np_random") or self.np_random is None:
            self.np_random, self._seed = gym.utils.seeding.np_random(None)

        self.agents = self.possible_agents[:]
        self.order_book = OrderBook(tick_size=self.tick_size)
        self.matching_engine = MatchingEngine(self.order_book)
        self.current_step = 0
        self.fees_collected = 0.0
        self._order_counter = 0
        self._all_orders = {}
        self.last_trade_price = None

        self.cash = {a: float(self._initial_cash[a]) for a in self.possible_agents}
        self.inventory = {a: int(self._initial_inventory[a]) for a in self.possible_agents}
        self.initial_total_value = sum(self.cash[a] for a in self.possible_agents)

        self.rejected_orders = {a: 0 for a in self.possible_agents}
        self.position_limit_rejections = {a: 0 for a in self.possible_agents}
        self.price_band_rejections = {a: 0 for a in self.possible_agents}
        self.expired_orders = {a: 0 for a in self.possible_agents}
        self.open_bid_qty = {a: 0 for a in self.possible_agents}
        self.open_ask_qty = {a: 0 for a in self.possible_agents}
        self._resting_order_queue = deque()

        # Calculate initial reference price and price bands
        self.min_price = None
        self.max_price = None
        self.episode_reference_price = self._get_reference_price()

        if self.price_band is not None:
            if isinstance(self.price_band, (int, float)):
                band = float(self.price_band)
                self.min_price = max(self.tick_size, round(self.episode_reference_price * (1.0 - band) / self.tick_size) * self.tick_size)
                self.max_price = round(self.episode_reference_price * (1.0 + band) / self.tick_size) * self.tick_size
                self.min_price = round(self.min_price, 4)
                self.max_price = round(self.max_price, 4)
            elif isinstance(self.price_band, (tuple, list)) and len(self.price_band) == 2:
                self.min_price = max(self.tick_size, round(float(self.price_band[0]) / self.tick_size) * self.tick_size)
                self.max_price = round(float(self.price_band[1]) / self.tick_size) * self.tick_size
                self.min_price = round(self.min_price, 4)
                self.max_price = round(self.max_price, 4)

        self.matching_engine.set_price_bounds(self.min_price, self.max_price)

        ref_price = self._get_reference_price()
        self.prev_portfolio_value = {
            a: self.cash[a] + self.inventory[a] * ref_price
            for a in self.possible_agents
        }

        observations = {a: self._get_obs(a) for a in self.agents}
        infos = {
            a: {
                "rejected_orders": 0,
                "position_limit_rejections": 0,
                "price_band_rejections": 0,
                "expired_orders": 0,
                "step_expired_orders": 0,
                "open_bid_qty": 0,
                "open_ask_qty": 0,
            }
            for a in self.agents
        }

        return observations, infos

    def step(
        self, actions: Dict[str, Any]
    ) -> Tuple[
        Dict[str, Any],
        Dict[str, float],
        Dict[str, bool],
        Dict[str, bool],
        Dict[str, Dict[str, Any]],
    ]:
        """
        Receives actions for all active agents and processes them through the CLOB and matching engine.
        Returns (observations, rewards, terminations, truncations, infos).
        """
        if not self.agents:
            return {}, {}, {}, {}, {}

        current_agents = list(self.agents)
        self.current_step += 1

        # Deterministic shuffle of active agents order using np_random
        agent_order = [a for a in current_agents if a in actions]
        if len(agent_order) > 1:
            perm = self.np_random.permutation(len(agent_order))
            agent_order = [agent_order[i] for i in perm]

        step_trades: List[Trade] = []
        agent_trades: Dict[str, List[Trade]] = {a: [] for a in current_agents}
        step_rejected_orders: Dict[str, int] = {a: 0 for a in current_agents}
        step_pos_rejections: Dict[str, int] = {a: 0 for a in current_agents}
        step_band_rejections: Dict[str, int] = {a: 0 for a in current_agents}
        step_expired_orders: Dict[str, int] = {a: 0 for a in current_agents}
        step_rejection_reasons: Dict[str, Optional[str]] = {a: None for a in current_agents}

        # Auto-cancel stale resting orders based on order_ttl
        if self.order_ttl is not None and self.order_ttl > 0:
            while self._resting_order_queue:
                ts, ord_id = self._resting_order_queue[0]
                if self.current_step - ts >= self.order_ttl:
                    self._resting_order_queue.popleft()
                    ord_obj = self._all_orders.get(ord_id)
                    if (
                        ord_obj is not None
                        and ord_obj.status in (Status.OPEN, Status.PARTIALLY_FILLED)
                        and ord_obj.quantity > 0
                    ):
                        rem_qty = ord_obj.quantity
                        self.order_book.cancel_order(ord_id)
                        ag = ord_obj.agent_id
                        if ord_obj.side == Side.BID:
                            self.open_bid_qty[ag] = max(0, self.open_bid_qty.get(ag, 0) - rem_qty)
                        else:
                            self.open_ask_qty[ag] = max(0, self.open_ask_qty.get(ag, 0) - rem_qty)
                        self.expired_orders[ag] = self.expired_orders.get(ag, 0) + 1
                        step_expired_orders[ag] = step_expired_orders.get(ag, 0) + 1
                else:
                    break

        # Process orders
        for agent_id in agent_order:
            action = actions[agent_id]
            order_type, side, price_offset, qty = self._parse_action(action)

            if order_type is None or qty <= 0:
                continue

            # Check max_position constraint (considering live resting exposure)
            if self.max_position is not None:
                current_inv = self.inventory[agent_id]
                open_bids = self.open_bid_qty.get(agent_id, 0)
                open_asks = self.open_ask_qty.get(agent_id, 0)

                if side == Side.BID and (current_inv + open_bids + qty > self.max_position):
                    self.position_limit_rejections[agent_id] = self.position_limit_rejections.get(agent_id, 0) + 1
                    self.rejected_orders[agent_id] = self.rejected_orders.get(agent_id, 0) + 1
                    step_rejected_orders[agent_id] += 1
                    step_pos_rejections[agent_id] += 1
                    step_rejection_reasons[agent_id] = "max_position_exceeded"
                    continue
                elif side == Side.ASK and (current_inv - open_asks - qty < -self.max_position):
                    self.position_limit_rejections[agent_id] = self.position_limit_rejections.get(agent_id, 0) + 1
                    self.rejected_orders[agent_id] = self.rejected_orders.get(agent_id, 0) + 1
                    step_rejected_orders[agent_id] += 1
                    step_pos_rejections[agent_id] += 1
                    step_rejection_reasons[agent_id] = "max_position_exceeded"
                    continue

            ref_price = self._get_reference_price()

            if order_type == OrderType.LIMIT:
                raw_price = ref_price + price_offset
                price = max(self.tick_size, round(raw_price / self.tick_size) * self.tick_size)
                price = round(price, 4)

                # Check price band constraint
                if self.min_price is not None and price < self.min_price:
                    self.price_band_rejections[agent_id] = self.price_band_rejections.get(agent_id, 0) + 1
                    self.rejected_orders[agent_id] = self.rejected_orders.get(agent_id, 0) + 1
                    step_rejected_orders[agent_id] += 1
                    step_band_rejections[agent_id] += 1
                    step_rejection_reasons[agent_id] = "price_below_band"
                    continue
                if self.max_price is not None and price > self.max_price:
                    self.price_band_rejections[agent_id] = self.price_band_rejections.get(agent_id, 0) + 1
                    self.rejected_orders[agent_id] = self.rejected_orders.get(agent_id, 0) + 1
                    step_rejected_orders[agent_id] += 1
                    step_band_rejections[agent_id] += 1
                    step_rejection_reasons[agent_id] = "price_above_band"
                    continue
            else:
                # Market order: price is set to current reference price
                price = ref_price

            self._order_counter += 1
            order = Order(
                order_id=f"ord_{self._order_counter}",
                agent_id=agent_id,
                side=side,
                order_type=order_type,
                price=price,
                quantity=qty,
                timestamp_step=self.current_step,
                run_id=self.run_id,
            )
            self._all_orders[order.order_id] = order

            trades = self.matching_engine.process(order)
            for t in trades:
                step_trades.append(t)
                buy_agent = self._all_orders[t.buy_order_id].agent_id
                sell_agent = self._all_orders[t.sell_order_id].agent_id

                if buy_agent in agent_trades:
                    agent_trades[buy_agent].append(t)
                if sell_agent in agent_trades:
                    agent_trades[sell_agent].append(t)

                trade_val = t.price * t.quantity
                fee = trade_val * self.fee_rate
                self.fees_collected += (2.0 * fee)

                if buy_agent in self.cash:
                    self.cash[buy_agent] -= (trade_val + fee)
                    self.inventory[buy_agent] += t.quantity

                if sell_agent in self.cash:
                    self.cash[sell_agent] += (trade_val - fee)
                    self.inventory[sell_agent] -= t.quantity

                # Release resting exposure for the counterparty order resting in the book
                if order.side == Side.BID:
                    # Incoming was BID; resting order was ASK
                    self.open_ask_qty[sell_agent] = max(0, self.open_ask_qty.get(sell_agent, 0) - t.quantity)
                else:
                    # Incoming was ASK; resting order was BID
                    self.open_bid_qty[buy_agent] = max(0, self.open_bid_qty.get(buy_agent, 0) - t.quantity)

                self.last_trade_price = t.price

            # If the limit order has remaining unfilled quantity, it rests on the book
            if order.order_type == OrderType.LIMIT and order.quantity > 0:
                if order.side == Side.BID:
                    self.open_bid_qty[agent_id] = self.open_bid_qty.get(agent_id, 0) + order.quantity
                else:
                    self.open_ask_qty[agent_id] = self.open_ask_qty.get(agent_id, 0) + order.quantity
                self._resting_order_queue.append((order.timestamp_step, order.order_id))

        # Call validators after each step
        validate_no_cross(self.order_book, run_id=self.run_id)

        agent_states = {
            a: {"cash": self.cash[a], "inventory": self.inventory[a]}
            for a in self.possible_agents
        }
        current_total_value = sum(self.cash[a] for a in self.possible_agents)
        validate_zero_sum(
            agents=agent_states,
            initial_total_value=self.initial_total_value,
            current_total_value=current_total_value,
            fees_collected=self.fees_collected,
            run_id=self.run_id,
        )

        # Rewards and infos computation
        new_ref_price = self._get_reference_price()
        rewards: Dict[str, float] = {}
        infos: Dict[str, Dict[str, Any]] = {}

        for a in current_agents:
            portfolio_val = self.cash[a] + self.inventory[a] * new_ref_price
            step_reward = float(portfolio_val - self.prev_portfolio_value[a])
            self.prev_portfolio_value[a] = portfolio_val
            rewards[a] = step_reward

            infos[a] = {
                "cash": float(self.cash[a]),
                "inventory": int(self.inventory[a]),
                "portfolio_value": float(portfolio_val),
                "pnl": float(portfolio_val - self._initial_cash[a]),
                "step_pnl": step_reward,
                "trades_count": len(agent_trades.get(a, [])),
                "rejected_orders": self.rejected_orders.get(a, 0),
                "position_limit_rejections": self.position_limit_rejections.get(a, 0),
                "price_band_rejections": self.price_band_rejections.get(a, 0),
                "expired_orders": self.expired_orders.get(a, 0),
                "step_rejected_orders": step_rejected_orders.get(a, 0),
                "step_position_limit_rejections": step_pos_rejections.get(a, 0),
                "step_price_band_rejections": step_band_rejections.get(a, 0),
                "step_expired_orders": step_expired_orders.get(a, 0),
                "open_bid_qty": int(self.open_bid_qty.get(a, 0)),
                "open_ask_qty": int(self.open_ask_qty.get(a, 0)),
                "order_rejected": bool(step_rejected_orders.get(a, 0) > 0),
                "rejection_reason": step_rejection_reasons.get(a, None),
            }

        # Check terminations & truncations
        is_truncated = bool(self.current_step >= self.max_cycles)
        terminations = {a: False for a in current_agents}
        truncations = {a: is_truncated for a in current_agents}

        observations = {a: self._get_obs(a) for a in current_agents}

        # Update active agents list
        self.agents = [
            a for a in current_agents
            if not (terminations[a] or truncations[a])
        ]

        return observations, rewards, terminations, truncations, infos

    def cancel_order(self, order_id: str) -> bool:
        """
        Explicitly cancels an active resting order and releases its open exposure.
        Returns True if the order was cancelled, False otherwise.
        """
        ord_obj = self._all_orders.get(order_id)
        if ord_obj is not None and ord_obj.status in (Status.OPEN, Status.PARTIALLY_FILLED) and ord_obj.quantity > 0:
            rem_qty = ord_obj.quantity
            self.order_book.cancel_order(order_id)
            ag = ord_obj.agent_id
            if ord_obj.side == Side.BID:
                self.open_bid_qty[ag] = max(0, self.open_bid_qty.get(ag, 0) - rem_qty)
            else:
                self.open_ask_qty[ag] = max(0, self.open_ask_qty.get(ag, 0) - rem_qty)
            return True
        return False

    def render(self) -> Optional[str]:
        depth = self.order_book.depth(levels=self.k)
        lines = [
            f"--- Step {self.current_step} ---",
            "=== BIDS ===",
        ]
        for p, v in depth.get("bids", []):
            lines.append(f"  Price: {p:.2f} | Vol: {v}")
        lines.append("=== ASKS ===")
        for p, v in depth.get("asks", []):
            lines.append(f"  Price: {p:.2f} | Vol: {v}")
        lines.append("=== AGENTS ===")
        for a in self.possible_agents:
            lines.append(f"  {a}: Cash={self.cash[a]:.2f}, Inv={self.inventory[a]}")
        text = "\n".join(lines)
        if self.render_mode == "human":
            print(text)
        return text

    def close(self):
        pass


# Aliases and factory helpers
MarketEnv = MarketEnvironment


def parallel_env(**kwargs) -> MarketEnvironment:
    return MarketEnvironment(**kwargs)
