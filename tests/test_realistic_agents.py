"""
Unit tests for realistic agent archetypes:
1. AvellanedaStoikovMM: analytical reservation price, optimal spread, rolling sigma returns, and quoting.
2. InformedTrader: oracle price jump anticipation, pre-jump market order execution, post-jump stopping.
3. ShockAgent: scripted liquidity shock window, flash crash / squeeze directional execution.
4. Small deterministic book matching & environment integration.
"""

from __future__ import annotations

import os
import shutil
import unittest
import numpy as np

from src.engine.order_book import OrderBook, Order, Side, OrderType
from src.engine.matching_engine import MatchingEngine
from src.env.market_env import MarketEnvironment
from src.agents.avellaneda_stoikov import AvellanedaStoikovMM
from src.agents.informed_trader import InformedTrader
from src.agents.shock_agent import ShockAgent
from scripts.run_simulation import run_simulation


class TestRealisticAgents(unittest.TestCase):
    def setUp(self):
        self.output_dir = "data/runs"
        self.test_run_id = "test_realistic_eval"

    def tearDown(self):
        p = os.path.join(self.output_dir, self.test_run_id)
        if os.path.exists(p):
            shutil.rmtree(p, ignore_errors=True)

    # ------------------------------------------------------------------
    # 1. AvellanedaStoikovMM Tests
    # ------------------------------------------------------------------

    def test_avellaneda_stoikov_formulas(self):
        """
        Verify the analytical Avellaneda-Stoikov reservation price and optimal spread formulas:
          r = s - q * gamma * sigma^2 * (T - t)
          delta = gamma * sigma^2 * (T - t) + (2 / gamma) * ln(1 + gamma / k)
          bid = r - delta / 2
          ask = r + delta / 2
        """
        gamma = 0.1
        k = 1.5
        as_mm = AvellanedaStoikovMM(
            agent_id="as_0",
            gamma=gamma,
            k=k,
            window=10,
            T=1.0,
            quantity=10,
        )

        mid = 100.0
        sigma = 0.05
        time_left = 1.0

        # When inventory q == 0, reservation price equals mid
        r_zero = as_mm.compute_reservation_price(mid=mid, inventory=0, sigma=sigma, time_left=time_left)
        self.assertAlmostEqual(r_zero, mid, places=6)

        # When long inventory (q > 0), reservation price shifts DOWN (wants to sell)
        r_long = as_mm.compute_reservation_price(mid=mid, inventory=10, sigma=sigma, time_left=time_left)
        expected_r_long = 100.0 - 10 * gamma * (sigma ** 2) * time_left
        self.assertAlmostEqual(r_long, expected_r_long, places=6)
        self.assertLess(r_long, mid)

        # When short inventory (q < 0), reservation price shifts UP (wants to buy)
        r_short = as_mm.compute_reservation_price(mid=mid, inventory=-10, sigma=sigma, time_left=time_left)
        expected_r_short = 100.0 - (-10) * gamma * (sigma ** 2) * time_left
        self.assertAlmostEqual(r_short, expected_r_short, places=6)
        self.assertGreater(r_short, mid)

        # Optimal spread delta
        delta = as_mm.compute_spread(sigma=sigma, time_left=time_left)
        expected_delta = gamma * (sigma ** 2) * time_left + (2.0 / gamma) * np.log(1.0 + gamma / k)
        self.assertAlmostEqual(delta, expected_delta, places=6)
        self.assertGreater(delta, 0.0)

        # Quotes around reservation price
        bid, ask = as_mm.compute_quotes(mid=mid, inventory=5, sigma=sigma, time_left=time_left)
        r_inv = as_mm.compute_reservation_price(mid=mid, inventory=5, sigma=sigma, time_left=time_left)
        self.assertAlmostEqual(bid, r_inv - delta / 2.0, places=6)
        self.assertAlmostEqual(ask, r_inv + delta / 2.0, places=6)
        self.assertAlmostEqual(ask - bid, delta, places=6)

    def test_avellaneda_stoikov_rolling_sigma(self):
        """Verify rolling return volatility computation from mid-price updates."""
        as_mm = AvellanedaStoikovMM(agent_id="as_0", window=4, default_sigma=0.01)

        # With 0 or 1 observations, sigma defaults to default_sigma
        self.assertEqual(as_mm.compute_sigma(), 0.01)

        obs1 = {"bids": np.array([[99.0, 10.0]]), "asks": np.array([[101.0, 10.0]])}
        as_mm.act(obs1)  # mid = 100.0
        self.assertEqual(as_mm.compute_sigma(), 0.01)

        # Step 2: mid = 102.0 -> return = (102 - 100) / 100 = +0.02
        obs2 = {"bids": np.array([[101.0, 10.0]]), "asks": np.array([[103.0, 10.0]])}
        as_mm.act(obs2)

        # Step 3: mid = 101.0 -> return = (101 - 102) / 102
        obs3 = {"bids": np.array([[100.0, 10.0]]), "asks": np.array([[102.0, 10.0]])}
        as_mm.act(obs3)

        self.assertEqual(len(as_mm.returns_history), 2)
        expected_std = float(np.std(list(as_mm.returns_history), ddof=1))
        self.assertAlmostEqual(as_mm.compute_sigma(), expected_std, places=6)

        # Reset clears returns and step
        as_mm.reset()
        self.assertEqual(len(as_mm.returns_history), 0)
        self.assertEqual(as_mm.current_step, 0)

    def test_avellaneda_stoikov_deterministic_book_quoting(self):
        """Test AS agent quoting on a small deterministic limit order book."""
        as_mm = AvellanedaStoikovMM(
            agent_id="as_0",
            gamma=0.1,
            k=1.5,
            quantity=15,
            default_sigma=0.02,
        )

        obs = {
            "bids": np.array([[99.5, 10.0]]),
            "asks": np.array([[100.5, 10.0]]),
            "inventory": np.array([0.0]),
        }

        # Step 0: should place a limit BID
        act_bid = as_mm.act(obs)
        self.assertEqual(act_bid["order_type"], 1)  # LIMIT
        self.assertEqual(act_bid["side"], 0)        # BID
        self.assertEqual(act_bid["quantity"], 15)
        self.assertLess(act_bid["price_offset"], 0.0)  # bid offset is below mid

        # Step 1: should place a limit ASK
        act_ask = as_mm.act(obs)
        self.assertEqual(act_ask["order_type"], 1)  # LIMIT
        self.assertEqual(act_ask["side"], 1)        # ASK
        self.assertEqual(act_ask["quantity"], 15)
        self.assertGreater(act_ask["price_offset"], 0.0)  # ask offset is above mid

    # ------------------------------------------------------------------
    # 2. InformedTrader Tests
    # ------------------------------------------------------------------

    def test_informed_trader_positive_jump(self):
        """Informed trader with upward price jump (+2.0) buys before jump_step then stops."""
        trader = InformedTrader(agent_id="inf_1", jump_step=4, jump_size=2.0, quantity=10)

        # Steps 0, 1, 2 (step < 4) -> aggressive market BUY
        for expected_step in range(3):
            act = trader.act({})
            self.assertEqual(act["order_type"], 2, f"Expected MARKET order at step {expected_step}")
            self.assertEqual(act["side"], 0, "Expected BID (BUY) side for positive jump")
            self.assertEqual(act["quantity"], 10)

        # Step 3 (step == 3 < 4) -> still buys
        act_last_buy = trader.act({})
        self.assertEqual(act_last_buy["order_type"], 2)
        self.assertEqual(act_last_buy["side"], 0)

        # Step 4 (step >= 4) -> STOPS trading (HOLD / NOOP)
        act_stop = trader.act({})
        self.assertEqual(act_stop["order_type"], 0)
        self.assertEqual(act_stop["quantity"], 0)

        # Subsequent steps remain stopped
        act_stop_2 = trader.act({})
        self.assertEqual(act_stop_2["order_type"], 0)

    def test_informed_trader_negative_jump(self):
        """Informed trader with downward price jump (-3.0) sells before jump_step then stops."""
        trader = InformedTrader(agent_id="inf_2", jump_step=3, jump_size=-3.0, quantity=25)

        # Steps 0, 1, 2 -> aggressive market SELL (3 orders before step 3)
        for _ in range(3):
            act = trader.act({})
            self.assertEqual(act["order_type"], 2)  # MARKET
            self.assertEqual(act["side"], 1)        # ASK (SELL)
            self.assertEqual(act["quantity"], 25)

        # Step 3 (at jump_step) -> stops
        act_stop = trader.act({})
        self.assertEqual(act_stop["order_type"], 0)

    def test_informed_trader_book_execution(self):
        """Test InformedTrader aggressive market order execution against standing book liquidity."""
        book = OrderBook(tick_size=0.01)
        engine = MatchingEngine(book)

        # Standing asks: 10 shares @ $101.0, 10 shares @ $102.0
        ask1 = Order("ask_1", "mm_0", Side.ASK, OrderType.LIMIT, price=101.0, quantity=10, timestamp_step=1)
        ask2 = Order("ask_2", "mm_0", Side.ASK, OrderType.LIMIT, price=102.0, quantity=10, timestamp_step=1)
        engine.process(ask1)
        engine.process(ask2)

        trader = InformedTrader("inf_0", jump_step=2, jump_size=1.5, quantity=15)
        action = trader.act({})
        self.assertEqual(action["order_type"], 2)
        self.assertEqual(action["side"], 0)

        # Process the market buy order
        buy_order = Order("buy_inf", "inf_0", Side.BID, OrderType.MARKET, price=101.0, quantity=action["quantity"], timestamp_step=2)
        trades = engine.process(buy_order)

        # Should fill 10 shares @ 101.0 and 5 shares @ 102.0
        self.assertEqual(len(trades), 2)
        self.assertEqual(trades[0].quantity, 10)
        self.assertEqual(trades[0].price, 101.0)
        self.assertEqual(trades[1].quantity, 5)
        self.assertEqual(trades[1].price, 102.0)

    # ------------------------------------------------------------------
    # 3. ShockAgent Tests
    # ------------------------------------------------------------------

    def test_shock_agent_flash_crash(self):
        """ShockAgent in flash crash mode fires market sells during shock window."""
        shock = ShockAgent(
            agent_id="shock_0",
            shock_step=2,
            n_steps=2,
            direction="sell",
            quantity=50,
        )

        # Step 0: prior to shock -> HOLD
        act0 = shock.act({})
        self.assertEqual(act0["order_type"], 0)
        self.assertEqual(act0["quantity"], 0)

        # Step 1: prior to shock -> HOLD
        act1 = shock.act({})
        self.assertEqual(act1["order_type"], 0)

        # Step 2: shock step -> MARKET SELL of 50 shares
        act2 = shock.act({})
        self.assertEqual(act2["order_type"], 2)
        self.assertEqual(act2["side"], 1)  # SELL
        self.assertEqual(act2["quantity"], 50)

        # Step 3: second shock step -> MARKET SELL of 50 shares
        act3 = shock.act({})
        self.assertEqual(act3["order_type"], 2)
        self.assertEqual(act3["side"], 1)
        self.assertEqual(act3["quantity"], 50)

        # Step 4: after shock window -> HOLD
        act4 = shock.act({})
        self.assertEqual(act4["order_type"], 0)
        self.assertEqual(act4["quantity"], 0)

    def test_shock_agent_short_squeeze(self):
        """ShockAgent in short squeeze mode fires market buys during shock window."""
        shock = ShockAgent(
            agent_id="squeeze_0",
            shock_step=1,
            n_steps=1,
            direction="squeeze",
            quantity=100,
        )

        # Step 0: prior to shock -> HOLD
        self.assertEqual(shock.act({})["order_type"], 0)

        # Step 1: shock active -> MARKET BUY of 100 shares
        act = shock.act({})
        self.assertEqual(act["order_type"], 2)
        self.assertEqual(act["side"], 0)  # BUY
        self.assertEqual(act["quantity"], 100)

        # Step 2: after shock -> HOLD
        self.assertEqual(shock.act({})["order_type"], 0)

    def test_shock_agent_clears_bid_ladder(self):
        """Verify flash crash shock sweeps through bids on a deterministic book."""
        book = OrderBook(tick_size=0.01)
        engine = MatchingEngine(book)

        # Add bid ladder: 10 @ 100.0, 10 @ 99.0, 10 @ 98.0
        for i, p in enumerate([100.0, 99.0, 98.0]):
            engine.process(Order(f"bid_{i}", "buyer", Side.BID, OrderType.LIMIT, price=p, quantity=10, timestamp_step=1))

        best_bid_before = book.best_bid()
        self.assertIsNotNone(best_bid_before)
        self.assertEqual(best_bid_before.price, 100.0)

        # Shock agent fires 25 share market sell
        shock_order = Order("shock_ord", "shocker", Side.ASK, OrderType.MARKET, price=100.0, quantity=25, timestamp_step=2)
        trades = engine.process(shock_order)

        self.assertEqual(len(trades), 3)  # fills 10 @ 100, 10 @ 99, 5 @ 98
        best_bid_after = book.best_bid()
        self.assertIsNotNone(best_bid_after)
        self.assertEqual(best_bid_after.price, 98.0)  # Price crashed from 100.0 down to 98.0
        self.assertEqual(best_bid_after.quantity, 5)

    # ------------------------------------------------------------------
    # 4. Simulation Runner Realistic Mode Integration
    # ------------------------------------------------------------------

    def test_run_simulation_realistic_mode(self):
        """Test full simulation execution with all realistic agents registered."""
        logger = run_simulation(
            mode="realistic",
            episodes=1,
            cycles_per_episode=15,
            snapshot_interval=5,
            seed=42,
            run_id=self.test_run_id,
            output_dir=self.output_dir,
            verbose=False,
        )

        run_summary = logger.get_run_summary(self.test_run_id)
        self.assertEqual(run_summary["status"], "COMPLETE")

        snapshots = logger.get_book_snapshots(self.test_run_id)
        self.assertEqual(len(snapshots), 3)  # 15 // 5 = 3 snapshots

        ts = logger.get_timeseries(self.test_run_id)
        self.assertEqual(len(ts), 3)

        agent_metrics = logger.get_agent_metrics(self.test_run_id)
        agent_names = set(m["agent_id"] for m in agent_metrics)
        self.assertIn("as_mm_0", agent_names)
        self.assertIn("informed_0", agent_names)
        self.assertIn("shock_0", agent_names)


if __name__ == "__main__":
    unittest.main()
