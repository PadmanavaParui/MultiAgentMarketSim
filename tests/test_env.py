import unittest
import numpy as np
from pettingzoo.test import parallel_api_test, parallel_seed_test
from pettingzoo.test.seed_test import data_equivalence

from src.env.market_env import MarketEnvironment, parallel_env
from src.engine.order_book import Side, OrderType, Status
from src.agents.zero_intelligence import ZeroIntelligenceAgent
from src.agents.market_maker import MarketMakerAgent
from src.agents.avellaneda_stoikov import AvellanedaStoikovMM
from src.agents.rl_directional import RLDirectionalAgent
from src.agents.informed_trader import InformedTrader
from src.agents.shock_agent import ShockAgent


class TestMarketEnvironment(unittest.TestCase):
    def test_parallel_api(self):
        """Runs PettingZoo's standard parallel_api_test to ensure full API conformance."""
        env = MarketEnvironment(dict_spaces=True)
        parallel_api_test(env, num_cycles=200)

    def test_parallel_api_flat_box(self):
        """Runs parallel_api_test with dict_spaces=False (flat Box observation/action)."""
        env = MarketEnvironment(dict_spaces=False)
        parallel_api_test(env, num_cycles=100)

    def test_seed_reproducibility_pettingzoo_tool(self):
        """Uses PettingZoo's parallel_seed_test to check deterministic environment runs."""
        parallel_seed_test(lambda: MarketEnvironment(dict_spaces=True), num_cycles=100)

    def test_same_seed_explicit_identical_results(self):
        """
        Explicitly checks that two environments seeded with the same seed
        produce identical observations, rewards, terminations, truncations, and infos.
        """
        seed = 42
        env1 = MarketEnvironment(dict_spaces=True)
        env2 = MarketEnvironment(dict_spaces=True)

        obs1, info1 = env1.reset(seed=seed)
        obs2, info2 = env2.reset(seed=seed)

        self.assertTrue(data_equivalence(obs1, obs2))
        self.assertTrue(data_equivalence(info1, info2))

        # Seed the action spaces identically
        for a in env1.possible_agents:
            env1.action_space(a).seed(seed)
            env2.action_space(a).seed(seed)

        for step_idx in range(50):
            actions1 = {a: env1.action_space(a).sample() for a in env1.agents}
            actions2 = {a: env2.action_space(a).sample() for a in env2.agents}
            self.assertTrue(data_equivalence(actions1, actions2), f"Action mismatch at step {step_idx}")

            o1, r1, term1, trunc1, inf1 = env1.step(actions1)
            o2, r2, term2, trunc2, inf2 = env2.step(actions2)

            self.assertTrue(data_equivalence(o1, o2), f"Obs mismatch at step {step_idx}")
            self.assertTrue(data_equivalence(r1, r2), f"Reward mismatch at step {step_idx}")
            self.assertTrue(data_equivalence(term1, term2), f"Term mismatch at step {step_idx}")
            self.assertTrue(data_equivalence(trunc1, trunc2), f"Trunc mismatch at step {step_idx}")
            self.assertTrue(data_equivalence(inf1, inf2), f"Info mismatch at step {step_idx}")

            # Check that underlying order book depths and cash balances are bit-for-bit equal
            depth1 = env1.order_book.depth()
            depth2 = env2.order_book.depth()
            self.assertEqual(depth1, depth2)
            self.assertEqual(env1.cash, env2.cash)
            self.assertEqual(env1.inventory, env2.inventory)

            if len(env1.agents) == 0:
                break

    def test_zero_sum_conservation(self):
        """
        Tests that across multiple random trading steps, total cash (minus fees)
        and total inventory across all agents are strictly conserved.
        """
        env = MarketEnvironment(dict_spaces=True, max_cycles=60)
        env.reset(seed=999)

        initial_total_cash = sum(env.cash.values())
        initial_total_inv = sum(env.inventory.values())

        for _ in range(50):
            actions = {a: env.action_space(a).sample() for a in env.agents}
            env.step(actions)

            current_total_cash = sum(env.cash.values())
            current_total_inv = sum(env.inventory.values())

            expected_cash = initial_total_cash - env.fees_collected
            self.assertAlmostEqual(current_total_cash, expected_cash, places=5)
            self.assertEqual(current_total_inv, initial_total_inv)

            if len(env.agents) == 0:
                break

    def test_crossing_orders_and_no_cross_invariant(self):
        """
        Tests that when agents submit crossing orders (e.g. bid price >= ask price),
        the matching engine executes trades immediately, and the remaining spread
        is never crossed.
        """
        env = MarketEnvironment(dict_spaces=True)
        env.reset(seed=1)

        # Agent 0 places a Limit Bid @ 102 (buying 10 shares)
        # Agent 1 places a Limit Ask @ 98 (selling 10 shares)
        actions = {
            "agent_0": {"order_type": 1, "side": 0, "price_offset": 2.0, "quantity": 10},
            "agent_1": {"order_type": 1, "side": 1, "price_offset": -2.0, "quantity": 10},
        }
        obs, rewards, term, trunc, info = env.step(actions)

        # Trade should have matched 10 shares
        self.assertEqual(env.inventory["agent_0"], 10)
        self.assertEqual(env.inventory["agent_1"], -10)
        self.assertEqual(sum(env.cash.values()), env.initial_total_value)

        # Order book should not have a crossed spread
        best_bid = env.order_book.best_bid()
        best_ask = env.order_book.best_ask()
        if best_bid and best_ask:
            self.assertLess(best_bid.price, best_ask.price)

    def test_market_orders(self):
        """
        Tests placing a resting limit ask followed by a market buy order.
        """
        env = MarketEnvironment(dict_spaces=True)
        env.reset(seed=10)

        # Step 1: Agent 1 places a Limit Ask @ 101.0 (quantity 20)
        actions1 = {
            "agent_0": {"order_type": 0, "side": 0, "price_offset": 0.0, "quantity": 0},  # HOLD
            "agent_1": {"order_type": 1, "side": 1, "price_offset": 1.0, "quantity": 20},
        }
        env.step(actions1)
        self.assertIsNotNone(env.order_book.best_ask())
        self.assertEqual(env.order_book.best_ask().price, 101.0)

        # Step 2: Agent 0 places a Market Buy order for 15 shares
        actions2 = {
            "agent_0": {"order_type": 2, "side": 0, "price_offset": 0.0, "quantity": 15},  # MARKET BUY
            "agent_1": {"order_type": 0, "side": 0, "price_offset": 0.0, "quantity": 0},  # HOLD
        }
        env.step(actions2)

        # 15 shares traded @ 101.0
        self.assertEqual(env.inventory["agent_0"], 15)
        self.assertEqual(env.inventory["agent_1"], -15)
        self.assertEqual(env.cash["agent_0"], 10000.0 - (15 * 101.0))
        self.assertEqual(env.cash["agent_1"], 10000.0 + (15 * 101.0))

    def test_max_position_limit_rejection(self):
        """
        Tests that orders that would push |inventory| beyond max_position are rejected
        and accurately counted in info dicts.
        """
        env = MarketEnvironment(
            dict_spaces=True,
            max_position=50,
            price_band=0.20,
        )
        obs, infos = env.reset(seed=42)

        # Step 1: Agent 0 places resting Ask @ 100.0 (qty 40)
        env.step({
            "agent_0": {"order_type": 1, "side": 1, "price_offset": 0.0, "quantity": 40},
            "agent_1": {"order_type": 0, "side": 0, "price_offset": 0.0, "quantity": 0},
        })

        # Step 2: Agent 1 buys 40 shares (inv: +40 <= 50)
        env.step({
            "agent_0": {"order_type": 0, "side": 0, "price_offset": 0.0, "quantity": 0},
            "agent_1": {"order_type": 2, "side": 0, "price_offset": 0.0, "quantity": 40},
        })
        self.assertEqual(env.inventory["agent_1"], 40)

        # Step 3: Agent 1 has inventory = 40. Submitting BUY of 20 would make inventory 60 > 50.
        # Should be rejected!
        obs, rewards, term, trunc, infos = env.step({
            "agent_0": {"order_type": 0, "side": 0, "price_offset": 0.0, "quantity": 0},
            "agent_1": {"order_type": 2, "side": 0, "price_offset": 0.0, "quantity": 20},
        })
        self.assertEqual(env.inventory["agent_1"], 40)
        self.assertEqual(infos["agent_1"]["position_limit_rejections"], 1)
        self.assertEqual(infos["agent_1"]["rejected_orders"], 1)
        self.assertTrue(infos["agent_1"]["order_rejected"])
        self.assertEqual(infos["agent_1"]["rejection_reason"], "max_position_exceeded")

        # Similarly on short side: Agent 0 has inventory -40.
        # Submitting SELL of 20 would make inventory -60 < -50.
        obs, rewards, term, trunc, infos = env.step({
            "agent_0": {"order_type": 2, "side": 1, "price_offset": 0.0, "quantity": 20},
            "agent_1": {"order_type": 0, "side": 0, "price_offset": 0.0, "quantity": 0},
        })
        self.assertEqual(env.inventory["agent_0"], -40)
        self.assertEqual(infos["agent_0"]["position_limit_rejections"], 1)
        self.assertEqual(infos["agent_0"]["rejected_orders"], 1)

    def test_price_band_limit_rejection(self):
        """
        Tests that limit orders outside the starting reference price band are rejected.
        """
        # initial_price = 100.0, price_band = 0.10 => min_price = 90.0, max_price = 110.0
        env = MarketEnvironment(
            dict_spaces=True,
            initial_price=100.0,
            price_band=0.10,
            max_price_offset=20.0,
        )
        obs, infos = env.reset(seed=123)

        self.assertAlmostEqual(env.min_price, 90.0)
        self.assertAlmostEqual(env.max_price, 110.0)

        # Order above max_price (e.g. limit bid at 100 + 15 = 115) should be rejected
        obs, rewards, term, trunc, infos = env.step({
            "agent_0": {"order_type": 1, "side": 0, "price_offset": 15.0, "quantity": 10},
            "agent_1": {"order_type": 0, "side": 0, "price_offset": 0.0, "quantity": 0},
        })
        self.assertEqual(infos["agent_0"]["price_band_rejections"], 1)
        self.assertEqual(infos["agent_0"]["rejected_orders"], 1)
        self.assertEqual(infos["agent_0"]["rejection_reason"], "price_above_band")
        self.assertIsNone(env.order_book.best_bid())

        # Order below min_price (e.g. limit ask at 100 - 15 = 85) should be rejected
        obs, rewards, term, trunc, infos = env.step({
            "agent_0": {"order_type": 0, "side": 0, "price_offset": 0.0, "quantity": 0},
            "agent_1": {"order_type": 1, "side": 1, "price_offset": -15.0, "quantity": 10},
        })
        self.assertEqual(infos["agent_1"]["price_band_rejections"], 1)
        self.assertEqual(infos["agent_1"]["rejected_orders"], 1)
        self.assertEqual(infos["agent_1"]["rejection_reason"], "price_below_band")
        self.assertIsNone(env.order_book.best_ask())

    def test_market_orders_cannot_sweep_past_price_band(self):
        """
        Tests that market orders cannot execute trades beyond the price band limits.
        """
        from src.engine.order_book import Order
        env = MarketEnvironment(
            dict_spaces=True,
            initial_price=100.0,
            price_band=0.10,  # [90.0, 110.0]
        )
        env.reset(seed=1)

        # Manually seed order book with one ask inside the band (105.0) and one outside (115.0)
        ord_inside = Order("o_inside", "agent_1", Side.ASK, OrderType.LIMIT, 105.0, 10, 1)
        ord_outside = Order("o_outside", "agent_1", Side.ASK, OrderType.LIMIT, 115.0, 10, 1)
        env.order_book.add_order(ord_inside)
        env.order_book.add_order(ord_outside)
        env._all_orders[ord_inside.order_id] = ord_inside
        env._all_orders[ord_outside.order_id] = ord_outside

        # Agent 0 submits a Market Buy of 25 shares (enough to sweep both orders)
        actions = {
            "agent_0": {"order_type": 2, "side": 0, "price_offset": 0.0, "quantity": 25},
            "agent_1": {"order_type": 0, "side": 0, "price_offset": 0.0, "quantity": 0},
        }
        env.step(actions)

        # Only 10 shares at 105.0 should have matched. The 115.0 ask must NOT be filled.
        self.assertEqual(env.inventory["agent_0"], 10)
        self.assertEqual(env.order_book.best_ask().price, 115.0)
        self.assertEqual(env.order_book.best_ask().quantity, 10)

    def test_runaway_scenario_prevented(self):
        """
        Simulates an aggressive buyer attempting to inflate price and inventory over 50 steps.
        Verifies that inventory is strictly capped at max_position and mid-price is capped by price_band.
        """
        env = MarketEnvironment(
            dict_spaces=True,
            max_cycles=50,
            initial_price=100.0,
            max_position=100,
            price_band=0.10,  # collar is [90.0, 110.0]
        )
        env.reset(seed=42)

        # Agent 0 aggressively buys each step, Agent 1 is a passive seller
        for step in range(50):
            actions = {
                "agent_0": {"order_type": 2, "side": 0, "price_offset": 0.0, "quantity": 10},
                "agent_1": {"order_type": 1, "side": 1, "price_offset": 1.0, "quantity": 10},
            }
            obs, rewards, term, trunc, infos = env.step(actions)
            ref_p = env._get_reference_price()

            # Invariant 1: Inventory never exceeds max_position
            self.assertLessEqual(abs(env.inventory["agent_0"]), 100)

            # Invariant 2: Price never exceeds max_price
            self.assertLessEqual(ref_p, 110.0 + 1e-6)
            self.assertGreaterEqual(ref_p, 90.0 - 1e-6)

        # At the end of the episode:
        self.assertLessEqual(env.inventory["agent_0"], 100)
        self.assertLessEqual(env._get_reference_price(), 110.0 + 1e-6)
        # Agent 0 should have rejected orders recorded
        self.assertGreater(infos["agent_0"]["position_limit_rejections"], 0)

    def test_property_incremental_exposure_matches_brute_force(self):
        """
        Property test (Task 4a):
        After sequences of random actions across 150 steps, verify that the incremental
        exposure counters (open_bid_qty, open_ask_qty) exactly equal a brute-force
        recomputation scanning all active orders in the order book.
        """
        env = MarketEnvironment(
            agents=["agent_0", "agent_1", "agent_2", "agent_3"],
            dict_spaces=True,
            max_cycles=150,
            initial_price=100.0,
            max_position=100,
            order_ttl=15,
        )
        env.reset(seed=123)

        for step in range(150):
            actions = {
                a: env.action_space(a).sample() for a in env.agents
            }
            obs, rewards, term, trunc, infos = env.step(actions)

            # Brute-force recomputation for each agent
            for a in env.possible_agents:
                expected_open_bids = sum(
                    o.quantity
                    for o in env._all_orders.values()
                    if o.agent_id == a
                    and o.side == Side.BID
                    and o.status in (Status.OPEN, Status.PARTIALLY_FILLED)
                    and o.quantity > 0
                )
                expected_open_asks = sum(
                    o.quantity
                    for o in env._all_orders.values()
                    if o.agent_id == a
                    and o.side == Side.ASK
                    and o.status in (Status.OPEN, Status.PARTIALLY_FILLED)
                    and o.quantity > 0
                )

                self.assertEqual(
                    env.open_bid_qty[a],
                    expected_open_bids,
                    f"Step {step} Agent {a}: incremental open_bid_qty {env.open_bid_qty[a]} != brute-force {expected_open_bids}",
                )
                self.assertEqual(
                    env.open_ask_qty[a],
                    expected_open_asks,
                    f"Step {step} Agent {a}: incremental open_ask_qty {env.open_ask_qty[a]} != brute-force {expected_open_asks}",
                )
                self.assertEqual(infos[a]["open_bid_qty"], expected_open_bids)
                self.assertEqual(infos[a]["open_ask_qty"], expected_open_asks)

    def test_inventory_never_exceeds_max_position_all_agents_500_steps(self):
        """
        Invariant test (Task 4b):
        Over 500 steps with all agent types active and interacting, verifies that
        |inventory| strictly never exceeds max_position for ANY agent on ANY step.
        """
        agent_ids = ["mm_0", "as_0", "rl_0", "zi_0", "inf_0", "shock_0"]
        env = MarketEnvironment(
            agents=agent_ids,
            dict_spaces=True,
            max_cycles=500,
            initial_price=100.0,
            max_position=100,
            price_band=0.15,
            order_ttl=20,
        )
        obs, infos = env.reset(seed=999)

        # Initialize policy instances
        agents = {
            "mm_0": MarketMakerAgent("mm_0", seed=101, spread=0.5, quantity=10),
            "as_0": AvellanedaStoikovMM("as_0", seed=102, gamma=0.1, k=1.5, quantity=10),
            "rl_0": RLDirectionalAgent("rl_0", seed=103, alpha=0.1, epsilon=0.5),
            "zi_0": ZeroIntelligenceAgent("zi_0", seed=104, aggression=0.7, max_quantity=15),
            "inf_0": InformedTrader("inf_0", jump_step=250, jump_size=2.0, quantity=10),
            "shock_0": ShockAgent("shock_0", shock_step=150, n_steps=5, direction="sell", quantity=40),
        }

        rng = np.random.default_rng(2026)

        for step in range(500):
            actions = {}
            for a_id in env.agents:
                # Mix of agent policy actions and random explorative actions
                if rng.random() < 0.7:
                    actions[a_id] = agents[a_id].act(obs[a_id])
                else:
                    actions[a_id] = env.action_space(a_id).sample()

            obs, rewards, term, trunc, infos = env.step(actions)

            for a_id in env.possible_agents:
                inv = env.inventory[a_id]
                self.assertLessEqual(
                    abs(inv),
                    100,
                    f"Step {step} Agent {a_id}: |inventory| {abs(inv)} exceeded max_position (100)!",
                )

    def test_market_maker_not_locked_out_under_20_percent_rejection(self):
        """
        Availability test (Task 4c):
        Over 300 steps with default order_ttl=20, verifies that MarketMakerAgent
        is not locked out by resting exposure and maintains a position-limit
        rejection rate strictly below 20%.
        """
        agent_ids = ["mm_0", "rl_0", "zi_0", "zi_1"]
        env = MarketEnvironment(
            agents=agent_ids,
            dict_spaces=True,
            max_cycles=300,
            initial_price=100.0,
            max_position=100,
            price_band=0.10,
            order_ttl=20,
        )
        obs, infos = env.reset(seed=42)

        agents = {
            "mm_0": MarketMakerAgent("mm_0", seed=42, spread=0.5, quantity=10),
            "rl_0": RLDirectionalAgent("rl_0", seed=43, alpha=0.1, epsilon=0.5),
            "zi_0": ZeroIntelligenceAgent("zi_0", seed=44, aggression=0.7, max_quantity=10),
            "zi_1": ZeroIntelligenceAgent("zi_1", seed=45, aggression=0.6, max_quantity=10),
        }

        mm_orders_submitted = 0

        for step in range(300):
            actions = {}
            for a_id in env.agents:
                act_val = agents[a_id].act(obs[a_id])
                actions[a_id] = act_val
                if a_id == "mm_0":
                    ot, _, _, q = env._parse_action(act_val)
                    if ot is not None and q > 0:
                        mm_orders_submitted += 1

            obs, rewards, term, trunc, infos = env.step(actions)

        mm_rejections = infos["mm_0"]["position_limit_rejections"]
        rejection_rate = mm_rejections / max(1, mm_orders_submitted)

        self.assertGreater(mm_orders_submitted, 250, "MM should actively submit quotes")
        self.assertLess(
            rejection_rate,
            0.20,
            f"MM rejection rate {rejection_rate:.2%} exceeded 20% threshold ({mm_rejections}/{mm_orders_submitted})",
        )


if __name__ == "__main__":
    unittest.main()
