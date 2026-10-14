import unittest
import numpy as np
from pettingzoo.test import parallel_api_test, parallel_seed_test
from pettingzoo.test.seed_test import data_equivalence

from src.env.market_env import MarketEnvironment, parallel_env
from src.engine.order_book import Side, OrderType


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


if __name__ == "__main__":
    unittest.main()
