"""
Unit and integration tests for stress testing scenarios and resilience metrics.
Verifies:
  1. Pure metric computations (max drawdown, spread delta, inventory, adverse selection, recovery).
  2. Scenario determinism and reproducibility across seeds.
  3. Execution of all 3 scenarios (flash_crash, informed_flow, liquidity_withdrawal).
  4. Comparison between MarketMakerAgent and AvellanedaStoikovMM.
  5. Persistence and schema of stress_report.json.
"""

from __future__ import annotations

import os
import shutil
import tempfile
import unittest
from dataclasses import asdict
import numpy as np

from src.engine.matching_engine import Trade
from src.engine.order_book import Order, Side, OrderType
from src.scenarios.stress import (
    ScenarioConfig,
    ScenarioMetrics,
    StressScenarioRunner,
    compute_max_mid_drawdown,
    compute_peak_vs_baseline_spread,
    compute_mm_max_inventory,
    compute_mm_pnl_vs_baseline,
    compute_adverse_selection,
    compute_recovery_time,
    load_stress_report,
)


class TestStressMetrics(unittest.TestCase):
    """Unit tests for pure resilience metric calculations."""

    def test_compute_max_mid_drawdown_synthetic(self):
        """Test peak-to-trough max drawdown on synthetic price paths."""
        # Flat path -> 0.0 drawdown
        self.assertEqual(compute_max_mid_drawdown([100.0, 100.0, 100.0]), 0.0)

        # Monotonically increasing -> 0.0 drawdown
        self.assertEqual(compute_max_mid_drawdown([100.0, 102.0, 105.0]), 0.0)

        # Peak at 110, drops to 95 -> drawdown is 15.0
        prices = [100.0, 110.0, 105.0, 95.0, 108.0]
        self.assertEqual(compute_max_mid_drawdown(prices), 15.0)

        # Two drawdowns: 100 -> 90 (drop 10), then 120 -> 100 (drop 20) -> max is 20.0
        prices2 = [100.0, 90.0, 120.0, 100.0, 115.0]
        self.assertEqual(compute_max_mid_drawdown(prices2), 20.0)

        # Handles None and NaNs
        prices_dirty = [100.0, None, 110.0, np.nan, 95.0]
        self.assertEqual(compute_max_mid_drawdown(prices_dirty), 15.0)

        # Single element or empty
        self.assertEqual(compute_max_mid_drawdown([100.0]), 0.0)
        self.assertEqual(compute_max_mid_drawdown([]), 0.0)

    def test_compute_peak_vs_baseline_spread(self):
        """Test peak spread vs baseline average spread computation without sentinel."""
        stressed = [0.5, 1.0, 5.0, 2.0]
        baseline = [0.5, 0.5, 0.6, 0.4]

        res = compute_peak_vs_baseline_spread(stressed, baseline)
        self.assertEqual(res["peak_spread"], 5.0)
        self.assertEqual(res["baseline_spread"], 0.5)
        self.assertAlmostEqual(res["spread_diff"], 4.5)
        self.assertAlmostEqual(res["spread_ratio"], 10.0)
        self.assertEqual(res["pct_steps_one_sided"], 0.0)

        # Handles None values without 20.0 sentinel and computes pct_steps_one_sided
        stressed_dirty = [0.5, None, 1.2, None]
        res_dirty = compute_peak_vs_baseline_spread(stressed_dirty, [1.0])
        self.assertEqual(res_dirty["peak_spread"], 1.2)
        self.assertEqual(res_dirty["pct_steps_one_sided"], 50.0)

    def test_compute_mm_max_inventory(self):
        """Test peak absolute inventory tracking."""
        self.assertEqual(compute_mm_max_inventory([]), 0)
        self.assertEqual(compute_mm_max_inventory([0, 10, -25, 15, -40, 20]), 40)
        self.assertEqual(compute_mm_max_inventory([-15, -30, -5]), 30)

    def test_compute_mm_pnl_vs_baseline(self):
        """Test PnL difference calculation between stressed and baseline runs."""
        # Under stress, MM lost money compared to baseline
        self.assertEqual(compute_mm_pnl_vs_baseline(stressed_pnl=-50.0, baseline_pnl=30.0), -80.0)
        # Stressed outperformed baseline
        self.assertEqual(compute_mm_pnl_vs_baseline(stressed_pnl=40.0, baseline_pnl=25.0), 15.0)

    def test_compute_adverse_selection_deterministic(self):
        """Test exact mathematical computation of adverse selection loss."""
        # MM sells 20 shares @ $100 to informed trader.
        # Terminal price jumps to $105.
        # MM received cash +$2000, inventory is -20.
        # At terminal price: pnl = 2000 + (-20 * 105) = 2000 - 2100 = -100.
        # Adverse selection loss = -pnl = +100.
        orders = {
            "ord_inf_buy": Order("ord_inf_buy", "inf_0", Side.BID, OrderType.MARKET, price=100.0, quantity=20, timestamp_step=1),
            "ord_mm_sell": Order("ord_mm_sell", "mm_0", Side.ASK, OrderType.LIMIT, price=100.0, quantity=20, timestamp_step=1),
        }
        trade = Trade("t1", "run1", "ord_inf_buy", "ord_mm_sell", price=100.0, quantity=20, timestamp_step=1)

        res = compute_adverse_selection(
            trades=[trade],
            all_orders=orders,
            mm_id="mm_0",
            target_agent_id="inf_0",
            terminal_price=105.0,
        )

        self.assertEqual(res["trades_count"], 1)
        self.assertEqual(res["inventory_accumulated"], -20)
        self.assertEqual(res["adverse_selection_pnl"], -100.0)
        self.assertEqual(res["adverse_selection_loss"], 100.0)

    def test_compute_recovery_time(self):
        """Test recovery detection and step delta."""
        # Baseline targets
        base_mid = 100.0
        base_sp = 1.0

        # Shock occurs at steps 2-4 (shock_end_step = 5).
        # Step 5: mid=95.0, sp=4.0 -> not recovered
        # Step 6: mid=96.0, sp=3.0 -> not recovered
        # Step 7: mid=99.5, sp=1.2 -> recovered! (within tolerance_mid=1.0 and spread <= 1.5 * 1.0 + 0.1)
        mids = [100.0, 95.0, 94.0, 94.0, 95.0, 96.0, 99.5, 100.1]
        spreads = [1.0, 4.0, 5.0, 5.0, 4.0, 3.0, 1.2, 1.0]

        rec, dt = compute_recovery_time(
            stressed_mids=mids,
            stressed_spreads=spreads,
            baseline_mid_target=base_mid,
            baseline_spread_target=base_sp,
            shock_end_step=5,
            tolerance_mid=1.0,
            tolerance_spread_factor=1.5,
        )

        self.assertTrue(rec)
        self.assertEqual(dt, 2)  # Step 7 is 2 steps after step 5

        # Does not recover if book is one-sided (spread is None) even if mid is recovered
        mids_onesided = [100.0, 95.0, 94.0, 94.0, 100.0, 100.0]
        spreads_onesided = [1.0, 4.0, 5.0, 5.0, None, None]
        rec_os, dt_os = compute_recovery_time(
            stressed_mids=mids_onesided,
            stressed_spreads=spreads_onesided,
            baseline_mid_target=base_mid,
            baseline_spread_target=base_sp,
            shock_end_step=5,
        )
        self.assertFalse(rec_os)
        self.assertIsNone(dt_os)

        # Never recovers case
        never_mids = [100.0, 90.0, 90.0, 90.0, 90.0]
        never_spreads = [1.0, 5.0, 5.0, 5.0, 5.0]
        rec_f, dt_f = compute_recovery_time(
            stressed_mids=never_mids,
            stressed_spreads=never_spreads,
            baseline_mid_target=100.0,
            baseline_spread_target=1.0,
            shock_end_step=3,
        )
        self.assertFalse(rec_f)
        self.assertIsNone(dt_f)


class TestStressDeterminismAndScenarios(unittest.TestCase):
    """Integration tests for scenario execution, determinism, and comparison."""

    def setUp(self):
        self.test_dir = tempfile.mkdtemp(prefix="test_stress_")

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_scenario_determinism_flash_crash(self):
        """Verify flash_crash scenario produces bit-identical metrics on identical seeds."""
        runner = StressScenarioRunner(episodes=1, cycles_per_episode=20, seed=12345, output_dir=self.test_dir)
        cfg = ScenarioConfig(name="flash_crash", shock_step=8, n_steps=3, shock_quantity=30)

        res1 = runner.evaluate_scenario(cfg, "MarketMakerAgent")
        res2 = runner.evaluate_scenario(cfg, "MarketMakerAgent")

        self.assertEqual(asdict(res1), asdict(res2))

    def test_scenario_determinism_informed_flow(self):
        """Verify informed_flow scenario produces bit-identical metrics on identical seeds."""
        runner = StressScenarioRunner(episodes=1, cycles_per_episode=20, seed=12345, output_dir=self.test_dir)
        cfg = ScenarioConfig(name="informed_flow", shock_step=10, jump_size=2.0, informed_quantity=10)

        res1 = runner.evaluate_scenario(cfg, "AvellanedaStoikovMM")
        res2 = runner.evaluate_scenario(cfg, "AvellanedaStoikovMM")

        self.assertEqual(asdict(res1), asdict(res2))

    def test_scenario_determinism_liquidity_withdrawal(self):
        """Verify liquidity_withdrawal scenario produces bit-identical metrics on identical seeds."""
        runner = StressScenarioRunner(episodes=1, cycles_per_episode=20, seed=12345, output_dir=self.test_dir)
        cfg = ScenarioConfig(name="liquidity_withdrawal", shock_step=8, withdrawal_steps=5)

        res1 = runner.evaluate_scenario(cfg, "MarketMakerAgent")
        res2 = runner.evaluate_scenario(cfg, "MarketMakerAgent")

        self.assertEqual(asdict(res1), asdict(res2))

    def test_run_suite_and_report_persistence(self):
        """Test full suite execution across all 3 scenarios, comparison, and stress_report.json."""
        runner = StressScenarioRunner(episodes=1, cycles_per_episode=25, seed=42, output_dir=self.test_dir)
        report = runner.run_suite(
            scenarios=["flash_crash", "informed_flow", "liquidity_withdrawal"],
            mm_types=["MarketMakerAgent", "AvellanedaStoikovMM"],
            run_id="integration_test_run",
        )

        # 1. Report structure checks
        self.assertIn("run_id", report)
        self.assertEqual(report["run_id"], "integration_test_run")
        self.assertIn("scenarios", report)

        scenarios = report["scenarios"]
        self.assertIn("flash_crash", scenarios)
        self.assertIn("informed_flow", scenarios)
        self.assertIn("liquidity_withdrawal", scenarios)

        # 2. Check each scenario has both MMs and a comparison block
        for sc_name in ["flash_crash", "informed_flow", "liquidity_withdrawal"]:
            sc_data = scenarios[sc_name]
            self.assertIn("MarketMakerAgent", sc_data)
            self.assertIn("AvellanedaStoikovMM", sc_data)
            self.assertIn("comparison", sc_data)

            # Check new schema fields
            for mm_key in ["MarketMakerAgent", "AvellanedaStoikovMM"]:
                mm_obj = sc_data[mm_key]
                self.assertIn("pct_steps_one_sided", mm_obj)
                self.assertIn("mm_pnl_vs_baseline_se", mm_obj)
                self.assertIn("mm_pnl_stressed_se", mm_obj)
                self.assertIsInstance(mm_obj["pct_steps_one_sided"], (int, float))
                self.assertIsInstance(mm_obj["mm_pnl_vs_baseline_se"], (int, float))
                self.assertIsInstance(mm_obj["mm_pnl_stressed_se"], (int, float))

            comp = sc_data["comparison"]
            self.assertIn("pnl_vs_baseline_advantage_as", comp)
            self.assertIn("inventory_reduction_as", comp)
            self.assertIn("adverse_selection_reduction_as", comp)
            self.assertIn("more_resilient_agent", comp)

        # 3. Check file persistence and loading
        loaded_report = load_stress_report("integration_test_run", output_dir=self.test_dir)
        self.assertEqual(loaded_report["run_id"], "integration_test_run")
        self.assertEqual(loaded_report["scenarios"]["flash_crash"]["MarketMakerAgent"]["scenario_name"], "flash_crash")

        # 4. Check FileNotFoundError on non-existent run_id
        with self.assertRaises(FileNotFoundError):
            load_stress_report("non_existent_run_id_xyz", output_dir=self.test_dir)


if __name__ == "__main__":
    unittest.main()
