"""
Tests for Phase 4: Dashboard & Charts (Streamlit + Plotly).
Covers TEST-010: Dashboard chart builders render completed run data correctly.
"""

import unittest
import plotly.graph_objects as go

from src.dashboard.charts import (
    plot_order_book_depth,
    plot_price_and_volume,
    plot_agent_pnl,
    plot_inventory_accumulation,
    plot_best_response_trace,
    plot_convergence_diagnostics,
    plot_training_curve,
)
from src.metrics.logger import MetricsLogger


class TestDashboard(unittest.TestCase):
    def test_test_010_chart_builders_render_completed_run(self):
        """
        TEST-010: Verifies that chart builder functions return valid Plotly
        figure objects with complete data for display in the dashboard.
        """
        # 1. Order book depth
        bids = [(99.0, 100), (98.5, 200)]
        asks = [(101.0, 150), (101.5, 250)]
        fig_depth = plot_order_book_depth(bids, asks, best_bid=99.0, best_ask=101.0)
        self.assertIsInstance(fig_depth, go.Figure)
        self.assertEqual(len(fig_depth.data), 2)  # Bids and Asks traces

        # 2. Price and volume
        price_series = [
            {"step": 1, "price": 100.0, "quantity": 10},
            {"step": 2, "price": 100.5, "quantity": 15},
            {"step": 3, "price": 99.8, "quantity": 25},
        ]
        fig_pv = plot_price_and_volume(price_series)
        self.assertIsInstance(fig_pv, go.Figure)
        self.assertEqual(len(fig_pv.data), 2)  # Price line + Volume bars

        # 3. Agent PnL
        agent_metrics = [
            {"agent_id": "rl_0", "episode": 1, "pnl": 12.5, "inventory": 5},
            {"agent_id": "rl_0", "episode": 2, "pnl": 25.0, "inventory": 0},
            {"agent_id": "zi_0", "episode": 1, "pnl": -12.5, "inventory": -5},
            {"agent_id": "zi_0", "episode": 2, "pnl": -25.0, "inventory": 0},
        ]
        fig_pnl = plot_agent_pnl(agent_metrics)
        self.assertIsInstance(fig_pnl, go.Figure)
        self.assertEqual(len(fig_pnl.data), 2)  # rl_0 and zi_0 traces

        # 4. Inventory accumulation
        fig_inv = plot_inventory_accumulation(agent_metrics)
        self.assertIsInstance(fig_inv, go.Figure)
        self.assertEqual(len(fig_inv.data), 2)

        # 5. Best-response trace
        trace = [
            {"step": 0, "p": [0.5, 0.5], "q": [0.5, 0.5]},
            {"step": 1, "p": [0.7, 0.3], "q": [0.6, 0.4]},
            {"step": 2, "p": [0.9, 0.1], "q": [0.8, 0.2]},
        ]
        fig_trace = plot_best_response_trace(trace)
        self.assertIsInstance(fig_trace, go.Figure)
        self.assertEqual(len(fig_trace.data), 4)  # p[0], p[1], q[0], q[1]

        # 6. Convergence diagnostics
        rewards = [10.0, 15.0, 18.0, 22.0]
        td_errors = [2.5, 1.8, 1.2, 0.8]
        fig_conv = plot_convergence_diagnostics(rewards, td_errors)
        self.assertIsInstance(fig_conv, go.Figure)
        self.assertEqual(len(fig_conv.data), 2)

    def test_empty_chart_states(self):
        """Tests that chart builders safely return empty figures without crashing."""
        fig_empty_pv = plot_price_and_volume([])
        self.assertIsInstance(fig_empty_pv, go.Figure)

        fig_empty_pnl = plot_agent_pnl([])
        self.assertIsInstance(fig_empty_pnl, go.Figure)

        fig_empty_trace = plot_best_response_trace([])
        self.assertIsInstance(fig_empty_trace, go.Figure)

        fig_empty_train = plot_training_curve([])
        self.assertIsInstance(fig_empty_train, go.Figure)

    def test_plot_training_curve(self):
        """Tests that plot_training_curve renders subplots, moving average, and secondary curves."""
        rewards = [5.0, 8.0, 12.0, 10.0, 15.0, 18.0]
        epsilons = [1.0, 0.95, 0.90, 0.85, 0.80, 0.75]
        td_errors = [1.2, 0.9, 0.7, 0.5, 0.4, 0.3]

        # Full figure with rewards, epsilons, and td_errors
        fig_full = plot_training_curve(rewards, epsilons, td_errors, window=3)
        self.assertIsInstance(fig_full, go.Figure)
        # Should have raw reward, rolling mean, epsilon, and td error traces (4 traces)
        self.assertGreaterEqual(len(fig_full.data), 4)

        # Rewards only figure
        fig_rewards_only = plot_training_curve(rewards)
        self.assertIsInstance(fig_rewards_only, go.Figure)
        self.assertEqual(len(fig_rewards_only.data), 2)  # raw + rolling mean

    def test_app_module_imports(self):
        """Tests that src.dashboard.app and modular tabs import without syntax or dependency errors."""
        import src.dashboard.app as app
        import src.dashboard.tab_stress as tab_stress
        import src.dashboard.tab_validation as tab_validation
        import src.dashboard.tab_nash as tab_nash
        import src.dashboard.tab_replay as tab_replay

        self.assertTrue(hasattr(app, "main"))
        self.assertTrue(hasattr(tab_stress, "render"))
        self.assertTrue(hasattr(tab_validation, "render"))
        self.assertTrue(hasattr(tab_nash, "render"))
        self.assertTrue(hasattr(tab_replay, "render"))

    def test_tab_missing_files_graceful_handling(self):
        """Verifies that all tab render() functions handle missing files gracefully without raising exceptions."""
        from src.dashboard import tab_stress, tab_validation, tab_nash, tab_replay

        # None run_dir
        tab_stress.render(None)
        tab_validation.render(None)
        tab_nash.render(None)
        tab_replay.render(None)

        # Non-existent run_dir
        non_existent = "data/runs/definitely_not_a_real_run_dir_12345"
        tab_stress.render(non_existent)
        tab_validation.render(non_existent)
        tab_nash.render(non_existent)
        tab_replay.render(non_existent)

    def test_tab_stress_chart_builders(self):
        """Verifies that tab_stress chart builders produce valid Plotly figures."""
        from src.dashboard.tab_stress import (
            plot_stress_mid_comparison,
            plot_stress_spread_comparison,
            plot_stress_inventory_comparison,
            plot_mm_resilience_bar_chart,
        )

        fig_mid = plot_stress_mid_comparison([100.0, 101.0, 100.5], [100.0, 95.0, 92.0])
        self.assertIsInstance(fig_mid, go.Figure)
        self.assertEqual(len(fig_mid.data), 2)

        fig_sp = plot_stress_spread_comparison([0.5, 0.5, 0.6], [0.5, 2.5, 4.0])
        self.assertIsInstance(fig_sp, go.Figure)
        self.assertEqual(len(fig_sp.data), 2)

        fig_inv = plot_stress_inventory_comparison([0, 5, -2], [0, -20, -40])
        self.assertIsInstance(fig_inv, go.Figure)
        self.assertEqual(len(fig_inv.data), 2)

        sc_data = {
            "MarketMakerAgent": {"max_drawdown_mid": 5.2, "peak_spread": 4.5, "mm_max_abs_inventory": 30, "adverse_selection_loss": 85.0},
            "AvellanedaStoikovMM": {"max_drawdown_mid": 3.1, "peak_spread": 3.8, "mm_max_abs_inventory": 18, "adverse_selection_loss": 32.0},
        }
        fig_bars = plot_mm_resilience_bar_chart(sc_data)
        self.assertIsInstance(fig_bars, go.Figure)
        self.assertEqual(len(fig_bars.data), 2)

    def test_tab_validation_chart_builders(self):
        """Verifies that tab_validation chart builders produce valid Plotly figures."""
        import numpy as np
        from src.dashboard.tab_validation import (
            plot_return_histogram_with_normal,
            plot_qq_plot,
            plot_return_acf_bar_chart,
            plot_squared_return_acf_bar_chart,
        )

        returns = np.array([-0.03, -0.015, -0.005, 0.0, 0.002, 0.008, 0.02, 0.035, -0.01, 0.005])
        fig_hist = plot_return_histogram_with_normal(returns)
        self.assertIsInstance(fig_hist, go.Figure)
        self.assertEqual(len(fig_hist.data), 2)  # empirical hist + normal curve

        fig_qq = plot_qq_plot(returns)
        self.assertIsInstance(fig_qq, go.Figure)
        self.assertEqual(len(fig_qq.data), 2)  # quantiles + normal reference line

        ret_lags = {"lag_1": -0.02, "lag_2": 0.01, "lag_3": 0.04, "lag_4": -0.03, "lag_5": 0.01}
        fig_ret_acf = plot_return_acf_bar_chart(ret_lags, threshold=0.10)
        self.assertIsInstance(fig_ret_acf, go.Figure)

        sq_lags = {f"lag_{i}": 0.2 / i for i in range(1, 11)}
        fig_sq_acf = plot_squared_return_acf_bar_chart(sq_lags)
        self.assertIsInstance(fig_sq_acf, go.Figure)
        self.assertEqual(len(fig_sq_acf.data), 2)  # bar + trend line

    def test_tab_nash_chart_builders(self):
        """Verifies that tab_nash payoff heatmap produces valid Plotly figures with standard errors."""
        from src.dashboard.tab_nash import plot_payoff_heatmap

        matrix = [[25.0, -10.0], [5.0, 15.0]]
        se = [[1.5, 2.0], [0.8, 1.2]]
        strats = ["Strategy 1", "Strategy 2"]

        fig_heat = plot_payoff_heatmap(matrix, standard_errors=se, strategies=strats)
        self.assertIsInstance(fig_heat, go.Figure)
        self.assertEqual(len(fig_heat.data), 1)

        # Without standard errors
        fig_heat_no_se = plot_payoff_heatmap(matrix, standard_errors=None, strategies=strats)
        self.assertIsInstance(fig_heat_no_se, go.Figure)

    def test_tab_renders_with_populated_temp_files(self):
        """Verifies that tab modules execute their render() pipeline with valid mock JSON data files."""
        import tempfile
        import json
        import os
        from src.dashboard import tab_stress, tab_validation, tab_nash

        with tempfile.TemporaryDirectory() as tmp_dir:
            # 1. Stress report mock
            stress_report = {
                "run_id": "test_stress",
                "config": {"cycles": 20, "seed": 42},
                "scenarios": {
                    "flash_crash": {
                        "MarketMakerAgent": {
                            "scenario_name": "flash_crash",
                            "mm_type": "MarketMakerAgent",
                            "max_drawdown_mid": 5.0,
                            "peak_spread": 4.0,
                            "baseline_spread": 0.5,
                            "spread_diff": 3.5,
                            "spread_ratio": 8.0,
                            "pct_steps_one_sided": 15.0,
                            "mm_max_abs_inventory": 40,
                            "mm_pnl_stressed": -50.0,
                            "mm_pnl_stressed_se": 3.5,
                            "mm_pnl_baseline": 20.0,
                            "mm_pnl_vs_baseline": -70.0,
                            "mm_pnl_vs_baseline_se": 4.2,
                            "adverse_selection_loss": 60.0,
                            "recovered": True,
                            "recovery_time": 8,
                        },
                        "AvellanedaStoikovMM": {
                            "scenario_name": "flash_crash",
                            "mm_type": "AvellanedaStoikovMM",
                            "max_drawdown_mid": 3.0,
                            "peak_spread": 2.5,
                            "baseline_spread": 0.5,
                            "spread_diff": 2.0,
                            "spread_ratio": 5.0,
                            "pct_steps_one_sided": 5.0,
                            "mm_max_abs_inventory": 15,
                            "mm_pnl_stressed": -10.0,
                            "mm_pnl_stressed_se": 1.8,
                            "mm_pnl_baseline": 25.0,
                            "mm_pnl_vs_baseline": -35.0,
                            "mm_pnl_vs_baseline_se": 2.1,
                            "adverse_selection_loss": 20.0,
                            "recovered": True,
                            "recovery_time": 4,
                        },
                        "comparison": {
                            "pnl_vs_baseline_advantage_as": 35.0,
                            "inventory_reduction_as": 25,
                            "adverse_selection_reduction_as": 40.0,
                            "more_resilient_agent": "AvellanedaStoikovMM",
                        },
                    }
                },
            }
            with open(os.path.join(tmp_dir, "stress_report.json"), "w") as f:
                json.dump(stress_report, f)

            tab_stress.render(tmp_dir)

            # 2. Validation report mock
            val_report = {
                "overall_passed": True,
                "summary": {"passed_checks": 5, "total_checks": 5},
                "resample_interval": 5,
                "num_resampled_observations": 100,
                "stylized_facts": {
                    "excess_kurtosis": {"passed": True, "raw_kurtosis": 4.5, "criterion": "raw > 3.0", "description": "Heavy tails"},
                    "hill_tail_index": {"passed": True, "tail_index": 3.2, "criterion": "2.0 <= alpha <= 5.0", "description": "Tail index"},
                    "return_autocorrelation": {"passed": True, "lags": {"lag_1": 0.02, "lag_2": -0.01}, "threshold": 0.10, "criterion": "|ACF| < 0.10", "description": "No linear ACF"},
                    "volatility_clustering": {"passed": True, "lags": {"lag_1": 0.12, "lag_2": 0.08}, "criterion": "Decaying squared ACF", "description": "ARCH effect"},
                    "spread_volatility_correlation": {"passed": True, "correlation": 0.35, "criterion": "corr > 0", "description": "Spread vol corr"},
                },
            }
            with open(os.path.join(tmp_dir, "validation_report.json"), "w") as f:
                json.dump(val_report, f)

            tab_validation.render(tmp_dir)

            # 3. Equilibrium analysis mock
            eq_analysis = {
                "run_id": "test_eq",
                "method": "support_enumeration",
                "strategies": ["MM", "AS_MM"],
                "payoff_matrix": [[10.0, -2.0], [4.0, 8.0]],
                "standard_errors": [[1.0, 0.5], [0.4, 0.8]],
                "equilibrium_candidates": [
                    {
                        "equilibrium_id": 1,
                        "player_1_strategy": [1.0, 0.0],
                        "player_2_strategy": [1.0, 0.0],
                        "player_1_payoff": 10.0,
                        "player_2_payoff": 10.0,
                        "is_pure": True,
                    }
                ],
                "best_response_trace": [
                    {"step": 0, "p": [0.5, 0.5], "q": [0.5, 0.5]},
                    {"step": 1, "p": [1.0, 0.0], "q": [1.0, 0.0]},
                ],
            }
            with open(os.path.join(tmp_dir, "equilibrium_analysis.json"), "w") as f:
                json.dump(eq_analysis, f)

            tab_nash.render(tmp_dir)

            # 4. Book snapshots and trades mock for replay
            snapshots = [
                {
                    "episode": 1,
                    "step": 5,
                    "best_bid": 99.5,
                    "best_ask": 100.5,
                    "mid": 100.0,
                    "spread": 1.0,
                    "bids": [[99.5, 20], [99.0, 30]],
                    "asks": [[100.5, 25], [101.0, 35]],
                    "inventories": {"mm_0": 0, "rl_0": 0},
                    "pnls": {"mm_0": 0.0, "rl_0": 0.0},
                },
                {
                    "episode": 1,
                    "step": 10,
                    "best_bid": 99.0,
                    "best_ask": 100.0,
                    "mid": 99.5,
                    "spread": 1.0,
                    "bids": [[99.0, 15], [98.5, 40]],
                    "asks": [[100.0, 20], [100.5, 30]],
                    "inventories": {"mm_0": -10, "rl_0": 10},
                    "pnls": {"mm_0": 5.0, "rl_0": -5.0},
                },
            ]
            with open(os.path.join(tmp_dir, "book_snapshots.json"), "w") as f:
                json.dump(snapshots, f)

            trades = [
                {"trade_id": "trd_1", "step": 5, "price": 100.0, "quantity": 10},
                {"trade_id": "trd_2", "step": 8, "price": 99.5, "quantity": 5},
            ]
            with open(os.path.join(tmp_dir, "trades.json"), "w") as f:
                json.dump(trades, f)

            from src.dashboard import tab_replay
            tab_replay.render(tmp_dir)

    def test_tab_replay_chart_builders(self):
        """Verifies that tab_replay Bookmap heatmap and live order book produce valid Plotly figures."""
        from src.dashboard.tab_replay import (
            plot_bookmap_depth_heatmap,
            plot_live_order_book,
            detect_flash_crash_step,
        )

        snapshots = [
            {
                "episode": 1,
                "step": 5,
                "best_bid": 99.5,
                "best_ask": 100.5,
                "mid": 100.0,
                "spread": 1.0,
                "bids": [[99.5, 10], [99.0, 20]],
                "asks": [[100.5, 15], [101.0, 25]],
                "inventories": {"mm_0": 0},
                "pnls": {"mm_0": 0.0},
            },
            {
                "episode": 1,
                "step": 10,
                "best_bid": 99.0,
                "best_ask": 100.0,
                "mid": 99.5,
                "spread": 1.0,
                "bids": [[99.0, 30], [98.5, 10]],
                "asks": [[100.0, 10], [100.5, 20]],
                "inventories": {"mm_0": 10},
                "pnls": {"mm_0": 12.0},
            },
            {
                "episode": 1,
                "step": 15,
                "best_bid": 96.0,
                "best_ask": 97.5,
                "mid": 96.75,
                "spread": 1.5,
                "bids": [[96.0, 50]],
                "asks": [[97.5, 40]],
                "inventories": {"mm_0": 40},
                "pnls": {"mm_0": -45.0},
            },
        ]
        trades = [
            {"trade_id": "trd_1", "step": 6, "price": 99.5, "quantity": 10},
            {"trade_id": "trd_2", "step": 15, "price": 96.0, "quantity": 40},
        ]

        # 1. Bookmap Heatmap with trades and flash-crash marker
        fig_bookmap = plot_bookmap_depth_heatmap(
            snapshots=snapshots,
            trades=trades,
            flash_crash_step=15,
            downsample_stride=1,
            price_bin_size=0.5,
        )
        self.assertIsInstance(fig_bookmap, go.Figure)
        # Should have heatmap trace + mid price line trace + executed trade bubbles trace
        self.assertGreaterEqual(len(fig_bookmap.data), 3)

        # 2. Live Order Book Ladder
        fig_ladder = plot_live_order_book(snapshots[0])
        self.assertIsInstance(fig_ladder, go.Figure)
        self.assertEqual(len(fig_ladder.data), 2)  # Bids and Asks bars

        # 3. Detect flash crash step
        step_found = detect_flash_crash_step(
            run_dir=None,
            snapshots=[{"agents": {"shock_0": {}}}],
            trades=[{"buy_order_id": "ord_shock_1", "step": 12}],
        )
        self.assertEqual(step_found, 12)

    def test_streamlit_apptest_demo_main_no_not_found_warnings(self):
        """Verifies via Streamlit AppTest that default demo_main has no 'not found' warnings across tabs."""
        import os
        from streamlit.testing.v1 import AppTest

        app_file = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src", "dashboard", "app.py"))
        at = AppTest.from_file(app_file)
        at.run(timeout=15)
        self.assertEqual(len(at.exception), 0, f"AppTest raised an exception: {at.exception}")

        # Ensure no tab displays a missing data warning
        not_found_warnings = [
            w.value for w in at.warning if "not found" in w.value.lower()
        ]
        self.assertEqual(
            not_found_warnings,
            [],
            f"Unexpected 'not found' warning(s) in dashboard: {not_found_warnings}",
        )


if __name__ == "__main__":
    unittest.main()
