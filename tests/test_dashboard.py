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

    def test_app_module_imports(self):
        """Tests that src.dashboard.app imports without syntax or dependency errors."""
        import src.dashboard.app as app
        self.assertTrue(hasattr(app, "main"))


if __name__ == "__main__":
    unittest.main()
