"""
Dashboard module for MultiAgentMarketSim using Streamlit and Plotly.
"""

from src.dashboard.charts import (
    plot_order_book_depth,
    plot_price_and_volume,
    plot_agent_pnl,
    plot_inventory_accumulation,
    plot_best_response_trace,
    plot_convergence_diagnostics,
)

from src.dashboard import tab_stress, tab_validation, tab_nash, tab_replay

__all__ = [
    "plot_order_book_depth",
    "plot_price_and_volume",
    "plot_agent_pnl",
    "plot_inventory_accumulation",
    "plot_best_response_trace",
    "plot_convergence_diagnostics",
    "tab_stress",
    "tab_validation",
    "tab_nash",
    "tab_replay",
]
