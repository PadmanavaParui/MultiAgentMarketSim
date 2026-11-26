"""
Scenarios module for MultiAgentMarketSim.
Includes stress testing scenarios and resilience benchmarking.
"""

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

__all__ = [
    "ScenarioConfig",
    "ScenarioMetrics",
    "StressScenarioRunner",
    "compute_max_mid_drawdown",
    "compute_peak_vs_baseline_spread",
    "compute_mm_max_inventory",
    "compute_mm_pnl_vs_baseline",
    "compute_adverse_selection",
    "compute_recovery_time",
    "load_stress_report",
]
