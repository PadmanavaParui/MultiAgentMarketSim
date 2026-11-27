"""
Stress Testing & Market Resilience Dashboard Tab for MultiAgentMarketSim.

Provides interactive visualization of market stress scenarios:
  - 'flash_crash' (large aggressive market sell orders)
  - 'informed_flow' (informed trader front-running price jump)
  - 'liquidity_withdrawal' (market makers pausing quotes and cancelling depth)

Exposes render(run_dir) with graceful missing-file handling.
"""

from __future__ import annotations

import os
import json
from typing import Any, Dict, List, Optional
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from src.dashboard.empty_state import render_empty_state, generate_stress_artifact
from src.dashboard.charts import downsample_series
from src.scenarios.stress import compute_resilience_verdict


@st.cache_data(max_entries=10)
def load_stress_report(report_path: str) -> Optional[Dict[str, Any]]:
    """Loads stress_report.json with caching."""
    if os.path.exists(report_path):
        with open(report_path, "r") as f:
            return json.load(f)
    return None


def _render_chart(fig: go.Figure):
    """Renders Plotly chart with modern width='stretch' and fallback."""
    try:
        st.plotly_chart(fig, width="stretch")
    except TypeError:
        st.plotly_chart(fig, use_container_width=True)


def _render_dataframe(df: Any):
    """Renders DataFrame with modern width='stretch' and fallback."""
    try:
        st.dataframe(df, width="stretch")
    except TypeError:
        st.dataframe(df, use_container_width=True)


def plot_stress_mid_comparison(
    baseline_mids: List[float],
    stressed_mids: List[float],
    title: str = "Mid Price: Baseline vs. Stressed",
) -> go.Figure:
    """Plots baseline vs stressed mid-price trajectories across steps."""
    if baseline_mids and len(baseline_mids) > 5000:
        baseline_mids = downsample_series(baseline_mids, 5000)
    if stressed_mids and len(stressed_mids) > 5000:
        stressed_mids = downsample_series(stressed_mids, 5000)

    fig = go.Figure()
    steps_base = list(range(1, len(baseline_mids) + 1))
    steps_stress = list(range(1, len(stressed_mids) + 1))

    if baseline_mids:
        fig.add_trace(go.Scatter(
            x=steps_base,
            y=baseline_mids,
            mode="lines",
            name="Baseline Mid Price",
            line=dict(color="#1f77b4", width=2, dash="dash"),
            hovertemplate="Step %{x}<br>Baseline Mid: $%{y:.2f}<extra></extra>",
        ))

    if stressed_mids:
        fig.add_trace(go.Scatter(
            x=steps_stress,
            y=stressed_mids,
            mode="lines+markers",
            name="Stressed Mid Price",
            line=dict(color="#d62728", width=2.5),
            marker=dict(size=4),
            hovertemplate="Step %{x}<br>Stressed Mid: $%{y:.2f}<extra></extra>",
        ))

    fig.update_layout(
        title=title,
        xaxis_title="Simulation Step",
        yaxis_title="Mid Price ($)",
        template="plotly_white",
        hovermode="x unified",
        legend=dict(x=0.02, y=0.98),
        margin=dict(l=40, r=40, t=50, b=40),
    )
    return fig


def plot_stress_spread_comparison(
    baseline_spreads: List[float],
    stressed_spreads: List[float],
    title: str = "Bid-Ask Spread: Baseline vs. Stressed",
) -> go.Figure:
    """Plots baseline vs stressed spread trajectories across steps."""
    if baseline_spreads and len(baseline_spreads) > 5000:
        baseline_spreads = downsample_series(baseline_spreads, 5000)
    if stressed_spreads and len(stressed_spreads) > 5000:
        stressed_spreads = downsample_series(stressed_spreads, 5000)

    fig = go.Figure()
    steps_base = list(range(1, len(baseline_spreads) + 1))
    steps_stress = list(range(1, len(stressed_spreads) + 1))

    if baseline_spreads:
        fig.add_trace(go.Scatter(
            x=steps_base,
            y=baseline_spreads,
            mode="lines",
            connectgaps=False,
            name="Baseline Spread",
            line=dict(color="#2ca02c", width=2, dash="dash"),
            hovertemplate="Step %{x}<br>Baseline Spread: $%{y:.2f}<extra></extra>",
        ))

    if stressed_spreads:
        fig.add_trace(go.Scatter(
            x=steps_stress,
            y=stressed_spreads,
            mode="lines+markers",
            connectgaps=False,
            name="Stressed Spread",
            line=dict(color="#ff7f0e", width=2.5),
            marker=dict(size=4),
            hovertemplate="Step %{x}<br>Stressed Spread: $%{y:.2f}<extra></extra>",
        ))

    fig.update_layout(
        title=title,
        xaxis_title="Simulation Step",
        yaxis_title="Spread ($)",
        template="plotly_white",
        hovermode="x unified",
        legend=dict(x=0.02, y=0.98),
        margin=dict(l=40, r=40, t=50, b=40),
    )
    return fig


def plot_stress_inventory_comparison(
    baseline_invs: List[int],
    stressed_invs: List[int],
    title: str = "MM Inventory Accumulation: Baseline vs. Stressed",
) -> go.Figure:
    """Plots baseline vs stressed market maker inventory positions."""
    if baseline_invs and len(baseline_invs) > 5000:
        baseline_invs = downsample_series(baseline_invs, 5000)
    if stressed_invs and len(stressed_invs) > 5000:
        stressed_invs = downsample_series(stressed_invs, 5000)

    fig = go.Figure()
    steps_base = list(range(1, len(baseline_invs) + 1))
    steps_stress = list(range(1, len(stressed_invs) + 1))

    if baseline_invs:
        fig.add_trace(go.Scatter(
            x=steps_base,
            y=baseline_invs,
            mode="lines",
            name="Baseline Inventory",
            line=dict(color="#17becf", width=2, dash="dash"),
            hovertemplate="Step %{x}<br>Baseline Inv: %{y} shares<extra></extra>",
        ))

    if stressed_invs:
        fig.add_trace(go.Scatter(
            x=steps_stress,
            y=stressed_invs,
            mode="lines+markers",
            name="Stressed Inventory",
            line=dict(color="#9467bd", width=2.5),
            marker=dict(size=4),
            hovertemplate="Step %{x}<br>Stressed Inv: %{y} shares<extra></extra>",
        ))

    fig.add_hline(y=0.0, line_dash="dash", line_color="gray", annotation_text="Neutral Inventory (0)")

    fig.update_layout(
        title=title,
        xaxis_title="Simulation Step",
        yaxis_title="Inventory (Shares)",
        template="plotly_white",
        hovermode="x unified",
        legend=dict(x=0.02, y=0.98),
        margin=dict(l=40, r=40, t=50, b=40),
    )
    return fig


def plot_mm_resilience_bar_chart(sc_data: Dict[str, Any]) -> go.Figure:
    """Grouped bar chart comparing MarketMakerAgent vs AvellanedaStoikovMM."""
    mm_res = sc_data.get("MarketMakerAgent", {})
    as_res = sc_data.get("AvellanedaStoikovMM", {})

    metrics = [
        ("Max Mid Drawdown ($)", "max_drawdown_mid"),
        ("Peak Spread ($)", "peak_spread"),
        ("MM Max |Inventory|", "mm_max_abs_inventory"),
        ("Adverse Selection Loss ($)", "adverse_selection_loss"),
    ]

    labels = [m[0] for m in metrics]
    mm_vals = [float(mm_res.get(m[1], 0.0)) for m in metrics]
    as_vals = [float(as_res.get(m[1], 0.0)) for m in metrics]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=labels,
        y=mm_vals,
        name="Standard MM (Fixed Spread)",
        marker_color="#1f77b4",
    ))
    fig.add_trace(go.Bar(
        x=labels,
        y=as_vals,
        name="Avellaneda-Stoikov MM (Inventory-Aware)",
        marker_color="#2ca02c",
    ))

    fig.update_layout(
        title="Comparative Resilience Metrics (Lower is More Resilient)",
        barmode="group",
        template="plotly_white",
        hovermode="x unified",
        margin=dict(l=40, r=40, t=50, b=40),
    )
    return fig


def _simulate_sample_trajectory(
    sc_name: str,
    mm_type: str,
    cycles: int,
    seed: int,
) -> Optional[Dict[str, Any]]:
    """Simulates a single paired episode to obtain representative trajectory curves."""
    try:
        from src.scenarios.stress import StressScenarioRunner, ScenarioConfig

        scenario_configs = {
            "flash_crash": ScenarioConfig(name="flash_crash", shock_step=15, n_steps=4, shock_quantity=40, direction="sell"),
            "informed_flow": ScenarioConfig(name="informed_flow", shock_step=18, jump_size=2.0, informed_quantity=10),
            "liquidity_withdrawal": ScenarioConfig(name="liquidity_withdrawal", shock_step=15, withdrawal_steps=6),
        }
        if sc_name not in scenario_configs:
            return None

        runner = StressScenarioRunner(episodes=1, cycles_per_episode=cycles, seed=seed)
        cfg = scenario_configs[sc_name]

        base_run = runner.run_episode(cfg, mm_type=mm_type, is_stressed=False, episode_seed=seed)
        stress_run = runner.run_episode(cfg, mm_type=mm_type, is_stressed=True, episode_seed=seed)

        return {
            "baseline": base_run,
            "stressed": stress_run,
        }
    except Exception:
        return None


def render(run_dir: Optional[str] = None):
    """
    Renders the Stress Testing tab inside the Streamlit dashboard.

    Parameters
    ----------
    run_dir : Optional[str]
        Directory path of the current simulation run (e.g. 'data/runs/<run_id>').
    """
    st.subheader("⚡ Market Microstructure Stress Testing")
    st.markdown(
        "Evaluates exchange resilience and agent performance under structural stress scenarios: "
        "**Flash Crash** (rapid aggressive selling), **Informed Flow** (informed front-running), "
        "and **Liquidity Withdrawal** (market makers pulling quotes)."
    )

    if not run_dir:
        st.info("ℹ️ No run directory specified. Please select a simulation run to inspect stress reports.")
        return

    report_path = os.path.join(run_dir, "stress_report.json")
    report = load_stress_report(report_path)

    if not report or not report.get("scenarios"):
        render_empty_state(
            tab_name="Market Microstructure Stress Testing",
            description="Evaluates exchange resilience and agent performance under structural stress scenarios: Flash Crash, Informed Flow, and Liquidity Withdrawal.",
            artifact_name="stress_report.json",
            run_dir=run_dir,
            generator_fn=generate_stress_artifact,
            button_label="⚡ Generate stress_report.json now",
            spinner_text="Simulating stress test scenarios (3 scenarios × 5 episodes × 100 cycles)...",
        )
        return

    scenarios = report.get("scenarios", {})

    # Scenario Selection
    col_sc1, col_sc2 = st.columns([1, 1])
    with col_sc1:
        scenario_options = list(scenarios.keys())
        sc_labels = {k: k.replace("_", " ").title() for k in scenario_options}
        selected_scenario = st.selectbox(
            "Select Stress Scenario",
            options=scenario_options,
            format_func=lambda k: sc_labels.get(k, k),
            key="stress_scenario_selector",
        )
    with col_sc2:
        mm_types_available = [k for k in scenarios[selected_scenario].keys() if k != "comparison"]
        selected_mm = st.selectbox(
            "Market Maker Model for Trajectories",
            options=mm_types_available,
            index=1 if "AvellanedaStoikovMM" in mm_types_available else 0,
            key="stress_mm_selector",
        )

    sc_data = scenarios[selected_scenario]
    mm_res = sc_data.get(selected_mm, {})
    comparison = sc_data.get("comparison", {})

    # Top KPI Metrics Cards
    col_m1, col_m2, col_m3, col_m4, col_m5 = st.columns(5)
    with col_m1:
        st.metric("Max Mid Drawdown", f"${mm_res.get('max_drawdown_mid', 0.0):.2f}")
    with col_m2:
        st.metric("Peak Spread", f"${mm_res.get('peak_spread', 0.0):.2f}", delta=f"{mm_res.get('spread_diff', 0.0):+.2f} vs Base", delta_color="inverse")
    with col_m3:
        st.metric("One-Sided Book Steps", f"{mm_res.get('pct_steps_one_sided', 0.0):.1f}%")
    with col_m4:
        se_pnl = mm_res.get('mm_pnl_vs_baseline_se', 0.0)
        st.metric("MM PnL vs Base", f"${mm_res.get('mm_pnl_vs_baseline', 0.0):.2f}", delta=f"±${se_pnl:.2f} SE")
    with col_m5:
        st.metric("Adverse Selection Loss", f"${mm_res.get('adverse_selection_loss', 0.0):.2f}")

    if comparison:
        mm_std = sc_data.get("MarketMakerAgent", {})
        mm_as = sc_data.get("AvellanedaStoikovMM", {})

        verdict = comparison.get("verdict")
        better_agent = comparison.get("better_agent")
        z_score = comparison.get("significance_ratio")
        combined_se = comparison.get("combined_se")

        if not verdict:
            verdict_info = compute_resilience_verdict(mm_std, mm_as)
            verdict = verdict_info["verdict"]
            better_agent = verdict_info["better_agent"]
            z_score = verdict_info["significance_ratio"]
            combined_se = verdict_info["combined_se"]

        adv_pnl = comparison.get("pnl_vs_baseline_advantage_as", 0.0)
        inv_red = comparison.get("inventory_reduction_as", 0)
        adv_loss_saved = comparison.get("adverse_selection_reduction_as", 0.0)

        std_peak_inv = mm_std.get("mm_max_abs_inventory", 0)
        as_peak_inv = mm_as.get("mm_max_abs_inventory", 0)

        z_display = f"{z_score:.2f}" if (isinstance(z_score, (int, float)) and z_score < 999.0) else "N/A"
        se_display = f"±${combined_se:.2f}" if isinstance(combined_se, (int, float)) else "N/A"

        if verdict == "clearly better":
            banner_title = f"🏆 **Resilience Comparison:** **{better_agent}** is **clearly better** (|diff| / combined SE = {z_display} ≥ 2.0)"
            alert_fn = st.success
        elif verdict == "weak evidence":
            banner_title = f"⚖️ **Resilience Comparison:** **Weak evidence** favoring **{better_agent}** (1.0 ≤ |diff| / combined SE = {z_display} < 2.0)"
            alert_fn = st.info
        else:
            banner_title = f"⚖️ **Resilience Comparison:** **No clear difference** between market makers (|diff| / combined SE = {z_display} < 1.0)"
            alert_fn = st.info

        alert_fn(
            f"{banner_title}\n\n"
            f"- **Peak Absolute Inventory:** Avellaneda-Stoikov **{as_peak_inv}** shares vs Standard MM **{std_peak_inv}** shares ({inv_red:+d} reduction)\n"
            f"- **Avellaneda-Stoikov PnL Delta Advantage:** ${adv_pnl:+.2f} (Combined SE: {se_display})\n"
            f"- **Adverse Selection Loss Mitigated:** ${adv_loss_saved:+.2f}"
        )

    st.divider()

    # Telemetry Curves: Baseline vs Stressed
    st.markdown(f"### Microstructure Trajectories ({selected_mm} under {sc_labels.get(selected_scenario, selected_scenario)})")

    # Check if pre-computed trajectory in sc_data or generate representative trajectory
    trajectories = sc_data.get("trajectories", {}).get(selected_mm)
    if not trajectories:
        config = report.get("config", {})
        cycles = config.get("cycles", 50)
        seed = config.get("seed", 42)
        traj_data = _simulate_sample_trajectory(selected_scenario, selected_mm, cycles=cycles, seed=seed)
        if traj_data:
            base_run = traj_data["baseline"]
            stress_run = traj_data["stressed"]
            base_mids = base_run.get("mids", [])
            stress_mids = stress_run.get("mids", [])
            base_spreads = base_run.get("spreads", [])
            stress_spreads = stress_run.get("spreads", [])
            base_invs = base_run.get("mm_inventories", [])
            stress_invs = stress_run.get("mm_inventories", [])
        else:
            base_mids, stress_mids = [], []
            base_spreads, stress_spreads = [], []
            base_invs, stress_invs = [], []
    else:
        base_mids = trajectories.get("baseline", {}).get("mids", [])
        stress_mids = trajectories.get("stressed", {}).get("mids", [])
        base_spreads = trajectories.get("baseline", {}).get("spreads", [])
        stress_spreads = trajectories.get("stressed", {}).get("spreads", [])
        base_invs = trajectories.get("baseline", {}).get("mm_inventories", [])
        stress_invs = trajectories.get("stressed", {}).get("mm_inventories", [])

    col_t1, col_t2 = st.columns(2)
    with col_t1:
        fig_mid = plot_stress_mid_comparison(
            base_mids, stress_mids,
            title=f"Mid Price Trajectory: Baseline vs. Stressed ({selected_scenario})"
        )
        _render_chart(fig_mid)

    with col_t2:
        fig_sp = plot_stress_spread_comparison(
            base_spreads, stress_spreads,
            title=f"Spread Trajectory: Baseline vs. Stressed ({selected_scenario})"
        )
        _render_chart(fig_sp)

    col_t3, col_t4 = st.columns(2)
    with col_t3:
        fig_inv = plot_stress_inventory_comparison(
            base_invs, stress_invs,
            title=f"MM Inventory Trajectory: Baseline vs. Stressed ({selected_scenario})"
        )
        _render_chart(fig_inv)

    with col_t4:
        fig_bars = plot_mm_resilience_bar_chart(sc_data)
        _render_chart(fig_bars)

    st.divider()

    # Comparative Metrics Scorecard Table
    st.markdown("### Scenario Metrics Scorecard")
    scorecard_rows = []
    metric_fields = [
        ("Max Mid Drawdown ($)", "max_drawdown_mid", "{:.2f}"),
        ("Peak Spread ($)", "peak_spread", "{:.2f}"),
        ("Baseline Spread ($)", "baseline_spread", "{:.2f}"),
        ("Spread Diff vs Base ($)", "spread_diff", "{:.2f}"),
        ("Spread Ratio (Stressed / Base)", "spread_ratio", "{:.2f}x"),
        ("One-Sided Book Steps (%)", "pct_steps_one_sided", "{:.2f}%"),
        ("MM Max |Inventory| (Shares)", "mm_max_abs_inventory", "{:d}"),
        ("MM Stressed PnL ($)", "mm_pnl_stressed", "{:.2f}"),
        ("MM Stressed PnL SE ($)", "mm_pnl_stressed_se", "±{:.2f}"),
        ("MM Baseline PnL ($)", "mm_pnl_baseline", "{:.2f}"),
        ("MM PnL vs Baseline ($)", "mm_pnl_vs_baseline", "{:.2f}"),
        ("MM PnL vs Baseline SE ($)", "mm_pnl_vs_baseline_se", "±{:.2f}"),
        ("Adverse-Selection Loss ($)", "adverse_selection_loss", "{:.2f}"),
        ("Recovered?", "recovered", lambda v: "✅ Yes" if v else "❌ No"),
        ("Recovery Time (Steps)", "recovery_time", lambda v: f"{v} steps" if v is not None else "Did not recover"),
    ]

    mm_std = sc_data.get("MarketMakerAgent", {})
    mm_as = sc_data.get("AvellanedaStoikovMM", {})

    for label, key, fmt in metric_fields:
        v1 = mm_std.get(key)
        v2 = mm_as.get(key)

        def _fmt(val):
            if val is None:
                return "N/A"
            if callable(fmt):
                return fmt(val)
            return fmt.format(val)

        scorecard_rows.append({
            "Resilience Metric": label,
            "MarketMakerAgent (Fixed Spread)": _fmt(v1),
            "AvellanedaStoikovMM (Inventory-Aware)": _fmt(v2),
        })

    _render_dataframe(scorecard_rows)
