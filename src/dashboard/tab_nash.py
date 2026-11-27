"""
Game Theory & Nash Equilibrium Analysis Dashboard Tab for MultiAgentMarketSim.

Provides interactive visualization of game-theoretic tournament dynamics:
  - Head-to-head normal-form payoff matrix (A) as an annotated heatmap with standard errors
  - Pure and mixed Nash equilibrium candidates solved via support enumeration
  - Best-response dynamics trajectory (fictitious play convergence)

Exposes render(run_dir) with graceful missing-file handling.
"""

from __future__ import annotations

import os
import json
from typing import Any, Dict, List, Optional
import numpy as np
import plotly.graph_objects as go
import streamlit as st

from src.dashboard.charts import plot_best_response_trace
from src.dashboard.empty_state import render_empty_state, generate_tournament_artifact


@st.cache_data(max_entries=10)
def load_equilibrium_analysis(eq_path: str) -> Optional[Dict[str, Any]]:
    """Loads equilibrium_analysis.json with caching."""
    if os.path.exists(eq_path):
        with open(eq_path, "r") as f:
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


def plot_payoff_heatmap(
    payoff_matrix: List[List[float]],
    standard_errors: Optional[List[List[float]]] = None,
    strategies: Optional[List[str]] = None,
    title: str = "Head-to-Head Payoff Matrix Heatmap (Player X Mean PnL ± SE)",
) -> go.Figure:
    """
    Renders an interactive heatmap of the 2-player normal-form payoff matrix.
    Each cell is annotated with mean payoff and standard error: 'mean +- SE'.
    """
    fig = go.Figure()
    if not payoff_matrix or not payoff_matrix[0]:
        fig.update_layout(title="No Payoff Matrix Data Available")
        return fig

    n_rows = len(payoff_matrix)
    n_cols = len(payoff_matrix[0])

    if not strategies or len(strategies) < n_cols:
        col_labels = [f"Strat {j+1}" for j in range(n_cols)]
        row_labels = [f"Strat {i+1}" for i in range(n_rows)]
    else:
        col_labels = strategies[:n_cols]
        row_labels = strategies[:n_rows]

    # Generate cell text annotations: "mean +- SE"
    text_matrix = []
    hover_matrix = []
    for i in range(n_rows):
        text_row = []
        hover_row = []
        for j in range(n_cols):
            val = payoff_matrix[i][j]
            se = standard_errors[i][j] if standard_errors and i < len(standard_errors) and j < len(standard_errors[i]) else None
            if se is not None and not np.isnan(se):
                text_row.append(f"{val:.2f} ± {se:.2f}")
                hover_row.append(
                    f"Row Strategy: {row_labels[i]}<br>"
                    f"Col Strategy: {col_labels[j]}<br>"
                    f"Mean Payoff: ${val:.2f}<br>"
                    f"Std Error: ±${se:.2f}"
                )
            else:
                text_row.append(f"{val:.2f}")
                hover_row.append(
                    f"Row Strategy: {row_labels[i]}<br>"
                    f"Col Strategy: {col_labels[j]}<br>"
                    f"Mean Payoff: ${val:.2f}"
                )
        text_matrix.append(text_row)
        hover_matrix.append(hover_row)

    z_vals = [[float(v) for v in row] for row in payoff_matrix]

    fig.add_trace(go.Heatmap(
        z=z_vals,
        x=col_labels,
        y=row_labels,
        text=text_matrix,
        texttemplate="%{text}",
        textfont=dict(size=13, color="black"),
        customdata=hover_matrix,
        hovertemplate="%{customdata}<extra></extra>",
        colorscale="RdBu",
        zmid=0.0,
        colorbar=dict(title="PnL ($)"),
    ))

    fig.update_layout(
        title=title,
        xaxis_title="Column Player Strategy (Player Y)",
        yaxis_title="Row Player Strategy (Player X)",
        template="plotly_white",
        margin=dict(l=50, r=40, t=50, b=50),
    )
    return fig


def render(run_dir: Optional[str] = None):
    """
    Renders the Game Theory & Nash Equilibria tab inside the Streamlit dashboard.

    Parameters
    ----------
    run_dir : Optional[str]
        Directory path of the current simulation run (e.g. 'data/runs/<run_id>').
    """
    st.subheader("🎯 Post-Hoc Game-Theoretic Equilibrium Analysis")
    st.markdown(
        "Analyzes strategic interactions between market agent archetypes via normal-form payoff matrices, "
        "Nash equilibrium computation (support enumeration), and fictitious play best-response dynamics."
    )

    if not run_dir:
        st.info("ℹ️ No run directory specified. Please select a simulation run to inspect game-theoretic analysis.")
        return

    eq_path = os.path.join(run_dir, "equilibrium_analysis.json")
    eq_analysis = load_equilibrium_analysis(eq_path)

    if not eq_analysis:
        # Check if agent_metrics.json exists to compute on-the-fly
        metrics_path = os.path.join(run_dir, "agent_metrics.json")
        if os.path.exists(metrics_path):
            try:
                with open(metrics_path, "r") as mf:
                    agent_metrics = json.load(mf)
                from src.gametheory.payoff_builder import PayoffBuilder
                from src.gametheory.nash_solver import NashSolver
                from src.gametheory.best_response import BestResponseDynamics

                archetype_pnls = {}
                for m in agent_metrics:
                    a_id = m.get("agent_id", "")
                    arch = "RL" if "rl" in a_id.lower() else ("MM" if "mm" in a_id.lower() else "ZI")
                    archetype_pnls.setdefault(arch, []).append(m.get("pnl", 0.0))

                if len(archetype_pnls) >= 2:
                    A, B, strats = PayoffBuilder.mean_pnl_ranking(archetype_pnls)
                    solver = NashSolver(A, B)
                    eq_candidates = solver.get_equilibrium_candidates(method="support_enumeration")
                    brd = BestResponseDynamics(A, B)
                    trace = brd.trace(max_iterations=20)
                    eq_analysis = {
                        "run_id": os.path.basename(os.path.normpath(run_dir)),
                        "method": "support_enumeration",
                        "payoff_matrix": A.tolist(),
                        "strategies": strats,
                        "equilibrium_candidates": eq_candidates,
                        "best_response_trace": trace,
                        "computed_at": "dynamic",
                    }
            except Exception:
                eq_analysis = None

    if not eq_analysis:
        render_empty_state(
            tab_name="Post-Hoc Game-Theoretic Equilibrium Analysis",
            description="Analyzes strategic interactions between market agent archetypes via normal-form payoff matrices, Nash equilibrium candidates, and best-response dynamics.",
            artifact_name="equilibrium_analysis.json",
            run_dir=run_dir,
            generator_fn=generate_tournament_artifact,
            button_label="⚡ Generate equilibrium_analysis.json now",
            spinner_text="Running pairwise tournament (5 episodes per cell)...",
        )
        return

    strategies = eq_analysis.get("strategies", [])
    payoff_matrix = eq_analysis.get("payoff_matrix", [])
    standard_errors = eq_analysis.get("standard_errors")
    candidates = eq_analysis.get("equilibrium_candidates", [])
    trace = eq_analysis.get("best_response_trace", [])
    solver_method = eq_analysis.get("method", "support_enumeration")

    # Overview Metrics Banner
    col_m1, col_m2, col_m3 = st.columns(3)
    with col_m1:
        st.metric("Interacting Strategies", len(strategies))
    with col_m2:
        st.metric("Equilibrium Candidates", len(candidates))
    with col_m3:
        st.metric("Solver Algorithm", str(solver_method).upper())

    st.divider()

    # Heatmap with Standard Errors
    st.markdown("### Head-to-Head Payoff Matrix ($A$) Heatmap")
    st.caption("Values reflect Player X (row) expected PnL with standard error ($mean \\pm SE$) against Player Y (column).")

    if payoff_matrix:
        fig_heat = plot_payoff_heatmap(
            payoff_matrix=payoff_matrix,
            standard_errors=standard_errors,
            strategies=strategies,
        )
        _render_chart(fig_heat)
    else:
        st.warning("No payoff matrix data found in equilibrium report.")

    st.divider()

    # Equilibrium Candidates & Best Response Trajectory
    col_eq1, col_eq2 = st.columns([1, 1])

    with col_eq1:
        st.markdown("### Nash Equilibrium Candidates")
        if candidates:
            for cand in candidates:
                eq_id = cand.get("equilibrium_id", 1)
                is_pure = cand.get("is_pure", False)
                p1_strat = cand.get("player_1_strategy", [])
                p2_strat = cand.get("player_2_strategy", [])
                p1_payoff = cand.get("player_1_payoff", 0.0)
                p2_payoff = cand.get("player_2_payoff", 0.0)

                # Format strategy distributions
                p1_desc = [
                    f"**{strategies[i]}**: {prob*100:.1f}%"
                    for i, prob in enumerate(p1_strat)
                    if prob > 1e-4 and i < len(strategies)
                ] or ["100.0%"]

                p2_desc = [
                    f"**{strategies[i]}**: {prob*100:.1f}%"
                    for i, prob in enumerate(p2_strat)
                    if prob > 1e-4 and i < len(strategies)
                ] or ["100.0%"]

                strat_type = "Pure Strategy" if is_pure else "Mixed Strategy"
                badge_icon = "🟢" if is_pure else "🟣"

                with st.container(border=True):
                    st.markdown(f"#### {badge_icon} Equilibrium #{eq_id} ({strat_type})")
                    st.markdown(f"- **Player 1 Strategy Profile:** {', '.join(p1_desc)}")
                    st.markdown(f"- **Player 1 Expected Payoff:** ${p1_payoff:.3f}")
                    st.markdown(f"- **Player 2 Strategy Profile:** {', '.join(p2_desc)}")
                    st.markdown(f"- **Player 2 Expected Payoff:** ${p2_payoff:.3f}")
        else:
            st.info("No pure or mixed strategy Nash equilibria identified for this payoff structure.")

    with col_eq2:
        st.markdown("### Best-Response Trajectory (Fictitious Play)")
        if trace:
            fig_trace = plot_best_response_trace(trace)
            _render_chart(fig_trace)
        else:
            st.info("No fictitious play best-response trajectory recorded.")

    st.divider()

    # Tabular Breakdown Table
    with st.expander("📊 View Payoff & Standard Error Numerical Tables"):
        col_tb1, col_tb2 = st.columns(2)
        with col_tb1:
            st.markdown("#### Mean Payoff Table ($A$)")
            if strategies and payoff_matrix:
                st.table({strategies[i]: [round(x, 2) for x in row] for i, row in enumerate(payoff_matrix)})
        with col_tb2:
            st.markdown("#### Standard Errors ($SE_A$)")
            if strategies and standard_errors:
                st.table({strategies[i]: [round(x, 2) for x in row] for i, row in enumerate(standard_errors)})
            else:
                st.info("Standard errors not available for this run.")
