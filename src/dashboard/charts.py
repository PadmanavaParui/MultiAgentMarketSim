"""
Plotly Chart Builders for MultiAgentMarketSim Dashboard (TR-007, UX Requirements).

Provides accessible, interactive visualizations:
  1. Order Book Depth Chart (bid/ask ladder + spread sanity marker)
  2. Price & Volume Time Series (dual-axis / subplots)
  3. Agent Performance (per-agent PnL, Sharpe, inventory accumulation)
  4. Game-Theoretic Analytics (best-response trace trajectory)
  5. Convergence Diagnostics (reward curves and TD errors)
"""

from __future__ import annotations

from typing import List, Dict, Any, Tuple, Optional
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def downsample_series(data: Any, max_points: int = 5000) -> Any:
    """Downsamples a sequence or array to at most max_points by uniform striding."""
    if data is None or len(data) <= max_points:
        return data
    stride = int(np.ceil(len(data) / max_points))
    return data[::stride]


def plot_order_book_depth(
    bids: List[Tuple[float, int]],
    asks: List[Tuple[float, int]],
    best_bid: Optional[float] = None,
    best_ask: Optional[float] = None,
) -> go.Figure:
    """
    Renders an interactive Order Book Depth chart (bid/ask ladder).
    Uses high-contrast colors and explicit labels (accessibility compliance).
    Includes a visible spread sanity check indicator.
    """
    fig = go.Figure()

    # Bids (Buy side)
    if bids:
        # Sort ascending by price for cumulative volume display
        bids_sorted = sorted(bids, key=lambda x: x[0])
        bid_prices = [p for p, _ in bids_sorted]
        bid_vols = [v for _, v in bids_sorted]
        cum_bid_vols = np.cumsum(bid_vols[::-1])[::-1]  # Higher prices have cumulative volume

        fig.add_trace(go.Scatter(
            x=bid_prices,
            y=cum_bid_vols,
            mode="lines+markers",
            name="Bids (Buy Queue)",
            line=dict(color="#2ca02c", width=3, shape="hv"),
            fill="tozeroy",
            fillcolor="rgba(44, 160, 44, 0.2)",
            hovertemplate="Bid Price: $%{x:.2f}<br>Cum Vol: %{y}<extra>BID</extra>",
        ))

    # Asks (Sell side)
    if asks:
        asks_sorted = sorted(asks, key=lambda x: x[0])
        ask_prices = [p for p, _ in asks_sorted]
        ask_vols = [v for _, v in asks_sorted]
        cum_ask_vols = np.cumsum(ask_vols)

        fig.add_trace(go.Scatter(
            x=ask_prices,
            y=cum_ask_vols,
            mode="lines+markers",
            name="Asks (Sell Queue)",
            line=dict(color="#d62728", width=3, shape="hv"),
            fill="tozeroy",
            fillcolor="rgba(214, 39, 40, 0.2)",
            hovertemplate="Ask Price: $%{x:.2f}<br>Cum Vol: %{y}<extra>ASK</extra>",
        ))

    # Spread annotation & sanity marker
    if best_bid is not None and best_ask is not None:
        spread = best_ask - best_bid
        status_text = "SANITY CHECK: SPREAD INTACT (best_bid < best_ask)" if spread > 0 else "VIOLATION DETECTED"
        color = "green" if spread > 0 else "red"
        fig.add_annotation(
            text=f"Spread: ${spread:.2f} | {status_text}",
            xref="paper", yref="paper",
            x=0.5, y=1.05,
            showarrow=False,
            font=dict(size=12, color=color, family="monospace"),
            bgcolor="rgba(255, 255, 255, 0.8)",
            bordercolor=color,
            borderwidth=1,
        )

    fig.update_layout(
        title="Order Book Depth (Bid/Ask Ladder)",
        xaxis_title="Price ($)",
        yaxis_title="Cumulative Volume (Shares)",
        template="plotly_white",
        hovermode="x unified",
        legend=dict(x=0.02, y=0.98),
        margin=dict(l=40, r=40, t=60, b=40),
    )
    return fig


def plot_price_and_volume(price_series: List[Dict[str, Any]]) -> go.Figure:
    """
    Renders dual-axis subplots for price time series and volume bars.
    Handles multi-episode and single-episode runs gracefully without backward loops.
    """
    if not price_series:
        fig = go.Figure()
        fig.update_layout(title="No Trade Data Available")
        return fig

    if len(price_series) > 5000:
        price_series = downsample_series(price_series, 5000)

    # Check if steps reset backwards (indicates multi-episode runs)
    raw_steps = [p.get("step", i + 1) for i, p in enumerate(price_series)]
    has_resets = any(raw_steps[i] < raw_steps[i - 1] for i in range(1, len(raw_steps)))

    if "global_step" in price_series[0]:
        x_vals = [p["global_step"] for p in price_series]
        x_label = "Global Simulation Step"
    elif has_resets:
        # Multi-episode run: use sequential trade progression to ensure strictly forward timeline
        x_vals = list(range(1, len(price_series) + 1))
        x_label = "Trade Execution Sequence (#)"
    else:
        x_vals = raw_steps
        x_label = "Simulation Step"

    prices = [p.get("price", 0.0) for p in price_series]
    quantities = [p.get("quantity", 0) for p in price_series]

    hover_texts = [
        f"Trade #{i + 1}<br>"
        + (f"Episode: {p.get('episode')}<br>" if 'episode' in p else "")
        + f"Step: {p.get('step', i + 1)}<br>"
        + f"Price: ${p.get('price', 0.0):.2f}<br>"
        + f"Volume: {p.get('quantity', 0)} shares"
        for i, p in enumerate(price_series)
    ]

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=("Simulated Execution Price", "Trade Volume"),
        row_heights=[0.7, 0.3],
    )

    # Price Line
    fig.add_trace(
        go.Scatter(
            x=x_vals,
            y=prices,
            mode="lines+markers",
            name="Trade Price",
            line=dict(color="#1f77b4", width=2),
            marker=dict(size=6),
            customdata=hover_texts,
            hovertemplate="%{customdata}<extra></extra>",
        ),
        row=1, col=1,
    )

    # Volume Bars
    fig.add_trace(
        go.Bar(
            x=x_vals,
            y=quantities,
            name="Volume",
            marker_color="#ff7f0e",
            customdata=hover_texts,
            hovertemplate="%{customdata}<extra></extra>",
        ),
        row=2, col=1,
    )

    fig.update_layout(
        template="plotly_white",
        hovermode="x unified",
        showlegend=False,
        margin=dict(l=40, r=40, t=50, b=40),
    )
    fig.update_xaxes(title_text=x_label, row=2, col=1)
    fig.update_yaxes(title_text="Price ($)", row=1, col=1)
    fig.update_yaxes(title_text="Shares", row=2, col=1)
    return fig


def plot_agent_pnl(agent_metrics: List[Dict[str, Any]]) -> go.Figure:
    """
    Plots per-agent PnL across episodes to compare RL agents vs. baselines.
    """
    fig = go.Figure()
    if not agent_metrics:
        fig.update_layout(title="No Agent Metric Data Available")
        return fig

    # Group by agent_id
    agents_data: Dict[str, List[Dict[str, Any]]] = {}
    for m in agent_metrics:
        a_id = m.get("agent_id", "unknown")
        agents_data.setdefault(a_id, []).append(m)

    palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]

    for idx, (a_id, records) in enumerate(agents_data.items()):
        if len(records) > 5000:
            records = downsample_series(records, 5000)
        episodes = [r.get("episode", i + 1) for i, r in enumerate(records)]
        pnls = [r.get("pnl", 0.0) for r in records]
        color = palette[idx % len(palette)]

        is_rl = "rl" in a_id.lower()
        line_style = dict(color=color, width=3 if is_rl else 2, dash="solid" if is_rl else "dot")

        fig.add_trace(go.Scatter(
            x=episodes,
            y=pnls,
            mode="lines+markers",
            name=f"{a_id} ({'RL' if is_rl else 'Baseline'})",
            line=line_style,
            marker=dict(size=6),
            hovertemplate=f"Agent: {a_id}<br>Episode: %{{x}}<br>Total PnL: $%{{y:.2f}}<extra></extra>",
        ))

    # Add reference horizontal zero line
    fig.add_hline(y=0.0, line_dash="dash", line_color="gray", annotation_text="Breakeven ($0)")

    fig.update_layout(
        title="Agent Cumulative PnL Across Episodes (RL vs. Baselines)",
        xaxis_title="Episode",
        yaxis_title="PnL ($)",
        template="plotly_white",
        hovermode="x unified",
        margin=dict(l=40, r=40, t=50, b=40),
    )
    return fig


def plot_inventory_accumulation(agent_metrics: List[Dict[str, Any]]) -> go.Figure:
    """
    Plots agent inventory accumulation across episodes.
    """
    fig = go.Figure()
    if not agent_metrics:
        fig.update_layout(title="No Inventory Data Available")
        return fig

    agents_data: Dict[str, List[Dict[str, Any]]] = {}
    for m in agent_metrics:
        a_id = m.get("agent_id", "unknown")
        agents_data.setdefault(a_id, []).append(m)

    for a_id, records in agents_data.items():
        if len(records) > 5000:
            records = downsample_series(records, 5000)
        episodes = [r.get("episode", i + 1) for i, r in enumerate(records)]
        invs = [r.get("inventory", 0) for r in records]

        fig.add_trace(go.Scatter(
            x=episodes,
            y=invs,
            mode="lines+markers",
            name=a_id,
            hovertemplate=f"Agent: {a_id}<br>Episode: %{{x}}<br>Inventory: %{{y}} shares<extra></extra>",
        ))

    fig.add_hline(y=0.0, line_dash="dash", line_color="gray")

    fig.update_layout(
        title="Inventory Accumulation Across Episodes",
        xaxis_title="Episode",
        yaxis_title="Inventory (Shares)",
        template="plotly_white",
        margin=dict(l=40, r=40, t=50, b=40),
    )
    return fig


def plot_best_response_trace(trace: List[Dict[str, Any]]) -> go.Figure:
    """
    Plots the strategy probability trajectory across fictitious play / best-response iterations.
    """
    fig = go.Figure()
    if not trace:
        fig.update_layout(title="No Best-Response Trace Available")
        return fig

    if len(trace) > 5000:
        trace = downsample_series(trace, 5000)

    steps = [t["step"] for t in trace]
    # Check dimensions
    p_len = len(trace[0]["p"]) if trace and "p" in trace[0] else 0
    q_len = len(trace[0]["q"]) if trace and "q" in trace[0] else 0

    for i in range(p_len):
        prob_series = [t["p"][i] for t in trace]
        fig.add_trace(go.Scatter(
            x=steps,
            y=prob_series,
            mode="lines",
            name=f"Player 1 Strat {i+1} prob",
            line=dict(width=2),
        ))

    for j in range(q_len):
        prob_series = [t["q"][j] for t in trace]
        fig.add_trace(go.Scatter(
            x=steps,
            y=prob_series,
            mode="lines",
            name=f"Player 2 Strat {j+1} prob",
            line=dict(width=2, dash="dash"),
        ))

    fig.update_layout(
        title="Best-Response Dynamics (Strategy Evolution Toward Equilibrium)",
        xaxis_title="Iteration",
        yaxis_title="Strategy Probability",
        template="plotly_white",
        margin=dict(l=40, r=40, t=50, b=40),
    )
    return fig


def plot_convergence_diagnostics(rewards: List[float], td_errors: Optional[List[float]] = None) -> go.Figure:
    """
    Plots RL reward convergence curve and TD error progression.
    """
    fig = make_subplots(
        rows=2 if td_errors else 1, cols=1,
        shared_xaxes=True,
        subplot_titles=("Reward Trajectory", "TD Error (Bellman Gap)") if td_errors else ("Reward Trajectory",),
        vertical_spacing=0.1,
    )

    if rewards and len(rewards) > 5000:
        rewards = downsample_series(rewards, 5000)
    if td_errors and len(td_errors) > 5000:
        td_errors = downsample_series(td_errors, 5000)

    steps = list(range(1, len(rewards) + 1))
    fig.add_trace(
        go.Scatter(x=steps, y=rewards, mode="lines", name="Reward", line=dict(color="#1f77b4")),
        row=1, col=1,
    )

    if td_errors:
        fig.add_trace(
            go.Scatter(x=steps[:len(td_errors)], y=td_errors, mode="lines", name="TD Error", line=dict(color="#d62728")),
            row=2, col=1,
        )

    fig.update_layout(
        title="RL Convergence Diagnostics (Reward / Loss Stability)",
        template="plotly_white",
        margin=dict(l=40, r=40, t=50, b=40),
    )
    return fig


def plot_training_curve(
    rewards: List[float],
    epsilons: Optional[List[float]] = None,
    td_errors: Optional[List[float]] = None,
    window: int = 5,
    title: str = "RL Agent Training Curve (Reward & Exploration Dynamics)",
) -> go.Figure:
    """
    Renders an RL training curve displaying raw episode/step rewards, a smoothed
    moving-average trendline, exploration rate (epsilon) decay, and TD errors.

    Parameters
    ----------
    rewards : List[float]
        Sequence of rewards per step or episode.
    epsilons : List[float] | None
        Exploration rate (epsilon) schedule history.
    td_errors : List[float] | None
        Temporal difference (TD) Bellman error history.
    window : int
        Window size for computing smoothed rolling average (default: 5).
    title : str
        Chart title.
    """
    if not rewards:
        fig = go.Figure()
        fig.update_layout(
            title="No Training Data Available",
            template="plotly_white",
            margin=dict(l=40, r=40, t=50, b=40),
        )
        return fig

    if len(rewards) > 5000:
        stride = int(np.ceil(len(rewards) / 5000))
        rewards = rewards[::stride]
        if epsilons:
            epsilons = epsilons[::stride]
        if td_errors:
            td_errors = td_errors[::stride]

    has_secondary = bool(epsilons or td_errors)
    steps = list(range(1, len(rewards) + 1))

    # Compute rolling mean for rewards
    w = max(1, min(window, len(rewards)))
    rolling_rewards = []
    for i in range(len(rewards)):
        start_idx = max(0, i - w + 1)
        rolling_rewards.append(float(np.mean(rewards[start_idx : i + 1])))

    if has_secondary:
        fig = make_subplots(
            rows=2,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.12,
            subplot_titles=(
                "Reward Trajectory & Smoothed Trend",
                "Exploration Rate (ε) & Bellman Error Progression",
            ),
            row_heights=[0.62, 0.38],
        )
    else:
        fig = make_subplots(
            rows=1,
            cols=1,
            subplot_titles=("Reward Trajectory & Smoothed Trend",),
        )

    # 1. Raw rewards
    fig.add_trace(
        go.Scatter(
            x=steps,
            y=rewards,
            mode="lines+markers",
            name="Raw Reward",
            line=dict(color="rgba(31, 119, 180, 0.35)", width=1.5),
            marker=dict(size=4, color="rgba(31, 119, 180, 0.5)"),
            hovertemplate="Step: %{x}<br>Reward: %{y:.2f}<extra>RAW</extra>",
        ),
        row=1,
        col=1,
    )

    # 2. Smoothed rolling mean
    fig.add_trace(
        go.Scatter(
            x=steps,
            y=rolling_rewards,
            mode="lines",
            name=f"Rolling Mean (w={w})",
            line=dict(color="#1f77b4", width=3),
            hovertemplate="Step: %{x}<br>Mean Reward: %{y:.2f}<extra>TREND</extra>",
        ),
        row=1,
        col=1,
    )

    # Baseline 0 reward reference line
    fig.add_hline(y=0.0, line_dash="dash", line_color="gray", row=1, col=1)

    # 3. Secondary subplots: Epsilon & TD Error
    if has_secondary:
        if epsilons:
            eps_x = steps[: len(epsilons)]
            fig.add_trace(
                go.Scatter(
                    x=eps_x,
                    y=epsilons,
                    mode="lines",
                    name="Exploration (ε)",
                    line=dict(color="#ff7f0e", width=2),
                    hovertemplate="Step: %{x}<br>Epsilon: %{y:.4f}<extra>EPSILON</extra>",
                ),
                row=2,
                col=1,
            )

        if td_errors:
            td_x = steps[: len(td_errors)]
            fig.add_trace(
                go.Scatter(
                    x=td_x,
                    y=td_errors,
                    mode="lines",
                    name="TD Error |δ|",
                    line=dict(color="#d62728", width=1.5, dash="dot"),
                    hovertemplate="Step: %{x}<br>TD Error: %{y:.4f}<extra>BELLMAN</extra>",
                ),
                row=2,
                col=1,
            )

    fig.update_layout(
        title=title,
        template="plotly_white",
        hovermode="x unified",
        margin=dict(l=40, r=40, t=60, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1.0),
    )
    fig.update_xaxes(title_text="Step / Episode", row=2 if has_secondary else 1, col=1)
    fig.update_yaxes(title_text="Reward ($)", row=1, col=1)
    if has_secondary:
        fig.update_yaxes(title_text="Rate / Error", row=2, col=1)

    return fig

