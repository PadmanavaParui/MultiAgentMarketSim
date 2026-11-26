"""
Order Book Replay & Bookmap-Style Depth Heatmap Dashboard Tab for MultiAgentMarketSim.

Provides high-fidelity market microstructure visualization:
  (a) Bookmap-style depth heatmap (x=step, y=price, colour=resting size) with trade dots overlaid
      and flash-crash marker on stress runs.
  (b) Interactive animated replay with a play/pause slider showing the live order book ladder,
      the trade tape, and each agent's real-time inventory and PnL.

Plotly only, zero external dependencies, optimized for smooth performance with downsampled data.
Exposes render(run_dir) with graceful missing-file handling.
"""

from __future__ import annotations

import os
import json
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st


def _render_chart(fig: go.Figure):
    """Renders Plotly chart with modern width='stretch' and backwards-compatible fallback."""
    try:
        st.plotly_chart(fig, width="stretch")
    except TypeError:
        st.plotly_chart(fig, use_container_width=True)


def _render_dataframe(df: Any):
    """Renders DataFrame with modern width='stretch' and backwards-compatible fallback."""
    try:
        st.dataframe(df, width="stretch")
    except TypeError:
        st.dataframe(df, use_container_width=True)


# ======================================================================
# 1. Stress Run & Flash-Crash Detection
# ======================================================================

def detect_flash_crash_step(
    run_dir: Optional[str] = None,
    snapshots: Optional[List[Dict[str, Any]]] = None,
    trades: Optional[List[Dict[str, Any]]] = None,
) -> Optional[int]:
    """
    Detects if the current run is a stress run with a flash crash shock,
    and returns the step at which the shock was initiated.
    """
    if run_dir:
        # 1. Check stress_report.json
        stress_report_path = os.path.join(run_dir, "stress_report.json")
        if os.path.exists(stress_report_path):
            try:
                with open(stress_report_path, "r") as f:
                    report = json.load(f)
                scenarios = report.get("scenarios", {})
                if "flash_crash" in scenarios:
                    # Default shock step in flash crash scenario config is 15
                    return int(report.get("config", {}).get("shock_step", 15))
            except Exception:
                pass

        # 2. Check run_summary.json
        summary_path = os.path.join(run_dir, "run_summary.json")
        if os.path.exists(summary_path):
            try:
                with open(summary_path, "r") as f:
                    summary = json.load(f)
                cfg = summary.get("config_snapshot", {})
                if "shock_step" in cfg:
                    return int(cfg["shock_step"])
                run_id = str(summary.get("run_id", "")).lower()
                if "flash_crash" in run_id or "shock" in run_id or "stress" in run_id:
                    return 15
            except Exception:
                pass

    # 3. Check for ShockAgent in agent states or trades
    if snapshots and len(snapshots) > 0:
        first_snap = snapshots[0]
        agent_keys = list(first_snap.get("inventories", {}).keys()) or list(first_snap.get("agents", {}).keys())
        if any("shock" in a.lower() for a in agent_keys):
            # Check trades to see when shock agent first traded
            if trades:
                for t in trades:
                    buy_id = str(t.get("buy_order_id", ""))
                    sell_id = str(t.get("sell_order_id", ""))
                    if "shock" in buy_id.lower() or "shock" in sell_id.lower():
                        return int(t.get("timestamp_step", t.get("step", 15)))
            return 15

    return None

    # 3. Check for ShockAgent in agent states or trades
    if snapshots and len(snapshots) > 0:
        first_snap = snapshots[0]
        agent_keys = list(first_snap.get("inventories", {}).keys()) or list(first_snap.get("agents", {}).keys())
        if any("shock" in a.lower() for a in agent_keys):
            # Check trades to see when shock agent first traded
            if trades:
                for t in trades:
                    buy_id = str(t.get("buy_order_id", ""))
                    sell_id = str(t.get("sell_order_id", ""))
                    if "shock" in buy_id.lower() or "shock" in sell_id.lower():
                        return int(t.get("timestamp_step", t.get("step", 15)))
            return 15

    return None


# ======================================================================
# 2. Bookmap-Style Depth Heatmap Builder
# ======================================================================

def plot_bookmap_depth_heatmap(
    snapshots: List[Dict[str, Any]],
    trades: Optional[List[Dict[str, Any]]] = None,
    flash_crash_step: Optional[int] = None,
    downsample_stride: int = 1,
    price_bin_size: Optional[float] = None,
) -> go.Figure:
    """
    Renders a Bookmap-style depth heatmap (x=step, y=price, colour=resting size)
    with trade dots overlaid, mid-price path, and an optional flash-crash marker.

    Parameters
    ----------
    snapshots : List[Dict[str, Any]]
        Records loaded from book_snapshots.json.
    trades : Optional[List[Dict[str, Any]]]
        Executed trades from trades.json or price_series.json.
    flash_crash_step : Optional[int]
        Step where flash crash market orders occur.
    downsample_stride : int
        Stride for temporal subsampling to maintain smooth frame rendering.
    price_bin_size : Optional[float]
        Price level bucket resolution (defaults to auto-calculated).

    Returns
    -------
    go.Figure
        Interactive Plotly Bookmap heatmap.
    """
    fig = go.Figure()
    if not snapshots:
        fig.update_layout(title="No Order Book Snapshot Telemetry Available")
        return fig

    # Subsample snapshots if stride > 1
    sampled_snaps = snapshots[::max(1, downsample_stride)]
    steps = [int(s.get("step", i + 1)) for i, s in enumerate(sampled_snaps)]
    if not steps:
        fig.update_layout(title="No Step Data in Book Snapshots")
        return fig

    # Collect all resting and traded prices to define vertical price grid
    price_pool: List[float] = []
    for s in sampled_snaps:
        bids = s.get("bids", []) or s.get("depth", {}).get("bids", [])
        asks = s.get("asks", []) or s.get("depth", {}).get("asks", [])
        for entry in bids + asks:
            if isinstance(entry, (list, tuple)) and len(entry) >= 2:
                price_pool.append(float(entry[0]))
        if s.get("mid") is not None:
            price_pool.append(float(s["mid"]))

    if trades:
        for t in trades:
            if "price" in t and t["price"] is not None:
                price_pool.append(float(t["price"]))

    if not price_pool:
        price_pool = [95.0, 100.0, 105.0]

    min_p = float(np.min(price_pool))
    max_p = float(np.max(price_pool))
    p_range = max(1.0, max_p - min_p)

    # Determine uniform price bin resolution
    if price_bin_size is None or price_bin_size <= 0:
        if p_range <= 5.0:
            bin_size = 0.10
        elif p_range <= 15.0:
            bin_size = 0.25
        elif p_range <= 40.0:
            bin_size = 0.50
        else:
            bin_size = 1.0
    else:
        bin_size = float(price_bin_size)

    y_grid = np.arange(np.floor(min_p) - bin_size, np.ceil(max_p) + bin_size * 2, bin_size)
    y_prices = [round(float(p), 4) for p in y_grid]

    # Construct resting depth 2D matrix Z (shape: len(y_prices) x len(steps))
    Z = np.zeros((len(y_prices), len(steps)), dtype=np.float64)

    for j, s in enumerate(sampled_snaps):
        bids = s.get("bids", []) or s.get("depth", {}).get("bids", [])
        asks = s.get("asks", []) or s.get("depth", {}).get("asks", [])
        for entry in bids + asks:
            if isinstance(entry, (list, tuple)) and len(entry) >= 2:
                p_val = float(entry[0])
                vol = float(entry[1])
                # Nearest bin index
                idx = int(np.argmin(np.abs(y_grid - p_val)))
                Z[idx, j] += vol

    # 1. Depth Heatmap (Bookmap style with dark background and vibrant heat intensity)
    fig.add_trace(go.Heatmap(
        x=steps,
        y=y_prices,
        z=Z,
        colorscale="Plasma",
        colorbar=dict(
            title=dict(text="Resting Depth<br>(Shares)", side="top"),
            thickness=15,
            len=0.8,
        ),
        hovertemplate="Step: %{x}<br>Price Level: $%{y:.2f}<br>Resting Size: %{z:,.0f} shares<extra>Bookmap Depth</extra>",
        showscale=True,
    ))

    # 2. Mid-Price Path
    valid_mid_steps = []
    valid_mids = []
    for s in sampled_snaps:
        if s.get("mid") is not None and not np.isnan(s["mid"]):
            valid_mid_steps.append(int(s["step"]))
            valid_mids.append(float(s["mid"]))

    if valid_mids:
        fig.add_trace(go.Scatter(
            x=valid_mid_steps,
            y=valid_mids,
            mode="lines",
            name="Mid Price",
            line=dict(color="#00ffff", width=1.75, dash="dot"),
            hovertemplate="Step: %{x}<br>Mid: $%{y:.2f}<extra>Mid Price</extra>",
        ))

    # 3. Executed Trade Dots Overlaid
    if trades:
        min_step = steps[0]
        max_step = steps[-1]
        t_steps = []
        t_prices = []
        t_sizes = []
        t_texts = []

        for i, t in enumerate(trades):
            st_val = int(t.get("timestamp_step", t.get("step", 0)))
            if min_step <= st_val <= max_step:
                p = float(t.get("price", 100.0))
                q = int(t.get("quantity", 1))
                t_steps.append(st_val)
                t_prices.append(p)
                # Scale dot radius by trade volume
                marker_size = max(5, min(22, int(np.sqrt(q) * 3.2)))
                t_sizes.append(marker_size)
                t_texts.append(
                    f"Trade #{i+1}<br>Step: {st_val}<br>Price: ${p:.2f}<br>Size: {q} shares"
                )

        if t_steps:
            fig.add_trace(go.Scatter(
                x=t_steps,
                y=t_prices,
                mode="markers",
                name="Executed Trades",
                marker=dict(
                    size=t_sizes,
                    color="#ffd700",  # Bright gold trade bubbles
                    symbol="circle",
                    line=dict(width=1, color="#ffffff"),
                    opacity=0.9,
                ),
                customdata=t_texts,
                hovertemplate="%{customdata}<extra>Trade Tape</extra>",
            ))

    # 4. Flash-Crash Marker (on stress runs)
    if flash_crash_step is not None:
        fig.add_vline(
            x=flash_crash_step,
            line_dash="dash",
            line_color="#ff1744",
            line_width=2.5,
            annotation_text="⚡ Flash Crash Shock",
            annotation_position="top left",
            annotation_font=dict(color="#ff1744", size=12, family="monospace"),
        )

    fig.update_layout(
        title="Bookmap Depth Heatmap (Resting Liquidity & Executed Flow)",
        xaxis_title="Simulation Step",
        yaxis_title="Price ($)",
        template="plotly_dark",
        plot_bgcolor="#0c0e14",
        paper_bgcolor="#0c0e14",
        hovermode="closest",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor="rgba(0,0,0,0.5)",
        ),
        margin=dict(l=50, r=40, t=65, b=45),
    )
    return fig


# ======================================================================
# 3. Live Order Book Ladder Builder
# ======================================================================

def plot_live_order_book(snapshot: Dict[str, Any]) -> go.Figure:
    """
    Renders the live order book depth ladder at a single replay step.
    Shows resting bids (green) and resting asks (red) with spread annotation.
    """
    fig = go.Figure()

    step = snapshot.get("step", 0)
    best_bid = snapshot.get("best_bid")
    best_ask = snapshot.get("best_ask")
    mid = snapshot.get("mid")
    spread = snapshot.get("spread")

    bids = snapshot.get("bids", []) or snapshot.get("depth", {}).get("bids", [])
    asks = snapshot.get("asks", []) or snapshot.get("depth", {}).get("asks", [])

    bid_prices = [float(b[0]) for b in bids if isinstance(b, (list, tuple))]
    bid_sizes = [int(b[1]) for b in bids if isinstance(b, (list, tuple))]

    ask_prices = [float(a[0]) for a in asks if isinstance(a, (list, tuple))]
    ask_sizes = [int(a[1]) for a in asks if isinstance(a, (list, tuple))]

    # Bids (green bars)
    if bid_prices:
        fig.add_trace(go.Bar(
            y=[f"${p:.2f}" for p in bid_prices],
            x=bid_sizes,
            orientation="h",
            name="Bids (Buy Queue)",
            marker_color="#00e676",
            hovertemplate="Bid: %{y}<br>Resting Size: %{x} shares<extra>BID</extra>",
        ))

    # Asks (red bars)
    if ask_prices:
        fig.add_trace(go.Bar(
            y=[f"${p:.2f}" for p in ask_prices],
            x=ask_sizes,
            orientation="h",
            name="Asks (Sell Queue)",
            marker_color="#ff1744",
            hovertemplate="Ask: %{y}<br>Resting Size: %{x} shares<extra>ASK</extra>",
        ))

    # Spread and Mid text annotation
    if best_bid is not None and best_ask is not None:
        spread_text = f"Spread: ${spread:.2f} | Mid: ${mid:.2f}"
    else:
        spread_text = "Empty Book Side"

    fig.update_layout(
        title=f"Live Order Book Ladder (Step {step}) — {spread_text}",
        xaxis_title="Resting Size (Shares)",
        yaxis_title="Price Level ($)",
        template="plotly_dark",
        plot_bgcolor="#121722",
        paper_bgcolor="#121722",
        barmode="group",
        legend=dict(x=0.02, y=0.98),
        margin=dict(l=60, r=40, t=50, b=40),
    )
    return fig


# ======================================================================
# 4. Animated Replay Tab Render Entrypoint
# ======================================================================

def render(run_dir: Optional[str] = None):
    """
    Renders the Order Book Replay tab inside the Streamlit dashboard.

    Parameters
    ----------
    run_dir : Optional[str]
        Directory path of the current simulation run (e.g. 'data/runs/<run_id>').
    """
    st.subheader("📼 Order Book Replay & Bookmap Heatmap")
    st.markdown(
        "Interactive microstructure replay visualizing **Bookmap-style resting liquidity heatmaps**, "
        "executed order flow, live order book dynamics, and agent-level telemetry across time steps."
    )

    if not run_dir:
        st.warning("No run directory specified. Please select a simulation run to inspect replay telemetry.")
        return

    snapshots_path = os.path.join(run_dir, "book_snapshots.json")
    if not os.path.exists(snapshots_path):
        st.warning(
            "No order book snapshot telemetry found (`book_snapshots.json`). "
            "Please select a run logged with book telemetry or launch a new simulation."
        )
        return

    # Load book snapshots
    try:
        with open(snapshots_path, "r") as f:
            snapshots = json.load(f)
    except Exception as e:
        st.error(f"Failed to load `{snapshots_path}`: {e}")
        return

    if not snapshots:
        st.info("The `book_snapshots.json` file is empty.")
        return

    # Load trade data if available
    trades: List[Dict[str, Any]] = []
    trades_path = os.path.join(run_dir, "trades.json")
    if os.path.exists(trades_path):
        try:
            with open(trades_path, "r") as f:
                trades = json.load(f)
        except Exception:
            pass

    # Detect stress run / flash crash marker
    flash_crash_step = detect_flash_crash_step(run_dir, snapshots, trades)

    # ------------------------------------------------------------------
    # PART (A): Bookmap-Style Depth Heatmap
    # ------------------------------------------------------------------
    st.markdown("### (a) Bookmap Depth Heatmap & Trade Execution Flow")
    st.caption("Visualizes order book depth intensity across steps with executed trades overlaid as gold bubbles.")

    col_ctrl1, col_ctrl2, col_ctrl3 = st.columns([1, 1, 1])
    with col_ctrl1:
        downsample_stride = st.selectbox(
            "Heatmap Temporal Stride",
            options=[1, 2, 5, 10],
            index=0 if len(snapshots) < 100 else (1 if len(snapshots) < 300 else 2),
            help="Subsample step frequency to ensure high framerates on large simulation runs.",
            key="replay_heatmap_stride",
        )
    with col_ctrl2:
        price_bin_options = ["Auto", "$0.10", "$0.25", "$0.50", "$1.00"]
        selected_bin = st.selectbox("Price Bin Resolution", options=price_bin_options, index=0, key="replay_bin_size")
        bin_val = None if selected_bin == "Auto" else float(selected_bin.replace("$", ""))

    with col_ctrl3:
        if flash_crash_step is not None:
            st.warning(f"⚡ Stress Run Detected: Flash Crash Shock Marker Active (Step {flash_crash_step})")
        else:
            st.info("Normal Trading Run: No external price shocks detected.")

    fig_bookmap = plot_bookmap_depth_heatmap(
        snapshots=snapshots,
        trades=trades,
        flash_crash_step=flash_crash_step,
        downsample_stride=downsample_stride,
        price_bin_size=bin_val,
    )
    _render_chart(fig_bookmap)

    st.divider()

    # ------------------------------------------------------------------
    # PART (B): Interactive Animated Replay with Play/Pause Slider
    # ------------------------------------------------------------------
    st.markdown("### (b) Live Book Replay Player, Trade Tape & Agent Telemetry")

    total_snapshots = len(snapshots)

    # Session state initialization for replay slider
    if "replay_step_idx" not in st.session_state:
        st.session_state["replay_step_idx"] = 0
    if "replay_playing" not in st.session_state:
        st.session_state["replay_playing"] = False

    # Clamp index within valid range
    st.session_state["replay_step_idx"] = max(0, min(total_snapshots - 1, st.session_state["replay_step_idx"]))

    # Play/Pause and Step Controls
    col_btn1, col_btn2, col_btn3, col_btn4, col_btn5 = st.columns([1, 1, 1, 1, 2])
    with col_btn1:
        if st.button("⏪ -5 Steps", key="btn_prev_5"):
            st.session_state["replay_step_idx"] = max(0, st.session_state["replay_step_idx"] - 5)
            st.session_state["replay_playing"] = False
    with col_btn2:
        play_label = "⏸️ Pause" if st.session_state["replay_playing"] else "▶️ Play"
        if st.button(play_label, key="btn_play_pause", type="primary" if not st.session_state["replay_playing"] else "secondary"):
            st.session_state["replay_playing"] = not st.session_state["replay_playing"]
            if st.session_state["replay_playing"] and st.session_state["replay_step_idx"] >= total_snapshots - 1:
                st.session_state["replay_step_idx"] = 0
    with col_btn3:
        if st.button("⏩ +5 Steps", key="btn_next_5"):
            st.session_state["replay_step_idx"] = min(total_snapshots - 1, st.session_state["replay_step_idx"] + 5)
            st.session_state["replay_playing"] = False
    with col_btn4:
        if st.button("🔄 Reset", key="btn_reset"):
            st.session_state["replay_step_idx"] = 0
            st.session_state["replay_playing"] = False

    # Auto-play tick advancement
    if st.session_state["replay_playing"]:
        if st.session_state["replay_step_idx"] < total_snapshots - 1:
            st.session_state["replay_step_idx"] += 1
        else:
            st.session_state["replay_playing"] = False

    # Play/Pause Step Slider
    step_labels = [f"Step {s.get('step', i+1)}" for i, s in enumerate(snapshots)]
    selected_slider_idx = st.slider(
        "Replay Timeline Scrub Slider",
        min_value=0,
        max_value=total_snapshots - 1,
        value=st.session_state["replay_step_idx"],
        format="%d",
        key="replay_slider",
        help="Drag slider to scrub through simulation steps.",
    )

    # Sync slider selection to session state
    if selected_slider_idx != st.session_state["replay_step_idx"]:
        st.session_state["replay_step_idx"] = selected_slider_idx
        st.session_state["replay_playing"] = False

    current_idx = st.session_state["replay_step_idx"]
    current_snap = snapshots[current_idx]
    current_step = current_snap.get("step", current_idx + 1)

    # Flash Crash Event Banner in Replay
    if flash_crash_step is not None:
        if current_step >= flash_crash_step:
            st.error(f"⚡ **FLASH CRASH ACTIVE:** Stress shock orders swept the bid ladder at Step {flash_crash_step}!")
        else:
            st.info(f"🛡️ **Pre-Shock State:** Order book operating normally (Flash Crash onset scheduled at Step {flash_crash_step}).")

    # Layout: Live Book Ladder (left) and Trade Tape (right)
    col_book, col_tape = st.columns([3, 2])

    with col_book:
        fig_ladder = plot_live_order_book(current_snap)
        _render_chart(fig_ladder)

    with col_tape:
        st.markdown(f"#### 📜 The Tape (Executed Trades up to Step {current_step})")
        # Filter trades executed up to current step
        tape_records = [
            t for t in trades
            if int(t.get("timestamp_step", t.get("step", 0))) <= current_step
        ]
        if tape_records:
            # Display latest 12 trades
            recent_trades = tape_records[-12:][::-1]
            tape_table = []
            for t in recent_trades:
                p = float(t.get("price", 100.0))
                q = int(t.get("quantity", 1))
                val = p * q
                t_step = t.get("timestamp_step", t.get("step", 0))
                tape_table.append({
                    "Step": t_step,
                    "Price ($)": f"${p:.2f}",
                    "Size": f"{q} shs",
                    "Total ($)": f"${val:.2f}",
                    "Trade ID": t.get("trade_id", "trd"),
                })
            _render_dataframe(tape_table)
        else:
            st.info("No trades executed yet at or prior to this step.")

    st.divider()

    # ------------------------------------------------------------------
    # Agent Inventory and PnL Telemetry
    # ------------------------------------------------------------------
    st.markdown(f"### 👥 Agent State Telemetry at Step {current_step}")

    inventories = current_snap.get("inventories", {})
    pnls = current_snap.get("pnls", {})
    agents_dict = current_snap.get("agents", {})

    agent_ids = sorted(list(set(list(inventories.keys()) + list(pnls.keys()) + list(agents_dict.keys()))))

    if agent_ids:
        # Dynamic Metric Tiles for Agents
        cols = st.columns(min(5, max(1, len(agent_ids))))
        for idx, a_id in enumerate(agent_ids):
            col_target = cols[idx % len(cols)]
            inv = inventories.get(a_id, agents_dict.get(a_id, {}).get("inventory", 0))
            pnl = pnls.get(a_id, agents_dict.get(a_id, {}).get("pnl", 0.0))

            a_role = "RL Trader" if "rl" in a_id.lower() else (
                "Shock Agent" if "shock" in a_id.lower() else (
                    "Informed Trader" if "informed" in a_id.lower() else (
                        "Avellaneda-Stoikov MM" if "as" in a_id.lower() else (
                            "Market Maker" if "mm" in a_id.lower() else "Zero Intelligence"
                        )
                    )
                )
            )

            with col_target:
                pnl_color = "normal" if pnl >= 0 else "inverse"
                st.metric(
                    label=f"{a_id} ({a_role})",
                    value=f"${pnl:+.2f}",
                    delta=f"{inv:+d} shares",
                    delta_color=pnl_color,
                )

        # Tabular Summary of Agent Positions
        agent_table = []
        for a_id in agent_ids:
            inv = inventories.get(a_id, agents_dict.get(a_id, {}).get("inventory", 0))
            pnl = pnls.get(a_id, agents_dict.get(a_id, {}).get("pnl", 0.0))
            a_role = "RL Trader" if "rl" in a_id.lower() else (
                "Shock Agent" if "shock" in a_id.lower() else (
                    "Informed Trader" if "informed" in a_id.lower() else (
                        "Avellaneda-Stoikov MM" if "as" in a_id.lower() else (
                            "Market Maker" if "mm" in a_id.lower() else "Zero Intelligence"
                        )
                    )
                )
            )
            agent_table.append({
                "Agent ID": a_id,
                "Archetype": a_role,
                "Inventory (Shares)": inv,
                "PnL ($)": round(float(pnl), 2),
                "Status": "Flat" if inv == 0 else ("Long" if inv > 0 else "Short"),
            })

        with st.expander(f"📋 View Full Agent Portfolio Table (Step {current_step})"):
            _render_dataframe(agent_table)
    else:
        st.info("No agent telemetry recorded in this snapshot.")
