"""
PnL, Sharpe ratio, and inventory calculations.

Provides pure functions that compute per-agent performance metrics from
reward/portfolio history.  Used by the logger and the dashboard.
"""

from __future__ import annotations

import numpy as np
from typing import Dict, List, Optional, Sequence, Tuple


def compute_pnl(
    final_cash: float,
    final_inventory: int,
    reference_price: float,
    initial_cash: float,
) -> float:
    """
    Compute total PnL (realized + unrealized) for a single agent.

    PnL = (final_cash + final_inventory × reference_price) − initial_cash
    """
    portfolio_value = final_cash + final_inventory * reference_price
    return portfolio_value - initial_cash


def compute_sharpe(
    returns: Sequence[float],
    risk_free_rate: float = 0.0,
    annualize: bool = False,
    periods_per_year: int = 252,
) -> float:
    """
    Compute the Sharpe ratio from a sequence of per-step returns.

    Sharpe = (mean(R) − Rf) / std(R)

    Parameters
    ----------
    returns : sequence of float
        Per-step returns (e.g. step PnL changes).
    risk_free_rate : float
        Per-step risk-free rate (default 0).
    annualize : bool
        If True, multiply by sqrt(periods_per_year).
    periods_per_year : int
        Number of trading steps per year (only used if annualize=True).

    Returns
    -------
    float
        Sharpe ratio.  Returns 0.0 if std(R) is effectively zero.
    """
    arr = np.asarray(returns, dtype=np.float64)
    if len(arr) < 2:
        return 0.0
    excess = arr - risk_free_rate
    std = float(np.std(excess, ddof=1))
    if std < 1e-12:
        return 0.0
    sharpe = float(np.mean(excess)) / std
    if annualize:
        sharpe *= np.sqrt(periods_per_year)
    return sharpe


def compute_drawdown(portfolio_values: Sequence[float]) -> Tuple[float, int]:
    """
    Compute maximum drawdown and the step at which it occurred.

    Returns
    -------
    (max_drawdown, peak_step) : (float, int)
        max_drawdown is a non-negative number representing the largest
        peak-to-trough decline.  peak_step is the step index of the
        preceding peak.
    """
    arr = np.asarray(portfolio_values, dtype=np.float64)
    if len(arr) < 2:
        return 0.0, 0
    running_max = np.maximum.accumulate(arr)
    drawdowns = running_max - arr
    max_dd_idx = int(np.argmax(drawdowns))
    max_dd = float(drawdowns[max_dd_idx])
    # Peak is the last index where the running max was achieved before the trough
    peak_idx = int(np.argmax(arr[:max_dd_idx + 1])) if max_dd_idx > 0 else 0
    return max_dd, peak_idx


def compute_inventory_stats(
    inventory_history: Sequence[int],
) -> Dict[str, float]:
    """
    Compute summary statistics over an agent's inventory trajectory.

    Returns a dict with keys:
        mean, std, min, max, abs_mean
    """
    arr = np.asarray(inventory_history, dtype=np.float64)
    if len(arr) == 0:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "abs_mean": 0.0}
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "abs_mean": float(np.mean(np.abs(arr))),
    }


def compute_agent_metrics(
    rewards: Sequence[float],
    portfolio_values: Sequence[float],
    inventory_history: Sequence[int],
    initial_cash: float,
    final_cash: float,
    final_inventory: int,
    reference_price: float,
) -> Dict[str, float]:
    """
    One-shot computation of all per-agent metrics.

    Returns a dict containing: pnl, sharpe, max_drawdown, and inventory stats.
    """
    pnl = compute_pnl(final_cash, final_inventory, reference_price, initial_cash)
    sharpe = compute_sharpe(rewards)
    max_dd, _ = compute_drawdown(portfolio_values)
    inv_stats = compute_inventory_stats(inventory_history)

    return {
        "pnl": pnl,
        "sharpe": sharpe,
        "max_drawdown": max_dd,
        "inventory_mean": inv_stats["mean"],
        "inventory_std": inv_stats["std"],
        "inventory_abs_mean": inv_stats["abs_mean"],
    }
