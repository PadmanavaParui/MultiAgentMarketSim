"""
Repository-style metrics and event logger for MultiAgentMarketSim.

Acts as the single write path for Order, Trade, AgentMetric, and SimulationRun records
(LLD, Data Design & API Specification in OJT documentation).
Stores run logs in JSON/CSV formats in data/runs/<run_id>/ and provides
standard query methods for the dashboard and analysis tools:
  - get_run_summary(run_id)
  - get_price_series(run_id)
  - get_agent_metrics(run_id, agent_id=None)
  - get_equilibrium_analysis(run_id)
  - get_book_snapshots(run_id)
  - get_timeseries(run_id)
"""

from __future__ import annotations

import os
import json
import csv
from datetime import datetime, timezone
from dataclasses import dataclass, asdict, field
from typing import Any, Dict, List, Optional, Union
import numpy as np


@dataclass
class SimulationRunRecord:
    run_id: str
    seed: Optional[int]
    config_snapshot: Dict[str, Any]
    started_at: str
    status: str = "RUNNING"  # RUNNING / COMPLETE / SIMULATION_INVALID
    completed_at: Optional[str] = None
    error_message: Optional[str] = None


@dataclass
class OrderRecord:
    order_id: str
    run_id: str
    agent_id: str
    side: str  # BID / ASK
    order_type: str  # LIMIT / MARKET
    price: float
    quantity: int
    timestamp_step: int
    status: str = "OPEN"


@dataclass
class TradeRecord:
    trade_id: str
    run_id: str
    buy_order_id: str
    sell_order_id: str
    price: float
    quantity: int
    timestamp_step: int


@dataclass
class AgentMetricRecord:
    run_id: str
    agent_id: str
    episode: int
    pnl: float
    sharpe_ratio: float
    inventory: int
    reward: float


@dataclass
class BookSnapshotRecord:
    episode: int
    step: int
    best_bid: Optional[float]
    best_ask: Optional[float]
    mid: Optional[float]
    spread: Optional[float]
    bids: List[List[Union[float, int]]] = field(default_factory=list)
    asks: List[List[Union[float, int]]] = field(default_factory=list)
    depth: Dict[str, List[List[Union[float, int]]]] = field(default_factory=dict)
    agents: Dict[str, Dict[str, Union[int, float]]] = field(default_factory=dict)
    inventories: Dict[str, int] = field(default_factory=dict)
    pnls: Dict[str, float] = field(default_factory=dict)


@dataclass
class TimeseriesRecord:
    episode: int
    step: int
    best_bid: Optional[float]
    best_ask: Optional[float]
    mid: Optional[float]
    spread: Optional[float]
    agents: Dict[str, Dict[str, Union[int, float]]] = field(default_factory=dict)
    inventories: Dict[str, int] = field(default_factory=dict)
    pnls: Dict[str, float] = field(default_factory=dict)
    bids: List[List[Union[float, int]]] = field(default_factory=list)
    asks: List[List[Union[float, int]]] = field(default_factory=list)
    depth: Dict[str, List[List[Union[float, int]]]] = field(default_factory=dict)


class MetricsLogger:
    """
    Repository-style simulation logger storing orders, trades, agent metrics,
    book snapshots, timeseries, and equilibrium results per run.
    """

    def __init__(self, base_dir: str = "data/runs"):
        self.base_dir = base_dir
        os.makedirs(self.base_dir, exist_ok=True)
        self.current_run: Optional[SimulationRunRecord] = None
        self.orders: List[Dict[str, Any]] = []
        self.trades: List[Dict[str, Any]] = []
        self.agent_metrics: List[Dict[str, Any]] = []
        self.price_series: List[Dict[str, Any]] = []
        self.equilibrium_analysis: Optional[Dict[str, Any]] = None
        self.book_snapshots: List[Dict[str, Any]] = []
        self.timeseries: List[Dict[str, Any]] = []

    def start_run(
        self,
        run_id: str,
        seed: Optional[int] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> SimulationRunRecord:
        """Initialize a new simulation run record."""
        started_at = datetime.now(timezone.utc).isoformat()
        self.current_run = SimulationRunRecord(
            run_id=run_id,
            seed=seed,
            config_snapshot=config or {},
            started_at=started_at,
            status="RUNNING"
        )
        self.orders = []
        self.trades = []
        self.agent_metrics = []
        self.price_series = []
        self.equilibrium_analysis = None
        self.book_snapshots = []
        self.timeseries = []

        run_path = os.path.join(self.base_dir, run_id)
        os.makedirs(run_path, exist_ok=True)
        self._flush_metadata()
        return self.current_run

    def log_order(
        self,
        order_id: str,
        run_id: str,
        agent_id: str,
        side: str,
        order_type: str,
        price: float,
        quantity: int,
        timestamp_step: int,
        status: str = "OPEN",
    ):
        """Append an order entry."""
        record = OrderRecord(
            order_id=order_id,
            run_id=run_id,
            agent_id=agent_id,
            side=side,
            order_type=order_type,
            price=price,
            quantity=quantity,
            timestamp_step=timestamp_step,
            status=status,
        )
        self.orders.append(asdict(record))

    def log_trade(
        self,
        trade_id: str,
        run_id: str,
        buy_order_id: str,
        sell_order_id: str,
        price: float,
        quantity: int,
        timestamp_step: int,
        episode: int = 1,
    ):
        """Append a trade entry and update price series."""
        record = TradeRecord(
            trade_id=trade_id,
            run_id=run_id,
            buy_order_id=buy_order_id,
            sell_order_id=sell_order_id,
            price=price,
            quantity=quantity,
            timestamp_step=timestamp_step,
        )
        self.trades.append(asdict(record))
        self.price_series.append({
            "episode": episode,
            "step": timestamp_step,
            "price": price,
            "quantity": quantity,
            "trade_id": trade_id,
        })

    def log_agent_metric(
        self,
        run_id: str,
        agent_id: str,
        episode: int,
        pnl: float,
        sharpe_ratio: float,
        inventory: int,
        reward: float,
    ):
        """Append an agent performance metric entry."""
        record = AgentMetricRecord(
            run_id=run_id,
            agent_id=agent_id,
            episode=episode,
            pnl=float(pnl),
            sharpe_ratio=float(sharpe_ratio),
            inventory=int(inventory),
            reward=float(reward),
        )
        self.agent_metrics.append(asdict(record))

    def log_book_snapshot(
        self,
        episode: int,
        step: int,
        best_bid: Optional[float],
        best_ask: Optional[float],
        mid: Optional[float],
        spread: Optional[float],
        bids: Optional[List[Any]] = None,
        asks: Optional[List[Any]] = None,
        depth: Optional[Dict[str, Any]] = None,
        agents: Optional[Dict[str, Any]] = None,
        inventories: Optional[Dict[str, int]] = None,
        pnls: Optional[Dict[str, float]] = None,
        run_id: Optional[str] = None,
    ):
        """Append a downsampled order book depth snapshot entry."""
        bids_clean = []
        asks_clean = []
        if depth:
            bids_clean = [list(x) if isinstance(x, (list, tuple)) else x for x in depth.get("bids", [])][:5]
            asks_clean = [list(x) if isinstance(x, (list, tuple)) else x for x in depth.get("asks", [])][:5]
        else:
            if bids:
                bids_clean = [list(x) if isinstance(x, (list, tuple)) else x for x in bids][:5]
            if asks:
                asks_clean = [list(x) if isinstance(x, (list, tuple)) else x for x in asks][:5]

        depth_dict = {"bids": bids_clean, "asks": asks_clean}
        agents_dict = agents or {}
        inv_dict = inventories or {a: data.get("inventory", 0) for a, data in agents_dict.items()}
        pnl_dict = pnls or {a: data.get("pnl", 0.0) for a, data in agents_dict.items()}

        record = {
            "episode": int(episode),
            "step": int(step),
            "best_bid": float(best_bid) if best_bid is not None else None,
            "best_ask": float(best_ask) if best_ask is not None else None,
            "mid": float(mid) if mid is not None else None,
            "spread": float(spread) if spread is not None else None,
            "bids": bids_clean,
            "asks": asks_clean,
            "depth": depth_dict,
            "agents": agents_dict,
            "inventories": inv_dict,
            "pnls": pnl_dict,
        }
        self.book_snapshots.append(record)

    def log_timeseries(
        self,
        episode: int,
        step: int,
        best_bid: Optional[float],
        best_ask: Optional[float],
        mid: Optional[float],
        spread: Optional[float],
        agents: Optional[Dict[str, Any]] = None,
        inventories: Optional[Dict[str, int]] = None,
        pnls: Optional[Dict[str, float]] = None,
        bids: Optional[List[Any]] = None,
        asks: Optional[List[Any]] = None,
        depth: Optional[Dict[str, Any]] = None,
        run_id: Optional[str] = None,
    ):
        """Append a telemetry timeseries point entry."""
        bids_clean = []
        asks_clean = []
        if depth:
            bids_clean = [list(x) if isinstance(x, (list, tuple)) else x for x in depth.get("bids", [])][:5]
            asks_clean = [list(x) if isinstance(x, (list, tuple)) else x for x in depth.get("asks", [])][:5]
        else:
            if bids:
                bids_clean = [list(x) if isinstance(x, (list, tuple)) else x for x in bids][:5]
            if asks:
                asks_clean = [list(x) if isinstance(x, (list, tuple)) else x for x in asks][:5]

        depth_dict = {"bids": bids_clean, "asks": asks_clean}
        agents_dict = agents or {}
        inv_dict = inventories or {a: data.get("inventory", 0) for a, data in agents_dict.items()}
        pnl_dict = pnls or {a: data.get("pnl", 0.0) for a, data in agents_dict.items()}

        record = {
            "episode": int(episode),
            "step": int(step),
            "best_bid": float(best_bid) if best_bid is not None else None,
            "best_ask": float(best_ask) if best_ask is not None else None,
            "mid": float(mid) if mid is not None else None,
            "spread": float(spread) if spread is not None else None,
            "agents": agents_dict,
            "inventories": inv_dict,
            "pnls": pnl_dict,
            "bids": bids_clean,
            "asks": asks_clean,
            "depth": depth_dict,
        }
        self.timeseries.append(record)

    def log_equilibrium_analysis(
        self,
        run_id: str,
        payoff_matrix: Any,
        equilibrium_candidates: List[Any],
        method: str = "LEMKE_HOWSON",
        strategies: Optional[List[str]] = None,
        best_response_trace: Optional[List[Any]] = None,
        standard_errors: Optional[Any] = None,
    ):
        """Save post-hoc game theoretic equilibrium analysis with per-cell standard errors."""
        matrix_list = (
            payoff_matrix.tolist()
            if isinstance(payoff_matrix, np.ndarray)
            else payoff_matrix
        )

        se_list = None
        if standard_errors is not None:
            if isinstance(standard_errors, np.ndarray):
                se_list = standard_errors.tolist()
            elif isinstance(standard_errors, dict):
                se_list = {
                    k: (v.tolist() if isinstance(v, np.ndarray) else v)
                    for k, v in standard_errors.items()
                }
            elif isinstance(standard_errors, (tuple, list)):
                se_list = [
                    (x.tolist() if isinstance(x, np.ndarray) else x)
                    for x in standard_errors
                ]
            else:
                se_list = standard_errors

        self.equilibrium_analysis = {
            "run_id": run_id,
            "method": method,
            "payoff_matrix": matrix_list,
            "strategies": strategies or [],
            "equilibrium_candidates": equilibrium_candidates,
            "best_response_trace": best_response_trace or [],
            "standard_errors": se_list,
            "computed_at": datetime.now(timezone.utc).isoformat(),
        }

    def finish_run(self, status: str = "COMPLETE", error_message: Optional[str] = None):
        """Mark run finished and persist all run artifacts to disk."""
        if self.current_run is not None:
            self.current_run.status = status
            self.current_run.completed_at = datetime.now(timezone.utc).isoformat()
            self.current_run.error_message = error_message
            self.save()

    def save(self, run_id: Optional[str] = None):
        """Persist all in-memory collections to data/runs/<run_id>/."""
        target_run_id = run_id or (self.current_run.run_id if self.current_run else None)
        if not target_run_id:
            return

        run_path = os.path.join(self.base_dir, target_run_id)
        os.makedirs(run_path, exist_ok=True)

        self._flush_metadata()

        # Save orders (compact)
        if self.orders:
            orders_file = os.path.join(run_path, "orders.json")
            with open(orders_file, "w") as f:
                json.dump(self.orders, f, separators=(',', ':'))

        # Save trades (compact)
        if self.trades:
            trades_file = os.path.join(run_path, "trades.json")
            with open(trades_file, "w") as f:
                json.dump(self.trades, f, separators=(',', ':'))

        # Save price series (compact)
        if self.price_series:
            price_file = os.path.join(run_path, "price_series.json")
            with open(price_file, "w") as f:
                json.dump(self.price_series, f, separators=(',', ':'))

        # Save agent metrics
        if self.agent_metrics:
            metrics_file = os.path.join(run_path, "agent_metrics.json")
            with open(metrics_file, "w") as f:
                json.dump(self.agent_metrics, f, indent=2)

        # Save equilibrium analysis
        if self.equilibrium_analysis:
            eq_file = os.path.join(run_path, "equilibrium_analysis.json")
            with open(eq_file, "w") as f:
                json.dump(self.equilibrium_analysis, f, indent=2)

        # Save book snapshots (compact downsampled)
        if self.book_snapshots:
            book_file = os.path.join(run_path, "book_snapshots.json")
            with open(book_file, "w") as f:
                json.dump(self.book_snapshots, f, separators=(',', ':'))

        # Save timeseries (compact)
        if self.timeseries:
            timeseries_file = os.path.join(run_path, "timeseries.json")
            with open(timeseries_file, "w") as f:
                json.dump(self.timeseries, f, separators=(',', ':'))

    def _flush_metadata(self):
        if not self.current_run:
            return
        run_path = os.path.join(self.base_dir, self.current_run.run_id)
        os.makedirs(run_path, exist_ok=True)
        summary_file = os.path.join(run_path, "run_summary.json")
        with open(summary_file, "w") as f:
            json.dump(asdict(self.current_run), f, indent=2)

    # ------------------------------------------------------------------
    # Query Interface (API Specification § TRD / LLD)
    # ------------------------------------------------------------------

    def get_run_summary(self, run_id: str) -> Dict[str, Any]:
        """Returns run config, seed, and status."""
        run_path = os.path.join(self.base_dir, run_id, "run_summary.json")
        if os.path.exists(run_path):
            with open(run_path, "r") as f:
                return json.load(f)
        if self.current_run and self.current_run.run_id == run_id:
            return asdict(self.current_run)
        raise FileNotFoundError(f"Run summary not found for run_id: {run_id}")

    def get_price_series(self, run_id: str) -> List[Dict[str, Any]]:
        """Returns the simulated time-series of trade prices and volumes."""
        run_path = os.path.join(self.base_dir, run_id, "price_series.json")
        if os.path.exists(run_path):
            with open(run_path, "r") as f:
                return json.load(f)
        if self.current_run and self.current_run.run_id == run_id:
            return self.price_series
        return []

    def get_agent_metrics(
        self, run_id: str, agent_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Returns per-agent PnL, Sharpe ratio, and inventory history."""
        run_path = os.path.join(self.base_dir, run_id, "agent_metrics.json")
        data = []
        if os.path.exists(run_path):
            with open(run_path, "r") as f:
                data = json.load(f)
        elif self.current_run and self.current_run.run_id == run_id:
            data = self.agent_metrics

        if agent_id is not None:
            return [m for m in data if m.get("agent_id") == agent_id]
        return data

    def get_equilibrium_analysis(self, run_id: str) -> Optional[Dict[str, Any]]:
        """Returns the logged Nashpy equilibrium candidates and best-response trace."""
        run_path = os.path.join(self.base_dir, run_id, "equilibrium_analysis.json")
        if os.path.exists(run_path):
            with open(run_path, "r") as f:
                return json.load(f)
        if self.current_run and self.current_run.run_id == run_id:
            return self.equilibrium_analysis
        return None

    def get_book_snapshots(self, run_id: str) -> List[Dict[str, Any]]:
        """Returns the downsampled order book depth snapshots."""
        run_path = os.path.join(self.base_dir, run_id, "book_snapshots.json")
        if os.path.exists(run_path):
            with open(run_path, "r") as f:
                return json.load(f)
        if self.current_run and self.current_run.run_id == run_id:
            return self.book_snapshots
        return []

    def get_timeseries(self, run_id: str) -> List[Dict[str, Any]]:
        """Returns the telemetry time-series of prices, spreads, and agent states."""
        run_path = os.path.join(self.base_dir, run_id, "timeseries.json")
        if os.path.exists(run_path):
            with open(run_path, "r") as f:
                return json.load(f)
        if self.current_run and self.current_run.run_id == run_id:
            return self.timeseries
        return []

    def get_validation_report(self, run_id: str) -> Optional[Dict[str, Any]]:
        """Returns the econometric stylized facts validation report if present."""
        run_path = os.path.join(self.base_dir, run_id, "validation_report.json")
        if os.path.exists(run_path):
            with open(run_path, "r") as f:
                return json.load(f)
        return None

    def get_stress_report(self, run_id: str) -> Optional[Dict[str, Any]]:
        """Returns the stress testing resilience report if present."""
        run_path = os.path.join(self.base_dir, run_id, "stress_report.json")
        if os.path.exists(run_path):
            with open(run_path, "r") as f:
                return json.load(f)
        return None

    def list_runs(self) -> List[str]:
        """List all available run IDs in the base directory."""
        if not os.path.exists(self.base_dir):
            return []
        return sorted([
            d for d in os.listdir(self.base_dir)
            if os.path.isdir(os.path.join(self.base_dir, d))
        ])


# Global default logger instance
default_logger = MetricsLogger()

def get_run_summary(run_id: str) -> Dict[str, Any]:
    return default_logger.get_run_summary(run_id)

def get_price_series(run_id: str) -> List[Dict[str, Any]]:
    return default_logger.get_price_series(run_id)

def get_agent_metrics(run_id: str, agent_id: Optional[str] = None) -> List[Dict[str, Any]]:
    return default_logger.get_agent_metrics(run_id, agent_id)

def get_equilibrium_analysis(run_id: str) -> Optional[Dict[str, Any]]:
    return default_logger.get_equilibrium_analysis(run_id)

def get_book_snapshots(run_id: str) -> List[Dict[str, Any]]:
    return default_logger.get_book_snapshots(run_id)

def get_timeseries(run_id: str) -> List[Dict[str, Any]]:
    return default_logger.get_timeseries(run_id)

def get_validation_report(run_id: str) -> Optional[Dict[str, Any]]:
    return default_logger.get_validation_report(run_id)

def get_stress_report(run_id: str) -> Optional[Dict[str, Any]]:
    return default_logger.get_stress_report(run_id)

