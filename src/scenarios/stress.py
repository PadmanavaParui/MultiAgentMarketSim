"""
Market Stress-Testing Framework for MultiAgentMarketSim.

Provides three canonical stress scenarios:
  1. 'flash_crash': ShockAgent fires large aggressive market sells, sweeping the bid ladder.
  2. 'informed_flow': InformedTrader aggressively front-runs a future price jump.
  3. 'liquidity_withdrawal': Market makers cease quoting for N steps (cancelling resting depth).

Each scenario executes paired, counterfactual baseline vs. stressed episodes using
identical random seeds. Computes standardized resilience and microstructure metrics:
  - max drawdown of mid
  - peak spread vs. baseline spread
  - MM max |inventory|
  - MM PnL vs. baseline
  - adverse-selection loss (MM PnL on trades against the informed / shock agent)
  - recovery time (steps after shock ceases until mid and spread return near baseline)

Saves results to data/runs/<run_id>/stress_report.json and provides side-by-side
resilience comparisons between MarketMakerAgent and AvellanedaStoikovMM.
"""

from __future__ import annotations

import os
import json
from datetime import datetime, timezone
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np

from src.env.market_env import MarketEnvironment
from src.agents.market_maker import MarketMakerAgent
from src.agents.avellaneda_stoikov import AvellanedaStoikovMM
from src.agents.zero_intelligence import ZeroIntelligenceAgent
from src.agents.informed_trader import InformedTrader
from src.agents.shock_agent import ShockAgent


# ======================================================================
# 1. Pure Metric Computation Functions
# ======================================================================

def compute_max_mid_drawdown(mid_prices: Sequence[Optional[float]]) -> float:
    """
    Compute peak-to-trough maximum drawdown of mid prices.

    Parameters
    ----------
    mid_prices : Sequence[Optional[float]]
        Time series of mid prices (None entries are skipped).

    Returns
    -------
    float
        Maximum drop from a preceding peak (non-negative).
    """
    clean = [float(p) for p in mid_prices if p is not None and not np.isnan(p)]
    if len(clean) < 2:
        return 0.0
    arr = np.asarray(clean, dtype=np.float64)
    running_max = np.maximum.accumulate(arr)
    drawdowns = running_max - arr
    return float(np.max(drawdowns))


def compute_peak_vs_baseline_spread(
    stressed_spreads: Sequence[Optional[float]],
    baseline_spreads: Sequence[Optional[float]],
    max_penalty_spread: float = 20.0,
) -> Dict[str, float]:
    """
    Compute peak spread in stressed episode compared to baseline average spread.

    Parameters
    ----------
    stressed_spreads : Sequence[Optional[float]]
        Spreads across steps in the stressed episode.
    baseline_spreads : Sequence[Optional[float]]
        Spreads across steps in the paired baseline episode.
    max_penalty_spread : float
        Fallback spread value when a book side is empty.

    Returns
    -------
    Dict[str, float]
        Contains 'peak_spread', 'baseline_spread', 'spread_diff', and 'spread_ratio'.
    """
    s_clean = [
        float(s) if (s is not None and not np.isnan(s)) else max_penalty_spread
        for s in stressed_spreads
    ]
    b_clean = [
        float(s) if (s is not None and not np.isnan(s)) else max_penalty_spread
        for s in baseline_spreads
    ]

    peak_stressed = float(np.max(s_clean)) if s_clean else 0.0
    avg_baseline = float(np.mean(b_clean)) if b_clean else 0.0
    diff = peak_stressed - avg_baseline
    ratio = (peak_stressed / avg_baseline) if avg_baseline > 1e-9 else 1.0

    return {
        "peak_spread": round(peak_stressed, 4),
        "baseline_spread": round(avg_baseline, 4),
        "spread_diff": round(diff, 4),
        "spread_ratio": round(ratio, 4),
    }


def compute_mm_max_inventory(inventories: Sequence[int]) -> int:
    """
    Compute peak absolute inventory exposure for the market maker.

    Parameters
    ----------
    inventories : Sequence[int]
        MM inventory trajectory across episode steps.

    Returns
    -------
    int
        Maximum absolute inventory observed.
    """
    if len(inventories) == 0:
        return 0
    arr = np.abs(np.asarray(inventories, dtype=np.int64))
    return int(np.max(arr))


def compute_mm_pnl_vs_baseline(stressed_pnl: float, baseline_pnl: float) -> float:
    """
    Compute delta of MM terminal PnL between stressed and baseline episodes.

    Returns
    -------
    float
        stressed_pnl - baseline_pnl
    """
    return round(float(stressed_pnl - baseline_pnl), 4)


def compute_adverse_selection(
    trades: List[Any],
    all_orders: Dict[str, Any],
    mm_id: str,
    target_agent_id: Optional[str],
    terminal_price: float,
) -> Dict[str, float]:
    """
    Compute adverse-selection loss: MM PnL on trades against the informed/shock agent.

    For trades where the MM was the counterparty to target_agent_id:
      - MM bought from target: cash -= price * qty, inv += qty
      - MM sold to target: cash += price * qty, inv -= qty
    Marked to market at terminal reference price S_T:
      PnL_against_target = cash_flow + inv_flow * S_T
      adverse_selection_loss = - PnL_against_target

    Returns
    -------
    Dict[str, float]
        Keys: 'adverse_selection_loss', 'adverse_selection_pnl', 'trades_count', 'inventory_accumulated'
    """
    if not target_agent_id:
        return {
            "adverse_selection_loss": 0.0,
            "adverse_selection_pnl": 0.0,
            "trades_count": 0,
            "inventory_accumulated": 0,
        }

    mm_cash = 0.0
    mm_inv = 0
    trade_count = 0

    for t in trades:
        buy_order = all_orders.get(t.buy_order_id)
        sell_order = all_orders.get(t.sell_order_id)
        if not buy_order or not sell_order:
            continue

        buy_agent = buy_order.agent_id
        sell_agent = sell_order.agent_id

        if buy_agent == mm_id and sell_agent == target_agent_id:
            # MM bought from target
            mm_cash -= float(t.price * t.quantity)
            mm_inv += int(t.quantity)
            trade_count += 1
        elif sell_agent == mm_id and buy_agent == target_agent_id:
            # MM sold to target
            mm_cash += float(t.price * t.quantity)
            mm_inv -= int(t.quantity)
            trade_count += 1

    pnl = mm_cash + float(mm_inv * terminal_price)
    loss = -pnl

    return {
        "adverse_selection_loss": round(loss, 4),
        "adverse_selection_pnl": round(pnl, 4),
        "trades_count": trade_count,
        "inventory_accumulated": mm_inv,
    }


def compute_recovery_time(
    stressed_mids: Sequence[Optional[float]],
    stressed_spreads: Sequence[Optional[float]],
    baseline_mid_target: float,
    baseline_spread_target: float,
    shock_end_step: int,
    tolerance_mid: float = 1.0,
    tolerance_spread_factor: float = 1.5,
    max_penalty_spread: float = 20.0,
) -> Tuple[bool, Optional[int]]:
    """
    Compute recovery time in steps after shock cessation until mid and spread return near baseline.

    Parameters
    ----------
    stressed_mids : Sequence[Optional[float]]
        Mid prices per step in the stressed episode.
    stressed_spreads : Sequence[Optional[float]]
        Spreads per step in the stressed episode.
    baseline_mid_target : float
        Reference mid price target (e.g. mean baseline mid).
    baseline_spread_target : float
        Reference spread target (e.g. mean baseline spread).
    shock_end_step : int
        Step at which the stress shock finishes (1-indexed).
    tolerance_mid : float
        Acceptable absolute difference in mid price (|mid - baseline_mid| <= tolerance_mid).
    tolerance_spread_factor : float
        Acceptable spread multiple (spread <= baseline_spread * tolerance_spread_factor + 0.1).
    max_penalty_spread : float
        Spread value assigned when book side is empty.

    Returns
    -------
    Tuple[bool, Optional[int]]
        (recovered, recovery_time_steps).
        If recovered is False, recovery_time_steps is None.
    """
    total_steps = len(stressed_mids)
    spread_threshold = baseline_spread_target * tolerance_spread_factor + 0.1

    for idx in range(total_steps):
        step = idx + 1
        if step < shock_end_step:
            continue

        m = stressed_mids[idx]
        s = stressed_spreads[idx]
        if m is None:
            continue
        spread_val = s if s is not None else max_penalty_spread

        mid_near = abs(m - baseline_mid_target) <= tolerance_mid
        spread_near = spread_val <= spread_threshold

        if mid_near and spread_near:
            recovery_steps = step - shock_end_step
            return True, recovery_steps

    return False, None


# ======================================================================
# 2. Scenario Configurations & Runner
# ======================================================================

@dataclass
class ScenarioConfig:
    name: str  # 'flash_crash', 'informed_flow', 'liquidity_withdrawal'
    shock_step: int = 15
    n_steps: int = 4
    shock_quantity: int = 40
    direction: str = "sell"
    jump_size: float = 2.0
    informed_quantity: int = 10
    withdrawal_steps: int = 6
    tolerance_mid: float = 1.0
    tolerance_spread_factor: float = 1.5


@dataclass
class ScenarioMetrics:
    scenario_name: str
    mm_type: str
    max_drawdown_mid: float
    peak_spread: float
    baseline_spread: float
    spread_diff: float
    spread_ratio: float
    mm_max_abs_inventory: int
    mm_pnl_stressed: float
    mm_pnl_baseline: float
    mm_pnl_vs_baseline: float
    adverse_selection_loss: float
    adverse_selection_pnl: float
    recovered: bool
    recovery_time: Optional[int]


def _instantiate_mm(mm_type: str, agent_id: str = "mm_0", seed: Optional[int] = None, max_steps: int = 50) -> Any:
    """Instantiate either MarketMakerAgent or AvellanedaStoikovMM."""
    if mm_type in ("MM", "MarketMakerAgent"):
        return MarketMakerAgent(agent_id=agent_id, seed=seed, spread=0.5, quantity=10, max_inventory=40)
    elif mm_type in ("AvellanedaStoikovMM", "AS_MM", "AS"):
        return AvellanedaStoikovMM(
            agent_id=agent_id,
            seed=seed,
            gamma=0.1,
            k=1.5,
            window=10,
            quantity=10,
            T=1.0,
            max_steps=max_steps,
        )
    else:
        raise ValueError(f"Unknown mm_type: {mm_type}")


class StressScenarioRunner:
    """
    Executes paired baseline and stressed episodes across stress scenarios
    and aggregates comparative resilience metrics.
    """

    def __init__(
        self,
        episodes: int = 5,
        cycles_per_episode: int = 50,
        seed: int = 42,
        output_dir: str = "data/runs",
    ):
        self.episodes = max(1, episodes)
        self.cycles = max(10, cycles_per_episode)
        self.seed = seed
        self.output_dir = output_dir

    def run_episode(
        self,
        scenario_config: ScenarioConfig,
        mm_type: str,
        is_stressed: bool,
        episode_seed: int,
    ) -> Dict[str, Any]:
        """
        Executes a single market simulation episode under baseline or stressed conditions.
        """
        sc_name = scenario_config.name
        shock_step = scenario_config.shock_step
        n_steps = scenario_config.n_steps

        # Construct agent pool
        agent_ids = ["mm_0", "zi_0", "zi_1"]
        target_agent_id: Optional[str] = None

        if sc_name == "flash_crash":
            agent_ids.append("shock_0")
            target_agent_id = "shock_0"
        elif sc_name == "informed_flow":
            agent_ids.extend(["informed_0", "oracle_0"])
            target_agent_id = "informed_0"
        elif sc_name == "liquidity_withdrawal":
            pass

        env = MarketEnvironment(
            agents=agent_ids,
            max_cycles=self.cycles,
            dict_spaces=True,
            run_id=f"stress_{sc_name}_{mm_type}_{'stressed' if is_stressed else 'baseline'}",
        )
        obs, _ = env.reset(seed=episode_seed)

        # Create agents
        mm_agent = _instantiate_mm(mm_type, "mm_0", seed=episode_seed, max_steps=self.cycles)
        zi0 = ZeroIntelligenceAgent("zi_0", seed=episode_seed + 101, aggression=0.7, max_quantity=10)
        zi1 = ZeroIntelligenceAgent("zi_1", seed=episode_seed + 202, aggression=0.7, max_quantity=10)
        agents_map: Dict[str, Any] = {"mm_0": mm_agent, "zi_0": zi0, "zi_1": zi1}

        if sc_name == "flash_crash":
            qty = scenario_config.shock_quantity if is_stressed else 0
            shock = ShockAgent(
                "shock_0",
                shock_step=shock_step,
                n_steps=n_steps,
                direction=scenario_config.direction,
                quantity=qty,
                seed=episode_seed + 303,
            )
            agents_map["shock_0"] = shock

        elif sc_name == "informed_flow":
            j_size = scenario_config.jump_size if is_stressed else 0.0
            inf = InformedTrader(
                "informed_0",
                jump_step=shock_step,
                jump_size=j_size,
                quantity=scenario_config.informed_quantity,
                seed=episode_seed + 303,
            )
            agents_map["informed_0"] = inf

        # Trajectory tracking
        mids: List[float] = []
        spreads: List[Optional[float]] = []
        mm_inventories: List[int] = []
        last_infos: Dict[str, Any] = {}

        for step in range(1, self.cycles + 1):
            if not env.agents:
                break

            # Action dispatch
            actions: Dict[str, Any] = {}
            for a_id in env.agents:
                if a_id not in obs:
                    continue

                # Liquidity withdrawal logic: MM halts quoting during [withdrawal_start, withdrawal_start + n_steps)
                if sc_name == "liquidity_withdrawal" and a_id == "mm_0" and is_stressed:
                    if shock_step <= step < (shock_step + scenario_config.withdrawal_steps):
                        # Quoting halted
                        actions["mm_0"] = {"order_type": 0, "side": 0, "price_offset": 0.0, "quantity": 0}
                        continue

                # Oracle agent for price jump in informed_flow
                if sc_name == "informed_flow" and a_id == "oracle_0":
                    if step == shock_step and scenario_config.jump_size != 0.0:
                        actions["oracle_0"] = {
                            "order_type": 1,  # LIMIT
                            "side": 0 if scenario_config.jump_size > 0 else 1,
                            "price_offset": scenario_config.jump_size,
                            "quantity": 10,
                        }
                    else:
                        actions["oracle_0"] = {"order_type": 0, "side": 0, "price_offset": 0.0, "quantity": 0}
                    continue

                actions[a_id] = agents_map[a_id].act(obs[a_id])

            # In liquidity withdrawal: cancel resting MM orders at onset of withdrawal
            if sc_name == "liquidity_withdrawal" and is_stressed and step == shock_step:
                for ord_id, o in list(env.order_book.orders.items()):
                    if o.agent_id == "mm_0":
                        env.order_book.cancel_order(ord_id)

            # Step the environment
            next_obs, rewards, terminations, truncations, step_infos = env.step(actions)
            last_infos = step_infos

            # Record telemetry
            bid = env.order_book.best_bid()
            ask = env.order_book.best_ask()
            if bid is not None and ask is not None:
                mid = round((bid.price + ask.price) / 2.0, 4)
                sp = round(ask.price - bid.price, 4)
            else:
                mid = env._get_reference_price()
                sp = 20.0

            mids.append(mid)
            spreads.append(sp)
            mm_inventories.append(int(env.inventory.get("mm_0", 0)))

            # Update agents
            for a_id in env.agents:
                if a_id in agents_map and a_id in rewards and a_id in next_obs:
                    agents_map[a_id].update(rewards[a_id], next_obs[a_id])

            obs = next_obs

        final_pnl = float(last_infos.get("mm_0", {}).get("pnl", 0.0))
        ref_price = env._get_reference_price()

        return {
            "mids": mids,
            "spreads": spreads,
            "mm_inventories": mm_inventories,
            "final_pnl": final_pnl,
            "terminal_price": ref_price,
            "trades": list(env.matching_engine.trades_history),
            "all_orders": dict(env._all_orders),
            "target_agent_id": target_agent_id,
        }

    def evaluate_scenario(
        self,
        scenario_config: ScenarioConfig,
        mm_type: str,
    ) -> ScenarioMetrics:
        """
        Runs paired baseline vs stressed episodes for the scenario and MM type,
        averaging metrics across seeds.
        """
        sc_name = scenario_config.name
        shock_step = scenario_config.shock_step

        if sc_name == "flash_crash":
            shock_end_step = shock_step + scenario_config.n_steps
        elif sc_name == "informed_flow":
            shock_end_step = shock_step + 1
        elif sc_name == "liquidity_withdrawal":
            shock_end_step = shock_step + scenario_config.withdrawal_steps
        else:
            shock_end_step = shock_step + 1

        drawdowns = []
        peak_spreads = []
        base_spreads = []
        spread_diffs = []
        spread_ratios = []
        max_inventories = []
        stressed_pnls = []
        base_pnls = []
        pnl_diffs = []
        adverse_losses = []
        adverse_pnls = []
        recovery_times = []
        recovered_count = 0

        for ep in range(self.episodes):
            ep_seed = self.seed + ep * 1000

            # 1. Baseline episode
            base_run = self.run_episode(
                scenario_config=scenario_config,
                mm_type=mm_type,
                is_stressed=False,
                episode_seed=ep_seed,
            )

            # 2. Stressed episode
            stress_run = self.run_episode(
                scenario_config=scenario_config,
                mm_type=mm_type,
                is_stressed=True,
                episode_seed=ep_seed,
            )

            # Compute metrics for this pair
            dd = compute_max_mid_drawdown(stress_run["mids"])
            drawdowns.append(dd)

            sp_res = compute_peak_vs_baseline_spread(stress_run["spreads"], base_run["spreads"])
            peak_spreads.append(sp_res["peak_spread"])
            base_spreads.append(sp_res["baseline_spread"])
            spread_diffs.append(sp_res["spread_diff"])
            spread_ratios.append(sp_res["spread_ratio"])

            max_inv = compute_mm_max_inventory(stress_run["mm_inventories"])
            max_inventories.append(max_inv)

            s_pnl = stress_run["final_pnl"]
            b_pnl = base_run["final_pnl"]
            stressed_pnls.append(s_pnl)
            base_pnls.append(b_pnl)
            pnl_diffs.append(compute_mm_pnl_vs_baseline(s_pnl, b_pnl))

            # Adverse selection against target agent
            target_id = stress_run["target_agent_id"]
            adv_res = compute_adverse_selection(
                trades=stress_run["trades"],
                all_orders=stress_run["all_orders"],
                mm_id="mm_0",
                target_agent_id=target_id,
                terminal_price=stress_run["terminal_price"],
            )
            adverse_losses.append(adv_res["adverse_selection_loss"])
            adverse_pnls.append(adv_res["adverse_selection_pnl"])

            # Recovery time
            base_mid_target = float(np.mean(base_run["mids"])) if base_run["mids"] else 100.0
            base_sp_target = float(np.mean(base_run["spreads"])) if base_run["spreads"] else 1.0

            rec_ok, rec_time = compute_recovery_time(
                stressed_mids=stress_run["mids"],
                stressed_spreads=stress_run["spreads"],
                baseline_mid_target=base_mid_target,
                baseline_spread_target=base_sp_target,
                shock_end_step=shock_end_step,
                tolerance_mid=scenario_config.tolerance_mid,
                tolerance_spread_factor=scenario_config.tolerance_spread_factor,
            )
            if rec_ok and rec_time is not None:
                recovered_count += 1
                recovery_times.append(rec_time)

        mean_rec_time = round(float(np.mean(recovery_times)), 2) if recovery_times else None

        return ScenarioMetrics(
            scenario_name=sc_name,
            mm_type=mm_type,
            max_drawdown_mid=round(float(np.mean(drawdowns)), 4),
            peak_spread=round(float(np.mean(peak_spreads)), 4),
            baseline_spread=round(float(np.mean(base_spreads)), 4),
            spread_diff=round(float(np.mean(spread_diffs)), 4),
            spread_ratio=round(float(np.mean(spread_ratios)), 4),
            mm_max_abs_inventory=int(round(float(np.mean(max_inventories)))),
            mm_pnl_stressed=round(float(np.mean(stressed_pnls)), 4),
            mm_pnl_baseline=round(float(np.mean(base_pnls)), 4),
            mm_pnl_vs_baseline=round(float(np.mean(pnl_diffs)), 4),
            adverse_selection_loss=round(float(np.mean(adverse_losses)), 4),
            adverse_selection_pnl=round(float(np.mean(adverse_pnls)), 4),
            recovered=(recovered_count > 0),
            recovery_time=int(round(mean_rec_time)) if mean_rec_time is not None else None,
        )

    def run_suite(
        self,
        scenarios: Optional[List[str]] = None,
        mm_types: Optional[List[str]] = None,
        run_id: str = "stress_run",
    ) -> Dict[str, Any]:
        """
        Executes stress scenarios for both MarketMakerAgent and AvellanedaStoikovMM,
        compares performance, and persists report to data/runs/<run_id>/stress_report.json.
        """
        if scenarios is None or "all" in scenarios:
            scenario_names = ["flash_crash", "informed_flow", "liquidity_withdrawal"]
        else:
            scenario_names = scenarios

        if mm_types is None or "both" in mm_types:
            evaluated_mm_types = ["MarketMakerAgent", "AvellanedaStoikovMM"]
        else:
            evaluated_mm_types = mm_types

        scenario_configs = {
            "flash_crash": ScenarioConfig(name="flash_crash", shock_step=15, n_steps=4, shock_quantity=40, direction="sell"),
            "informed_flow": ScenarioConfig(name="informed_flow", shock_step=18, jump_size=2.0, informed_quantity=10),
            "liquidity_withdrawal": ScenarioConfig(name="liquidity_withdrawal", shock_step=15, withdrawal_steps=6),
        }

        report_scenarios: Dict[str, Any] = {}

        for sc_name in scenario_names:
            if sc_name not in scenario_configs:
                continue
            cfg = scenario_configs[sc_name]
            sc_results: Dict[str, Any] = {}

            for m_type in evaluated_mm_types:
                metrics = self.evaluate_scenario(cfg, m_type)
                sc_results[m_type] = asdict(metrics)

            # Compute comparative deltas if both MMs were evaluated
            if "MarketMakerAgent" in sc_results and "AvellanedaStoikovMM" in sc_results:
                mm_std = sc_results["MarketMakerAgent"]
                mm_as = sc_results["AvellanedaStoikovMM"]

                sc_results["comparison"] = {
                    "pnl_vs_baseline_advantage_as": round(mm_as["mm_pnl_vs_baseline"] - mm_std["mm_pnl_vs_baseline"], 4),
                    "inventory_reduction_as": mm_std["mm_max_abs_inventory"] - mm_as["mm_max_abs_inventory"],
                    "adverse_selection_reduction_as": round(mm_std["adverse_selection_loss"] - mm_as["adverse_selection_loss"], 4),
                    "drawdown_reduction_as": round(mm_std["max_drawdown_mid"] - mm_as["max_drawdown_mid"], 4),
                    "more_resilient_agent": (
                        "AvellanedaStoikovMM"
                        if mm_as["mm_pnl_vs_baseline"] >= mm_std["mm_pnl_vs_baseline"]
                        else "MarketMakerAgent"
                    ),
                }

            report_scenarios[sc_name] = sc_results

        # Complete stress report structure
        report = {
            "run_id": run_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "config": {
                "scenarios": scenario_names,
                "mm_types": evaluated_mm_types,
                "episodes": self.episodes,
                "cycles": self.cycles,
                "seed": self.seed,
            },
            "scenarios": report_scenarios,
        }

        # Persist report to data/runs/<run_id>/stress_report.json
        run_dir = os.path.join(self.output_dir, run_id)
        os.makedirs(run_dir, exist_ok=True)
        report_path = os.path.join(run_dir, "stress_report.json")
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)

        return report


def load_stress_report(run_id: str, output_dir: str = "data/runs") -> Dict[str, Any]:
    """Load a previously saved stress report JSON."""
    report_path = os.path.join(output_dir, run_id, "stress_report.json")
    if not os.path.exists(report_path):
        raise FileNotFoundError(f"Stress report not found at {report_path}")
    with open(report_path, "r") as f:
        return json.load(f)
