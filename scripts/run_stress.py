"""
CLI Runner for Market Microstructure Stress Testing Scenarios.

Executes paired baseline vs. stressed episodes for:
  - flash_crash
  - informed_flow
  - liquidity_withdrawal

Evaluates and compares MarketMakerAgent vs. AvellanedaStoikovMM resilience metrics:
  - Max drawdown of mid-price
  - Peak spread vs. baseline spread
  - MM peak |inventory|
  - MM PnL vs. baseline
  - Adverse-selection loss
  - Recovery time (steps)

Persists report to data/runs/<run_id>/stress_report.json.

Usage:
    python scripts/run_stress.py --episodes 5 --cycles 50 --seed 42 --run-id my_stress_run
"""

from __future__ import annotations

import os
import sys
import argparse
from datetime import datetime, timezone

# Ensure project root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.scenarios.stress import StressScenarioRunner


def format_table(scenario_name: str, results: dict) -> str:
    """Format scenario comparison as an ASCII table."""
    lines = [
        f"\n==================================================================",
        f"  SCENARIO: {scenario_name.upper()}",
        f"==================================================================",
        f"{'Metric':<32} | {'MarketMakerAgent':<16} | {'AvellanedaStoikovMM':<16}",
        f"---------------------------------+------------------+-----------------",
    ]

    mm_res = results.get("MarketMakerAgent", {})
    as_res = results.get("AvellanedaStoikovMM", {})

    metrics_display = [
        ("Max Mid Drawdown ($)", "max_drawdown_mid", "{:.2f}"),
        ("Peak Spread ($)", "peak_spread", "{:.2f}"),
        ("Baseline Spread ($)", "baseline_spread", "{:.2f}"),
        ("Spread Diff vs Base ($)", "spread_diff", "{:.2f}"),
        ("MM Max |Inventory|", "mm_max_abs_inventory", "{:d}"),
        ("MM Stressed PnL ($)", "mm_pnl_stressed", "{:.2f}"),
        ("MM Baseline PnL ($)", "mm_pnl_baseline", "{:.2f}"),
        ("MM PnL vs Baseline ($)", "mm_pnl_vs_baseline", "{:.2f}"),
        ("Adverse-Selection Loss ($)", "adverse_selection_loss", "{:.2f}"),
        ("Recovered?", "recovered", lambda v: "Yes" if v else "No"),
        ("Recovery Time (steps)", "recovery_time", lambda v: str(v) if v is not None else "N/A"),
    ]

    for label, key, fmt in metrics_display:
        v1 = mm_res.get(key)
        v2 = as_res.get(key)

        def _fmt(val):
            if val is None:
                return "N/A"
            if callable(fmt):
                return fmt(val)
            return fmt.format(val)

        s1 = _fmt(v1)
        s2 = _fmt(v2)
        lines.append(f"{label:<32} | {s1:<16} | {s2:<16}")

    comp = results.get("comparison")
    if comp:
        lines.append(f"---------------------------------+------------------+-----------------")
        lines.append(f"Advantage AS PnL Delta: {comp.get('pnl_vs_baseline_advantage_as', 0):+.2f}")
        lines.append(f"AS Inventory Reduction: {comp.get('inventory_reduction_as', 0):+d}")
        lines.append(f"AS Adverse Selection Loss Saved: {comp.get('adverse_selection_reduction_as', 0):+.2f}")
        lines.append(f"More Resilient Agent: {comp.get('more_resilient_agent', 'N/A')}")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="MultiAgentMarketSim Market Stress Testing Suite")
    parser.add_argument(
        "--scenarios",
        nargs="+",
        default=["flash_crash", "informed_flow", "liquidity_withdrawal"],
        help="Stress scenarios to execute (choices: flash_crash, informed_flow, liquidity_withdrawal, all)",
    )
    parser.add_argument(
        "--mm-types",
        nargs="+",
        default=["MarketMakerAgent", "AvellanedaStoikovMM"],
        help="Market maker agents to evaluate (choices: MarketMakerAgent, AvellanedaStoikovMM, both)",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=5,
        help="Number of paired baseline/stressed episodes per scenario (default: 5)",
    )
    parser.add_argument(
        "--cycles",
        type=int,
        default=50,
        help="Number of trading steps per episode (default: 50)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base random seed (default: 42)",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Custom run identifier (default: auto-generated timestamp)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/runs",
        help="Base directory for saving run artifacts (default: data/runs)",
    )

    args = parser.parse_args()

    run_id = args.run_id or f"stress_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"

    print(f"\n==================================================================")
    print(f"  STARTING STRESS TEST RUN: {run_id}")
    print(f"  Episodes: {args.episodes} | Cycles/Ep: {args.cycles} | Seed: {args.seed}")
    print(f"  Scenarios: {args.scenarios}")
    print(f"  MM Types: {args.mm_types}")
    print(f"==================================================================")

    runner = StressScenarioRunner(
        episodes=args.episodes,
        cycles_per_episode=args.cycles,
        seed=args.seed,
        output_dir=args.output_dir,
    )

    report = runner.run_suite(
        scenarios=args.scenarios,
        mm_types=args.mm_types,
        run_id=run_id,
    )

    # Print summary tables
    for sc_name, sc_data in report.get("scenarios", {}).items():
        print(format_table(sc_name, sc_data))

    report_path = os.path.join(args.output_dir, run_id, "stress_report.json")
    print(f"\n[+] Stress report saved to: {report_path}\n")


if __name__ == "__main__":
    main()
