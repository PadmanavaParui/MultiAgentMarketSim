"""
CLI Runner for Econometric Stylized Facts Validation.

Validates telemetry (timeseries.json) from any past simulation run against
the 5 canonical empirical microstructure stylized facts:
  1. Heavy Tails (Excess Kurtosis > 0 / Kurtosis > 3)
  2. Hill Tail Index (2.0 <= alpha <= 5.0)
  3. Absence of Linear Autocorrelation (|ACF(1..5)| < 0.10)
  4. Volatility Clustering (ACF(r^2) positive and slowly decaying)
  5. Spread-Volatility Correlation (corr(spread, |r|) > 0)

Usage:
    python scripts/run_validation.py --run-id demo_run_001 --interval 5
"""

from __future__ import annotations

import os
import sys
import argparse

# Ensure project root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.validation.stylized_facts import validate_stylized_facts


def format_validation_scorecard(report: dict) -> str:
    """Format stylized facts validation report as an ASCII table."""
    summary = report.get("summary", {})
    facts = report.get("stylized_facts", {})

    status_str = "PASSED" if report.get("overall_passed") else "FAILED (Honest Report)"

    lines = [
        f"\n==================================================================",
        f"  ECONOMETRIC STYLIZED FACTS VALIDATION REPORT",
        f"  Run ID: {report.get('run_id')} | Observations: {report.get('num_observations')} | Interval: {report.get('resample_interval')} steps",
        f"  Overall Status: {status_str} ({summary.get('passed_checks', 0)}/{summary.get('total_checks', 0)} Passed)",
        f"==================================================================",
        f"{'Stylized Fact':<30} | {'Status':<8} | {'Empirical Value':<20} | {'Criterion / Threshold'}",
        f"-------------------------------+----------+----------------------+-----------------------------",
    ]

    # 1. Excess Kurtosis
    kurt = facts.get("excess_kurtosis", {})
    k_stat = "PASS" if kurt.get("passed") else "FAIL"
    k_val = f"raw={kurt.get('raw_kurtosis'):.2f}, exc={kurt.get('excess_kurtosis'):.2f}"
    lines.append(f"{'1. Excess Kurtosis':<30} | {k_stat:<8} | {k_val:<20} | {kurt.get('criterion', '')}")

    # 2. Hill Tail Index
    hill = facts.get("hill_tail_index", {})
    h_stat = "PASS" if hill.get("passed") else "FAIL"
    h_idx = hill.get("tail_index")
    h_val = f"alpha={h_idx:.2f}" if isinstance(h_idx, (int, float)) else f"alpha={h_idx}"
    lines.append(f"{'2. Hill Tail Index':<30} | {h_stat:<8} | {h_val:<20} | {hill.get('criterion', '')}")

    # 3. Return Autocorrelation
    acf_ret = facts.get("return_autocorrelation", {})
    r_stat = "PASS" if acf_ret.get("passed") else "FAIL"
    r_val = f"max|ACF|={acf_ret.get('max_abs_acf'):.4f}"
    lines.append(f"{'3. Return Autocorrelation':<30} | {r_stat:<8} | {r_val:<20} | {acf_ret.get('criterion', '')}")

    # 4. Volatility Clustering
    vol = facts.get("volatility_clustering", {})
    v_stat = "PASS" if vol.get("passed") else "FAIL"
    v_val = f"lag1={vol.get('lag_1'):.2f}, mean5={vol.get('mean_lags_1_5'):.2f}"
    lines.append(f"{'4. Volatility Clustering':<30} | {v_stat:<8} | {v_val:<20} | {vol.get('criterion', '')}")

    # 5. Spread vs Volatility Correlation
    sp_vol = facts.get("spread_volatility_correlation", {})
    s_stat = "PASS" if sp_vol.get("passed") else "FAIL"
    s_val = f"rho={sp_vol.get('correlation'):.4f}"
    lines.append(f"{'5. Spread-Vol Correlation':<30} | {s_stat:<8} | {s_val:<20} | {sp_vol.get('criterion', '')}")

    lines.append(f"==================================================================")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="MultiAgentMarketSim Stylized Facts Validator")
    parser.add_argument(
        "--run-id",
        type=str,
        required=True,
        help="Run identifier whose timeseries.json to validate",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/runs",
        help="Directory containing run outputs (default: data/runs)",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=5,
        help="Step resampling interval to eliminate bid-ask bounce (default: 5)",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default=None,
        help="Custom path to save validation_report.json",
    )

    args = parser.parse_args()

    try:
        report = validate_stylized_facts(
            run_id=args.run_id,
            data_dir=args.data_dir,
            resample_interval=args.interval,
            save_report=True,
            output_path=args.output_path,
        )
        print(format_validation_scorecard(report))
        out_dest = args.output_path or os.path.join(args.data_dir, args.run_id, "validation_report.json")
        print(f"[+] Validation report successfully written to: {out_dest}\n")
    except Exception as e:
        print(f"[-] Validation failed: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
