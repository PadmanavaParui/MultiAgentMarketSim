"""
Econometric Validation of Market Microstructure Stylized Facts.

Evaluates simulation telemetry (timeseries.json) against five canonical empirical
stylized facts observed across real-world financial markets (Cont, 2001; Bouchaud & Potters, 2003):

1. Heavy Tails (Leptokurtosis):
   Unconditional return distributions display fat tails with excess kurtosis > 0
   (raw kurtosis > 3.0), far exceeding Gaussian distribution limits.

2. Heavy Tail Exponent (Hill Estimator):
   The tail index alpha of extreme returns follows a power-law P(|R| > x) ~ x^(-alpha)
   with the tail index universally falling within 2.0 <= alpha <= 5.0 (the 'inverse cubic law'
   and finite-variance heavy-tail regime).

3. Absence of Linear Autocorrelations:
   Asset returns exhibit no significant linear autocorrelation for lags >= 1:
   |ACF(k)| < 0.10 for k in {1, 2, 3, 4, 5}, consistent with the Efficient Market Hypothesis
   once bid-ask bounce is eliminated by resampling at fixed time intervals.

4. Volatility Clustering:
   While returns are linearly uncorrelated, squared returns r_t^2 (and absolute returns |r_t|)
   exhibit positive, slowly decaying autocorrelation: ACF(r^2, lag=1) > 0 and
   mean(ACF(r^2, lags=1..5)) > 0, decaying over higher lags (the ARCH/GARCH effect).

5. Spread-Volatility Correlation:
   Bid-ask spread correlates positively with instantaneous price volatility (corr(spread, |r|) > 0),
   reflecting market makers widening quotes to offset heightened inventory risk and adverse selection.

Resampling Methodology:
To eliminate high-frequency microstructural noise such as bid-ask bounce (Roll, 1984),
mid prices and spreads are resampled at fixed step intervals (default: every 5 steps).
Returns are calculated strictly intra-episode (r_t = ln(P_t / P_{t-1})) to avoid artificial
discontinuities across episode boundaries.

Reporting Policy:
Failures are reported honestly and transparently; the simulation dynamics are not tuned
to force synthetic passes.
"""

from __future__ import annotations

import os
import json
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import numpy as np


# ======================================================================
# 1. Resampling & Intra-Episode Return Extraction
# ======================================================================

def resample_timeseries(
    timeseries_records: List[Dict[str, Any]],
    interval: int = 5,
    fallback_price: float = 100.0,
) -> Dict[str, np.ndarray]:
    """
    Resample mid prices and spreads at fixed intervals to avoid bid-ask bounce artifacts.

    Computes intra-episode log returns: r_t = ln(P_t / P_{t-1}).

    Parameters
    ----------
    timeseries_records : List[Dict[str, Any]]
        Records loaded from timeseries.json, each containing 'episode', 'step', 'mid', 'spread'.
    interval : int
        Step interval for resampling (default: 5 steps).
    fallback_price : float
        Reference price used if leading mid prices are missing (default: 100.0).

    Returns
    -------
    Dict[str, np.ndarray]
        Contains 'returns' (float64 array of log returns) and
        'spreads' (float64 array of corresponding spreads).
    """
    if not timeseries_records:
        return {"returns": np.empty(0, dtype=np.float64), "spreads": np.empty(0, dtype=np.float64)}

    # Group records by episode to prevent cross-episode return contamination
    episodes: Dict[int, List[Dict[str, Any]]] = {}
    for r in timeseries_records:
        ep = r.get("episode", 1)
        if ep not in episodes:
            episodes[ep] = []
        episodes[ep].append(r)

    all_returns: List[float] = []
    all_spreads: List[float] = []

    for ep, records in episodes.items():
        # Sort chronologically by step
        records_sorted = sorted(records, key=lambda x: x.get("step", 0))

        # Forward fill missing mid prices and spreads
        last_mid = fallback_price
        last_spread = 1.0
        cleaned_records: List[Tuple[int, float, float]] = []

        for r in records_sorted:
            step = int(r.get("step", 0))
            mid = r.get("mid")
            spread = r.get("spread")

            if mid is not None and not np.isnan(mid) and float(mid) > 0:
                last_mid = float(mid)
            if spread is not None and not np.isnan(spread) and float(spread) >= 0:
                last_spread = float(spread)

            cleaned_records.append((step, last_mid, last_spread))

        if not cleaned_records:
            continue

        # Sample at fixed intervals
        if interval <= 1:
            sampled = cleaned_records
        else:
            # Pick records where step % interval == 0; if none match, sample every interval-th element
            matching = [cr for cr in cleaned_records if cr[0] % interval == 0]
            sampled = matching if len(matching) >= 2 else cleaned_records[::interval]

        if len(sampled) < 2:
            continue

        # Compute log returns intra-episode
        for i in range(1, len(sampled)):
            prev_p = sampled[i - 1][1]
            curr_p = sampled[i][1]
            curr_sp = sampled[i][2]

            if prev_p > 0 and curr_p > 0:
                ret = float(np.log(curr_p / prev_p))
                all_returns.append(ret)
                all_spreads.append(curr_sp)

    return {
        "returns": np.asarray(all_returns, dtype=np.float64),
        "spreads": np.asarray(all_spreads, dtype=np.float64),
    }


# ======================================================================
# 2. Econometric Estimators
# ======================================================================

def compute_excess_kurtosis(returns: Sequence[float]) -> Dict[str, float]:
    """
    Compute sample excess kurtosis and raw kurtosis of returns.

    Excess Kurtosis = mu_4 / (sigma^2)^2 - 3.0
    where a standard Gaussian distribution has excess kurtosis = 0.0 (raw kurtosis = 3.0).

    Returns
    -------
    Dict[str, float]
        Keys: 'excess_kurtosis', 'raw_kurtosis'.
    """
    arr = np.asarray(returns, dtype=np.float64)
    if len(arr) < 4:
        return {"excess_kurtosis": 0.0, "raw_kurtosis": 3.0}

    mean = float(np.mean(arr))
    var = float(np.mean((arr - mean) ** 2))
    if var < 1e-12:
        return {"excess_kurtosis": 0.0, "raw_kurtosis": 3.0}

    m4 = float(np.mean((arr - mean) ** 4))
    raw_kurt = m4 / (var ** 2)
    excess_kurt = raw_kurt - 3.0

    return {
        "excess_kurtosis": round(float(excess_kurt), 4),
        "raw_kurtosis": round(float(raw_kurt), 4),
    }


def compute_hill_tail_index(
    returns: Sequence[float],
    tail_fraction: float = 0.10,
) -> Dict[str, Any]:
    """
    Compute the Hill tail index estimator (Hill, 1975) for extreme returns.

    Models the distribution tail as a Pareto-type power law: P(|R| > x) ~ x^(-alpha).
    Let absolute returns be ordered descending: x_(1) >= x_(2) >= ... >= x_(k) > x_(k+1).
    Hill shape parameter:
        gamma_hat = (1 / k) * sum_{i=1}^k ln(x_(i) / x_(k+1))
    Tail index:
        alpha_hat = 1 / gamma_hat

    In empirical finance, alpha universally falls within [2.0, 5.0] (the 'inverse cubic law').

    Parameters
    ----------
    returns : Sequence[float]
        Return series.
    tail_fraction : float
        Fraction of upper-tail observations used in the Hill estimator (default: 0.10 = 10%).

    Returns
    -------
    Dict[str, Any]
        Keys: 'tail_index', 'k', 'tail_fraction', 'gamma'.
    """
    arr = np.abs(np.asarray(returns, dtype=np.float64))
    arr = arr[arr > 1e-9]

    if len(arr) < 10:
        return {
            "tail_index": None,
            "k": 0,
            "tail_fraction": tail_fraction,
            "gamma": None,
        }

    # Sort in descending order
    arr_sorted = np.sort(arr)[::-1]
    n = len(arr_sorted)
    k = max(2, min(n - 1, int(np.floor(tail_fraction * n))))

    x_tail = arr_sorted[:k]
    x_cutoff = arr_sorted[k]  # x_(k+1)

    if x_cutoff <= 0:
        return {
            "tail_index": None,
            "k": k,
            "tail_fraction": tail_fraction,
            "gamma": None,
        }

    log_ratios = np.log(x_tail / x_cutoff)
    gamma = float(np.mean(log_ratios))

    if gamma <= 1e-12:
        tail_index = float("inf")
    else:
        tail_index = float(1.0 / gamma)

    return {
        "tail_index": round(tail_index, 4) if not np.isinf(tail_index) else "inf",
        "k": k,
        "tail_fraction": tail_fraction,
        "gamma": round(gamma, 6),
    }


def compute_autocorrelation(
    series: Sequence[float],
    max_lag: int = 10,
) -> Dict[int, float]:
    """
    Compute sample autocorrelation function (ACF) for lags 1 through max_lag.

    ACF(k) = sum_{t=k+1}^T (y_t - y_bar)(y_{t-k} - y_bar) / sum_{t=1}^T (y_t - y_bar)^2

    Parameters
    ----------
    series : Sequence[float]
        Data series.
    max_lag : int
        Maximum lag horizon (default: 10).

    Returns
    -------
    Dict[int, float]
        Mapping from lag (1..max_lag) to autocorrelation coefficient.
    """
    arr = np.asarray(series, dtype=np.float64)
    n = len(arr)

    if n <= max_lag:
        return {lag: 0.0 for lag in range(1, max_lag + 1)}

    mean = float(np.mean(arr))
    var = float(np.sum((arr - mean) ** 2))

    if var < 1e-12:
        return {lag: 0.0 for lag in range(1, max_lag + 1)}

    acfs: Dict[int, float] = {}
    for lag in range(1, max_lag + 1):
        cov = float(np.sum((arr[lag:] - mean) * (arr[:-lag] - mean)))
        acfs[lag] = round(float(cov / var), 4)

    return acfs


def compute_spread_volatility_correlation(
    spreads: Sequence[float],
    returns: Sequence[float],
) -> float:
    """
    Compute Pearson correlation between bid-ask spread and return volatility |r_t|.

    Stylized Fact: corr(spread, volatility) > 0.

    Parameters
    ----------
    spreads : Sequence[float]
        Bid-ask spreads at resampled steps.
    returns : Sequence[float]
        Returns at resampled steps.

    Returns
    -------
    float
        Pearson correlation coefficient in [-1, 1]. Returns 0.0 if either series has zero variance.
    """
    sp_arr = np.asarray(spreads, dtype=np.float64)
    vol_arr = np.abs(np.asarray(returns, dtype=np.float64))

    if len(sp_arr) != len(vol_arr) or len(sp_arr) < 3:
        return 0.0

    sp_std = float(np.std(sp_arr))
    vol_std = float(np.std(vol_arr))

    if sp_std < 1e-12 or vol_std < 1e-12:
        return 0.0

    corr = float(np.corrcoef(sp_arr, vol_arr)[0, 1])
    return round(float(corr), 4)


# ======================================================================
# 3. Stylized Facts Validation Suite
# ======================================================================

def compute_stylized_facts(
    timeseries_records: List[Dict[str, Any]],
    resample_interval: int = 5,
    kurtosis_threshold: float = 3.0,
    hill_bounds: Tuple[float, float] = (2.0, 5.0),
    max_return_acf_threshold: float = 0.10,
) -> Dict[str, Any]:
    """
    Compute all five stylized facts from timeseries records and evaluate against thresholds.

    Threshold Specifications:
    -------------------------
    1. Excess Kurtosis (Leptokurtosis):
       Raw kurtosis > 3.0 (equivalently, excess kurtosis > 0.0).
       If kurtosis_threshold is set to 3.0, checks raw_kurtosis > 3.0 (excess_kurtosis > 0.0).

    2. Hill Tail Index:
       Checks 2.0 <= alpha <= 5.0.
       In empirical finance, asset return tails decay with a power-law exponent between 2 and 5.
       Values > 5 indicate Gaussian-like thin tails; values < 2 indicate infinite variance.

    3. Return Autocorrelation:
       Checks max(|ACF(1)| .. |ACF(5)|) < max_return_acf_threshold (default: 0.10).
       Asset returns show negligible linear autocorrelation beyond high-frequency bounce.

    4. Volatility Clustering:
       Squared returns r_t^2 exhibit positive and persistent autocorrelation.
       Checks: ACF(r^2, lag=1) > 0 and mean(ACF(r^2, lags=1..5)) > 0, with slow decay
       (ACF(r^2, lag=10) <= ACF(r^2, lag=1)).

    5. Spread vs. Volatility Correlation:
       Checks corr(spread, |r|) > 0.0.
       Market makers widen quotes during volatile periods to offset risk.

    Parameters
    ----------
    timeseries_records : List[Dict[str, Any]]
        Records from timeseries.json.
    resample_interval : int
        Step downsampling interval (default: 5).
    kurtosis_threshold : float
        Threshold for raw kurtosis (default: 3.0).
    hill_bounds : Tuple[float, float]
        Acceptable bounds for Hill tail exponent alpha (default: (2.0, 5.0)).
    max_return_acf_threshold : float
        Max acceptable magnitude for return autocorrelation at lags 1-5 (default: 0.10).

    Returns
    -------
    Dict[str, Any]
        Dictionary with structured results for each stylized fact, raw metrics, and pass/fail status.
    """
    resampled = resample_timeseries(timeseries_records, interval=resample_interval)
    returns = resampled["returns"]
    spreads = resampled["spreads"]

    num_obs = len(returns)

    # 1. Excess Kurtosis
    kurt_res = compute_excess_kurtosis(returns)
    raw_kurt = kurt_res["raw_kurtosis"]
    excess_kurt = kurt_res["excess_kurtosis"]
    kurt_passed = bool(raw_kurt > kurtosis_threshold)

    # 2. Hill Tail Index
    hill_res = compute_hill_tail_index(returns, tail_fraction=0.10)
    alpha = hill_res["tail_index"]
    if alpha is not None and isinstance(alpha, (int, float)) and not np.isnan(alpha) and not np.isinf(alpha):
        hill_passed = bool(hill_bounds[0] <= alpha <= hill_bounds[1])
    else:
        hill_passed = False

    # 3. Return Autocorrelation (lags 1-5)
    ret_acf = compute_autocorrelation(returns, max_lag=5)
    max_abs_ret_acf = max([abs(v) for v in ret_acf.values()]) if ret_acf else 0.0
    ret_acf_passed = bool(max_abs_ret_acf < max_return_acf_threshold and num_obs >= 10)

    # 4. Volatility Clustering (squared returns lags 1-10)
    sq_returns = returns ** 2
    sq_acf = compute_autocorrelation(sq_returns, max_lag=10)
    lag1_sq_acf = sq_acf.get(1, 0.0)
    lags1_5_sq_acf = [sq_acf.get(i, 0.0) for i in range(1, 6)]
    mean_sq_acf_1_5 = float(np.mean(lags1_5_sq_acf)) if lags1_5_sq_acf else 0.0
    lag10_sq_acf = sq_acf.get(10, 0.0)

    # Volatility clustering requires positive lag 1, positive short-horizon mean, and decaying trend
    vol_clustering_passed = bool(
        lag1_sq_acf > 0.0
        and mean_sq_acf_1_5 > 0.0
        and lag10_sq_acf <= lag1_sq_acf
        and num_obs >= 15
    )

    # 5. Spread vs. Volatility Correlation
    spread_vol_corr = compute_spread_volatility_correlation(spreads, returns)
    spread_vol_passed = bool(spread_vol_corr > 0.0 and num_obs >= 5)

    checks = {
        "excess_kurtosis": {
            "passed": kurt_passed,
            "raw_kurtosis": raw_kurt,
            "excess_kurtosis": excess_kurt,
            "threshold": kurtosis_threshold,
            "criterion": f"raw_kurtosis > {kurtosis_threshold} (excess_kurtosis > 0.0)",
            "description": (
                "Heavy Tails (Leptokurtosis): Unconditional returns exhibit heavy tails "
                "with excess kurtosis > 0 (kurtosis > 3), exceeding the Gaussian normal distribution."
            ),
        },
        "hill_tail_index": {
            "passed": hill_passed,
            "tail_index": alpha,
            "k_tail_obs": hill_res["k"],
            "tail_fraction": hill_res["tail_fraction"],
            "bounds": list(hill_bounds),
            "criterion": f"{hill_bounds[0]} <= tail_index <= {hill_bounds[1]}",
            "description": (
                "Hill Tail Index: Power-law tail exponent alpha falls within [2.0, 5.0], "
                "characterizing the heavy-tailed, finite-variance regime universally observed in empirical finance."
            ),
        },
        "return_autocorrelation": {
            "passed": ret_acf_passed,
            "lags": {f"lag_{k}": v for k, v in ret_acf.items()},
            "max_abs_acf": round(max_abs_ret_acf, 4),
            "threshold": max_return_acf_threshold,
            "criterion": f"max(|ACF(1..5)|) < {max_return_acf_threshold}",
            "description": (
                "Absence of Linear Autocorrelations: Linear autocorrelation of returns decays "
                "rapidly to near zero (|ACF| < 0.10 for lags 1-5), consistent with the Efficient Market Hypothesis."
            ),
        },
        "volatility_clustering": {
            "passed": vol_clustering_passed,
            "lags": {f"lag_{k}": v for k, v in sq_acf.items()},
            "lag_1": round(lag1_sq_acf, 4),
            "mean_lags_1_5": round(mean_sq_acf_1_5, 4),
            "lag_10": round(lag10_sq_acf, 4),
            "criterion": "ACF(r^2, lag=1) > 0 and mean(ACF(r^2, lags=1..5)) > 0 with slow decay (lag_10 <= lag_1)",
            "description": (
                "Volatility Clustering: While returns are linearly uncorrelated, squared returns "
                "exhibit positive, slowly decaying autocorrelation across lags 1-10 (ARCH effect)."
            ),
        },
        "spread_volatility_correlation": {
            "passed": spread_vol_passed,
            "correlation": spread_vol_corr,
            "threshold": 0.0,
            "criterion": "corr(spread, |returns|) > 0.0",
            "description": (
                "Spread-Volatility Correlation: Market makers widen bid-ask spreads during "
                "periods of elevated volatility to manage inventory risk and adverse selection."
            ),
        },
    }

    total_checks = len(checks)
    passed_checks = sum(1 for c in checks.values() if c["passed"])
    overall_passed = bool(passed_checks == total_checks)

    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "resample_interval": resample_interval,
        "num_observations": num_obs,
        "overall_passed": overall_passed,
        "summary": {
            "total_checks": total_checks,
            "passed_checks": passed_checks,
            "failed_checks": total_checks - passed_checks,
        },
        "stylized_facts": checks,
    }


def validate_stylized_facts(
    run_id: Optional[str] = None,
    data_dir: str = "data/runs",
    timeseries_data: Optional[List[Dict[str, Any]]] = None,
    resample_interval: int = 5,
    save_report: bool = True,
    output_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Validate stylized facts for a simulation run and optionally persist validation_report.json.

    Parameters
    ----------
    run_id : str | None
        Identifier of the simulation run. If provided, loads timeseries from data_dir/run_id/timeseries.json.
    data_dir : str
        Directory containing run outputs (default: data/runs).
    timeseries_data : List[Dict[str, Any]] | None
        In-memory timeseries records. If provided, overrides reading from disk.
    resample_interval : int
        Step interval for resampling mid prices (default: 5).
    save_report : bool
        If True, saves results as validation_report.json (default: True).
    output_path : str | None
        Custom file path to save validation_report.json. Defaults to data_dir/run_id/validation_report.json.

    Returns
    -------
    Dict[str, Any]
        Validation report dictionary with full metrics, thresholds, and pass/fail results.
    """
    if timeseries_data is not None:
        records = timeseries_data
    elif run_id is not None:
        ts_path = os.path.join(data_dir, run_id, "timeseries.json")
        if not os.path.exists(ts_path):
            raise FileNotFoundError(f"Timeseries data not found at {ts_path}")
        with open(ts_path, "r") as f:
            records = json.load(f)
    else:
        raise ValueError("Either run_id or timeseries_data must be provided.")

    report = compute_stylized_facts(
        timeseries_records=records,
        resample_interval=resample_interval,
    )
    report["run_id"] = run_id or "in_memory_run"

    if save_report:
        if output_path is not None:
            save_file = output_path
        elif run_id is not None:
            save_file = os.path.join(data_dir, run_id, "validation_report.json")
        else:
            save_file = os.path.join(data_dir, "validation_report.json")

        os.makedirs(os.path.dirname(os.path.abspath(save_file)), exist_ok=True)
        with open(save_file, "w") as f:
            json.dump(report, f, indent=2)

    return report


def load_validation_report(run_id: str, data_dir: str = "data/runs") -> Dict[str, Any]:
    """Load an existing validation_report.json for a given run ID."""
    report_path = os.path.join(data_dir, run_id, "validation_report.json")
    if not os.path.exists(report_path):
        raise FileNotFoundError(f"Validation report not found at {report_path}")
    with open(report_path, "r") as f:
        return json.load(f)
