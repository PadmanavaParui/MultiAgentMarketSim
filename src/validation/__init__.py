"""
Validation module for MultiAgentMarketSim.
Includes econometric stylized facts testing and reporting.
"""

from src.validation.stylized_facts import (
    resample_timeseries,
    compute_excess_kurtosis,
    compute_hill_tail_index,
    compute_autocorrelation,
    compute_spread_volatility_correlation,
    compute_stylized_facts,
    validate_stylized_facts,
    load_validation_report,
)

__all__ = [
    "resample_timeseries",
    "compute_excess_kurtosis",
    "compute_hill_tail_index",
    "compute_autocorrelation",
    "compute_spread_volatility_correlation",
    "compute_stylized_facts",
    "validate_stylized_facts",
    "load_validation_report",
]
