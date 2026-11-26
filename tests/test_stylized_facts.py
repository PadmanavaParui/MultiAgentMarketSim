"""
Unit and integration tests for econometric stylized facts validation.
Verifies:
  1. Pure econometric estimators on synthetic statistical distributions.
  2. Resampling logic at fixed intervals (preventing bid-ask bounce and inter-episode jumps).
  3. Honest pass/fail reporting against documented thresholds.
  4. Generation and persistence of validation_report.json.
"""

from __future__ import annotations

import os
import shutil
import tempfile
import unittest
import numpy as np

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


class TestEconometricEstimators(unittest.TestCase):
    """Unit tests for individual econometric estimators on known statistical distributions."""

    def setUp(self):
        self.rng = np.random.default_rng(42)

    def test_excess_kurtosis_gaussian_vs_student_t(self):
        """Gaussian data has excess kurtosis ~0; Student-t(df=4) has heavy tails (excess kurtosis > 3)."""
        # Gaussian distribution -> excess kurtosis ~ 0
        normal_data = self.rng.normal(loc=0.0, scale=0.02, size=10000)
        norm_res = compute_excess_kurtosis(normal_data)
        self.assertAlmostEqual(norm_res["excess_kurtosis"], 0.0, delta=0.3)
        self.assertAlmostEqual(norm_res["raw_kurtosis"], 3.0, delta=0.3)

        # Student-t with df=3 has theoretical infinite kurtosis / extreme leptokurtosis
        t_data = self.rng.standard_t(df=3, size=10000) * 0.01
        t_res = compute_excess_kurtosis(t_data)
        self.assertGreater(t_res["excess_kurtosis"], 3.0)
        self.assertGreater(t_res["raw_kurtosis"], 6.0)

        # Edge cases: constant or very small sample
        self.assertEqual(compute_excess_kurtosis([0.01, 0.01, 0.01])["excess_kurtosis"], 0.0)

    def test_hill_tail_index_heavy_tailed(self):
        """Hill estimator on Pareto / Student-t distribution recovers heavy-tail exponent alpha."""
        # Pareto distribution with alpha = 3.0
        # P(X > x) = (x_m / x)^alpha -> tail index alpha = 3.0
        pareto_samples = (self.rng.pareto(a=3.0, size=10000) + 1.0) * 0.01
        hill_res = compute_hill_tail_index(pareto_samples, tail_fraction=0.10)

        self.assertIsNotNone(hill_res["tail_index"])
        self.assertAlmostEqual(hill_res["tail_index"], 3.0, delta=0.6)
        self.assertGreater(hill_res["k"], 50)

        # Small sample edge case returns None
        small_hill = compute_hill_tail_index([0.01, 0.02, 0.03])
        self.assertIsNone(small_hill["tail_index"])

    def test_autocorrelation_returns_white_noise_vs_ar1(self):
        """White noise has zero return autocorrelation; AR(1) has significant positive lag 1."""
        # White noise
        wn = self.rng.normal(0, 1, 5000)
        acf_wn = compute_autocorrelation(wn, max_lag=5)
        for lag in range(1, 6):
            self.assertLess(abs(acf_wn[lag]), 0.06)

        # AR(1) with phi = 0.7
        ar1 = np.zeros(5000)
        for t in range(1, 5000):
            ar1[t] = 0.7 * ar1[t - 1] + wn[t]
        acf_ar1 = compute_autocorrelation(ar1, max_lag=5)
        self.assertGreater(acf_ar1[1], 0.60)

    def test_volatility_clustering_arch_vs_iid(self):
        """GARCH-like clustered volatility displays positive, slowly decaying ACF of squared returns."""
        n = 5000
        returns_clustered = np.zeros(n)
        vol = 0.01
        for t in range(n):
            # Stochastic volatility regime
            if t % 50 == 0:
                vol = self.rng.choice([0.005, 0.04])
            returns_clustered[t] = self.rng.normal(0, vol)

        sq_acf = compute_autocorrelation(returns_clustered ** 2, max_lag=10)
        self.assertGreater(sq_acf[1], 0.05)
        self.assertGreater(float(np.mean([sq_acf[k] for k in range(1, 6)])), 0.05)

        # I.i.d. Gaussian white noise has ~0 squared return autocorrelation
        iid = self.rng.normal(0, 0.01, 5000)
        sq_acf_iid = compute_autocorrelation(iid ** 2, max_lag=10)
        self.assertLess(abs(sq_acf_iid[1]), 0.05)

    def test_spread_volatility_correlation(self):
        """Spread correlated with absolute returns produces positive correlation coefficient."""
        rets = self.rng.normal(0, 0.02, 1000)
        abs_rets = np.abs(rets)
        spreads = 0.5 + 20.0 * abs_rets + self.rng.normal(0, 0.05, 1000)

        corr = compute_spread_volatility_correlation(spreads, rets)
        self.assertGreater(corr, 0.6)

        # Uncorrelated constant spread
        const_spreads = np.ones(1000) * 0.5
        corr_const = compute_spread_volatility_correlation(const_spreads, rets)
        self.assertEqual(corr_const, 0.0)


class TestResamplingLogic(unittest.TestCase):
    """Tests for fixed-interval time-series resampling and return calculation."""

    def test_resampling_fixed_intervals(self):
        """Verify sampling at step % interval == 0 and intra-episode return computation."""
        records = [
            {"episode": 1, "step": 1, "mid": 100.0, "spread": 0.5},
            {"episode": 1, "step": 2, "mid": 101.0, "spread": 0.5},
            {"episode": 1, "step": 3, "mid": 102.0, "spread": 0.5},
            {"episode": 1, "step": 4, "mid": 103.0, "spread": 0.5},
            {"episode": 1, "step": 5, "mid": 105.0, "spread": 0.6},  # Sampled
            {"episode": 1, "step": 6, "mid": 106.0, "spread": 0.6},
            {"episode": 1, "step": 10, "mid": 110.0, "spread": 0.7}, # Sampled
        ]

        res = resample_timeseries(records, interval=5)
        # 2 sampled points: 105.0 and 110.0 -> 1 return
        self.assertEqual(len(res["returns"]), 1)
        expected_ret = np.log(110.0 / 105.0)
        self.assertAlmostEqual(res["returns"][0], expected_ret, places=6)
        self.assertEqual(res["spreads"][0], 0.7)

    def test_resampling_prevents_inter_episode_jumps(self):
        """Verify returns are computed strictly intra-episode and do not jump between episodes."""
        records = [
            {"episode": 1, "step": 5, "mid": 100.0, "spread": 0.5},
            {"episode": 1, "step": 10, "mid": 102.0, "spread": 0.5},
            # Episode 2 starts fresh at 100.0
            {"episode": 2, "step": 5, "mid": 100.0, "spread": 0.5},
            {"episode": 2, "step": 10, "mid": 104.0, "spread": 0.5},
        ]

        res = resample_timeseries(records, interval=5)
        # Episode 1 has 1 return (102 / 100), Episode 2 has 1 return (104 / 100)
        self.assertEqual(len(res["returns"]), 2)
        self.assertAlmostEqual(res["returns"][0], np.log(102.0 / 100.0), places=6)
        self.assertAlmostEqual(res["returns"][1], np.log(104.0 / 100.0), places=6)

    def test_resampling_handles_missing_and_none_values(self):
        """Missing or None mid values are gracefully forward-filled."""
        records = [
            {"episode": 1, "step": 5, "mid": 100.0, "spread": 0.5},
            {"episode": 1, "step": 10, "mid": None, "spread": None},  # should ffill 100.0, 0.5
            {"episode": 1, "step": 15, "mid": 105.0, "spread": 0.8},
        ]

        res = resample_timeseries(records, interval=5)
        self.assertEqual(len(res["returns"]), 2)
        self.assertAlmostEqual(res["returns"][0], 0.0)  # 100.0 to 100.0
        self.assertAlmostEqual(res["returns"][1], np.log(105.0 / 100.0))


class TestStylizedFactsValidationAndReporting(unittest.TestCase):
    """Tests for complete stylized facts validation and validation_report.json persistence."""

    def setUp(self):
        self.test_dir = tempfile.mkdtemp(prefix="test_val_")

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_validation_report_generation_and_persistence(self):
        """Generate validation report on synthetic market telemetry and verify JSON output."""
        # Create synthetic market telemetry with realistic traits
        rng = np.random.default_rng(123)
        records = []
        price = 100.0

        for ep in range(1, 4):
            price = 100.0
            for step in range(5, 105, 5):
                ret = rng.standard_t(df=4) * 0.015
                price = max(10.0, price * np.exp(ret))
                spread = max(0.1, 0.5 + 10.0 * abs(ret) + rng.normal(0, 0.02))
                records.append({
                    "episode": ep,
                    "step": step,
                    "mid": round(price, 4),
                    "spread": round(spread, 4),
                })

        report = validate_stylized_facts(
            run_id="test_run_001",
            data_dir=self.test_dir,
            timeseries_data=records,
            resample_interval=5,
            save_report=True,
        )

        # 1. Check top-level report schema
        self.assertIn("run_id", report)
        self.assertEqual(report["run_id"], "test_run_001")
        self.assertIn("timestamp", report)
        self.assertIn("resample_interval", report)
        self.assertIn("num_observations", report)
        self.assertIn("overall_passed", report)
        self.assertIn("summary", report)
        self.assertIn("stylized_facts", report)

        summary = report["summary"]
        self.assertEqual(summary["total_checks"], 5)
        self.assertEqual(summary["passed_checks"] + summary["failed_checks"], 5)

        # 2. Check each stylized fact entry
        facts = report["stylized_facts"]
        expected_keys = [
            "excess_kurtosis",
            "hill_tail_index",
            "return_autocorrelation",
            "volatility_clustering",
            "spread_volatility_correlation",
        ]
        for key in expected_keys:
            self.assertIn(key, facts)
            fact = facts[key]
            self.assertIn("passed", fact)
            self.assertIsInstance(fact["passed"], bool)
            self.assertIn("criterion", fact)
            self.assertIn("description", fact)

        # 3. Check file was saved and can be loaded
        loaded = load_validation_report("test_run_001", data_dir=self.test_dir)
        self.assertEqual(loaded["run_id"], "test_run_001")
        self.assertEqual(loaded["summary"]["total_checks"], 5)

        # 4. Check FileNotFoundError for invalid run
        with self.assertRaises(FileNotFoundError):
            load_validation_report("non_existent_run", data_dir=self.test_dir)

    def test_honest_failure_reporting(self):
        """Verify that when a data series violates stylized facts, it is reported honestly without errors."""
        # Flat constant prices -> no kurtosis, no tail index, no volatility correlation
        records = [
            {"episode": 1, "step": i * 5, "mid": 100.0, "spread": 0.5}
            for i in range(1, 30)
        ]

        report = compute_stylized_facts(records, resample_interval=5)
        self.assertFalse(report["overall_passed"])
        self.assertFalse(report["stylized_facts"]["excess_kurtosis"]["passed"])
        self.assertFalse(report["stylized_facts"]["hill_tail_index"]["passed"])
        self.assertFalse(report["stylized_facts"]["spread_volatility_correlation"]["passed"])


if __name__ == "__main__":
    unittest.main()
