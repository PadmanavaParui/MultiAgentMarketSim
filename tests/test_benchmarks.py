"""
Unit and integration tests for scripts/benchmark.py.
Verifies steps/sec measurement, CLOB orders/sec benchmark, determinism check, and report generation.
"""

import os
import tempfile
import unittest

from scripts.benchmark import (
    benchmark_engine_orders,
    benchmark_environment_steps,
    check_determinism,
    generate_benchmark_markdown,
    get_system_info,
    run_all_benchmarks,
    write_benchmark_report,
)


class TestBenchmarks(unittest.TestCase):
    def test_get_system_info(self):
        """Verifies system info returns OS, Python, and timestamp."""
        info = get_system_info()
        self.assertIn("os", info)
        self.assertIn("python_version", info)
        self.assertIn("timestamp_utc", info)
        self.assertTrue(len(info["python_version"]) > 0)

    def test_benchmark_environment_steps(self):
        """Tests that environment steps benchmark measures throughput on 4 agents."""
        res = benchmark_environment_steps(n_agents=4, steps=100, seed=42, max_cycles=50)
        self.assertEqual(res["n_agents"], 4)
        self.assertEqual(res["completed_steps"], 100)
        self.assertGreater(res["steps_per_sec"], 0)
        self.assertGreater(res["ms_per_step"], 0)
        self.assertGreaterEqual(res["trades_executed"], 0)

    def test_benchmark_engine_orders(self):
        """Tests raw CLOB matching engine benchmark with synthetic orders."""
        res = benchmark_engine_orders(num_orders=2000, seed=42)
        self.assertEqual(res["num_orders"], 2000)
        self.assertGreater(res["orders_per_sec"], 0)
        self.assertGreater(res["us_per_order"], 0)
        self.assertGreater(res["trades_executed"], 0)

    def test_check_determinism_identical_and_distinct(self):
        """
        Verifies bit-for-bit determinism:
        Same seed yields identical trade hash, different seed yields different trade hash.
        """
        det = check_determinism(seed=42, steps=100, n_agents=4)
        self.assertTrue(det["passed"])
        self.assertTrue(det["identical_seed_match"])
        self.assertTrue(det["control_seed_distinct"])
        self.assertEqual(det["trial_1_hash"], det["trial_2_hash"])
        self.assertNotEqual(det["trial_1_hash"], det["control_hash"])
        self.assertEqual(len(det["trial_1_hash"]), 64)  # Valid SHA-256 hex string

    def test_report_generation_and_write(self):
        """Tests that markdown benchmark report is generated and successfully written."""
        results = run_all_benchmarks(
            agent_counts=(4,),
            env_steps=50,
            engine_orders=1000,
            seed=42,
        )
        md = generate_benchmark_markdown(results)
        self.assertIn("# MultiAgentMarketSim — System Benchmarks & Verification", md)
        self.assertIn("Cryptographic Determinism Verification", md)
        self.assertIn("PASS", md)

        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = os.path.join(tmpdir, "BENCHMARKS_TEST.md")
            written = write_benchmark_report(results, output_path=out_path)
            self.assertEqual(written, out_path)
            self.assertTrue(os.path.exists(out_path))
            with open(out_path, "r", encoding="utf-8") as f:
                content = f.read()
            self.assertIn("MultiAgentMarketSim", content)


if __name__ == "__main__":
    unittest.main()
