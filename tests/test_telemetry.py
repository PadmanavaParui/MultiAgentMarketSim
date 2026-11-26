"""
Unit tests for book snapshots and telemetry time-series logging.

Verifies:
  1. Schema conformance for book_snapshots.json (best bid, best ask, mid, spread, top-5 depth, agent state).
  2. Schema conformance for timeseries.json (telemetry prices, spread, agent inventories and PnL).
  3. Downsampling and file size sanity checks (bounded size, expected sample counts per episode).
  4. Configurable downsampling interval K (default 5).
  5. Determinism and reproducibility across identical random seeds.
  6. Backward compatibility with older runs that lack book_snapshots.json and timeseries.json.
"""

from __future__ import annotations

import os
import json
import shutil
import unittest

from src.metrics.logger import MetricsLogger, get_book_snapshots, get_timeseries
from scripts.run_simulation import run_simulation


class TestTelemetryLogging(unittest.TestCase):
    def setUp(self):
        self.test_run_id = "test_telemetry_eval"
        self.custom_k_run_id = "test_telemetry_custom_k"
        self.output_dir = "data/runs"

    def tearDown(self):
        for run_id in [self.test_run_id, self.custom_k_run_id, "test_det_run_1", "test_det_run_2"]:
            p = os.path.join(self.output_dir, run_id)
            if os.path.exists(p):
                shutil.rmtree(p, ignore_errors=True)

    def test_schema_book_snapshots(self):
        """
        Verify book_snapshots.json exists and strictly adheres to the required schema:
        best bid, best ask, mid, spread, top-5 depth per side, and each agent's inventory & PnL.
        """
        run_simulation(
            mode="rl_vs_baseline",
            episodes=2,
            cycles_per_episode=15,
            snapshot_interval=5,
            seed=42,
            run_id=self.test_run_id,
            output_dir=self.output_dir,
            verbose=False,
        )

        logger = MetricsLogger(base_dir=self.output_dir)
        snapshots = logger.get_book_snapshots(self.test_run_id)

        self.assertIsInstance(snapshots, list)
        self.assertGreater(len(snapshots), 0)

        for s in snapshots:
            # Required top-level keys
            self.assertIn("episode", s)
            self.assertIn("step", s)
            self.assertIn("best_bid", s)
            self.assertIn("best_ask", s)
            self.assertIn("mid", s)
            self.assertIn("spread", s)
            self.assertIn("bids", s)
            self.assertIn("asks", s)
            self.assertIn("depth", s)
            self.assertIn("agents", s)

            self.assertIsInstance(s["episode"], int)
            self.assertIsInstance(s["step"], int)
            self.assertEqual(s["step"] % 5, 0)

            # Check prices
            if s["best_bid"] is not None:
                self.assertIsInstance(s["best_bid"], (float, int))
            if s["best_ask"] is not None:
                self.assertIsInstance(s["best_ask"], (float, int))

            if s["best_bid"] is not None and s["best_ask"] is not None:
                self.assertLessEqual(s["best_bid"], s["best_ask"])
                self.assertIsNotNone(s["spread"])
                self.assertIsNotNone(s["mid"])
                self.assertAlmostEqual(s["spread"], s["best_ask"] - s["best_bid"], places=3)
                self.assertAlmostEqual(s["mid"], (s["best_bid"] + s["best_ask"]) / 2.0, places=3)

            # Check top-5 depth
            self.assertIsInstance(s["bids"], list)
            self.assertIsInstance(s["asks"], list)
            self.assertLessEqual(len(s["bids"]), 5)
            self.assertLessEqual(len(s["asks"]), 5)

            for p, q in s["bids"]:
                self.assertGreater(p, 0.0)
                self.assertGreater(q, 0)

            for p, q in s["asks"]:
                self.assertGreater(p, 0.0)
                self.assertGreater(q, 0)

            # Check agent inventory and PnL
            self.assertIsInstance(s["agents"], dict)
            for agent_id, data in s["agents"].items():
                self.assertIn("inventory", data)
                self.assertIn("pnl", data)
                self.assertIsInstance(data["inventory"], int)
                self.assertIsInstance(data["pnl"], (float, int))

    def test_schema_timeseries(self):
        """
        Verify timeseries.json exists and adheres to the telemetry schema:
        episode, step, best bid, best ask, mid, spread, and agent states.
        """
        run_simulation(
            mode="rl_vs_baseline",
            episodes=2,
            cycles_per_episode=15,
            snapshot_interval=5,
            seed=42,
            run_id=self.test_run_id,
            output_dir=self.output_dir,
            verbose=False,
        )

        logger = MetricsLogger(base_dir=self.output_dir)
        ts = logger.get_timeseries(self.test_run_id)

        self.assertIsInstance(ts, list)
        self.assertGreater(len(ts), 0)

        for entry in ts:
            self.assertIn("episode", entry)
            self.assertIn("step", entry)
            self.assertIn("best_bid", entry)
            self.assertIn("best_ask", entry)
            self.assertIn("mid", entry)
            self.assertIn("spread", entry)
            self.assertIn("agents", entry)
            self.assertIn("inventories", entry)
            self.assertIn("pnls", entry)

            self.assertEqual(entry["step"] % 5, 0)

            for a_id in ["rl_0", "mm_0", "zi_0"]:
                self.assertIn(a_id, entry["agents"])
                self.assertIn("inventory", entry["agents"][a_id])
                self.assertIn("pnl", entry["agents"][a_id])
                self.assertIsInstance(entry["agents"][a_id]["inventory"], int)
                self.assertIsInstance(entry["agents"][a_id]["pnl"], (float, int))
                self.assertEqual(entry["inventories"][a_id], entry["agents"][a_id]["inventory"])
                self.assertEqual(entry["pnls"][a_id], entry["agents"][a_id]["pnl"])

    def test_file_size_sanity_and_downsampling(self):
        """
        Tests file size sanity and verifies that saving every K steps downsamples the records correctly.
        """
        episodes = 2
        cycles = 25
        k = 5
        run_simulation(
            mode="rl_vs_baseline",
            episodes=episodes,
            cycles_per_episode=cycles,
            snapshot_interval=k,
            seed=100,
            run_id=self.test_run_id,
            output_dir=self.output_dir,
            verbose=False,
        )

        run_path = os.path.join(self.output_dir, self.test_run_id)
        book_file = os.path.join(run_path, "book_snapshots.json")
        ts_file = os.path.join(run_path, "timeseries.json")

        self.assertTrue(os.path.exists(book_file))
        self.assertTrue(os.path.exists(ts_file))

        # Size sanity: files must be non-empty and reasonably sized (under 100 KB for small runs)
        book_size = os.path.getsize(book_file)
        ts_size = os.path.getsize(ts_file)

        self.assertGreater(book_size, 0)
        self.assertLess(book_size, 100_000)
        self.assertGreater(ts_size, 0)
        self.assertLess(ts_size, 100_000)

        with open(book_file) as f:
            book_data = json.load(f)
        with open(ts_file) as f:
            ts_data = json.load(f)

        # Expected record count: episodes * (cycles // k)
        expected_records = episodes * (cycles // k)
        self.assertEqual(len(book_data), expected_records)
        self.assertEqual(len(ts_data), expected_records)

        # Verify steps sampled
        expected_steps = [5, 10, 15, 20, 25] * episodes
        actual_steps = [r["step"] for r in book_data]
        self.assertEqual(actual_steps, expected_steps)

    def test_custom_k_downsampling(self):
        """
        Verify that configuring a custom K (e.g., K=3) accurately downsamples every K steps.
        """
        run_simulation(
            mode="baselines",
            episodes=1,
            cycles_per_episode=12,
            snapshot_interval=3,
            seed=77,
            run_id=self.custom_k_run_id,
            output_dir=self.output_dir,
            verbose=False,
        )

        logger = MetricsLogger(base_dir=self.output_dir)
        snapshots = logger.get_book_snapshots(self.custom_k_run_id)

        # 12 cycles // 3 = 4 snapshots: steps [3, 6, 9, 12]
        self.assertEqual(len(snapshots), 4)
        self.assertEqual([s["step"] for s in snapshots], [3, 6, 9, 12])

    def test_seeded_determinism(self):
        """
        Verify that runs with identical seeds produce bit-for-bit identical telemetry files.
        """
        run_simulation(
            mode="rl_vs_baseline",
            episodes=2,
            cycles_per_episode=15,
            snapshot_interval=5,
            seed=888,
            run_id="test_det_run_1",
            output_dir=self.output_dir,
            verbose=False,
        )
        run_simulation(
            mode="rl_vs_baseline",
            episodes=2,
            cycles_per_episode=15,
            snapshot_interval=5,
            seed=888,
            run_id="test_det_run_2",
            output_dir=self.output_dir,
            verbose=False,
        )

        logger = MetricsLogger(base_dir=self.output_dir)
        snap1 = logger.get_book_snapshots("test_det_run_1")
        snap2 = logger.get_book_snapshots("test_det_run_2")
        self.assertEqual(snap1, snap2)

        ts1 = logger.get_timeseries("test_det_run_1")
        ts2 = logger.get_timeseries("test_det_run_2")
        self.assertEqual(ts1, ts2)

    def test_backward_compatibility_old_runs(self):
        """
        Verify that old runs lacking book_snapshots.json and timeseries.json still load cleanly.
        """
        logger = MetricsLogger(base_dir=self.output_dir)

        # demo_gt_run is an existing run created before book_snapshots existed
        self.assertIn("demo_gt_run", logger.list_runs())

        summary = logger.get_run_summary("demo_gt_run")
        self.assertIsInstance(summary, dict)
        self.assertEqual(summary["run_id"], "demo_gt_run")

        prices = logger.get_price_series("demo_gt_run")
        self.assertIsInstance(prices, list)
        self.assertGreater(len(prices), 0)

        metrics = logger.get_agent_metrics("demo_gt_run")
        self.assertIsInstance(metrics, list)
        self.assertGreater(len(metrics), 0)

        # Calling new query methods on old runs safely returns empty list without error
        old_snapshots = logger.get_book_snapshots("demo_gt_run")
        self.assertEqual(old_snapshots, [])

        old_timeseries = logger.get_timeseries("demo_gt_run")
        self.assertEqual(old_timeseries, [])

        # Module-level convenience functions also return empty list
        self.assertEqual(get_book_snapshots("demo_gt_run"), [])
        self.assertEqual(get_timeseries("demo_gt_run"), [])


if __name__ == "__main__":
    unittest.main()
