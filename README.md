# MultiAgentMarketSim

Multi-Agent Market Simulation platform in which independent, self-interested reinforcement-learning (RL) agents trade against one another inside a simulated central limit order book (CLOB) exchange.

[![Live Demo](https://img.shields.io/badge/Live%20Demo-Streamlit%20Cloud-FF4B4B?style=for-the-badge&logo=streamlit)](https://multiagentmarketsim.streamlit.app)
[![Tests Passing](https://img.shields.io/badge/Tests-88%20Passing-2ea44f?style=for-the-badge&logo=pytest)](https://github.com/PadmanavaParui/MultiAgentMarketSim/tree/main/tests)
[![Python 3.11+](https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python)](https://python.org)
[![PettingZoo Conformance](https://img.shields.io/badge/PettingZoo-ParallelEnv-blue?style=for-the-badge)](https://pettingzoo.farama.org)
[![Benchmarks](https://img.shields.io/badge/Engine-500k%20orders%2Fsec-purple?style=for-the-badge)](docs/BENCHMARKS.md)

> 🚀 **Live Demo:** Explore the interactive dashboard, Bookmap replay, and Nash equilibrium analysis at **[https://multiagentmarketsim.streamlit.app](https://multiagentmarketsim.streamlit.app)**.

---

## ⚡ One-Command Quickstart

Clone the repository and launch the full interactive Streamlit dashboard in a single command:

```bash
git clone https://github.com/PadmanavaParui/MultiAgentMarketSim.git && cd MultiAgentMarketSim && pip install -r requirements.txt && streamlit run src/dashboard/app.py
```

*Or using [uv](https://github.com/astral-sh/uv):*
```bash
git clone https://github.com/PadmanavaParui/MultiAgentMarketSim.git && cd MultiAgentMarketSim && uv run --with-requirements requirements.txt streamlit run src/dashboard/app.py
```

---

## 🖼️ Visual Highlights & Previews

| 📼 Bookmap Depth Heatmap & Replay | ♟️ Game Theory & Nash Equilibria |
| :---: | :---: |
| ![Bookmap Depth Heatmap & Replay](docs/assets/replay_heatmap_preview.png)<br><sub>*Bookmap-style resting depth heatmap ($x=\text{step}, y=\text{price}$) with trade dots and shock markers*</sub> | ![Nash Equilibrium Heatmap](docs/assets/nash_heatmap_preview.png)<br><sub>*Head-to-head empirical payoff matrix with Nash equilibria & standard errors*</sub> |

| 📊 Microstructure & Depth Ladder | 📉 Econometric Stylized Facts |
| :---: | :---: |
| ![Live CLOB Ladder](docs/assets/clob_depth_preview.png)<br><sub>*Interactive dual priority-queue bid/ask depth ladder with spread sanity checks*</sub> | ![Validation Stylized Facts](docs/assets/stylized_facts_preview.png)<br><sub>*Fat tails, Hill tail index, volatility clustering, and spread-vol correlation*</sub> |

---

## ⚠️ Limitations & Modeling Assumptions

While `MultiAgentMarketSim` provides high throughput, strict price-time priority matching, and bit-for-bit determinism, users and researchers should be aware of the following engineering and modeling boundaries:

1. **Tabular Q-Learning (No Deep Function Approximation):**
   - The `RLDirectionalAgent` implements classic Bellman Q-learning over a **coarsely discretized tabular state space** (trend, spread, inventory, and cash buckets) rather than continuous deep reinforcement learning (such as DQN, PPO, or Ray RLlib).
   - *Rationale:* Tabular representation guarantees 100% mathematical transparency, microsecond execution speeds, reproducible transitions, and zero GPU overhead, but does not generalize across unvisited continuous state regimes.
2. **Soft Position Limit (Checked at Order Submission Time):**
   - Position limits (`max_position=100` by default, configured via `--max-position`) operate as a **soft pre-trade gate enforced at order submission** in `MarketEnvironment.step()`.
   - *Mechanism:* Orders that would push an agent's net position beyond `max_position` (evaluating current inventory plus pending order side and quantity) are rejected and logged in `info['rejected_orders']`. Because fills against resting orders occur asynchronously inside the matching engine, an agent quoting both sides can briefly hold an inventory up to the limit if both legs execute before cancellation.
3. **Econometric Stylized Facts (Passes 2 of 5 Documented Facts):**
   - Empirical validation honestly reports passes against documented quantitative financial thresholds without artificial simulator tuning or curve-fitting:
     - **Passes (2/5):** Excess kurtosis / fat tails ($\kappa > 3$) and absence of linear autocorrelation in raw returns ($|ACF_k| < 0.15$ for lags 1–5).
     - **Fails (3/5):** Hill tail index estimate ($\alpha \approx 2\text{--}4$ in real markets), long-range volatility clustering (slowly decaying positive squared-return autocorrelation), and strong positive spread-volatility correlation.
     - *Design Principle:* The simulator intentionally avoids hardcoding artificial price shocks or synthetic volatility feedback to force stylized fact passes, preserving honest, authentic emergent dynamics from zero-intelligence and tabular RL agent interactions.
4. **Price Band Safeguards (Episode Reference Bands):**
   - The simulation enforces a protective price corridor (`price_band=0.10` by default, configured via `--price-band`) evaluated as $\pm 10\%$ of the episode's initial reference price. Aggressive market orders cannot sweep past the band boundary, and limit orders outside the corridor are rejected at entry.
5. **Single-Asset Order Book:**
   - The exchange simulates a single central limit order book per environment instance. Cross-asset pairs, triangular arbitrage, and ETF basket creations are not currently simulated.
6. **Discrete Synchronous Steps:**
   - Time moves forward in discrete environment cycles ($t = 1, 2, \dots$) where all agents submit actions synchronously rather than continuous event-driven nanosecond tick time.
7. **Endogenous Liquidity-Based Market Impact:**
   - Price impact emerges purely through the depletion of resting orders in the CLOB queues; exogenous statistical impact functions (such as Almgren-Chriss or square-root laws) are not artificially superimposed.
8. **Zero-Jitter Idealized Network:**
   - Orders reach the matching engine immediately in deterministic priority order without simulated TCP packet drops or exchange gateway queuing delay.

---

## Project Status & Roadmap Completion

| Phase | Milestone / Deliverable | Status |
|---|---|---|
| **Phase 1: Core Engine** (Weeks 1–4) | Dual priority-queue CLOB (`OrderBook`), price-time priority `MatchingEngine`, simulation invariant validators (`validate_no_cross`, `validate_zero_sum`), and PettingZoo `ParallelEnv` wrapper (`MarketEnvironment`). | **100% COMPLETE** |
| **Phase 2: Agent Logic** (Weeks 5–8) | Baseline Zero-Intelligence (ZI) agents, Market-Maker (MM) quoting agents, Avellaneda-Stoikov MM, Informed traders, Shock agents, tabular Q-learning RL directional agents, PnL/Sharpe metrics, and repository logger. | **100% COMPLETE** |
| **Phase 3: Game Theory** (Weeks 9–11) | Empirical payoff matrix builder, Nash equilibrium computation via Nashpy (Lemke-Howson & Support Enumeration), and Best-Response Dynamics (Fictitious Play). | **100% COMPLETE** |
| **Phase 4: Dashboard & Analytics** (Week 12) | Streamlit & Plotly interactive visualizer, Bookmap-style depth replay, stress testing scenarios, econometric stylized facts validation, and run reproducibility. | **100% COMPLETE** |
| **Phase 5: Verification & Benchmarks** | Comprehensive benchmarking suite measuring steps/sec (4, 8, 16 agents), engine orders/sec, and cryptographic trade SHA-256 determinism checks. | **100% COMPLETE** |

---

## System Architecture

```
+-------------------------------------------------------------------+
|                        Agent Archetypes                           |
|  - ZeroIntelligenceAgent (random order flow baseline)             |
|  - MarketMakerAgent (two-sided quotes, inventory skewing)         |
|  - AvellanedaStoikovMM (optimal inventory reservation pricing)    |
|  - InformedTrader (oracle price-jump aggressive flow)             |
|  - ShockAgent (scripted flash-crash / squeeze liquidity drain)   |
|  - RLDirectionalAgent (Bellman tabular Q-learning trader)         |
+---------------------------------+---------------------------------+
                                  | actions (Limit / Market orders)
                                  v
+-------------------------------------------------------------------+
|               PettingZoo ParallelEnv (MarketEnvironment)          |
|  - Top-K depth observation spaces (Dict & flat Box)               |
|  - Order routing & deterministic execution                        |
|  - Per-step invariant validators (validate_no_cross, zero-sum)    |
+---------------------------------+---------------------------------+
                                  |
                                  v
+-------------------------------------------------------------------+
|            Matching Engine & Central Limit Order Book             |
|  - Dual priority-queue order book (O(log n) insert, O(1) best)    |
|  - Strict price-time priority (FIFO matching)                     |
|  - >500,000 orders/sec engine throughput (docs/BENCHMARKS.md)    |
+---------------------------------+---------------------------------+
                                  |
                                  v
+-------------------------------------------------------------------+
|                  Metrics & Trace Logger (Repository)              |
|  - Appends Order, Trade, and AgentMetric records                  |
|  - Persists runs to data/runs/<run_id>/ (JSON / Parquet-ready)    |
|  - Bit-for-bit SHA-256 deterministic reproducibility check        |
+---------------------------------+---------------------------------+
                                  |
                                  v
+-------------------------------------------------------------------+
|                   Post-Hoc Game-Theoretic Layer                   |
|  - PayoffBuilder: empirical normal-form matrices (A, B)           |
|  - NashSolver: Nashpy Lemke-Howson & Support Enumeration          |
|  - BestResponseDynamics: strategy trajectory & stability trace   |
+---------------------------------+---------------------------------+
                                  |
                                  v
+-------------------------------------------------------------------+
|                Streamlit + Plotly Interactive Dashboard           |
|  - Live & Replay simulation visualization with Bookmap heatmap    |
|  - Order book depth ladders with spread sanity check marker       |
|  - Stress scenario analysis (flash crash, informed, withdrawal)   |
|  - Econometric stylized facts (kurtosis, Hill index, clustering)  |
|  - Game-theoretic payoff tables & Nash equilibrium profiles       |
+-------------------------------------------------------------------+
```

---

## 📊 Empirical Benchmarks

Full benchmark numbers and hardware specifications are documented in **[docs/BENCHMARKS.md](docs/BENCHMARKS.md)**:

- **CLOB Engine Throughput:** `~500,000+ orders / second` (~$1.99\,\mu\text{s}$ per order match)
- **Multi-Agent Simulation Speed:**
  - `4 Agents:` **~3,200 steps / sec** (0.31 ms / step)
  - `8 Agents:` **~980 steps / sec** (1.02 ms / step)
  - `16 Agents:` **~285 steps / sec** (3.50 ms / step)
- **Cryptographic Determinism (BR-009 / TEST-007):** Re-running identical seeds produces bit-for-bit identical trade logs validated via SHA-256 hashing.

---

## Test Cases & Verification (88 Tests Passing)

All 88 tests pass cleanly across 11 test suites:

```bash
pytest -v
# ======================== 88 passed, 1 warning in 5.41s =========================
```

### Test Suite Breakdown
- **`tests/test_engine.py`**: CLOB dual priority queues, FIFO matching, price-time priority, limit & market order fills.
- **`tests/test_env.py`**: PettingZoo `ParallelEnv` API conformance, seed reproducibility, zero-sum cash conservation, and no-cross spread invariance.
- **`tests/test_agents.py`**: Zero-Intelligence, Market Maker inventory skewing, and tabular Q-learning `RLDirectionalAgent` with step and per-episode epsilon decay.
- **`tests/test_realistic_agents.py`**: Avellaneda-Stoikov inventory skewing, InformedTrader jump anticipation, ShockAgent flash crash execution.
- **`tests/test_stress.py`**: Flash crash, informed order flow, and liquidity withdrawal stress testing scenarios.
- **`tests/test_stylized_facts.py`**: Excess kurtosis, Hill tail index, linear autocorrelation decay, squared-return clustering, and spread-volatility correlation.
- **`tests/test_gametheory.py`**: Normal-form payoff matrix construction, Nash equilibrium solvers (Prisoner's Dilemma & Matching Pennies), and Fictitious Play best-response dynamics.
- **`tests/test_tournament.py`**: Multi-seed head-to-head strategy tournaments with bootstrap standard error estimation.
- **`tests/test_telemetry.py`**: Book snapshots, depth levels, and time series logging.
- **`tests/test_dashboard.py`**: Plotly chart builders, Bookmap heatmap, live ladder, training curve, and modular tabs.
- **`tests/test_benchmarks.py`**: Steps/sec measurement, CLOB orders/sec benchmark, and trade SHA-256 determinism checks.

---

## Quickstart & Usage

### 1. Running the Test Suite
```bash
pytest -v
# or using uv:
uv run --with-requirements requirements.txt python3 -m pytest -v
```

### 2. Running Simulation Modes
```bash
# Baseline agents only (Zero-Intelligence + Market Maker)
python scripts/run_simulation.py --mode baselines --episodes 5 --cycles 30 --seed 42

# RL directional trader vs. frozen baselines (TEST-006)
python scripts/run_simulation.py --mode rl_vs_baseline --episodes 10 --cycles 50 --seed 42

# Multiple concurrent RL agents
python scripts/run_simulation.py --mode multi_rl --episodes 10 --cycles 50 --seed 42
```

### 3. Running Stress Scenarios
```bash
# Run flash-crash, informed-flow, and liquidity-withdrawal scenarios
python scripts/run_stress.py --seed 42 --episodes 5 --cycles 100
```

### 4. Running Benchmarks
```bash
# Run performance and determinism benchmarks
python scripts/benchmark.py --output docs/BENCHMARKS.md
```

### 5. Launching the Interactive Dashboard
```bash
streamlit run src/dashboard/app.py
```

The visualizer opens at `http://localhost:8501`, featuring 8 dedicated tabs:
1. **Overview**: Executive summary, run selector, and quick metrics.
2. **Microstructure & Depth**: Dual-priority bid/ask depth ladders with spread sanity checks.
3. **Order Book Replay**: Bookmap-style depth heatmap, scrubber playback, trade tape feed, and flash-crash markers.
4. **Agent Performance**: Comparative PnL curves, inventory accumulation, and Sharpe ratios.
5. **Stress Scenarios**: Baseline vs. stressed mid-price, peak spreads, and adverse selection metrics.
6. **Econometric Validation**: Stylized facts scorecard (kurtosis, Hill index, volatility clustering, QQ plot).
7. **Game Theory & Nash**: Head-to-head empirical payoff heatmaps and Nash equilibria.
8. **Convergence & Telemetry**: RL policy training curves, epsilon decay schedules, and reproducibility snapshots.
