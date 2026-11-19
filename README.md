# MultiAgentMarketSim
Multi-Agent Market Simulation platform in which independent, self-interested reinforcement-learning (RL) agents trade against one another inside a simulated limit-order-book (CLOB) exchange.

---

## Project Status & Roadmap Completion

| Phase | Milestone / Deliverable | Status |
|---|---|---|
| **Phase 1: Core Engine** (Weeks 1–4) | Dual priority-queue CLOB (`OrderBook`), price-time priority `MatchingEngine`, simulation invariant validators (`validate_no_cross`, `validate_zero_sum`), and PettingZoo `ParallelEnv` wrapper (`MarketEnvironment`). | **100% COMPLETE** |
| **Phase 2: Agent Logic** (Weeks 5–8) | Baseline Zero-Intelligence (ZI) agents, Market-Maker (MM) quoting agents, tabular Q-learning RL directional agents, PnL/Sharpe metric calculations, and repository logger. | **100% COMPLETE** |
| **Phase 3: Game Theory** (Weeks 9–11) | Empirical payoff matrix builder, Nash equilibrium computation via Nashpy (Lemke-Howson & Support Enumeration), and Best-Response Dynamics (Fictitious Play). | **100% COMPLETE** |
| **Phase 4: Dashboard & Viva** (Week 12) | Streamlit & Plotly interactive visualizer, replay analytics, order book ladders, convergence diagnostics, and run reproducibility. | **100% COMPLETE** |

---

## System Architecture

```
+-------------------------------------------------------------------+
|                        Agent Archetypes                           |
|  - ZeroIntelligenceAgent (random order flow baseline)             |
|  - MarketMakerAgent (two-sided quotes, inventory skewing)         |
|  - RLDirectionalAgent (Bellman Q-learning directional trader)     |
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
+---------------------------------+---------------------------------+
                                  |
                                  v
+-------------------------------------------------------------------+
|                  Metrics & Trace Logger (Repository)              |
|  - Appends Order, Trade, and AgentMetric records                  |
|  - Persists runs to data/runs/<run_id>/ (JSON / Parquet-ready)    |
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
|  - Live & Replay simulation visualization                         |
|  - Order book depth ladders with spread sanity check marker       |
|  - Price & Volume time series / Agent PnL & Inventory curves      |
|  - Game-theoretic payoff tables & Nash equilibrium profiles       |
+-------------------------------------------------------------------+
```

---

## Test Cases & Verification (23 Tests Passing)

All 23 tests pass cleanly across `tests/test_engine.py`, `tests/test_env.py`, `tests/test_agents.py`, `tests/test_gametheory.py`, and `tests/test_dashboard.py`:

```
tests/test_agents.py::TestAgents::test_agent_environment_integration_step PASSED
tests/test_agents.py::TestAgents::test_market_maker_agent PASSED
tests/test_agents.py::TestAgents::test_rl_directional_agent_learning PASSED
tests/test_agents.py::TestAgents::test_test_006_rl_vs_frozen_baseline PASSED
tests/test_agents.py::TestAgents::test_zero_intelligence_agent PASSED
tests/test_dashboard.py::TestDashboard::test_app_module_imports PASSED
tests/test_dashboard.py::TestDashboard::test_empty_chart_states PASSED
tests/test_dashboard.py::TestDashboard::test_test_010_chart_builders_render_completed_run PASSED
tests/test_engine.py::TestMatchingEngine::test_crossing_orders_match PASSED
tests/test_engine.py::TestMatchingEngine::test_fifo_within_same_price_level PASSED
tests/test_engine.py::TestMatchingEngine::test_submit_valid_limit_order PASSED
tests/test_env.py::TestMarketEnvironment::test_crossing_orders_and_no_cross_invariant PASSED
tests/test_env.py::TestMarketEnvironment::test_market_orders PASSED
tests/test_env.py::TestMarketEnvironment::test_parallel_api PASSED
tests/test_env.py::TestMarketEnvironment::test_parallel_api_flat_box PASSED
tests/test_env.py::TestMarketEnvironment::test_same_seed_explicit_identical_results PASSED
tests/test_env.py::TestMarketEnvironment::test_seed_reproducibility_pettingzoo_tool PASSED
tests/test_env.py::TestMarketEnvironment::test_zero_sum_conservation PASSED
tests/test_gametheory.py::TestGameTheory::test_best_response_dynamics PASSED
tests/test_gametheory.py::TestGameTheory::test_payoff_builder_archetypes PASSED
tests/test_gametheory.py::TestGameTheory::test_payoff_builder_from_metrics PASSED
tests/test_gametheory.py::TestGameTheory::test_test_008_nash_solver_matching_pennies PASSED
tests/test_gametheory.py::TestGameTheory::test_test_008_nash_solver_prisoners_dilemma PASSED
============================= 23 passed in 1.75s ==============================
```

### Key Conformance & OJT Tests
- **TEST-001 to TEST-005**: Matching engine price-time priority, FIFO ordering, limit & market order execution, partial fills, zero-sum conservation, and strict no-cross invariant validation.
- **TEST-006**: RL directional trader trained against frozen Zero-Intelligence baseline demonstrating positive Q-learning updates and PnL adaptation.
- **TEST-007 / BR-009**: Seeded reproducibility verifying bit-for-bit identical trade sequence and portfolio values when running with the same seed.
- **TEST-008**: Nashpy equilibrium solver validation on known analytical games (Prisoner's Dilemma pure Nash equilibrium and Matching Pennies mixed Nash equilibrium).
- **TEST-009**: Official PettingZoo `parallel_api_test` and `parallel_seed_test` full conformance.
- **TEST-010**: Streamlit and Plotly dashboard rendering all charts (depth ladders, price/volume series, agent metrics, and game theory candidate tables).

---

## Quickstart & Usage

### 1. Running the Test Suite
```bash
pytest -v
# or using uv:
uv run --with-requirements requirements.txt python3 -m pytest -v
```

### 2. Running Simulations (Curriculum Modes)
Execute multi-agent simulations across different training stages:

```bash
# Stage 1: Baseline agents only (Zero-Intelligence + Market Maker)
python scripts/run_simulation.py --mode baselines --episodes 5 --cycles 30 --seed 42

# Stage 2: RL directional trader vs. frozen baselines (TEST-006)
python scripts/run_simulation.py --mode rl_vs_baseline --episodes 10 --cycles 50 --seed 42

# Stage 3: Multiple concurrent RL agents
python scripts/run_simulation.py --mode multi_rl --episodes 10 --cycles 50 --seed 42
```

All runs automatically compute post-hoc game theory metrics and save complete datasets into `data/runs/<run_id>/`.

### 3. Launching the Interactive Dashboard
```bash
streamlit run src/dashboard/app.py
# or using uv:
uv run --with-requirements requirements.txt streamlit run src/dashboard/app.py
```

The dashboard opens in your browser at `http://localhost:8501`, featuring:
- **Interactive Run Launcher & Inspector**: Select from saved runs or launch new experiments.
- **Market & Price Microstructure**: Price and volume time-series alongside the live order book depth ladder.
- **Agent Performance**: Comparative PnL curves (RL vs baselines), inventory accumulation, and Sharpe ratio summary table.
- **Game Theory & Nash Equilibria**: Payoff matrix display, Nash equilibrium candidate strategies, and Fictitious Play best-response trajectory.
- **Convergence Diagnostics**: RL reward trajectories and loss stability indicators.
- **Reproducibility Verification**: Configuration snapshots and seed replay assurance.
