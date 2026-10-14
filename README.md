# MultiAgentMarketSim
Multi-Agent Market Simulation platform in which independent, self-interested reinforcement-learning (RL) agents trade against one another inside a simulated limit-order-book (CLOB) exchange.

---

## Project Status & Roadmap Completion

| Phase | Milestone / Deliverable | Status |
|---|---|---|
| **Phase 1: Core Engine** (Weeks 1–4) | Dual priority-queue CLOB (`OrderBook`), price-time priority `MatchingEngine`, simulation invariant validators (`validate_no_cross`, `validate_zero_sum`), and PettingZoo `ParallelEnv` wrapper (`MarketEnvironment`). | **100% COMPLETE** |
| **Phase 2: Agent Logic** (Weeks 5–8) | Baseline Zero-Intelligence (ZI) agents, Market-Maker (MM) quoting agents, and RL directional agents (Ray RLlib / Gymnasium). | Up next |
| **Phase 3: Game Theory** (Weeks 9–11) | Payoff matrix construction from logged runs, Nash equilibrium computation (Nashpy / Lemke-Howson), best-response dynamics. | Planned |
| **Phase 4: Dashboard & Viva** (Week 12) | Streamlit & Plotly interactive visualizer, replay analytics, and final report. | Planned |

---

## What Was Implemented / Changed

### 1. PettingZoo Multi-Agent Environment (`src/env/market_env.py`)
Completed `MarketEnvironment` (inheriting from PettingZoo's `ParallelEnv`) wrapping the `OrderBook` and `MatchingEngine`:
- **Observation Space**: Top-$K$ bid/ask prices and volumes from `OrderBook.depth(levels=K)`, along with the agent's individual cash balance and share inventory.
- **Action Space**: Supports order type (`0: HOLD`, `1: LIMIT`, `2: MARKET`), side (`0: BID`, `1: ASK`), price offset relative to reference/mid price, and order quantity. Supports both structured `gymnasium.spaces.Dict` and flat `gymnasium.spaces.Box` formats.
- **Order Tracking**: Internal `_all_orders` dictionary tracks every submitted order so filled and cancelled orders can always be mapped back to their originating agent for trade settlements.
- **Invariant Validation**: Enforces `validate_no_cross` and `validate_zero_sum` after every single step; raises `SimulationInvalidError` immediately if an invariant is violated.
- **PnL-based Mark-to-Market Rewards**: Step rewards correspond to changes in total portfolio value ($Cash + Inventory \times MidPrice$).

### 2. Comprehensive Conformance & Invariant Tests (`tests/test_env.py`)
Implemented full PettingZoo compliance tests, reproducibility tests, zero-sum conservation tests, and order-matching edge case tests.

### 3. Repository Configuration (`.gitignore`)
Added standard Python and OS ignore rules to exclude `venv/`, `__pycache__/`, `.pytest_cache/`, and system files from git commits.

---

## The 9 Test Cases Explained

All 9 tests pass cleanly across `tests/test_engine.py` and `tests/test_env.py`:

```
tests/test_engine.py::TestMatchingEngine::test_crossing_orders_match PASSED
tests/test_engine.py::TestMatchingEngine::test_submit_valid_limit_order PASSED
tests/test_env.py::TestMarketEnvironment::test_crossing_orders_and_no_cross_invariant PASSED
tests/test_env.py::TestMarketEnvironment::test_market_orders PASSED
tests/test_env.py::TestMarketEnvironment::test_parallel_api PASSED
tests/test_env.py::TestMarketEnvironment::test_parallel_api_flat_box PASSED
tests/test_env.py::TestMarketEnvironment::test_same_seed_explicit_identical_results PASSED
tests/test_env.py::TestMarketEnvironment::test_seed_reproducibility_pettingzoo_tool PASSED
tests/test_env.py::TestMarketEnvironment::test_zero_sum_conservation PASSED
============================== 9 passed in 0.50s ==============================
```

### Matching Engine Unit Tests (`tests/test_engine.py`)
1. **`test_submit_valid_limit_order`**:
   - Submits a valid limit buy order to an empty book.
   - Verifies 0 trades are generated and the order rests at the top of the bid heap with correct price and quantity.
2. **`test_crossing_orders_match`**:
   - Submits a resting bid order at $100.00 followed by an incoming ask at $99.00.
   - Verifies immediate trade matching at the resting order's price ($100.00) under price-time priority, with correct partial fill statuses.

### PettingZoo Environment Integration Tests (`tests/test_env.py`)
3. **`test_parallel_api`**:
   - Runs PettingZoo's official `parallel_api_test(env, num_cycles=200)` suite.
   - Verifies observation/action space immutability, agent dictionary keys across `step()` and `reset()`, live agent filtering, and termination/truncation compliance.
4. **`test_parallel_api_flat_box`**:
   - Validates that the environment also conforms to `parallel_api_test` when initialized with flat Box observation and action spaces (`dict_spaces=False`).
5. **`test_seed_reproducibility_pettingzoo_tool`**:
   - Uses PettingZoo's official `parallel_seed_test` suite to verify environment determinism across independent runs.
6. **`test_same_seed_explicit_identical_results`**:
   - Instantiates two separate environments with the identical random seed (`seed=42`).
   - Steps them forward for 50 cycles with identical seeded action samples.
   - Asserts bit-for-bit equivalence across all observations, rewards, terminations, truncations, infos, internal cash balances, and order book depth.
7. **`test_zero_sum_conservation`**:
   - Simulates 50 random trading steps.
   - Verifies after every step that total cash is strictly conserved ($\sum Cash_i = InitialTotalCash - Fees$) and total share inventory across all agents is conserved ($\sum Inventory_i = InitialTotalInventory$).
8. **`test_crossing_orders_and_no_cross_invariant`**:
   - Simultaneously submits crossing orders (e.g., Bid @ $102 vs. Ask @ $98).
   - Verifies trades execute immediately, balances update, and the remaining order book satisfies `best_bid < best_ask` without crossed spreads.
9. **`test_market_orders`**:
   - Places a resting limit sell order and executes an aggressive market buy order against it.
   - Verifies the market order consumes liquidity at the resting ask price, transfers shares, updates cash balances, and clears the book.

---

## How to Run the Tests

To run the entire test suite:

```bash
pytest -v
# or
python -m pytest -v
```

To run only the environment tests:

```bash
pytest tests/test_env.py -v
```
