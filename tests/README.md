# tests/

Unit tests for the modules in this repo. Run them with:

```bash
.venv\Scripts\python -m pytest tests
```

Tests that touch ArcticDB open a throw-away LMDB store under pytest's `tmp_path` with `?map_size=20MB`. Without that, each store preallocates GBs of disk on Windows. They never touch the real store.

| File | Covers |
|---|---|
| `conftest.py` | Shared fixtures: a sample futures `Bar`, `OptionContract` and `OptionChain`. |
| `unit/test_black76.py` | Placeholder only (one trivial test); Black-76 is exercised through the IV-solver and enricher tests. |
| `unit/test_iv_solver.py` | IV round-trips (price → IV → price) for ATM/OTM calls and puts, high vol, short DTE, index scale; edge cases (zero/negative price, zero time). |
| `unit/test_chain_enricher.py` | `ChainEnricher`: IV recovered for every contract, Greek signs (delta, gamma, theta, vega), ATM delta ≈ 0.5, synthetic spread, zero-price contracts skipped, chain IV summary. |
| `unit/test_engine.py` | `CommissionModel` (buy cost, STT on sells), `ExecutionSimulator` (fill emitted, slippage direction, commission in fill), `RiskManager` (approve, drawdown reject) and `BacktestEngine` on synthetic snapshots. |
| `unit/test_slim_layout.py` | Store writes drop copied columns; reads join Kite spot back. Only Kite-stamped `spot` symbols are used: Kite `minute` beats an Upstox `1minute`, and a ticker with only Upstox spot gets `spot_missing`. |
| `unit/test_fetch_ticker_history.py` | `fetch_ticker --history`: intraday bars floored and kept in session, daily bars normalised, F1/F2/F3 slots roll after expiry with `cycle_id` counting F1 changes, borrow formulas match a hand calculation. |
| `unit/test_feature_builder_columns.py` | Feature list contents and the `feat_flow_intensity` definition (net roll flow / total OI, not scaled by days to expiry). |
| `unit/test_feature_builder_lookahead.py` | `feat_roll_deviation` / `feat_b12_percentile` use earlier cycles only (no look-ahead). |
| `unit/test_borrow_continuous.py` | Roll-stitched and constant-maturity continuous borrow series. |
