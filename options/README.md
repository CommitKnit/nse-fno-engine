# options/

Option-specific logic: turning raw candles into enriched chains, pricing, execution costs, and the strategies the backtests run.

## `options/chain/`

| File | What it does |
|---|---|
| `build_enriched_chains.py` | The enrich + clean core used by `scripts/fetch_options.py`, see below. |
| `chain_enricher.py` | `ChainEnricher`, see below. |

### `build_enriched_chains.py`

This is a library with no CLI. `run_stage2_enrich(raw_df, …)` runs these steps:
1. `pre_solver_filter`
2. fractional trading-day DTE
3. build `OptionChain`s per (timestamp, expiry)
4. one-pass Black-76 IV + Greeks (`ChainEnricher.enrich_dataframe`)
5. the `in_last_{N}_td` flag
6. IV cleanup → arbitrage flags → delta band

`_write_enriched_to_arctic` upserts the result into `options_enriched/{T}/{iv}/{expiry}`. It uses `update()`, so re-runs overwrite rather than duplicate.

The old Hive-parquet medallion pipeline (raw/enriched/surfaces parquet layers, the IV-surface stage, and its `main`) was removed.

### `chain_enricher.py`

`ChainEnricher` attaches IV, delta/gamma/theta/vega, moneyness class and synthetic bid/ask to chains, on the futures price (Black-76).
- `enrich_dataframe` handles a whole frame in one pass. It classifies each row as:
  - traded,
  - put-call-parity repaired (the other leg traded), or
  - forward-filled (both legs stale; `is_synthetic`).
- `enrich(chain)` handles a single chain.

## `options/pricing/`

| File | What it does |
|---|---|
| `black76.py` | Black-76 price and Greeks for options on futures, scalar and vectorised (`black76_price`, `black76_greeks`, `*_vec`). No spot is involved: the future already carries the cost of carry. |
| `iv_solver.py` | Implied vol from a Black-76 price: Newton (`iv_newton`), with Brent (`iv_brent`) as fallback, behind `solve_iv`; `solve_iv_chain` solves a whole chain. |
| `dte_utils.py` | See below. |

### `dte_utils.py`

The single source of truth for the NSE trading calendar and time-to-expiry:
- `is_trading_day`, `trading_days_inclusive`, `nth_trading_day_before`.
- Interval arithmetic: `interval_to_minutes`, `candles_per_day` (375-minute session).
- The intraday-decrementing `fractional_dte` and `year_fraction`.
- The `in_last_N_td` window flag helpers.

## `options/execution/`

`slippage_model.py` contains `SlippageModel`. It estimates a synthetic bid-ask spread for options from moneyness, OI, volume and DTE, because Upstox history has no quotes.

## `options/strategies/`

All three subclass `core/strategy/base_strategy.py::BaseStrategy`. Each takes its entry/exit timestamps from the runner (no signal logic of its own) and records per-trade attribution for the CSVs.

| File | What it does |
|---|---|
| `roll_spread_put_hedge.py` | See below. Run by `backtests/run_roll_put_hedge.py`. |
| `timed_long_options.py` | `TimedLongOptionsStrategy`. Buys a set of legs (`("CE"\|"PE", "atm"\|"otm")`) on a chosen chain at a scheduled bar and sells them at a scheduled bar, unhedged. Run by `backtests/run_trough_peak_options.py`. |
| `expiry_straddle.py` | `ExpiryStraddleStrategy`. Long one lot each of the F1 ATM call and put from a scheduled entry to a scheduled exit, unhedged. Run by `backtests/run_expiry_straddle.py`. |

### `roll_spread_put_hedge.py`

`RollSpreadPutHedgeStrategy` buys an F2 put (ATM, or the nearest OTM strike with OI ≥ `min_oi`) and delta-hedges it with the F1 future. The position is long about |Δ| × (F1 − F2), so it profits when the F1−F2 spread widens in the final expiry week.
- **Hedge:** rehedges inside a configurable |Δ| band.
- **Mark-to-market:** marks the held strike with Black-76 at its last observed IV (sticky strike).
- **P&L attribution:** splits P&L into spread / gamma / vega / theta.
