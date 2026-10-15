# data/

Getting market data in (Upstox, Kite), storing it (ArcticDB), shaping it (models, cleaner), and handing it to the backtest (snapshot builder).

## `data/borrow_continuous.py`

Builds cross-cycle borrow-rate series from a joined `borrow_rates` frame. It is used by `scripts/build_borrow_continuous.py`. There are two families:
- **Roll-stitched** `b1/b2/b12`:
  - plus `cycle`, a `roll` flag on the first bar of each new F1 contract,
  - and `b1_ex_expiry`, which masks the last `NEAR_EXPIRY_DAYS` (b1 blows up as t1 → 0).
- **Unclipped constant-maturity** `b1_cm30`, `b2_cm60`, `b12_cm30_60`:
  - read off the cumulative borrow curve `B(t) = r·t − ln(F/S)`, linearly interpolated at fixed 30/60-day tenors;
  - no roll jumps by construction;
  - they need Kite spot.

## `data/fetchers/`

| File | What it does |
|---|---|
| `api_fetcher.py` | `UpstoxFetcher`, see below. |
| `base_fetcher.py` | Abstract `BaseFetcher` interface that `UpstoxFetcher` implements: expiries, contracts, expired candles, historical candles. |
| `kite_fetcher.py` | `KiteFetcher` and the daily login, see below. |
| `snapshot_builder.py` | `LocalSnapshotBuilder`, see below. |

### `api_fetcher.py`

`UpstoxFetcher` is the Upstox REST client that every Upstox call goes through.
- It enforces a rate limiter: 20/s, 230/min, 950/30min (from `settings`).
- Endpoints:
  - option and futures expiries and contracts, live and expired (`fetch_expiries`, `fetch_option_contracts`, `fetch_future_contracts`, `fetch_live_*_contracts`);
  - candles: expired instruments (`fetch_expired_candles`), v2 historical, and the v3 historical endpoint (`fetch_historical_candles_v3`; v2 no longer serves 5/15-minute bars);
  - instrument search and key resolution (`resolve_spot_key`, `resolve_underlying`).
- It returns `list[Bar]`.

### `kite_fetcher.py`

`KiteFetcher` is a rate-limited, retrying wrapper over the Kite Connect SDK.
- **Instruments:** instrument dump with a disk cache; `resolve_equity` / `resolve_equities` / `resolve_front_month`.
- **Candles:** `fetch_candles` and the chunked `fetch_candles_chunked`, which returns an IST-indexed OHLCV DataFrame.
- **Daily login (from a request token you supply by hand):**
  - `exchange_request_token(token_or_url)` calls `generate_session`, validates the session with `profile()`, and writes `KITE_ACCESS_TOKEN` into `.env`.
  - `login_main` is the `kite-login` command: with no arguments it prints the login URL; `--request-token X` exchanges the token.
  - `resolve_kite_access_token` re-reads `.env` on every run, so a token saved after a process started is still picked up.

### `snapshot_builder.py`

`LocalSnapshotBuilder` turns ArcticDB data into `list[MarketSnapshot]` for the engine. **It reads ArcticDB only.**
- Inputs:
  - the `futures/{T}/{iv}` wide panel, with Kite spot joined;
  - `options_enriched/{T}/{iv}/{expiry}` for every expiry in the F1/F2/F3 slots.
- Each chain is priced on its own expiry's futures close.
- `contract_ffill_days=N` makes each chain carry every contract's latest row from the last N days, for thin chains.
- It reads both stored chain schemas.
- Lot size comes from the option row, then the `contract_master` table, then 1.

## `data/models/`

| File | What it does |
|---|---|
| `bar.py` | `Bar`: one OHLCV(+OI) candle. `from_upstox_candle` / `from_kite_candle` parse provider rows; `to_dict` flattens a bar. |
| `option_contract.py` | `OptionContract`: one option at one bar. It holds instrument key, strike, type, expiry, lot size, OHLCV/OI, fractional DTE, moneyness, IV, Greeks and synthetic bid/ask. |
| `option_chain.py` | `OptionChain`: all strikes of one expiry at one timestamp, priced on `futures_price`. It has strike lookups, ATM helpers, and `to_dataframe()`. `spot_price` is reference only (Kite). |
| `market_snapshot.py` | `MarketSnapshot`: what a strategy sees at one bar. It holds the F1 futures bar, the Kite spot bar (or None), and the chains keyed by expiry. |

## `data/processors/`

### `data_cleaner.py`

`OptionsDataCleaner` is the vectorised option-chain cleaner. `options/chain/build_enriched_chains.py::run_stage2_enrich` uses it in this order:
1. `pre_solver_filter`: market hours, OI/DTE/price floors, intrinsic-value bound.
2. `post_solver_cleanup`: solved-IV sanity range.
3. `flag_arbitrage`: butterfly/calendar flags; nothing is removed.
4. `filter_delta_band`: keeps strikes inside a delta band.

`calculate_synthetic_forward`, `remove_arbitrage` and `repair_stale_prices` remain available for other uses.

## `data/storage/`

| File | What it does |
|---|---|
| `arctic_store.py` | `ArcticStore`, the single storage layer, see below. |
| `schemas.py` | The canonical MultiIndex option-chain schema (`symbol, expiration, strike, option_type`) and the underlying-state schema, plus `make_empty_*` and `validate_chain_df`. It is imported by `OptionChain` and the store. |

### `arctic_store.py`

`ArcticStore` handles the libraries, symbol naming, and typed read/write for each of them:

| Library | Symbol |
|---|---|
| `futures` | `{T}/{iv}` |
| `futures_contracts` | `{T}/{iv}/{expiry}` |
| `options_enriched` | `{T}/{iv}/{expiry}` |
| `borrow_rates` | `{T}/{iv}` (also `/cycle_summary` and `/continuous`). Columns are listed in `scripts/fetch_ticker.py::BORROW_COLUMNS`, including `spot_f1_pct` and `f1_f2_pct` in percent. |
| `spot` | `{T}/{iv}` |
| `indices` | `{NAME}/{iv}` |
| `metadata` | tables |

How it stores and reads:
- **Slim layout:** panels and borrow series store no copied columns. `read_futures(with_spot=True)` and `read_borrow_rates(joined=True)` join the inputs and **Kite** spot back at read time.
- **Spot guard:** `spot_symbol_for` only accepts `spot` symbols whose metadata `source` starts with `fetch_kite`. Upstox spot series are never used.
- `spot` and `spot_adjusted` live in their own LMDB path (`ARCTICDB_SPOT_DIR`).
