# nse-fno-engine

A generic pipeline for NSE stock/index F&O research. It works for any ticker and interval:

1. **Fetch:** Upstox futures and options, plus Kite spot and indices.
2. **Store:** everything goes into ArcticDB.
3. **Features:** borrow rates (b1/b2/b12) and roll/OI features.
4. **Backtest:** an event-driven engine.
5. **Results:** CSV outputs and an HTML report.

Every folder has its own `README.md` describing each file in it.

## Data flow

```
 FETCH                                   STORE  (ArcticDB, LMDB)                         DERIVE / USE
 ─────                                   ──────────────────────                          ────────────
 Upstox ── fetch-ticker --history ─────► futures_contracts/{T}/{iv}/{expiry}  raw contract OHLCV+OI ─┐
                                         futures/{T}/{iv}                     F1/F2/F3 wide panel     │
                                         borrow_rates/{T}/{iv}  (+/cycle_summary)  ◄── b1,b2,b12…  ◄──┤ needs Kite spot
                                                                                                      │
 Kite ──── fetch-kite-equities ────────► spot/{T}/{iv}   (metadata source=fetch_kite_equities) ───────┘
 Kite ──── fetch-kite-indices ─────────► indices/{NAME}/{iv}
 (daily token: kite-login)

 Upstox ── fetch-options ──────────────► options_enriched/{T}/{iv}/{expiry}
           option candles only; priced     ▲ IV + Greeks (Black-76 on the future)
           on futures_contracts close at   │ options/chain/build_enriched_chains.py
           the SAME timestamp ─────────────┘

 borrow_rates ── build-features ───────► borrow_rates/{T}/{iv}  (+ feat_* columns; IV feats only if chains exist)
 borrow_rates ── build-borrow-continuous► borrow_rates/{T}/{iv}/continuous

 futures + options_enriched ── LocalSnapshotBuilder ──► MarketSnapshot per bar
        ──► BacktestEngine + strategy (backtest-*) ──► results/{t}_<strategy>/*.csv
        ──► report-roll-put-hedge ──► plots/{t}_roll_put_hedge.html
```

**Spot rule:** option IV and Greeks are priced on the **future** (Black-76), so they never use spot. Wherever spot *is* used (the b1/b2/spot spreads in borrow rates, `SPOT_*` joins, snapshot spot bars), it is always the **Kite** series. `ArcticStore.spot_symbol_for` only accepts `spot` symbols stamped `source=fetch_kite*`; anything else gives NaN spot.

## Layout

| Folder | What lives there |
|---|---|
| [config/](config/README.md) | `.env`-backed settings, enums, instrument registry, NSE holidays |
| [data/](data/README.md) | Upstox/Kite fetchers, ArcticDB store, data models, option cleaner, snapshot builder, continuous borrow |
| [options/](options/README.md) | Chain enrichment, Black-76/IV/DTE pricing, slippage, the three option strategies |
| [core/](core/README.md) | Event-driven backtest engine: events, orders/execution, portfolio, risk, strategy base |
| [scripts/](scripts/README.md) | CLI entry points for fetch → store → features |
| [backtests/](backtests/README.md) | Backtest runners and the HTML report |
| [tests/](tests/README.md) | Unit tests (run on a temporary LMDB store) |

## Setup (Windows)

```bash
py -3.13 -m venv .venv
.venv\Scripts\python -m pip install -r requirements.lock
.venv\Scripts\python -m pip install -e . --no-deps
```

- `requirements.lock` holds the exact versions of the original working environment (Python 3.13.3).
- `pip install -e .` registers the console commands below in `.venv\Scripts`.
- Copy `.env.example` to `.env` and fill in the keys.
  - `ARCTICDB_DIR` / `ARCTICDB_SPOT_DIR` point at the ArcticDB store.
  - The shipped `.env` points at the existing store in `C:\Users\risha\backtesting-engine\backtesting-engine\data_cache`, so nothing was copied. **Don't run writers from two repos at the same time against the same store.**

Activate with `.venv\Scripts\activate`, or call `.venv\Scripts\<command>.exe` directly.

## Daily Kite login

Kite access tokens expire at about 06:00 IST every day.

```bash
kite-login
```
This prints the login URL. Open it, log in, and copy the `request_token` from the redirect URL. Then run:
```bash
kite-login --request-token <request_token or the whole redirect URL>
```
That saves `KITE_ACCESS_TOKEN` into `.env`, and every Kite command reads it from there.

## Typical run for a new ticker

```bash
fetch-kite-equities TCS --intervals 5minute 15minute day
```
Fetches Kite spot. It must exist before the borrow rates are computed.

```bash
fetch-ticker TCS --history --intervals 5minute 15minute 1day
```
Fetches futures contracts and builds the panel and borrow rates.

```bash
fetch-options TCS --interval 5minute
```
Fetches option candles and prices them on the stored futures.

```bash
build-features TCS --interval 5minute
```
Builds the OI/roll/b12 features, plus IV features when chains exist.

```bash
build-borrow-continuous TCS --intervals 5minute
```
Builds the roll-stitched and constant-maturity b1/b2/b12 series.

```bash
backtest-roll-put-hedge --symbol TCS --interval 5minute
```
Writes `results/tcs_roll_put_hedge/*.csv`.

```bash
report-roll-put-hedge --symbol TCS --interval 5minute
```
Writes `plots/tcs_roll_put_hedge.html`.

## Commands

| Command | Module | Does |
|---|---|---|
| `fetch-ticker` | `scripts/fetch_ticker.py` | Upstox futures contracts → panel → borrow rates |
| `fetch-options` | `scripts/fetch_options.py` | Upstox option candles → enriched chains (priced on stored futures) |
| `fetch-kite-equities` | `scripts/fetch_kite_equities.py` | Kite equity spot → `spot` |
| `fetch-kite-indices` | `scripts/fetch_kite_indices.py` | Kite NSE indices → `indices` |
| `kite-login` | `data/fetchers/kite_fetcher.py:login_main` | request_token → daily access token in `.env` |
| `build-features` | `scripts/feature_builder.py` | `feat_*` columns on `borrow_rates/{T}/{iv}` |
| `build-borrow-continuous` | `scripts/build_borrow_continuous.py` | `borrow_rates/{T}/{iv}/continuous` |
| `test-ewma-b12` | `scripts/ewma_b12_signal.py` | Grid-search a fast/slow EWMA divergence of b12 as a predictor of future b12 (train/test by cycle) |
| `b12-crossings` | `scripts/crossings.py` | Where in the cycle fast/slow b12 EWMA crossings and big divergences happen (last N days vs earlier) |
| `b12-crossings-last10` | `scripts/crossings_last10.py` | Count crossings per ticker/cycle/day in the last N trading days (lasting vs whipsaw) |
| `b12-regimes` | `scripts/regimes_last7.py` | Near expiry: does b12 keep moving with the fast/slow state? (segment and fixed-horizon tests) |
| `backtest-roll-put-hedge` | `backtests/run_roll_put_hedge.py` | F2 put delta-hedged with F1, final-week roll window |
| `backtest-trough-peak` | `backtests/run_trough_peak_options.py` | Unhedged long calls/puts over the same window |
| `backtest-expiry-straddle` | `backtests/run_expiry_straddle.py` | Long F1 ATM straddle into expiry |
| `report-roll-put-hedge` | `backtests/report_roll_put_hedge.py` | Self-contained HTML report of a roll-put-hedge run |

Every command takes `--help`. Each is also runnable as `python scripts/<file>.py …` / `python backtests/<file>.py …`.

### Why the entry points were rebuilt

In the original `pyproject.toml`, 4 of the 5 entry points were dead and the scripts people actually used had none:

| Old entry | Problem |
|---|---|
| `backtest = scripts.run_backtest:main` | `run_backtest.py` was a 4-line TODO stub, so the command did nothing. |
| `fetch-data = scripts.fetch_historical_data:main` | The file had been deleted, so `fetch-data.exe` crashed with `ModuleNotFoundError` on launch. |
| `fetch-spot = scripts.fetch_spot:main` | Upstox spot fetcher, now removed: it overwrote the Kite spot that the borrow rates depend on. |
| `build-chains = scripts.build_enriched_chains:main` | The old parquet medallion CLI, now removed; the module is a library under `options/chain/`. |
| *(none)* `fetch_options.py` | Sat at the repo root outside any package, so it *couldn't* be an entry point. |
| *(none)* Kite fetchers, feature builder | Had `main()` but were never registered. |
| `packages = [...]` | Did not list `scripts`, so even the working entries only resolved in editable installs, by accident. |

The fix:
- `fetch_options.py` moved into `scripts/`.
- Every entry-point function is `main(argv=None)`, so it can be called from code or tests.
- The IREDA-only runners became `backtests/*` with `--symbol/--interval`.
- `kite-login` was added for the daily token.
- Hatch `packages` now lists `config, core, data, options, scripts, backtests`, so every target imports in editable and regular installs alike.
- `pip install -e .` regenerates the `.exe` launchers.

## Tests

```bash
.venv\Scripts\python -m pytest tests
```
The tests create throw-away LMDB stores with `?map_size=20MB`. Without that, each store would preallocate GBs on Windows.
