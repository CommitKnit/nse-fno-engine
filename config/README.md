# config/

Runtime settings and static constants. Every other package imports from here; nothing here imports from the rest of the repo.

| File | What it does |
|---|---|
| `__init__.py` | Re-exports `settings` and the `NIFTY` / `BANKNIFTY` instrument configs. |
| `settings.py` | See below. |
| `constants.py` | See below. |

## `settings.py`

`Settings`, a pydantic-settings class loaded from the repo `.env`, exposed as the singleton `settings`.

- **Upstox:** `UPSTOX_API_KEY`, `UPSTOX_API_SECRET`, `UPSTOX_ACCESS_TOKEN`, `UPSTOX_BASE_URL`.
- **Kite:** `KITE_API_KEY`, `KITE_API_SECRET`, `KITE_ACCESS_TOKEN`.
  - `KITE_ENV_PATH` is the `.env` that `kite-login` writes the daily token to; it defaults to this repo's `.env`.
  - `KITE_HISTORICAL_RPS` and `KITE_TIMEOUT` set the request rate and timeout.
  - `KITE_INSTRUMENTS_CACHE_DIR` caches the Kite instrument dump.
- **Rate limits:** the Upstox per-second / per-minute / per-30-minute caps.
- **Storage:**
  - `ARCTICDB_DIR` is the main LMDB store.
  - `ARCTICDB_SPOT_DIR` is a separate LMDB for the `spot` and `spot_adjusted` libraries, which have a bigger `map_size`.
  - Both are absolute paths; override them in `.env` to point at an existing store.
  - The `ARCTICDB_URI` and `ARCTICDB_SPOT_URI` properties build the `lmdb://` URIs.
- **Backtest and risk defaults:** capital, timezone (`Asia/Kolkata`), `RISK_FREE_RATE`, drawdown/position/Greek limits.
- `ensure_dirs()` creates the data, ArcticDB and log folders.

## `constants.py`

Static constants shared by every layer.

- **Enums:** `Exchange`, `OptionType` (`CE`/`PE`), `Side` (`BUY`/`SELL`), `Interval`.
- **Instruments:** `InstrumentConfig` records (Upstox instrument key, lot size, tick size, strike interval) in `INSTRUMENTS`. `fetch_options.py` uses it to resolve index keys such as NIFTY without an API search.
- **Holidays:** `TRADING_HOLIDAYS_2025` / `TRADING_HOLIDAYS_2026` / `ALL_TRADING_HOLIDAYS`. These are the NSE holidays behind the trading-day calendar in `options/pricing/dte_utils.py`.
  - **Only 2025 and 2026 are listed.** Bars in other years are treated as if they had no holidays.
