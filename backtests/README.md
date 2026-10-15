# backtests/

Runners that build snapshots from ArcticDB, run a strategy through `core/engine/backtest_engine.py`, and write CSV results. Each one takes `--symbol` and `--interval` (defaults IREDA / 5minute). Each runs every variant twice:
- **gross:** zero costs;
- **net:** engine-default NSE costs plus 0.1% slippage.

Prerequisites for a symbol:
- the `futures/{T}/{iv}` panel (`fetch-ticker --history`);
- `options_enriched/{T}/{iv}/{expiry}` chains (`fetch-options`).

## `run_roll_put_hedge.py` (`backtest-roll-put-hedge`)

Runs the roll-spread put hedge (`options/strategies/roll_spread_put_hedge.py`). For each F1 expiry:
- **Window:** the final `--final-week-td` trading days (default 5).
- **Entry:** the F1−F2 spread trough before the week's peak.
- **Exit:** the peak.
- **These windows use look-ahead by design**, so the result is an upper bound, not a tradeable result.

Options:
- Hedge bands: `--hedges tight,medium,relaxed,initial`.
- Strike choice: `--strike atm|otm_liquid` with `--min-oi`.

Output in `results/{symbol}_roll_put_hedge/` (or `--out`):
- `windows.csv`
- `cycles.csv`: per cycle and band, with P&L attribution and the IV source of each fill.
- `paths.csv`
- `summary.csv`
- `skipped.csv`

It also exports `lookahead_windows(store, symbol, interval, …)`, which the trough-peak runner reuses.

## `run_trough_peak_options.py` (`backtest-trough-peak`)

Same look-ahead windows. Two unhedged strategies, one lot per leg:
- **calls:** ATM call plus the first OTM call with OI ≥ `--min-oi`;
- **puts:** the same with puts.

`--chain F2` (default) or `F1` picks the chain. Output in `results/{symbol}_trough_peak_options[_f1]/`: `legs.csv`, `cycles.csv`, `summary.csv`, `skipped.csv`, `windows.csv`.

## `run_expiry_straddle.py` (`backtest-expiry-straddle`)

Long F1 ATM straddle:
- **Entry:** the 15:00 bar of expiry − N calendar days (`--entries`).
- **Exit:** the last bar of expiry − M days (`--exits`).

Every (N, M) pair is run. Output in `results/{symbol}_expiry_straddle/`: `schedule.csv`, `cycles.csv`, `summary.csv`.

## `report_roll_put_hedge.py` (`report-roll-put-hedge`)

Reads a `run_roll_put_hedge` results folder (`--results`, default `results/{symbol}_roll_put_hedge`) and writes a self-contained HTML report (`--html`, default `plots/{symbol}_roll_put_hedge.html`). Plotly is inlined and the page is light/dark aware. It shows:
- KPI tiles;
- net P&L per cycle by hedge band;
- the attribution split into spread/gamma/vega/theta;
- a per-cycle drill-down with the final-week spread path, in-trade P&L and the hedge.
