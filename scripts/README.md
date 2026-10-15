# scripts/

Command-line entry points for **fetch → store → features**. Each has `main(argv=None)` and is registered as a console command in `pyproject.toml`. All of them are ticker- and interval-agnostic.

Run them in this order for a new ticker:
1. `fetch-kite-equities` (spot)
2. `fetch-ticker --history` (futures + borrow)
3. `fetch-options`
4. `build-features`
5. `build-borrow-continuous`

## `fetch_ticker.py` (`fetch-ticker`)

Upstox futures for any NSE F&O ticker.

**`--history` mode** (the main path; intervals `5minute`, `15minute`, `1day`):
1. Discovers expired and live contracts (Upstox expired-instruments API).
2. Stores each one's full OHLCV+OI in `futures_contracts/{T}/{iv}/{expiry}`. Fetching resumes from each contract's last stored bar, so complete contracts cost 0 calls.
3. Rebuilds the F1/F2/F3 wide panel into `futures/{T}/{iv}` (slot/roll rules, `cycle_id`, `is_expiry_day`).
4. Computes borrow rates into `borrow_rates/{T}/{iv}` (21 columns, `BORROW_COLUMNS`). The rates use `dte_star` = (dte + 1) − bar-of-day / bars-per-day (75 at 5minute, 25 at 15minute), r = 0.0625, clipped at 0:
   - `b1 = r − ln(F1/S)/t1`
   - `b2 = r − ln(F2/S)/t2`
   - `b12 = r − ln(F2/F1)/(t2−t1)`, plus `b3`, `b23`, `b13`
   - spot spreads in ₹ (`spread_spot_f1`, `spread_spot_f2`, `spread_f1_f2`) and in percent: `spot_f1_pct` = (S − F1)/S × 100, `f1_f2_pct` = (F1 − F2)/F1 × 100
   - `OI_ratio`, `U_t`, `in_backwardation`
   - trade flags: `both_traded` = F1 and F2 both had volume in the bar (otherwise one leg's close is a stale, carried-forward price); `f2_thin_early` = 10+ trading days before F1 expiry, both legs traded, and F2 volume < 10% of F1 volume in the bar
   - missing-data flags: `data_missing` = the F1, F2 or spot close is missing on the bar, and `missing_inputs` names which (`"F2"`, `"F1,F2,SPOT"`, …). Missing closes are never filled; the rates that need them stay NaN.
5. **S is Kite spot** from `spot/{T}/{iv}`. It isn't written here (`cycle_summary` isn't either).
6. **OI = 0 on a bar with a price** (Upstox does this often) is replaced by that contract's previous OI when the panel is built, and `F{k}_oi_filled` marks the bar. The raw `futures_contracts` keep the zeros.

Options:
- `--borrow-only` recomputes the borrow rates from what is already stored.
- `--fill-missing` (no API calls) completes what is stored, without recomputing anything that exists:
  - a futures panel with no borrow series gets the full borrow series;
  - an existing borrow series that lacks `spot_f1_pct` / `f1_f2_pct` gets them from its own stored spreads, and one that lacks `both_traded` / `f2_thin_early` gets them from the panel's F1/F2 volume. Every other column is kept as stored.
- `--rebuild-from-stored` (no API calls) rebuilds the panel from the stored contracts and recomputes the borrow rates, as a full rewrite. A ticker whose stored panel has rows no stored contract explains is skipped, not overwritten.
- `--all` runs every ticker that has a futures panel at `--intervals`. For example, `fetch-ticker --fill-missing --all --intervals 5minute 15minute`.
- `--dry-run` prints the plan and writes nothing.

`borrow_rates` has a fixed column schema, so a series is rewritten in full once when it gains new columns. After that, incremental updates work as normal.

**Default mode** (no `--history`): uses the Upstox instrument-master JSON (live contracts only) to build the panel without spot.

## `fetch_options.py` (`fetch-options`)

Upstox option chains for any ticker:
1. Enumerates monthly expiries (`--expiries all` adds weeklies) and their last-N-trading-day windows (`--lookback-td`, default 40).
2. Fetches **option candles only**.
3. Prices each option bar on the close of `futures_contracts/{T}/{iv}/{expiry}` at **exactly the same timestamp**:
   - Weeklies are priced on the nearest monthly contract.
   - Bars with no futures bar at that instant are dropped and counted in the log.
   - No futures are fetched; run `fetch-ticker --history` first.
4. Enriches via `options/chain/build_enriched_chains.py` and writes `options_enriched/{T}/{iv}/{expiry}`.

**Spot is neither fetched nor written.**

Flags:
- Incremental: only the uncovered leading/trailing parts of each window are fetched; `--force` refetches everything.
- `--no-arctic` reads futures but writes nothing.
- `--dry-run` shows the plan, contract counts and whether each expiry's futures are stored.

## `fetch_kite_equities.py` (`fetch-kite-equities`)

Kite equity OHLCV for one or more tickers (or `--tickers-file`) at any Kite interval (`minute`, `5minute`, `15minute`, `30minute`, `60minute`, `day`).
- Fetching is chunked to Kite's per-request span caps.
- Writes `spot/{T}/{iv}` with metadata `source='fetch_kite_equities'`. That stamp is what makes the series count as Kite spot everywhere downstream.
- Needs a valid daily token (`kite-login`).

## `fetch_kite_indices.py` (`fetch-kite-indices`)

Broad-market and sectoral NSE indices from Kite into `indices/{NAME}/{iv}` (`15minute`/`30minute`/`60minute`/`day`).
- History is bounded by each index's launch date.
- Volume is dropped, because index volume is not traded quantity.

## `feature_builder.py` (`build-features TICKER… --interval IV`)

Adds feature columns to `borrow_rates/{T}/{iv}` for any ticker at any interval:

| Group | Columns |
|---|---|
| Volume | `F1_volume`, `F2_volume`, `feat_Vshare`, `feat_volflow` |
| OI / roll flow | `feat_oi_concentration`, `feat_roll_velocity`, `feat_roll_acceleration`, `feat_relative_oi_change`, `feat_net_roll_flow`, `feat_flow_intensity`, `feat_roll_deviation` (vs earlier cycles only, so no look-ahead) |
| Borrow / price | `feat_b12_percentile` (earlier cycles only), `feat_b12_slope_1d`, `feat_b12_accel`, within-cycle and rolling OLS of log F1 on DTE* (`feat_beta_*`, `feat_r2_*`) |
| IV, only if chains exist | `feat_atm_iv`, 25Δ/15Δ call/put IVs, `feat_25d_risk_reversal`, `feat_residual_atm_iv` |

How it runs:
- **Windows** are defined in trading time and converted to bars with `candles_per_day(interval)`. 15minute reproduces the original 250/25/10/5 bars exactly.
- **IV group:** built only when `options_enriched/{T}/{iv}` has chains. Both stored chain schemas are read. Otherwise it is skipped and not written.
- **Re-runs:** each run drops old feature columns first, then overwrites the series.
- `--features F…` builds only the listed features. Feature columns not listed keep their stored values, and groups nothing asked for (IV, b12 slope/accel) are skipped.
- `--dry-run` prints the coverage table without writing.

## `build_borrow_continuous.py` (`build-borrow-continuous`)

Writes `borrow_rates/{T}/{iv}/continuous`:
- the roll-stitched b1/b2/b12,
- the unclipped constant-maturity `b1_cm30`, `b2_cm60`, `b12_cm30_60`.

See `data/borrow_continuous.py`. `--intervals` defaults to `15minute 1day`.

## `build_gap_timeseries.py` (`build-gap-timeseries TICKER… --interval IV`)

Writes `results/ewma_b12/gap_timeseries/{t}_gap_timeseries.csv`, one row per bar:
- `b12_pts` = 100 × b12, and the EWMA gaps `gap_0.1_2`, `gap_0.1_1`, `gap_0.05_1` (fast − slow half-life in trading days, restarted every roll);
- `gap_0.05_1_acc`, the second difference of `gap_0.05_1` within the cycle;
- `F1_volume`, `F2_volume`, `both_traded`, `f2_thin_early`, `data_missing`, `missing_inputs` from the borrow series;
- the 7 OI roll features (`feat_oi_concentration` … `feat_roll_deviation`) as stored in the borrow series, so run `build-features` first.

Where F1 or F2 is missing, `b12_pts` and every gap are NaN and `data_missing` is True; nothing is filled. Floats keep full precision and dates are ISO, so don't re-save these files in Excel.

## `plot_htb_cycles.py` (`plot-htb-cycles TICKER… --interval IV`)

Static charts of hard-to-borrow cycles: every contract expiry whose smoothed b12 (causal EWMA, 0.1-day half-life) reaches `--threshold` (default 30%) in the final 5 trading days. One chart per expiry covers the whole time that contract was front month.
- **x-axis:** trading time (one step per bar). Nights, weekends and holidays take no space, so lines are continuous.
- **Panels:** b12 (centred 1-hour Gaussian smoothing over faint raw b12), spot−F1 and F1−F2 basis, F1/F2 OI with OI concentration, roll flow with the prototype active-roll toggle, and hourly F1/F2 volume with F2 share.
- **Data problems are shown, not hidden:** bars where the F1 slot held another contract are hatched and left out of the lines, and bars with a missing F1 or F2 close are marked. Roll flow is hidden on the bar after a missing OI print, because the stored OI change there is an artefact.
- **Output:** `results/charts/htb_cycles/{T}/{T}_{expiry}.png`, one `{T}.pdf` per ticker, and a `README.md` index.

## `ewma_b12_signal.py` (`test-ewma-b12 TICKER… --interval IV`)

Tests whether the gap between a fast and a slow EWMA of b12 predicts where b12 goes next. It is read-only and writes only CSV/HTML output.

**What it does:**
- **EWMAs:** both run on the stored b12, are restarted at every roll, and have half-lives given in **trading days** (`--fast`, `--slow`). Those are converted to bars and α for the interval (α = 1 − 0.5^(1/bars)). A bar with NaN b12 gets a NaN EWMA; the next valid bar continues from the state before the gap.
- **Divergence:** fast − slow.
- **Target:** b12[t+h] − b12[t] within the same cycle, for every `--horizons` value in bars.
- **Grid:** every fast < slow pair × every horizon, scored on the earlier cycles (train) and on the most recent `--test-frac` cycles (test).

**Metrics:**
- **IC:** the Spearman rank correlation inside each cycle, with a t-stat across cycles.
- **Direction hit rate,** compared with the base rate (the share of up moves).
- **Balanced accuracy:** 50% means no skill. Unlike the hit rate, it can't be inflated by always guessing "up".
- **Conditional means:** the average future change when the divergence is positive vs negative.

**Direction:** the best combination is the one with the strongest train IC **in either direction**.
- "trend": fast above slow → b12 keeps rising.
- "contrarian": fast above slow → b12 falls back.

`test_*_dir` columns re-express the test metrics in the direction learned on train; positive means the rule still worked.

**Flags:**
- `--from` / `--to`: date range.
- `--warmup`: skip the first warmup × slow-half-life of each cycle.
- `--exclude-last-days`: drop bars near F1 expiry.
- `--phase expiry-week|pre-expiry-week` (with `--expiry-week-td`, default 5): score only bars inside, or before, the last N trading days of each cycle.
- `--no-expiry-day-target`: drop targets that start or end on F1 expiry day, which removes the expiry-day b12 collapse.
- `--require-f2-trade`: keep only bars where F2 traded at both t and t+h.
- `--min-div`: only make a direction call when |divergence| exceeds this.

**Output:** `results/ewma_b12/<run>/`, containing `grid.csv`, `best.csv`, `per_cycle_best.csv` and `heatmap.html`.

**Watch out:**
- A long slow half-life with the default warm-up skips the start of each cycle. With 10 days, only the second half of a cycle (including expiry week) is scored. Compare against `--warmup 0.5` or `--exclude-last-days` before trusting a result.
- At 5-minute, stale F2 prints create fake b12 moves. Check with `--require-f2-trade`.

## `crossings.py` (`b12-crossings`)

Where in the cycle does the fast b12 EWMA cross the slow one?
- **Bar labels:**
  - up- and down-crossings;
  - "big" bars, where the divergence is in the ticker's top 10%;
  - "significant" up-crosses, which reach big within a day.
- **Comparison:** the last `--last-td`+1 trading days (default 10, expiry day included) against the earlier part of each cycle:
  - by days-to-expiry bucket;
  - cycle by cycle, with a sign test across cycles.
- `--pairs` takes fast/slow half-lives in days (e.g. `0.1/10 0.1/2`).
- `--warmup-days` skips the start of each cycle.

Output: `results/ewma_b12/crossings_<interval>/` (`by_bucket.csv`, `per_cycle.csv`, `summary.csv`).

## `crossings_last10.py` (`b12-crossings-last10`)

Counts every crossing in the last `--last-td`+1 trading days of each **complete** cycle:
- up and down;
- "lasting" (the new state holds at least `--lasting-minutes`) vs whipsaws.

It summarises per ticker (total, per cycle, per day, share lasting, on expiry day, cycles with none) and per day-to-expiry.

Output: `results/ewma_b12/crossings_last10_<interval>/` (`per_cycle.csv`, `by_day.csv`, `per_ticker.csv`).

## `regimes_last7.py` (`b12-regimes`)

In the last `--last-td`+1 trading days (default 7, expiry day included): while fast stays above slow, does b12 keep rising, and while it stays below, does b12 fall?

**Test A, segments:**
- **Entry:** once the state has held `--confirm-minutes`.
- **Exit:** when it flips, or at expiry.
- **Measures:** Δb12 and its excess over the normal drift at each day to expiry.
- Exiting at the flip makes this lean against the hypothesis by construction. It measures what a crossover rule would earn.

**Test B, fixed horizon:** the neutral version. Every bar whose state has held at least 60 minutes is scored on the b12 change over `--horizon-minutes`.

Both get a per-cycle sign test. The output also prints segments by entry day, and "below" segments that run into expiry day vs those that end before it.

Output: `results/ewma_b12/regimes_<interval>/`.

