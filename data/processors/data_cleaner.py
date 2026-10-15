"""Data cleaner — handle missing bars, outliers, and after-hours data.

Two public classes
------------------
DataCleaner
    Generic OHLCV cleaner (market hours, price outliers, gap-filling).

OptionsDataCleaner
    Institutional-grade, fully vectorised multi-stage cleaner for option
    chain DataFrames.  No Python-level row iterations (no iterrows).

Full stage order of OptionsDataCleaner (as originally designed)
----------------------------------------------------------------
  Stage 0  calculate_synthetic_forward()   PCP-derived forward per (ts, dte)
  Stage 1  pre_solver_filter()             Microstructure cleaning before IV solve
  Stage 2  remove_arbitrage()              Butterfly + calendar-spread arb removal
  ── external ──────────────────────────────────────────────────────────────────
  [ChainEnricher computes IV + Greeks using synthetic_forward as F]
  ── external ──────────────────────────────────────────────────────────────────
  Stage 3  post_solver_cleanup()           IV sanity-bound filter
  Stage 4  repair_stale_prices()           Stale-price interpolation (ffill + B76)

The other pipeline entry point, options/chain/build_enriched_chains.py::run_stage2_enrich
(used by fetch_options.py), diverges: it skips calculate_synthetic_forward and
remove_arbitrage entirely (prices on the raw future, flags rather than removes
arbitrage via flag_arbitrage), and — instead of a separate repair_stale_prices
pass after enrichment — does IV solving and stale-price repair together in one
pass via ChainEnricher.enrich_dataframe. See that method's docstring, and the
note on repair_stale_prices below, for why.

Expected column names (raw schema)
-----------------------------------
  timestamp, open, high, low, close, volume, oi,
  instrument_key, strike_price, instrument_type  ("CE" / "PE"),
  dte, futures_price, moneyness

Column aliases auto-normalised on every entry point:
  strike       → strike_price
  option_type  → instrument_type
  F            → futures_price
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from loguru import logger

from options.pricing.dte_utils import TRADING_DAYS_PER_YEAR, year_fraction


# ══════════════════════════════════════════════════════════════════════════════
# Generic OHLCV cleaner
# ══════════════════════════════════════════════════════════════════════════════

class DataCleaner:
    """Cleans raw OHLCV data from the API.

    Handles:
    - Missing bars (forward-fill within trading hours)
    - Price outliers (spikes > N standard deviations)
    - After-hours data filtering
    - Zero-volume bars (flag as stale)
    - Negative or zero prices
    """

    def __init__(
        self,
        market_open: str = "09:15",
        market_close: str = "15:30",
        timezone: str = "Asia/Kolkata",
        outlier_std_threshold: float = 5.0,
    ):
        self.market_open = market_open
        self.market_close = market_close
        self.timezone = timezone
        self.outlier_std_threshold = outlier_std_threshold

    def clean(self, df: pd.DataFrame, intraday: bool = False) -> pd.DataFrame:
        """Run all cleaning steps on a candle DataFrame.

        Args:
            df: DataFrame with columns [timestamp, open, high, low, close, volume]
                and optionally [oi, instrument_key].
            intraday: If True, filter to market hours and forward-fill gaps.

        Returns:
            Cleaned DataFrame.
        """
        if df.empty:
            return df

        df = df.copy()
        initial_len = len(df)

        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.sort_values("timestamp").reset_index(drop=True)
        df = df.drop_duplicates(subset=["timestamp"])
        df = self._remove_invalid_prices(df)

        if intraday:
            df = self._filter_market_hours(df)

        df = self._handle_outliers(df)
        df["is_stale"] = df["volume"] == 0

        removed = initial_len - len(df)
        if removed > 0:
            logger.debug(f"DataCleaner.clean: removed {removed}/{initial_len} rows.")

        return df.reset_index(drop=True)

    def _remove_invalid_prices(self, df: pd.DataFrame) -> pd.DataFrame:
        price_cols = ["open", "high", "low", "close"]
        for col in price_cols:
            if col in df.columns:
                mask = df[col] > 0
                bad = (~mask).sum()
                if bad > 0:
                    logger.warning(f"Removing {bad} rows with {col} <= 0.")
                df = df[mask]
        return df

    def _filter_market_hours(self, df: pd.DataFrame) -> pd.DataFrame:
        ts = df["timestamp"]
        if ts.dt.tz is None:
            ts = ts.dt.tz_localize(self.timezone)
        else:
            ts = ts.dt.tz_convert(self.timezone)

        t_only    = ts.dt.time
        open_t    = pd.Timestamp(self.market_open).time()
        close_t   = pd.Timestamp(self.market_close).time()
        mask      = (t_only >= open_t) & (t_only <= close_t)
        kept      = mask.sum()
        if kept < len(df):
            logger.debug(f"Market hours filter: kept {kept}/{len(df)} bars.")
        return df[mask]

    def _handle_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        if len(df) < 20:
            df["is_outlier"] = False
            return df

        rolling_mean = df["close"].rolling(window=20, min_periods=5).mean()
        rolling_std  = df["close"].rolling(window=20, min_periods=5).std()
        deviation    = (df["close"] - rolling_mean).abs()
        threshold    = rolling_std * self.outlier_std_threshold
        df["is_outlier"] = deviation > threshold

        n = df["is_outlier"].sum()
        if n > 0:
            logger.warning(f"Found {n} outlier bars (>{self.outlier_std_threshold} std devs).")
        return df

    def forward_fill_gaps(self, df: pd.DataFrame, freq: str = "1min") -> pd.DataFrame:
        """Forward-fill missing time slots within trading hours."""
        if df.empty:
            return df

        df = df.set_index("timestamp")
        full_idx = pd.date_range(start=df.index.min(), end=df.index.max(), freq=freq)
        df = df.reindex(full_idx)

        for col in ["open", "high", "low", "close"]:
            if col in df.columns:
                df[col] = df[col].ffill()

        if "volume" in df.columns:
            df["volume"] = df["volume"].fillna(0).astype(int)
        if "oi" in df.columns:
            df["oi"] = df["oi"].ffill().fillna(0).astype(int)

        df.index.name = "timestamp"
        return df.reset_index()


# ══════════════════════════════════════════════════════════════════════════════
# Options-specific cleaner
# ══════════════════════════════════════════════════════════════════════════════

class OptionsDataCleaner:
    """Institutional-grade vectorised six-stage cleaner for option chain DataFrames.

    All filtering stages are fully vectorised (Pandas / NumPy).
    No Python-level row iterations (no iterrows / itertuples).

    Class-level thresholds (override via constructor kwargs or subclassing):

    +-----------------------+----------+--------------------------------------+
    | Constant              | Value    | Meaning                              |
    +=======================+==========+======================================+
    | MARKET_OPEN_TIME      | 09:20    | First clean intraday bar             |
    | MARKET_CLOSE_TIME     | 15:25    | Last clean intraday bar              |
    | STRIKE_BAND           | 1 200    | Absolute F ± band for all strikes    |
    | CALL_ITM_DEPTH        | 100      | Calls: keep K ≥ F − 100             |
    | PUT_ITM_DEPTH         | 100      | Puts:  keep K ≤ F + 100             |
    | ARB_FLOOR             | 0.01     | Minimum lower arbitrage bound        |
    | MIN_IV                | 0.01     | Post-solve IV lower bound (1 %)      |
    | MAX_IV                | 3.00     | Post-solve IV upper bound (300 %)    |
    +-----------------------+----------+--------------------------------------+
    """

    MARKET_OPEN_TIME  : str   = "09:15"   # full NSE session
    MARKET_CLOSE_TIME : str   = "15:30"

    ARB_FLOOR         : float = 0.01
    MIN_IV            : float = 0.01
    MAX_IV            : float = 3.00
    # Strike retention is delta-based (see filter_delta_band), not a fixed price
    # band: keep contracts with ITM_DELTA_MAX >= |delta| >= OTM_DELTA_MIN.
    ITM_DELTA_MAX     : float = 0.94
    OTM_DELTA_MIN     : float = 0.02

    def __init__(
        self,
        risk_free_rate: float | None = None,
        timezone: str = "Asia/Kolkata",
    ) -> None:
        """
        Args:
            risk_free_rate: Annualised risk-free rate for Black-76 synthetic
                            pricing.  Defaults to ``settings.RISK_FREE_RATE``.
            timezone:       Timezone string used for market-hours filtering.
        """
        if risk_free_rate is not None:
            self.r = risk_free_rate
        else:
            from config.settings import settings
            self.r = settings.RISK_FREE_RATE

        self.timezone = timezone

    # ══════════════════════════════════════════════════════════════════════════
    # Stage 0 — Synthetic Forward via Put-Call Parity
    # ══════════════════════════════════════════════════════════════════════════

    def calculate_synthetic_forward(self, df: pd.DataFrame) -> pd.DataFrame:
        """Derive an implied-forward price for every (timestamp, dte) snapshot.

        Algorithm
        ---------
        For each (timestamp, dte) group:

        1. Inner-join CE and PE prices on ``strike_price``.
        2. Find the "ATM" strike K* where |C(K) − P(K)| is minimised.
        3. Apply Put-Call Parity:  F* = K* + C(K*) − P(K*)
        4. Broadcast F* to **all** rows that share the same (timestamp, dte).

        The resulting ``synthetic_forward`` column supersedes ``futures_price``
        for all downstream moneyness / IV calculations.

        Rows with no matched call/put pair fall back to ``futures_price``.

        Args:
            df: Raw option-chain DataFrame (one row per contract per timestamp).

        Returns:
            Copy of ``df`` with ``synthetic_forward`` column added (or replaced).
        """
        if df.empty:
            return df.copy()

        df = df.copy()
        df = self._normalise_columns(df)
        n_initial = len(df)

        itype   = df["instrument_type"].str.upper().str.strip()
        is_call = itype.isin({"CE", "CALL"})
        is_put  = itype.isin({"PE", "PUT"})

        # ── Build call and put sub-frames (deduplicated per strike per snapshot)
        call_cols = ["timestamp", "dte", "strike_price", "close"]
        put_cols  = ["timestamp", "dte", "strike_price", "close"]

        calls = (
            df.loc[is_call, call_cols]
            .groupby(["timestamp", "dte", "strike_price"], sort=False)
            .first()
            .reset_index()
            .rename(columns={"close": "_call_close"})
        )
        puts = (
            df.loc[is_put, put_cols]
            .groupby(["timestamp", "dte", "strike_price"], sort=False)
            .first()
            .reset_index()
            .rename(columns={"close": "_put_close"})
        )

        # ── Inner join to get call/put pairs at the same strike ───────────────
        pairs = pd.merge(
            calls, puts,
            on=["timestamp", "dte", "strike_price"],
            how="inner",
        )

        if pairs.empty:
            logger.warning(
                "calculate_synthetic_forward: no call/put pairs found across any "
                "snapshot.  Falling back to futures_price for all rows."
            )
            df["synthetic_forward"] = df.get("futures_price", pd.Series(np.nan, index=df.index))
            return df.reset_index(drop=True)

        # ── Per (timestamp, dte): find the strike minimising |C − P| ──────────
        pairs["_abs_diff"] = (pairs["_call_close"] - pairs["_put_close"]).abs()
        atm_idx = (
            pairs
            .groupby(["timestamp", "dte"], sort=False)["_abs_diff"]
            .idxmin()
        )
        atm = pairs.loc[atm_idx.values, [
            "timestamp", "dte", "strike_price",
            "_call_close", "_put_close",
        ]].copy()

        # ── PCP:  F* = K_ATM + C_ATM − P_ATM ─────────────────────────────────
        atm["synthetic_forward"] = (
            atm["strike_price"]
            + atm["_call_close"]
            - atm["_put_close"]
        )

        fwd_map = (
            atm[["timestamp", "dte", "synthetic_forward"]]
            .reset_index(drop=True)
        )

        # Drop existing column to prevent merge suffix collision
        if "synthetic_forward" in df.columns:
            df = df.drop(columns=["synthetic_forward"])

        # ── Broadcast synthetic_forward to every row in the snapshot ──────────
        df = df.merge(fwd_map, on=["timestamp", "dte"], how="left")

        # Fallback: rows with no pair use raw futures_price
        n_missing = int(df["synthetic_forward"].isna().sum())
        if n_missing > 0 and "futures_price" in df.columns:
            df["synthetic_forward"] = df["synthetic_forward"].fillna(df["futures_price"])
            logger.debug(
                f"calculate_synthetic_forward: {n_missing} rows had no "
                "call/put pair — fell back to futures_price."
            )

        n_computed = int(df["synthetic_forward"].notna().sum())
        n_snapshots = int(fwd_map["timestamp"].nunique())
        logger.info(
            f"calculate_synthetic_forward: "
            f"synthetic_forward set for {n_computed}/{n_initial} rows "
            f"across {n_snapshots} unique timestamps."
        )
        return df.reset_index(drop=True)

    # ══════════════════════════════════════════════════════════════════════════
    # Stage 1 — Pre-solver microstructure filter
    # ══════════════════════════════════════════════════════════════════════════

    def pre_solver_filter(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove noise and arbitrage-violating rows before the IV solver.

        Filters applied in order
        ------------------------
        1. **Market-hours** — keep 09:15–15:30 IST (full session).
        2. **Liquidity** — drop rows where ``oi == 0``.
        3. **DTE** — keep ``dte >= 0`` (0-DTE rows are kept intentionally).
        4. **Arbitrage bounds** (F = ``futures_price``):

           * Call: ``close ≥ max(F − K, 0.01)``
           * Put:  ``close ≥ max(K − F, 0.01)``

        There is **no** fixed strike / moneyness band — strike retention is
        delta-based and applied post-solve by :meth:`filter_delta_band`. Options
        are priced on the **future** (``futures_price``); the synthetic (put-call
        parity) forward is not used.

        All filtering is vectorised — no iterrows.

        Args:
            df: Raw option-chain DataFrame (one row per contract per timestamp).

        Returns:
            Filtered copy with index reset.
        """
        if df.empty:
            return df.copy()

        df = df.copy()
        df = self._normalise_columns(df)
        n_initial = len(df)

        # ── Forward price = the future (options are priced on the future) ──────
        if "futures_price" not in df.columns:
            raise ValueError(
                "pre_solver_filter: DataFrame must contain 'futures_price'."
            )
        df["_fwd"] = df["futures_price"]

        # ── 1. Market-hours filter ─────────────────────────────────────────────
        ts = pd.to_datetime(df["timestamp"])
        ts_tz = (
            ts.dt.tz_localize(self.timezone)
            if ts.dt.tz is None
            else ts.dt.tz_convert(self.timezone)
        )
        t_only  = ts_tz.dt.time
        open_t  = pd.Timestamp(f"2000-01-01 {self.MARKET_OPEN_TIME}").time()
        close_t = pd.Timestamp(f"2000-01-01 {self.MARKET_CLOSE_TIME}").time()
        hours_mask = (t_only >= open_t) & (t_only <= close_t)

        n_before = len(df)
        df = df[hours_mask].reset_index(drop=True)
        logger.info(
            f"pre_solver_filter | [1] market hours "
            f"({self.MARKET_OPEN_TIME}–{self.MARKET_CLOSE_TIME} IST): "
            f"kept {len(df)}/{n_before}, dropped {n_before - len(df)}."
        )
        if df.empty:
            return df

        # ── 2. Liquidity filter: drop zero open-interest ───────────────────────
        if "oi" in df.columns:
            n_before = len(df)
            oi_mask = df["oi"] > 0
            df = df[oi_mask].reset_index(drop=True)
            logger.info(
                f"pre_solver_filter | [2] OI > 0: "
                f"kept {len(df)}/{n_before}, dropped {n_before - len(df)}."
            )
        else:
            logger.warning("pre_solver_filter: 'oi' column not found; skipping liquidity filter.")

        if df.empty:
            return df

        # ── 3. DTE filter: keep dte >= 0 (include expiry day) ─────────────────
        if "dte" in df.columns:
            n_before = len(df)
            dte_mask = df["dte"].astype(float) >= 0  # fractional trading-day DTE
            df = df[dte_mask].reset_index(drop=True)
            logger.info(
                f"pre_solver_filter | [3] DTE ≥ 0: "
                f"kept {len(df)}/{n_before}, dropped {n_before - len(df)}."
            )
        else:
            logger.warning("pre_solver_filter: 'dte' column not found; skipping DTE filter.")

        if df.empty:
            return df

        # ── 4. Arbitrage-bound filter ──────────────────────────────────────────
        itype   = df["instrument_type"].str.upper().str.strip()
        is_call = itype.isin({"CE", "CALL"}).values
        K       = df["strike_price"].values.astype(float)
        F       = df["_fwd"].values.astype(float)

        call_lb     = np.maximum(F - K, self.ARB_FLOOR)
        put_lb      = np.maximum(K - F, self.ARB_FLOOR)
        lower_bound = np.where(is_call, call_lb, put_lb)
        arb_mask    = df["close"].values >= lower_bound

        n_before = len(df)
        df = df[arb_mask].reset_index(drop=True)
        logger.info(
            f"pre_solver_filter | [4] arbitrage bounds "
            f"(floor={self.ARB_FLOOR}): "
            f"kept {len(df)}/{n_before}, dropped {n_before - len(df)}."
        )

        # ── Remove internal helper column ──────────────────────────────────────
        df = df.drop(columns=["_fwd"], errors="ignore")

        logger.info(
            f"pre_solver_filter | TOTAL: "
            f"{n_initial} → {len(df)} rows "
            f"({n_initial - len(df)} dropped)."
        )
        return df.reset_index(drop=True)

    # ══════════════════════════════════════════════════════════════════════════
    # Stage — Arbitrage FLAGGING (butterfly + calendar); runs after stale repair
    # ══════════════════════════════════════════════════════════════════════════

    def flag_arbitrage(self, df: pd.DataFrame) -> pd.DataFrame:
        """Flag (never drop) butterfly and calendar-spread arbitrage violations.

        Adds three boolean columns and leaves every row in place:

        * ``butterfly_arb``    — interior strike with negative discrete second
          derivative of ``close`` w.r.t. strike (concave smile) within each
          ``(timestamp, dte, instrument_type)`` group.
        * ``calendar_arb``     — a back-month ``close`` strictly below the
          preceding shorter-DTE ``close`` within each
          ``(timestamp, strike_price, instrument_type)`` group.
        * ``arb_has_synthetic``— set on a flagged row when the row itself **or**
          any strike/leg involved in that violation (butterfly neighbours, or the
          calendar front-month leg) was repaired synthetically
          (``is_synthetic`` or ``synthetic_via_putcallparity``).

        Must run **after** :meth:`repair_stale_prices` so the synthetic labels
        exist. All operations are vectorised (``groupby + transform(shift)``).

        .. note::
           The calendar-spread check is currently disabled (see the comment
           in this method) — it's a no-op on the single-expiry frames every
           caller passes in, and was the source of a severe performance
           pathology on wide chains. ``calendar_arb`` is still added as a
           column (always ``False``) so the output schema is unchanged.
        """
        if df.empty:
            return df.copy()

        df = self._normalise_columns(df.copy())

        # Synthetic indicator = either repair label (missing columns → all False).
        syn = pd.Series(False, index=df.index)
        for col in ("is_synthetic", "synthetic_via_putcallparity"):
            if col in df.columns:
                syn = syn | df[col].fillna(False).astype(bool)
        df["_syn"] = syn.values

        df["butterfly_arb"]     = False
        df["calendar_arb"]      = False
        df["arb_has_synthetic"] = False

        # ── Butterfly (per smile, sorted by strike) ────────────────────────────
        df = df.sort_values(
            ["timestamp", "dte", "instrument_type", "strike_price"]
        ).reset_index(drop=True)
        g_bf = df.groupby(["timestamp", "dte", "instrument_type"], sort=False)
        c_prev   = g_bf["close"].transform(lambda s: s.shift(1))
        c_next   = g_bf["close"].transform(lambda s: s.shift(-1))
        syn_prev = g_bf["_syn"].transform(lambda s: s.shift(1)).eq(True)
        syn_next = g_bf["_syn"].transform(lambda s: s.shift(-1)).eq(True)
        d2c = c_prev - 2.0 * df["close"] + c_next
        bf_bad = d2c.notna() & (d2c < 0.0)
        df.loc[bf_bad, "butterfly_arb"] = True
        df.loc[bf_bad & (df["_syn"] | syn_prev | syn_next), "arb_has_synthetic"] = True
        n_bf = int(bf_bad.sum())

        # ── Calendar (per strike, sorted by DTE) — DISABLED for now ────────────
        # Every caller in this codebase invokes flag_arbitrage on a single-
        # expiry frame (fetch_options.py processes one expiry at a time), so
        # this group key — (timestamp, strike_price, instrument_type) — always
        # collapses to one row per group: there's no second expiry at the same
        # strike/timestamp to compare against, so c_front is always NaN and
        # calendar_arb is structurally guaranteed to be all-False. Meanwhile
        # the lambda-wrapped .transform() defeats pandas' vectorized path and
        # falls back to a per-group Python loop (_transform_general) — with
        # ~1 row/group that's millions of near-singleton groups on a wide,
        # 1-minute chain (confirmed live via py-spy on a BANKNIFTY run: this
        # was the entire multi-hour stall). Commented out rather than left to
        # run for a guaranteed-zero result. Re-enable if this is ever called
        # on a genuinely multi-expiry frame (calendar spread only means
        # something when >1 expiry is present).
        # df = df.sort_values(
        #     ["timestamp", "strike_price", "instrument_type", "dte"]
        # ).reset_index(drop=True)
        # g_cal = df.groupby(["timestamp", "strike_price", "instrument_type"], sort=False)
        # c_front   = g_cal["close"].transform(lambda s: s.shift(1))
        # syn_front = g_cal["_syn"].transform(lambda s: s.shift(1)).eq(True)
        # cal_bad = c_front.notna() & (df["close"] < c_front)
        # df.loc[cal_bad, "calendar_arb"] = True
        # df.loc[cal_bad & (df["_syn"] | syn_front), "arb_has_synthetic"] = True
        n_cal = 0

        df = df.drop(columns=["_syn"], errors="ignore")
        logger.info(
            f"flag_arbitrage: {n_bf} butterfly, {n_cal} calendar, "
            f"{int(df['arb_has_synthetic'].sum())} involving a synthetic "
            f"(no rows dropped)."
        )
        return df.reset_index(drop=True)

    # ══════════════════════════════════════════════════════════════════════════
    # Stage 3 — Post-solver IV cleanup
    # ══════════════════════════════════════════════════════════════════════════

    def post_solver_cleanup(self, df: pd.DataFrame) -> pd.DataFrame:
        """Drop rows whose **solved** IV is out of sane range; keep NaN-IV rows.

        Drops only rows where a solved ``iv`` is ``< MIN_IV`` (< 1 %) or
        ``> MAX_IV`` (> 300 %) — numerical-floor / anomaly violations.

        Rows with ``iv`` NaN (solver did not converge, or greeks undefined at
        expiry) are **kept** — edge/expiry bars are preserved and the downstream
        delta band decides their fate. This is a deliberate change from the
        earlier behaviour that dropped every NaN-IV row.

        Args:
            df: Post-IV-solver DataFrame.  Must contain an ``iv`` column.

        Returns:
            Cleaned copy with index reset.
        """
        if df.empty:
            return df.copy()

        df = df.copy()
        n_initial = len(df)

        if "iv" not in df.columns:
            logger.warning("post_solver_cleanup: 'iv' column not found — skipping.")
            return df

        # Keep NaN IV; drop only solved-but-out-of-range IV.
        iv = pd.to_numeric(df["iv"], errors="coerce")
        out_of_range = iv.notna() & ((iv < self.MIN_IV) | (iv > self.MAX_IV))
        df = df[~out_of_range].reset_index(drop=True)
        logger.info(
            f"post_solver_cleanup: dropped {int(out_of_range.sum())} out-of-range "
            f"IV rows (range [{self.MIN_IV}, {self.MAX_IV}]); kept NaN-IV rows. "
            f"{n_initial} → {len(df)}."
        )
        return df.reset_index(drop=True)

    # ══════════════════════════════════════════════════════════════════════════
    # Stage 4 — Stale-price repair (post-IV-solve)
    # ══════════════════════════════════════════════════════════════════════════

    def repair_stale_prices(self, df: pd.DataFrame) -> pd.DataFrame:
        """Repair zero-volume (stale) bars, preferring put-call parity.

        .. note::
           ``options/chain/build_enriched_chains.py::run_stage2_enrich`` (used by
           ``fetch_options.py``) no longer calls this — it now does the
           equivalent classification in a single pass via
           ``ChainEnricher.enrich_dataframe``, which also fixes an ordering
           bug present here: this method's both-stale forward-fill
           (Tier 2, below) can read a put-call-parity row's *pre-repair* IV
           (Tier 1 only re-solves IV for its own rows in a separate loop that
           runs after both tiers), so a downstream ffill can silently inherit
           a stale value. Nothing in this repo calls this method; it is kept
           for direct use on other chain frames.

        A bar is **stale** when ``volume == 0`` (regardless of whether an IV was
        solved from its stale price). Two tiers of repair:

        * **One side traded, the other stale** — reprice the stale leg from its
          live counterpart via put-call parity on the future::

              stale call  C = P_live + e^{-rT}(F - K)
              stale put   P = C_live - e^{-rT}(F - K)

          Overwrite ``close``, set ``synthetic_via_putcallparity = True``, and
          re-solve IV + greeks from the repaired price so ``delta`` is current.
        * **Both sides stale** (or no traded counterpart) — forward-fill IV per
          instrument and reprice ``close`` via Black-76 from the carried IV,
          setting ``is_synthetic = True``. Stale rows with no prior IV to carry
          forward are dropped.

        ``T`` uses the trading-day year fraction (``dte / 252``); options are
        priced on the future (``futures_price``).

        Args:
            df: Post-enrichment DataFrame (must carry ``iv``/greeks columns).

        Returns:
            Repaired copy with index reset.
        """
        if df.empty:
            return df.copy()

        from options.pricing.black76 import black76_greeks, black76_price_vec
        from options.pricing.iv_solver import solve_iv

        df = self._normalise_columns(df.copy())
        n_initial = len(df)

        # Ensure label columns exist.
        if "is_synthetic" not in df.columns:
            df["is_synthetic"] = False
        df["is_synthetic"] = df["is_synthetic"].fillna(False).astype(bool)
        df["synthetic_via_putcallparity"] = False

        # Instrument group column (for the both-stale forward-fill).
        if "instrument_key" in df.columns:
            group_col = "instrument_key"
        elif "instrument_token" in df.columns:
            group_col = "instrument_token"
        else:
            logger.warning(
                "repair_stale_prices: neither 'instrument_key' nor "
                "'instrument_token' found; grouping all rows together."
            )
            df["_grp"] = "all"
            group_col = "_grp"

        # Order by instrument then time so the Tier-2 ffill propagates correctly.
        df = df.sort_values([group_col, "timestamp"]).reset_index(drop=True)

        df["_stale"] = pd.to_numeric(df["volume"], errors="coerce").fillna(0) == 0
        if not df["_stale"].any():
            return df.drop(columns=["_stale", "_grp"], errors="ignore").reset_index(drop=True)

        itype   = df["instrument_type"].str.upper().str.strip()
        is_call = itype.isin({"CE", "CALL"}).to_numpy()
        is_put  = itype.isin({"PE", "PUT"}).to_numpy()

        pair_keys = (["timestamp", "expiry", "strike_price"]
                     if "expiry" in df.columns
                     else ["timestamp", "dte", "strike_price"])

        # One row per (pair): opposite leg's close + stale flag. merge how="left"
        # preserves df's (already time-sorted) row order.
        calls = (df.loc[is_call, pair_keys + ["close", "_stale"]]
                   .groupby(pair_keys, sort=False).first().reset_index()
                   .rename(columns={"close": "_ce_close", "_stale": "_ce_stale"}))
        puts  = (df.loc[is_put, pair_keys + ["close", "_stale"]]
                   .groupby(pair_keys, sort=False).first().reset_index()
                   .rename(columns={"close": "_pe_close", "_stale": "_pe_stale"}))
        df = df.merge(pd.merge(calls, puts, on=pair_keys, how="outer"),
                      on=pair_keys, how="left")

        F   = pd.to_numeric(df["futures_price"], errors="coerce").to_numpy(dtype=float)
        K   = pd.to_numeric(df["strike_price"], errors="coerce").to_numpy(dtype=float)
        dte = pd.to_numeric(df["dte"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        disc = np.exp(-self.r * np.maximum(dte / TRADING_DAYS_PER_YEAR, 0.0))
        stale = df["_stale"].to_numpy()

        # ── Tier 1: put-call parity (stale leg, live counterpart) ──────────────
        pe_live = df["_pe_stale"].eq(False).to_numpy() & df["_pe_close"].notna().to_numpy()
        ce_live = df["_ce_stale"].eq(False).to_numpy() & df["_ce_close"].notna().to_numpy()
        call_pcp = is_call & stale & pe_live
        put_pcp  = is_put  & stale & ce_live
        pcp_mask = call_pcp | put_pcp

        close    = df["close"].to_numpy(dtype=float).copy()
        pe_close = pd.to_numeric(df["_pe_close"], errors="coerce").to_numpy(dtype=float)
        ce_close = pd.to_numeric(df["_ce_close"], errors="coerce").to_numpy(dtype=float)
        close[call_pcp] = np.maximum(pe_close[call_pcp] + disc[call_pcp] * (F[call_pcp] - K[call_pcp]), 0.0)
        close[put_pcp]  = np.maximum(ce_close[put_pcp]  - disc[put_pcp]  * (F[put_pcp]  - K[put_pcp]),  0.0)
        df["close"] = close
        df.loc[pcp_mask, "synthetic_via_putcallparity"] = True
        n_pcp = int(pcp_mask.sum())

        # ── Tier 2: both stale (or no live counterpart) → ffill IV + Black-76 ──
        tier2 = stale & ~pcp_mask
        n_dropped = 0
        if tier2.any():
            iv_work = pd.to_numeric(df["iv"], errors="coerce")
            df["_ivw"] = iv_work.mask(pd.Series(tier2, index=df.index))  # blank stale IVs
            df["_ivw"] = df.groupby(group_col, sort=False)["_ivw"].transform(lambda s: s.ffill())

            tier2_ok   = tier2 & df["_ivw"].notna().to_numpy()
            tier2_drop = tier2 & df["_ivw"].isna().to_numpy()
            n_dropped  = int(tier2_drop.sum())

            if tier2_ok.any():
                iv_ok = df["_ivw"].to_numpy(dtype=float)
                T_ok  = np.maximum(dte / TRADING_DAYS_PER_YEAR, 1e-6)
                synth = black76_price_vec(
                    F=F[tier2_ok], K=K[tier2_ok], T=T_ok[tier2_ok],
                    sigma=iv_ok[tier2_ok], r=self.r, is_call=is_call[tier2_ok],
                )
                close = df["close"].to_numpy(dtype=float).copy()
                close[tier2_ok] = np.maximum(synth, 0.0)
                df["close"] = close
                df.loc[tier2_ok, "iv"] = iv_ok[tier2_ok]
                df.loc[tier2_ok, "is_synthetic"] = True

            if n_dropped:
                df = df[~tier2_drop].reset_index(drop=True)
            df = df.drop(columns=["_ivw"], errors="ignore")

        # ── Re-enrich every repaired row so IV + greeks match the new close ────
        repaired = (df["synthetic_via_putcallparity"] | df["is_synthetic"]).to_numpy()
        it2 = df["instrument_type"].str.upper().str.strip()
        for idx in df.index[repaired]:
            Kx = float(df.at[idx, "strike_price"]); Fx = float(df.at[idx, "futures_price"])
            Tx = year_fraction(float(df.at[idx, "dte"]))
            callx = it2.at[idx] in ("CE", "CALL")
            if bool(df.at[idx, "synthetic_via_putcallparity"]):
                px = float(df.at[idx, "close"])
                ivx = (solve_iv(market_price=px, F=Fx, K=Kx, T=Tx, r=self.r, is_call=callx)
                       if (Tx > 0 and px > 0 and Fx > 0 and Kx > 0) else None)
                df.at[idx, "iv"] = ivx
            else:
                ivx = df.at[idx, "iv"]
            if ivx is not None and not pd.isna(ivx) and Tx > 0:
                g = black76_greeks(F=Fx, K=Kx, T=Tx, sigma=float(ivx), r=self.r, is_call=callx)
                for c in ("delta", "gamma", "theta", "vega", "rho"):
                    if c in df.columns:
                        df.at[idx, c] = g.get(c)

        df = df.drop(
            columns=["_stale", "_grp", "_ce_close", "_ce_stale", "_pe_close", "_pe_stale"],
            errors="ignore",
        )
        logger.info(
            f"repair_stale_prices | {n_initial} → {len(df)} rows | "
            f"{n_pcp} put-call-parity, {int(df['is_synthetic'].sum())} ffill-synthetic | "
            f"{n_dropped} dropped (no prior IV)."
        )
        return df.reset_index(drop=True)

    # ══════════════════════════════════════════════════════════════════════════
    # Stage — Delta-band strike retention (post-solve)
    # ══════════════════════════════════════════════════════════════════════════

    def filter_delta_band(
        self,
        df: pd.DataFrame,
        itm_max: float | None = None,
        otm_min: float | None = None,
    ) -> pd.DataFrame:
        """Keep strikes whose ``|delta|`` lies within the retention band.

        Retains rows with ``otm_min <= |delta| <= itm_max`` (default 0.02–0.94):
        drops both the too-deep-ITM leg (``|delta| > itm_max``) and the
        too-far-OTM tail (``|delta| < otm_min``). Rows with **NaN delta** (IV did
        not solve, or greeks undefined at expiry) are **kept** so edge/expiry
        bars survive.

        Args:
            df: Enriched (and stale-repaired) DataFrame with a ``delta`` column.

        Returns:
            Filtered copy with index reset.
        """
        if df.empty:
            return df.copy()
        itm_max = self.ITM_DELTA_MAX if itm_max is None else itm_max
        otm_min = self.OTM_DELTA_MIN if otm_min is None else otm_min

        if "delta" not in df.columns:
            logger.warning("filter_delta_band: 'delta' column not found — skipping.")
            return df.copy()

        n_initial = len(df)
        ad = pd.to_numeric(df["delta"], errors="coerce").abs()
        keep = ad.isna() | ((ad >= otm_min) & (ad <= itm_max))
        out = df[keep].reset_index(drop=True)
        logger.info(
            f"filter_delta_band [{otm_min}, {itm_max}]: "
            f"{n_initial} → {len(out)} rows "
            f"(kept {int(ad.isna().sum())} NaN-delta)."
        )
        return out

    # ══════════════════════════════════════════════════════════════════════════
    # Internal helpers
    # ══════════════════════════════════════════════════════════════════════════

    @staticmethod
    def _normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
        """Rename aliased column names to the canonical names used here.

        Handles column aliases (e.g. ``strike``/``option_type``/``F``) from any
        upstream code that uses different field names.
        """
        alias_map: dict[str, str] = {
            "strike":      "strike_price",
            "option_type": "instrument_type",
            "F":           "futures_price",
        }
        to_rename = {k: v for k, v in alias_map.items() if k in df.columns}
        if to_rename:
            df = df.rename(columns=to_rename)
        return df
