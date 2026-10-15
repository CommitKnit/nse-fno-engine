"""ArcticDB storage — the canonical, versioned market-data store for the engine.

ArcticDB (LMDB backend) is the single store for every dataset in this repo. It gives
append-in-place, fast ``date_range`` slicing, and version history for reproducible
backtests.

This layer preserves the **source schemas verbatim**. It does NOT reshape option
chains into the engine's canonical MultiIndex chain schema
(``data/storage/schemas.py``) — that conversion still happens in the snapshot
builder, downstream of this store.

Layout (LMDB at ``settings.ARCTICDB_DIR``)::

    library            symbol                                    source today
    ----------------   ---------------------------------------   ----------------------------------------
    futures            {symbol}/{interval}                       F1/F2/F3 wide panel, futures columns only (slim:
                                                                   SPOT_* joined from `spot` by read_futures)
    futures_contracts  {symbol}/{interval}/{expiry}              raw per-contract OHLCV+OI the panels are built from
    options_enriched   {symbol}/{interval}/{expiry}              enriched chain (one symbol per expiry; fetch_options.py)
    options_raw        {symbol}/{interval}/{expiry}              raw option candles before IV/Greeks
    borrow_rates       {symbol}/{interval}                       computed columns only (slim: futures inputs and
                                                                   SPOT_close joined by read_borrow_rates)
    borrow_rates       {symbol}/{interval}/cycle_summary
    spot_upstox        {symbol}/{interval}                       Upstox spot archive (reads use Kite `spot`)
    spot               {symbol}/{interval}                       flat NSE_EQ OHLCV series (open/high/low/close/volume);
                                                                   only Kite-written symbols (metadata source
                                                                   'fetch_kite*') are ever joined as SPOT_*
    spot_adjusted      {symbol}/{interval}                       DERIVED from `spot`: dividend back-adjusted OHLC
                                                                   (+ close_raw, adj_factor). Splits and bonuses
                                                                   arrive ALREADY adjusted from Kite; only cash
                                                                   dividends are applied here. Full rewrite only
                                                                   -- never appended (see spot_adjusted_symbol)
    factors            {name}/{interval}                         market/sector factor returns -- one symbol per
                                                                   NAMED series (e.g. "MARKET", "EX_TATASTEEL"),
                                                                   never a wide multi-ticker table (see module note)
    fundamentals       {symbol}/{statement}/{type}/{time_period}  income_statement/balance_sheet/cash_flow tidy rows
    residuals          {symbol}/{model_version}/{interval}        per-stock factor-model residuals; model_version
                                                                   lets the spec evolve without touching old runs
    signals            {symbol}/{model_version}/{interval}        per-stock rank/bucket/target-weight time series
    corporate_actions  {symbol}                                   per-company event log (splits, bonuses, OFS, ...);
                                                                   no interval -- these are events, not sampled bars
    metadata           contract_master | fetch_log               metadata.db tables

ALL of the above are ticker-wise (one symbol = one instrument's own full time
series across its fields), by deliberate choice -- never field-wise (one
symbol = one field, tickers as columns). Ticker-wise keeps writes, corrections
and version history isolated per instrument: fixing one stock's data touches
exactly one symbol, and a fetch failure on one name never blocks or corrupts
another. A wide cross-section (e.g. every stock's residual on one date, for
ranking) is built by reading the relevant per-ticker symbols and pivoting at
query time -- store a *derived* wide cache under a synthetic name in
``factors``/``signals`` if that pivot is expensive enough to want to reuse,
but the per-ticker symbols stay the source of truth.

``cycle_id`` is kept as a queryable column on the continuous futures / borrow
symbols; per-cycle slices are derived with a ``QueryBuilder`` filter rather than
stored as separate symbols.

Usage::

    store = ArcticStore()
    store.write_futures("RVNL", "15minute", df)
    df = store.read_futures("RVNL", "15minute", date_range=(t0, t1))

    from arcticdb import QueryBuilder
    q = QueryBuilder(); q = q[q["cycle_id"] == 7]
    cycle7 = store.read_futures("RVNL", "15minute", query_builder=q)
"""

from __future__ import annotations

import re
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from arcticdb import Arctic
from loguru import logger

from config.settings import settings

# ── Library names ──
LIB_FUTURES = "futures"
LIB_OPTIONS = "options_enriched"
LIB_BORROW = "borrow_rates"
LIB_SPOT = "spot"
LIB_SPOT_ADJUSTED = "spot_adjusted"
LIB_METADATA = "metadata"
LIB_FACTORS = "factors"
LIB_FUNDAMENTALS = "fundamentals"
LIB_RESIDUALS = "residuals"
LIB_SIGNALS = "signals"
LIB_CORPORATE_ACTIONS = "corporate_actions"
# NSE index OHLC (Nifty 50, sectorals, ...). Deliberately NOT in ``spot``:
# that library is the Nifty 500 equity universe, and several scripts build
# their universe by globbing every symbol in it. An index landing there
# would silently enter cross-sectional work as if it were a constituent.
LIB_INDICES = "indices"
# NSE corporate announcements (press releases / investor-presentation PDF URLs),
# symbols ``{TICKER}/investor_presentations``.
LIB_ANNOUNCEMENTS = "announcements"
_ALL_LIBS = (
    LIB_FUTURES, LIB_OPTIONS, LIB_BORROW, LIB_SPOT, LIB_SPOT_ADJUSTED, LIB_METADATA,
    LIB_FACTORS, LIB_FUNDAMENTALS, LIB_RESIDUALS, LIB_SIGNALS, LIB_CORPORATE_ACTIONS,
    LIB_INDICES, LIB_ANNOUNCEMENTS,
)

# Date range is a (start, end) pair of timestamps; either side may be None for
# an open-ended bound, matching ArcticDB's ``read(date_range=...)`` contract.
DateRange = tuple[pd.Timestamp | None, pd.Timestamp | None]


def ensure_dt_index(df: pd.DataFrame, col: str = "timestamp") -> pd.DataFrame:
    """Return ``df`` with a sorted ``DatetimeIndex``.

    If the index is not already datetime-like but a ``col`` column exists, that
    column is promoted to the index. The result is sorted ascending (stable) so
    ArcticDB ``append`` / ``date_range`` operations work. Non-unique indices
    (e.g. one row per strike per bar in option chains) are allowed.
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        if col in df.columns:
            df = df.set_index(col)
        else:
            raise ValueError(
                f"ensure_dt_index: frame has no DatetimeIndex and no '{col}' "
                f"column; columns={list(df.columns)[:10]}"
            )
    if not df.index.is_monotonic_increasing:
        df = df.sort_index(kind="stable")
    return df


# Derived columns computed from ``close``. Raw ``spot`` carries LOG_COLUMNS only
# on the 1-minute intervals, which have no ``spot_adjusted`` counterpart.
# ``spot_adjusted`` carries SPOT_ADJUSTED_LOG_COLUMNS on every interval,
# computed from the ADJUSTED close (see data/dividend_adjustment.py) and
# suffixed so they can never be mistaken for raw-price returns.
LOG_COLUMNS = ("log_price", "log_return", "corp_action_flag")
SPOT_LOG_INTERVALS = frozenset({"minute", "1minute"})
SPOT_ADJUSTED_SUFFIX = "_spot_adjusted"
SPOT_ADJUSTED_LOG_COLUMNS = (
    f"log_price{SPOT_ADJUSTED_SUFFIX}", f"log_return{SPOT_ADJUSTED_SUFFIX}", "corp_action_flag",
)

# A |log return| this large in one bar is an unadjusted corporate action
# (demerger etc. -- neither Kite nor Upstox back-adjusts those), not a move.
CORP_ACTION_LOGRET_THRESHOLD = 0.5

# ── Slim layout: every value is stored exactly once ──────────────────────────
# Spot lives only in the (Kite) ``spot`` library, futures columns only in the
# ``futures`` wide panel, and ``borrow_rates`` keeps only what it computes.
# Writes strip the copies below; ``read_futures`` / ``read_borrow_rates`` join
# them back at read time, so readers still see the familiar wide columns.
FUTURES_SPOT_COLUMNS = (
    "SPOT_open", "SPOT_high", "SPOT_low", "SPOT_close", "SPOT_volume", "SPOT_ticker", "spot_missing",
)
BORROW_COPIED_COLUMNS = (
    "F1_close", "F2_close", "F3_close", "SPOT_close",
    "F1_dte", "F2_dte", "F3_dte", "F1_oi", "F2_oi", "F3_oi",
    "cycle_id", "is_expiry_day", "F1_volume", "F2_volume",
)
# Any raw futures-panel field (F1_close, F2_expiry, …; not derived ones like F1_dte_star).
_PANEL_FIELD = re.compile(r"^F[123]_(open|high|low|close|volume|oi|dte|expiry|ticker)$")
# Futures panels use Upstox interval names, Kite spot uses its own for two of them.
SPOT_INTERVAL_ALIASES = {"1day": "day", "1minute": "minute", "1hour": "60minute"}


def flag_corp_actions(log_return: pd.Series) -> pd.Series:
    """Boolean ``corp_action_flag``: ``|log_return| > 0.5``; NaN counts as False."""
    return (log_return.abs() > CORP_ACTION_LOGRET_THRESHOLD).astype(bool)


def add_log_columns(
    df: pd.DataFrame, prev_close: float | None = None, suffix: str = ""
) -> pd.DataFrame:
    """Return a copy of *df* with ``log_price{suffix} = ln(close)``,
    ``log_return{suffix} = diff(log_price)`` and ``corp_action_flag``.

    ``prev_close`` is the stored close of the bar immediately before ``df``'s
    first row; pass it when ``df`` continues an existing series so the first
    row's return spans the join instead of being NaN.
    """
    lp, lr = f"log_price{suffix}", f"log_return{suffix}"
    out = df.copy()
    close = out["close"].astype("float64")
    out[lp] = np.log(close.where(close > 0))   # bad print -> NaN, not -inf
    out[lr] = out[lp].diff()
    if prev_close is not None and len(out):
        out.iloc[0, out.columns.get_loc(lr)] = out[lp].iloc[0] - np.log(float(prev_close))
    out["corp_action_flag"] = flag_corp_actions(out[lr])
    return out


class ArcticStore:
    """Read/write versioned market data to ArcticDB (LMDB backend).

    A single connection object, loguru logging, and dataset-typed convenience
    methods on top of a small generic core (``write`` / ``append`` / ``read``).
    """

    # Libraries big enough to need a bigger LMDB map_size than everything else
    # -- see the constructor docstring/comment below for why they get their own
    # physical path instead of a bump to the shared default.
    _BIG_LIBS = frozenset({LIB_SPOT, LIB_SPOT_ADJUSTED})

    def __init__(self, uri: str | None = None, spot_uri: str | None = None):
        # `spot` and `spot_adjusted` live at their own physical LMDB path with a
        # larger map_size -- see settings.ARCTICDB_SPOT_URI. NOT a second
        # connection over the SAME path as `self.arctic`: LMDB explicitly does
        # not support one path being opened twice in a process (ArcticDB warns
        # loudly if you try, and it is not a warning to route around --
        # unverified behaviour under a multi-hour unattended write is not a risk
        # worth taking). Also: on this Windows/NTFS build, opening a library
        # EAGERLY allocates real disk equal to whatever map_size is requested,
        # regardless of actual content (verified 2026-09-20) -- so these two
        # libraries' bigger ceiling must stay isolated to its own path, or every
        # other library would grow right along with them.
        #
        # Only split them out when connecting to the DEFAULT store (or when a
        # caller explicitly opts in via `spot_uri`). A caller that passes a
        # custom `uri` -- e.g. index_rebalancing/store.py's own, entirely
        # separate LMDB root -- gets the pre-existing behaviour untouched: both
        # stay inside that same custom connection, because defaulting to THIS
        # project's dedicated path would silently point a different project's
        # `spot`/`spot_adjusted` at the wrong data.
        using_default_uri = uri is None
        self.uri = uri or settings.ARCTICDB_URI
        if spot_uri is not None:
            self.spot_uri = spot_uri
        elif using_default_uri:
            self.spot_uri = settings.ARCTICDB_SPOT_URI
        else:
            self.spot_uri = self.uri  # custom store: keep everything in one connection

        # LMDB needs its backing directory to exist before we connect. Both URIs
        # may carry a `?map_size=...` query string, so strip that off before
        # mkdir -- otherwise it's treated as part of the literal path and mkdir
        # fails with an invalid-name error.
        def _mkdir_for(u: str) -> None:
            if u.startswith("lmdb://"):
                Path(u[len("lmdb://"):].split("?", 1)[0]).mkdir(parents=True, exist_ok=True)

        _mkdir_for(self.uri)
        if self.spot_uri != self.uri:
            _mkdir_for(self.spot_uri)
        self.arctic = Arctic(self.uri)
        # Same physical path as the main connection -> reuse it (LMDB forbids
        # opening one path twice in a process); different path -> its own connection.
        self._spot_arctic = self.arctic if self.spot_uri == self.uri else Arctic(self.spot_uri)

        self._libs: dict[str, Any] = {
            name: (self._spot_arctic if name in self._BIG_LIBS else self.arctic).get_library(
                name, create_if_missing=True
            )
            for name in _ALL_LIBS
        }
        logger.debug(f"ArcticStore connected: {self.uri}  (spot: {self.spot_uri})")

    # ── internals ──────────────────────────────────────────────────────────

    def _lib(self, library: str):
        lib = self._libs.get(library)
        if lib is None:  # allow ad-hoc libraries beyond the defaults
            lib = self.arctic.get_library(library, create_if_missing=True)
            self._libs[library] = lib
        return lib

    @staticmethod
    def _sort_if_dt(df: pd.DataFrame) -> pd.DataFrame:
        if isinstance(df.index, pd.DatetimeIndex) and not df.index.is_monotonic_increasing:
            return df.sort_index(kind="stable")
        return df

    @staticmethod
    def _wants_spot_logs(library: str, symbol: str, df: pd.DataFrame) -> bool:
        """True for a raw OHLCV frame bound for a 1-minute ``spot`` symbol that
        the caller did not already give log columns."""
        return (
            library == LIB_SPOT
            and symbol.rsplit("/", 1)[-1] in SPOT_LOG_INTERVALS
            and isinstance(df.index, pd.DatetimeIndex)
            and "close" in df.columns
            and not set(LOG_COLUMNS) & set(df.columns)
        )

    @staticmethod
    def _slim(library: str, symbol: str, df: pd.DataFrame) -> pd.DataFrame:
        """Drop columns that belong to another library (see FUTURES_SPOT_COLUMNS /
        BORROW_COPIED_COLUMNS), so no write can re-introduce a stored copy."""
        if library == LIB_FUTURES:
            drop = [c for c in FUTURES_SPOT_COLUMNS if c in df.columns]
        elif library == LIB_BORROW and not symbol.endswith("/cycle_summary"):
            drop = [c for c in df.columns
                    if c in BORROW_COPIED_COLUMNS or c in FUTURES_SPOT_COLUMNS
                    or _PANEL_FIELD.match(str(c))]
        else:
            return df
        if drop:
            logger.debug(f"[{library}] {symbol}: not storing copied columns {drop}")
            df = df.drop(columns=drop)
        return df

    @staticmethod
    def _stored_columns(lib, symbol: str) -> list[str]:
        return [c.name for c in lib.get_description(symbol).columns]

    @staticmethod
    def _expiry_str(expiry: str | date) -> str:
        if isinstance(expiry, date):
            return expiry.isoformat()
        return str(expiry)[:10]

    # ── symbol naming ────────────────────────────────────────────────────────

    @staticmethod
    def futures_symbol(symbol: str, interval: str) -> str:
        return f"{symbol}/{interval}"

    @staticmethod
    def spot_symbol(symbol: str, interval: str) -> str:
        return f"{symbol}/{interval}"

    @staticmethod
    def spot_adjusted_symbol(symbol: str, interval: str) -> str:
        """Same shape as ``spot_symbol`` on purpose -- ``spot_adjusted`` is a
        drop-in replacement for ``spot``, same symbols and same OHLCV column
        names, with cash dividends back-adjusted out of the history."""
        return f"{symbol}/{interval}"

    @classmethod
    def options_symbol(cls, symbol: str, interval: str, expiry: str | date) -> str:
        return f"{symbol}/{interval}/{cls._expiry_str(expiry)}"

    @staticmethod
    def borrow_symbol(symbol: str, interval: str) -> str:
        return f"{symbol}/{interval}"

    @staticmethod
    def cycle_summary_symbol(symbol: str, interval: str) -> str:
        return f"{symbol}/{interval}/cycle_summary"

    @staticmethod
    def indices_symbol(name: str, interval: str) -> str:
        """``name`` is the index's canonical short name (e.g. ``"NIFTY50"``,
        ``"NIFTYBANK"``), not Kite's spaced tradingsymbol."""
        return f"{name}/{interval}"

    @staticmethod
    def factors_symbol(name: str, interval: str) -> str:
        """``name`` is the factor's own identity, e.g. ``"MARKET"`` or
        ``"EX_TATASTEEL"`` (Nifty Metal with Tata Steel's own contribution
        backed out) -- NOT necessarily a real tradeable ticker. Still one
        symbol per named series, never a wide multi-name table."""
        return f"{name}/{interval}"

    @classmethod
    def fundamentals_symbol(
        cls, symbol: str, statement: str, statement_type: str, time_period: str
    ) -> str:
        """``statement``: income_statement|balance_sheet|cash_flow.
        ``statement_type``: consolidated|standalone. ``time_period``: yearly|quarterly.
        Matches the tidy long-format rows ``fetch_income_statements.py`` produces."""
        return f"{symbol}/{statement}/{statement_type}/{time_period}"

    @staticmethod
    def residuals_symbol(symbol: str, model_version: str, interval: str) -> str:
        """``model_version`` (e.g. ``"v1"``) isolates a factor-model spec change
        to its own symbol -- improving the model never overwrites or disturbs
        an older version's stored residuals."""
        return f"{symbol}/{model_version}/{interval}"

    @staticmethod
    def signals_symbol(symbol: str, model_version: str, interval: str) -> str:
        """Per-stock rank/bucket/target-weight time series, keyed the same way
        as ``residuals_symbol`` -- one symbol per ticker per model version, NOT
        one wide table of the whole universe's signal on a given date. Build
        the cross-section by reading the relevant per-ticker symbols and
        pivoting at query/rebalance time."""
        return f"{symbol}/{model_version}/{interval}"

    @staticmethod
    def corporate_actions_symbol(symbol: str) -> str:
        """No interval -- corporate actions are discrete events, not sampled bars."""
        return symbol

    # ── generic core ─────────────────────────────────────────────────────────

    def write(
        self,
        library: str,
        symbol: str,
        df: pd.DataFrame,
        *,
        metadata: dict | None = None,
        prune_previous: bool = False,
    ) -> str:
        """Full (over)write of a symbol — creates a new version.

        ``prune_previous=True`` drops older versions (used by the one-time
        backfill so re-runs don't accumulate dead versions).

        1-minute ``spot`` frames get ``LOG_COLUMNS`` added (see
        ``SPOT_LOG_INTERVALS``).
        """
        df = self._slim(library, symbol, self._sort_if_dt(df))
        if self._wants_spot_logs(library, symbol, df):
            df = add_log_columns(df)
        self._lib(library).write(
            symbol, df, metadata=metadata, prune_previous_versions=prune_previous
        )
        logger.debug(f"[{library}] wrote {len(df)} rows -> {symbol}")
        return symbol

    def append(self, library: str, symbol: str, df: pd.DataFrame,
               *, metadata: dict | None = None) -> int:
        """Append only rows strictly newer than the existing tail (dedup-safe).

        Writes the symbol fresh if it does not exist yet. Returns the number of
        rows actually appended.

        ArcticDB stamps each appended version with only the metadata passed to
        that call, so ``metadata=None`` carries the stored symbol's metadata
        forward (otherwise e.g. the Kite ``source`` stamp that ``spot_symbol_for``
        requires would be lost). Pass a dict to replace it.

        On 1-minute ``spot``, if the stored symbol carries ``LOG_COLUMNS``, they are
        computed for the new rows too, the first one chained off the stored
        tail's close (otherwise the schema mismatch would reject the append).
        """
        lib = self._lib(library)
        df = self._slim(library, symbol, self._sort_if_dt(df))
        if not lib.has_symbol(symbol):
            if self._wants_spot_logs(library, symbol, df):
                df = add_log_columns(df)
            lib.write(symbol, df, metadata=metadata)
            logger.debug(f"[{library}] wrote (new) {len(df)} rows -> {symbol}")
            return len(df)
        tail = None
        if isinstance(df.index, pd.DatetimeIndex):
            tail = lib.tail(symbol, 1).data
            if len(tail):
                last_ts = tail.index.max()
                df = df[df.index > last_ts]
        if len(df) == 0:
            logger.debug(f"[{library}] nothing new to append -> {symbol}")
            return 0
        if self._wants_spot_logs(library, symbol, df):
            stored = self._stored_columns(lib, symbol)
            if set(stored) == set(df.columns) | set(LOG_COLUMNS):
                prev = tail["close"].iloc[-1] if tail is not None and len(tail) else None
                df = add_log_columns(df, prev)[stored]  # static schema: stored order
        lib.append(symbol, df, metadata=self._carry_metadata(lib, symbol, metadata))
        logger.debug(f"[{library}] appended {len(df)} rows -> {symbol}")
        return len(df)

    def update(self, library: str, symbol: str, df: pd.DataFrame,
               *, metadata: dict | None = None) -> int:
        """Upsert: replace any overlapping date range, insert the rest.

        Use for incremental fetches that may re-fetch the current (partial) bar.

        Like ``append``, ``metadata=None`` carries the stored metadata forward.

        On 1-minute ``spot`` symbols carrying ``LOG_COLUMNS``, the new rows get log
        columns chained off the stored bar just before them, and the stored bar
        just after them (if any) is re-emitted with its ``log_return`` rebased
        onto the new close.
        """
        lib = self._lib(library)
        df = self._slim(library, symbol, self._sort_if_dt(df))
        if not lib.has_symbol(symbol):
            if self._wants_spot_logs(library, symbol, df):
                df = add_log_columns(df)
            lib.write(symbol, df, metadata=metadata)
        else:
            if len(df) and self._wants_spot_logs(library, symbol, df):
                stored = self._stored_columns(lib, symbol)
                if set(stored) == set(df.columns) | set(LOG_COLUMNS):
                    one_ns = pd.Timedelta(1, "ns")
                    first, last = df.index.min(), df.index.max()
                    before = lib.read(symbol, date_range=(None, first - one_ns),
                                      columns=["close"]).data
                    df = add_log_columns(df, before["close"].iloc[-1] if len(before) else None)
                    after = lib.read(symbol, date_range=(last + one_ns, None)).data.head(1)
                    if len(after):
                        after = after.copy()
                        after["log_return"] = after["log_price"] - df["log_price"].iloc[-1]
                        after["corp_action_flag"] = flag_corp_actions(after["log_return"])
                        df = pd.concat([df, after[df.columns]])
                    df = df[stored]  # static schema: stored order
            lib.update(symbol, df, metadata=self._carry_metadata(lib, symbol, metadata))
        logger.debug(f"[{library}] upserted {len(df)} rows -> {symbol}")
        return len(df)

    @staticmethod
    def _carry_metadata(lib, symbol: str, metadata: dict | None) -> dict | None:
        """*metadata* if given, else the stored symbol's current metadata."""
        return metadata if metadata is not None else lib.read_metadata(symbol).metadata

    def read(
        self,
        library: str,
        symbol: str,
        *,
        date_range: DateRange | None = None,
        columns: list[str] | None = None,
        query_builder: Any | None = None,
    ) -> pd.DataFrame:
        """Read a symbol (latest version). Returns an empty frame if absent."""
        lib = self._lib(library)
        if not lib.has_symbol(symbol):
            return pd.DataFrame()
        return lib.read(
            symbol,
            date_range=date_range,
            columns=columns,
            query_builder=query_builder,
        ).data

    def has_symbol(self, library: str, symbol: str) -> bool:
        return self._lib(library).has_symbol(symbol)

    def list_symbols(self, library: str) -> list[str]:
        return sorted(self._lib(library).list_symbols())

    def stored_date_span(
        self, library: str, symbol: str
    ) -> tuple[pd.Timestamp, pd.Timestamp] | None:
        """(first, last) index timestamps already stored for *symbol*, or ``None``.

        Backed by ``Library.get_description``, which reads index metadata only
        (no data pages) — this is the cheap coverage check fetch pipelines use
        to decide what's already covered before hitting the API, e.g.
        ``fetch_options.py``'s incremental skip logic. Returns ``None`` when the
        symbol doesn't exist or is empty (``row_count == 0``), so callers can
        treat that uniformly as "nothing stored yet".
        """
        lib = self._lib(library)
        if not lib.has_symbol(symbol):
            return None
        desc = lib.get_description(symbol)
        if not desc.row_count or desc.date_range is None:
            return None
        lo, hi = desc.date_range
        if lo is None or hi is None:
            return None
        return pd.Timestamp(lo), pd.Timestamp(hi)

    def read_metadata(self, library: str, symbol: str) -> dict | None:
        lib = self._lib(library)
        if not lib.has_symbol(symbol):
            return None
        return lib.read_metadata(symbol).metadata

    def delete(self, library: str, symbol: str) -> None:
        lib = self._lib(library)
        if lib.has_symbol(symbol):
            lib.delete(symbol)
            logger.debug(f"[{library}] deleted -> {symbol}")

    # ── typed: futures (continuous wide panel) ────────────────────────────────

    def write_futures(self, symbol: str, interval: str, df: pd.DataFrame,
                      *, metadata: dict | None = None, prune_previous: bool = False) -> str:
        return self.write(LIB_FUTURES, self.futures_symbol(symbol, interval), df,
                          metadata=metadata, prune_previous=prune_previous)

    def append_futures(self, symbol: str, interval: str, df: pd.DataFrame,
                       *, metadata: dict | None = None) -> int:
        return self.append(LIB_FUTURES, self.futures_symbol(symbol, interval), df, metadata=metadata)

    def read_futures(self, symbol: str, interval: str, *, date_range: DateRange | None = None,
                     columns: list[str] | None = None, query_builder: Any | None = None,
                     with_spot: bool = True) -> pd.DataFrame:
        """Read a futures symbol; F1/F2/F3 wide panels get Kite spot joined as ``SPOT_*``.

        Panels store futures columns only (slim layout). ``with_spot=True`` attaches
        ``SPOT_open/high/low/close/volume/ticker`` and ``spot_missing`` from the
        ``spot`` library at read time, so callers see the familiar wide panel.
        """
        want_spot = with_spot and (columns is None or any(c in FUTURES_SPOT_COLUMNS for c in columns))
        fut_cols = None if columns is None else [c for c in columns if c not in FUTURES_SPOT_COLUMNS]
        df = self.read(LIB_FUTURES, self.futures_symbol(symbol, interval),
                       date_range=date_range, columns=fut_cols or None, query_builder=query_builder)
        if want_spot and len(df) and self._is_wide_panel(symbol, interval):
            df = self._attach_spot(symbol, interval, df)
            if columns is not None:
                df = df[[c for c in columns if c in df.columns]]
        return df

    def _is_wide_panel(self, symbol: str, interval: str) -> bool:
        sym = self.futures_symbol(symbol, interval)
        lib = self._lib(LIB_FUTURES)
        return lib.has_symbol(sym) and "F1_close" in self._stored_columns(lib, sym)

    def spot_symbol_for(self, symbol: str, interval: str) -> str | None:
        """The Kite ``spot`` symbol matching a futures interval (``1day`` -> ``day``,
        ``1minute`` -> ``minute``), or ``None`` if no Kite series is stored.

        Only series written by the Kite fetchers (symbol metadata ``source`` starting
        with ``fetch_kite``) qualify. Older Upstox series in the same library (e.g.
        ``SBICARD/1minute``, ``NIFTY/15minute``, written by the old fetch_options.py
        spot stage) carry no such metadata and are never used, so every spot-dependent
        value (borrow rates, ``SPOT_*`` joins, snapshot spot bars) is Kite spot or NaN.
        """
        lib = self._lib(LIB_SPOT)
        for iv in (SPOT_INTERVAL_ALIASES.get(interval), interval):
            if iv and self._is_kite_spot(lib, self.spot_symbol(symbol, iv)):
                return self.spot_symbol(symbol, iv)
        return None

    def _is_kite_spot(self, lib, sym: str) -> bool:
        """True when ``spot/{sym}`` exists and was written by a Kite fetcher (cached)."""
        cache = self.__dict__.setdefault("_kite_spot_cache", {})
        if sym not in cache:
            ok = False
            if lib.has_symbol(sym):
                md = lib.read_metadata(sym).metadata or {}
                ok = str(md.get("source", "")).startswith("fetch_kite")
            cache[sym] = ok
        return cache[sym]

    def _attach_spot(self, symbol: str, interval: str, df: pd.DataFrame) -> pd.DataFrame:
        """Join Kite spot OHLCV onto *df*'s timestamps as ``SPOT_*`` + ``spot_missing``."""
        sym = self.spot_symbol_for(symbol, interval)
        cols = ["open", "high", "low", "close", "volume"]
        spot = pd.DataFrame(columns=cols)
        if sym is not None:
            spot = self._lib(LIB_SPOT).read(
                sym, date_range=(df.index.min(), df.index.max()), columns=cols).data
            spot = spot[~spot.index.duplicated(keep="last")]
        spot = spot.reindex(df.index)
        out = df.copy()
        for c in cols:
            out[f"SPOT_{c}"] = spot[c].astype("float64")
        out["SPOT_volume"] = out["SPOT_volume"].fillna(0)
        out["SPOT_ticker"] = symbol if sym is not None else ""
        out["spot_missing"] = out["SPOT_close"].isna()
        return out

    # ── typed: spot (flat NSE_EQ OHLCV series) ─────────────────────────────────

    def write_spot(self, symbol: str, interval: str, df: pd.DataFrame,
                   *, metadata: dict | None = None, prune_previous: bool = False) -> str:
        return self.write(LIB_SPOT, self.spot_symbol(symbol, interval), df,
                          metadata=metadata, prune_previous=prune_previous)

    def append_spot(self, symbol: str, interval: str, df: pd.DataFrame,
                    *, metadata: dict | None = None) -> int:
        return self.append(LIB_SPOT, self.spot_symbol(symbol, interval), df, metadata=metadata)

    def read_spot(self, symbol: str, interval: str, *, date_range: DateRange | None = None,
                  columns: list[str] | None = None, query_builder: Any | None = None) -> pd.DataFrame:
        return self.read(LIB_SPOT, self.spot_symbol(symbol, interval),
                         date_range=date_range, columns=columns, query_builder=query_builder)

    # ── typed: NSE indices ───────────────────────────────────────────────────

    def write_indices(self, name: str, interval: str, df: pd.DataFrame,
                      *, metadata: dict | None = None,
                      prune_previous: bool = False) -> str:
        return self.write(LIB_INDICES, self.indices_symbol(name, interval), df,
                          metadata=metadata, prune_previous=prune_previous)

    def append_indices(self, name: str, interval: str, df: pd.DataFrame,
                       *, metadata: dict | None = None) -> int:
        return self.append(LIB_INDICES, self.indices_symbol(name, interval), df, metadata=metadata)

    def read_indices(self, name: str, interval: str, *,
                     date_range: DateRange | None = None,
                     columns: list[str] | None = None,
                     query_builder: Any | None = None) -> pd.DataFrame:
        return self.read(LIB_INDICES, self.indices_symbol(name, interval),
                         date_range=date_range, columns=columns,
                         query_builder=query_builder)

    # ── typed: dividend back-adjusted spot (derived from ``spot``) ────────────
    #
    # There is deliberately NO ``append_spot_adjusted``. Every bar's adjustment
    # factor is a product over all *future* dividends, so one new ex-date
    # rescales the entire history -- ``append`` and ``update`` cannot express
    # "rescale everything before t". Full ``write`` only; omitting the method
    # stops anyone reaching for the wrong one.

    def write_spot_adjusted(self, symbol: str, interval: str, df: pd.DataFrame,
                            *, metadata: dict | None = None,
                            prune_previous: bool = False) -> str:
        return self.write(LIB_SPOT_ADJUSTED, self.spot_adjusted_symbol(symbol, interval), df,
                          metadata=metadata, prune_previous=prune_previous)

    def read_spot_adjusted(self, symbol: str, interval: str, *,
                           date_range: DateRange | None = None,
                           columns: list[str] | None = None,
                           query_builder: Any | None = None) -> pd.DataFrame:
        return self.read(LIB_SPOT_ADJUSTED, self.spot_adjusted_symbol(symbol, interval),
                         date_range=date_range, columns=columns, query_builder=query_builder)

    # ── typed: enriched options (one symbol per expiry) ───────────────────────

    def write_options(self, symbol: str, interval: str, expiry: str | date, df: pd.DataFrame,
                      *, metadata: dict | None = None, prune_previous: bool = False) -> str:
        return self.write(LIB_OPTIONS, self.options_symbol(symbol, interval, expiry), df,
                          metadata=metadata, prune_previous=prune_previous)

    def append_options(self, symbol: str, interval: str, expiry: str | date, df: pd.DataFrame) -> int:
        return self.append(LIB_OPTIONS, self.options_symbol(symbol, interval, expiry), df)

    def read_options(self, symbol: str, interval: str, expiry: str | date,
                     *, date_range: DateRange | None = None, columns: list[str] | None = None,
                     query_builder: Any | None = None) -> pd.DataFrame:
        return self.read(LIB_OPTIONS, self.options_symbol(symbol, interval, expiry),
                         date_range=date_range, columns=columns, query_builder=query_builder)

    def list_option_expiries(self, symbol: str, interval: str) -> list[str]:
        """Expiry strings present for a symbol+interval (sorted)."""
        prefix = f"{symbol}/{interval}/"
        out = []
        for s in self.list_symbols(LIB_OPTIONS):
            if s.startswith(prefix):
                tail = s[len(prefix):]
                if tail != "cycle_summary":
                    out.append(tail)
        return sorted(out)

    # ── typed: borrow rates ───────────────────────────────────────────────────

    def write_borrow_rates(self, symbol: str, interval: str, df: pd.DataFrame,
                           *, metadata: dict | None = None, prune_previous: bool = False) -> str:
        return self.write(LIB_BORROW, self.borrow_symbol(symbol, interval), df,
                          metadata=metadata, prune_previous=prune_previous)

    def append_borrow_rates(self, symbol: str, interval: str, df: pd.DataFrame) -> int:
        return self.append(LIB_BORROW, self.borrow_symbol(symbol, interval), df)

    def read_borrow_rates(self, symbol: str, interval: str, *, date_range: DateRange | None = None,
                          columns: list[str] | None = None, query_builder: Any | None = None,
                          joined: bool = True) -> pd.DataFrame:
        """Read a borrow-rate series.

        ``borrow_rates`` stores only computed columns (slim layout). With
        ``joined=True`` the futures-panel inputs (F1–F3 close/OI/DTE, cycle_id,
        is_expiry_day, F1/F2 volume) and Kite ``SPOT_close`` are joined back on
        the same timestamps, giving the familiar full frame. ``query_builder``
        filters on futures-panel columns (e.g. ``cycle_id``) when joined.
        """
        sym = self.borrow_symbol(symbol, interval)
        if not joined:
            return self.read(LIB_BORROW, sym, date_range=date_range, columns=columns,
                             query_builder=query_builder)
        br = self.read(LIB_BORROW, sym, date_range=date_range)
        if br.empty:
            return br
        inputs = [c for c in BORROW_COPIED_COLUMNS if c != "SPOT_close"]
        fut = self.read_futures(symbol, interval, date_range=(br.index.min(), br.index.max()),
                                query_builder=query_builder, with_spot=False)
        have = [c for c in inputs if c in fut.columns]
        out = fut[have].join(br, how="inner" if query_builder is not None else "right")
        out = self._attach_spot(symbol, interval, out)[
            have[:3] + ["SPOT_close"] + have[3:] + list(br.columns)]
        if columns is not None:
            out = out[[c for c in columns if c in out.columns]]
        return out

    def write_cycle_summary(self, symbol: str, interval: str, df: pd.DataFrame,
                            *, metadata: dict | None = None, prune_previous: bool = False) -> str:
        return self.write(LIB_BORROW, self.cycle_summary_symbol(symbol, interval), df,
                          metadata=metadata, prune_previous=prune_previous)

    def read_cycle_summary(self, symbol: str, interval: str) -> pd.DataFrame:
        return self.read(LIB_BORROW, self.cycle_summary_symbol(symbol, interval))

    # ── typed: factors (market/sector, one symbol per named series) ───────────

    def write_factor(self, name: str, interval: str, df: pd.DataFrame,
                     *, metadata: dict | None = None, prune_previous: bool = False) -> str:
        return self.write(LIB_FACTORS, self.factors_symbol(name, interval), df,
                          metadata=metadata, prune_previous=prune_previous)

    def append_factor(self, name: str, interval: str, df: pd.DataFrame) -> int:
        return self.append(LIB_FACTORS, self.factors_symbol(name, interval), df)

    def read_factor(self, name: str, interval: str, *, date_range: DateRange | None = None,
                    columns: list[str] | None = None, query_builder: Any | None = None) -> pd.DataFrame:
        return self.read(LIB_FACTORS, self.factors_symbol(name, interval),
                         date_range=date_range, columns=columns, query_builder=query_builder)

    # ── typed: fundamentals (per-company financial-statement rows) ────────────

    def write_fundamentals(self, symbol: str, statement: str, statement_type: str,
                           time_period: str, df: pd.DataFrame, *, metadata: dict | None = None,
                           prune_previous: bool = False) -> str:
        return self.write(
            LIB_FUNDAMENTALS,
            self.fundamentals_symbol(symbol, statement, statement_type, time_period),
            df, metadata=metadata, prune_previous=prune_previous,
        )

    def update_fundamentals(self, symbol: str, statement: str, statement_type: str,
                            time_period: str, df: pd.DataFrame) -> int:
        """Upsert, not append -- financials get RESTATED (we measured this
        directly: NMDC's balance sheet sat a full fiscal year stale relative to
        its own income statement). ``update`` replaces the overlapping period
        range instead of assuming newer data only ever adds new periods."""
        return self.update(
            LIB_FUNDAMENTALS,
            self.fundamentals_symbol(symbol, statement, statement_type, time_period),
            df,
        )

    def read_fundamentals(self, symbol: str, statement: str, statement_type: str,
                          time_period: str, *, date_range: DateRange | None = None,
                          columns: list[str] | None = None,
                          query_builder: Any | None = None) -> pd.DataFrame:
        return self.read(
            LIB_FUNDAMENTALS,
            self.fundamentals_symbol(symbol, statement, statement_type, time_period),
            date_range=date_range, columns=columns, query_builder=query_builder,
        )

    # ── typed: company profile (sector snapshot, also in `fundamentals`) ──────

    @staticmethod
    def profile_symbol(ticker: str) -> str:
        """Company-profile snapshot -- sector, description, market cap.

        Deliberately NOT ``fundamentals_symbol``'s
        ``{symbol}/{statement}/{type}/{period}`` shape: a profile has no
        statement, no consolidated/standalone axis and no reporting period. It
        is one wide row of current facts, so it gets a two-part symbol of its
        own inside the same library.
        """
        return f"{ticker}/profile"

    def write_profile(self, ticker: str, df: pd.DataFrame, *,
                      metadata: dict | None = None,
                      prune_previous: bool = False) -> str:
        """Full overwrite -- NOT ``update_fundamentals``' upsert.

        A company's sector is a single current fact, and ``update``'s
        range-replace semantics assume a date-indexed series being extended.
        Re-fetching simply supersedes; ArcticDB keeps the prior version in
        history either way.
        """
        return self.write(LIB_FUNDAMENTALS, self.profile_symbol(ticker), df,
                          metadata=metadata, prune_previous=prune_previous)

    def read_profile(self, ticker: str, *, columns: list[str] | None = None) -> pd.DataFrame:
        return self.read(LIB_FUNDAMENTALS, self.profile_symbol(ticker), columns=columns)

    def list_profiles(self) -> list[str]:
        """Tickers that currently have a stored profile."""
        suffix = "/profile"
        return sorted(
            s[: -len(suffix)] for s in self.list_symbols(LIB_FUNDAMENTALS)
            if s.endswith(suffix)
        )

    # ── typed: residuals (per-stock factor-model output) ──────────────────────

    def write_residuals(self, symbol: str, model_version: str, interval: str, df: pd.DataFrame,
                        *, metadata: dict | None = None, prune_previous: bool = False) -> str:
        return self.write(LIB_RESIDUALS, self.residuals_symbol(symbol, model_version, interval), df,
                          metadata=metadata, prune_previous=prune_previous)

    def append_residuals(self, symbol: str, model_version: str, interval: str, df: pd.DataFrame) -> int:
        return self.append(LIB_RESIDUALS, self.residuals_symbol(symbol, model_version, interval), df)

    def read_residuals(self, symbol: str, model_version: str, interval: str, *,
                       date_range: DateRange | None = None, columns: list[str] | None = None,
                       query_builder: Any | None = None) -> pd.DataFrame:
        return self.read(LIB_RESIDUALS, self.residuals_symbol(symbol, model_version, interval),
                         date_range=date_range, columns=columns, query_builder=query_builder)

    # ── typed: signals (per-stock rank/bucket/target-weight) ───────────────────

    def write_signal(self, symbol: str, model_version: str, interval: str, df: pd.DataFrame,
                     *, metadata: dict | None = None, prune_previous: bool = False) -> str:
        return self.write(LIB_SIGNALS, self.signals_symbol(symbol, model_version, interval), df,
                          metadata=metadata, prune_previous=prune_previous)

    def append_signal(self, symbol: str, model_version: str, interval: str, df: pd.DataFrame) -> int:
        return self.append(LIB_SIGNALS, self.signals_symbol(symbol, model_version, interval), df)

    def read_signal(self, symbol: str, model_version: str, interval: str, *,
                    date_range: DateRange | None = None, columns: list[str] | None = None,
                    query_builder: Any | None = None) -> pd.DataFrame:
        return self.read(LIB_SIGNALS, self.signals_symbol(symbol, model_version, interval),
                         date_range=date_range, columns=columns, query_builder=query_builder)

    # ── typed: corporate actions (per-company event log) ───────────────────────

    def write_corporate_actions(self, symbol: str, df: pd.DataFrame, *,
                                metadata: dict | None = None, prune_previous: bool = False) -> str:
        return self.write(LIB_CORPORATE_ACTIONS, self.corporate_actions_symbol(symbol), df,
                          metadata=metadata, prune_previous=prune_previous)

    def update_corporate_actions(self, symbol: str, df: pd.DataFrame) -> int:
        """Upsert -- a previously-recorded action can be corrected (e.g. an
        OFS's confirmed executed size revising an earlier estimate), and a
        pending action (board approval) can later be confirmed or withdrawn
        in place (the Adani FPO case: fully subscribed, then withdrawn --
        the record needs to change, not just gain a new row)."""
        return self.update(LIB_CORPORATE_ACTIONS, self.corporate_actions_symbol(symbol), df)

    def read_corporate_actions(self, symbol: str, *, date_range: DateRange | None = None,
                               columns: list[str] | None = None,
                               query_builder: Any | None = None) -> pd.DataFrame:
        return self.read(LIB_CORPORATE_ACTIONS, self.corporate_actions_symbol(symbol),
                         date_range=date_range, columns=columns, query_builder=query_builder)

    # ── typed: metadata tables ────────────────────────────────────────────────

    def write_table(self, name: str, df: pd.DataFrame, *, metadata: dict | None = None,
                    prune_previous: bool = False) -> str:
        """Write a small reference table (e.g. ``contract_master``) into ``metadata``."""
        return self.write(LIB_METADATA, name, df, metadata=metadata, prune_previous=prune_previous)

    def read_table(self, name: str) -> pd.DataFrame:
        return self.read(LIB_METADATA, name)

    def __repr__(self) -> str:
        return f"ArcticStore(uri={self.uri!r}, libraries={list(self._libs)})"
