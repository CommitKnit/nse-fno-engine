"""fetch_options.py — any-ticker options pipeline: fetch → build → enrich → clean → ArcticDB.

For a ticker at a given interval this:
  0. resolves the underlying instrument key (used only to look up option contracts),
  1. enumerates option expiries (expired-instruments history + the live current month)
     and, for each expiry, computes its last-N-trading-day window (``--lookback-td``,
     default 40 TD ≈ 56 calendar days ≈ 1.9 expiry cycles). These windows OVERLAP by
     design — a given date typically falls inside two expiries' windows at once (its
     current front-month contract and the still-live next-month one), which is what
     "every expiry trading in the period" means here. Then, per expiry:
       2. check ArcticDB for what's already stored for that (ticker, interval, expiry)
          and skip entirely if the window is fully covered, else narrow the fetch down
          to just the missing leading/trailing sub-window (see Incremental skip below),
       3. fetch the full option chain's candles from Upstox (expired-instruments
          endpoint for past expiries, v3 for the live one),
       4. read the pricing future's close from ArcticDB —
          ``futures_contracts/{TICKER}/{interval}/{expiry}``, written by
          ``fetch-ticker {TICKER} --history`` — and join it to every option bar on the
          EXACT timestamp. Option bars with no futures bar at that timestamp are dropped
          (they cannot be priced); no forward-fill. Weekly option expiries are priced on
          the nearest monthly contract. No futures candles are fetched here,
       5. run the shared enrich+clean core from options/chain/build_enriched_chains.py
          (build chains → Black-76 IV/Greeks on the future → OptionsDataCleaner: stale
          repair via put-call parity / ffill, arbitrage flagging, delta band, fractional
          trading-day DTE),
       6. gate on the ``in_last_{N}_td`` flag — only bars inside the lookback window
          survive; everything else enrichment produced is dropped before the write,
       7. store the enriched chain.

Spot is neither fetched nor written by this script. Black-76 prices options on the
future, so IV and Greeks never use spot; wherever spot is needed downstream (borrow
rates, snapshots) it is the Kite series in the ``spot`` library
(``fetch-kite-equities``), see ``ArcticStore.spot_symbol_for``.

Incremental skip: coverage is tracked per (ticker, interval, expiry) via
``ArcticStore.stored_date_span`` — an index-metadata read, not a full data read. Only
two gaps are ever re-fetched: the leading gap before what's already stored, and the
trailing gap from the last stored date onward (inclusive, so a partially-fetched final
day gets repaired rather than left half-written). A fully-covered expiry costs zero API
calls. Pass ``--force`` to bypass this and refetch/overwrite the whole window regardless.

ArcticDB layout:
    futures_contracts {TICKER}/{interval}/{expiry}     READ  — pricing future (fetch-ticker --history)
    options_enriched  {TICKER}/{interval}/{expiry}     WRITE — enriched chain (in_last_{N}_td rows only)

Prerequisite: run ``fetch-ticker TICKER --history --intervals <interval>`` up to the
same ``--to`` date first, so every option bar has its future in ``futures_contracts``.

Coverage note: history is bounded by what Upstox retains for expired instruments
(~20 months observed for equity F&O) plus the live front-month; a longer --from is
silently clipped to what's actually available.

Usage:
    fetch-options TCS --interval 15minute
    fetch-options TCS --interval 15minute --from 2025-01-01 --to 2026-06-30
    fetch-options TCS --interval 15minute --lookback-td 60
    fetch-options TCS --interval 15minute --dry-run
    fetch-options TCS --interval 15minute --no-arctic     # reads futures, writes nothing
    fetch-options TCS --interval 15minute --force
    (or: python scripts/fetch_options.py TCS ...)
"""

from __future__ import annotations

import argparse
import sys
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
from loguru import logger

# ── Project root on sys.path ───────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.constants import INSTRUMENTS
from config.settings import settings
from data.fetchers.api_fetcher import UpstoxFetcher
from data.models.bar import Bar
from data.models.option_contract import OptionContract
from data.processors.data_cleaner import OptionsDataCleaner
from data.storage.arctic_store import LIB_OPTIONS, ArcticStore
from options.chain.build_enriched_chains import _write_enriched_to_arctic, run_stage2_enrich
from options.chain.chain_enricher import ChainEnricher
from options.pricing.dte_utils import (
    DEFAULT_LOOKBACK_TD,
    candles_per_day,
    last_n_td_column,
    nth_trading_day_before,
    trading_days_inclusive,
)

_IST = "Asia/Kolkata"

# interval string → (v3 unit, v3 multiplier). The same string is passed to the
# expired-instruments endpoint and to run_stage2_enrich / dte_utils.
_V3_UNIT: dict[str, tuple[str, int]] = {
    "1minute": ("minutes", 1),
    "3minute": ("minutes", 3),
    "5minute": ("minutes", 5),
    "15minute": ("minutes", 15),
    "30minute": ("minutes", 30),
    "60minute": ("minutes", 60),
    "day": ("days", 1),
}

# options interval → the interval name fetch_ticker.py stores futures_contracts under.
_FUTURES_INTERVAL: dict[str, str] = {"day": "1day"}
LIB_FUTURES_CONTRACTS = "futures_contracts"

_MONTHLY_MIN_GAP = 21     # days; drops weekly expiries (monthlies are ~28-31 apart)
_CHUNK_DAYS = 28          # per-request candle window (v3 has a ~30-day limit)


# ══════════════════════════════════════════════════════════════════════════════
# Small helpers
# ══════════════════════════════════════════════════════════════════════════════

def _month_chunks(from_d: date, to_d: date, days: int = _CHUNK_DAYS) -> list[tuple[date, date]]:
    """Split [from_d, to_d] into contiguous ≤``days``-wide (from, to) windows."""
    chunks: list[tuple[date, date]] = []
    cur = from_d
    while cur <= to_d:
        end = min(cur + timedelta(days=days - 1), to_d)
        chunks.append((cur, end))
        cur = end + timedelta(days=1)
    return chunks


def _fetch_candles(
    fetcher: UpstoxFetcher, key: str, interval: str,
    from_d: date, to_d: date, is_expired: bool,
) -> list[Bar]:
    """Fetch OHLCV bars for one instrument key over [from_d, to_d], month-chunked.

    Expired keys (``NSE_FO|token|DD-MM-YYYY``) use the v2 expired-instruments endpoint;
    live keys use the v3 endpoint. Bars are de-duplicated by timestamp across chunks.
    """
    by_ts: dict = {}
    for clo, chi in _month_chunks(from_d, to_d):
        try:
            if is_expired:
                bars = fetcher.fetch_expired_candles(key, interval, clo, chi)
            else:
                unit, mult = _V3_UNIT[interval]
                bars = fetcher.fetch_historical_candles_v3(key, unit, mult, clo, chi)
        except Exception as exc:  # noqa: BLE001 — a dead contract/chunk shouldn't abort the run
            logger.debug(f"  candle fetch failed {key} {clo}→{chi}: {exc}")
            continue
        for b in bars:
            by_ts[b.timestamp] = b
    return sorted(by_ts.values(), key=lambda b: b.timestamp)


# ══════════════════════════════════════════════════════════════════════════════
# Expiry enumeration → last-N-trading-day windows (overlapping)
# ══════════════════════════════════════════════════════════════════════════════

def _is_monthly(expiry: date) -> bool:
    """Return True if *expiry* is the last occurrence of its weekday in its month.

    Weekday-agnostic (checks "does +7 days cross into next month", not a specific
    weekday) — correct across NSE's historical expiry-weekday changes (e.g.
    Thursday → Tuesday in 2024) without hardcoding either.
    """
    return (expiry + timedelta(days=7)).month != expiry.month


def _nearest_monthly_expiry(expiry: date) -> date:
    """Return the nearest monthly expiry on or after *expiry*.

    Index futures only expire monthly, so a weekly option expiry has no same-date
    futures contract — this resolves the monthly contract to price it against
    instead. Detects the target weekday dynamically from *expiry* itself (not
    hardcoded), so it stays correct across future NSE weekday changes too.
    """
    if _is_monthly(expiry):
        return expiry

    dow = expiry.weekday()
    for months_ahead in range(3):
        month = expiry.month + months_ahead
        year = expiry.year + (month - 1) // 12
        month = ((month - 1) % 12) + 1

        if month == 12:
            last_day = date(year + 1, 1, 1) - timedelta(days=1)
        else:
            last_day = date(year, month + 1, 1) - timedelta(days=1)

        days_back = (last_day.weekday() - dow) % 7
        last_occurrence = last_day - timedelta(days=days_back)

        if last_occurrence >= expiry:
            return last_occurrence

    return expiry  # unreachable safety fallback


def _filter_monthly(expiries: list[date], min_gap: int = _MONTHLY_MIN_GAP) -> list[date]:
    """Keep only expiries spaced ``>= min_gap`` days apart (drops weeklies)."""
    if not expiries:
        return []
    out = [expiries[0]]
    for e in expiries[1:]:
        if (e - out[-1]).days >= min_gap:
            out.append(e)
    return out


def _enumerate_windows(
    fetcher: UpstoxFetcher, underlying_key: str, ticker: str,
    global_from: date, global_to: date, expiries_mode: str = "monthly",
    lookback_td: int = DEFAULT_LOOKBACK_TD, exchange: str = "NSE",
) -> list[tuple[date, date, date]]:
    """Return ``[(expiry, window_from, window_to), …]`` — one *lookback_td*-trading-day
    window per expiry that traded during ``[global_from, global_to]``.

    Combines expired-instruments expiries with the live upcoming expiries. When
    ``expiries_mode == "monthly"`` (default), keeps only monthly expiries. When
    ``"all"``, keeps every expiry (weekly + monthly).

    Each window is ``[nth_trading_day_before(e, lookback_td), e]`` clipped to
    ``[global_from, global_to]`` — independent per expiry, unlike the old
    ``(prev_expiry, expiry]`` partition. At the default lookback_td=40 (≈56
    calendar days ≈ 1.9 expiry cycles), consecutive expiries' windows OVERLAP,
    so a date generally belongs to two windows at once (current F1 and the
    still-live F2) rather than exactly one. That overlap is the point: it's
    what makes this "every expiry trading in the period" rather than only the
    front-month contract.
    """
    exp: set[date] = set()
    try:
        exp.update(fetcher.fetch_expiries(underlying_key))
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"fetch_expiries failed: {exc}")
    try:
        for s in fetcher.resolve_underlying(ticker, exchange=exchange).get("expiries", []):
            if s:
                exp.add(date.fromisoformat(str(s)[:10]))
    except Exception as exc:  # noqa: BLE001
        logger.debug(f"live expiry resolve failed: {exc}")

    selected = _filter_monthly(sorted(exp)) if expiries_mode == "monthly" else sorted(exp)

    windows: list[tuple[date, date, date]] = []
    for e in selected:
        lo = max(nth_trading_day_before(e, lookback_td), global_from)
        hi = min(e, global_to)
        if lo <= hi:
            windows.append((e, lo, hi))
    return windows


def _missing_subwindows(
    store: ArcticStore | None, ticker: str, interval: str,
    expiry: date, lo: date, hi: date, force: bool,
) -> list[tuple[date, date]]:
    """Contiguous ``(from, to)`` sub-windows of ``[lo, hi]`` not yet stored in Arctic.

    Returns ``[]`` when the whole window is already covered — the caller should
    skip this expiry with zero API calls. Returns ``[(lo, hi)]`` unchanged when
    ``force`` is set, when *store* is ``None`` (``--no-arctic``), or when nothing
    is stored yet for this ``(ticker, interval, expiry)``.

    Only the two edges of the stored span are ever treated as missing:
      * **leading** — before the earliest stored date.
      * **trailing** — from (and *including*) the latest stored date onward, so
        a partially-fetched final day (e.g. an in-progress session at the time
        of the previous run) gets repaired rather than left half-written.

    A hole in the *middle* of the stored span (e.g. a date where every row got
    filtered out by ``pre_solver_filter`` / the delta band) is treated as
    already attempted, not missing — otherwise an empty date would be
    re-fetched forever. ``--force`` is the escape hatch for a full redo.
    """
    if force or store is None:
        return [(lo, hi)]

    symbol = ArcticStore.options_symbol(ticker, interval, expiry)
    span = store.stored_date_span(LIB_OPTIONS, symbol)
    if span is None:
        return [(lo, hi)]

    covered_lo = span[0].date()
    covered_hi = span[1].date()

    gaps: list[tuple[date, date]] = []
    if lo < covered_lo:
        leading_hi = min(covered_lo - timedelta(days=1), hi)
        if lo <= leading_hi and trading_days_inclusive(lo, leading_hi) > 0:
            gaps.append((lo, leading_hi))
    if hi >= covered_hi:
        trailing_lo = max(covered_hi, lo)
        if trailing_lo <= hi and trading_days_inclusive(trailing_lo, hi) > 0:
            gaps.append((trailing_lo, hi))
    return gaps


# ══════════════════════════════════════════════════════════════════════════════
# Raw DataFrame assembly (schema consumed by run_stage2_enrich)
# ══════════════════════════════════════════════════════════════════════════════

def _assemble_raw_df(
    ticker: str, underlying_key: str, expiry: date,
    contracts: list[OptionContract],
    contract_bars: dict[str, list[Bar]],
    fut_close_by_ts: dict,
) -> pd.DataFrame:
    """One row per (timestamp, strike, opt_type), priced on the expiry's future.

    *fut_close_by_ts* maps UTC epoch ns → futures close (``stored_futures_close``).
    Option rows with no futures bar at exactly their timestamp are dropped (can't
    be priced). ``dte`` is a placeholder here — run_stage2_enrich recomputes the
    fractional trading-day value from timestamp/expiry/interval.
    """
    rows: list[dict] = []
    for c in contracts:
        for b in contract_bars.get(c.instrument_key, []):
            fp = fut_close_by_ts.get(pd.Timestamp(b.timestamp).value)
            if fp is None or fp <= 0:
                continue
            strike = float(c.strike)
            rows.append({
                "underlying_symbol": ticker,
                "timestamp": b.timestamp,
                "futures_price": fp,
                "instrument_key": c.instrument_key,
                "trading_symbol": c.trading_symbol,
                "underlying_key": underlying_key,
                "strike_price": strike,
                "instrument_type": c.option_type.value,
                "expiry": expiry.isoformat(),
                "lot_size": c.lot_size,
                "tick_size": c.tick_size,
                "open": b.open, "high": b.high, "low": b.low, "close": b.close,
                "volume": b.volume, "oi": b.oi or 0,
                "dte": 0.0,
                "moneyness": strike / fp,
            })
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["expiry"] = pd.to_datetime(df["expiry"]).dt.date
    return (df.sort_values(["timestamp", "strike_price", "instrument_type"])
              .reset_index(drop=True))


# ══════════════════════════════════════════════════════════════════════════════
# Per-expiry pipeline (fetch → assemble → enrich/clean → load)
# ══════════════════════════════════════════════════════════════════════════════

def _resolve_contracts(
    fetcher: UpstoxFetcher, underlying_key: str, spot_key: str,
    expiry: date, is_expired: bool,
) -> list[OptionContract]:
    """Option contracts for *expiry* (expired-instruments API, or the live chain)."""
    if is_expired:
        return fetcher.fetch_option_contracts(underlying_key, expiry)
    return fetcher.fetch_live_option_contracts(spot_key, expiry_date=expiry)


def futures_contract_symbol(ticker: str, interval: str, expiry: date) -> str:
    """``futures_contracts`` symbol holding the future that prices *expiry*'s options.

    A weekly option expiry has no same-date futures contract (index/stock futures
    only expire monthly), so it is priced on the nearest monthly contract.
    """
    pricing_expiry = _nearest_monthly_expiry(expiry)
    iv = _FUTURES_INTERVAL.get(interval, interval)
    return f"{ticker}/{iv}/{pricing_expiry.isoformat()}"


def stored_futures_close(
    store: ArcticStore, ticker: str, interval: str, expiry: date,
    win_from: date, win_to: date,
) -> dict[int, float] | None:
    """Pricing-future closes from ArcticDB, keyed by bar timestamp (epoch ns).

    Reads ``futures_contracts/{ticker}/{interval}/{expiry}`` (written by
    ``fetch-ticker --history``) over ``[win_from, win_to]``. Keys are UTC epoch
    nanoseconds so an option bar matches only a futures bar at the EXACT same instant,
    whatever tz object each side carries. Returns ``None`` when the contract has not
    been stored at all.
    """
    sym = futures_contract_symbol(ticker, interval, expiry)
    if not store.has_symbol(LIB_FUTURES_CONTRACTS, sym):
        return None
    lo = pd.Timestamp(win_from, tz=_IST)
    hi = pd.Timestamp(win_to, tz=_IST) + pd.Timedelta(days=1)
    df = store.read(LIB_FUTURES_CONTRACTS, sym, date_range=(lo, hi), columns=["close"])
    close = df["close"].dropna()
    close = close[close > 0]
    return {int(ts.value): float(px) for ts, px in close.items()}


def process_expiry(
    ticker: str, underlying_key: str, spot_key: str,
    expiry: date, win_from: date, win_to: date, interval: str,
    fetcher: UpstoxFetcher, enricher: ChainEnricher, cleaner: OptionsDataCleaner,
    store: ArcticStore, write: bool = True,
    lookback_td: int = DEFAULT_LOOKBACK_TD,
) -> int:
    """Fetch, enrich, clean and store one expiry's ``[win_from, win_to]`` sub-window.

    *win_from*/*win_to* may be the expiry's full last-``lookback_td``-TD window or a
    narrower leading/trailing gap from ``_missing_subwindows`` — this function itself
    is unaware of incremental skip, it just fetches+stores whatever range it's given.

    Options are priced on the future already stored in ``futures_contracts`` (see
    ``stored_futures_close``); *store* is always read, and written only when *write*.

    After enrichment, rows are gated on ``in_last_{lookback_td}_td`` (see
    ``dte_utils.last_n_td_column``) before the Arctic write — only bars inside the
    lookback window are ever stored, so a fetch window wider than the flag's own
    definition (shouldn't happen here, but keeps the two independently correct)
    can't leak rows past it.

    Returns the number of enriched rows stored (0 if nothing was produced).
    """
    is_expired = expiry < date.today()
    tag = f"[{expiry} | {'expired' if is_expired else 'live'}]"

    fut_close_by_ts = stored_futures_close(store, ticker, interval, expiry, win_from, win_to)
    fut_sym = futures_contract_symbol(ticker, interval, expiry)
    if fut_close_by_ts is None:
        iv = _FUTURES_INTERVAL.get(interval, interval)
        logger.warning(f"{tag} futures_contracts/{fut_sym} not in ArcticDB — cannot price "
                       f"options; run `fetch-ticker {ticker} --history --intervals {iv}` first. Skipping.")
        return 0
    if not fut_close_by_ts:
        logger.warning(f"{tag} futures_contracts/{fut_sym} has no bars in {win_from}→{win_to}; skipping.")
        return 0

    contracts = _resolve_contracts(fetcher, underlying_key, spot_key, expiry, is_expired)
    if not contracts:
        logger.warning(f"{tag} no option contracts — skipping.")
        return 0

    logger.info(f"{tag} {len(contracts)} contracts | window {win_from}→{win_to} | "
                f"priced on futures_contracts/{fut_sym} ({len(fut_close_by_ts)} bars)")

    # ── Option candles (full chain) ────────────────────────────────────────────
    contract_bars: dict[str, list[Bar]] = {}
    for i, c in enumerate(contracts, 1):
        contract_bars[c.instrument_key] = _fetch_candles(
            fetcher, c.instrument_key, interval, win_from, win_to, is_expired
        )
        if i % 40 == 0:
            got = sum(1 for v in contract_bars.values() if v)
            logger.info(f"{tag}   {i}/{len(contracts)} contracts ({got} w/ data | calls={fetcher.request_count})")

    # ── Assemble → enrich/clean → store ────────────────────────────────────────
    raw_df = _assemble_raw_df(ticker, underlying_key, expiry, contracts, contract_bars, fut_close_by_ts)
    n_bars = sum(len(v) for v in contract_bars.values())
    if n_bars:
        logger.info(f"{tag} {n_bars - len(raw_df)} of {n_bars} option bars had no futures bar "
                    "at the same timestamp — dropped (unpriceable).")
    if raw_df.empty:
        logger.warning(f"{tag} assembled raw frame is empty — skipping.")
        return 0

    enriched = run_stage2_enrich(raw_df, ticker, enricher, cleaner, interval, lookback_td=lookback_td)
    if enriched.empty:
        logger.warning(f"{tag} enrichment produced no rows — skipping.")
        return 0

    # ── Last-N-TD gate: only store bars inside the lookback window ────────────
    flag_col = last_n_td_column(lookback_td)
    if flag_col in enriched.columns:
        mask = enriched[flag_col].fillna(False).astype(bool)
        kept, dropped = int(mask.sum()), int((~mask).sum())
        enriched = enriched[mask].reset_index(drop=True)
        logger.info(f"{tag} {flag_col} gate: kept {kept}, dropped {dropped}")
    else:
        logger.warning(f"{tag} '{flag_col}' column missing after enrich — storing all rows.")
    if enriched.empty:
        logger.warning(f"{tag} nothing left after the {flag_col} gate — skipping.")
        return 0

    if write:
        _write_enriched_to_arctic(store, enriched, ticker, interval)
    logger.info(f"{tag} ✓ {len(enriched)} enriched rows"
                + ("" if write else " (not stored — --no-arctic)"))
    return len(enriched)


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════

def main(argv: list[str] | None = None) -> None:
    for stream in (sys.stdout, sys.stderr):   # help/log text has →, — (cp1252 can't encode them)
        try:
            stream.reconfigure(encoding="utf-8")
        except Exception:  # noqa: BLE001
            pass
    ap = argparse.ArgumentParser(
        description="Fetch → build → enrich → clean → ArcticDB for any ticker's options "
                    "(every expiry trading in range, last-N-TD windowed), priced on the "
                    "futures closes already stored in ArcticDB at the same timestamp. "
                    "Incrementally skips what's already in ArcticDB.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("ticker", help="Equity or index ticker, e.g. TCS, NIFTY, BANKNIFTY")
    ap.add_argument("--interval", default="15minute", choices=list(_V3_UNIT),
                    help="Candle interval (default: 15minute).")
    ap.add_argument("--from", dest="from_date", default=None, metavar="YYYY-MM-DD",
                    help="Start date (default: ~2 years before --to).")
    ap.add_argument("--to", dest="to_date", default=None, metavar="YYYY-MM-DD",
                    help="End date (default: today).")
    ap.add_argument("--expiries", default="monthly", choices=["monthly", "all"],
                    help="'monthly' (default): only monthly expiries. 'all': also include "
                         "weekly expiries (e.g. NIFTY weeklies).")
    ap.add_argument("--lookback-td", type=int, default=DEFAULT_LOOKBACK_TD, metavar="N",
                    help="Per-expiry trading-day window AND the Arctic write gate: only "
                         f"bars within the last N trading days of each expiry are stored, "
                         f"under column 'in_last_N_td' (default: {DEFAULT_LOOKBACK_TD}).")
    ap.add_argument("--no-arctic", action="store_true",
                    help="Fetch + enrich but do not write to ArcticDB (futures are still READ "
                         "from futures_contracts; the incremental-skip check is disabled).")
    ap.add_argument("--force", action="store_true",
                    help="Bypass the incremental skip — refetch and overwrite every window "
                         "in range even if already covered in ArcticDB.")
    ap.add_argument("--dry-run", action="store_true",
                    help="List resolved keys, expiries, windows, and what the incremental "
                         "skip would fetch vs. skip; fetch no candles, write nothing.")
    args = ap.parse_args(argv)
    if args.lookback_td <= 0:
        ap.error("--lookback-td must be a positive integer.")

    ticker = args.ticker.strip().upper()
    interval = args.interval
    to_d = date.fromisoformat(args.to_date) if args.to_date else date.today()
    from_d = date.fromisoformat(args.from_date) if args.from_date else (to_d - timedelta(days=730))

    fetcher = UpstoxFetcher()
    # `store` is always opened: the pricing futures are read from it. It is the write
    # target unless --dry-run / --no-arctic. --no-arctic also disables the
    # incremental-skip coverage check (`coverage_store` None → fetch full windows).
    store = ArcticStore()
    write = not (args.dry_run or args.no_arctic)
    coverage_store = None if args.no_arctic else store

    # ── Stage 0: resolve keys ──────────────────────────────────────────────────
    # Registry first (exact, no API call — covers known indices like NIFTY whose
    # trading_symbol doesn't equal the bare ticker via search, e.g. "Nifty 50").
    # Live search (equity then index segment) as the fallback for anything else.
    registered = INSTRUMENTS.get(ticker)
    if registered:
        spot_key = registered.instrument_key
        exchange = "BSE" if spot_key.startswith("BSE") else "NSE"
    else:
        exchange = "NSE"
        spot_key = fetcher.resolve_spot_key(ticker, exchange=exchange)
        if not spot_key:
            # Not every index/underlying lives on NSE (e.g. SENSEX is BSE-only) —
            # retry on BSE before giving up.
            exchange = "BSE"
            spot_key = fetcher.resolve_spot_key(ticker, exchange=exchange)
    if not spot_key:
        logger.error(f"Could not resolve spot instrument_key for '{ticker}' on NSE or BSE. Aborting.")
        return
    underlying_key = spot_key  # equity/index key doubles as the expired-instruments underlying key

    logger.info("=" * 66)
    logger.info(f"fetch_options  |  {ticker}  |  {interval}  |  {from_d} → {to_d}")
    logger.info(f"  spot/underlying key : {spot_key}  ({exchange})")
    logger.info(f"  ArcticDB            : {store.uri}  (writes "
                f"{'OFF — dry-run' if args.dry_run else 'OFF — --no-arctic' if args.no_arctic else 'ON'})")
    logger.info(f"  pricing futures     : futures_contracts/{ticker}/{_FUTURES_INTERVAL.get(interval, interval)}/<expiry>")
    logger.info("=" * 66)

    # ── Stage 1: expiries → last-N-TD windows (overlapping) ───────────────────
    windows = _enumerate_windows(fetcher, underlying_key, ticker, from_d, to_d,
                                  args.expiries, args.lookback_td, exchange=exchange)
    logger.info(f"{len(windows)} {args.expiries} window(s) in [{from_d}, {to_d}] "
                f"(lookback={args.lookback_td} TD).")

    if args.dry_run:
        total_est_bars = 0
        total_est_calls = 0
        for e, lo, hi in windows:
            is_exp = e < date.today()
            try:
                cs = (fetcher.fetch_option_contracts(underlying_key, e) if is_exp
                      else fetcher.fetch_live_option_contracts(spot_key, expiry_date=e))
                n_contracts = len(cs)
            except Exception:  # noqa: BLE001
                n_contracts = None

            subwindows = _missing_subwindows(coverage_store, ticker, interval, e, lo, hi, args.force)
            has_fut = store.has_symbol(LIB_FUTURES_CONTRACTS, futures_contract_symbol(ticker, interval, e))
            tag = (f"  {e}  [{'expired' if is_exp else 'live'}]  "
                   f"contracts={n_contracts if n_contracts is not None else '?'}  "
                   f"futures={'stored' if has_fut else 'MISSING — run fetch-ticker --history'}")
            if not subwindows:
                logger.info(f"{tag}  full window {lo}→{hi}  SKIP — already fully covered")
                continue
            for s_lo, s_hi in subwindows:
                tdays = trading_days_inclusive(s_lo, s_hi)
                cal_days = (s_hi - s_lo).days + 1
                if n_contracts is not None:
                    est_bars = n_contracts * tdays * candles_per_day(interval)
                    est_calls = n_contracts * -(-cal_days // _CHUNK_DAYS)
                    total_est_bars += est_bars
                    total_est_calls += est_calls
                    cost = f"trading_days={tdays}  est_bars≤{est_bars:,}  est_api_calls≈{est_calls:,}"
                else:
                    cost = f"trading_days={tdays}  est_bars/calls=? (contract lookup failed)"
                logger.info(f"{tag}  fetch {s_lo}→{s_hi}  {cost}")
        logger.info(
            "Dry run complete — no candles fetched, nothing written. Upper-bound estimate "
            f"(pre_solver_filter / filter_delta_band will remove a large share of est_bars): "
            f"bars≤{total_est_bars:,}, api_calls≈{total_est_calls:,}. "
            f"Actual API calls made so far (contract lookups only): {fetcher.request_count}"
        )
        return

    # ── Stages 2-7: per expiry, incrementally (only missing sub-windows fetched) ──
    covered: list[date] = []
    skipped_full: list[date] = []
    total_rows = 0
    for i, (e, lo, hi) in enumerate(windows, 1):
        subwindows = _missing_subwindows(coverage_store, ticker, interval, e, lo, hi, args.force)
        if not subwindows:
            logger.info(f"\n[{i}/{len(windows)}] expiry {e} — SKIP, already fully covered in ArcticDB.")
            skipped_full.append(e)
            continue

        logger.info(f"\n[{i}/{len(windows)}] expiry {e}  ({len(subwindows)} sub-window(s) to fetch: "
                    f"{', '.join(f'{s}→{h}' for s, h in subwindows)})")
        expiry_rows = 0
        for sub_lo, sub_hi in subwindows:
            try:
                n = process_expiry(ticker, underlying_key, spot_key, e, sub_lo, sub_hi, interval,
                                   fetcher, ChainEnricher(),
                                   OptionsDataCleaner(risk_free_rate=settings.RISK_FREE_RATE),
                                   store, write=write, lookback_td=args.lookback_td)
                expiry_rows += n
            except Exception as exc:  # noqa: BLE001
                logger.exception(f"expiry {e} sub-window {sub_lo}→{sub_hi} failed: {exc}")
        if expiry_rows:
            covered.append(e)
            total_rows += expiry_rows

    # ── Summary ────────────────────────────────────────────────────────────────
    logger.info("\n" + "=" * 66)
    logger.info("Done.")
    logger.info(f"  expiries fetched : {len(covered)}/{len(windows)}"
                + (f"  ({covered[0]} → {covered[-1]})" if covered else ""))
    logger.info(f"  expiries skipped : {len(skipped_full)}  (already fully covered)")
    logger.info(f"  enriched rows    : {total_rows:,}")
    logger.info(f"  API calls        : {fetcher.request_count}")
    logger.info("=" * 66)


if __name__ == "__main__":
    main()
