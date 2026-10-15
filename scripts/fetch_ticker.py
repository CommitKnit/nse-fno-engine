"""fetch_ticker.py — Fetch NSE futures lifecycle data from Upstox and store in ArcticDB.

Reads every NSE_FO FUT record for the given ticker from the Upstox instrument master
JSON, fetches the full OHLCV lifecycle for each contract (expired and live), assembles
the F1/F2/F3 wide panel (the same layout LocalSnapshotBuilder reads), and upserts it
into the ``futures`` ArcticDB library as ``{SYMBOL}/{interval}``.

Note on coverage
----------------
The JSON only lists currently-traded instruments — typically the 3 active monthly
contracts plus any quarterly/long-dated ones.  Contracts that expired months ago and
are no longer listed will not appear in the JSON and will therefore not be fetched.
Re-run whenever Upstox refreshes the master JSON to pick up newly-listed expiries.

History mode (``--history``)
---------------------------
Adds what the JSON path cannot: expired contracts (Upstox expired-instruments API)
plus the live ones, stored per contract in ``futures_contracts/{SYMBOL}/{interval}/{expiry}``
and resumed from each contract's last print (contracts already complete cost zero
API calls). The F1/F2/F3 panel is rebuilt from those contracts in the canonical
stored layout (slot rules ported from src ``data/slot.py``) and the borrow rates
(src ``update_intraday.py`` formulas, Kite spot) are written to
``borrow_rates/{SYMBOL}/{interval}``. Only rows from the stored series' last day
onward are rewritten, unless older contracts were added in front, in which case the
panel and borrow series are rewritten in full (previous version kept).
Intervals: ``5minute``, ``15minute``, ``1day``.

Borrow columns (``BORROW_COLUMNS``): ``F1/F2/F3_dte_star``, ``OI_ratio``, ``U_t``,
``spread_spot_f1``, ``spread_spot_f2``, ``spread_f1_f2``, ``spot_f1_pct`` = (S−F1)/S×100,
``f1_f2_pct`` = (F1−F2)/F1×100, ``b1``, ``b2``, ``b3``, ``b12``, ``b23``, ``b13``,
``in_backwardation``, ``both_traded``, ``f2_thin_early``. ``dte_star`` uses the interval's
bars per day (75 at 5minute, 25 at 15minute).

Trade flags: ``both_traded`` = F1 and F2 both had volume in the bar (otherwise one leg's
close is a carried-forward, stale price). ``f2_thin_early`` = 10+ trading days before F1
expiry, both legs traded, and F2 volume < 10% of F1 volume in the bar.

``--fill-missing`` (no API calls) completes what is stored: a futures panel without a
borrow series gets the full borrow series; an existing borrow series without the two
``*_pct`` columns or the two trade flags gets them derived from its own stored spreads and
the panel's volumes (nothing else recomputed).
``--all`` runs every ticker that has a futures panel at ``--intervals``.

Usage
-----
    fetch-ticker bajaj-auto
    fetch-ticker bajaj-auto --intervals day 15minute
    fetch-ticker bajaj-auto --from 2024-01-01 --force
    fetch-ticker bajaj-auto --dry-run
    fetch-ticker bajaj-auto --json "C:\\Users\\risha\\Downloads\\complete (1).json.gz"
    fetch-ticker IREDA SBICARD --history --intervals 5minute 15minute 1day --to 2026-10-09
    fetch-ticker IREDA --history --dry-run
    fetch-ticker IREDA --history --borrow-only --from 2026-09-18   # recompute borrow
    fetch-ticker --fill-missing --all --intervals 5minute 15minute --dry-run
    fetch-ticker --fill-missing --all --intervals 5minute 15minute
    (or: python scripts/fetch_ticker.py ...)

Spot: the borrow rates read Kite spot from ``spot/{SYMBOL}/{interval}`` (run
``fetch-kite-equities SYMBOL --intervals ...`` first); see ``ArcticStore.spot_symbol_for``.
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.settings import settings  # noqa: E402
from data.fetchers.api_fetcher import UpstoxFetcher  # noqa: E402
from data.models.bar import Bar  # noqa: E402
from data.storage.arctic_store import ArcticStore  # noqa: E402
from options.pricing.dte_utils import trading_days_inclusive  # noqa: E402

_DEFAULT_JSON = Path(r"C:\Users\risha\Downloads\complete (1).json.gz")
_DEFAULT_INTERVALS = ["day", "15minute", "5minute"]
_LIFECYCLE_DAYS = 120  # per-contract fetch window: expiry − N days
_IST = "Asia/Kolkata"


# ── Phase 1: Instrument resolution ────────────────────────────────────────────


def load_instrument_json(path: Path) -> list[dict]:
    """Load Upstox instrument master JSON (.json or .json.gz)."""
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as fh:
        return json.load(fh)


def resolve_futures_contracts(instruments: list[dict], ticker: str) -> list[dict]:
    """Return all NSE_FO FUT records for *ticker*, sorted by expiry ascending.

    Matches against ``underlying_symbol`` (the NSE ticker, e.g. 'BAJAJ-AUTO')
    with a fallback to ``name`` (for index futures where both fields equal the
    short ticker, e.g. 'NIFTY').
    """
    target = ticker.upper().strip()

    def _matches(r: dict) -> bool:
        sym = r.get("underlying_symbol", "").upper().strip()
        name = r.get("name", "").upper().strip()
        return sym == target or name == target

    hits = [
        r for r in instruments
        if r.get("segment") == "NSE_FO"
        and r.get("instrument_type") == "FUT"
        and _matches(r)
    ]
    if not hits:
        # Suggest possible near-matches to help the user correct the ticker
        candidates = sorted(
            {
                r.get("underlying_symbol", "")
                for r in instruments
                if r.get("segment") == "NSE_FO"
                and r.get("instrument_type") == "FUT"
                and target[:4] in r.get("underlying_symbol", "").upper()
            }
        )
        hint = f"  Possible matches: {candidates}" if candidates else ""
        raise ValueError(
            f"No NSE_FO FUT contracts found for '{ticker}'. "
            f"Check the 'underlying_symbol' field in the JSON "
            f"(e.g. 'BAJAJ-AUTO', 'RVNL', 'NIFTY').{hint}"
        )
    hits.sort(key=lambda r: r["expiry"])  # expiry is Unix milliseconds
    return hits


def record_expiry(record: dict) -> date:
    """Unpack the Unix-ms expiry field to a date."""
    return date.fromtimestamp(record["expiry"] / 1000)


# ── Phase 2: Per-contract lifecycle fetch ─────────────────────────────────────


def fetch_lifecycle(
    fetcher: UpstoxFetcher,
    instrument_key: str,
    contract_expiry: date,
    interval: str,
    from_date: date | None,
) -> pd.DataFrame:
    """Fetch OHLCV candles for one futures contract, returned as an IST DatetimeIndex frame.

    Uses the expired-instruments endpoint for expired contracts and the standard
    historical-candle endpoint for the active (non-expired) contract.
    """
    today = date.today()
    start = from_date if from_date else (contract_expiry - timedelta(days=_LIFECYCLE_DAYS))
    start = max(start, date(2015, 1, 1))

    if contract_expiry < today:
        bars = fetcher.fetch_expired_candles(instrument_key, interval, start, contract_expiry)
    else:
        bars = fetcher.fetch_historical_candles(instrument_key, interval, start, today)

    if not bars:
        return pd.DataFrame()

    rows = [b.to_dict() for b in bars]
    df = pd.DataFrame(rows)
    df["timestamp"] = (
        pd.to_datetime(df["timestamp"], utc=True).dt.tz_convert(_IST)
    )
    df = df.set_index("timestamp").sort_index()
    df = df.drop(columns=["instrument_key"], errors="ignore")
    keep = [c for c in ("open", "high", "low", "close", "volume", "oi") if c in df.columns]
    return df[keep]


# ── Phase 3: F1/F2/F3 wide panel assembly ─────────────────────────────────────


def assemble_wide_panel(
    contract_data: list[tuple[date, int, str, pd.DataFrame]],
) -> pd.DataFrame:
    """Build the F1/F2/F3 wide panel from per-contract OHLCV DataFrames.

    Slot assignment at timestamp *t*:
      Fn = contract with n-th smallest expiry among those with expiry >= t.date()

    For contracts sorted by expiry (c_0 < c_1 < c_2 ...), contract c_i occupies
    slot s during the period  (E_(i-s), E_(i-s+1)], where the boundary is the
    prior-slot anchor's expiry.  Requires i >= s-1 (earlier contracts cannot fill
    higher-numbered slots that need two prior contracts to define the period).
    """
    if not contract_data:
        return pd.DataFrame()

    expiries = [c[0] for c in contract_data]

    # Detect timezone from the first non-empty frame
    tz = None
    for _, _, _, df in contract_data:
        if not df.empty and df.index.tz is not None:
            tz = df.index.tz
            break

    def _ts(d: date) -> pd.Timestamp:
        ts = pd.Timestamp(d)
        return ts.tz_localize(tz) if tz else ts

    slot_pieces: dict[int, list[pd.DataFrame]] = {1: [], 2: [], 3: []}

    for i, (expiry, cycle_id, ticker, df) in enumerate(contract_data):
        if df.empty:
            continue

        for s in (1, 2, 3):
            # Need at least s-1 prior contracts to define the period start
            if i < s - 1:
                continue

            # Period start: day after E_(i-s), or open-ended if no prior anchor
            if i >= s:
                start_ts = _ts(expiries[i - s] + timedelta(days=1))
            else:
                start_ts = None  # no lower bound — covers data from the very start

            # Period end: E_(i-s+1) inclusive → use next day as exclusive upper bound
            end_anchor = expiries[i - s + 1]
            end_ts = _ts(end_anchor + timedelta(days=1))

            sliced = (
                df[(df.index >= start_ts) & (df.index < end_ts)]
                if start_ts is not None
                else df[df.index < end_ts]
            )
            if sliced.empty:
                continue

            slot = f"F{s}"
            renamed = sliced.rename(columns={c: f"{slot}_{c}" for c in sliced.columns})
            renamed[f"{slot}_ticker"] = ticker
            renamed[f"{slot}_expiry"] = expiry.isoformat()
            if s == 1:
                renamed["cycle_id"] = cycle_id

            slot_pieces[s].append(renamed)

    if not slot_pieces[1]:
        logger.warning("Wide panel assembly produced no F1 rows — check that contract data is not empty.")
        return pd.DataFrame()

    wide = pd.concat(slot_pieces[1]).sort_index()

    for s in (2, 3):
        if slot_pieces[s]:
            extra = pd.concat(slot_pieces[s]).sort_index()
            wide = wide.merge(extra, left_index=True, right_index=True, how="left")

    wide["spot_missing"] = True
    return wide


# ══════════════════════════════════════════════════════════════════════════════
# History mode: expired + live contracts → futures_contracts → panel → borrow
# ══════════════════════════════════════════════════════════════════════════════

_HISTORY_INTERVALS = ("5minute", "15minute", "1day")
_API_INTERVAL = {"5minute": "5minute", "15minute": "15minute", "1day": "day"}
_V3_UNIT = {"5minute": ("minutes", 5), "15minute": ("minutes", 15), "1day": ("days", 1)}
_CHUNK_DAYS = {"5minute": 28, "15minute": 28, "1day": 365}
_BARS_PER_DAY = {"5minute": 75, "15minute": 25, "1day": 1}
_LAST_BAR = {"5minute": "15:25", "15minute": "15:15"}   # session's final candle start
_HISTORY_LOOKBACK_DAYS = 180   # per new contract, as src fetch_expired_symbol (fills F2/F3)
_ROLL_ZONE_DTE = 0             # src data/config.py
_BORROW_R = 0.0625             # src update_intraday.py R
_MIN_TRACED_SHARE = 0.99       # stored values that must trace to stored contracts before a rebuild may replace them

CONTRACT_COLUMNS = ["open", "high", "low", "close", "volume", "oi",
                    "instrument_token", "symbol", "expiry", "interval", "dte"]
PANEL_COLUMNS = (
    [f"F{k}_{f}" for f in ("open", "high", "low", "close", "volume", "oi", "dte", "expiry", "ticker")
     for k in (1, 2, 3)]
    + ["cycle_id", "is_expiry_day", "symbol", "interval"]
    + [f"F{k}_oi_filled" for k in (1, 2, 3)]
)
_PANEL_INT = [f"F{k}_{f}" for f in ("volume", "oi", "dte") for k in (1, 2, 3)]
_PANEL_STR = [f"F{k}_{f}" for f in ("expiry", "ticker") for k in (1, 2, 3)]
BORROW_COLUMNS = ["F1_dte_star", "F2_dte_star", "F3_dte_star", "OI_ratio", "U_t",
                  "spread_spot_f1", "spread_spot_f2", "spread_f1_f2",
                  "spot_f1_pct", "f1_f2_pct",
                  "b1", "b2", "b3", "b12", "b23", "b13", "in_backwardation",
                  "both_traded", "f2_thin_early", "data_missing", "missing_inputs"]
MISSING_FLAG_COLUMNS = ["data_missing", "missing_inputs"]
# Columns added after the original src layout; stored series lacking them get them
# derived from their own stored values (add_missing_borrow_columns), nothing recomputed.
DERIVED_PCT_COLUMNS = ["spot_f1_pct", "f1_f2_pct"]
TRADE_FLAG_COLUMNS = ["both_traded", "f2_thin_early"]
_THIN_F2_SHARE = 0.10          # f2_thin_early: F2 bar volume below this share of F1's
_EARLY_TD = 10                 # ... at 10+ trading days to F1 expiry (before the last 10)
_HISTORY_SNAPSHOT_LIBS = ("futures", "futures_contracts", "borrow_rates")


def contract_symbol(symbol: str, interval: str, expiry: date) -> str:
    return f"{symbol}/{interval}/{expiry.isoformat()}"


def discover_contracts(fetcher: UpstoxFetcher, ticker: str, to_date: date) -> dict[date, dict]:
    """All futures contracts Upstox knows for *ticker*: expired (expired-instruments
    API) plus live (search API). Returns {expiry: {"key", "expired"}} sorted by expiry."""
    spot_key = fetcher.resolve_spot_key(ticker)
    if not spot_key:
        raise ValueError(f"could not resolve '{ticker}' to an NSE instrument key")
    out: dict[date, dict] = {}
    try:
        expiries = fetcher.fetch_expiries(spot_key)
    except Exception as exc:  # noqa: BLE001 — a ticker with no expired history still has live contracts
        logger.warning(f"{ticker}: fetch_expiries failed ({exc}); live contracts only")
        expiries = []
    for exp in expiries:
        if exp > to_date:
            continue
        try:
            recs = fetcher.fetch_future_contracts(spot_key, exp)
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"{ticker}: future contract lookup failed for {exp}: {exc}")
            continue
        if recs:
            out[exp] = {"key": recs[0]["instrument_key"], "expired": True}
    for rec in fetcher.fetch_live_future_contracts(ticker):
        exp = date.fromisoformat(str(rec.get("expiry", ""))[:10])
        if exp not in out:
            out[exp] = {"key": rec["instrument_key"], "expired": False}
    return dict(sorted(out.items()))


def _chunks(lo: date, hi: date, days: int):
    cur = lo
    while cur <= hi:
        end = min(cur + timedelta(days=days - 1), hi)
        yield cur, end
        cur = end + timedelta(days=1)


def bars_to_contract_frame(bars: list[Bar], symbol: str, expiry: date, interval: str,
                           token: str) -> pd.DataFrame:
    """Candles → stored ``futures_contracts`` schema, with src ``_parse_candles`` timestamp
    rules: daily bars normalised to midnight; intraday floored to the candle boundary and
    kept inside 09:15–15:30 (drops the expired endpoint's offset and 03:30 summary bars)."""
    if not bars:
        return pd.DataFrame(columns=CONTRACT_COLUMNS)
    df = pd.DataFrame([b.to_dict() for b in bars])
    idx = pd.DatetimeIndex(pd.to_datetime(df["timestamp"], utc=True)).tz_convert(_IST)
    if interval == "1day":
        idx = idx.normalize()
    else:
        idx = idx.floor(f"{_V3_UNIT[interval][1]}min")
    df.index = idx
    df.index.name = "timestamp"
    if interval != "1day":
        mins = df.index.hour * 60 + df.index.minute
        df = df[(mins >= 9 * 60 + 15) & (mins <= 15 * 60 + 30)]
    df = df[~df.index.duplicated(keep="last")].sort_index()
    out = pd.DataFrame(index=df.index)
    for c in ("open", "high", "low", "close"):
        out[c] = df[c].astype("float64")
    out["volume"] = df["volume"].fillna(0).astype("int64")
    out["oi"] = df["oi"].fillna(0).astype("int64")
    out["instrument_token"] = token
    out["symbol"] = symbol
    out["expiry"] = expiry.isoformat()
    out["interval"] = interval
    out["dte"] = np.maximum(0, (pd.Timestamp(expiry) - out.index.tz_localize(None).normalize()).days).astype("int64")
    return out[CONTRACT_COLUMNS]


def fetch_contract_window(fetcher: UpstoxFetcher, key: str, expired: bool, interval: str,
                          lo: date, hi: date) -> tuple[list[Bar], int]:
    """Fetch [lo, hi] in chunks; expired → v2 expired endpoint, live → v3. Returns (bars, n_failed_chunks)."""
    by_ts: dict = {}
    failed = 0
    for clo, chi in _chunks(lo, hi, _CHUNK_DAYS[interval]):
        try:
            if expired:
                bars = fetcher.fetch_expired_candles(key, _API_INTERVAL[interval], clo, chi)
            else:
                unit, mult = _V3_UNIT[interval]
                bars = fetcher.fetch_historical_candles_v3(key, unit, mult, clo, chi)
        except Exception as exc:  # noqa: BLE001 — one bad chunk must not abort the run
            logger.warning(f"    chunk {clo}→{chi} failed for {key}: {exc}")
            failed += 1
            continue
        for b in bars:
            by_ts[b.timestamp] = b
    return sorted(by_ts.values(), key=lambda b: b.timestamp), failed


def contract_fetch_window(store: ArcticStore, symbol: str, interval: str, expiry: date,
                          to_date: date) -> tuple[date, date] | None:
    """Window still missing for one contract, or None when it is already complete.

    Not stored → its lifecycle [expiry − 180d, min(expiry, to_date)]. Stored → from the
    day of its last print (inclusive, so a partial last day is repaired) to
    min(expiry, to_date); None if that last print is the expiry day's final candle.
    """
    end = min(expiry, to_date)
    sym = contract_symbol(symbol, interval, expiry)
    if not store.has_symbol("futures_contracts", sym):
        return (expiry - timedelta(days=_HISTORY_LOOKBACK_DAYS), end)
    _, last = store.stored_date_span("futures_contracts", sym)
    last_day = last.date()
    if last_day > end:
        return None
    if last_day == end and (end < to_date or end == expiry):
        if interval == "1day" or last.strftime("%H:%M") >= _LAST_BAR[interval]:
            return None
    return (last_day, end)


# ── Wide panel (src data/slot.py rules) ───────────────────────────────────────

def build_panel_from_contracts(contracts: list[pd.DataFrame], symbol: str, interval: str) -> pd.DataFrame:
    """F1/F2/F3 wide panel in the canonical stored layout from per-contract frames.

    Contracts are ranked by expiry once per trading DATE (not per bar); while the
    front contract's DTE < ROLL_ZONE_DTE ranks shift up by one. cycle_id = cumulative
    count of F1-expiry changes (F1_expiry ffilled/bfilled first); is_expiry_day = the
    bar's date is a date on which some contract had DTE 0.

    Upstox reports OI = 0 on many bars that have a price: there the contract's previous
    non-zero OI is carried forward (within that contract only) and ``F{k}_oi_filled``
    marks the bar. A zero before the contract's first OI print stays 0. Slots with no
    candle are not filled.
    """
    frames = [c for c in contracts if len(c)]
    if not frames:
        return pd.DataFrame(columns=PANEL_COLUMNS)
    long = pd.concat(frames)
    long = long[~long.reset_index().duplicated(["timestamp", "expiry"]).to_numpy()]
    long = long.sort_index(kind="stable")
    zero_oi = long["close"].notna() & (long["oi"] <= 0)
    filled_oi = long["oi"].where(~zero_oi).groupby(long["expiry"]).ffill()
    long["oi_filled"] = zero_oi & filled_oi.notna()
    long["oi"] = filled_oi.fillna(0).astype("int64")
    trade_date = long.index.normalize()
    expiry_days = set(long.index[long["dte"].to_numpy() == 0].strftime("%Y-%m-%d"))
    rank = long.groupby(trade_date)["expiry"].rank(method="dense").astype(int) - 1
    front_dte = long["dte"].where(rank == 0).groupby(trade_date).transform("first")
    slot_idx = rank - (front_dte < _ROLL_ZONE_DTE).astype(int)
    long = long[(slot_idx >= 0) & (slot_idx <= 2)].copy()
    long["slot"] = slot_idx[(slot_idx >= 0) & (slot_idx <= 2)].map({0: "F1", 1: "F2", 2: "F3"})

    fields = ["open", "high", "low", "close", "volume", "oi", "dte", "expiry", "symbol", "oi_filled"]
    mi = long.set_index("slot", append=True)[fields]
    mi = mi[~mi.index.duplicated(keep="first")]
    wide = mi.unstack("slot")
    wide.columns = [f"{slot}_{field}" for field, slot in wide.columns]
    wide = wide.rename(columns={f"F{k}_symbol": f"F{k}_ticker" for k in (1, 2, 3)})
    fill = {"open": np.nan, "high": np.nan, "low": np.nan, "close": np.nan,
            "volume": 0, "oi": 0, "dte": 0, "expiry": None, "ticker": None, "oi_filled": False}
    for k in (1, 2, 3):   # every slot present even when raw data is sparse
        for f, v in fill.items():
            if f"F{k}_{f}" not in wide.columns:
                wide[f"F{k}_{f}"] = v
    for c in _PANEL_STR:   # empty slot label = missing (None), as in the stored panels
        wide[c] = wide[c].astype(object).where(wide[c].notna(), None)
    for c in _PANEL_INT:
        wide[c] = wide[c].fillna(0).astype("int64")
    for k in (1, 2, 3):
        wide[f"F{k}_oi_filled"] = wide[f"F{k}_oi_filled"].astype("boolean").fillna(False).astype(bool)
    wide = wide.sort_index()
    f1 = wide["F1_expiry"].replace("", pd.NA).ffill().bfill()
    wide["F1_expiry"] = f1.astype(str)
    wide["cycle_id"] = (f1 != f1.shift(1)).cumsum().astype("int64")
    wide["is_expiry_day"] = pd.Index(wide.index.strftime("%Y-%m-%d")).isin(expiry_days)
    wide["symbol"] = symbol
    wide["interval"] = interval
    for k in (1, 2, 3):
        for f in ("open", "high", "low", "close"):
            wide[f"F{k}_{f}"] = wide[f"F{k}_{f}"].astype("float64")
    return wide[PANEL_COLUMNS]


# ── Borrow rates (src update_intraday.py formulas) ────────────────────────────

def spot_f1_pct(spread_spot_f1: pd.Series, spot: pd.Series) -> pd.Series:
    """(spot − F1) / spot × 100 — near basis in percent of spot."""
    return (spread_spot_f1 / spot.where(spot > 0) * 100).astype("float64")


def f1_f2_pct(spread_f1_f2: pd.Series, f1: pd.Series) -> pd.Series:
    """(F1 − F2) / F1 × 100 — calendar spread in percent of F1."""
    return (spread_f1_f2 / f1.where(f1 > 0) * 100).astype("float64")


def trade_flags(panel: pd.DataFrame) -> pd.DataFrame:
    """``both_traded`` and ``f2_thin_early`` per bar from the panel's F1/F2 volume and F1 expiry.

    both_traded   = F1_volume > 0 and F2_volume > 0 (else a leg's close is a stale price);
    f2_thin_early = trading days to F1 expiry >= 10, both legs traded, F2 volume < 10% of F1's.
    Missing volume counts as 0; a missing or unparsable F1 expiry is never "early".
    """
    idx = panel.index
    zero = pd.Series(0.0, index=idx)
    v1 = panel["F1_volume"].fillna(0).astype("float64") if "F1_volume" in panel else zero
    v2 = panel["F2_volume"].fillna(0).astype("float64") if "F2_volume" in panel else zero
    early = np.zeros(len(idx), dtype=bool)
    if "F1_expiry" in panel and len(idx):
        days = idx.date
        exps = panel["F1_expiry"].astype(str).str[:10].to_numpy()
        td = {}
        for d, e in set(zip(days, exps)):
            try:
                td[(d, e)] = trading_days_inclusive(d, date.fromisoformat(e)) - 1
            except ValueError:
                pass
        early = np.array([td.get(k, -1) >= _EARLY_TD for k in zip(days, exps)], dtype=bool)
    both = ((v1 > 0) & (v2 > 0)).to_numpy()
    thin = early & both & (v2 < _THIN_F2_SHARE * v1).to_numpy()
    return pd.DataFrame({"both_traded": both, "f2_thin_early": thin}, index=idx)


def compute_borrow(panel: pd.DataFrame, interval: str, r: float = _BORROW_R) -> pd.DataFrame:
    """Borrow-rate columns from a panel carrying SPOT_close (read_futures(with_spot=True)).

    dte_star = (dte + 1) − bar_rank_in_day / bars_per_day; b_i = r − ln(F_i/S)/t_i,
    b_ij = r − ln(F_j/F_i)/(t_j − t_i), t in years (/365), all clipped >= 0;
    OI_ratio = F1_oi/F2_oi and U_t = OI_ratio/(F1_dte_star + 0.04·is_expiry_day), intraday only.
    spot_f1_pct = (S − F1)/S × 100 and f1_f2_pct = (F1 − F2)/F1 × 100, in percent
    (NaN where an input is missing, e.g. spot outside the Kite series).
    Callers pass whole trading days so the per-day bar ranks are right.
    """
    df = panel.sort_index()
    out = pd.DataFrame(index=df.index)
    rank = df.groupby(df.index.normalize()).cumcount() + 1
    bpd = _BARS_PER_DAY[interval]
    for k in (1, 2, 3):
        out[f"F{k}_dte_star"] = (df[f"F{k}_dte"] + 1) - rank / bpd
    S = df["SPOT_close"] if "SPOT_close" in df.columns else pd.Series(np.nan, index=df.index)
    F = {k: df[f"F{k}_close"] for k in (1, 2, 3)}
    t = {k: out[f"F{k}_dte_star"] / 365 for k in (1, 2, 3)}

    def logr(num, den):
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where((num > 0) & (den > 0), np.log(num / den), np.nan)

    def rate(num, den, dt):
        return pd.Series(np.where(dt > 0, r - logr(num, den) / dt, np.nan), index=df.index).clip(lower=0)

    if interval != "1day":
        out["OI_ratio"] = df["F1_oi"] / df["F2_oi"].replace(0, np.nan)
        out["U_t"] = out["OI_ratio"] / (out["F1_dte_star"] + df["is_expiry_day"].astype(float) * 0.04)
    out["spread_spot_f1"] = S - F[1]
    out["spread_spot_f2"] = S - F[2]
    out["spread_f1_f2"] = F[1] - F[2]
    out["spot_f1_pct"] = spot_f1_pct(out["spread_spot_f1"], S)
    out["f1_f2_pct"] = f1_f2_pct(out["spread_f1_f2"], F[1])
    out["b1"] = rate(F[1], S, t[1])
    out["b2"] = rate(F[2], S, t[2])
    out["b3"] = rate(F[3], S, t[3])
    out["b12"] = rate(F[2], F[1], t[2] - t[1])
    out["b23"] = rate(F[3], F[2], t[3] - t[2])
    out["b13"] = rate(F[3], F[1], t[3] - t[1])
    out["in_backwardation"] = (F[1] < S).astype(bool)
    flags = trade_flags(df)
    for c in TRADE_FLAG_COLUMNS:
        out[c] = flags[c].to_numpy()
    miss = missing_flags(F[1], F[2], S)
    for c in MISSING_FLAG_COLUMNS:
        out[c] = miss[c].to_numpy()
    cols = [c for c in BORROW_COLUMNS if c in out.columns]
    return out[cols]


def missing_flags(f1: pd.Series, f2: pd.Series, spot: pd.Series) -> pd.DataFrame:
    """``data_missing`` / ``missing_inputs`` per bar: which of the F1, F2 and spot closes
    are missing (NaN or non-positive). Nothing is filled — the rates that need a missing
    input stay NaN. ``missing_inputs`` is e.g. "F2" or "F1,F2,SPOT"; "" when complete."""
    parts = [(name, ~(s > 0)) for name, s in (("F1", f1), ("F2", f2), ("SPOT", spot))]
    labels = pd.Series("", index=f1.index, dtype=object)
    for name, m in parts:
        labels = labels.where(~m, labels + "," + name)
    labels = labels.str.lstrip(",")
    return pd.DataFrame({"data_missing": labels != "", "missing_inputs": labels}, index=f1.index)


def _align_to_stored(store: ArcticStore, library: str, symbol: str, df: pd.DataFrame,
                     existing: pd.DataFrame | None = None, keep_new: bool = False) -> pd.DataFrame:
    """Reorder *df* to the stored symbol's columns (static schema). Columns the stored
    symbol has but *df* lacks (e.g. ``feat_*``) are taken from *existing* by timestamp
    when given, else NaN. *keep_new* (full rewrites only — ``update`` cannot add columns)
    appends columns of *df* the stored schema lacks instead of dropping them."""
    lib = store._lib(library)
    if not lib.has_symbol(symbol):
        return df
    cols = [c.name for c in lib.get_description(symbol).columns]
    out = df.copy()
    for c in cols:
        if c not in out.columns:
            if existing is not None and c in existing.columns:
                out[c] = existing[c].reindex(out.index)
            else:
                out[c] = np.nan
    if keep_new:
        cols = cols + [c for c in df.columns if c not in cols]
    return out[cols]


def _traced_share(stored_rows: pd.DataFrame, contract_frames: list[pd.DataFrame]) -> float:
    """Share of the stored panel's F1/F2 closes (on *stored_rows*) that equal the close of
    the contract named by that row's F1/F2_expiry at the same bar. NaN slots count as traced."""
    closes = {f["expiry"].iloc[0]: f["close"] for f in contract_frames if len(f)}
    ok = total = 0
    for slot in ("F1", "F2"):
        px, exp = stored_rows[f"{slot}_close"], stored_rows[f"{slot}_expiry"]
        for e, grp in px.groupby(exp.fillna("")):
            total += len(grp)
            if e == "" or grp.isna().all():
                ok += int(grp.isna().sum()) if e == "" else len(grp)
                continue
            src = closes.get(e)
            if src is None:
                continue
            ref = src.reindex(grp.index)
            ok += int(((ref == grp) | (ref.isna() & grp.isna())).sum())
    return ok / total if total else 1.0


def _snapshot_once(store: ArcticStore, name: str) -> None:
    for libname in _HISTORY_SNAPSHOT_LIBS:
        lib = store._lib(libname)
        if name not in lib.list_snapshots():
            lib.snapshot(name)
            logger.info(f"snapshot {libname}@{name}")


def update_panel_and_borrow(store: ArcticStore, symbol: str, interval: str,
                            contract_frames: list[pd.DataFrame], added_before_start: bool,
                            force_full: bool = False) -> dict:
    """Rebuild the panel from all stored contracts and write it plus the borrow rates.

    Tail-only (default): rows from the start of the stored panel's last day onward are
    upserted; cycle_id continues the stored numbering. Full rewrite when the symbol is
    new, older contracts were added in front, or the rebuild re-slots stored rows whose
    values all trace to stored contracts (the rebuild is then authoritative). If stored
    values don't trace to stored contracts, the stored rows are kept and only the tail
    is extended. *force_full* (``--rebuild-from-stored``) always rewrites in full — the
    rebuild may differ only in columns the overlap check does not compare (e.g. filled
    OI) — but still keeps the stored panel when stored rows have no source contract.
    """
    fsym = store.futures_symbol(symbol, interval)
    bsym = store.borrow_symbol(symbol, interval)
    rebuilt = build_panel_from_contracts(contract_frames, symbol, interval)
    if rebuilt.empty:
        return {"panel": "no data"}
    has_panel = store.has_symbol("futures", fsym)
    # Full rewrite when the contracts reach back before the stored panel (whether added
    # this run or an earlier one), so a rerun also completes an earlier partial fill.
    starts_earlier = has_panel and rebuilt.index.min() < store.stored_date_span("futures", fsym)[0]
    mode = "full" if (not has_panel or added_before_start or starts_earlier) else "tail"

    if has_panel:
        stored = store.read("futures", fsym)
        common = stored.index.intersection(rebuilt.index)
        chk = ["F1_close", "F2_close", "F1_expiry", "F2_expiry"]
        a, b = stored.loc[common, chk], rebuilt.loc[common, chk]
        mism = common[(~((a == b) | (a.isna() & b.isna()))).any(axis=1).to_numpy()]
        diff = len(mism)
        # Where the rebuild differs, check the stored values trace back to stored
        # contracts (stored F1/F2 close == that contract's close at that bar). If they
        # do, the difference is only re-slotting — e.g. contracts missing when the panel
        # was first built, or a contract's expiry-day bars now fetched — and the rebuild
        # is authoritative: rewrite in full. If they don't, the stored panel came from
        # data the contracts lack: keep it and only extend the tail.
        traced = _traced_share(stored.loc[mism], contract_frames) if diff else 1.0
        if diff and traced < _MIN_TRACED_SHARE:
            logger.warning(f"{fsym}: rebuild differs on {diff} rows and only {traced:.1%} of the stored "
                           f"values trace to stored contracts — keeping stored rows, tail-only update")
            mode = "tail"
        elif diff:
            mode = "full"
        mism_days = []
        stored_only = stored.index.difference(rebuilt.index)
        if force_full:
            if (diff and traced < _MIN_TRACED_SHARE) or len(stored_only):
                reason = (f"{len(stored_only)} stored rows have no source contract" if len(stored_only)
                          else f"{diff} rows differ and only {traced:.1%} trace to stored contracts")
                logger.warning(f"{fsym}: forced rebuild skipped — {reason}; stored panel kept")
                return {"panel": f"skipped: {reason}", "mode": "none", "overlap_mismatch": diff}
            mode = "full"
        if mode == "full" and len(stored_only):
            logger.warning(f"{fsym}: {len(stored_only)} stored rows have no source contract "
                           f"(e.g. {stored_only[0]}) — a full rewrite would drop them; tail-only update")
            mode = "tail"
        if mode == "tail":
            tail_start = stored.index.max().normalize()
            anchor_exp = stored["F1_expiry"].iloc[-1]
            anchor_cyc = int(stored["cycle_id"].iloc[-1])
            hit = rebuilt.loc[rebuilt["F1_expiry"] == anchor_exp, "cycle_id"]
            new = rebuilt[rebuilt.index >= tail_start].copy()
            offset = (anchor_cyc - int(hit.iloc[-1]) if len(hit)
                      else anchor_cyc - int(new["cycle_id"].iloc[0]) + 1 if len(new) else 0)
            repaired = [d for d in mism_days if d < tail_start]
            if new.empty and not repaired:
                return {"panel": "up to date", "mode": mode}
            for day in repaired:   # rewrite each corrected day in place
                rows = rebuilt[rebuilt.index.normalize() == day].copy()
                rows["cycle_id"] = (rows["cycle_id"] + offset).astype("int64")
                store.update("futures", fsym, _align_to_stored(store, "futures", fsym, rows))
                write_borrow(store, symbol, interval, day, to_ts=day + pd.Timedelta(days=1))
            if len(new):
                new["cycle_id"] = (new["cycle_id"] + offset).astype("int64")
                store.update("futures", fsym, _align_to_stored(store, "futures", fsym, new))
            panel_info = {"panel": f"tail {len(new)} rows from {tail_start.date()}", "mode": mode,
                          "overlap_mismatch": diff,
                          "repaired_days": [str(d.date()) for d in repaired]}
            borrow_from = tail_start if len(new) else None
            if borrow_from is None:
                panel_info["borrow"] = "repaired days only"
                return panel_info
        else:
            store.write("futures", fsym, _align_to_stored(store, "futures", fsym, rebuilt, keep_new=True),
                        metadata={"source": "fetch_ticker --history", "rebuilt": date.today().isoformat()})
            panel_info = {"panel": f"full {len(rebuilt)} rows", "mode": mode, "overlap_mismatch": diff}
            borrow_from = None
    else:
        store.write("futures", fsym, rebuilt,
                    metadata={"source": "fetch_ticker --history", "rebuilt": date.today().isoformat()})
        panel_info = {"panel": f"new {len(rebuilt)} rows", "mode": mode}
        borrow_from = None

    panel_info["borrow"] = write_borrow(store, symbol, interval, borrow_from)
    return panel_info


def write_borrow(store: ArcticStore, symbol: str, interval: str, from_ts: pd.Timestamp | None,
                 to_ts: pd.Timestamp | None = None) -> str:
    """(Re)compute borrow rates from the stored panel + Kite spot and write them.
    *from_ts* None → full series; else upsert rows from that day onward (up to *to_ts*,
    exclusive, when given — used to rewrite single repaired days)."""
    bsym = store.borrow_symbol(symbol, interval)
    rng = None if from_ts is None else (from_ts.normalize(),
                                        None if to_ts is None else to_ts - pd.Timedelta(1, "ns"))
    panel = store.read_futures(symbol, interval, with_spot=True, date_range=rng)
    if panel.empty:
        return "no panel rows"
    br = compute_borrow(panel, interval)
    has = store.has_symbol("borrow_rates", bsym)
    if from_ts is not None and has:
        # static schema: give an older series the new columns first, so update() matches
        add_missing_borrow_columns(store, symbol, interval)
        store.update("borrow_rates", bsym, _align_to_stored(store, "borrow_rates", bsym, br))
        return f"tail {len(br)} rows from {from_ts.date()}"
    existing = store.read("borrow_rates", bsym) if has else None
    store.write("borrow_rates", bsym,
                _align_to_stored(store, "borrow_rates", bsym, br, existing, keep_new=True),
                metadata={"source": "fetch_ticker --history (update_intraday formulas)",
                          "spot_source": "kite spot library", "r": _BORROW_R,
                          "built": date.today().isoformat()})
    return f"full {len(br)} rows"


def add_missing_borrow_columns(store: ArcticStore, symbol: str, interval: str,
                               dry_run: bool = False) -> str:
    """Give an existing ``borrow_rates/{symbol}/{interval}`` series any of
    ``DERIVED_PCT_COLUMNS`` / ``TRADE_FLAG_COLUMNS`` it lacks:

        spot_f1_pct = stored spread_spot_f1 / Kite SPOT_close × 100
        f1_f2_pct   = stored spread_f1_f2   / panel F1_close  × 100
        both_traded, f2_thin_early = trade_flags(panel F1/F2 volume, F1 expiry)

    Nothing else is recomputed — every stored column (b1…b13, dte_star, spreads,
    feat_*, …) is written back unchanged. One full write (the library has a static
    schema, so ``update`` cannot add columns); the previous version is kept.
    """
    bsym = store.borrow_symbol(symbol, interval)
    if not store.has_symbol("borrow_rates", bsym):
        return "no series"
    stored_cols = [c.name for c in store._lib("borrow_rates").get_description(bsym).columns]
    missing = [c for c in DERIVED_PCT_COLUMNS + TRADE_FLAG_COLUMNS if c not in stored_cols]
    if not missing:
        return "up to date"
    if dry_run:
        return f"would add {missing}"
    stored = store.read("borrow_rates", bsym)
    rng = (stored.index.min(), stored.index.max())
    out = stored.copy()
    meta = dict(store.read_metadata("borrow_rates", bsym) or {})
    pct = [c for c in missing if c in DERIVED_PCT_COLUMNS]
    if pct:
        inputs = store.read_futures(symbol, interval, with_spot=True, date_range=rng,
                                    columns=["F1_close", "SPOT_close"])
        inputs = inputs[~inputs.index.duplicated(keep="last")].reindex(stored.index)
        if "spot_f1_pct" in pct:
            out["spot_f1_pct"] = spot_f1_pct(stored["spread_spot_f1"], inputs["SPOT_close"])
        if "f1_f2_pct" in pct:
            out["f1_f2_pct"] = f1_f2_pct(stored["spread_f1_f2"], inputs["F1_close"])
        meta["pct_columns_added"] = {"columns": pct, "date": date.today().isoformat(),
                                     "from": "stored spread_spot_f1 / spread_f1_f2"}
    flags = [c for c in missing if c in TRADE_FLAG_COLUMNS]
    if flags:
        vol = store.read_futures(symbol, interval, with_spot=False, date_range=rng,
                                 columns=["F1_volume", "F2_volume", "F1_expiry"])
        vol = vol[~vol.index.duplicated(keep="last")].reindex(stored.index)
        tf = trade_flags(vol)
        for c in flags:
            out[c] = tf[c].to_numpy()
        meta["trade_flags_added"] = {"columns": flags, "date": date.today().isoformat(),
                                     "rule": f"both legs volume > 0; thin = F2 vol < {_THIN_F2_SHARE:.0%} "
                                             f"of F1 vol at >= {_EARLY_TD} trading days to expiry"}
    store.write("borrow_rates", bsym, out, metadata=meta)
    return f"added {missing}"


def all_panel_tickers(store: ArcticStore, intervals: list[str]) -> list[str]:
    """Tickers with a wide F1/F2/F3 panel (``futures/{T}/{iv}`` carrying F1_close) at any
    of *intervals* — the per-expiry ``futures/{T}/{iv}/{expiry}`` symbols are skipped."""
    lib = store._lib("futures")
    out = set()
    for sym in lib.list_symbols():
        parts = sym.split("/")
        if len(parts) == 2 and parts[1] in intervals and store._is_wide_panel(parts[0], parts[1]):
            out.add(parts[0])
    return sorted(out)


def fill_missing(store: ArcticStore, symbol: str, interval: str, dry_run: bool = False) -> str:
    """--fill-missing: no API calls. Panel but no borrow series → compute the full borrow
    series; borrow series present → only add missing DERIVED_PCT_COLUMNS (from its stored
    values) and TRADE_FLAG_COLUMNS (from the panel's volumes); no panel → skip."""
    if not store._is_wide_panel(symbol, interval):
        return "no panel - needs fetch-ticker --history"
    if not store.has_symbol("borrow_rates", store.borrow_symbol(symbol, interval)):
        return "would compute full borrow series" if dry_run else f"computed {write_borrow(store, symbol, interval, None)}"
    return add_missing_borrow_columns(store, symbol, interval, dry_run=dry_run)


def run_history(args) -> int:
    """--history entry point: per ticker and interval, fetch missing contract windows,
    then rebuild/write the panel and borrow rates."""
    to_date = args.to_date or date.today()
    intervals = [iv for iv in args.intervals if iv in _HISTORY_INTERVALS]
    bad = [iv for iv in args.intervals if iv not in _HISTORY_INTERVALS]
    if bad:
        logger.error(f"--history supports {list(_HISTORY_INTERVALS)}; got {bad}")
        return 2
    offline = args.fill_missing or args.borrow_only or args.rebuild_from_stored
    fetcher = None if offline else UpstoxFetcher()
    store = ArcticStore()
    if not args.dry_run:
        _snapshot_once(store, f"pre_fetch_ticker_history_{date.today().isoformat()}")

    summary = []
    tickers = all_panel_tickers(store, intervals) if args.all else args.ticker
    if args.all:
        logger.info(f"--all: {len(tickers)} tickers with a wide futures panel at {intervals}")
    for raw in tickers:
        symbol = raw.upper().strip()
        if args.fill_missing:
            for iv in intervals:
                summary.append((symbol, iv, fill_missing(store, symbol, iv, args.dry_run)))
            continue
        if args.borrow_only:
            for iv in intervals:
                if not store.has_symbol("futures", store.futures_symbol(symbol, iv)):
                    summary.append((symbol, iv, "no panel")); continue
                frm = pd.Timestamp(args.from_date, tz=_IST) if args.from_date else None
                res = "dry-run" if args.dry_run else write_borrow(store, symbol, iv, frm)
                summary.append((symbol, iv, f"borrow {res}"))
            continue
        if args.rebuild_from_stored:
            for iv in intervals:
                frames = [store.read("futures_contracts", s) for s in store.list_symbols("futures_contracts")
                          if s.startswith(f"{symbol}/{iv}/")]
                if not frames:
                    summary.append((symbol, iv, "no stored contracts")); continue
                if args.dry_run:
                    summary.append((symbol, iv, f"dry-run: would rebuild from {len(frames)} stored contracts"))
                    continue
                info = update_panel_and_borrow(store, symbol, iv, frames, False, force_full=True)
                logger.info(f"     {store.futures_symbol(symbol, iv)}: {info}")
                summary.append((symbol, iv, str(info)))
            continue
        try:
            contracts = discover_contracts(fetcher, symbol, to_date)
        except Exception as exc:  # noqa: BLE001
            logger.error(f"{symbol}: {exc}")
            summary.append((symbol, "-", f"error: {exc}"))
            continue
        logger.info(f"\n{'═' * 66}\n{symbol}: {len(contracts)} contracts "
                    f"({min(contracts)} → {max(contracts)})" if contracts else f"{symbol}: no contracts")
        for iv in intervals:
            fsym = store.futures_symbol(symbol, iv)
            panel_start = store.stored_date_span("futures", fsym)[0] if store.has_symbol("futures", fsym) else None
            plan = []
            for exp, meta in contracts.items():
                win = contract_fetch_window(store, symbol, iv, exp, to_date)
                if win is not None and win[0] <= win[1]:
                    plan.append((exp, meta, win))
            est = sum(len(list(_chunks(w[0], w[1], _CHUNK_DAYS[iv]))) for _, _, w in plan)
            logger.info(f"── {symbol} {iv}: {len(plan)}/{len(contracts)} contracts to fetch, ~{est} calls"
                        + (f" | stored panel from {panel_start.date()}" if panel_start is not None else " | no panel"))
            if args.dry_run:
                for exp, meta, (lo, hi) in plan:
                    tag = "new " if not store.has_symbol("futures_contracts", contract_symbol(symbol, iv, exp)) else "tail"
                    logger.info(f"     {exp} [{'expired' if meta['expired'] else 'live   '}] {tag} {lo} → {hi}")
                summary.append((symbol, iv, f"dry-run: {len(plan)} contracts, ~{est} calls"))
                continue

            added_before_start = False
            failed_total = 0
            for exp, meta, (lo, hi) in plan:
                csym = contract_symbol(symbol, iv, exp)
                is_new = not store.has_symbol("futures_contracts", csym)
                bars, failed = fetch_contract_window(fetcher, meta["key"], meta["expired"], iv, lo, hi)
                failed_total += failed
                token = "|".join(meta["key"].split("|")[:2])
                df = bars_to_contract_frame(bars, symbol, exp, iv, token)
                if df.empty:
                    continue
                if is_new:
                    store.write("futures_contracts", csym, df)
                    if panel_start is not None and df.index.min() < panel_start:
                        added_before_start = True
                else:
                    store.update("futures_contracts", csym, df)
                logger.info(f"     {exp}: {len(df):>6} bars {df.index.min():%Y-%m-%d} → {df.index.max():%Y-%m-%d}"
                            f"{' (new)' if is_new else ''}")
            frames = [store.read("futures_contracts", s) for s in store.list_symbols("futures_contracts")
                      if s.startswith(f"{symbol}/{iv}/")]
            info = update_panel_and_borrow(store, symbol, iv, frames, added_before_start)
            info["failed_chunks"] = failed_total
            logger.info(f"     {fsym}: {info}")
            summary.append((symbol, iv, str(info)))

    logger.info("\n" + "═" * 66 + "\nSummary")
    for s, iv, msg in summary:
        logger.info(f"  {s:12s} {iv:9s} {msg}")
    logger.info(f"API calls: {fetcher.request_count if fetcher else 0}")
    return 0


# ── CLI ────────────────────────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Fetch NSE futures lifecycle data from Upstox API "
            "and store in ArcticDB as an F1/F2/F3 wide panel."
        )
    )
    ap.add_argument(
        "ticker",
        nargs="*",
        help=(
            "Ticker name as it appears in the JSON 'name' field "
            "(e.g. bajaj-auto, RVNL, NIFTY). Case-insensitive. Several tickers "
            "are accepted with --history."
        ),
    )
    ap.add_argument(
        "--json",
        dest="json_path",
        type=Path,
        default=_DEFAULT_JSON,
        metavar="PATH",
        help=f"Path to the Upstox instrument master JSON/GZ (default: {_DEFAULT_JSON})",
    )
    ap.add_argument(
        "--intervals",
        nargs="+",
        default=_DEFAULT_INTERVALS,
        choices=["1minute", "3minute", "5minute", "15minute", "30minute", "day", "1day"],
        metavar="INTERVAL",
        help="One or more intervals to fetch (default: day 15minute 5minute)",
    )
    ap.add_argument(
        "--from",
        dest="from_date",
        type=lambda s: date.fromisoformat(s),
        default=None,
        metavar="YYYY-MM-DD",
        help=f"Override from-date for every contract (default: expiry minus {_LIFECYCLE_DAYS} days)",
    )
    ap.add_argument(
        "--force",
        action="store_true",
        help="Re-fetch all contracts and overwrite existing Arctic data",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Print resolved contracts and exit without fetching or writing",
    )
    ap.add_argument(
        "--history",
        action="store_true",
        help="Expired + live contracts via the Upstox API, per-contract resume, panel + borrow "
             "rates in ArcticDB (intervals 5minute 15minute 1day). See module docstring.",
    )
    ap.add_argument(
        "--to",
        dest="to_date",
        type=lambda s: date.fromisoformat(s),
        default=None,
        metavar="YYYY-MM-DD",
        help="--history: last date to fetch (default: today)",
    )
    ap.add_argument(
        "--borrow-only",
        action="store_true",
        help="--history: skip fetching; recompute borrow rates from the stored panel "
             "(from --from onward, or the full series)",
    )
    ap.add_argument(
        "--fill-missing",
        action="store_true",
        help="--history (implied), no API calls: compute the borrow series where a futures "
             "panel exists without one, and add the spot_f1_pct / f1_f2_pct columns (from "
             "stored spreads) and the both_traded / f2_thin_early flags (from panel volumes) "
             "to existing borrow series. Nothing already stored is recomputed.",
    )
    ap.add_argument(
        "--rebuild-from-stored",
        action="store_true",
        help="--history, no API calls: rebuild the panel from the stored futures_contracts "
             "(OI carry-forward included) and recompute borrow rates, as a full rewrite. "
             "A ticker whose stored panel has rows no stored contract explains is skipped.",
    )
    ap.add_argument(
        "--all",
        action="store_true",
        help="--history: run every ticker that has a wide futures panel at --intervals "
             "(instead of naming tickers).",
    )
    args = ap.parse_args(argv)

    if args.fill_missing or args.rebuild_from_stored:
        args.history = True
    if args.all and not args.history:
        ap.error("--all needs --history (or --fill-missing)")
    if not args.ticker and not args.all:
        ap.error("give at least one ticker, or --all")

    if args.history:
        if args.intervals == _DEFAULT_INTERVALS:
            args.intervals = list(_HISTORY_INTERVALS)
        return run_history(args)
    if len(args.ticker) != 1 or "1day" in args.intervals:
        logger.error("Without --history: exactly one ticker, and use 'day' (not '1day').")
        return 2
    args.ticker = args.ticker[0]

    if not args.json_path.exists():
        logger.error(f"Instrument JSON not found: {args.json_path}")
        return 2

    logger.info(f"Loading instrument master from {args.json_path} …")
    instruments = load_instrument_json(args.json_path)
    logger.info(f"Loaded {len(instruments):,} instrument records")

    try:
        fut_records = resolve_futures_contracts(instruments, args.ticker)
    except ValueError as exc:
        logger.error(str(exc))
        return 2

    symbol = args.ticker.upper().strip()
    today = date.today()

    logger.info(f"Found {len(fut_records)} NSE_FO FUT contracts for {symbol}:")
    for r in fut_records:
        exp = record_expiry(r)
        status = "ACTIVE " if exp >= today else "EXPIRED"
        logger.info(
            f"  [{status}]  {r.get('trading_symbol', r['instrument_key']):<45}"
            f"  expiry={exp}  key={r['instrument_key']}"
        )

    if args.dry_run:
        return 0

    # Stable cycle_id: 1 = oldest expiry, 2 = next, …
    global_cycle = {r["instrument_key"]: idx + 1 for idx, r in enumerate(fut_records)}

    fetcher = UpstoxFetcher()
    store = ArcticStore()

    for interval in args.intervals:
        logger.info(f"\n{'─' * 60}")
        logger.info(f"Interval: {interval}")

        already_exists = store.has_symbol("futures", f"{symbol}/{interval}")
        is_incremental = already_exists and not args.force

        if is_incremental:
            # Fetch only the 3 most recently-expired contracts plus all active ones.
            # This is enough to correctly assign F1/F2/F3 slots for new timestamps.
            expired = [r for r in fut_records if record_expiry(r) < today]
            active = [r for r in fut_records if record_expiry(r) >= today]
            records_to_fetch = expired[-3:] + active
            logger.info(
                f"Incremental mode: {len(records_to_fetch)} contract(s) "
                f"({len(active)} active + up to 3 recent expired)"
            )
        else:
            records_to_fetch = fut_records
            mode = "full overwrite" if args.force and already_exists else "first write"
            logger.info(f"{mode.capitalize()}: fetching all {len(records_to_fetch)} contract(s)")

        contract_data: list[tuple[date, int, str, pd.DataFrame]] = []

        for r in records_to_fetch:
            exp = record_expiry(r)
            key = r["instrument_key"]
            cycle_id = global_cycle[key]
            sym_label = r.get("trading_symbol", key)

            logger.info(f"  Fetching {sym_label}  (cycle {cycle_id}, expiry {exp}) …")
            df = fetch_lifecycle(fetcher, key, exp, interval, args.from_date)

            if df.empty:
                logger.warning(f"    No data returned — skipping {sym_label}")
                continue

            logger.info(
                f"    {len(df):>5} bars  "
                f"[{df.index[0].date()} — {df.index[-1].date()}]"
            )
            contract_data.append((exp, cycle_id, key, df))

        if not contract_data:
            logger.info(f"No data fetched for {symbol}/{interval} — nothing to write")
            continue

        logger.info(f"Assembling wide panel from {len(contract_data)} contract(s) …")
        wide = assemble_wide_panel(contract_data)

        if wide.empty:
            logger.warning(f"Wide panel is empty for {symbol}/{interval} — skipping write")
            continue

        logger.info(f"Wide panel: {len(wide)} rows × {len(wide.columns)} columns")

        meta = {
            "source": "fetch_ticker",
            "symbol": symbol,
            "interval": interval,
            "contracts": [r["instrument_key"] for r in records_to_fetch],
            "fetched_at": today.isoformat(),
        }

        if is_incremental:
            n = store.append_futures(symbol, interval, wide)
            logger.info(f"Appended {n} new rows  →  futures/{symbol}/{interval}")
        else:
            store.write_futures(symbol, interval, wide, prune_previous=args.force, metadata=meta)
            logger.info(
                f"Wrote {len(wide)} rows "
                f"(prune_previous={'yes' if args.force else 'no'})  "
                f"→  futures/{symbol}/{interval}"
            )

    logger.info("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
