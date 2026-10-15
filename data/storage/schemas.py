"""Canonical schema definitions for the options backtesting engine.

Two canonical structures are defined here:

1.  **Options chain DataFrame** — MultiIndex (symbol, expiration, strike,
    option_type) with 15 data columns covering OHLCV, bid/ask/mid, Greeks,
    and DTE.  ``bid`` and ``ask`` are nullable so historical batches (which
    only carry OHLCV) can coexist with live snapshots (which carry bid/ask).

2.  **Underlying state DataFrame** — flat, indexed by ``symbol``, holding
    spot price, macro metrics, and volatility estimates.
"""

from __future__ import annotations

import pandas as pd
import pyarrow as pa


# ── MultiIndex level names ──────────────────────────────────────────────────
CHAIN_INDEX_NAMES: list[str] = ["symbol", "expiration", "strike", "option_type"]

# ── Ordered list of data columns ───────────────────────────────────────────
CHAIN_COLUMNS: list[str] = [
    "open", "high", "low", "close",
    "bid", "ask", "mid",
    "volume", "open_interest",
    "implied_vol", "delta", "gamma", "vega", "theta",
    "dte",
]

# ── Underlying state columns ────────────────────────────────────────────────
UNDERLYING_COLUMNS: list[str] = [
    "spot_price",
    "next_earnings_date",
    "rfr",
    "dividend_yield",
    "historic_volatility",
]

# ── PyArrow schema (index levels are stored as regular columns by pyarrow) ──
# Used for schema enforcement on read; pyarrow reinstates the MultiIndex
# automatically when the parquet file was written with index=True.
CHAIN_PARQUET_SCHEMA = pa.schema([
    # Index levels
    ("symbol",        pa.string()),
    ("expiration",    pa.timestamp("ns")),
    ("strike",        pa.float64()),
    ("option_type",   pa.string()),
    # Data columns — float64 is nullable in Arrow (NaN / null)
    ("open",          pa.float64()),
    ("high",          pa.float64()),
    ("low",           pa.float64()),
    ("close",         pa.float64()),
    ("bid",           pa.float64()),   # NaN for historical data
    ("ask",           pa.float64()),   # NaN for historical data
    ("mid",           pa.float64()),
    ("volume",        pa.float64()),
    ("open_interest", pa.float64()),
    ("implied_vol",   pa.float64()),
    ("delta",         pa.float64()),
    ("gamma",         pa.float64()),
    ("vega",          pa.float64()),
    ("theta",         pa.float64()),
    ("dte",           pa.float64()),   # fractional, intraday trading-day DTE
])


# ── Factory helpers ─────────────────────────────────────────────────────────

def make_empty_chain_df() -> pd.DataFrame:
    """Return a zero-row options chain DataFrame with the canonical MultiIndex.

    All float columns use pandas nullable ``Float64`` dtype so that
    ``pd.NA`` (not ``0.0``) represents missing bid/ask on historical rows.
    ``dte`` also uses nullable ``Float64`` (fractional trading-day DTE).
    """
    index = pd.MultiIndex.from_tuples([], names=CHAIN_INDEX_NAMES)
    df = pd.DataFrame(index=index, columns=CHAIN_COLUMNS)
    for col in CHAIN_COLUMNS:
        df[col] = pd.array([], dtype="Float64")
    return df


def make_empty_underlying_df() -> pd.DataFrame:
    """Return a zero-row underlying state DataFrame indexed by ``symbol``."""
    df = pd.DataFrame(columns=UNDERLYING_COLUMNS).rename_axis("symbol")
    for col in ["spot_price", "rfr", "dividend_yield", "historic_volatility"]:
        df[col] = pd.array([], dtype="Float64")
    df["next_earnings_date"] = pd.Series([], dtype="datetime64[ns]")
    return df


# ── Validation ──────────────────────────────────────────────────────────────

def validate_chain_df(df: pd.DataFrame) -> None:
    """Raise ``ValueError`` if *df* does not conform to the canonical chain schema.

    Checks:
    * DataFrame must have a ``pd.MultiIndex``.
    * Index names must be ``CHAIN_INDEX_NAMES`` in the correct order.
    * All ``CHAIN_COLUMNS`` must be present.
    """
    if not isinstance(df.index, pd.MultiIndex):
        raise ValueError(
            "Options chain DataFrame must have a pd.MultiIndex; "
            f"got {type(df.index).__name__}."
        )
    if list(df.index.names) != CHAIN_INDEX_NAMES:
        raise ValueError(
            f"MultiIndex names must be {CHAIN_INDEX_NAMES} "
            f"(in that order); got {list(df.index.names)}."
        )
    missing = set(CHAIN_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(
            f"Options chain DataFrame is missing required columns: "
            f"{sorted(missing)}."
        )
