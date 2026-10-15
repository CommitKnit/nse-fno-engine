"""build_enriched_chains — raw option candles → enriched chain rows → ArcticDB.

The shared enrich + clean core used by ``scripts/fetch_options.py``. It is a library
(no CLI): ``fetch_options.py`` assembles a raw frame (one row per timestamp × strike ×
option type, each priced on its expiry's futures close from ``futures_contracts``) and
hands it to :func:`run_stage2_enrich`, then writes the result with
:func:`_write_enriched_to_arctic`.

Pipeline inside :func:`run_stage2_enrich`
    pre-solver filter (market hours / OI / DTE / price floor / intrinsic bound)
    → fractional trading-day DTE (``options/pricing/dte_utils.py``)
    → build ``OptionChain`` objects per (timestamp, expiry) → flatten
    → single-pass Black-76 IV + Greeks (``ChainEnricher.enrich_dataframe``: traded /
      put-call-parity / forward-fill classification)
    → ``in_last_{N}_td`` flag → solved-IV cleanup → arbitrage flags → delta band.

Input (raw) columns, one row per contract × timestamp:
    underlying_symbol, timestamp, futures_price, instrument_key, trading_symbol,
    underlying_key, strike_price, instrument_type, expiry, lot_size, tick_size,
    open, high, low, close, volume, oi, dte, moneyness

Output adds: iv, delta, gamma, theta, vega, synthetic_bid, synthetic_ask,
    in_last_{N}_td, synthetic_via_putcallparity, is_synthetic, butterfly_arb,
    calendar_arb, arb_has_synthetic

Storage: ArcticDB library ``options_enriched``, symbol ``{underlying}/{interval}/{expiry}``.
"""

from __future__ import annotations

from datetime import date

import pandas as pd
from loguru import logger

from config.constants import OptionType
from data.models.option_chain import OptionChain
from data.models.option_contract import OptionContract
from data.processors.data_cleaner import OptionsDataCleaner
from data.storage.arctic_store import LIB_OPTIONS, ArcticStore, ensure_dt_index
from options.chain.chain_enricher import ChainEnricher
from options.pricing.dte_utils import (
    DEFAULT_LOOKBACK_TD,
    fractional_dte,
    in_last_n_trading_days,
    last_n_td_column,
)

# ══════════════════════════════════════════════════════════════════════════
# Stage helpers
# ══════════════════════════════════════════════════════════════════════════

def _row_to_contract(row: pd.Series) -> OptionContract | None:
    """Convert one DataFrame row (raw schema) → OptionContract.

    Returns None if required fields are missing or invalid.
    """
    try:
        expiry_val = row["expiry"]
        if isinstance(expiry_val, str):
            expiry_val = date.fromisoformat(expiry_val)
        elif hasattr(expiry_val, "date"):
            expiry_val = expiry_val.date()

        return OptionContract(
            instrument_key=str(row["instrument_key"]),
            trading_symbol=str(row.get("trading_symbol", "")),
            underlying_key=str(row.get("underlying_key", "")),
            strike=float(row["strike_price"]),
            option_type=OptionType(str(row["instrument_type"])),
            expiry=expiry_val,
            lot_size=int(row.get("lot_size", 75)),
            tick_size=float(row.get("tick_size", 0.05)),
            timestamp=pd.Timestamp(row["timestamp"]).to_pydatetime(),
            open=float(row.get("open", 0.0)),
            high=float(row.get("high", 0.0)),
            low=float(row.get("low", 0.0)),
            close=float(row.get("close", 0.0)),
            volume=int(row.get("volume", 0)),
            oi=int(row.get("oi", 0)),
            dte=float(row.get("dte", 0.0)),
            moneyness=float(row["moneyness"])
                      if pd.notna(row.get("moneyness")) else None,
        )
    except (KeyError, ValueError, TypeError) as exc:
        logger.warning(
            f"Skipping bad row ({row.get('instrument_key', '?')}): {exc}"
        )
        return None


def _recompute_dte(df: pd.DataFrame, interval: str) -> pd.DataFrame:
    """Overwrite ``dte`` with the fractional, intraday trading-day value,
    keyed on each row's own timestamp/expiry.

    Must run *before* the DataFrame is converted into ``OptionContract``
    objects, since ``dte`` is a real field on that dataclass and needs to
    survive the round-trip through chain assembly / enrichment.

    Recomputed on the unique ``(timestamp, expiry)`` pairs present in *df* and
    mapped back, rather than row-by-row, so cost scales with distinct bars
    rather than total rows (many strikes share a bar).
    """
    if df.empty:
        return df

    df = df.copy()
    ts = pd.to_datetime(df["timestamp"])
    exp = pd.to_datetime(df["expiry"]).dt.date

    pairs = pd.DataFrame({"_ts": ts, "_exp": exp}).drop_duplicates()
    pairs["_dte"] = pairs.apply(
        lambda r: fractional_dte(r["_ts"].to_pydatetime(), r["_exp"], interval), axis=1
    )

    key = pd.DataFrame({"_ts": ts, "_exp": exp})
    merged = key.merge(pairs, on=["_ts", "_exp"], how="left")
    df["dte"] = merged["_dte"].to_numpy()
    return df


def _attach_in_last_n_td(df: pd.DataFrame, n: int = DEFAULT_LOOKBACK_TD) -> pd.DataFrame:
    """Add the ``in_last_{n}_td`` flag, keyed on each row's timestamp/expiry.

    Applied *after* chain assembly (``OptionContract`` has no such field, so
    it cannot survive that round-trip) — needs only ``timestamp``/``expiry``,
    both of which do survive, so it is recomputed here rather than carried.

    Column name follows ``dte_utils.last_n_td_column`` so every reader
    (``fetch_options.py``, ``analyze_enriched_options.py``,
    ``fit_ssvi_surface.py``) resolves it the same way regardless of *n*.
    """
    col = last_n_td_column(n)
    if df.empty:
        df = df.copy()
        df[col] = pd.array([], dtype="boolean")
        return df

    df = df.copy()
    ts = pd.to_datetime(df["timestamp"])
    exp = pd.to_datetime(df["expiry"]).dt.date

    pairs = pd.DataFrame({"_ts": ts, "_exp": exp}).drop_duplicates()
    pairs["_inN"] = pairs.apply(
        lambda r: in_last_n_trading_days(r["_ts"].date(), r["_exp"], n), axis=1
    )

    key = pd.DataFrame({"_ts": ts, "_exp": exp})
    merged = key.merge(pairs, on=["_ts", "_exp"], how="left")
    df[col] = merged["_inN"].to_numpy()
    return df


def _write_enriched_to_arctic(
    arctic_store: ArcticStore,
    enriched_df: pd.DataFrame,
    underlying: str,
    interval: str,
) -> None:
    """Write enriched rows into ArcticDB, one symbol per expiry.

    Library ``options_enriched``, symbol ``{underlying}/{interval}/{expiry}``,
    ``timestamp``-indexed (non-unique — many rows per bar, one per strike ×
    option type). Uses ``update()`` (range-replace over the min/max timestamp
    in *enriched_df*) rather than ``append()``: ``append()`` only accepts rows
    strictly newer than the symbol's existing tail, so reprocessing an
    already-written window (e.g. under ``--force``) would be silently dropped —
    ``update()`` correctly overwrites that span regardless of where it falls in
    the symbol's history.

    Caller contract: *enriched_df* must hold every row of the fetched window
    (as ``fetch_options.process_expiry`` passes), since ``update()`` replaces the
    exact [min, max] timestamp range it's given — a partial frame would leave
    the rest of that range's prior rows in place.

    A failure writing one expiry's group is logged and skipped rather than
    raised, so a single bad group doesn't abort the whole run.

    Bar timestamps carry an unnamed ``pytz.FixedOffset(330)`` tzinfo, which
    ArcticDB's normalizer rejects (``bad argument type for built-in
    operation``), so they are converted to the named ``Asia/Kolkata`` zone here.
    """
    if enriched_df.empty or "expiry" not in enriched_df.columns:
        return

    for expiry, grp in enriched_df.groupby("expiry"):
        symbol = ArcticStore.options_symbol(underlying, interval, str(expiry))
        try:
            grp_indexed = ensure_dt_index(grp.reset_index(drop=True))
            if grp_indexed.index.tz is not None:
                grp_indexed.index = grp_indexed.index.tz_convert("Asia/Kolkata")
            arctic_store.update(LIB_OPTIONS, symbol, grp_indexed)
            logger.debug(f"[Arctic] {symbol}: upserted {len(grp_indexed)} rows.")
        except Exception as exc:
            logger.warning(f"[Arctic] {symbol}: write failed — {exc}")


def _raw_df_to_chains(
    raw_df: pd.DataFrame,
    underlying_symbol: str,
) -> list[OptionChain]:
    """Convert a raw DataFrame (one full date) into OptionChain objects.

    Groups rows by (timestamp, expiry). Each group → one OptionChain.
    futures_price is read from the `futures_price` column (constant per group).
    """
    if raw_df.empty:
        return []

    df = raw_df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["expiry"]    = pd.to_datetime(df["expiry"]).dt.date

    chains: list[OptionChain] = []

    for (ts, expiry), group in df.groupby(["timestamp", "expiry"]):
        futures_price = float(group["futures_price"].iloc[0])

        spot_price: float | None = None
        if "spot_price" in group.columns:
            val = group["spot_price"].iloc[0]
            if pd.notna(val):
                spot_price = float(val)

        contracts: list[OptionContract] = []
        for _, row in group.iterrows():
            contract = _row_to_contract(row)
            if contract is not None:
                contracts.append(contract)

        if not contracts:
            continue

        ts_dt = ts.to_pydatetime() if hasattr(ts, "to_pydatetime") else ts

        chains.append(OptionChain(
            underlying_symbol=underlying_symbol,
            expiry=expiry,
            timestamp=ts_dt,
            futures_price=futures_price,
            spot_price=spot_price,
            contracts=contracts,
        ))

    return chains


def _chains_to_df(chains: list[OptionChain]) -> pd.DataFrame:
    """Flatten OptionChain objects into a single DataFrame.

    Each chain contributes one block of rows via chain.to_dataframe().
    Adds futures_price and underlying_symbol as extra columns. Works whether
    the chains have been enriched yet or not — ``run_stage2_enrich`` calls
    this *before* enrichment (``iv``/``delta``/... come back as unset/NaN)
    so ``ChainEnricher.enrich_dataframe`` can classify and solve every row
    for the whole date in one pass.
    """
    frames: list[pd.DataFrame] = []
    for chain in chains:
        df = chain.to_dataframe()
        if df.empty:
            continue
        df["futures_price"]     = chain.futures_price
        df["underlying_symbol"] = chain.underlying_symbol
        frames.append(df)

    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()



# ══════════════════════════════════════════════════════════════════════════
# Stage runners
# ══════════════════════════════════════════════════════════════════════════

def run_stage2_enrich(
    raw_df: pd.DataFrame,
    underlying_symbol: str,
    enricher: ChainEnricher,
    cleaner: OptionsDataCleaner,
    interval: str,
    lookback_td: int = DEFAULT_LOOKBACK_TD,
) -> pd.DataFrame:
    """Stage 2 — Raw DataFrame → Enriched DataFrame (IV + Greeks via Black-76).

    Order: pre-solver filter (market hours / OI / DTE / price floor / intrinsic
    bound — no strike band, no synthetic forward) -> recompute fractional DTE ->
    build chains -> flatten (unenriched) -> single-pass IV solve
    (``ChainEnricher.enrich_dataframe``: traded / put-call-parity /
    forward-fill classification, solved once per row, vectorized Greeks) ->
    last-N-trading-days flag -> solved-IV range cleanup (NaN kept) ->
    arbitrage flagging (non-destructive) -> delta-band strike retention.

    Chains are flattened *before* enrichment (rather than enriched per-chain
    then flattened) because ``enrich_dataframe`` needs the whole date's rows
    together to classify each leg against its opposite leg and, for
    both-stale legs, forward-fill from that instrument's own history — a
    single ``OptionChain`` (one timestamp × expiry snapshot) can't see either.
    This also replaces the old separate ``OptionsDataCleaner.
    repair_stale_prices`` pass — see that method's docstring for why it's
    deprecated in this path.

    Args:
        raw_df:            Raw option snapshot for one date.
        underlying_symbol: Human-readable underlying name e.g. "Nifty 50".
        enricher:          Pre-constructed ChainEnricher (shared across dates).
        cleaner:            Pre-constructed OptionsDataCleaner (shared across dates).
        interval:           Candle interval (e.g. "15minute") for DTE decrement.
        lookback_td:        Trading-day window for the ``in_last_{N}_td`` flag
                             (default 40 — keeps this script's own CLI/behaviour
                             unchanged unless it opts into a different value).

    Returns:
        Enriched DataFrame with iv, delta, gamma, theta, vega populated.
        Empty DataFrame if no valid chains could be built or enriched.
    """
    filtered_df = cleaner.pre_solver_filter(raw_df)
    if filtered_df.empty:
        logger.warning(
            f"[Stage 2] {underlying_symbol} — nothing survived pre_solver_filter."
        )
        return pd.DataFrame()

    filtered_df = _recompute_dte(filtered_df, interval)

    chains = _raw_df_to_chains(filtered_df, underlying_symbol)
    if not chains:
        logger.warning(
            f"[Stage 2] No chains built for {underlying_symbol} — "
            "check raw file schema."
        )
        return pd.DataFrame()

    flat_df = _chains_to_df(chains)
    enriched_df = enricher.enrich_dataframe(flat_df)
    solved_total = int(enriched_df["iv"].notna().sum()) if not enriched_df.empty else 0

    enriched_df = _attach_in_last_n_td(enriched_df, lookback_td)
    enriched_df = cleaner.post_solver_cleanup(enriched_df)
    enriched_df = cleaner.flag_arbitrage(enriched_df)
    enriched_df = cleaner.filter_delta_band(enriched_df)

    logger.info(
        f"[Stage 2] {underlying_symbol} — "
        f"{len(chains)} chain(s), {len(enriched_df)} rows, "
        f"{solved_total} IV(s) solved."
    )
    return enriched_df
