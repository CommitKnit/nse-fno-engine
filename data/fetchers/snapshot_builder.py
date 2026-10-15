"""snapshot_builder — build MarketSnapshot objects from ArcticDB for the backtest engine.

Reads, for one symbol:
    futures           {symbol}/{futures_interval}            F1/F2/F3 wide panel (+ Kite spot joined
                                                              as SPOT_* by ArcticStore.read_futures)
    options_enriched  {symbol}/{options_interval}/{expiry}   enriched chains for every expiry that
                                                              appears in the panel's F1/F2/F3 slots
and emits one ``MarketSnapshot`` per futures bar that has at least one option chain.

Usage:
    builder = LocalSnapshotBuilder(symbol="IREDA", futures_interval="5minute",
                                   options_interval="5minute")
    snapshots = builder.build(from_date="2025-04-01", to_date="2026-06-30")
    engine = BacktestEngine(snapshots, strategy, ...)
    results = engine.run()

Cross-interval join:
    When futures_interval != options_interval (e.g. 5min futures, 15min options),
    each futures bar gets the most recent prior options snapshot via forward-fill
    (binary search on the sorted options timestamp index).

Pricing: every chain is priced on its OWN expiry's futures close from the panel slot
(F2 options on F2), falling back to the chain rows' stored ``futures_price`` and then F1.
"""

from __future__ import annotations

from datetime import date

import pandas as pd

from config.constants import OptionType
from config.settings import settings
from data.models.bar import Bar
from data.models.market_snapshot import MarketSnapshot
from data.models.option_chain import OptionChain
from data.models.option_contract import OptionContract
from data.storage.arctic_store import ArcticStore

_OPT_TYPE: dict[str, OptionType] = {
    "CE": OptionType.CE,
    "PE": OptionType.PE,
}


def _opt_field(row: pd.Series, *names: str):
    """First present, non-null value among *names* — bridges the two options_enriched
    schemas (older ``strike``/``opt_type`` vs fetch_options.py
    ``strike_price``/``instrument_type``)."""
    for n in names:
        v = row.get(n)
        if v is not None and not (isinstance(v, float) and pd.isna(v)):
            return v
    return None


def _opt_float(row: pd.Series, name: str) -> float | None:
    v = row.get(name)
    return float(v) if v is not None and pd.notna(v) else None


class LocalSnapshotBuilder:
    """Reads the engine's ArcticDB store and builds list[MarketSnapshot].

    Args:
        symbol:            e.g. "IREDA".
        futures_interval:  e.g. "5minute", "15minute", "1day".
        options_interval:  Interval of the ``options_enriched`` chains to load.
        contract_ffill_days: None (default) — each bar's chain is the rows of the
                           latest option timestamp only, so a strike that did not
                           print on that bar is absent. An int N makes each chain
                           carry EVERY contract's latest row from the last N calendar
                           days (per-contract forward-fill), so thin chains still
                           expose each strike's last OI/IV.
        store:             Optional existing ArcticStore (one is opened otherwise).
    """

    def __init__(
        self,
        symbol: str,
        futures_interval: str = "15minute",
        options_interval: str = "15minute",
        contract_ffill_days: int | None = None,
        store: ArcticStore | None = None,
    ):
        self.symbol = symbol
        self.futures_interval = futures_interval
        self.options_interval = options_interval
        self.contract_ffill_days = contract_ffill_days
        self._store = store or ArcticStore()
        self._lot_size = self._read_lot_size()

    # ── Date bounds helper ───────────────────────────────────────────────────

    def _bounds(
        self, from_date: str | None, to_date: str | None
    ) -> tuple[pd.Timestamp | None, pd.Timestamp | None]:
        """Convert ISO date strings to a tz-aware (start, end) range for ArcticDB.

        ``end`` is pushed to the start of the day *after* ``to_date`` so the whole
        ``to_date`` trading day is covered; an exact ``.loc[from:to]`` slice is
        applied afterwards.
        """
        tz = settings.DEFAULT_TIMEZONE
        start = pd.Timestamp(from_date, tz=tz) if from_date else None
        end = (pd.Timestamp(to_date, tz=tz) + pd.Timedelta(days=1)) if to_date else None
        return (start, end)

    # ── Setup ──────────────────────────────────────────────────────────────────

    def _read_lot_size(self) -> int | None:
        """Fallback lot size from the ``contract_master`` table, if stored.

        Option rows written by fetch_options.py carry their own ``lot_size``, which
        always takes precedence (see ``_make_option_contract``).
        """
        try:
            cm = self._store.read_table("contract_master")
            if not cm.empty and {"symbol", "lot_size"} <= set(cm.columns):
                hit = cm.loc[cm["symbol"] == self.symbol, "lot_size"].dropna()
                if not hit.empty and hit.iloc[0]:
                    return int(hit.iloc[0])
        except Exception:
            pass
        return None

    # ── Data loading ───────────────────────────────────────────────────────────

    def _load_wide(self, from_date: str, to_date: str) -> pd.DataFrame:
        wide = self._store.read_futures(
            self.symbol, self.futures_interval, date_range=self._bounds(from_date, to_date)
        )
        if wide.empty:
            raise FileNotFoundError(
                f"No futures data in ArcticDB for {self.symbol}/{self.futures_interval}. "
                f"Run: fetch-ticker {self.symbol} --history --intervals {self.futures_interval}"
            )
        wide = wide.sort_index()
        wide = wide.loc[from_date:to_date]
        wide = wide[wide["F1_close"].notna()]
        return wide

    def _load_options_for_expiry(
        self, expiry: str, from_date: str, to_date: str
    ) -> pd.DataFrame:
        df = self._store.read_options(
            self.symbol, self.options_interval, expiry,
            date_range=self._bounds(from_date, to_date),
        )
        if df.empty:
            return df
        df = df.sort_index()
        df = df.loc[from_date:to_date]
        if "in_window" in df.columns:   # older chain schema's window flag
            df = df[df["in_window"].astype(bool)]
        return df

    # ── Main builder ───────────────────────────────────────────────────────────

    def build(self, from_date: str, to_date: str) -> list[MarketSnapshot]:
        """Return the sorted ``list[MarketSnapshot]`` for ``[from_date, to_date]``.

        Args:
            from_date: ISO date string, inclusive, e.g. "2025-03-01".
            to_date:   ISO date string, inclusive, e.g. "2025-12-31".
        """
        wide = self._load_wide(from_date, to_date)
        print(
            f"LocalSnapshotBuilder: {len(wide)} wide-panel rows "
            f"({self.symbol} {self.futures_interval})"
        )

        # Active expiries — collect from all three slots
        active_expiries: set[str] = set()
        for slot in ("F1", "F2", "F3"):
            col = f"{slot}_expiry"
            if col in wide.columns:
                active_expiries.update(
                    str(v) for v in wide[col].dropna().unique()
                )
        active_expiries_sorted = sorted(active_expiries)

        # Load and pre-index options data per expiry
        # opts_index: expiry -> sorted pd.DatetimeIndex of option timestamps
        # opts_data:  expiry -> dict[opt_ts -> DataFrame of contracts]
        opts_index: dict[str, pd.DatetimeIndex] = {}
        opts_data: dict[str, dict] = {}

        for expiry in active_expiries_sorted:
            n_days = self.contract_ffill_days
            load_from = (from_date if not n_days else
                         (pd.Timestamp(from_date) - pd.Timedelta(days=n_days)).date().isoformat())
            df = self._load_options_for_expiry(expiry, load_from, to_date)
            if df.empty:
                continue
            unique_ts = df.index.unique().sort_values()
            opts_index[expiry] = unique_ts
            if not n_days:
                opts_data[expiry] = {ts: df.loc[ts] for ts in unique_ts}
            else:
                key = "instrument_key" if "instrument_key" in df.columns else "instrument_token"
                win = pd.Timedelta(days=n_days)
                opts_data[expiry] = {
                    ts: df[(df.index <= ts) & (df.index > ts - win)].groupby(key, sort=False).tail(1)
                    for ts in unique_ts
                }
            print(f"  Options loaded: {expiry}  {len(unique_ts)} bars  "
                  f"{len(df)} contracts total")

        if not opts_data:
            print(
                f"WARNING: No options data in options_enriched/{self.symbol}/"
                f"{self.options_interval}/<expiry> for this range.\n"
                f"Snapshots will have empty chains."
            )

        # Build snapshots bar by bar
        futures_timestamps = wide.index.unique().sort_values()
        snapshots: list[MarketSnapshot] = []

        for ts in futures_timestamps:
            wide_row = wide.loc[ts]
            if isinstance(wide_row, pd.DataFrame):
                wide_row = wide_row.iloc[0]

            futures_bar = self._make_futures_bar(ts, wide_row)
            spot_bar = self._make_spot_bar(ts, wide_row)
            chains = self._make_chains(ts, wide_row, opts_index, opts_data)

            # Only add snapshots that have at least one option chain
            if not chains:
                continue

            snapshots.append(
                MarketSnapshot(
                    timestamp=ts.to_pydatetime(),
                    futures_bar=futures_bar,
                    spot_bar=spot_bar,
                    chains=chains,
                )
            )

        print(f"Built {len(snapshots)} MarketSnapshot objects")
        return snapshots

    # ── Bar construction ───────────────────────────────────────────────────────

    def _make_futures_bar(self, ts: pd.Timestamp, row: pd.Series) -> Bar:
        return Bar(
            timestamp=ts.to_pydatetime(),
            open=float(row["F1_open"]),
            high=float(row["F1_high"]),
            low=float(row["F1_low"]),
            close=float(row["F1_close"]),
            volume=int(row.get("F1_volume", 0) or 0),
            oi=int(row.get("F1_oi", 0) or 0),
            instrument_key=str(row.get("F1_ticker", self.symbol)),
        )

    def _make_spot_bar(self, ts: pd.Timestamp, row: pd.Series) -> Bar | None:
        """Kite spot bar (joined by ArcticStore.read_futures), or None when missing."""
        if row.get("spot_missing", True) or pd.isna(row.get("SPOT_close")):
            return None
        return Bar(
            timestamp=ts.to_pydatetime(),
            open=float(row["SPOT_open"]),
            high=float(row["SPOT_high"]),
            low=float(row["SPOT_low"]),
            close=float(row["SPOT_close"]),
            volume=int(row.get("SPOT_volume", 0) or 0),
            oi=None,
            instrument_key=str(row.get("SPOT_ticker", self.symbol + "_SPOT")),
        )

    # ── Chain construction ─────────────────────────────────────────────────────

    def _make_chains(
        self,
        ts: pd.Timestamp,
        wide_row: pd.Series,
        opts_index: dict[str, pd.DatetimeIndex],
        opts_data: dict[str, dict],
    ) -> dict[str, OptionChain]:
        f1_price = float(wide_row["F1_close"])
        # Price each chain off ITS OWN expiry's future (F2 options on F2, not F1);
        # fall back to F1 when the slot's close is missing on this bar.
        slot_price: dict[str, float] = {}
        for slot in ("F1", "F2", "F3"):
            exp, px = wide_row.get(f"{slot}_expiry"), wide_row.get(f"{slot}_close")
            if exp is not None and pd.notna(exp) and px is not None and pd.notna(px):
                slot_price[str(exp)[:10]] = float(px)
        chains: dict[str, OptionChain] = {}

        for expiry, idx in opts_index.items():
            # Forward-fill: find most recent option bar <= futures ts
            pos = idx.searchsorted(ts, side="right") - 1
            if pos < 0:
                continue
            opt_ts = idx[pos]
            opt_rows = opts_data[expiry][opt_ts]

            # opts_data[expiry][opt_ts] is either a Series (1 row) or DataFrame
            if isinstance(opt_rows, pd.Series):
                opt_rows = opt_rows.to_frame().T

            futures_price = slot_price.get(expiry)
            if futures_price is None and "futures_price" in opt_rows.columns:
                # fetch_options.py rows carry the expiry's own pricing future; take
                # the most recent row's (ffilled chains mix row timestamps). Covers
                # e.g. an expiry day the wide panel has already rolled past.
                fp = pd.to_numeric(opt_rows["futures_price"], errors="coerce")
                fp = fp[fp.notna()]
                if len(fp) and isinstance(opt_rows.index, pd.DatetimeIndex):
                    fp = fp.sort_index(kind="stable")
                futures_price = float(fp.iloc[-1]) if len(fp) else None
            if futures_price is None:
                futures_price = f1_price
            contracts = [
                self._make_option_contract(opt_row, ts, expiry, futures_price)
                for _, opt_row in opt_rows.iterrows()
            ]

            if not contracts:
                continue

            expiry_date = date.fromisoformat(expiry)
            chains[expiry] = OptionChain(
                underlying_symbol=self.symbol,
                expiry=expiry_date,
                timestamp=ts.to_pydatetime(),
                futures_price=futures_price,
                spot_price=(
                    float(wide_row["SPOT_close"])
                    if not wide_row.get("spot_missing", True)
                    and pd.notna(wide_row.get("SPOT_close"))
                    else None
                ),
                contracts=contracts,
            )

        return chains

    def _make_option_contract(
        self,
        row: pd.Series,
        ts: pd.Timestamp,
        expiry: str,
        futures_price: float,
    ) -> OptionContract:
        # Handles both options_enriched schemas (see _opt_field).
        strike = float(_opt_field(row, "strike", "strike_price"))
        opt_code = str(_opt_field(row, "opt_type", "instrument_type"))
        opt_type = _OPT_TYPE[opt_code]
        inst_key = str(
            row.get("instrument_key", row.get("instrument_token", ""))
        )
        lot_size = _opt_field(row, "lot_size") or self._lot_size or 1

        return OptionContract(
            instrument_key=inst_key,
            trading_symbol=str(_opt_field(row, "trading_symbol")
                               or f"{self.symbol}{strike:.0f}{opt_code} {expiry}"),
            underlying_key=str(_opt_field(row, "underlying_key") or f"NSE_EQ|{self.symbol}"),
            strike=strike,
            option_type=opt_type,
            expiry=date.fromisoformat(expiry),
            lot_size=int(lot_size),
            tick_size=0.05,
            timestamp=ts.to_pydatetime(),
            open=float(row["open"]),
            high=float(row["high"]),
            low=float(row["low"]),
            close=float(row["close"]),
            volume=int(row.get("volume", 0) or 0),
            oi=int(row.get("oi", 0) or 0),
            iv=float(row["iv"]) if pd.notna(row.get("iv")) else None,
            delta=float(row["delta"]) if pd.notna(row.get("delta")) else None,
            gamma=float(row["gamma"]) if pd.notna(row.get("gamma")) else None,
            theta=float(row["theta"]) if pd.notna(row.get("theta")) else None,
            vega=float(row["vega"]) if pd.notna(row.get("vega")) else None,
            # float, not int: the DTE is fractional intraday (int() truncated it).
            dte=_opt_float(row, "dte"),
            moneyness=strike / futures_price if futures_price > 0 else None,
            synthetic_bid=_opt_float(row, "synthetic_bid"),
            synthetic_ask=_opt_float(row, "synthetic_ask"),
        )
