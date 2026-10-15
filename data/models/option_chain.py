
"""Option chain snapshot — all strikes for one expiry at one point in time."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime

import pandas as pd

from config.constants import OptionType
from data.models.option_contract import OptionContract

# option_type value → canonical single-char code used in MultiIndex
_OT_CANONICAL = {OptionType.CE: "C", OptionType.PE: "P"}


@dataclass
class OptionChain:
    """Full option chain for one underlying + one expiry at one timestamp.

    This is the core data structure fed to strategies each bar.
    """

    underlying_symbol: str           # e.g. "NIFTY"
    expiry: date
    timestamp: datetime
    futures_price: float             # near-month future close at this timestamp
    spot_price: float | None = None  # spot index (optional, for reference)
    contracts: list[OptionContract] = field(default_factory=list)

    @property
    def calls(self) -> list[OptionContract]:
        return [c for c in self.contracts if c.is_call]

    @property
    def puts(self) -> list[OptionContract]:
        return [c for c in self.contracts if c.is_put]

    @property
    def strikes(self) -> list[float]:
        """Sorted unique strikes."""
        return sorted(set(c.strike for c in self.contracts))

    @property
    def atm_strike(self) -> float:
        """Strike closest to futures price."""
        if not self.strikes:
            return 0.0
        return min(self.strikes, key=lambda s: abs(s - self.futures_price))

    @property
    def dte(self) -> float:
        """Days to expiry.

        Returns the carried fractional, intraday-decrementing trading-day DTE
        from the chain's contracts when available (all contracts in a chain
        share the same timestamp/expiry/interval, so the first populated value
        is authoritative). Falls back to a plain calendar-day count when no
        contract carries a DTE (e.g. an unpopulated chain).
        """
        for c in self.contracts:
            if c.dte is not None:
                return float(c.dte)
        return float((self.expiry - self.timestamp.date()).days)

    def get_contract(self, strike: float, option_type: OptionType) -> OptionContract | None:
        """Look up a specific contract by strike and type."""
        for c in self.contracts:
            if c.strike == strike and c.option_type == option_type:
                return c
        return None

    def get_by_delta(
        self, target_delta: float, option_type: OptionType, tolerance: float = 0.05
    ) -> OptionContract | None:
        """Find the contract closest to a target delta.

        Args:
            target_delta: Desired delta (e.g. 0.30 for 30-delta).
                          For puts, pass negative (e.g. -0.30).
            option_type: CE or PE
            tolerance: Max acceptable deviation from target.

        Returns:
            The contract with delta closest to target, or None.
        """
        candidates = self.calls if option_type == OptionType.CE else self.puts
        candidates = [c for c in candidates if c.delta is not None]
        if not candidates:
            return None
        best = min(candidates, key=lambda c: abs((c.delta or 0) - target_delta))
        if abs((best.delta or 0) - target_delta) <= tolerance:
            return best
        return None

    def get_strikes_around_atm(self, n: int = 5) -> list[float]:
        """Get `n` strikes above and below ATM (2n+1 total)."""
        atm = self.atm_strike
        all_strikes = self.strikes
        if not all_strikes:
            return []
        atm_idx = all_strikes.index(atm)
        lo = max(0, atm_idx - n)
        hi = min(len(all_strikes), atm_idx + n + 1)
        return all_strikes[lo:hi]

    def to_dataframe(self) -> pd.DataFrame:
        """Convert chain to a DataFrame for analysis."""
        if not self.contracts:
            return pd.DataFrame()
        rows = [c.to_dict() for c in self.contracts]
        df = pd.DataFrame(rows)
        df["futures_price"] = self.futures_price
        # Preserve each contract's carried (fractional) DTE; only fill gaps with
        # the chain-level fallback so a per-bar intraday DTE is not flattened.
        if "dte" not in df.columns:
            df["dte"] = self.dte
        else:
            df["dte"] = pd.to_numeric(df["dte"], errors="coerce").fillna(self.dte)
        return df.sort_values(["strike_price", "instrument_type"]).reset_index(drop=True)

    def to_multiindex_df(self) -> pd.DataFrame:
        """Convert chain to the canonical MultiIndex DataFrame.

        MultiIndex levels (in order):
            symbol, expiration, strike, option_type ('C' or 'P')

        All float columns use nullable ``Float64``; ``dte`` uses ``Float64``
        (fractional, intraday trading-day DTE).
        ``bid`` and ``ask`` are populated from ``synthetic_bid`` /
        ``synthetic_ask`` when set, otherwise ``pd.NA``.
        ``mid`` = (bid + ask) / 2 when both are valid, else ``close``.
        The result is sorted on the full index before returning.
        """
        from data.storage.schemas import CHAIN_COLUMNS, CHAIN_INDEX_NAMES, make_empty_chain_df

        if not self.contracts:
            return make_empty_chain_df()

        records: list[dict] = []
        chain_dte = self.dte  # fallback when contract.dte is None

        for c in self.contracts:
            bid: float | pd.NAType = (
                float(c.synthetic_bid) if c.synthetic_bid is not None else pd.NA
            )
            ask: float | pd.NAType = (
                float(c.synthetic_ask) if c.synthetic_ask is not None else pd.NA
            )
            if bid is not pd.NA and ask is not pd.NA:
                mid: float | pd.NAType = (float(bid) + float(ask)) / 2.0
            else:
                mid = float(c.close)

            records.append(
                {
                    "symbol":        self.underlying_symbol,
                    "expiration":    pd.Timestamp(self.expiry),
                    "strike":        float(c.strike),
                    "option_type":   _OT_CANONICAL[c.option_type],
                    "open":          float(c.open),
                    "high":          float(c.high),
                    "low":           float(c.low),
                    "close":         float(c.close),
                    "bid":           bid,
                    "ask":           ask,
                    "mid":           mid,
                    "volume":        float(c.volume),
                    "open_interest": float(c.oi),
                    "implied_vol":   float(c.iv) if c.iv is not None else pd.NA,
                    "delta":         float(c.delta) if c.delta is not None else pd.NA,
                    "gamma":         float(c.gamma) if c.gamma is not None else pd.NA,
                    "vega":          float(c.vega) if c.vega is not None else pd.NA,
                    "theta":         float(c.theta) if c.theta is not None else pd.NA,
                    "dte":           c.dte if c.dte is not None else chain_dte,
                }
            )

        df = pd.DataFrame(records)
        df["expiration"] = pd.to_datetime(df["expiration"])
        df = df.set_index(CHAIN_INDEX_NAMES)

        # Cast to canonical nullable types
        for col in CHAIN_COLUMNS:
            if col == "dte":
                df[col] = pd.to_numeric(df[col], errors="coerce").astype("Float64")
            else:
                df[col] = pd.to_numeric(df[col], errors="coerce").astype("Float64")

        return df.sort_index()

    def __len__(self) -> int:
        return len(self.contracts)

    def __repr__(self) -> str:
        return (
            f"OptionChain({self.underlying_symbol} {self.expiry} "
            f"F={self.futures_price:.0f} "
            f"strikes={len(self.strikes)} "
            f"contracts={len(self.contracts)} "
            f"DTE={self.dte})"
        )
