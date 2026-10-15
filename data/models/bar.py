"""Bar — one OHLCV(+OI) candle, the unit every fetcher returns.

``Bar.from_upstox_candle`` / ``Bar.from_kite_candle`` parse a provider candle; ``to_dict`` flattens a bar
for DataFrame construction. Used for futures, options and spot candles alike.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import pandas as pd


@dataclass(slots=True)
class Bar:
    """Single OHLCV candle bar.

    Works for spot, futures, and option contracts.
    The `oi` field is populated for F&O instruments, None for spot/equity.
    """

    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int
    oi: int | None = None           # open interest (F&O only)
    instrument_key: str = ""        # source instrument id (Upstox key, or "KITE_MCX|<token>")

    @property
    def typical_price(self) -> float:
        return (self.high + self.low + self.close) / 3.0

    @property
    def hl_range(self) -> float:
        return self.high - self.low

    @property
    def mid(self) -> float:
        return (self.high + self.low) / 2.0

    @classmethod
    def from_upstox_candle(cls, candle: list, instrument_key: str = "") -> Bar:
        """Parse a candle row from Upstox API response.

        Upstox returns: [timestamp, open, high, low, close, volume, oi]
        The OI field is present for F&O, absent for equity/index.
        """
        ts = pd.Timestamp(candle[0]).to_pydatetime()
        return cls(
            timestamp=ts,
            open=float(candle[1]),
            high=float(candle[2]),
            low=float(candle[3]),
            close=float(candle[4]),
            volume=int(candle[5]),
            oi=int(candle[6]) if len(candle) > 6 else None,
            instrument_key=instrument_key,
        )

    @classmethod
    def from_kite_candle(cls, candle: dict, instrument_key: str = "") -> Bar:
        """Parse a candle from the Kite Connect python client.

        Unlike Upstox (positional list), ``KiteConnect.historical_data()``
        returns dicts already keyed by name::

            {"date": datetime(tz=+05:30), "open": .., "high": .., "low": ..,
             "close": .., "volume": .., "oi": ..}

        ``oi`` is present only when the request passed ``oi=1`` -- the SDK omits
        the key entirely otherwise (it only sets it when the raw row has 7
        fields).

        ``date`` is already tz-aware IST, so the resulting ``timestamp`` is
        tz-aware, matching what ``from_upstox_candle`` produces. Downstream
        ``pd.to_datetime(..., utc=True).dt.tz_convert("Asia/Kolkata")``
        normalisation therefore works identically for both sources.
        """
        ts = pd.Timestamp(candle["date"]).to_pydatetime()
        oi_raw = candle.get("oi")
        return cls(
            timestamp=ts,
            open=float(candle["open"]),
            high=float(candle["high"]),
            low=float(candle["low"]),
            close=float(candle["close"]),
            volume=int(candle["volume"]),
            oi=int(oi_raw) if oi_raw is not None else None,
            instrument_key=instrument_key,
        )

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "open": self.open,
            "high": self.high,
            "low": self.low,
            "close": self.close,
            "volume": self.volume,
            "oi": self.oi,
            "instrument_key": self.instrument_key,
        }
