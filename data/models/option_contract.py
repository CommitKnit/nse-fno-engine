"""Single option contract with market data, IV, and Greeks."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime

from config.constants import OptionType


@dataclass(slots=True)
class OptionContract:
    """Represents one option contract at a point in time.

    Contains both static metadata (strike, type, expiry) and
    time-varying data (OHLCV, OI, computed IV and Greeks).
    """

    # ── Static metadata (from contract endpoint) ──
    instrument_key: str              # e.g. "NSE_FO|37590"
    trading_symbol: str              # e.g. "NIFTY 23500 CE 30 JAN 25"
    underlying_key: str              # e.g. "NSE_INDEX|Nifty 50"
    strike: float                    # strike price
    option_type: OptionType          # CE or PE
    expiry: date                     # expiry date
    lot_size: int = 75
    tick_size: float = 0.05

    # ── Time-varying market data (from candle endpoint) ──
    timestamp: datetime | None = None
    open: float = 0.0
    high: float = 0.0
    low: float = 0.0
    close: float = 0.0
    volume: int = 0
    oi: int = 0

    # ── Computed fields (from IV solver + Greeks engine) ──
    iv: float | None = None          # implied volatility
    delta: float | None = None
    gamma: float | None = None
    theta: float | None = None
    vega: float | None = None
    rho: float | None = None

    # ── Derived ──
    moneyness: float | None = None   # strike / futures_price
    dte: float | None = None         # fractional, intraday trading-day DTE
    synthetic_bid: float | None = None
    synthetic_ask: float | None = None

    @property
    def mid_price(self) -> float:
        """Mid of synthetic bid-ask, or close if not available."""
        if self.synthetic_bid is not None and self.synthetic_ask is not None:
            return (self.synthetic_bid + self.synthetic_ask) / 2.0
        return self.close

    @property
    def is_call(self) -> bool:
        return self.option_type == OptionType.CE

    @property
    def is_put(self) -> bool:
        return self.option_type == OptionType.PE

    @property
    def intrinsic_value(self) -> float:
        """Intrinsic value — requires moneyness to be set."""
        if self.moneyness is None:
            return 0.0
        futures_price = self.strike / self.moneyness if self.moneyness != 0 else 0
        if self.is_call:
            return max(0.0, futures_price - self.strike)
        return max(0.0, self.strike - futures_price)

    @classmethod
    def from_upstox_contract(cls, data: dict) -> OptionContract:
        """Parse from Upstox expired-instruments/option/contract response."""
        return cls(
            instrument_key=data["instrument_key"],
            trading_symbol=data.get("trading_symbol", ""),
            underlying_key=data.get("underlying_key", ""),
            strike=float(data["strike_price"]),
            option_type=OptionType(data["instrument_type"]),
            expiry=date.fromisoformat(data["expiry"]),
            lot_size=int(data.get("lot_size", 75)),
            tick_size=float(data.get("tick_size", 0.05)),
        )

    def to_dict(self) -> dict:
        return {
            # ── keys match from_upstox_contract expectations ──
            "instrument_key": self.instrument_key,
            "trading_symbol": self.trading_symbol,
            "underlying_key": self.underlying_key,
            "strike_price": self.strike,
            "instrument_type": self.option_type.value,
            "expiry": self.expiry.isoformat(),
            "lot_size": self.lot_size,
            "tick_size": self.tick_size,
            # ── time-varying market data ──
            "timestamp": self.timestamp,
            "open": self.open,
            "high": self.high,
            "low": self.low,
            "close": self.close,
            "volume": self.volume,
            "oi": self.oi,
            # ── computed ──
            "iv": self.iv,
            "delta": self.delta,
            "gamma": self.gamma,
            "theta": self.theta,
            "vega": self.vega,
            "moneyness": self.moneyness,
            "dte": self.dte,
            "synthetic_bid": self.synthetic_bid,
            "synthetic_ask": self.synthetic_ask,
        }

