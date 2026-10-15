"""Static constants shared by every layer.

Enums ``Exchange``, ``OptionType`` (CE/PE), ``Side`` (BUY/SELL), ``Interval``;
``InstrumentConfig`` records (lot size, strike step, Upstox instrument key) for the
registered underlyings in ``INSTRUMENTS`` (used by fetch_options.py to resolve index
keys such as NIFTY without an API search); and the NSE trading-holiday lists
(``ALL_TRADING_HOLIDAYS``) that drive the trading-day calendar in
``options/pricing/dte_utils.py``. Holidays are listed for 2025 and 2026 only.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


# ── Enums ──

class Exchange(str, Enum):
    NSE_EQ = "NSE_EQ"
    NSE_FO = "NSE_FO"
    NSE_INDEX = "NSE_INDEX"
    BSE_EQ = "BSE_EQ"


class OptionType(str, Enum):
    CE = "CE"
    PE = "PE"


class Side(str, Enum):
    BUY = "BUY"
    SELL = "SELL"


class Interval(str, Enum):
    MIN_1 = "1minute"
    MIN_3 = "3minute"
    MIN_5 = "5minute"
    MIN_15 = "15minute"
    MIN_30 = "30minute"
    DAY = "day"


# ── Instrument definitions ──

@dataclass(frozen=True)
class InstrumentConfig:
    """Configuration for an underlying instrument."""

    name: str
    symbol: str
    instrument_key: str             # Upstox key e.g. "NSE_INDEX|Nifty 50"
    lot_size: int
    tick_size: float
    strike_interval: float          # gap between strikes
    exchange: Exchange = Exchange.NSE_FO
    market_open: str = "09:15"      # IST
    market_close: str = "15:30"     # IST
    weekly_expiry_day: str = "THU"  # day of week for weekly expiry
    monthly_expiry_last_thu: bool = True
    freeze_quantity: int = 1800
    aliases: list[str] = field(default_factory=list)


NIFTY = InstrumentConfig(
    name="NIFTY",
    symbol="NIFTY",
    instrument_key="NSE_INDEX|Nifty 50",
    lot_size=75,
    tick_size=0.05,
    strike_interval=50.0,
    freeze_quantity=1800,
    aliases=["NIFTY50", "NIFTY 50"],
)

BANKNIFTY = InstrumentConfig(
    name="BANKNIFTY",
    symbol="BANKNIFTY",
    instrument_key="NSE_INDEX|Nifty Bank",
    lot_size=30,
    tick_size=0.05,
    strike_interval=100.0,
    freeze_quantity=900,
    aliases=["NIFTYBANK", "BANK NIFTY"],
)

FINNIFTY = InstrumentConfig(
    name="FINNIFTY",
    symbol="FINNIFTY",
    instrument_key="NSE_INDEX|Nifty Fin Service",
    lot_size=65,
    tick_size=0.05,
    strike_interval=50.0,
    freeze_quantity=1300,
    aliases=["NIFTY FIN SERVICE"],
)

SBICARD = InstrumentConfig(
    name="SBICARD",
    symbol="SBICARD",
    instrument_key="NSE_EQ|INE018E01016",
    lot_size=800,                            # verified from NSE contract data
    tick_size=0.05,
    strike_interval=50.0,
    exchange=Exchange.NSE_FO,
    weekly_expiry_day="THU",
    monthly_expiry_last_thu=True,
    freeze_quantity=4500,
    aliases=["SBI CARD", "SBICARDS"],
)

# Registry for quick lookup
INSTRUMENTS: dict[str, InstrumentConfig] = {
    "NIFTY": NIFTY,
    "BANKNIFTY": BANKNIFTY,
    "FINNIFTY": FINNIFTY,
    "SBICARD": SBICARD,
}


def get_instrument(name: str) -> InstrumentConfig:
    """Look up instrument by name or alias."""
    name_upper = name.upper().strip()
    if name_upper in INSTRUMENTS:
        return INSTRUMENTS[name_upper]
    for inst in INSTRUMENTS.values():
        if name_upper in [a.upper() for a in inst.aliases]:
            return inst
    raise ValueError(f"Unknown instrument: {name}. Available: {list(INSTRUMENTS.keys())}")


# ── Market calendar constants ──

TRADING_HOLIDAYS_2025 = [
    "2025-02-26",  # Mahashivratri
    "2025-03-14",  # Holi
    "2025-03-31",  # Id-Ul-Fitr
    "2025-04-10",  # Shri Mahavir Jayanti
    "2025-04-14",  # Dr. Ambedkar Jayanti
    "2025-04-18",  # Good Friday
    "2025-05-01",  # Maharashtra Day
    "2025-08-15",  # Independence Day
    "2025-08-27",  # Ganesh Chaturthi
    "2025-10-02",  # Mahatma Gandhi Jayanti / Dussehra
    "2025-10-21",  # Diwali Laxmi Pujan
    "2025-10-22",  # Diwali Balipratipada
    "2025-11-05",  # Prakash Gurpurb Sri Guru Nanak Dev
    "2025-12-25",  # Christmas
]

# NSE equity-segment trading holidays for 2026 (weekday closures only).
# Source: NSE 2026 holiday circular, cross-checked against Zerodha + ClearTax.
# Nov 8, 2026 (Diwali Laxmi Pujan) is a Sunday with a special Muhurat session — not
# a full closure, so it is intentionally omitted.
TRADING_HOLIDAYS_2026 = [
    "2026-01-15",  # Municipal Corporation Elections (Maharashtra)
    "2026-01-26",  # Republic Day
    "2026-03-03",  # Holi
    "2026-03-26",  # Shri Ram Navami
    "2026-03-31",  # Shri Mahavir Jayanti
    "2026-04-03",  # Good Friday
    "2026-04-14",  # Dr. Baba Saheb Ambedkar Jayanti
    "2026-05-01",  # Maharashtra Day
    "2026-05-28",  # Bakri Id
    "2026-06-26",  # Muharram
    "2026-09-14",  # Ganesh Chaturthi
    "2026-10-02",  # Mahatma Gandhi Jayanti
    "2026-10-20",  # Dussehra
    "2026-11-10",  # Diwali - Balipratipada
    "2026-11-24",  # Prakash Gurpurb Sri Guru Nanak Dev
    "2026-12-25",  # Christmas
]

# Union of every year's holidays — consumed by the trading-day calendar in
# options/pricing/dte_utils.py. Extend by adding new TRADING_HOLIDAYS_<year> lists here.
ALL_TRADING_HOLIDAYS = TRADING_HOLIDAYS_2025 + TRADING_HOLIDAYS_2026
