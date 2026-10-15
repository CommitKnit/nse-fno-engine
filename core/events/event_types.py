"""Event types that flow through the backtesting engine.

Every action in the engine is an Event. The event loop processes them
in priority order: Market → Signal → Order → Fill.

This decouples modules: the strategy doesn't call the executor directly,
it emits a SignalEvent, and the risk manager picks it up.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import IntEnum
from typing import Any

from config.constants import OptionType, Side

class EventPriority(IntEnum):
    """Processing order — lower number = higher priority."""
    MARKET = 0      # process market data first
    SIGNAL = 1      # then strategy signals
    ORDER = 2       # then validated orders
    FILL = 3        # then execution fills
    EXPIRY = 4      # then expiry events
    SYSTEM = 10     # system events (logging, snapshots)


@dataclass
class Event:
    """Base event class."""
    timestamp: datetime
    priority: EventPriority = EventPriority.SYSTEM
    event_id: str = ""

    def __post_init__(self):
        if not self.event_id:
            import uuid
            self.event_id = str(uuid.uuid4())[:8]

    def __lt__(self, other: Event) -> bool:
        """For priority queue ordering."""
        if self.priority != other.priority:
            return self.priority < other.priority
        return self.timestamp < other.timestamp


@dataclass
class MarketEvent(Event):
    """New market data arrived — triggers strategy evaluation.

    Carries a MarketSnapshot with futures bar + option chains.
    """
    priority: EventPriority = field(default=EventPriority.MARKET, init=False)
    snapshot: Any = None  # MarketSnapshot — Any to avoid circular import


@dataclass
class SignalEvent(Event):
    """Strategy wants to trade — needs risk validation before execution.

    A signal is an *intent* to trade, not a confirmed order.
    The risk manager converts approved signals into OrderEvents.
    """
    priority: EventPriority = field(default=EventPriority.SIGNAL, init=False)
    symbol: str = ""
    instrument_key: str = ""
    side: Side = Side.BUY
    quantity: int = 0               # in lots
    option_type: OptionType | None = None
    strike: float | None = None
    expiry: str = ""                # ISO date string
    signal_type: str = ""           # "entry", "exit", "adjustment"
    strategy_name: str = ""
    target_price: float | None = None  # limit price hint
    metadata: dict = field(default_factory=dict)

    @property
    def is_option(self) -> bool:
        return self.option_type is not None

    @property
    def is_buy(self) -> bool:
        return self.side == Side.BUY


@dataclass
class OrderEvent(Event):
    """Risk-approved order ready for execution.

    Created by the risk manager after validating a SignalEvent.
    """
    priority: EventPriority = field(default=EventPriority.ORDER, init=False)
    order_id: str = ""
    instrument_key: str = ""
    side: Side = Side.BUY
    quantity: int = 0               # in units (lots × lot_size)
    lots: int = 0
    order_type: str = "MARKET"      # "MARKET" or "LIMIT"
    limit_price: float | None = None
    strategy_name: str = ""
    signal_id: str = ""             # links back to the SignalEvent
    metadata: dict = field(default_factory=dict)


@dataclass
class FillEvent(Event):
    """Order has been filled (simulated execution).

    Created by the execution simulator after processing an OrderEvent.
    """
    priority: EventPriority = field(default=EventPriority.FILL, init=False)
    order_id: str = ""
    instrument_key: str = ""
    side: Side = Side.BUY
    quantity: int = 0
    lots: int = 0
    fill_price: float = 0.0
    commission: float = 0.0
    slippage: float = 0.0
    strategy_name: str = ""
    metadata: dict = field(default_factory=dict)

    @property
    def total_cost(self) -> float:
        """Total cash impact of this fill."""
        direction = 1.0 if self.side == Side.BUY else -1.0
        return direction * self.fill_price * self.quantity + self.commission


@dataclass
class ExpiryEvent(Event):
    """An option contract has expired.

    Created by the expiry handler at the end of expiry day.
    """
    priority: EventPriority = field(default=EventPriority.EXPIRY, init=False)
    instrument_key: str = ""
    settlement_price: float = 0.0
    is_itm: bool = False
    pnl: float = 0.0
