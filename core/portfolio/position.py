"""Position — tracks a single open position with entry, quantity, and P&L.

A Position represents one instrument holding. For options, each leg
of a spread is a separate Position. The portfolio aggregates them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from config.constants import Side


@dataclass
class Position:
    """A single open position in one instrument.

    Tracks entry price, current market price, quantity, and P&L.
    Supports averaging up/down (multiple fills on the same instrument).
    """

    instrument_key: str
    side: Side                       # BUY (long) or SELL (short)
    quantity: int = 0                # total units held (lots × lot_size)
    lots: int = 0                    # total lots held
    avg_entry_price: float = 0.0     # volume-weighted average entry
    current_price: float = 0.0       # latest market price
    realized_pnl: float = 0.0       # P&L from closed portions
    commission_paid: float = 0.0     # total commissions on this position
    strategy_name: str = ""
    opened_at: datetime | None = None
    last_updated: datetime | None = None
    metadata: dict = field(default_factory=dict)

    @property
    def is_long(self) -> bool:
        return self.side == Side.BUY

    @property
    def is_short(self) -> bool:
        return self.side == Side.SELL

    @property
    def is_open(self) -> bool:
        return self.quantity > 0

    @property
    def direction(self) -> int:
        """+1 for long, -1 for short."""
        return 1 if self.is_long else -1

    @property
    def market_value(self) -> float:
        """Current market value of the position."""
        return self.current_price * self.quantity * self.direction

    @property
    def cost_basis(self) -> float:
        """Total cost to enter the position."""
        return self.avg_entry_price * self.quantity

    @property
    def unrealized_pnl(self) -> float:
        """Unrealized P&L based on current price."""
        if self.quantity == 0:
            return 0.0
        price_diff = self.current_price - self.avg_entry_price
        return price_diff * self.quantity * self.direction

    @property
    def total_pnl(self) -> float:
        """Realized + unrealized P&L."""
        return self.realized_pnl + self.unrealized_pnl

    @property
    def pnl_pct(self) -> float:
        """P&L as percentage of cost basis."""
        if self.cost_basis == 0:
            return 0.0
        return self.unrealized_pnl / self.cost_basis * 100.0

    def add_fill(self, fill_price: float, fill_quantity: int, fill_lots: int,
                 commission: float = 0.0, timestamp: datetime | None = None) -> None:
        """Process a new fill that adds to the position.

        Updates average entry price using volume-weighted averaging.
        """
        if self.quantity == 0:
            self.avg_entry_price = fill_price
            self.opened_at = timestamp
        else:
            # Volume-weighted average
            total_cost = self.avg_entry_price * self.quantity + fill_price * fill_quantity
            self.avg_entry_price = total_cost / (self.quantity + fill_quantity)

        self.quantity += fill_quantity
        self.lots += fill_lots
        self.commission_paid += commission
        self.last_updated = timestamp

    def reduce(self, fill_price: float, fill_quantity: int, fill_lots: int,
               commission: float = 0.0, timestamp: datetime | None = None) -> float:
        """Process a fill that reduces the position.

        Returns the realized P&L from this partial close.
        """
        if fill_quantity > self.quantity:
            fill_quantity = self.quantity
            fill_lots = self.lots

        # Calculate realized P&L for the closed portion
        price_diff = fill_price - self.avg_entry_price
        realized = price_diff * fill_quantity * self.direction
        self.realized_pnl += realized

        self.quantity -= fill_quantity
        self.lots -= fill_lots
        self.commission_paid += commission
        self.last_updated = timestamp

        if self.quantity == 0:
            self.lots = 0

        return realized

    def mark_to_market(self, price: float, timestamp: datetime | None = None) -> None:
        """Update the current market price."""
        self.current_price = price
        if timestamp:
            self.last_updated = timestamp

    def close(self, fill_price: float, commission: float = 0.0,
              timestamp: datetime | None = None) -> float:
        """Fully close the position. Returns realized P&L."""
        return self.reduce(fill_price, self.quantity, self.lots, commission, timestamp)

    def to_dict(self) -> dict:
        return {
            "instrument_key": self.instrument_key,
            "side": self.side.value,
            "quantity": self.quantity,
            "lots": self.lots,
            "avg_entry_price": round(self.avg_entry_price, 2),
            "current_price": round(self.current_price, 2),
            "unrealized_pnl": round(self.unrealized_pnl, 2),
            "realized_pnl": round(self.realized_pnl, 2),
            "total_pnl": round(self.total_pnl, 2),
            "commission_paid": round(self.commission_paid, 2),
            "strategy_name": self.strategy_name,
            "opened_at": self.opened_at,
        }

    def __repr__(self) -> str:
        side_str = "LONG" if self.is_long else "SHORT"
        return (
            f"Position({side_str} {self.lots}L {self.instrument_key} "
            f"entry={self.avg_entry_price:.2f} "
            f"curr={self.current_price:.2f} "
            f"pnl={self.total_pnl:.2f})"
        )
