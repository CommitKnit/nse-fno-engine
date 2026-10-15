"""Portfolio — tracks cash, all positions, and net asset value (NAV).

The portfolio is the single source of truth for "how much money do I have"
at any point during the backtest.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from loguru import logger

from config.constants import Side
from core.events.event_types import FillEvent
from core.portfolio.position import Position


class Portfolio:
    """Manages cash, positions, and NAV.

    Usage:
        portfolio = Portfolio(initial_capital=1_000_000)
        portfolio.on_fill(fill_event)       # process fills
        portfolio.mark_to_market(prices)    # update all positions
        print(portfolio.nav)                # total value
    """

    def __init__(self, initial_capital: float = 1_000_000.0):
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.positions: dict[str, Position] = {}  # instrument_key → Position
        self.closed_trades: list[dict] = []       # history of closed positions
        self.nav_history: list[dict] = []         # time series of NAV
        self._trade_count = 0

    @property
    def nav(self) -> float:
        """Net Asset Value = cash + unrealized P&L of all positions."""
        unrealized = sum(p.unrealized_pnl for p in self.positions.values() if p.is_open)
        return self.cash + unrealized

    @property
    def total_unrealized_pnl(self) -> float:
        return sum(p.unrealized_pnl for p in self.positions.values() if p.is_open)

    @property
    def total_realized_pnl(self) -> float:
        return sum(t.get("realized_pnl", 0) for t in self.closed_trades)

    @property
    def total_pnl(self) -> float:
        return self.nav - self.initial_capital

    @property
    def total_commission(self) -> float:
        open_comm = sum(p.commission_paid for p in self.positions.values())
        closed_comm = sum(t.get("commission", 0) for t in self.closed_trades)
        return open_comm + closed_comm

    @property
    def open_positions(self) -> list[Position]:
        return [p for p in self.positions.values() if p.is_open]

    @property
    def position_count(self) -> int:
        return len(self.open_positions)

    def on_fill(self, event: FillEvent) -> None:
        """Process a fill event — update positions and cash.

        For a BUY: cash decreases, position quantity increases.
        For a SELL: cash increases, position quantity decreases (or opens short).
        """
        key = event.instrument_key
        self._trade_count += 1

        if key in self.positions and self.positions[key].is_open:
            position = self.positions[key]

            if event.side == position.side:
                # Adding to existing position
                position.add_fill(
                    event.fill_price, event.quantity, event.lots,
                    event.commission, event.timestamp
                )
                # Cash decreases for buys, increases for sells
                cash_impact = event.fill_price * event.quantity
                if event.side == Side.BUY:
                    self.cash -= cash_impact + event.commission
                else:
                    self.cash += cash_impact - event.commission

            else:
                # Reducing or closing existing position
                realized = position.reduce(
                    event.fill_price, event.quantity, event.lots,
                    event.commission, event.timestamp
                )
                # Cash changes by fill amount
                cash_impact = event.fill_price * event.quantity
                if event.side == Side.BUY:
                    self.cash -= cash_impact + event.commission
                else:
                    self.cash += cash_impact - event.commission

                if not position.is_open:
                    self._record_closed_trade(position, event.timestamp)

        else:
            # New position
            position = Position(
                instrument_key=key,
                side=event.side,
                strategy_name=event.strategy_name,
            )
            position.add_fill(
                event.fill_price, event.quantity, event.lots,
                event.commission, event.timestamp
            )
            self.positions[key] = position

            cash_impact = event.fill_price * event.quantity
            if event.side == Side.BUY:
                self.cash -= cash_impact + event.commission
            else:
                self.cash += cash_impact - event.commission

        logger.debug(
            f"Fill: {event.side.value} {event.lots}L {key} "
            f"@{event.fill_price:.2f} | Cash={self.cash:.0f} NAV={self.nav:.0f}"
        )

    def mark_to_market(self, prices: dict[str, float],
                       timestamp: datetime | None = None) -> None:
        """Update all positions with latest market prices.

        Args:
            prices: Dict of instrument_key → current price.
            timestamp: Current bar timestamp.
        """
        for key, position in self.positions.items():
            if position.is_open and key in prices:
                position.mark_to_market(prices[key], timestamp)

    def record_nav(self, timestamp: datetime) -> None:
        """Snapshot the current NAV for equity curve tracking."""
        self.nav_history.append({
            "timestamp": timestamp,
            "nav": self.nav,
            "cash": self.cash,
            "unrealized_pnl": self.total_unrealized_pnl,
            "position_count": self.position_count,
        })

    def get_position(self, instrument_key: str) -> Position | None:
        """Get an open position by instrument key."""
        pos = self.positions.get(instrument_key)
        if pos and pos.is_open:
            return pos
        return None

    def has_position(self, instrument_key: str) -> bool:
        return self.get_position(instrument_key) is not None

    def get_exposure(self) -> dict[str, float]:
        """Get total long and short exposure."""
        long_exp = sum(
            p.market_value for p in self.positions.values()
            if p.is_open and p.is_long
        )
        short_exp = sum(
            abs(p.market_value) for p in self.positions.values()
            if p.is_open and p.is_short
        )
        return {
            "long": long_exp,
            "short": short_exp,
            "net": long_exp - short_exp,
            "gross": long_exp + short_exp,
        }

    def _record_closed_trade(self, position: Position, timestamp: datetime | None) -> None:
        """Record a closed position for trade history."""
        self.closed_trades.append({
            "instrument_key": position.instrument_key,
            "side": position.side.value,
            "lots": position.lots,
            "avg_entry": position.avg_entry_price,
            "avg_exit": position.current_price,
            "realized_pnl": position.realized_pnl,
            "commission": position.commission_paid,
            "strategy": position.strategy_name,
            "opened_at": position.opened_at,
            "closed_at": timestamp,
        })

    def summary(self) -> dict:
        return {
            "nav": round(self.nav, 2),
            "cash": round(self.cash, 2),
            "initial_capital": self.initial_capital,
            "total_pnl": round(self.total_pnl, 2),
            "total_pnl_pct": round(self.total_pnl / self.initial_capital * 100, 2),
            "unrealized_pnl": round(self.total_unrealized_pnl, 2),
            "realized_pnl": round(self.total_realized_pnl, 2),
            "open_positions": self.position_count,
            "total_trades": len(self.closed_trades),
            "total_commission": round(self.total_commission, 2),
        }

    def __repr__(self) -> str:
        return (
            f"Portfolio(NAV={self.nav:.0f} Cash={self.cash:.0f} "
            f"Positions={self.position_count} PnL={self.total_pnl:.0f})"
        )
