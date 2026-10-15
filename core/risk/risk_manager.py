"""Risk manager — validates signals before they become orders.

Sits between the strategy and the executor. Every SignalEvent passes
through here. The risk manager checks position limits, drawdown,
and exposure before converting a signal into an OrderEvent.
"""

from __future__ import annotations

from datetime import datetime
import uuid

from loguru import logger

from config.constants import Side
from config.settings import settings
from core.events.event_bus import EventBus
from core.events.event_types import SignalEvent, OrderEvent
from core.portfolio.portfolio import Portfolio


class RiskManager:
    """Validates trading signals against risk limits.

    Subscribes to SignalEvent, validates, and emits OrderEvent if approved.

    Checks:
    - Max position size (per instrument)
    - Max portfolio drawdown
    - Max number of open positions
    - Available cash for new positions
    """

    def __init__(
        self,
        event_bus: EventBus,
        portfolio: Portfolio,
        max_position_lots: int = 50,
        max_open_positions: int = 20,
        max_drawdown_pct: float = 0.15,
        lot_size: int = 75,
    ):
        self.event_bus = event_bus
        self.portfolio = portfolio
        self.max_position_lots = max_position_lots
        self.max_open_positions = max_open_positions
        self.max_drawdown_pct = max_drawdown_pct
        self.lot_size = lot_size
        self._approved = 0
        self._rejected = 0

        self.event_bus.subscribe(SignalEvent, self.on_signal)

    def on_signal(self, event: SignalEvent) -> None:
        """Validate a signal and emit an order if approved."""
        rejection_reason = self._validate(event)

        if rejection_reason:
            self._rejected += 1
            logger.warning(
                f"Signal REJECTED: {event.side.value} {event.quantity}L "
                f"{event.instrument_key} — {rejection_reason}"
            )
            return

        # Convert signal to order
        self._approved += 1
        quantity_units = event.quantity * self.lot_size

        order = OrderEvent(
            timestamp=event.timestamp,
            order_id=str(uuid.uuid4())[:8],
            instrument_key=event.instrument_key,
            side=event.side,
            quantity=quantity_units,
            lots=event.quantity,
            order_type="MARKET",
            limit_price=event.target_price,
            strategy_name=event.strategy_name,
            signal_id=event.event_id,
            metadata=event.metadata,
        )

        logger.debug(
            f"Signal APPROVED: {event.side.value} {event.quantity}L "
            f"{event.instrument_key} → Order {order.order_id}"
        )

        self.event_bus.emit(order)

    def _validate(self, signal: SignalEvent) -> str | None:
        """Run all risk checks. Returns rejection reason or None if OK."""

        # Check 1: Max drawdown
        if self.portfolio.initial_capital > 0:
            drawdown = (self.portfolio.initial_capital - self.portfolio.nav) / self.portfolio.initial_capital
            if drawdown >= self.max_drawdown_pct:
                return f"Max drawdown breached ({drawdown:.1%} >= {self.max_drawdown_pct:.1%})"

        # Check 2: Max open positions (for new entries only)
        if signal.signal_type == "entry":
            if self.portfolio.position_count >= self.max_open_positions:
                return f"Max positions reached ({self.portfolio.position_count}/{self.max_open_positions})"

        # Check 3: Position size limit
        existing = self.portfolio.get_position(signal.instrument_key)
        current_lots = existing.lots if existing else 0
        if signal.side == (existing.side if existing else signal.side):
            # Adding to same direction
            new_total = current_lots + signal.quantity
            if new_total > self.max_position_lots:
                return f"Position size limit ({new_total} > {self.max_position_lots} lots)"

        # Check 4: Sufficient cash for buys
        # For options: you pay the premium (price × quantity)
        # For futures: you pay margin (~15-20% of notional)
        if signal.side == Side.BUY and signal.target_price:
            if signal.is_option:
                required_cash = signal.target_price * signal.quantity * self.lot_size
            else:
                # Futures margin — approximate at 15% of notional
                required_cash = signal.target_price * signal.quantity * self.lot_size * 0.15
            if required_cash > self.portfolio.cash * 0.95:  # keep 5% buffer
                return f"Insufficient cash (need {required_cash:.0f}, have {self.portfolio.cash:.0f})"

        return None

    @property
    def approved_count(self) -> int:
        return self._approved

    @property
    def rejected_count(self) -> int:
        return self._rejected
