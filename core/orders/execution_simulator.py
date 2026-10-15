"""Order model and execution simulator.

Handles the lifecycle: OrderEvent → simulated fill → FillEvent.
Models slippage, commissions, and partial fills.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from loguru import logger

from config.constants import Side
from core.events.event_bus import EventBus
from core.events.event_types import OrderEvent, FillEvent


@dataclass
class CommissionModel:
    """Commission structure for Indian F&O.

    NSE F&O commissions:
    - Brokerage: flat per order or per lot
    - STT: on sell side for options (0.0625% of premium)
    - Exchange charges: ~0.05% of turnover
    - GST: 18% on brokerage + exchange charges
    - Stamp duty: 0.003% on buy side
    """

    brokerage_per_order: float = 20.0      # flat ₹20 per order (discount broker)
    stt_rate: float = 0.000625             # 0.0625% on sell premium
    exchange_rate: float = 0.0005          # 0.05% of turnover
    gst_rate: float = 0.18                 # 18% on brokerage + exchange charges
    stamp_duty_rate: float = 0.00003       # 0.003% on buy side

    def calculate(self, price: float, quantity: int, side: Side) -> float:
        """Calculate total commission for a trade."""
        turnover = price * quantity

        brokerage = self.brokerage_per_order
        exchange_charges = turnover * self.exchange_rate
        gst = (brokerage + exchange_charges) * self.gst_rate

        stt = 0.0
        if side == Side.SELL:
            stt = turnover * self.stt_rate

        stamp = 0.0
        if side == Side.BUY:
            stamp = turnover * self.stamp_duty_rate

        total = brokerage + stt + exchange_charges + gst + stamp
        return round(total, 2)


class ExecutionSimulator:
    """Simulates order execution with slippage and commissions.

    Subscribes to OrderEvent, processes the order, and emits FillEvent.

    Usage:
        executor = ExecutionSimulator(event_bus)
        # executor automatically listens for OrderEvents
    """

    def __init__(
        self,
        event_bus: EventBus,
        commission_model: CommissionModel | None = None,
        slippage_pct: float = 0.001,  # 0.1% default slippage
    ):
        self.event_bus = event_bus
        self.commission = commission_model or CommissionModel()
        self.slippage_pct = slippage_pct
        self._fill_count = 0

        # Subscribe to order events
        self.event_bus.subscribe(OrderEvent, self.on_order)

    def on_order(self, event: OrderEvent) -> None:
        """Process an order event and emit a fill.

        For backtesting, we assume all market orders fill immediately
        at the current price ± slippage.
        """
        if event.quantity <= 0:
            logger.warning(f"Skipping zero-quantity order: {event.order_id}")
            return

        # Determine fill price
        if event.order_type == "LIMIT" and event.limit_price is not None:
            base_price = event.limit_price
        else:
            # Market order — use limit_price as reference (set by risk manager)
            base_price = event.limit_price or 0.0

        if base_price <= 0:
            logger.warning(f"No valid price for order {event.order_id}")
            return

        # Apply slippage
        fill_price = self._apply_slippage(base_price, event.side)

        # Calculate commission
        commission = self.commission.calculate(
            fill_price, event.quantity, event.side
        )

        # Calculate slippage cost
        slippage = abs(fill_price - base_price) * event.quantity

        self._fill_count += 1

        # Emit fill event
        fill = FillEvent(
            timestamp=event.timestamp,
            order_id=event.order_id,
            instrument_key=event.instrument_key,
            side=event.side,
            quantity=event.quantity,
            lots=event.lots,
            fill_price=round(fill_price, 2),
            commission=commission,
            slippage=round(slippage, 2),
            strategy_name=event.strategy_name,
            metadata=event.metadata,
        )

        logger.debug(
            f"Fill: {event.side.value} {event.lots}L {event.instrument_key} "
            f"@{fill_price:.2f} (slip={slippage:.2f}, comm={commission:.2f})"
        )

        self.event_bus.emit(fill)

    def _apply_slippage(self, price: float, side: Side) -> float:
        """Apply slippage to the fill price.

        Buys fill higher (adverse), sells fill lower (adverse).
        """
        slippage_amount = price * self.slippage_pct
        if side == Side.BUY:
            return price + slippage_amount
        else:
            return max(0.01, price - slippage_amount)

    @property
    def fill_count(self) -> int:
        return self._fill_count
