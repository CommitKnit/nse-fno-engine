"""Event bus — pub/sub dispatcher for the backtesting engine.

Modules subscribe to event types they care about. When an event is
emitted, all subscribers for that type get called in order.

Usage:
    bus = EventBus()
    bus.subscribe(MarketEvent, strategy.on_market)
    bus.subscribe(SignalEvent, risk_manager.on_signal)
    bus.subscribe(FillEvent, portfolio.on_fill)

    bus.emit(MarketEvent(timestamp=now, snapshot=snap))
    # → strategy.on_market(event) gets called
"""

from __future__ import annotations

from collections import defaultdict
from typing import Callable, Type

from loguru import logger

from core.events.event_types import Event


class EventBus:
    """Simple synchronous pub/sub event dispatcher.

    Events are dispatched immediately when emitted (synchronous).
    Subscribers are called in the order they were registered.
    """

    def __init__(self):
        self._subscribers: dict[Type[Event], list[Callable]] = defaultdict(list)
        self._event_count: dict[str, int] = defaultdict(int)

    def subscribe(self, event_type: Type[Event], handler: Callable) -> None:
        """Register a handler for an event type.

        Args:
            event_type: The Event subclass to listen for.
            handler: Callable that takes one argument (the event).
        """
        self._subscribers[event_type].append(handler)
        logger.debug(
            f"Subscribed {handler.__qualname__} to {event_type.__name__}"
        )

    def unsubscribe(self, event_type: Type[Event], handler: Callable) -> None:
        """Remove a handler for an event type."""
        handlers = self._subscribers.get(event_type, [])
        if handler in handlers:
            handlers.remove(handler)

    def emit(self, event: Event) -> None:
        """Dispatch an event to all registered handlers.

        Args:
            event: The event instance to dispatch.
        """
        event_type = type(event)
        self._event_count[event_type.__name__] += 1

        handlers = self._subscribers.get(event_type, [])
        for handler in handlers:
            try:
                handler(event)
            except Exception as e:
                logger.error(
                    f"Handler {handler.__qualname__} failed on "
                    f"{event_type.__name__}: {e}"
                )
                raise

    def emit_many(self, events: list[Event]) -> None:
        """Dispatch multiple events in order."""
        for event in events:
            self.emit(event)

    @property
    def subscriber_count(self) -> int:
        return sum(len(h) for h in self._subscribers.values())

    @property
    def event_counts(self) -> dict[str, int]:
        return dict(self._event_count)

    def reset_counts(self) -> None:
        self._event_count.clear()

    def clear(self) -> None:
        """Remove all subscribers."""
        self._subscribers.clear()
        self._event_count.clear()
