"""Backtest engine — the main orchestrator.

Wires together all modules (event bus, strategy, risk manager, executor,
portfolio) and runs the bar-by-bar simulation loop.

Usage:
    engine = BacktestEngine(
        snapshots=market_snapshots,
        strategy=my_strategy,
        initial_capital=1_000_000,
    )
    results = engine.run()
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from loguru import logger

from core.events.event_bus import EventBus
from core.events.event_types import MarketEvent, FillEvent
from core.portfolio.portfolio import Portfolio
from core.orders.execution_simulator import ExecutionSimulator, CommissionModel
from core.risk.risk_manager import RiskManager
from core.strategy.base_strategy import BaseStrategy
from data.models.market_snapshot import MarketSnapshot


class BacktestEngine:
    """Main backtesting orchestrator.

    Wires: EventBus → Strategy → RiskManager → Executor → Portfolio
    Runs: iterates through MarketSnapshots bar-by-bar.
    """

    def __init__(
        self,
        snapshots: list[MarketSnapshot],
        strategy: BaseStrategy,
        initial_capital: float = 1_000_000.0,
        commission_model: CommissionModel | None = None,
        slippage_pct: float = 0.001,
        max_position_lots: int = 50,
        max_open_positions: int = 20,
        max_drawdown_pct: float = 0.15,
        lot_size: int = 75,
    ):
        self.snapshots = sorted(snapshots, key=lambda s: s.timestamp)

        # Core components
        self.event_bus = EventBus()
        self.portfolio = Portfolio(initial_capital)

        # Execution
        self.executor = ExecutionSimulator(
            self.event_bus,
            commission_model=commission_model,
            slippage_pct=slippage_pct,
        )

        # Risk
        self.risk_manager = RiskManager(
            self.event_bus,
            self.portfolio,
            max_position_lots=max_position_lots,
            max_open_positions=max_open_positions,
            max_drawdown_pct=max_drawdown_pct,
            lot_size=lot_size,
        )

        # Strategy — connect to event bus
        self.strategy = strategy
        self.strategy.event_bus = self.event_bus
        self.event_bus.subscribe(MarketEvent, self.strategy._handle_market)
        self.event_bus.subscribe(FillEvent, self.strategy._handle_fill)

        # Connect portfolio to fills
        self.event_bus.subscribe(FillEvent, self.portfolio.on_fill)

        # State
        self._current_bar = 0
        self._is_running = False

    def run(self) -> dict:
        """Run the full backtest.

        Returns:
            Dict with portfolio summary, trade list, NAV history.
        """
        total_bars = len(self.snapshots)
        if total_bars == 0:
            logger.warning("No snapshots to process")
            return self._build_results()

        logger.info(
            f"Starting backtest: {total_bars} bars, "
            f"capital={self.portfolio.initial_capital:,.0f}, "
            f"strategy={self.strategy.name}"
        )

        self._is_running = True

        for i, snapshot in enumerate(self.snapshots):
            self._current_bar = i

            # 1. Update positions with current prices
            self._mark_positions(snapshot)

            # 2. Record NAV
            self.portfolio.record_nav(snapshot.timestamp)

            # 3. Emit market event → strategy reacts → signals → risk → orders → fills
            market_event = MarketEvent(
                timestamp=snapshot.timestamp,
                snapshot=snapshot,
            )
            self.event_bus.emit(market_event)

            # Progress logging
            if (i + 1) % 50 == 0 or i + 1 == total_bars:
                logger.info(
                    f"Bar {i + 1}/{total_bars} | "
                    f"{snapshot.timestamp:%Y-%m-%d} | "
                    f"NAV={self.portfolio.nav:,.0f} | "
                    f"PnL={self.portfolio.total_pnl:,.0f} | "
                    f"Positions={self.portfolio.position_count}"
                )

        self._is_running = False

        logger.info(
            f"Backtest complete: {total_bars} bars processed, "
            f"Final NAV={self.portfolio.nav:,.0f}, "
            f"PnL={self.portfolio.total_pnl:,.0f} "
            f"({self.portfolio.total_pnl / self.portfolio.initial_capital:.1%})"
        )

        return self._build_results()

    def _mark_positions(self, snapshot: MarketSnapshot) -> None:
        """Update all open positions with latest prices from snapshot."""
        prices: dict[str, float] = {}

        # Futures price
        if snapshot.futures_bar:
            prices[snapshot.futures_bar.instrument_key] = snapshot.futures_bar.close

        # Option prices from all chains
        for chain in snapshot.chains.values():
            for contract in chain.contracts:
                if contract.close > 0:
                    prices[contract.instrument_key] = contract.close

        self.portfolio.mark_to_market(prices, snapshot.timestamp)

    def _build_results(self) -> dict:
        """Build the results dictionary."""
        return {
            "summary": self.portfolio.summary(),
            "nav_history": self.portfolio.nav_history,
            "closed_trades": self.portfolio.closed_trades,
            "event_counts": self.event_bus.event_counts,
            "strategy_stats": {
                "name": self.strategy.name,
                "bars_processed": self.strategy.bar_count,
                "signals_generated": self.strategy.signal_count,
            },
            "risk_stats": {
                "approved": self.risk_manager.approved_count,
                "rejected": self.risk_manager.rejected_count,
            },
            "execution_stats": {
                "fills": self.executor.fill_count,
            },
        }
