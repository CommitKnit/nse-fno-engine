"""Tests for execution simulator and full backtest engine integration."""

import pytest
from datetime import datetime, date

from config.constants import Side, OptionType
from core.events.event_bus import EventBus
from core.events.event_types import OrderEvent, FillEvent, SignalEvent, MarketEvent
from core.orders.execution_simulator import ExecutionSimulator, CommissionModel
from core.portfolio.portfolio import Portfolio
from core.risk.risk_manager import RiskManager
from core.strategy.base_strategy import BaseStrategy
from core.engine.backtest_engine import BacktestEngine
from data.models.bar import Bar
from data.models.market_snapshot import MarketSnapshot
from data.models.option_chain import OptionChain


class TestCommissionModel:

    def test_buy_commission(self):
        model = CommissionModel()
        comm = model.calculate(price=300.0, quantity=75, side=Side.BUY)
        assert comm > 0
        # Should include brokerage + exchange + GST + stamp duty (no STT on buy)
        assert comm < 300  # sanity check

    def test_sell_has_stt(self):
        model = CommissionModel()
        buy_comm = model.calculate(300.0, 75, Side.BUY)
        sell_comm = model.calculate(300.0, 75, Side.SELL)
        assert sell_comm > buy_comm  # sell has STT


class TestExecutionSimulator:

    def test_fill_emitted_on_order(self):
        bus = EventBus()
        executor = ExecutionSimulator(bus, slippage_pct=0.001)

        fills = []
        bus.subscribe(FillEvent, lambda e: fills.append(e))

        order = OrderEvent(
            timestamp=datetime(2025, 1, 15),
            order_id="test1",
            instrument_key="NSE_FO|12345",
            side=Side.BUY,
            quantity=75,
            lots=1,
            limit_price=300.0,
            strategy_name="test",
        )
        bus.emit(order)

        assert len(fills) == 1
        assert fills[0].instrument_key == "NSE_FO|12345"
        assert fills[0].side == Side.BUY
        assert fills[0].quantity == 75

    def test_buy_fills_above_reference(self):
        bus = EventBus()
        executor = ExecutionSimulator(bus, slippage_pct=0.01)

        fills = []
        bus.subscribe(FillEvent, lambda e: fills.append(e))

        order = OrderEvent(
            timestamp=datetime(2025, 1, 15),
            order_id="test2",
            instrument_key="X",
            side=Side.BUY,
            quantity=75,
            lots=1,
            limit_price=100.0,
        )
        bus.emit(order)

        assert fills[0].fill_price > 100.0  # slippage pushes buy up

    def test_sell_fills_below_reference(self):
        bus = EventBus()
        executor = ExecutionSimulator(bus, slippage_pct=0.01)

        fills = []
        bus.subscribe(FillEvent, lambda e: fills.append(e))

        order = OrderEvent(
            timestamp=datetime(2025, 1, 15),
            order_id="test3",
            instrument_key="X",
            side=Side.SELL,
            quantity=75,
            lots=1,
            limit_price=100.0,
        )
        bus.emit(order)

        assert fills[0].fill_price < 100.0  # slippage pushes sell down

    def test_commission_included_in_fill(self):
        bus = EventBus()
        executor = ExecutionSimulator(bus)

        fills = []
        bus.subscribe(FillEvent, lambda e: fills.append(e))

        order = OrderEvent(
            timestamp=datetime(2025, 1, 15),
            order_id="test4",
            instrument_key="X",
            side=Side.BUY,
            quantity=75,
            lots=1,
            limit_price=300.0,
        )
        bus.emit(order)

        assert fills[0].commission > 0


class TestRiskManager:

    def test_signal_approved_and_order_emitted(self):
        bus = EventBus()
        portfolio = Portfolio(1_000_000)
        risk = RiskManager(bus, portfolio, lot_size=75)

        orders = []
        bus.subscribe(OrderEvent, lambda e: orders.append(e))

        signal = SignalEvent(
            timestamp=datetime(2025, 1, 15),
            instrument_key="X",
            side=Side.BUY,
            quantity=1,
            signal_type="entry",
            target_price=300.0,
        )
        bus.emit(signal)

        assert len(orders) == 1
        assert orders[0].quantity == 75  # 1 lot × 75

    def test_signal_rejected_max_drawdown(self):
        bus = EventBus()
        portfolio = Portfolio(1_000_000)
        portfolio.cash = 800_000  # simulate 20% drawdown
        risk = RiskManager(bus, portfolio, max_drawdown_pct=0.15)

        orders = []
        bus.subscribe(OrderEvent, lambda e: orders.append(e))

        signal = SignalEvent(
            timestamp=datetime(2025, 1, 15),
            instrument_key="X",
            side=Side.BUY,
            quantity=1,
            signal_type="entry",
            target_price=300.0,
        )
        bus.emit(signal)

        assert len(orders) == 0  # rejected
        assert risk.rejected_count == 1


class TestBacktestEngine:
    """Integration test — full engine with a simple strategy."""

    def _make_snapshots(self, n_bars=10):
        """Create simple MarketSnapshots with rising futures."""
        snapshots = []
        for i in range(n_bars):
            price = 23500 + i * 50
            ts = datetime(2025, 1, 15 + i, 15, 30)
            futures_bar = Bar(
                timestamp=ts, open=price - 10, high=price + 20,
                low=price - 30, close=price, volume=1_000_000,
                instrument_key="NSE_FO|FUT",
            )
            snapshots.append(MarketSnapshot(
                timestamp=ts,
                futures_bar=futures_bar,
                chains={},
            ))
        return snapshots

    def test_engine_runs_without_error(self):
        """Engine should complete even with a no-op strategy."""

        class NoOpStrategy(BaseStrategy):
            def on_market(self, snapshot):
                pass

        snapshots = self._make_snapshots()
        strategy = NoOpStrategy(name="noop")
        engine = BacktestEngine(snapshots=snapshots, strategy=strategy)
        results = engine.run()

        assert results["summary"]["nav"] == 1_000_000
        assert results["strategy_stats"]["bars_processed"] == 10

    def test_engine_with_simple_buy_strategy(self):
        """Strategy that buys on bar 2 and sells on bar 5."""

        class SimpleBuyStrategy(BaseStrategy):
            def on_market(self, snapshot):
                if self.bar_count == 2:
                    self.add_signal(SignalEvent(
                        timestamp=snapshot.timestamp,
                        instrument_key="OPT_CE_23500",
                        side=Side.BUY,
                        quantity=1,
                        signal_type="entry",
                        target_price=300.0,  # option premium, not futures notional
                        option_type=OptionType.CE,
                    ))
                elif self.bar_count == 5:
                    self.add_signal(SignalEvent(
                        timestamp=snapshot.timestamp,
                        instrument_key="OPT_CE_23500",
                        side=Side.SELL,
                        quantity=1,
                        signal_type="exit",
                        target_price=350.0,
                        option_type=OptionType.CE,
                    ))

        snapshots = self._make_snapshots(n_bars=10)
        strategy = SimpleBuyStrategy(name="simple_buy")
        engine = BacktestEngine(
            snapshots=snapshots,
            strategy=strategy,
            slippage_pct=0.0,
        )
        results = engine.run()

        assert results["strategy_stats"]["signals_generated"] == 2
        assert results["risk_stats"]["approved"] == 2
        assert results["execution_stats"]["fills"] == 2

    def test_engine_nav_history_recorded(self):
        class NoOpStrategy(BaseStrategy):
            def on_market(self, snapshot):
                pass

        snapshots = self._make_snapshots(n_bars=5)
        strategy = NoOpStrategy(name="noop")
        engine = BacktestEngine(snapshots=snapshots, strategy=strategy)
        results = engine.run()

        assert len(results["nav_history"]) == 5
