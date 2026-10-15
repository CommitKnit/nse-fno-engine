# core/

The event-driven backtest engine.

**Bar loop:**
- Each bar's `MarketSnapshot` goes to the strategy.
- The strategy emits `SignalEvent`s.
- The risk manager approves them into `OrderEvent`s.
- The execution simulator fills them as `FillEvent`s.
- The portfolio books fills and marks to market.

| File | What it does |
|---|---|
| `engine/backtest_engine.py` | `BacktestEngine`, see below. |
| `events/event_types.py` | Event dataclasses with a priority order: `MarketEvent`, `SignalEvent` (strategy intent: instrument, side, quantity, price), `OrderEvent` (risk-approved), `FillEvent` (executed price, commission, slippage), `ExpiryEvent`. |
| `events/event_bus.py` | `EventBus`: pub/sub dispatcher. Modules `subscribe(EventType, handler)`, and `emit` calls the handlers in order. |
| `orders/execution_simulator.py` | `ExecutionSimulator` and `CommissionModel`, see below. |
| `portfolio/position.py` | `Position`: one instrument holding with average entry, quantity, realised/unrealised P&L and commission. Each option leg is its own position. |
| `portfolio/portfolio.py` | `Portfolio`, see below. |
| `risk/risk_manager.py` | `RiskManager`: validates each `SignalEvent` against max open positions, max lots and the drawdown circuit breaker, then emits an `OrderEvent` or rejects it. |
| `strategy/base_strategy.py` | `BaseStrategy`: the abstract interface every strategy implements. `on_market(snapshot)` is called every bar, `get_signals()` returns the signals to trade, and `on_fill(fill)` is notified of executions. |

### `engine/backtest_engine.py`

`BacktestEngine(snapshots, strategy, …)` wires the event bus, risk manager, execution simulator and portfolio together. `run()` runs the bar-by-bar loop and returns a dict with:
- `summary` (NAV and P&L totals),
- `closed_trades`,
- the equity curve,
- `execution_stats` and `risk_stats`.

### `orders/execution_simulator.py`

- `ExecutionSimulator` fills `OrderEvent`s at the bar price, applying percentage slippage.
- `CommissionModel` is NSE F&O costs: brokerage, STT, exchange charges, GST, stamp duty. The backtests pass a zero-cost instance for "gross" runs.

### `portfolio/portfolio.py`

`Portfolio` holds cash, positions and NAV.
- `on_fill` books trades.
- `mark_to_market` revalues open positions.
- `record_nav` appends to the equity curve.
- `summary()` returns the totals.
