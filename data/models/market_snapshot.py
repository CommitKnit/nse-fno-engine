"""MarketSnapshot — everything a strategy sees at one bar.

Built by ``data/fetchers/snapshot_builder.py`` and consumed bar-by-bar by
``core/engine/backtest_engine.py``: the F1 futures bar, the Kite spot bar (or None),
and the option chains keyed by expiry string.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from data.models.bar import Bar
from data.models.option_chain import OptionChain


@dataclass
class MarketSnapshot:
    """Everything the strategy sees at one timestamp.

    Attributes:
        timestamp: The bar timestamp.
        futures_bar: OHLCV bar for the near-month future.
        spot_bar: OHLCV bar for the underlying index (optional).
        chains: Option chains keyed by expiry date string.
                Usually 1-3 chains (current week, next week, monthly).
    """

    timestamp: datetime
    futures_bar: Bar
    spot_bar: Bar | None = None
    chains: dict[str, OptionChain] = field(default_factory=dict)

    @property
    def futures_price(self) -> float:
        """Current futures close price."""
        return self.futures_bar.close

    @property
    def spot_price(self) -> float | None:
        """Current spot close price, if available."""
        return self.spot_bar.close if self.spot_bar else None

    @property
    def basis(self) -> float | None:
        """Absolute futures-spot gap (cost of carry indicator).

        Always non-negative. A negative futures-spot difference (backwardation)
        is folded into the absolute value here — check `is_dividend_borrow` to
        tell that case apart from ordinary positive cost-of-carry (contango).
        """
        if self.spot_bar is None:
            return None
        return abs(self.futures_bar.close - self.spot_bar.close)

    @property
    def is_dividend_borrow(self) -> bool | None:
        """True if futures trade below spot (backwardation) — dividend/borrow-rate driven."""
        if self.spot_bar is None:
            return None
        return self.futures_bar.close < self.spot_bar.close

    @property
    def basis_pct(self) -> float | None:
        """Basis as percentage of spot. Always non-negative (basis already is)."""
        if self.spot_bar is None or self.spot_bar.close == 0:
            return None
        return (self.basis or 0) / self.spot_bar.close * 100.0

    @property
    def nearest_chain(self) -> OptionChain | None:
        """Chain with the smallest DTE (current week/month)."""
        if not self.chains:
            return None
        return min(self.chains.values(), key=lambda c: c.dte)

    @property
    def active_expiries(self) -> list[str]:
        """Sorted list of expiry date strings with DTE > 0."""
        return sorted(
            k for k, v in self.chains.items() if v.dte > 0
        )

    def __repr__(self) -> str:
        chain_info = ", ".join(
            f"{k}(DTE={v.dte})" for k, v in sorted(self.chains.items())
        )
        return (
            f"MarketSnapshot({self.timestamp:%Y-%m-%d %H:%M} "
            f"F={self.futures_price:.0f} "
            f"chains=[{chain_info}])"
        )
