"""Slippage model — estimate synthetic bid-ask spread for options.

Since Upstox historical data gives OHLCV but not actual bid-ask quotes,
we need to model the spread. The spread depends on:
    1. Moneyness — ATM options have tighter spreads than deep OTM
    2. Open Interest — high OI = more liquidity = tighter spread
    3. Volume — high volume = active trading = tighter spread
    4. Time to expiry — near-expiry illiquid strikes widen dramatically
    5. Option price level — ₹0.50 options can't have a ₹5 spread

This model produces synthetic bid/ask prices used by the execution
simulator to model realistic fill prices.
"""

from __future__ import annotations

import math

from data.models.option_contract import OptionContract


class SlippageModel:
    """Estimate synthetic bid-ask spread for option contracts.

    Usage:
        model = SlippageModel()
        bid, ask = model.estimate_spread(contract, futures_price=23500)
        # bid = 295.0, ask = 305.0 (for a ₹300 ATM option)

        # Or get the half-spread (slippage per side)
        slippage = model.estimate_slippage(contract, futures_price=23500)
        # slippage = 5.0 (you pay 5 points worse than mid)
    """

    def __init__(
        self,
        base_spread_pct: float = 0.005,
        moneyness_multiplier: float = 2.0,
        liquidity_scale: float = 1.0,
        min_spread_ticks: float = 1.0,
        tick_size: float = 0.05,
    ):
        """
        Args:
            base_spread_pct: Base spread as fraction of option price (0.5% default).
                ATM liquid options typically have 0.3-0.5% spread.
            moneyness_multiplier: How much spread widens for OTM options.
                At 2.0, a 10% OTM option has 2x the base spread.
            liquidity_scale: Global scaling factor (1.0 = normal, 2.0 = wide markets).
            min_spread_ticks: Minimum spread in tick size units.
            tick_size: Minimum price movement (₹0.05 for NSE options).
        """
        self.base_spread_pct = base_spread_pct
        self.moneyness_multiplier = moneyness_multiplier
        self.liquidity_scale = liquidity_scale
        self.min_spread_ticks = min_spread_ticks
        self.tick_size = tick_size

    def estimate_spread(
        self,
        contract: OptionContract,
        futures_price: float,
    ) -> tuple[float, float]:
        """Estimate synthetic bid and ask prices.

        Args:
            contract: Option contract with close, volume, OI, strike.
            futures_price: Current futures price (for moneyness calc).

        Returns:
            Tuple of (bid_price, ask_price).
        """
        mid = contract.close
        if mid <= 0:
            return (0.0, 0.0)

        half_spread = self.estimate_slippage(contract, futures_price)

        bid = max(0.0, mid - half_spread)
        ask = mid + half_spread

        # Round to tick size
        bid = self._round_to_tick(bid)
        ask = self._round_to_tick(ask)

        # Ensure ask > bid
        if ask <= bid:
            ask = bid + self.tick_size

        return (bid, ask)

    def estimate_slippage(
        self,
        contract: OptionContract,
        futures_price: float,
    ) -> float:
        """Estimate half-spread (slippage per side) in price points.

        This is what you'd add to mid price when buying,
        or subtract when selling.

        Args:
            contract: Option contract.
            futures_price: Current futures price.

        Returns:
            Half-spread in absolute price points.
        """
        mid = contract.close
        if mid <= 0:
            return 0.0

        # ── Factor 1: Base spread from price level ──
        base = mid * self.base_spread_pct

        # ── Factor 2: Moneyness adjustment ──
        # OTM options have wider spreads
        moneyness_factor = self._moneyness_factor(
            contract.strike, futures_price, contract.is_call
        )

        # ── Factor 3: Liquidity adjustment ──
        # High OI + volume = tighter spread
        liquidity_factor = self._liquidity_factor(contract.oi, contract.volume)

        # ── Factor 4: DTE adjustment ──
        # Very near expiry + illiquid = wider spread
        dte_factor = self._dte_factor(contract.dte)

        # ── Factor 5: Price level floor ──
        # Cheap options have a wider spread relative to price
        # A ₹2 option can't have a ₹0.01 spread — minimum is 1-2 ticks
        price_floor = self._price_level_factor(mid)

        # Combine factors
        half_spread = (
            base
            * moneyness_factor
            * liquidity_factor
            * dte_factor
            * price_floor
            * self.liquidity_scale
        )

        # Enforce minimum spread
        min_spread = self.min_spread_ticks * self.tick_size
        half_spread = max(half_spread, min_spread / 2)

        # Cap at 20% of mid price (spread can't be wider than the option itself)
        half_spread = min(half_spread, mid * 0.20)

        return half_spread

    def fill_price(
        self,
        contract: OptionContract,
        futures_price: float,
        is_buy: bool,
    ) -> float:
        """Get the simulated fill price for a trade.

        Args:
            contract: Option contract.
            futures_price: Current futures price.
            is_buy: True if buying, False if selling.

        Returns:
            Simulated fill price (worse than mid by half-spread).
        """
        bid, ask = self.estimate_spread(contract, futures_price)
        if is_buy:
            return ask  # buyer pays the ask
        return bid  # seller receives the bid

    # ── Internal factor calculations ──

    def _moneyness_factor(
        self, strike: float, futures_price: float, is_call: bool
    ) -> float:
        """Wider spread for OTM options."""
        if futures_price <= 0:
            return 1.0

        if is_call:
            otm_pct = (strike - futures_price) / futures_price
        else:
            otm_pct = (futures_price - strike) / futures_price

        # ITM or ATM: factor = 1.0
        # OTM: increases linearly with distance
        if otm_pct <= 0:
            return 1.0

        return 1.0 + otm_pct * self.moneyness_multiplier * 10

    def _liquidity_factor(self, oi: int, volume: int) -> float:
        """Tighter spread for liquid contracts."""
        # High OI (>100k) → factor ≈ 0.7 (30% tighter)
        # Low OI (<1k) → factor ≈ 2.0 (100% wider)
        if oi <= 0 and volume <= 0:
            return 3.0  # completely illiquid

        oi_factor = 1.0
        if oi > 0:
            # Log scale: OI=1000 → 1.5, OI=10000 → 1.0, OI=100000 → 0.7
            oi_factor = max(0.5, 2.0 - 0.25 * math.log10(max(oi, 1)))

        vol_factor = 1.0
        if volume > 0:
            vol_factor = max(0.5, 2.0 - 0.25 * math.log10(max(volume, 1)))

        # Average of OI and volume factors
        return (oi_factor + vol_factor) / 2

    def _dte_factor(self, dte: int | None) -> float:
        """Wider spreads near expiry for illiquid strikes."""
        if dte is None:
            return 1.0
        if dte <= 0:
            return 2.0  # expiry day — very wide
        if dte <= 1:
            return 1.5
        if dte <= 3:
            return 1.2
        return 1.0

    def _price_level_factor(self, price: float) -> float:
        """Cheap options have relatively wider spreads."""
        if price >= 50:
            return 1.0
        if price >= 10:
            return 1.3
        if price >= 2:
            return 1.8
        if price >= 0.5:
            return 2.5
        return 3.0  # sub-₹0.50 options — very wide relative spread

    def _round_to_tick(self, price: float) -> float:
        """Round price to nearest tick size."""
        return round(round(price / self.tick_size) * self.tick_size, 2)