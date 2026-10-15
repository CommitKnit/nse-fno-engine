"""Roll-spread long put: buy an F2 put (ATM or liquid OTM), delta-hedge it with the F1 future.

Thesis: during the final expiry week the F1-F2 calendar spread (F1 - F2) widens as
the roll gets under way. A long F2 put hedged with a long F1 future is, to first
order, long ``|delta| x (F1 - F2)``: the put gains as F2 falls while the F1 hedge
gains as F1 rises, so the book profits when the spread widens. On top of that it
carries the put's gamma/vega/theta.

LOOK-AHEAD BIAS (deliberate): entry and exit timestamps are supplied by the
harness from the *realised* spread path of each final expiry week (entry = the
spread trough before the week's maximum, exit = the week's maximum). This is an
upper bound on what the roll trade can capture — not a tradeable signal.

Strike: ``strike_mode="atm"`` buys the put nearest F2; ``"otm_liquid"`` buys the
highest strike below F2 whose open interest is at least ``min_oi`` (a liquidity
screen; the cycle is skipped if none qualifies).

Hedge: a no-trade band on net delta (or, with ``rehedge=False``, only the initial
delta hedge, held unchanged to exit). Each bar the put's delta is recomputed
(Black-76 on the F2 future at the last observed IV of the held strike); when
``|net delta| > band x put_qty`` the F1 hedge is traded back to delta-flat.

Units: all quantities are SHARES (put = lots x lot_size, hedge = whole shares), so
the engine must run with ``lot_size=1`` and a large ``max_position_lots`` — same
convention as SBICardLongStrangleStrategy.

Marks: the enriched chain is thin (few strikes trade per 5-min bar), so the held
put is marked by Black-76 at the last observed IV of its strike re-priced on the
current F2 future and the current fractional DTE (sticky-strike). Entry and exit
fills use the same mark, so the strategy's own per-bar P&L attribution sums to the
engine's realised P&L when costs are off.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime

import pandas as pd
from loguru import logger

from config.constants import OptionType, Side
from config.settings import settings
from core.events.event_types import FillEvent, SignalEvent
from core.strategy.base_strategy import BaseStrategy
from data.models.market_snapshot import MarketSnapshot
from options.pricing.black76 import (
    black76_delta,
    black76_gamma,
    black76_price,
    black76_theta,
    black76_vega,
)
from options.pricing.dte_utils import fractional_dte, year_fraction


@dataclass
class RollCycle:
    """One F1 expiry cycle's (look-ahead) trade window."""

    f1_expiry: date
    f2_expiry: date
    entry_ts: datetime
    exit_ts: datetime


@dataclass
class RollSpreadPutHedgeConfig:
    cycles: list[RollCycle]
    band: float = 0.05            # rehedge when |net delta| > band x put shares
    rehedge: bool = True          # False: hedge once at entry, hold that hedge to exit
    strike_mode: str = "atm"      # "atm" | "otm_liquid" (nearest OTM put with oi >= min_oi)
    min_oi: int = 200_000         # shares of open interest, for strike_mode="otm_liquid"
    lots: int = 1
    interval: str = "5minute"     # for fractional DTE
    risk_free_rate: float = field(default_factory=lambda: settings.RISK_FREE_RATE)


class RollSpreadPutHedgeStrategy(BaseStrategy):
    """Long F2 ATM put, band-delta-hedged with the F1 future, over look-ahead windows."""

    def __init__(self, config: RollSpreadPutHedgeConfig, name: str = "RollSpreadPutHedge"):
        super().__init__(name=name)
        self.config = config
        self._cycles = sorted(config.cycles, key=lambda c: c.entry_ts)
        self._done: set[date] = set()

        # Open-position state
        self.cycle: RollCycle | None = None
        self.put_key: str | None = None
        self.strike: float = 0.0
        self.qty: int = 0
        self.iv: float | None = None
        self.hedge_key: str | None = None
        self.hedge_shares: int = 0
        self._prev: dict | None = None   # previous-bar state for attribution

        # Outputs for the harness
        self.records: list[dict] = []    # one per cycle
        self.skipped: list[dict] = []    # cycles with no eligible strike
        self.paths: list[dict] = []      # one per bar while in a position
        self._rec: dict | None = None

    # ── Pricing helpers ───────────────────────────────────────────────────────

    def _T(self, ts: datetime, expiry: date) -> float:
        return year_fraction(fractional_dte(ts, expiry, self.config.interval))

    def _greeks(self, F2: float, T: float) -> dict:
        K, s, r = self.strike, self.iv, self.config.risk_free_rate
        return {
            "price": black76_price(F2, K, T, s, r, False),
            "delta": black76_delta(F2, K, T, s, r, False),
            "gamma": black76_gamma(F2, K, T, s, r),
            "vega": black76_vega(F2, K, T, s, r),     # per 1 vol point
            "theta": black76_theta(F2, K, T, s, r, False),  # per calendar day
        }

    @staticmethod
    def _find(chain, key: str):
        for c in chain.contracts:
            if c.instrument_key == key:
                return c
        return None

    def _signal(self, ts, key, side, qty, price, kind, **meta) -> None:
        self.add_signal(SignalEvent(
            timestamp=ts, instrument_key=key, side=side, quantity=int(qty),
            option_type=meta.pop("option_type", None), strike=meta.pop("strike", None),
            expiry=meta.pop("expiry", ""), signal_type=kind, target_price=price,
            metadata={"strategy": self.name, **meta},
        ))

    # ── Core hook ─────────────────────────────────────────────────────────────

    def on_market(self, snapshot: MarketSnapshot) -> None:
        ts = snapshot.timestamp
        if self.cycle is None:
            for cyc in self._cycles:
                if cyc.f1_expiry not in self._done and cyc.entry_ts <= ts <= cyc.exit_ts:
                    self._enter(snapshot, cyc)
                    break
            return

        chain = snapshot.chains.get(self.cycle.f2_expiry.isoformat())
        if chain is None or not chain.futures_price:
            return
        F1, F2 = snapshot.futures_price, chain.futures_price
        c = self._find(chain, self.put_key)
        if c is not None and c.iv:
            self.iv = c.iv
        g = self._greeks(F2, self._T(ts, self.cycle.f2_expiry))
        self._attribute(ts, F1, F2, g)

        if ts >= self.cycle.exit_ts:
            self._exit(snapshot, g, F1, F2)
        else:
            self._rebalance(ts, F1, g)

    # ── Entry / exit ──────────────────────────────────────────────────────────

    def _enter(self, snapshot: MarketSnapshot, cyc: RollCycle) -> None:
        ts = snapshot.timestamp
        chain = snapshot.chains.get(cyc.f2_expiry.isoformat())
        if chain is None or not chain.futures_price:
            return
        F1, F2 = snapshot.futures_price, chain.futures_price
        puts = [p for p in chain.puts if p.iv]
        if self.config.strike_mode == "otm_liquid":
            puts = [p for p in puts if p.strike < F2 and (p.oi or 0) >= self.config.min_oi]
            if not puts:
                self._done.add(cyc.f1_expiry)
                self.skipped.append({"f1_expiry": cyc.f1_expiry, "ts": ts, "F2": F2,
                                     "reason": f"no OTM put with oi >= {self.config.min_oi:,}"})
                logger.warning(f"{self.name}: skip {cyc.f1_expiry} — no OTM put with "
                               f"oi >= {self.config.min_oi:,} at {ts:%m-%d %H:%M}")
                return
            put = max(puts, key=lambda p: p.strike)            # nearest to ATM from below
        elif not puts:
            return
        else:
            put = min(puts, key=lambda p: abs(p.strike - F2))

        self.cycle, self.put_key, self.strike, self.iv = cyc, put.instrument_key, put.strike, put.iv
        self.qty = self.config.lots * put.lot_size
        g = self._greeks(F2, self._T(ts, cyc.f2_expiry))
        self.hedge_key = snapshot.futures_bar.instrument_key
        self.hedge_shares = 0

        self._signal(ts, self.put_key, Side.BUY, self.qty, g["price"], "entry",
                     option_type=OptionType.PE, strike=self.strike,
                     expiry=cyc.f2_expiry.isoformat(), leg="f2_put")
        self._rec = {
            "f1_expiry": cyc.f1_expiry, "f2_expiry": cyc.f2_expiry,
            "entry_ts": ts, "strike": self.strike, "lot_size": put.lot_size, "qty": self.qty,
            "F1_entry": F1, "F2_entry": F2, "spread_entry": F1 - F2,
            "put_entry": g["price"], "iv_entry": self.iv, "delta_entry": g["delta"],
            "put_observed_at_entry": put.close, "oi_entry": put.oi, "rehedges": 0, "hedge_turnover_shares": 0,
            "max_hedge_shares": 0,
            "pnl_spread": 0.0, "pnl_hedge_residual": 0.0, "pnl_gamma": 0.0,
            "pnl_vega": 0.0, "pnl_theta": 0.0, "pnl_mark": 0.0,
        }
        self._prev = {"ts": ts, "F1": F1, "F2": F2, "iv": self.iv, "g": g,
                      "dte": fractional_dte(ts, cyc.f2_expiry, self.config.interval),
                      "hedge": 0}
        self._rebalance(ts, F1, g, force=True)
        logger.info(f"{self.name}[{self.config.band}]: enter {cyc.f1_expiry} @ {ts:%m-%d %H:%M} "
                    f"K={self.strike} put={g['price']:.2f} d={g['delta']:.2f} "
                    f"spread={F1 - F2:.2f}")

    def _exit(self, snapshot: MarketSnapshot, g: dict, F1: float, F2: float) -> None:
        ts = snapshot.timestamp
        self._signal(ts, self.put_key, Side.SELL, self.qty, max(g["price"], 0.05), "exit",
                     leg="f2_put")
        if self.hedge_shares > 0:
            self._signal(ts, self.hedge_key, Side.SELL, self.hedge_shares, F1, "exit",
                         role="close_hedge")
        self._rec.update({
            "exit_ts": ts, "F1_exit": F1, "F2_exit": F2, "spread_exit": F1 - F2,
            "put_exit": g["price"], "iv_exit": self.iv, "delta_exit": g["delta"],
        })
        self._rec["hedge_turnover_shares"] += self.hedge_shares
        self.records.append(self._rec)
        self._done.add(self.cycle.f1_expiry)
        logger.info(f"{self.name}[{self.config.band}]: exit {self.cycle.f1_expiry} @ {ts:%m-%d %H:%M} "
                    f"put={g['price']:.2f} spread={F1 - F2:.2f} rehedges={self._rec['rehedges']}")
        self.cycle, self.put_key, self.hedge_shares, self._rec, self._prev = None, None, 0, None, None

    # ── Delta hedge ───────────────────────────────────────────────────────────

    def _rebalance(self, ts, F1: float, g: dict, force: bool = False) -> None:
        net = g["delta"] * self.qty + self.hedge_shares        # shares of delta
        if not force and not self.config.rehedge:
            return
        if not force and abs(net) <= self.config.band * self.qty:
            return
        target = max(0, round(-g["delta"] * self.qty))         # put delta <= 0 -> long F1
        adj = target - self.hedge_shares
        if adj == 0:
            return
        self._signal(ts, self.hedge_key, Side.BUY if adj > 0 else Side.SELL, abs(adj), F1,
                     "entry" if force else "adjustment", role="delta_hedge",
                     net_delta_shares=round(net, 1))
        self.hedge_shares = target
        self._prev["hedge"] = target
        if not force:
            self._rec["rehedges"] += 1
        self._rec["hedge_turnover_shares"] += abs(adj)
        self._rec["max_hedge_shares"] = max(self._rec["max_hedge_shares"], target)

    # ── P&L attribution (gross, mark-to-model) ────────────────────────────────

    def _attribute(self, ts, F1: float, F2: float, g: dict) -> None:
        p, q = self._prev, self.qty
        if p is None or ts <= p["ts"]:
            return
        dF1, dF2 = F1 - p["F1"], F2 - p["F2"]
        dte = fractional_dte(ts, self.cycle.f2_expiry, self.config.interval)
        tau_td = max(p["dte"] - dte, 0.0)                       # trading days elapsed
        pg, h = p["g"], p["hedge"]
        r = self._rec
        r["pnl_spread"] += -pg["delta"] * q * (dF1 - dF2)       # |delta| x d(F1-F2)
        r["pnl_hedge_residual"] += (h + pg["delta"] * q) * dF1  # band slack x dF1
        r["pnl_gamma"] += 0.5 * pg["gamma"] * q * dF2 ** 2
        r["pnl_vega"] += pg["vega"] * q * (self.iv - p["iv"]) * 100.0
        r["pnl_theta"] += pg["theta"] * q * tau_td * 365.0 / 252.0
        r["pnl_mark"] += (g["price"] - pg["price"]) * q + h * dF1
        self.paths.append({
            "f1_expiry": self.cycle.f1_expiry, "ts": ts, "F1": F1, "F2": F2,
            "spread": F1 - F2, "put": g["price"], "iv": self.iv, "delta": g["delta"],
            "hedge_shares": h, "net_delta_shares": g["delta"] * q + h,
            "cum_pnl": r["pnl_mark"],
        })
        self._prev = {"ts": ts, "F1": F1, "F2": F2, "iv": self.iv, "g": g, "dte": dte,
                      "hedge": self.hedge_shares}

    def on_fill(self, event: FillEvent) -> None:
        pass  # state is tracked when signals are emitted (as in SBICardLongStrangle)
