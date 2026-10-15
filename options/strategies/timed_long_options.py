"""Buy a set of option legs at a scheduled bar and sell them at a scheduled bar (no hedge).

Each window names the chain (option expiry) to trade and the entry/exit
timestamps; the harness supplies them, so the strategy holds no signal logic.

Leg selection at entry, relative to that chain's own future F:
    ("CE"|"PE", "atm")  strike nearest F among strikes with an IV
    ("CE"|"PE", "otm")  first strike BEYOND the ATM strike (above for calls, below
                        for puts) whose open interest is >= ``min_oi``; the leg is
                        skipped if none qualifies

Marks: Black-76 on the chain's future at each strike's last observed IV and the
current fractional DTE (sticky strike); fills use the same mark. P&L is attributed
per bar with previous-bar Greeks into delta / gamma / vega / theta.

Units: SHARES (lots x lot_size) — run the engine with ``lot_size=1``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime

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
class TradeWindow:
    key: date            # cycle id (e.g. F1 expiry)
    chain_expiry: date   # option expiry to trade (e.g. F2)
    entry_ts: datetime
    exit_ts: datetime


@dataclass
class TimedLongOptionsConfig:
    windows: list[TradeWindow]
    legs: list[tuple[str, str]]          # e.g. [("CE", "atm"), ("CE", "otm")]
    min_oi: int = 200_000
    lots: int = 1
    interval: str = "5minute"
    risk_free_rate: float = field(default_factory=lambda: settings.RISK_FREE_RATE)


class TimedLongOptionsStrategy(BaseStrategy):
    def __init__(self, config: TimedLongOptionsConfig, name: str = "TimedLongOptions"):
        super().__init__(name=name)
        self.config = config
        self._windows = sorted(config.windows, key=lambda w: w.entry_ts)
        self._done: set[date] = set()
        self.win: TradeWindow | None = None
        self.legs: list[dict] = []
        self.records: list[dict] = []        # one per (cycle, leg)
        self.skipped: list[dict] = []

    # ── pricing ───────────────────────────────────────────────────────────────

    def _greeks(self, leg: dict, F: float, T: float) -> dict:
        K, s, r, c = leg["strike"], leg["iv"], self.config.risk_free_rate, leg["is_call"]
        return {"price": black76_price(F, K, T, s, r, c), "delta": black76_delta(F, K, T, s, r, c),
                "gamma": black76_gamma(F, K, T, s, r), "vega": black76_vega(F, K, T, s, r),
                "theta": black76_theta(F, K, T, s, r, c)}

    def _dte(self, ts) -> float:
        return fractional_dte(ts, self.win.chain_expiry, self.config.interval)

    # ── core hook ─────────────────────────────────────────────────────────────

    def on_market(self, snapshot: MarketSnapshot) -> None:
        ts = snapshot.timestamp
        if self.win is None:
            for w in self._windows:
                if w.key not in self._done and w.entry_ts <= ts < w.exit_ts:
                    self._enter(snapshot, w)
                    break
            return
        chain = snapshot.chains.get(self.win.chain_expiry.isoformat())
        if chain is None:
            return
        F, dte = chain.futures_price, self._dte(ts)
        T = year_fraction(dte)
        by_key = {c.instrument_key: c for c in chain.contracts}
        for leg in self.legs:
            c = by_key.get(leg["key"])
            if c is not None and c.iv:
                leg["iv_new"] = c.iv
        for leg in self.legs:
            new_iv = leg.pop("iv_new", leg["iv"])
            p, q = leg["g"], leg["qty"]
            tau = max(leg["dte"] - dte, 0.0) * 365.0 / 252.0
            dF = F - leg["F"]
            leg["iv_prev"], leg["iv"] = leg["iv"], new_iv
            g = self._greeks(leg, F, T)
            r = leg["rec"]
            r["pnl_delta"] += p["delta"] * q * dF
            r["pnl_gamma"] += 0.5 * p["gamma"] * q * dF ** 2
            r["pnl_vega"] += p["vega"] * q * (new_iv - leg["iv_prev"]) * 100.0
            r["pnl_theta"] += p["theta"] * q * tau
            r["pnl_mark"] += (g["price"] - p["price"]) * q
            leg.update(g=g, F=F, dte=dte)
        if ts >= self.win.exit_ts:
            self._exit(ts, F)

    def _pick(self, chain, ot: str, mode: str, F: float):
        side = chain.calls if ot == "CE" else chain.puts
        pool = [c for c in side if c.iv]
        if not pool:
            return None
        atm = min(pool, key=lambda c: abs(c.strike - F))
        if mode == "atm":
            return atm
        beyond = [c for c in pool if (c.strike > atm.strike if ot == "CE" else c.strike < atm.strike)
                  and (c.oi or 0) >= self.config.min_oi]
        if not beyond:
            return None
        return min(beyond, key=lambda c: abs(c.strike - atm.strike))

    def _enter(self, snapshot: MarketSnapshot, w: TradeWindow) -> None:
        chain = snapshot.chains.get(w.chain_expiry.isoformat())
        if chain is None:
            return
        ts, F = snapshot.timestamp, chain.futures_price
        self.win = w
        dte = self._dte(ts)
        T = year_fraction(dte)
        legs = []
        for ot, mode in self.config.legs:
            c = self._pick(chain, ot, mode, F)
            if c is None:
                self.skipped.append({"key": w.key, "leg": f"{mode}_{ot}", "ts": ts, "F": F,
                                     "reason": "no strike" if mode == "atm" else f"no OTM strike with oi >= {self.config.min_oi:,}"})
                continue
            leg = {"name": f"{mode}_{ot}", "key": c.instrument_key, "strike": c.strike,
                   "is_call": ot == "CE", "iv": c.iv, "qty": self.config.lots * c.lot_size,
                   "F": F, "dte": dte}
            leg["g"] = self._greeks(leg, F, T)
            leg["rec"] = {"key": w.key, "chain_expiry": w.chain_expiry, "leg": leg["name"],
                          "instrument_key": c.instrument_key, "strike": c.strike, "lot_size": c.lot_size, "qty": leg["qty"],
                          "oi_entry": c.oi, "entry_ts": ts, "F_entry": F,
                          "px_entry": leg["g"]["price"], "iv_entry": c.iv, "delta_entry": leg["g"]["delta"],
                          "pnl_delta": 0.0, "pnl_gamma": 0.0, "pnl_vega": 0.0, "pnl_theta": 0.0, "pnl_mark": 0.0}
            self.add_signal(SignalEvent(
                timestamp=ts, instrument_key=c.instrument_key, side=Side.BUY, quantity=leg["qty"],
                option_type=OptionType.CE if ot == "CE" else OptionType.PE, strike=c.strike,
                expiry=w.chain_expiry.isoformat(), signal_type="entry",
                target_price=leg["g"]["price"], metadata={"strategy": self.name, "leg": leg["name"]}))
            legs.append(leg)
        if not legs:
            self._done.add(w.key)
            self.win = None
            return
        self.legs = legs
        logger.info(f"{self.name}: enter {w.key} @ {ts:%m-%d %H:%M} F={F:.2f} "
                    + ", ".join(f"{l['name']} K={l['strike']} px={l['g']['price']:.2f}" for l in legs))

    def _exit(self, ts, F: float) -> None:
        for leg in self.legs:
            self.add_signal(SignalEvent(
                timestamp=ts, instrument_key=leg["key"], side=Side.SELL, quantity=leg["qty"],
                signal_type="exit", target_price=max(leg["g"]["price"], 0.05),
                metadata={"strategy": self.name, "leg": leg["name"]}))
            leg["rec"].update(exit_ts=ts, F_exit=F, px_exit=leg["g"]["price"], iv_exit=leg["iv"],
                              delta_exit=leg["g"]["delta"])
            self.records.append(leg["rec"])
        logger.info(f"{self.name}: exit {self.win.key} @ {ts:%m-%d %H:%M} F={F:.2f}")
        self._done.add(self.win.key)
        self.win, self.legs = None, []

    def on_fill(self, event: FillEvent) -> None:
        pass
