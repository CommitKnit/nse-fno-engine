"""Timed long ATM straddle in the near-month (F1) options around expiry.

Buys one lot each of the ATM call and put on the F1 chain at a scheduled entry
bar and sells both at a scheduled exit bar. There is no delta hedge. Entry and
exit timestamps come from the harness (calendar-day offsets from expiry), so the
strategy contains no signal logic — it is a fixed-schedule test.

ATM = the strike nearest the F1 future at entry among strikes where BOTH the call
and the put carry an IV. Each leg is marked by Black-76 on the F1 future at its
strike's last observed IV and the current fractional DTE (sticky strike), so thin
5-minute chains still give a mark every bar; fills use the same mark.

Units: quantities are SHARES (lots x lot_size); run the engine with ``lot_size=1``.
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
class StraddleWindow:
    expiry: date
    entry_ts: datetime
    exit_ts: datetime


@dataclass
class ExpiryStraddleConfig:
    windows: list[StraddleWindow]
    lots: int = 1
    interval: str = "5minute"
    risk_free_rate: float = field(default_factory=lambda: settings.RISK_FREE_RATE)


class ExpiryStraddleStrategy(BaseStrategy):
    """Long F1 ATM straddle over fixed (entry, exit) windows; no hedge."""

    def __init__(self, config: ExpiryStraddleConfig, name: str = "ExpiryStraddle"):
        super().__init__(name=name)
        self.config = config
        self._windows = sorted(config.windows, key=lambda w: w.entry_ts)
        self._done: set[date] = set()
        self.win: StraddleWindow | None = None
        self.legs: dict[str, dict] = {}       # "CE"/"PE" -> {key, iv}
        self.strike = 0.0
        self.qty = 0
        self._prev: dict | None = None
        self._rec: dict | None = None
        self.records: list[dict] = []

    # ── pricing ───────────────────────────────────────────────────────────────

    def _leg_greeks(self, F: float, T: float, iv: float, is_call: bool) -> dict:
        K, r = self.strike, self.config.risk_free_rate
        return {"price": black76_price(F, K, T, iv, r, is_call),
                "delta": black76_delta(F, K, T, iv, r, is_call),
                "gamma": black76_gamma(F, K, T, iv, r),
                "vega": black76_vega(F, K, T, iv, r),
                "theta": black76_theta(F, K, T, iv, r, is_call)}

    def _book(self, F: float, ts: datetime) -> dict:
        T = year_fraction(fractional_dte(ts, self.win.expiry, self.config.interval))
        return {lt: self._leg_greeks(F, T, self.legs[lt]["iv"], lt == "CE") for lt in ("CE", "PE")}

    # ── core hook ─────────────────────────────────────────────────────────────

    def on_market(self, snapshot: MarketSnapshot) -> None:
        ts = snapshot.timestamp
        if self.win is None:
            for w in self._windows:
                if w.expiry not in self._done and w.entry_ts <= ts < w.exit_ts:
                    self._enter(snapshot, w)
                    break
            return
        chain = snapshot.chains.get(self.win.expiry.isoformat())
        if chain is None:
            return
        F = chain.futures_price
        for lt, leg in self.legs.items():
            c = next((x for x in chain.contracts if x.instrument_key == leg["key"]), None)
            if c is not None and c.iv:
                leg["iv"] = c.iv
        book = self._book(F, ts)
        self._attribute(ts, F, book)
        if ts >= self.win.exit_ts:
            self._exit(ts, F, book)

    def _enter(self, snapshot: MarketSnapshot, w: StraddleWindow) -> None:
        chain = snapshot.chains.get(w.expiry.isoformat())
        if chain is None:
            return
        F, ts = chain.futures_price, snapshot.timestamp
        calls = {c.strike: c for c in chain.calls if c.iv}
        puts = {p.strike: p for p in chain.puts if p.iv}
        strikes = sorted(set(calls) & set(puts), key=lambda k: abs(k - F))
        if not strikes:
            return
        K = strikes[0]
        self.win, self.strike = w, K
        self.legs = {"CE": {"key": calls[K].instrument_key, "iv": calls[K].iv},
                     "PE": {"key": puts[K].instrument_key, "iv": puts[K].iv}}
        self.qty = self.config.lots * calls[K].lot_size
        book = self._book(F, ts)
        for lt, ot in (("CE", OptionType.CE), ("PE", OptionType.PE)):
            self.add_signal(SignalEvent(
                timestamp=ts, instrument_key=self.legs[lt]["key"], side=Side.BUY,
                quantity=self.qty, option_type=ot, strike=K, expiry=w.expiry.isoformat(),
                signal_type="entry", target_price=book[lt]["price"],
                metadata={"strategy": self.name, "leg": lt}))
        self._rec = {
            "expiry": w.expiry, "entry_ts": ts, "strike": K, "lot_size": calls[K].lot_size,
            "qty": self.qty, "F_entry": F,
            "ce_entry": book["CE"]["price"], "pe_entry": book["PE"]["price"],
            "ce_iv_entry": self.legs["CE"]["iv"], "pe_iv_entry": self.legs["PE"]["iv"],
            "ce_oi_entry": calls[K].oi, "pe_oi_entry": puts[K].oi,
            "delta_entry": book["CE"]["delta"] + book["PE"]["delta"],
            "pnl_delta": 0.0, "pnl_gamma": 0.0, "pnl_vega": 0.0, "pnl_theta": 0.0, "pnl_mark": 0.0,
        }
        self._prev = {"ts": ts, "F": F, "book": book,
                      "iv": {lt: self.legs[lt]["iv"] for lt in self.legs},
                      "dte": fractional_dte(ts, w.expiry, self.config.interval)}
        logger.info(f"{self.name}: enter {w.expiry} @ {ts:%m-%d %H:%M} K={K} F={F:.2f} "
                    f"straddle={book['CE']['price'] + book['PE']['price']:.2f}")

    def _exit(self, ts, F: float, book: dict) -> None:
        for lt in ("CE", "PE"):
            self.add_signal(SignalEvent(
                timestamp=ts, instrument_key=self.legs[lt]["key"], side=Side.SELL,
                quantity=self.qty, signal_type="exit",
                target_price=max(book[lt]["price"], 0.05),
                metadata={"strategy": self.name, "leg": lt}))
        self._rec.update({
            "exit_ts": ts, "F_exit": F,
            "ce_exit": book["CE"]["price"], "pe_exit": book["PE"]["price"],
            "ce_iv_exit": self.legs["CE"]["iv"], "pe_iv_exit": self.legs["PE"]["iv"],
            "delta_exit": book["CE"]["delta"] + book["PE"]["delta"],
        })
        self.records.append(self._rec)
        self._done.add(self.win.expiry)
        logger.info(f"{self.name}: exit {self.win.expiry} @ {ts:%m-%d %H:%M} F={F:.2f} "
                    f"straddle={book['CE']['price'] + book['PE']['price']:.2f}")
        self.win, self.legs, self._rec, self._prev = None, {}, None, None

    # ── P&L attribution (per bar, previous-bar Greeks) ────────────────────────

    def _attribute(self, ts, F: float, book: dict) -> None:
        p, q = self._prev, self.qty
        if p is None or ts <= p["ts"]:
            return
        dF = F - p["F"]
        dte = fractional_dte(ts, self.win.expiry, self.config.interval)
        tau = max(p["dte"] - dte, 0.0) * 365.0 / 252.0          # calendar-day equivalent
        r = self._rec
        for lt in ("CE", "PE"):
            g = p["book"][lt]
            r["pnl_delta"] += g["delta"] * q * dF
            r["pnl_gamma"] += 0.5 * g["gamma"] * q * dF ** 2
            r["pnl_vega"] += g["vega"] * q * (self.legs[lt]["iv"] - p["iv"][lt]) * 100.0
            r["pnl_theta"] += g["theta"] * q * tau
            r["pnl_mark"] += (book[lt]["price"] - g["price"]) * q
        self._prev = {"ts": ts, "F": F, "book": book,
                      "iv": {lt: self.legs[lt]["iv"] for lt in self.legs}, "dte": dte}

    def on_fill(self, event: FillEvent) -> None:
        pass
