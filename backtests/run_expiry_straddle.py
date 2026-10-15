"""run_expiry_straddle.py — long F1 ATM straddle into expiry, on the engine (any symbol).

Buy 1 lot ATM call + put on the F1 chain at the 15:00 bar of expiry-N (calendar
days, weekends counted; a non-trading day rolls back to the previous trading day)
and sell both at the last bar (close) of expiry-M. Every (N, M) in --entries x
--exits with exit after entry, x {gross, net of engine-default costs}. No hedge.

    backtest-expiry-straddle                                  # IREDA 5minute
    backtest-expiry-straddle --symbol SBICARD --interval 15minute --entries 6,5 --exits 0,1,4
    (or: python backtests/run_expiry_straddle.py ...)

Outputs: results/{symbol}_expiry_straddle/{cycles,summary,schedule}.csv
"""

from __future__ import annotations

import argparse
import sys
from datetime import date, time, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.engine.backtest_engine import BacktestEngine
from core.orders.execution_simulator import CommissionModel
from data.fetchers.snapshot_builder import LocalSnapshotBuilder
from options.pricing.dte_utils import is_trading_day
from options.strategies.expiry_straddle import (
    ExpiryStraddleConfig,
    ExpiryStraddleStrategy,
    StraddleWindow,
)

DEFAULT_SYMBOL, DEFAULT_INTERVAL = "IREDA", "5minute"
ENTRY_TIME = time(15, 0)
RESULTS = ROOT / "results"
ZERO_COST = CommissionModel(brokerage_per_order=0.0, stt_rate=0.0, exchange_rate=0.0,
                            gst_rate=0.0, stamp_duty_rate=0.0)


def on_or_before_trading_day(d: date) -> date:
    while not is_trading_day(d):
        d -= timedelta(days=1)
    return d


def schedule(expiries: list[date], bar_ts: pd.DatetimeIndex, entries, exits) -> pd.DataFrame:
    """One row per (expiry, N, M): entry = 15:00 bar of E-N, exit = last bar of E-M."""
    by_day = pd.Series(bar_ts, index=bar_ts.date).groupby(level=0)
    first_le = {d: g for d, g in by_day}
    rows = []
    for e in expiries:
        for n in entries:
            for m in exits:
                de = on_or_before_trading_day(e - timedelta(days=n))
                dx = on_or_before_trading_day(e - timedelta(days=m))
                eb, xb = first_le.get(de), first_le.get(dx)
                ent = eb[eb.dt.time <= ENTRY_TIME].max() if eb is not None else pd.NaT
                ext = xb.max() if xb is not None else pd.NaT
                ok = pd.notna(ent) and pd.notna(ext) and ext > ent
                rows.append({"expiry": e, "weekday": e.strftime("%a"), "entry_n": n, "exit_m": m,
                             "entry_day": de, "exit_day": dx, "entry_ts": ent, "exit_ts": ext,
                             "hold_h": (ext - ent).total_seconds() / 3600 if ok else np.nan,
                             "valid": ok})
    return pd.DataFrame(rows)


def iv_source(store, symbol: str, interval: str, rec: pd.DataFrame) -> pd.DataFrame:
    """Tag each fill's IV source: the leg's last enriched row at/before the fill."""
    out = []
    for r in rec.itertuples():
        o = store.read_options(symbol, interval, str(r.expiry)).sort_index()
        o = o[o["strike_price"] == r.strike]
        tag = {"expiry": r.expiry, "entry_n": r.entry_n, "exit_m": r.exit_m}
        for lt in ("CE", "PE"):
            ol = o[o["instrument_type"] == lt]
            for leg, ts in (("entry", r.entry_ts), ("exit", r.exit_ts)):
                q = ol[ol.index <= ts]
                if q.empty:
                    tag[f"{lt}_{leg}_src"] = "none"
                    continue
                last = q.iloc[-1]
                tag[f"{lt}_{leg}_src"] = ("pcp" if last["synthetic_via_putcallparity"]
                                          else "ffill" if last["is_synthetic"] else "traded")
                tag[f"{lt}_{leg}_age_min"] = (ts - q.index[-1]).total_seconds() / 60
        out.append(tag)
    return pd.DataFrame(out)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--symbol", default=DEFAULT_SYMBOL, help=f"Underlying (default {DEFAULT_SYMBOL}).")
    ap.add_argument("--interval", default=DEFAULT_INTERVAL,
                    help=f"Futures + options interval (default {DEFAULT_INTERVAL}).")
    ap.add_argument("--from", dest="from_d", default="2025-04-24")
    ap.add_argument("--to", dest="to_d", default="2026-06-30")
    ap.add_argument("--entries", default="6,5", help="Calendar days before expiry to enter.")
    ap.add_argument("--exits", default="0,1,4", help="Calendar days before expiry to exit.")
    ap.add_argument("--lots", type=int, default=1)
    ap.add_argument("--out", default=None,
                    help="Output folder (default results/{symbol}_expiry_straddle).")
    args = ap.parse_args(argv)
    symbol, interval = args.symbol.upper(), args.interval
    entries = [int(x) for x in args.entries.split(",")]
    exits = [int(x) for x in args.exits.split(",")]
    out = Path(args.out) if args.out else RESULTS / f"{symbol.lower()}_expiry_straddle"
    out.mkdir(parents=True, exist_ok=True)

    logger.remove()
    logger.add(sys.stderr, level="WARNING", filter=lambda r: not r["name"].startswith("core."))

    builder = LocalSnapshotBuilder(symbol=symbol, futures_interval=interval, options_interval=interval,
                                   contract_ffill_days=5)
    store = builder._store
    lo, hi = date.fromisoformat(args.from_d), date.fromisoformat(args.to_d)
    expiries = [date.fromisoformat(e) for e in store.list_option_expiries(symbol, interval)
                if lo <= date.fromisoformat(e) <= hi]

    snaps = []
    for e in expiries:
        s = builder.build((e - timedelta(days=max(entries) + 4)).isoformat(), e.isoformat())
        snaps.extend(s)
    snaps.sort(key=lambda s: s.timestamp)
    bar_ts = pd.DatetimeIndex([s.timestamp for s in snaps])
    sch = schedule(expiries, bar_ts, entries, exits)
    sch.to_csv(out / "schedule.csv", index=False)
    print(f"{len(expiries)} expiries, {len(snaps)} snapshots; "
          f"{int(sch.valid.sum())}/{len(sch)} (expiry, entry, exit) windows valid")

    frames, summary = [], []
    for n in entries:
        for m in exits:
            sub = sch[(sch.entry_n == n) & (sch.exit_m == m) & sch.valid]
            if sub.empty:
                continue
            wins = [StraddleWindow(r.expiry, r.entry_ts.to_pydatetime(), r.exit_ts.to_pydatetime())
                    for r in sub.itertuples()]
            for costs in (False, True):
                strat = ExpiryStraddleStrategy(ExpiryStraddleConfig(windows=wins, lots=args.lots,
                                                                    interval=interval))
                eng = BacktestEngine(snaps, strat, initial_capital=100_000_000, max_drawdown_pct=10.0,
                                     commission_model=None if costs else ZERO_COST,
                                     slippage_pct=0.001 if costs else 0.0,
                                     max_position_lots=10**9, max_open_positions=10, lot_size=1)
                res = eng.run()
                trades = pd.DataFrame(res["closed_trades"])
                rec = pd.DataFrame(strat.records)
                pnl = []
                for r in rec.itertuples():
                    t = trades[(trades.closed_at >= r.entry_ts) & (trades.closed_at <= r.exit_ts)]
                    pnl.append((t.realized_pnl.sum(), t.commission.sum()))
                rec["engine_realized"] = [p[0] for p in pnl]
                rec["engine_commission"] = [p[1] for p in pnl]
                rec["engine_net"] = rec.engine_realized - rec.engine_commission
                rec.insert(0, "costs", "net" if costs else "gross")
                rec.insert(0, "exit_m", m)
                rec.insert(0, "entry_n", n)
                frames.append(rec)
                summary.append({"entry_n": n, "exit_m": m, "costs": "net" if costs else "gross",
                                **res["summary"], "fills": res["execution_stats"]["fills"],
                                "rejected": res["risk_stats"]["rejected"]})

    cyc = pd.concat(frames, ignore_index=True)
    tags = iv_source(store, symbol, interval, cyc[cyc.costs == "gross"])
    cyc = cyc.merge(tags, on=["expiry", "entry_n", "exit_m"], how="left")
    cyc.to_csv(out / "cycles.csv", index=False)
    pd.DataFrame(summary).to_csv(out / "summary.csv", index=False)

    pd.set_option("display.width", 220)
    view = cyc.pivot_table(index="expiry", columns=["entry_n", "exit_m", "costs"], values="engine_net")
    print("\nEngine net P&L per cycle (INR):\n", view.round(0).to_string())
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
