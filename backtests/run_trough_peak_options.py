"""run_trough_peak_options.py — unhedged long F2 calls / puts over the roll window (any symbol).

Same LOOK-AHEAD windows as run_roll_put_hedge.py: in each final expiry week,
entry at the F1-F2 spread trough before the week's peak, exit at the peak. Two
strategies, each 1 lot per leg on the F2 chain (or F1 with --chain F1), no hedge:

    calls   ATM call + first liquid OTM call above it (OI >= --min-oi)
    puts    ATM put  + first liquid OTM put below it  (OI >= --min-oi)

x {gross, net of engine-default costs}.

--windows calendar swaps the look-ahead windows for the long-calendar rule's own
entry and exit times (results/backtests/b12_long_short_final/trades.csv): enter t-4
from 09:30, exit on the b12 stop, the t-1 gap signal or t-1 15:15. Only cycles whose
smoothed b12 reached --min-b12 % before entry are traded.

    backtest-trough-peak                                      # IREDA 5minute, F2 chain
    backtest-trough-peak --symbol SBICARD --interval 15minute --chain F1
    backtest-trough-peak --windows calendar --chain F1 --from 2025-01-01 --to 2026-12-31
    (or: python backtests/run_trough_peak_options.py ...)
Outputs: results/{symbol}_{trough_peak|calendar}_options[_f1]/{legs,cycles,summary,skipped,windows}.csv
"""

from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.engine.backtest_engine import BacktestEngine
from data.fetchers.snapshot_builder import LocalSnapshotBuilder
from options.strategies.timed_long_options import (
    TimedLongOptionsConfig,
    TimedLongOptionsStrategy,
    TradeWindow,
)
from backtests.run_roll_put_hedge import (
    DEFAULT_INTERVAL,
    DEFAULT_SYMBOL,
    RESULTS,
    ZERO_COST,
    lookahead_windows,
)

CALENDAR_TRADES = ROOT / "results" / "backtests" / "b12_long_short_final" / "trades.csv"
STRATEGIES = {"calls": [("CE", "atm"), ("CE", "otm")], "puts": [("PE", "atm"), ("PE", "otm")]}


def calendar_windows(store, symbol: str, interval: str, from_d: str, to_d: str,
                     trades: Path, min_b12: float, chain: str = "F2") -> pd.DataFrame:
    """Per F1 expiry: entry/exit = the long calendar's own fills (t-4 09:30 -> stop / signal / t-1 15:15)."""
    T = pd.read_csv(trades)
    T = T[(T.ticker == symbol) & (T.expiry >= from_d) & (T.expiry <= to_d)]
    fut = store.read_futures(symbol, interval, with_spot=False)
    opt_exps = set(store.list_option_expiries(symbol, interval))
    rows = []
    for r in T.itertuples():
        entry, exit_ = pd.Timestamp(r.entry_ts), pd.Timestamp(r.long_exit_ts)
        e2 = str(fut.loc[fut.index >= entry, "F2_expiry"].iloc[0])[:10]
        skip = ("" if r.b12_max_to_entry >= min_b12 else f"b12 max before entry {r.b12_max_to_entry:.1f}% < {min_b12:g}%")
        if not skip and (e2 if chain == "F2" else r.expiry) not in opt_exps:
            skip = f"no {chain} option chain in Arctic"
        rows.append({"f1_expiry": date.fromisoformat(r.expiry), "f2_expiry": date.fromisoformat(e2),
                     "week_start": entry.date(), "entry_ts": entry, "exit_ts": exit_,
                     "exit_reason": r.long_exit_reason, "b12_max_to_entry": r.b12_max_to_entry,
                     "calendar_long_net_bps": r.long_net, "skip": skip})
    return pd.DataFrame(rows)


def iv_src(store, symbol: str, interval: str, legs: pd.DataFrame) -> pd.DataFrame:
    tags = []
    for r in legs.itertuples():
        o = store.read_options(symbol, interval, str(r.chain_expiry)).sort_index()
        o = o[o["instrument_key"] == r.instrument_key]
        t = {"key": r.key, "leg": r.leg}
        for side, ts in (("entry", r.entry_ts), ("exit", r.exit_ts)):
            q = o[o.index <= ts]
            t[f"iv_src_{side}"] = ("none" if q.empty else "pcp" if q.iloc[-1]["synthetic_via_putcallparity"]
                                   else "ffill" if q.iloc[-1]["is_synthetic"] else "traded")
        tags.append(t)
    return pd.DataFrame(tags)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--symbol", default=DEFAULT_SYMBOL, help=f"Underlying (default {DEFAULT_SYMBOL}).")
    ap.add_argument("--interval", default=DEFAULT_INTERVAL,
                    help=f"Futures + options interval (default {DEFAULT_INTERVAL}).")
    ap.add_argument("--from", dest="from_d", default="2025-04-01")
    ap.add_argument("--to", dest="to_d", default="2026-06-30")
    ap.add_argument("--min-oi", type=int, default=200_000)
    ap.add_argument("--lots", type=int, default=1)
    ap.add_argument("--chain", choices=["F1", "F2"], default="F2",
                    help="Option chain to trade: F2 (next month, default) or F1 (expiring).")
    ap.add_argument("--windows", choices=["lookahead", "calendar"], default="lookahead",
                    help="lookahead: spread trough -> peak (default); calendar: long-calendar rule's entry/exit.")
    ap.add_argument("--trades", default=str(CALENDAR_TRADES),
                    help="Calendar backtest trades.csv (for --windows calendar).")
    ap.add_argument("--min-b12", type=float, default=30.0,
                    help="--windows calendar: trade only cycles whose smoothed b12 reached this %% before entry.")
    ap.add_argument("--out", default=None,
                    help="Output folder (default results/{symbol}_trough_peak_options[_f1]).")
    args = ap.parse_args(argv)
    symbol, interval = args.symbol.upper(), args.interval
    out = (Path(args.out) if args.out else
           RESULTS / (f"{symbol.lower()}_{'trough_peak' if args.windows == 'lookahead' else 'calendar'}_options"
                      + ("" if args.chain == "F2" else "_f1")))
    out.mkdir(parents=True, exist_ok=True)
    logger.remove()
    logger.add(sys.stderr, level="WARNING", filter=lambda r: not r["name"].startswith("core."))

    builder = LocalSnapshotBuilder(symbol=symbol, futures_interval=interval, options_interval=interval,
                                   contract_ffill_days=5)
    win = (lookahead_windows(builder._store, symbol, interval, args.from_d, args.to_d, 5)
           if args.windows == "lookahead" else
           calendar_windows(builder._store, symbol, interval, args.from_d, args.to_d,
                            Path(args.trades), args.min_b12, args.chain))
    win.to_csv(out / "windows.csv", index=False)
    live = win[win["skip"] == ""].reset_index(drop=True)
    snaps = []
    for w in live.itertuples():
        s = builder.build(w.week_start.isoformat(), w.f1_expiry.isoformat())
        snaps.extend(s)
    windows = [TradeWindow(w.f1_expiry, w.f2_expiry if args.chain == "F2" else w.f1_expiry,
                           w.entry_ts.to_pydatetime(), w.exit_ts.to_pydatetime())
               for w in live.itertuples()]

    legs, summary, skipped = [], [], []
    for name, spec in STRATEGIES.items():
        for costs in (False, True):
            strat = TimedLongOptionsStrategy(TimedLongOptionsConfig(windows=windows, legs=spec,
                                             min_oi=args.min_oi, lots=args.lots, interval=interval))
            res = BacktestEngine(snaps, strat, initial_capital=100_000_000, max_drawdown_pct=10.0,
                                 commission_model=None if costs else ZERO_COST,
                                 slippage_pct=0.001 if costs else 0.0,
                                 max_position_lots=10**9, max_open_positions=10, lot_size=1).run()
            trades = pd.DataFrame(res["closed_trades"])
            rec = pd.DataFrame(strat.records)
            got = [trades[(trades.instrument_key == r.instrument_key) & (trades.closed_at >= r.entry_ts)
                          & (trades.closed_at <= r.exit_ts)] for r in rec.itertuples()]
            rec["engine_realized"] = [t.realized_pnl.sum() for t in got]
            rec["engine_commission"] = [t.commission.sum() for t in got]
            rec["engine_net"] = rec.engine_realized - rec.engine_commission
            rec.insert(0, "costs", "net" if costs else "gross")
            rec.insert(0, "strategy", name)
            legs.append(rec)
            if not costs:
                skipped += [{"strategy": name, **s} for s in strat.skipped]
            summary.append({"strategy": name, "costs": "net" if costs else "gross", **res["summary"],
                            "fills": res["execution_stats"]["fills"], "rejected": res["risk_stats"]["rejected"]})

    L = pd.concat(legs, ignore_index=True)
    L = L.merge(iv_src(builder._store, symbol, interval, L[L.costs == "gross"]),
                on=["key", "leg"], how="left")
    L.to_csv(out / "legs.csv", index=False)
    C = (L.groupby(["strategy", "costs", "key"])
           .agg(net=("engine_net", "sum"), gross=("engine_realized", "sum"), legs=("leg", "count"))
           .reset_index())
    C.to_csv(out / "cycles.csv", index=False)
    pd.DataFrame(summary).to_csv(out / "summary.csv", index=False)
    pd.DataFrame(skipped).to_csv(out / "skipped.csv", index=False)
    pd.set_option("display.width", 220)
    print(C.pivot_table(index="key", columns=["strategy", "costs"], values="net").round(0).to_string())
    print(C.groupby(["strategy", "costs"]).net.agg(["sum", "mean", "std", lambda x: (x > 0).sum()]).round(0))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
