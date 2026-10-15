"""crossings_last10.py — how many times does the fast b12 EWMA cross the slow one in the
last N trading days of each cycle (expiry day included)?

For every ticker and (fast, slow) EWMA pair (half-lives in trading days) it counts, in
the window trading-days-to-expiry 0..``--last-td`` (default 9 = last 10 days):

    up       fast moves from <= slow to > slow
    down     fast moves from >= slow to < slow
    lasting  the new state then holds for at least ``--lasting-minutes`` (default 60);
             the rest are whipsaws that flip back sooner

per cycle and per day-to-expiry, then summarises per ticker: total, per cycle (mean /
median / min / max), per trading day, share lasting, crossings on expiry day, cycles
with no crossing at all.

Only COMPLETE cycles are used: the data reaches expiry day, and the cycle starts at least
2 trading days before the window (the EWMAs run from each cycle's start, so they have
history when the window begins). Read-only; writes CSVs.

Usage
-----
    b12-crossings-last10 --all --interval 5minute
    b12-crossings-last10 IREDA RVNL --interval 15minute --pairs 0.1/2 0.5/5 --last-td 4
    (or: python scripts/crossings_last10.py ...)

Outputs (``--out``, default ``results/ewma_b12/crossings_last10_<interval>/``):
    per_cycle.csv     ticker x pair x cycle: up, down, lasting, on expiry day
    by_day.csv        ticker x pair x trading-days-to-expiry
    per_ticker.csv    the summary table printed below
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # noqa: BLE001
    pass

from data.storage.arctic_store import ArcticStore  # noqa: E402
from options.pricing.dte_utils import interval_to_minutes  # noqa: E402
from scripts.crossings import parse_pairs, tickers_for  # noqa: E402
from scripts.ewma_b12_signal import ewma_by_cycle, halflife_to_alpha, load  # noqa: E402


def count_ticker(df: pd.DataFrame, t: str, interval: str, pairs, args) -> tuple[list, list]:
    span = df.groupby("cycle").td_to_exp.agg(["min", "max"])
    full = span[(span["min"] == 0) & (span["max"] >= args.last_td + 2)].index   # complete cycles only
    lasting_bars = max(1, int(round(args.lasting_minutes / interval_to_minutes(interval))))
    cyc_rows, day_frames = [], []
    for f, sl in pairs:
        pair = f"{f:g}/{sl:g}"
        div = ewma_by_cycle(df.b12, df.cycle, halflife_to_alpha(f, interval)[0]) - \
            ewma_by_cycle(df.b12, df.cycle, halflife_to_alpha(sl, interval)[0])
        state = np.sign(div).fillna(0).astype(int)
        prev = state.groupby(df.cycle).shift(1)
        up = (prev <= 0) & (state > 0) & prev.notna()
        down = (prev >= 0) & (state < 0) & prev.notna()
        new_run = (state != prev) | prev.isna()
        run_len = df.groupby(new_run.cumsum()).cycle.transform("size")      # length of the run starting here
        lasting = run_len >= lasting_bars
        win = (df.td_to_exp <= args.last_td) & df.cycle.isin(full)
        d = pd.DataFrame({"cycle": df.cycle, "td": df.td_to_exp, "up": up & win, "down": down & win,
                          "up_lasting": up & win & lasting, "down_lasting": down & win & lasting})[win]
        for c, g in d.groupby("cycle"):
            cyc_rows.append({"ticker": t, "pair": pair, "cycle": c,
                             "up": int(g.up.sum()), "down": int(g.down.sum()),
                             "up_lasting": int(g.up_lasting.sum()), "down_lasting": int(g.down_lasting.sum()),
                             "on_expiry_day": int((g.up | g.down)[g.td == 0].sum())})
        by_day = d.groupby("td")[["up", "down", "up_lasting", "down_lasting"]].sum()
        by_day["cycles"] = d.groupby("td").cycle.nunique()
        day_frames.append(by_day.reset_index().assign(ticker=t, pair=pair))
    return cyc_rows, day_frames


def per_ticker_table(c: pd.DataFrame, n_days: int) -> pd.DataFrame:
    T = c.groupby("ticker").agg(cycles=("cycle", "nunique"), up=("up", "sum"), down=("down", "sum"),
                                total=("total", "sum"), lasting=("lasting", "sum"),
                                per_cycle_mean=("total", "mean"), per_cycle_median=("total", "median"),
                                per_cycle_min=("total", "min"), per_cycle_max=("total", "max"),
                                lasting_per_cycle=("lasting", "mean"), on_expiry_day=("on_expiry_day", "sum"),
                                cycles_zero=("total", lambda x: int((x == 0).sum())))
    tot = pd.DataFrame([{"cycles": c.cycle.nunique(), "up": c.up.sum(), "down": c.down.sum(), "total": c.total.sum(),
                         "lasting": c.lasting.sum(), "per_cycle_mean": c.total.mean(),
                         "per_cycle_median": c.total.median(), "per_cycle_min": c.total.min(),
                         "per_cycle_max": c.total.max(), "lasting_per_cycle": c.lasting.mean(),
                         "on_expiry_day": c.on_expiry_day.sum(), "cycles_zero": int((c.total == 0).sum())}], index=["ALL"])
    T = pd.concat([T, tot])
    T["per_day"] = T.total / (T.cycles * n_days)
    T["share_lasting"] = T.lasting / T.total.replace(0, np.nan)
    return T


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("tickers", nargs="*", help="Tickers (or use --all).")
    ap.add_argument("--all", action="store_true", help="Every ticker with a borrow series at --interval.")
    ap.add_argument("--interval", default="5minute")
    ap.add_argument("--from", dest="from_date", default="2025-01-01", metavar="YYYY-MM-DD",
                    help="Start date (default 2025-01-01; the NSE holiday calendar covers 2025-2026).")
    ap.add_argument("--to", dest="to_date", default=None, metavar="YYYY-MM-DD")
    ap.add_argument("--pairs", nargs="+", default=["0.1/2", "0.1/10", "0.5/5"],
                    help="fast/slow half-lives in trading days (default 0.1/2 0.1/10 0.5/5).")
    ap.add_argument("--last-td", type=int, default=9,
                    help="Window = trading days to expiry 0..LAST_TD (default 9 = last 10 days).")
    ap.add_argument("--lasting-minutes", type=float, default=60,
                    help="A crossing is 'lasting' if the new state holds this long (default 60).")
    ap.add_argument("--out", default=None, help="Output folder (default results/ewma_b12/crossings_last10_<interval>).")
    args = ap.parse_args(argv)

    store = ArcticStore()
    tickers = tickers_for(store, args.interval) if args.all else [t.upper() for t in args.tickers]
    if not tickers:
        ap.error("give tickers or --all")
    pairs = parse_pairs(args.pairs)
    out = Path(args.out) if args.out else ROOT / "results" / "ewma_b12" / f"crossings_last10_{args.interval}"
    out.mkdir(parents=True, exist_ok=True)

    cyc_rows, day_frames = [], []
    for t in tickers:
        df = load(store, t, args.interval, args.from_date, args.to_date)
        if df.empty or df.td_to_exp.isna().all():
            print(f"{t}: no data — skipped")
            continue
        c, d = count_ticker(df, t, args.interval, pairs, args)
        cyc_rows += c
        day_frames += d
    C = pd.DataFrame(cyc_rows)
    if C.empty:
        print("no complete cycles in range")
        return 1
    D = pd.concat(day_frames)
    C["total"] = C.up + C.down
    C["lasting"] = C.up_lasting + C.down_lasting

    pd.set_option("display.width", 260); pd.set_option("display.max_rows", 400)
    tables = []
    n_days = args.last_td + 1
    for pair in [f"{f:g}/{s:g}" for f, s in pairs]:
        T = per_ticker_table(C[C.pair == pair], n_days)
        tables.append(T.assign(pair=pair))
        print(f"\n=== fast/slow {pair} days — crossings in the last {n_days} trading days "
              f"(complete cycles, {args.from_date}+) ===")
        print(T.round(2).to_string())
        dd = D[D.pair == pair].groupby("td")[["up", "down", "up_lasting", "down_lasting", "cycles"]].sum()
        dd["per_cycle"] = (dd.up + dd.down) / dd.cycles
        dd["lasting_per_cycle"] = (dd.up_lasting + dd.down_lasting) / dd.cycles
        print("by trading days to expiry (all tickers):\n", dd.round(2).to_string())

    C.to_csv(out / "per_cycle.csv", index=False)
    D.to_csv(out / "by_day.csv", index=False)
    pd.concat(tables).rename_axis("ticker").reset_index().to_csv(out / "per_ticker.csv", index=False)
    print(f"\nWrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
