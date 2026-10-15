"""run_b12_long_short_expiry.py — long b12 into t−1, then flip short into expiry day.

Leg 1 (long spread = long b12): BUY F1 / SELL F2 at the t−4 open; exit on t−1 by the chosen
rule (see run_b12_long_expiry.py), else at t−1 15:15.
Leg 2 (short spread = short b12): at the SAME fill as the long exit, SELL F1 / BUY F2; exit
on expiry day at ``--short-exit`` (default 14:45, filled at that bar's open, or the next bar
with both legs priced).

Costs are charged per leg per side on each trade (the flip pays two round trips).
P&L in ₹ per share and in bps of the leg's F1 entry price; the combined cycle P&L is in bps of
the long leg's F1 entry price.

Outputs: results/backtests/b12_long_short_expiry/{trades.csv, summary.csv, short_exit_sensitivity.csv}
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

from backtests.run_b12_long_expiry import COSTS_BPS, Cycle, implied_b12, lot_sizes, ticker_frame  # noqa: E402
from data.storage.arctic_store import ArcticStore  # noqa: E402
from scripts.plot_htb_cycles import select_htb_expiries  # noqa: E402

OUT = ROOT / "results" / "backtests" / "b12_long_short_expiry"
LONG_RULES = {"gap x=3 (gap>0, t-1)": ("fall", 3, "gap>0", "t1"), "hold to t-1 15:15": None}
SENS_TIMES = ("09:15", "10:30", "12:00", "13:30", "14:45", "15:15")


def fill_upto(c: Cycle, i: int, last: int):
    """Open of bar i (or the next bar with both legs priced, up to *last*); else the last
    priced close at or before *last*."""
    x = c.s
    for j in range(i, last + 1):
        if x.F1o.iat[j] > 0 and x.F2o.iat[j] > 0:
            return j, x.F1o.iat[j], x.F2o.iat[j]
    k = last
    while k > i and not (x.F1c.iat[k] > 0 and x.F2c.iat[k] > 0):
        k -= 1
    return k, x.F1c.iat[k], x.F2c.iat[k]


def short_leg(c: Cycle, p_in: int, f1e: float, f2e: float, at: str) -> dict | None:
    x = c.s
    if c.t0 is None:
        return None
    t0 = np.flatnonzero((x.day == c.t0).to_numpy())
    if not len(t0):
        return None
    hit = [p for p in t0 if x.hm.iat[p] == at]
    le = [p for p in t0 if x.hm.iat[p] <= at]
    p_at = hit[0] if hit else (le[-1] if le else t0[0])
    px, f1x, f2x = fill_upto(c, p_at, int(t0[-1]))
    s_e, s_x = f1e - f2e, f1x - f2x
    gross = s_e - s_x                                       # short spread profits when F1−F2 falls
    out = dict(short_entry_ts=x.timestamp.iat[p_in], short_exit_ts=x.timestamp.iat[px],
               short_F1_exit=f1x, short_F2_exit=f2x, short_spread_exit=s_x,
               short_b12_exit=implied_b12(f1x, f2x, x.d1.iat[px], x.d2.iat[px]),
               short_pnl_per_share=gross, short_pnl_bps=gross / f1e * 1e4,
               short_hold_h=(px - p_in) * 5 / 60)
    for cst in COSTS_BPS[1:]:
        cost = cst / 1e4 * (f1e + f2e + f1x + f2x)
        out[f"short_net{cst}_bps"] = (gross - cost) / f1e * 1e4
        out[f"short_net{cst}_per_share"] = gross - cost
    return out


def run_cycle(c: Cycle, rule_name: str, rule, short_at: str) -> dict | None:
    pos = None if rule is None else c.trigger(*rule)
    long = c.trade(pos, rule_name)
    x = c.s
    p_out = int(np.flatnonzero((x.timestamp == long["exit_ts"]).to_numpy())[0])
    sh = short_leg(c, p_out, long["F1_exit"], long["F2_exit"], short_at)
    if sh is None:
        return None
    r = {**long, **sh}
    f1_long = long["F1_entry"]
    r["combined_per_share"] = long["pnl_per_share"] + sh["short_pnl_per_share"]
    r["combined_bps"] = r["combined_per_share"] / f1_long * 1e4
    for cst in COSTS_BPS[1:]:
        r[f"combined_net{cst}_bps"] = (long[f"net{cst}_per_share"] + sh[f"short_net{cst}_per_share"]) / f1_long * 1e4
    if c.lot:
        r["combined_net10_per_lot"] = (long["net10_per_share"] + sh["short_net10_per_share"]) * c.lot
    return r


def stats(df: pd.DataFrame, col: str) -> dict:
    s = df[col].dropna()
    return dict(n=len(s), mean=s.mean(), median=s.median(), win=(s > 0).mean(), worst=s.min(), best=s.max(),
                sharpe=s.mean() / s.std() if len(s) > 1 and s.std() > 0 else np.nan)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("tickers", nargs="+")
    ap.add_argument("--threshold", type=float, default=0.30)
    ap.add_argument("--short-exit", default="14:45")
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    store = ArcticStore()

    rows, sens = [], []
    for t in (x.upper() for x in args.tickers):
        d = ticker_frame(store, t)
        lots = lot_sizes(store, t)
        known = sorted(lots)
        htb = set(select_htb_expiries(d.exp, d.td, d.b12s, args.threshold, 4).expiry)
        for E in sorted({e for e in d.exp.unique() if len(e) == 10 and e[4] == "-"}):
            lot = lots.get(E) or (lots[max([k for k in known if k <= E], default=known[0])] if known else None)
            c = Cycle(d, t, E, lot)
            if not c.ok:
                continue
            for name, rule in LONG_RULES.items():
                r = run_cycle(c, name, rule, args.short_exit)
                if r is not None:
                    rows.append({**r, "htb": E in htb})
                    if name == "hold to t-1 15:15":
                        for at in SENS_TIMES:
                            rr = run_cycle(c, name, rule, at)
                            if rr is not None:
                                sens.append(dict(ticker=t, expiry=E, htb=E in htb, short_exit=at,
                                                 short_net10_bps=rr["short_net10_bps"], short_pnl_bps=rr["short_pnl_bps"]))
        print(f"{t}: done", flush=True)
    T = pd.DataFrame(rows)
    T.to_csv(args.out / "trades.csv", index=False)
    pd.DataFrame(sens).to_csv(args.out / "short_exit_sensitivity.csv", index=False)

    summ = []
    for scope, m in (("HTB, IREDA+RVNL", T.htb & T.ticker.isin(["IREDA", "RVNL"])),
                     ("HTB, other tickers", T.htb & ~T.ticker.isin(["IREDA", "RVNL"])),
                     ("HTB, all tickers", T.htb), ("ALL cycles (no hindsight filter)", T.htb | ~T.htb)):
        for rule in LONG_RULES:
            g = T[m & (T.rule == rule)]
            for leg, col in (("long", "net10_bps"), ("short", "short_net10_bps"), ("combined", "combined_net10_bps"),
                             ("long gross", "pnl_bps"), ("short gross", "short_pnl_bps"), ("combined gross", "combined_bps")):
                summ.append(dict(scope=scope, long_exit=rule, leg=leg, **stats(g, col)))
    S = pd.DataFrame(summ)
    S.to_csv(args.out / "summary.csv", index=False)
    pd.set_option("display.width", 220)
    print(S[S.leg.isin(["long", "short", "combined"])].round(1).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
