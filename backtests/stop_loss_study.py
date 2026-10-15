"""stop_loss_study.py — can a stop cut the losers of the long-then-short b12 expiry trade early?

Legs (as in run_b12_long_short_expiry.py):
  long  : BUY F1 / SELL F2 at the t−4 open; scheduled exit = gap_0.05_1 rule (x=3, gap>0, t−1)
          else t−1 15:15
  short : SELL F1 / BUY F2 at the long's exit fill; scheduled exit = expiry day 14:45
Cycle sets: long — HTB cycles (final-week peak b12 >= 30%); short — HTB cycles, and the
ex-ante set (at the short entry: b12 >= 30% and b12 up >= 10 pts since the t−4 open).

1. Path anatomy: MAE / MFE (worst / best mark-to-market before the scheduled exit, bar closes)
   for winners vs losers, and how early the losers go underwater.
2. Stops (decided on a bar close, filled at the next bar's open, never past the scheduled exit):
     fixed    — unrealised P&L <= −X bps
     trail    — P&L falls X bps below its best so far (only once it has been >= +X/2)
     time     — at a checkpoint, P&L < 0  (long: close of t−3 / t−2 / t−1 10:30; short: t0 11:00 / 12:30)
     b12      — smoothed b12 moves Z pts against the position from its entry level
3. Validation: the best stop picked on 2025 expiries is scored on 2026 expiries, and a
   leave-one-cycle-out pick; a stop is only worth using if it beats "no stop" out of sample.

Outputs: results/backtests/stop_loss_study/{paths.csv, stops.csv, validation.csv}
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backtests.run_b12_long_expiry import Cycle, lot_sizes, ticker_frame  # noqa: E402
from backtests.run_b12_long_short_expiry import fill_upto  # noqa: E402
from data.storage.arctic_store import ArcticStore  # noqa: E402
from scripts.plot_htb_cycles import select_htb_expiries  # noqa: E402

OUT = ROOT / "results" / "backtests" / "stop_loss_study"
COST = 10                                   # bps per leg per side
TICKERS = ["360ONE", "ASTRAL", "ATHERENERG", "BDL", "BLUESTARCO", "COCHINSHIP", "DIXON", "HYUNDAI", "IREDA",
           "IRFC", "JUBLFOOD", "KAYNES", "KFINTECH", "KPITTECH", "LODHA", "LTM", "MANKIND", "NAM-INDIA", "NYKAA",
           "OFSS", "PIIND", "POWERGRID", "PREMIERENE", "RECLTD", "RVNL", "SBICARD", "VOLTAS"]


def leg(c: Cycle, p_in: int, p_out: int, f1e: float, f2e: float, side: int, d_b12: np.ndarray) -> dict:
    """Bar-close mark-to-market path of one leg. side=+1 long spread, −1 short spread."""
    x = c.s
    s_e = f1e - f2e
    sc = ((x.F1c - x.F2c).where((x.F1c > 0) & (x.F2c > 0))).to_numpy()
    path = side * (sc[p_in:p_out] - s_e) / f1e * 1e4               # decision bars p_in .. p_out-1
    return dict(c=c, p_in=p_in, p_out=p_out, f1e=f1e, f2e=f2e, s_e=s_e, side=side, path=path,
                b12=d_b12[p_in:p_out], b12_in=d_b12[p_in])


def realise(L: dict, p_fill: int) -> float:
    """Net P&L (bps of F1 entry) when the leg is closed at bar p_fill's open (or next priced)."""
    c = L["c"]
    j, f1x, f2x = fill_upto(c, p_fill, L["p_out"])
    gross = L["side"] * ((f1x - f2x) - L["s_e"])
    cost = COST / 1e4 * (L["f1e"] + L["f2e"] + f1x + f2x)
    return (gross - cost) / L["f1e"] * 1e4


def stop_exit(L: dict, kind: str, p: float, checkpoints: dict) -> int | None:
    """Bar (absolute position) whose OPEN the stop fills at, or None if it never fires."""
    path, b12, side = L["path"], L["b12"], L["side"]
    best = -np.inf
    for k in range(len(path) - 1):                                  # fill must be < scheduled exit
        v = path[k]
        if not np.isfinite(v):
            continue
        best = max(best, v)
        pos = L["p_in"] + k
        fire = False
        if kind == "fixed":
            fire = v <= -p
        elif kind == "trail":
            fire = best >= p / 2 and best - v >= p
        elif kind == "time":
            fire = pos == checkpoints.get(p, -1) and v < 0
        elif kind == "b12":
            fire = np.isfinite(b12[k]) and side * (b12[k] - L["b12_in"]) <= -p
        elif kind == "b12+fixed":                     # p = (b12 pts, bps)
            fire = (np.isfinite(b12[k]) and side * (b12[k] - L["b12_in"]) <= -p[0]) or v <= -p[1]
        if fire:
            return pos + 1
    return None


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    store = ArcticStore()
    longs, shorts = [], []
    for t in TICKERS:
        d = ticker_frame(store, t)
        htb = set(select_htb_expiries(d.exp, d.td, d.b12s, 0.30, 4).expiry)
        lots = lot_sizes(store, t)
        for E in sorted({e for e in d.exp.unique() if len(e) == 10 and e[4] == "-"}):
            c = Cycle(d, t, E, None)
            if not c.ok or c.t0 is None:
                continue
            x = c.s
            b12s = (x.timestamp.map(d.b12s) * 100).to_numpy()       # smoothed b12, pts
            # long leg
            trig = c.trigger("fall", 3, "gap>0", "t1")
            p_long_out, f1x, f2x = c.fill(c.p_forced if trig is None else trig)   # trigger() returns the fill bar
            pe = c.p_entry
            Lg = leg(c, pe, p_long_out, x.F1o.iat[pe], x.F2o.iat[pe], +1, b12s)
            # short leg: scheduled exit at t0 14:45
            t0 = np.flatnonzero((x.day == c.t0).to_numpy())
            le = [p for p in t0 if x.hm.iat[p] <= "14:45"]
            if not le:
                continue
            p_sx = fill_upto(c, le[-1], int(t0[-1]))[0]
            Sg = leg(c, p_long_out, p_sx, f1x, f2x, -1, b12s)
            b12_entry_long = d.b12s.loc[x.timestamp.iat[pe]] * 100
            b12_at_short = b12s[p_long_out]
            meta = dict(ticker=t, expiry=E, htb=E in htb, year=int(E[:4]),
                        exante_short=(b12_at_short >= 30) and (b12_at_short - b12_entry_long >= 10))
            # checkpoints
            days = x.day.to_numpy()
            def last_bar(day):
                b = np.flatnonzero(days == day)
                return int(b[-1]) if len(b) else -1
            dd = sorted(set(days))
            td_day = {int(x.td.iat[i]): x.day.iat[i] for i in range(len(x)) if np.isfinite(x.td.iat[i])}
            ck_long = {1: last_bar(td_day.get(3)), 2: last_bar(td_day.get(2)),
                       3: int(np.flatnonzero((days == c.t1) & (x.hm.to_numpy() == "10:30"))[0]) if ((days == c.t1) & (x.hm.to_numpy() == "10:30")).any() else -1}
            ck_short = {1: int(np.flatnonzero((days == c.t0) & (x.hm.to_numpy() == "11:00"))[0]) if ((days == c.t0) & (x.hm.to_numpy() == "11:00")).any() else -1,
                        2: int(np.flatnonzero((days == c.t0) & (x.hm.to_numpy() == "12:30"))[0]) if ((days == c.t0) & (x.hm.to_numpy() == "12:30")).any() else -1}
            longs.append({**meta, "L": Lg, "ck": ck_long, "base": realise(Lg, Lg["p_out"])})
            shorts.append({**meta, "L": Sg, "ck": ck_short, "base": realise(Sg, Sg["p_out"])})
        print(f"{t} done", flush=True)

    # ── 1. path anatomy ──
    rows = []
    for name, legs, sel in (("long (HTB)", longs, lambda r: r["htb"]), ("short (HTB)", shorts, lambda r: r["htb"]),
                            ("short (ex-ante set)", shorts, lambda r: r["exante_short"])):
        for r in filter(sel, legs):
            p = r["L"]["path"]
            p = p[np.isfinite(p)]
            if not len(p):
                continue
            k_mae = int(np.argmin(p))
            first_neg100 = np.flatnonzero(p <= -100)
            rows.append(dict(set=name, ticker=r["ticker"], expiry=r["expiry"], year=r["year"], pnl=r["base"],
                             mae=p.min(), mfe=p.max(), mae_h=k_mae * 5 / 60, hours=len(p) * 5 / 60,
                             h_to_minus100=first_neg100[0] * 5 / 60 if len(first_neg100) else np.nan))
    P = pd.DataFrame(rows)
    P.to_csv(OUT / "paths.csv", index=False)
    print("\n== 1. PATH ANATOMY (bps of F1, bar closes, before the scheduled exit)")
    for name, g in P.groupby("set"):
        w, l = g[g.pnl > 0], g[g.pnl <= 0]
        print(f"\n{name}: n={len(g)}  winners {len(w)}  losers {len(l)}")
        print(f"   MAE   winners median {w.mae.median():6.0f} (10th pct {w.mae.quantile(.1):6.0f}) | losers median {l.mae.median():6.0f}")
        print(f"   MFE   winners median {w.mfe.median():6.0f}                  | losers median {l.mfe.median():6.0f}  (losers that were once +100: {(l.mfe >= 100).mean():.0%})")
        for th in (50, 100, 150, 200):
            print(f"   dipped to −{th:<3}: winners {(w.mae <= -th).mean():4.0%}  losers {(l.mae <= -th).mean():4.0%}")
        print(f"   losers: median hours to −100 {l.h_to_minus100.median():.1f}h of {l.hours.median():.1f}h held; MAE at median hour {l.mae_h.median():.1f}")

    # ── 2. stops ──
    grids = {"fixed": [50, 75, 100, 150, 200, 300], "trail": [100, 150, 200, 300], "b12": [5, 10, 15, 25],
             "time": None, "b12+fixed": [(10, 150), (10, 200), (15, 150), (15, 200)]}
    res = []
    for name, legs, sel in (("long", longs, lambda r: r["htb"]), ("short", shorts, lambda r: r["htb"]),
                            ("short ex-ante", shorts, lambda r: r["exante_short"])):
        S = list(filter(sel, legs))
        tk = [1, 2, 3] if name == "long" else [1, 2]
        for kind, ps in grids.items():
            for p in (tk if kind == "time" else ps):
                for r in S:
                    pos = stop_exit(r["L"], kind, p, r["ck"])
                    pnl = realise(r["L"], pos) if pos is not None else r["base"]
                    res.append(dict(leg=name, stop=f"{kind} {p}", ticker=r["ticker"], expiry=r["expiry"], year=r["year"],
                                    pnl=pnl, base=r["base"], stopped=pos is not None))
        for r in S:
            res.append(dict(leg=name, stop="no stop", ticker=r["ticker"], expiry=r["expiry"], year=r["year"],
                            pnl=r["base"], base=r["base"], stopped=False))
    D = pd.DataFrame(res)
    D.to_csv(OUT / "stops.csv", index=False)
    labels = {"time 1": None}
    print("\n== 2. STOPS (net 10 bps/leg/side, bps of F1)")
    for name, g in D.groupby("leg"):
        a = g.groupby("stop").agg(n=("pnl", "size"), mean=("pnl", "mean"), median=("pnl", "median"),
                                  win=("pnl", lambda s: (s > 0).mean()), worst=("pnl", "min"),
                                  stopped=("stopped", "mean"),
                                  stopped_were_losers=("base", lambda s: np.nan)).drop(columns="stopped_were_losers")
        # how many stopped trades would have lost anyway
        st = g[g.stopped]
        a["saved_losers"] = st.groupby("stop").apply(lambda h: ((h.base <= 0)).sum())
        a["cut_winners"] = st.groupby("stop").apply(lambda h: ((h.base > 0)).sum())
        a["bps_saved_on_losers"] = st.groupby("stop").apply(lambda h: (h.pnl - h.base)[h.base <= 0].sum())
        a["bps_lost_on_winners"] = st.groupby("stop").apply(lambda h: (h.pnl - h.base)[h.base > 0].sum())
        print(f"\n{name}:")
        print(a.fillna(0).sort_values("mean", ascending=False).round(2).to_string())

    # ── 3. validation ──
    val = []
    for name, g in D.groupby("leg"):
        tr, te = g[g.year <= 2025], g[g.year >= 2026]
        if tr.expiry.nunique() >= 5:
            best = tr.groupby("stop").pnl.mean().idxmax()
            val.append(dict(leg=name, method="pick on 2025, test 2026", chosen=best,
                            test_mean=te[te.stop == best].pnl.mean(), test_no_stop=te[te.stop == "no stop"].pnl.mean(),
                            test_worst=te[te.stop == best].pnl.min(), test_worst_no_stop=te[te.stop == "no stop"].pnl.min(),
                            n_test=te[te.stop == "no stop"].shape[0]))
        cyc = g.ticker + "|" + g.expiry
        oos = []
        for cy in cyc.unique():
            best = g[cyc != cy].groupby("stop").pnl.mean().idxmax()
            oos.append(g[(cyc == cy) & (g.stop == best)].iloc[0])
        O = pd.DataFrame(oos)
        val.append(dict(leg=name, method="leave-one-cycle-out", chosen=O.stop.value_counts().index[0],
                        test_mean=O.pnl.mean(), test_no_stop=g[g.stop == "no stop"].pnl.mean(),
                        test_worst=O.pnl.min(), test_worst_no_stop=g[g.stop == "no stop"].pnl.min(), n_test=len(O)))
    V = pd.DataFrame(val)
    V.to_csv(OUT / "validation.csv", index=False)
    print("\n== 3. OUT-OF-SAMPLE CHECK")
    print(V.round(1).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
