"""run_b12_long_short_final.py — the full long + short b12 expiry strategy on HTB cycles.

HTB cycles: smoothed b12 (causal EWMA, 0.1-day half-life) peaked >= 30% in the final 5
trading days (hindsight filter — see the all-cycles line for the no-hindsight comparison).

LONG  (long F1 / short F2): enter t−4 open. Exit at the first of
        1. stop   — smoothed b12 10 pts below its level at entry (any time from entry)
        2. signal — on t−1, gap_0.05_1 falls 3 bars in a row after being > 0
        3. time   — t−1 15:15
SHORT (short F1 / long F2): enter at the scheduled long exit (signal or 15:15 — NOT the stop),
        exit expiry day 14:45, no stop. "filtered" = only when, at the short entry, b12 >= 30%
        and b12 rose >= 10 pts since the t−4 open (known at that moment).
All decisions on a bar close, fills at the next bar's open. Costs 10 bps per leg per side.

Outputs: results/backtests/b12_long_short_final/{trades.csv, summary.txt}
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
from backtests.stop_loss_study import TICKERS, leg, realise, stop_exit  # noqa: E402
from data.storage.arctic_store import ArcticStore  # noqa: E402
from scripts.plot_htb_cycles import select_htb_expiries  # noqa: E402

OUT = ROOT / "results" / "backtests" / "b12_long_short_final"


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    store = ArcticStore()
    rows = []
    for t in TICKERS:
        d = ticker_frame(store, t)
        htb = set(select_htb_expiries(d.exp, d.td, d.b12s, 0.30, 4).expiry)
        lots = lot_sizes(store, t)
        known = sorted(lots)
        for E in sorted({e for e in d.exp.unique() if len(e) == 10 and e[4] == "-"}):
            c = Cycle(d, t, E, None)
            if not c.ok or c.t0 is None:
                continue
            lot = lots.get(E) or (lots[max([k for k in known if k <= E], default=known[0])] if known else None)
            x = c.s
            b12s = (x.timestamp.map(d.b12s) * 100).to_numpy()
            pe = c.p_entry
            trig = c.trigger("fall", 3, "gap>0", "t1")              # returns the fill bar
            p_sched, f1x, f2x = c.fill(c.p_forced if trig is None else trig)
            L = leg(c, pe, p_sched, x.F1o.iat[pe], x.F2o.iat[pe], +1, b12s)
            p_stop = stop_exit(L, "b12", 10, {})
            long_net = realise(L, p_stop) if p_stop is not None else realise(L, p_sched)
            long_nostop = realise(L, p_sched)
            long_reason = "b12 stop" if p_stop is not None else ("signal" if trig is not None else "t-1 15:15")
            t0 = np.flatnonzero((x.day == c.t0).to_numpy())
            le = [p for p in t0 if x.hm.iat[p] <= "14:45"]
            if not le:
                continue
            p_sx = fill_upto(c, le[-1], int(t0[-1]))[0]
            S = leg(c, p_sched, p_sx, f1x, f2x, -1, b12s)
            short_net = realise(S, p_sx)
            b12_in, b12_sh = b12s[pe], b12s[p_sched]
            filt = (b12_sh >= 30) and (b12_sh - b12_in >= 10)
            # gross versions (cost 0) for reference
            g_long = ((x.F1o.iat[p_stop] - x.F2o.iat[p_stop]) if p_stop is not None else (f1x - f2x)) - (x.F1o.iat[pe] - x.F2o.iat[pe])
            ts_entry = x.timestamp.iat[pe]
            b12_pre_max = float(d.loc[(d.exp == E) & (d.index <= ts_entry), "b12s"].max() * 100)
            rows.append(dict(
                ticker=t, expiry=E, year=int(E[:4]), htb=E in htb, lot=lot, b12_max_to_entry=b12_pre_max, peak_b12=d.loc[d.exp == E].query("td <= 4").b12s.max() * 100,
                entry_ts=x.timestamp.iat[pe], b12_entry=b12_in, long_exit_ts=x.timestamp.iat[p_stop if p_stop is not None else p_sched],
                long_exit_reason=long_reason, short_entry_ts=x.timestamp.iat[p_sched], b12_short_entry=b12_sh,
                short_exit_ts=x.timestamp.iat[p_sx], short_filter_pass=filt,
                long_net=long_net, long_net_nostop=long_nostop, long_gross_approx=g_long / x.F1o.iat[pe] * 1e4,
                short_net=short_net, F1_entry=x.F1o.iat[pe], F1_short_entry=f1x))
        print(f"{t} done", flush=True)
    T = pd.DataFrame(rows)
    # combined in bps of the long leg's F1 entry (short leg rescaled to that base)
    scale = T.F1_short_entry / T.F1_entry
    T["combined_net"] = T.long_net + T.short_net * scale
    T["combined_net_filtered"] = T.long_net + np.where(T.short_filter_pass, T.short_net * scale, 0.0)
    T["combined_net_nostop"] = T.long_net_nostop + T.short_net * scale
    for col in ("long_net", "short_net", "combined_net", "combined_net_filtered"):
        T[col + "_rs_per_lot"] = np.where(T.lot.notna(), T[col] / 1e4 * T.F1_entry * T.lot, np.nan)
    T.to_csv(OUT / "trades.csv", index=False)

    H = T[T.htb]
    lines = []
    def stat(name, s):
        s = s.dropna()
        lines.append(f"{name:<46} n={len(s):3d}  mean {s.mean():+7.1f}  median {s.median():+7.1f}  win {(s > 0).mean():4.0%}  "
                     f"worst {s.min():+7.0f}  best {s.max():+7.0f}  sum {s.sum():+8.0f}  sharpe {s.mean() / s.std():+.2f}")
    lines.append(f"HTB cycles: {len(H)} on {H.ticker.nunique()} tickers | long exits: {H.long_exit_reason.value_counts().to_dict()} "
                 f"| short filter passes: {int(H.short_filter_pass.sum())}")
    lines.append("\n-- net bps of F1 (10 bps/leg/side)")
    stat("LONG  (stop + gap rule + 15:15)", H.long_net)
    stat("LONG  without the b12 stop", H.long_net_nostop)
    stat("SHORT (every cycle)", H.short_net)
    stat("SHORT (filtered at t-1)", H[H.short_filter_pass].short_net)
    stat("LONG + SHORT every cycle", H.combined_net)
    stat("LONG + SHORT filtered", H.combined_net_filtered)
    stat("LONG + SHORT, no stop", H.combined_net_nostop)
    lines.append("\n-- no hindsight filter, ALL cycles (for comparison)")
    stat("LONG + SHORT every cycle (all 27 tickers)", T.combined_net)
    stat("SHORT filtered only (all cycles)", T[T.short_filter_pass].short_net)
    lines.append("\n-- by year (HTB, long + short every cycle / filtered)")
    for y, g in H.groupby("year"):
        lines.append(f"   {y}: n={len(g):2d}  every {g.combined_net.mean():+6.0f} (win {(g.combined_net > 0).mean():.0%})  "
                     f"filtered {g.combined_net_filtered.mean():+6.0f} (win {(g.combined_net_filtered > 0).mean():.0%})")
    lines.append("\n-- by ticker (HTB): n | long | short | long+short | long+short filtered")
    bt = H.groupby("ticker").agg(n=("long_net", "size"), long=("long_net", "mean"), short=("short_net", "mean"),
                                 combined=("combined_net", "mean"), filtered=("combined_net_filtered", "mean"),
                                 rs_per_lot=("combined_net_rs_per_lot", "sum")).sort_values("combined", ascending=False)
    lines.append(bt.round(0).to_string())
    lines.append("\n-- by long exit reason (HTB): long net")
    lines.append(H.groupby("long_exit_reason").long_net.agg(["size", "mean", "median"]).round(0).to_string())
    txt = "\n".join(lines)
    (OUT / "summary.txt").write_text(txt, encoding="utf-8")
    print(txt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
