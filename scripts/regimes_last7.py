"""regimes_last7.py — near expiry, while the fast b12 EWMA STAYS above (below) the slow one,
does b12 keep rising (falling)?

Only bars in the last N trading days of each cycle are tested (``--last-td``, default
6 = last 7 trading days, expiry day included). Two tests, per (fast, slow) EWMA pair:

A. Segments ("hold while the state lasts")
   A state is fast-above-slow (+1) or fast-below-slow (−1). Inside the window, ENTER at
   the first bar where the state has held for ``--confirm-minutes`` (so the move that
   created the state is not counted) and EXIT at the bar where the state flips, or at
   the end of the cycle (expiry). Measure  Δb12 = b12[exit] − b12[entry]  and
   excess = Δb12 − normal drift (mean 1-bar change of b12 at each day-to-expiry,
   summed over the same bars). Note: exiting at the flip leans the result by
   construction (an above-segment ends when b12 has started to fall) — it measures what
   a crossover rule would earn, not a neutral forecast; test B is the neutral one.

B. Fixed horizon
   Every window bar whose state has held >= 60 minutes is scored on the b12 change over
   the next ``--horizon-minutes`` (default 60 and 375 = 1 trading day), raw and in excess
   of the normal drift.

Both are summarised per ticker and pooled, with a per-cycle sign test (in how many
cycles did above-segments beat below-segments). Also printed: segments by entry day,
and below-segments that run into expiry day vs those that end before it (the expiry-day
b12 collapse).

Usage
-----
    b12-regimes IREDA RVNL SBICARD IRFC --interval 5minute --from 2025-04-01
    b12-regimes --all --interval 5minute --pairs 0.1/2 0.1/10 0.5/5 --confirm-minutes 15 60
    (or: python scripts/regimes_last7.py ...)

Outputs (``--out``, default ``results/ewma_b12/regimes_<interval>/``):
    segments.csv          one row per segment (ticker, pair, confirm, cycle, state, Δb12, excess …)
    fixed_horizon.csv     one row per scored bar (ticker, pair, h, cycle, state, fwd, fwd_excess)
    summary_segments.csv  per scope x pair x confirm
    summary_fixed.csv     per scope x pair x horizon
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest

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

STATE_HELD_MIN = 60          # test B: state must have held this many minutes


def bars(minutes: float, interval: str) -> int:
    return max(1, int(round(minutes / interval_to_minutes(interval))))


def analyse_ticker(df: pd.DataFrame, t: str, interval: str, pairs, args) -> tuple[list, list]:
    """Segments (test A) and fixed-horizon rows (test B) for one ticker."""
    df = df.reset_index(drop=True)
    win = df.td_to_exp <= args.last_td
    step = df.b12.groupby(df.cycle).diff()
    drift_by_td = step[win].groupby(df.td_to_exp[win]).mean()     # normal 1-bar change per day-to-expiry
    drift = df.td_to_exp.map(drift_by_td).fillna(0.0)

    seg_rows, bar_frames = [], []
    for f, sl in pairs:
        pair = f"{f:g}/{sl:g}"
        div = ewma_by_cycle(df.b12, df.cycle, halflife_to_alpha(f, interval)[0]) - \
            ewma_by_cycle(df.b12, df.cycle, halflife_to_alpha(sl, interval)[0])
        state = np.sign(div).fillna(0).astype(int)
        new_run = (state != state.groupby(df.cycle).shift(1)) | (df.cycle != df.cycle.shift(1))
        run_id = new_run.cumsum()
        run_pos = df.groupby(run_id).cumcount()

        # ── A. segments ──
        for cm in args.confirm_minutes:
            K = bars(cm, interval)
            for _, g in df.groupby(run_id):
                st = int(state[g.index[0]])
                if st == 0:
                    continue
                inwin = g.index[win[g.index].to_numpy() & (run_pos[g.index] >= K - 1).to_numpy()]
                if len(inwin) == 0:
                    continue
                entry, last = inwin[0], g.index[-1]
                nxt = last + 1
                exit_ = nxt if (nxt < len(df) and df.cycle[nxt] == df.cycle[last]) else last
                if exit_ <= entry:
                    continue
                d_b12 = df.b12[exit_] - df.b12[entry]
                seg_rows.append({"ticker": t, "pair": pair, "confirm_min": cm, "cycle": df.cycle[entry],
                                 "state": "above" if st > 0 else "below", "bars": exit_ - entry,
                                 "td_entry": df.td_to_exp[entry], "ends_at_expiry": exit_ == last,
                                 "d_b12": d_b12, "excess": d_b12 - drift[entry + 1: exit_ + 1].sum()})

        # ── B. fixed horizon ──
        conf = win & (run_pos >= bars(STATE_HELD_MIN, interval) - 1) & (state != 0)
        for hm in args.horizon_minutes:
            h = bars(hm, interval)
            fwd = df.b12.groupby(df.cycle).shift(-h) - df.b12
            exp_fwd = drift[::-1].groupby(df.cycle[::-1]).transform(
                lambda x: x.rolling(h, min_periods=h).sum().shift(1))[::-1]
            m = conf & fwd.notna()
            bar_frames.append(pd.DataFrame({"ticker": t, "pair": pair, "horizon_min": hm, "cycle": df.cycle[m],
                                            "state": np.where(state[m] > 0, "above", "below"),
                                            "fwd": fwd[m], "fwd_excess": fwd[m] - exp_fwd[m]}))
    return seg_rows, bar_frames


def _sign_test(x: pd.DataFrame, value: str) -> tuple[str, float]:
    pc = x.groupby(["cycle", "state"])[value].mean().unstack()
    if not {"above", "below"} <= set(pc.columns):
        return "0/0", np.nan
    diff = (pc["above"] - pc["below"]).dropna()
    nz = diff[diff != 0]
    p = binomtest(int((nz > 0).sum()), len(nz), 0.5).pvalue if len(nz) else np.nan
    return f"{(nz > 0).sum()}/{len(nz)}", p


def seg_summary(x: pd.DataFrame, label: str) -> dict:
    r = {"scope": label}
    for st in ("above", "below"):
        g = x[x.state == st]
        r[f"n_{st}"] = len(g)
        r[f"{st}_mean_dB12_pts"] = g.d_b12.mean() * 100
        r[f"{st}_median_dB12_pts"] = g.d_b12.median() * 100
        r[f"{st}_share_rise"] = (g.d_b12 > 0).mean()
        r[f"{st}_mean_excess_pts"] = g.excess.mean() * 100
    r["cycles_above>below"], r["p_sign"] = _sign_test(x, "d_b12")
    return r


def bar_summary(x: pd.DataFrame, label: str) -> dict:
    r = {"scope": label}
    for st in ("above", "below"):
        g = x[x.state == st]
        r[f"{st}_mean_fwd_pts"] = g.fwd.mean() * 100
        r[f"{st}_share_fwd_up"] = (g.fwd > 0).mean()
        r[f"{st}_mean_excess_pts"] = g.fwd_excess.mean() * 100
    r["cycles_above>below"], r["p_sign"] = _sign_test(x, "fwd")
    return r


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("tickers", nargs="*", help="Tickers (or use --all).")
    ap.add_argument("--all", action="store_true", help="Every ticker with a borrow series at --interval.")
    ap.add_argument("--interval", default="5minute")
    ap.add_argument("--from", dest="from_date", default="2025-04-01", metavar="YYYY-MM-DD")
    ap.add_argument("--to", dest="to_date", default=None, metavar="YYYY-MM-DD")
    ap.add_argument("--pairs", nargs="+", default=["0.1/2", "0.1/10", "0.5/5"],
                    help="fast/slow half-lives in trading days (default 0.1/2 0.1/10 0.5/5).")
    ap.add_argument("--last-td", type=int, default=6,
                    help="Window = trading days to expiry 0..LAST_TD (default 6 = last 7 days, expiry day included).")
    ap.add_argument("--confirm-minutes", nargs="+", type=float, default=[15, 60],
                    help="Test A: minutes the state must have held before entry (default 15 60).")
    ap.add_argument("--horizon-minutes", nargs="+", type=float, default=[60, 375],
                    help="Test B: forward horizons in minutes (default 60 375 = 1 trading day).")
    ap.add_argument("--out", default=None, help="Output folder (default results/ewma_b12/regimes_<interval>).")
    args = ap.parse_args(argv)

    store = ArcticStore()
    tickers = tickers_for(store, args.interval) if args.all else [t.upper() for t in args.tickers]
    if not tickers:
        ap.error("give tickers or --all")
    pairs = parse_pairs(args.pairs)
    out = Path(args.out) if args.out else ROOT / "results" / "ewma_b12" / f"regimes_{args.interval}"
    out.mkdir(parents=True, exist_ok=True)

    seg_rows, bar_frames = [], []
    for t in tickers:
        df = load(store, t, args.interval, args.from_date, args.to_date)
        if df.empty or df.td_to_exp.isna().all():
            print(f"{t}: no data — skipped")
            continue
        s, b = analyse_ticker(df, t, args.interval, pairs, args)
        seg_rows += s
        bar_frames += b
    S = pd.DataFrame(seg_rows)
    Bb = pd.concat(bar_frames) if bar_frames else pd.DataFrame()
    if S.empty:
        print("nothing to score")
        return 1

    scopes = sorted(S.ticker.unique())
    pd.set_option("display.width", 260); pd.set_option("display.max_rows", 300)
    ss, sb = [], []
    for pair in [f"{f:g}/{s:g}" for f, s in pairs]:
        for cm in args.confirm_minutes:
            x = S[(S.pair == pair) & (S.confirm_min == cm)]
            rows = [seg_summary(x[x.ticker == t], t) for t in scopes]
            if len(scopes) > 1:
                rows.append(seg_summary(x, f"ALL ({len(scopes)})"))
            R = pd.DataFrame(rows).assign(pair=pair, confirm_min=cm)
            ss.append(R)
            print(f"\n=== SEGMENTS  fast/slow {pair}  state held {cm:g} min before entry ===")
            print(R.drop(columns=["pair", "confirm_min"]).round(3).to_string(index=False))
        for hm in args.horizon_minutes:
            x = Bb[(Bb.pair == pair) & (Bb.horizon_min == hm)]
            rows = [bar_summary(x[x.ticker == t], t) for t in scopes]
            if len(scopes) > 1:
                rows.append(bar_summary(x, f"ALL ({len(scopes)})"))
            R = pd.DataFrame(rows).assign(pair=pair, horizon_min=hm)
            sb.append(R)
            print(f"\n--- FIXED HORIZON  fast/slow {pair}  state held >= {STATE_HELD_MIN} min, next {hm:g} min ---")
            print(R.drop(columns=["pair", "horizon_min"]).round(3).to_string(index=False))

    p0, c0 = f"{pairs[0][0]:g}/{pairs[0][1]:g}", max(args.confirm_minutes)
    x = S[(S.pair == p0) & (S.confirm_min == c0)]
    print(f"\nsegments by entry day ({p0}, confirm {c0:g} min), all tickers — count and mean Δb12 (pts):")
    print(x.groupby(["td_entry", "state"]).d_b12.agg(["count", "mean"])
          .assign(mean=lambda d: d["mean"] * 100).round(2).unstack().to_string())
    print(f"\nbelow-segments ending at expiry vs not ({p0}, confirm {c0:g} min) — Δb12 pts:")
    print((x[x.state == "below"].groupby("ends_at_expiry").d_b12.agg(["count", "mean", "median"])
           * [1, 100, 100]).round(2).to_string())

    S.to_csv(out / "segments.csv", index=False)
    Bb.to_csv(out / "fixed_horizon.csv", index=False)
    pd.concat(ss).to_csv(out / "summary_segments.csv", index=False)
    pd.concat(sb).to_csv(out / "summary_fixed.csv", index=False)
    print(f"\nWrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
