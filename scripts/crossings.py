"""crossings.py — where in the cycle does the fast b12 EWMA cross the slow one?

For each ticker and each (fast, slow) EWMA pair (half-lives in trading days) this
labels every bar of b12 (stored ``borrow_rates/{T}/{interval}``):

    up-cross     fast moves from <= slow to > slow (within the cycle)
    down-cross   fast moves from >= slow to < slow
    big          divergence (fast − slow) is in the ticker's top (1 − --big-quantile)
                 of positive values  (descriptive: the cut-off uses the whole sample)
    sig up-cross an up-cross after which the divergence reaches "big" within
                 --sig-horizon-minutes

and compares the LAST N trading days of each cycle (``--last-td``, default 9 = last 10
days, expiry day included) with the earlier part of the same cycle:

  * by trading days to expiry (buckets 0-4 / 5-9 / 10-14 / 15+): crossings per day,
    significant up-crosses per day, share of bars in a big divergence;
  * cycle by cycle: rate in the last N days vs earlier, and a sign test across cycles
    (how many cycles have the higher rate in the last N days; binomial p-value).

The first ``--warmup-days`` of every cycle are skipped (the slow EWMA has no history
yet). Read-only; writes CSVs.

Usage
-----
    b12-crossings IREDA RVNL SBICARD IRFC --interval 5minute --from 2025-04-01
    b12-crossings --all --interval 5minute --pairs 0.1/10 0.1/2
    (or: python scripts/crossings.py ...)

Outputs (``--out``, default ``results/ewma_b12/crossings_<interval>/``):
    by_bucket.csv     ticker x pair x days-to-expiry bucket
    per_cycle.csv     ticker x pair x cycle: last-N vs earlier rates
    summary.csv       per ticker (+ pooled) x pair: means and sign tests
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
from options.pricing.dte_utils import candles_per_day, interval_to_minutes  # noqa: E402
from scripts.ewma_b12_signal import ewma_by_cycle, halflife_to_alpha, load  # noqa: E402

BUCKETS = [(0, 4, "0-4 (expiry wk)"), (5, 9, "5-9"), (10, 14, "10-14"), (15, 999, "15+")]


def parse_pairs(items: list[str]) -> list[tuple[float, float]]:
    """'0.1/10' -> (0.1, 10.0)."""
    out = []
    for it in items:
        f, s = it.split("/")
        out.append((float(f), float(s)))
    return out


def tickers_for(store: ArcticStore, interval: str) -> list[str]:
    """Every ticker with a borrow_rates/{T}/{interval} series."""
    return sorted(x.split("/")[0] for x in store._lib("borrow_rates").list_symbols()
                  if x.count("/") == 1 and x.endswith(f"/{interval}"))


def forward_max(x: pd.Series, cycle: pd.Series, h: int) -> pd.Series:
    """Max of x over this bar and the next h-1 bars of the same cycle (reverse-rolling)."""
    return x[::-1].groupby(cycle[::-1]).transform(lambda s: s.rolling(h, min_periods=1).max())[::-1]


def label_bars(df: pd.DataFrame, interval: str, fast: float, slow: float, args) -> pd.DataFrame:
    """One row per usable bar with up / down / sig / big flags and trading days to expiry."""
    bpd = candles_per_day(interval)
    div = ewma_by_cycle(df.b12, df.cycle, halflife_to_alpha(fast, interval)[0]) - \
        ewma_by_cycle(df.b12, df.cycle, halflife_to_alpha(slow, interval)[0])
    sign = np.sign(div)
    prev = sign.groupby(df.cycle).shift(1)
    ok = (df.pos >= args.warmup_days * bpd) & df.td_to_exp.notna() & df.b12.notna()
    up = ok & (prev <= 0) & (sign > 0)
    down = ok & (prev >= 0) & (sign < 0)
    thr = div[ok & (div > 0)].quantile(args.big_quantile)
    h = max(1, int(round(args.sig_horizon_minutes / interval_to_minutes(interval))))
    sig = up & (forward_max(div, df.cycle, h) >= thr)
    big = ok & (div >= thr)
    d = pd.DataFrame({"cycle": df.cycle, "td": df.td_to_exp, "ok": ok, "up": up, "down": down,
                      "sig": sig, "big": big})
    d = d[d.ok].drop(columns="ok")
    d["last"] = d.td <= args.last_td
    return d


def summarise(c: pd.DataFrame, label: str) -> dict:
    """Last-N vs earlier, per cycle, with a sign test across cycles."""
    r = {"scope": label, "cycles": len(c)}
    for k in ("up", "sig", "big"):
        a, e = c[f"{k}_last"], c[f"{k}_early"]
        r[f"{k}_last"], r[f"{k}_early"] = a.mean(), e.mean()
        diff = (a - e).dropna()
        nz = diff[diff != 0]
        r[f"{k}_cycles_last_higher"] = f"{(nz > 0).sum()}/{len(nz)}"
        r[f"{k}_p_sign"] = binomtest(int((nz > 0).sum()), len(nz), 0.5).pvalue if len(nz) else np.nan
    tot = c.sig_n_last.sum() + c.sig_n_early.sum()
    r["share_of_sig_in_last"] = c.sig_n_last.sum() / tot if tot else np.nan   # share of ALL events
    r["share_of_bars_in_last"] = c.bars_last.sum() / (c.bars_last.sum() + c.bars_early.sum())
    return r


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("tickers", nargs="*", help="Tickers (or use --all).")
    ap.add_argument("--all", action="store_true", help="Every ticker with a borrow series at --interval.")
    ap.add_argument("--interval", default="5minute")
    ap.add_argument("--from", dest="from_date", default="2025-01-01", metavar="YYYY-MM-DD",
                    help="Start date (default 2025-01-01; the NSE holiday calendar covers 2025-2026).")
    ap.add_argument("--to", dest="to_date", default=None, metavar="YYYY-MM-DD")
    ap.add_argument("--pairs", nargs="+", default=["0.1/10", "0.1/2"],
                    help="fast/slow half-lives in trading days (default 0.1/10 0.1/2).")
    ap.add_argument("--last-td", type=int, default=9,
                    help="Last-window = trading days to expiry 0..LAST_TD (default 9 = last 10 days).")
    ap.add_argument("--warmup-days", type=float, default=2.0, help="Skip the first N trading days of each cycle.")
    ap.add_argument("--big-quantile", type=float, default=0.9, help="'Big' divergence cut-off (default 0.9).")
    ap.add_argument("--sig-horizon-minutes", type=float, default=375,
                    help="An up-cross is significant if it reaches 'big' within this many minutes (default 375 = 1 day).")
    ap.add_argument("--out", default=None, help="Output folder (default results/ewma_b12/crossings_<interval>).")
    args = ap.parse_args(argv)

    store = ArcticStore()
    tickers = tickers_for(store, args.interval) if args.all else [t.upper() for t in args.tickers]
    if not tickers:
        ap.error("give tickers or --all")
    pairs = parse_pairs(args.pairs)
    bpd = candles_per_day(args.interval)
    out = Path(args.out) if args.out else ROOT / "results" / "ewma_b12" / f"crossings_{args.interval}"
    out.mkdir(parents=True, exist_ok=True)

    bucket_rows, cyc_rows = [], []
    for t in tickers:
        df = load(store, t, args.interval, args.from_date, args.to_date)
        if df.empty or df.td_to_exp.isna().all():
            print(f"{t}: no data — skipped")
            continue
        for f, sl in pairs:
            pair = f"{f:g}/{sl:g}"
            d = label_bars(df, args.interval, f, sl, args)
            for lo, hi, lab in BUCKETS:
                b = d[(d.td >= lo) & (d.td <= hi)]
                days = len(b) / bpd
                bucket_rows.append({"ticker": t, "pair": pair, "bucket": lab, "trading_days": days,
                                    "up_cross_per_day": b.up.sum() / days if days else np.nan,
                                    "sig_up_cross_per_day": b.sig.sum() / days if days else np.nan,
                                    "share_bars_big_div": b.big.mean() if len(b) else np.nan})
            for c, g in d.groupby("cycle"):
                L, E = g[g["last"]], g[~g["last"]]
                if len(L) < bpd or len(E) < bpd:            # need >= 1 day on each side
                    continue
                cyc_rows.append({"ticker": t, "pair": pair, "cycle": c,
                                 "bars_last": len(L), "bars_early": len(E),
                                 "up_last": L.up.sum() / (len(L) / bpd), "up_early": E.up.sum() / (len(E) / bpd),
                                 "sig_last": L.sig.sum() / (len(L) / bpd), "sig_early": E.sig.sum() / (len(E) / bpd),
                                 "big_last": L.big.mean(), "big_early": E.big.mean(),
                                 "sig_n_last": int(L.sig.sum()), "sig_n_early": int(E.sig.sum())})

    B, C = pd.DataFrame(bucket_rows), pd.DataFrame(cyc_rows)
    summ = []
    pd.set_option("display.width", 260); pd.set_option("display.max_rows", 300)
    for pair in [f"{f:g}/{s:g}" for f, s in pairs]:
        cp = C[C.pair == pair]
        rows = [summarise(cp[cp.ticker == t], t) for t in sorted(cp.ticker.unique())]
        if cp.ticker.nunique() > 1:
            rows.append(summarise(cp, f"ALL ({cp.ticker.nunique()} tickers)"))
        S = pd.DataFrame(rows).assign(pair=pair)
        summ.append(S)
        print(f"\n=== fast/slow {pair} days — last {args.last_td + 1} trading days vs earlier ===")
        print(S.drop(columns="pair").round(3).to_string(index=False))
        bb = B[B.pair == pair]
        w = bb.groupby("bucket")[["trading_days", "up_cross_per_day", "sig_up_cross_per_day", "share_bars_big_div"]].apply(lambda g: pd.Series({
            "trading_days": g.trading_days.sum(),
            "up_cross_per_day": (g.up_cross_per_day * g.trading_days).sum() / g.trading_days.sum(),
            "sig_up_cross_per_day": (g.sig_up_cross_per_day * g.trading_days).sum() / g.trading_days.sum(),
            "share_bars_big_div": (g.share_bars_big_div * g.trading_days).sum() / g.trading_days.sum()}))
        print("all tickers, by trading days to expiry:\n", w.round(3).to_string())

    B.to_csv(out / "by_bucket.csv", index=False)
    C.to_csv(out / "per_cycle.csv", index=False)
    pd.concat(summ).to_csv(out / "summary.csv", index=False)
    print(f"\nWrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
