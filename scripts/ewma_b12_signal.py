"""ewma_b12_signal.py — does a fast-vs-slow EWMA divergence of b12 predict where b12 goes next?

For every ticker you name, at any interval and date range, this script:

1. reads ``borrow_rates/{TICKER}/{interval}`` (stored ``b12``, clipped at 0) from ArcticDB;
2. builds a FAST and a SLOW exponentially weighted moving average of b12. Both restart
   at every roll (``cycle_id``) so one contract never leaks into the next. Their
   memories are given as half-lives in TRADING DAYS and converted to bars / alpha
   for the interval (75 bars/day at 5minute, 25 at 15minute);
3. takes the divergence  ``div = EWMA_fast − EWMA_slow``  (fast above slow = b12 has
   recently been rising faster than its slow trend);
4. measures the future change ``b12[t+h] − b12[t]`` h bars ahead, within the same cycle;
5. scores every (fast, slow, h) combination: per-cycle rank IC (with a t-stat across
   cycles), direction hit rate against the base rate, and the average future change
   when the divergence is positive vs negative;
6. picks the best combination on the TRAIN cycles (the earlier ones) — the strongest IC
   in EITHER direction: "trend" (IC > 0, fast above slow -> b12 keeps rising) or
   "contrarian" (IC < 0, fast above slow -> b12 falls back) — and reports how that rule,
   with the direction learned on train, does on the TEST cycles (the most recent
   ``--test-frac``) it never saw (``test_*_dir`` columns: positive = it still worked).

Each ticker is scored on its own and, with several tickers, also pooled.

Usage
-----
    test-ewma-b12 IREDA --interval 15minute
    test-ewma-b12 IREDA RVNL SBICARD IRFC --interval 5minute --from 2025-04-01 --to 2026-09-30
    test-ewma-b12 IREDA --interval 5minute --fast 0.1 0.25 0.5 --slow 2 5 10 \\
        --horizons 15 25 75 150 --require-f2-trade --exclude-last-days 1
    (or: python scripts/ewma_b12_signal.py ...)

Outputs (``--out``, default ``results/ewma_b12/<tickers>_<interval>_<from>_<to>/``):
    grid.csv            every scope x fast x slow x h, train and test metrics
    best.csv            the combination chosen on train, per scope, with its test result
    per_cycle_best.csv  that combination's metrics cycle by cycle
    heatmap.html        train IC over fast x slow at each scope's best horizon
"""

from __future__ import annotations

import argparse
import sys
from itertools import product
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

from config.settings import settings  # noqa: E402
from data.storage.arctic_store import ArcticStore  # noqa: E402
from options.pricing.dte_utils import candles_per_day, trading_days_inclusive  # noqa: E402

TZ = settings.DEFAULT_TIMEZONE
MIN_CYCLE_OBS = 10          # a cycle needs this many usable bars to get its own IC
POOLED = "POOLED"


# ══════════════════════════════════════════════════════════════════════════════
# 1. Data
# ══════════════════════════════════════════════════════════════════════════════

def load(store: ArcticStore, ticker: str, interval: str,
         from_date: str | None, to_date: str | None) -> pd.DataFrame:
    """b12 and what the masks need, one row per bar, for one ticker.

    ``cycle`` is "TICKER|cycle_id" so cycles stay distinct when tickers are pooled;
    ``pos`` is the bar's position inside its cycle (0 = first bar after the roll);
    ``td_to_exp`` is trading days left to F1 expiry (0 on expiry day, 4 = start of a
    5-day expiry week); ``is_expiry_day`` flags bars on an F1 expiry day.
    """
    lo = pd.Timestamp(from_date, tz=TZ) if from_date else None
    hi = pd.Timestamp(to_date, tz=TZ) + pd.Timedelta(days=1) if to_date else None
    br = store.read_borrow_rates(ticker, interval, date_range=(lo, hi) if (lo or hi) else None)
    if br is None or br.empty or "b12" not in br.columns:
        return pd.DataFrame()
    br = br[~br.index.duplicated(keep="last")].sort_index()
    df = pd.DataFrame(index=br.index)
    df["b12"] = br["b12"].astype("float64")
    df["cycle"] = ticker + "|" + br["cycle_id"].astype(int).astype(str)
    df["F1_dte_star"] = br["F1_dte_star"].astype("float64") if "F1_dte_star" in br else np.nan
    df["F2_volume"] = br["F2_volume"].astype("float64") if "F2_volume" in br else np.nan
    df["is_expiry_day"] = br["is_expiry_day"].astype(bool) if "is_expiry_day" in br else False
    exp = store.read_futures(ticker, interval, with_spot=False, columns=["F1_expiry"],
                             date_range=(df.index.min(), df.index.max()))["F1_expiry"]
    exp = exp[~exp.index.duplicated(keep="last")].reindex(df.index).astype(str).str[:10]
    pairs = pd.DataFrame({"d": df.index.date, "e": exp.to_numpy()})
    uniq = pairs.drop_duplicates().copy()
    uniq["td"] = [trading_days_inclusive(d, pd.Timestamp(e).date()) - 1 if e not in ("None", "nan", "") else np.nan
                  for d, e in zip(uniq.d, uniq.e)]
    df["td_to_exp"] = pairs.merge(uniq, on=["d", "e"], how="left")["td"].to_numpy()
    df["pos"] = df.groupby("cycle", sort=False).cumcount()
    df["ticker"] = ticker
    return df


def split_cycles(df: pd.DataFrame, test_frac: float) -> pd.Series:
    """'train' / 'test' per row: the most recent ``test_frac`` of the ticker's cycles
    (ordered by their first bar) are test, the rest train."""
    first = pd.Series(df.index, index=df.index).groupby(df["cycle"]).min().sort_values()
    n = len(first)
    n_test = 0
    if n >= 2 and test_frac > 0:
        n_test = min(max(1, int(round(test_frac * n))), n - 1)   # at least 1 train and 1 test cycle
    test = set(first.index[n - n_test:]) if n_test else set()
    return df["cycle"].map(lambda c: "test" if c in test else "train")


# ══════════════════════════════════════════════════════════════════════════════
# 2. EWMAs and targets
# ══════════════════════════════════════════════════════════════════════════════

def halflife_to_alpha(hl_days: float, interval: str) -> tuple[float, float]:
    """Half-life in trading days -> (half-life in bars, alpha per bar).

    Weight on a value k bars old is alpha*(1-alpha)^k; it halves every ``bars`` bars,
    so (1-alpha)^bars = 1/2  ->  alpha = 1 - 0.5**(1/bars).
    """
    bars = hl_days * candles_per_day(interval)
    return bars, 1 - 0.5 ** (1 / bars)


def ewma_by_cycle(x: pd.Series, cycle: pd.Series, hl_bars: float) -> pd.Series:
    """EWMA of *x* with half-life *hl_bars*, restarted at every cycle (roll).

    A bar where *x* is NaN (a missing F1/F2 close) gets NaN, not the last value carried
    through; the next valid bar continues from the state before the gap (``ignore_na``).
    """
    ew = x.groupby(cycle, sort=False).transform(
        lambda s: s.ewm(halflife=hl_bars, adjust=True, ignore_na=True).mean())
    return ew.where(x.notna())


def future_change(x: pd.Series, cycle: pd.Series, h: int) -> pd.Series:
    """x[t+h] - x[t] inside the same cycle; NaN when t+h falls in the next contract."""
    return x.groupby(cycle, sort=False).shift(-h) - x


# ══════════════════════════════════════════════════════════════════════════════
# 3. Scoring
# ══════════════════════════════════════════════════════════════════════════════

def _group_corr(a: pd.Series, b: pd.Series, g: pd.Series) -> pd.Series:
    """Pearson correlation of a and b within each group g (vectorised)."""
    df = pd.DataFrame({"a": a, "b": b, "g": g})
    grp = df.groupby("g", sort=False)
    da = df.a - grp.a.transform("mean")
    db = df.b - grp.b.transform("mean")
    tmp = pd.DataFrame({"ab": da * db, "aa": da * da, "bb": db * db, "g": df.g})
    s = tmp.groupby("g", sort=False).sum()
    with np.errstate(invalid="ignore", divide="ignore"):
        return s.ab / np.sqrt(s.aa * s.bb)


def per_cycle_ic(div: pd.Series, fut: pd.Series, cycle: pd.Series) -> pd.Series:
    """Spearman rank IC of div vs fut inside each cycle with >= MIN_CYCLE_OBS bars."""
    df = pd.DataFrame({"d": div, "f": fut, "c": cycle}).dropna()
    n = df.groupby("c").size()
    df = df[df.c.isin(n[n >= MIN_CYCLE_OBS].index)]
    if df.empty:
        return pd.Series(dtype=float)
    rd = df.groupby("c", sort=False).d.rank()
    rf = df.groupby("c", sort=False).f.rank()
    return _group_corr(rd, rf, df.c).dropna()


def score(div: pd.Series, fut: pd.Series, cycle: pd.Series, min_div: float = 0.0) -> dict:
    """All metrics for one (fast, slow, h) on one set of bars."""
    df = pd.DataFrame({"d": div, "f": fut, "c": cycle}).dropna()
    out = {"n_obs": len(df), "n_cycles": df.c.nunique()}
    if df.empty:
        return out
    ic = per_cycle_ic(df.d, df.f, df.c)
    out["ic_mean"] = ic.mean() if len(ic) else np.nan
    out["ic_t"] = (ic.mean() / (ic.std(ddof=1) / np.sqrt(len(ic)))) if len(ic) > 2 and ic.std() > 0 else np.nan
    out["ic_pos_share"] = (ic > 0).mean() if len(ic) else np.nan
    out["ic_cycles"] = len(ic)
    call = (df.d.abs() > min_div) & (df.f != 0)
    if call.any():
        d, f = df.d[call], df.f[call]
        hit = (np.sign(d) == np.sign(f)).mean()
        up = (f > 0).mean()
        # balanced accuracy: average of "called up when it went up" and "called down when it
        # went down" — unlike the hit rate it can't be inflated by always guessing the common side
        tpr = (d[f > 0] > 0).mean() if (f > 0).any() else np.nan
        tnr = (d[f < 0] < 0).mean() if (f < 0).any() else np.nan
        out.update(hit_rate=hit, base_up=up, edge=hit - max(up, 1 - up),
                   balanced_acc=np.nanmean([tpr, tnr]), n_calls=int(call.sum()))
    pos, neg = df.f[df.d > min_div], df.f[df.d < -min_div]
    out["mean_fut_when_div_pos"] = pos.mean() if len(pos) else np.nan
    out["mean_fut_when_div_neg"] = neg.mean() if len(neg) else np.nan
    out["mean_diff_pos_minus_neg"] = out["mean_fut_when_div_pos"] - out["mean_fut_when_div_neg"]
    return out


def per_cycle_table(div, fut, cycle, split, min_div) -> pd.DataFrame:
    """Metrics of one combination, cycle by cycle (for per_cycle_best.csv)."""
    df = pd.DataFrame({"d": div, "f": fut, "c": cycle, "split": split}).dropna(subset=["d", "f"])
    rows = []
    for c, g in df.groupby("c", sort=False):
        r = score(g.d, g.f, g.c, min_div)
        rows.append({"cycle": c, "split": g.split.iloc[0], **r})
    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════════════════
# 4. Grid search
# ══════════════════════════════════════════════════════════════════════════════

def prepare(df: pd.DataFrame, interval: str, fast: list[float], slow: list[float],
            horizons: list[int], args) -> dict:
    """Everything a ticker needs for the grid, computed once: EWMA per half-life,
    future change per horizon, and the masks that do not depend on the combination."""
    hls = sorted(set(fast) | set(slow))
    ew = {hl: ewma_by_cycle(df.b12, df.cycle, halflife_to_alpha(hl, interval)[0]) for hl in hls}
    fut = {h: future_change(df.b12, df.cycle, h) for h in horizons}
    base = df.b12.notna()
    if args.exclude_last_days > 0:
        base &= df.F1_dte_star >= args.exclude_last_days
    if args.phase == "expiry-week":            # bar t inside the last N trading days of the cycle
        base &= df.td_to_exp < args.expiry_week_td
    elif args.phase == "pre-expiry-week":
        base &= df.td_to_exp >= args.expiry_week_td
    no_exp_target = {}
    if args.no_expiry_day_target:              # t+h must not land on (or start on) F1 expiry day
        for h in horizons:
            end_on_exp = df.is_expiry_day.groupby(df.cycle, sort=False).shift(-h).fillna(True).astype(bool)
            no_exp_target[h] = ~end_on_exp & ~df.is_expiry_day
    f2 = {}
    if args.require_f2_trade:
        traded = df.F2_volume.fillna(0) > 0
        for h in horizons:
            f2[h] = traded & traded.groupby(df.cycle, sort=False).shift(-h).fillna(False).astype(bool)
    return {"ew": ew, "fut": fut, "base": base, "f2": f2, "no_exp": no_exp_target,
            "split": split_cycles(df, args.test_frac)}


def run_grid(frames: dict[str, pd.DataFrame], prep: dict[str, dict], interval: str, args) -> pd.DataFrame:
    """Score every (scope, fast, slow, h) on train and test."""
    bpd = candles_per_day(interval)
    rows = []
    pairs = [(f, s) for f, s in product(args.fast, args.slow) if f < s]
    for (f, s), h in product(pairs, args.horizons):
        f_bars, f_alpha = halflife_to_alpha(f, interval)
        s_bars, s_alpha = halflife_to_alpha(s, interval)
        parts = []
        for t, df in frames.items():
            p = prep[t]
            m = p["base"] & (df.pos >= args.warmup * s_bars) & p["fut"][h].notna()
            if args.require_f2_trade:
                m &= p["f2"][h]
            if args.no_expiry_day_target:
                m &= p["no_exp"][h]
            parts.append(pd.DataFrame({"div": (p["ew"][f] - p["ew"][s])[m], "fut": p["fut"][h][m],
                                       "cycle": df.cycle[m], "split": p["split"][m], "ticker": t}))
        allp = pd.concat(parts)
        scopes = [(t, allp[allp.ticker == t]) for t in frames]
        if len(frames) > 1:
            scopes.append((POOLED, allp))
        for scope, d in scopes:
            row = {"scope": scope, "fast_hl_days": f, "slow_hl_days": s, "h_bars": h,
                   "h_days": h / bpd, "fast_hl_bars": f_bars, "slow_hl_bars": s_bars,
                   "fast_alpha": f_alpha, "slow_alpha": s_alpha}
            for sp in ("train", "test"):
                ds = d[d.split == sp]
                for k, v in score(ds["div"], ds["fut"], ds["cycle"], args.min_div).items():
                    row[f"{sp}_{k}"] = v
            rows.append(row)
    return pd.DataFrame(rows)


def add_direction(grid: pd.DataFrame) -> pd.DataFrame:
    """The divergence can predict in either direction:
        trend      — IC > 0: fast above slow -> b12 keeps rising;
        contrarian — IC < 0: fast above slow -> b12 falls back.
    The direction is fixed on TRAIN (sign of train IC) and the test metrics are
    re-expressed in that direction, so a positive ``test_*_dir`` means the train-learned
    rule still worked out of sample."""
    g = grid.copy()
    sgn = np.sign(g["train_ic_mean"]).replace(0, 1)
    g["direction"] = np.where(sgn > 0, "trend", "contrarian")
    g["abs_train_ic"] = g["train_ic_mean"].abs()
    for sp in ("train", "test"):
        if f"{sp}_ic_mean" in g:
            g[f"{sp}_ic_dir"] = sgn * g[f"{sp}_ic_mean"]
        if f"{sp}_hit_rate" in g:
            g[f"{sp}_hit_dir"] = np.where(sgn > 0, g[f"{sp}_hit_rate"], 1 - g[f"{sp}_hit_rate"])
            g[f"{sp}_edge_dir"] = g[f"{sp}_hit_dir"] - np.maximum(g[f"{sp}_base_up"], 1 - g[f"{sp}_base_up"])
        if f"{sp}_balanced_acc" in g:
            g[f"{sp}_bal_acc_dir"] = np.where(sgn > 0, g[f"{sp}_balanced_acc"], 1 - g[f"{sp}_balanced_acc"])
    return g


def pick_best(grid: pd.DataFrame) -> pd.DataFrame:
    """Per scope: the combination with the strongest TRAIN mean IC in either direction
    (largest |IC|; ties: larger |t|)."""
    g = grid.dropna(subset=["train_ic_mean"]).copy()
    g["_abs_t"] = g["train_ic_t"].abs()
    g = g.sort_values(["scope", "abs_train_ic", "_abs_t"], ascending=[True, False, False])
    return g.groupby("scope", sort=False).head(1).drop(columns="_abs_t").reset_index(drop=True)


# ══════════════════════════════════════════════════════════════════════════════
# 5. Output
# ══════════════════════════════════════════════════════════════════════════════

def verdict(r: pd.Series) -> str:
    def f(x, fmt):
        return "n/a" if pd.isna(x) else format(x, fmt)
    base = r.get("test_base_up", np.nan)
    return (f"{r.scope:<10} best: fast {r.fast_hl_days:g}d / slow {r.slow_hl_days:g}d / "
            f"h={int(r.h_bars)} bars ({r.h_days:.2f} d), {r.direction.upper()} — "
            f"train IC {f(r.train_ic_mean, '+.3f')} (t {f(r.train_ic_t, '.1f')}) | "
            f"TEST IC {f(r.get('test_ic_mean'), '+.3f')} (t {f(r.get('test_ic_t'), '.1f')}; "
            f"{f(r.get('test_ic_dir'), '+.3f')} in the train direction), "
            f"test hit {f(r.get('test_hit_dir'), '.1%')} vs base {f(max(base, 1 - base), '.1%')}, "
            f"test balanced acc {f(r.get('test_bal_acc_dir'), '.1%')} (50% = no skill)")


def write_heatmap(grid: pd.DataFrame, best: pd.DataFrame, path: Path) -> None:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    scopes = list(best.scope)
    fig = make_subplots(rows=len(scopes), cols=1, vertical_spacing=0.08,
                        subplot_titles=[f"{r.scope}: train IC at h={int(r.h_bars)} bars" for r in best.itertuples()])
    for i, r in enumerate(best.itertuples(), 1):
        g = grid[(grid.scope == r.scope) & (grid.h_bars == r.h_bars)]
        z = g.pivot_table(index="fast_hl_days", columns="slow_hl_days", values="train_ic_mean")
        fig.add_trace(go.Heatmap(z=z.values, x=[f"slow {c:g}d" for c in z.columns],
                                 y=[f"fast {i_:g}d" for i_ in z.index], colorscale="RdBu", zmid=0,
                                 text=np.round(z.values, 3), texttemplate="%{text}", showscale=(i == 1)),
                      row=i, col=1)
    fig.update_layout(height=320 * len(scopes), title="EWMA divergence of b12: train IC (fast x slow)")
    fig.write_html(str(path), include_plotlyjs=True)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("tickers", nargs="+", help="One or more tickers (scored separately and pooled).")
    ap.add_argument("--interval", default="15minute", help="borrow_rates interval (default 15minute).")
    ap.add_argument("--from", dest="from_date", default=None, metavar="YYYY-MM-DD")
    ap.add_argument("--to", dest="to_date", default=None, metavar="YYYY-MM-DD")
    ap.add_argument("--fast", nargs="+", type=float, default=[0.1, 0.25, 0.5, 1.0],
                    help="Fast EWMA half-lives in trading days (default 0.1 0.25 0.5 1).")
    ap.add_argument("--slow", nargs="+", type=float, default=[1.0, 2.0, 3.0, 5.0, 10.0],
                    help="Slow EWMA half-lives in trading days (default 1 2 3 5 10). Pairs need fast < slow.")
    ap.add_argument("--horizons", nargs="+", type=int, default=[1, 5, 15, 25, 75],
                    help="Bars ahead to predict (default 1 5 15 25 75).")
    ap.add_argument("--test-frac", type=float, default=0.3,
                    help="Share of each ticker's most recent cycles held out as test (default 0.3).")
    ap.add_argument("--warmup", type=float, default=1.0,
                    help="Skip the first WARMUP x slow-half-life bars of every cycle (default 1).")
    ap.add_argument("--exclude-last-days", type=float, default=0.0,
                    help="Drop bars with fewer than this many days to F1 expiry (default 0 = keep all).")
    ap.add_argument("--require-f2-trade", action="store_true",
                    help="Keep only bars where F2 traded at both t and t+h (guards against stale F2 prints).")
    ap.add_argument("--phase", choices=["all", "expiry-week", "pre-expiry-week"], default="all",
                    help="Score only bars inside the expiry week (last --expiry-week-td trading days "
                         "of the cycle, expiry day included), only bars before it, or all (default).")
    ap.add_argument("--expiry-week-td", type=int, default=5,
                    help="Length of the expiry week in trading days (default 5).")
    ap.add_argument("--no-expiry-day-target", action="store_true",
                    help="Drop bars on F1 expiry day and bars whose t+h lands on it (removes the "
                         "expiry-day b12 collapse from the target).")
    ap.add_argument("--min-div", type=float, default=0.0,
                    help="Only make a direction call when |divergence| exceeds this (b12 units, default 0).")
    ap.add_argument("--top", type=int, default=15, help="Rows of the grid to print per scope (default 15).")
    ap.add_argument("--out", default=str(ROOT / "results" / "ewma_b12"), help="Output folder.")
    ap.add_argument("--no-plot", action="store_true", help="Skip heatmap.html.")
    args = ap.parse_args(argv)

    tickers = [t.upper() for t in args.tickers]
    store = ArcticStore()
    frames, prep = {}, {}
    for t in tickers:
        df = load(store, t, args.interval, args.from_date, args.to_date)
        if df.empty:
            print(f"{t}: no borrow_rates/{t}/{args.interval} data in range — skipped")
            continue
        frames[t] = df
        prep[t] = prepare(df, args.interval, args.fast, args.slow, args.horizons, args)
        sp = prep[t]["split"]
        print(f"{t}: {len(df):,} bars, {df.cycle.nunique()} cycles "
              f"({df[sp == 'train'].cycle.nunique()} train / {df[sp == 'test'].cycle.nunique()} test), "
              f"{df.index.min().date()} → {df.index.max().date()}")
    if not frames:
        return 1

    grid = add_direction(run_grid(frames, prep, args.interval, args))
    best = pick_best(grid)

    tag = f"{'-'.join(frames)[:60]}_{args.interval}_{args.from_date or 'start'}_{args.to_date or 'end'}"
    if args.exclude_last_days:
        tag += f"_xlast{args.exclude_last_days:g}d"
    if args.require_f2_trade:
        tag += "_f2trade"
    if args.min_div:
        tag += f"_mindiv{args.min_div:g}"
    if args.phase != "all":
        tag += f"_{args.phase}{args.expiry_week_td}td"
    if args.no_expiry_day_target:
        tag += "_noexpdaytarget"
    out = Path(args.out) / tag
    out.mkdir(parents=True, exist_ok=True)
    grid.to_csv(out / "grid.csv", index=False)
    best.to_csv(out / "best.csv", index=False)

    pcs = []
    for r in best.itertuples():
        f_hl, s_hl, h = r.fast_hl_days, r.slow_hl_days, int(r.h_bars)
        for t in (frames if r.scope == POOLED else [r.scope]):
            df, p = frames[t], prep[t]
            m = p["base"] & (df.pos >= args.warmup * halflife_to_alpha(s_hl, args.interval)[0]) & p["fut"][h].notna()
            if args.require_f2_trade:
                m &= p["f2"][h]
            if args.no_expiry_day_target:
                m &= p["no_exp"][h]
            tab = per_cycle_table((p["ew"][f_hl] - p["ew"][s_hl])[m], p["fut"][h][m], df.cycle[m],
                                  p["split"][m], args.min_div)
            tab.insert(0, "scope", r.scope)
            pcs.append(tab)
    if pcs:
        pd.concat(pcs).to_csv(out / "per_cycle_best.csv", index=False)
    if not args.no_plot and len(best):
        write_heatmap(grid, best, out / "heatmap.html")

    pd.set_option("display.width", 250)
    show = ["fast_hl_days", "slow_hl_days", "h_bars", "h_days", "direction", "train_ic_mean", "train_ic_t",
            "train_ic_pos_share", "train_hit_dir", "train_edge_dir", "test_ic_mean", "test_ic_t",
            "test_ic_dir", "test_hit_dir", "test_edge_dir", "test_bal_acc_dir"]
    for scope in best.scope:
        g = grid[grid.scope == scope].sort_values("abs_train_ic", ascending=False).head(args.top)
        print(f"\n── {scope}: top {args.top} combinations by |TRAIN IC| ──")
        print(g[show].round(4).to_string(index=False))
    print("\n── Chosen on train, scored on test ──")
    for _, r in best.iterrows():
        print(verdict(r))
    print(f"\nWrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
