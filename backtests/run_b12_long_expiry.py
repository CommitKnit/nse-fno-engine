"""run_b12_long_expiry.py — long b12 into the expiry week, exit when gap_0.05_1 rolls over.

Trade (one per qualifying contract expiry E):
  * long b12 = long the calendar spread: BUY F1, SELL F2 (b12 = r − ln(F2/F1)/(t2−t1) rises
    exactly when F1 gains on F2). P&L per share = Δ(F1 − F2).
  * entry: t−4, first bar at/after 09:30 with both legs priced (the 09:15–09:25 bars are skipped as noisy).
  * exit:  first bar where the gap_0.05_1 exit rule fires (decided on that bar's CLOSE,
           filled at the NEXT bar's open); if it never fires, at t−1 15:15.
  * cycles: expiries whose smoothed b12 (causal EWMA, 0.1-day half-life) peaked >= 30%
           in the final 5 trading days — NOTE this filter uses the future (the peak is after
           entry); results on all cycles are reported alongside.

gap_0.05_1 = EWMA(b12 pts, 0.05-day half-life) − EWMA(b12 pts, 1-day), restarted every roll,
NaN where b12 is missing (scripts/build_gap_timeseries.gap_frame).

Exit rules searched:
  fall  — gap_0.05_1 lower than the previous bar for x bars in a row; optional prior filter
          "gap>0" (the gap was positive when the fall started, i.e. b12 had been rising)
  dd    — gap_0.05_1 θ points below its high since the signal window opened (high > 0)
  window "t1"  : signal only on t−1 (exit happens on t−1, as specified)
         "all" : signal from entry onward (can exit before t−1)

Outputs: results/backtests/b12_long_expiry/{trades_grid,trades_best,grid,loco,post_exit,baselines}.csv
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # noqa: BLE001
    pass

from data.storage.arctic_store import ArcticStore  # noqa: E402
from scripts.build_gap_timeseries import gap_frame  # noqa: E402
from scripts.ewma_b12_signal import ewma_by_cycle, load  # noqa: E402
from scripts.plot_htb_cycles import select_htb_expiries  # noqa: E402

R = 0.0625
OUT = ROOT / "results" / "backtests" / "b12_long_expiry"
FORCED_EXIT = "15:15"
ENTRY_TIME = "09:30"              # t-4 entry: first bar at/after this time
COSTS_BPS = (0, 5, 10)                      # per leg, per side, of that leg's notional
FALL_X = (1, 2, 3, 4, 5, 6, 8, 10, 12)
DD_THETA = (0.5, 1, 2, 3, 5, 8)             # b12 points


# ── data ───────────────────────────────────────────────────────────────────────

def ticker_frame(store: ArcticStore, t: str, iv: str = "5minute") -> pd.DataFrame:
    br = store.read_borrow_rates(t, iv)
    br = br[~br.index.duplicated(keep="last")].sort_index()
    fut = store.read_futures(t, iv, with_spot=False)
    fut = fut[~fut.index.duplicated(keep="last")].reindex(br.index)
    d = pd.DataFrame(index=br.index)
    d["F1o"], d["F2o"] = fut["F1_open"].astype(float), fut["F2_open"].astype(float)
    d["F1c"], d["F2c"] = fut["F1_close"].astype(float), fut["F2_close"].astype(float)
    d["exp"] = fut["F1_expiry"].astype(str).str[:10]
    d["cyc"] = br["cycle_id"].astype(int)
    d["d1"], d["d2"] = br["F1_dte_star"].astype(float), br["F2_dte_star"].astype(float)
    d["b12"] = br["b12"].astype(float) * 100                      # b12 points (% p.a.)
    d["td"] = load(store, t, iv, None, None)["td_to_exp"].reindex(br.index)
    d["b12s"] = ewma_by_cycle(br["b12"].astype(float), d["cyc"], 7.5)
    d["gap"] = gap_frame(d["b12"], t + "|" + d["cyc"].astype(str), iv)["gap_0.05_1"]
    d["hm"] = d.index.strftime("%H:%M")
    d["day"] = d.index.strftime("%Y-%m-%d")
    return d


def lot_sizes(store: ArcticStore, t: str) -> dict[str, int]:
    out = {}
    for iv in ("5minute", "15minute"):
        for e in store.list_option_expiries(t, iv):
            o = store.read_options(t, iv, e)
            if "lot_size" in o.columns and o["lot_size"].notna().any():
                out[e] = int(o["lot_size"].dropna().iloc[0])
    return out


def implied_b12(f1, f2, d1, d2) -> float:
    dt = (d2 - d1) / 365
    if not (f1 > 0 and f2 > 0 and dt > 0):
        return np.nan
    return max(0.0, R - np.log(f2 / f1) / dt) * 100


# ── one cycle ──────────────────────────────────────────────────────────────────

class Cycle:
    """Bars of expiry E while it is the F1 contract, with entry/forced-exit positions."""

    def __init__(self, d: pd.DataFrame, t: str, E: str, lot: int | None):
        self.t, self.E, self.lot = t, E, lot
        self.ok, self.why = False, ""
        own = d[d.exp == E]
        days = own.groupby("day").td.first()
        t4, t1, t0 = days[days == 4].index, days[days == 1].index, days[days == 0].index
        if not len(t4) or not len(t1):
            self.why = "no t-4 or t-1 session with this contract in F1"
            return
        self.t4, self.t1 = t4[0], t1[0]
        self.t0 = t0[0] if len(t0) else None
        # wrong-front: the F1 slot held another contract on a day of the trade window
        win_days = sorted(set(d.day[(d.day >= self.t4) & (d.day <= self.t1)]))
        own_days = set(own.day)
        if any(x not in own_days for x in win_days):
            self.why = "F1 slot held another contract during t-4..t-1"
            return
        s = own[(own.day >= self.t4)].copy()
        self.s = s.reset_index()
        x = self.s
        # entry: first bar at/after ENTRY_TIME on t-4 (the opening bars are noisy)
        t4_ok = np.flatnonzero(((x.day == self.t4) & (x.hm >= ENTRY_TIME)).to_numpy())
        self.p_entry = self._first_priced(int(t4_ok[0])) if len(t4_ok) else None
        t1_bars = np.flatnonzero((x.day == self.t1).to_numpy())
        at = np.flatnonzero(((x.day == self.t1) & (x.hm == FORCED_EXIT)).to_numpy())
        le = np.flatnonzero(((x.day == self.t1) & (x.hm <= FORCED_EXIT)).to_numpy())
        self.p_forced = int(at[0] if len(at) else le[-1])
        self.p_t1_start, self.p_t1_end = int(t1_bars[0]), int(t1_bars[-1])
        if self.p_entry is None or self.p_entry >= self.p_forced:
            self.why = "no priced entry bar"
            return
        # gap falling-run length, peak info
        g = x.gap.to_numpy()
        run = np.zeros(len(g), int)
        for i in range(1, len(g)):
            if np.isfinite(g[i]) and np.isfinite(g[i - 1]) and g[i] < g[i - 1]:
                run[i] = run[i - 1] + 1
        self.g, self.run = g, run
        self.peak_b12 = float(d.loc[d.exp == E].query("td <= 4").b12s.max() * 100)
        self.ok = True

    def _first_priced(self, i: int, last: int | None = None):
        x = self.s
        last = len(x) - 1 if last is None else last
        for j in range(i, last + 1):
            if x.F1o.iat[j] > 0 and x.F2o.iat[j] > 0:
                return j
        return None

    def fill(self, i: int):
        """(position, F1, F2) at bar i's open, or the next bar with both legs priced up to
        the forced-exit bar; else t-1's last close."""
        j = self._first_priced(i, self.p_forced)
        x = self.s
        if j is not None:
            return j, x.F1o.iat[j], x.F2o.iat[j]
        k = self.p_t1_end
        while k > i and not (x.F1c.iat[k] > 0 and x.F2c.iat[k] > 0):
            k -= 1
        return k, x.F1c.iat[k], x.F2c.iat[k]

    def trigger(self, method: str, param: float, prior: str, window: str):
        """Bar to FILL at: the bar after the first bar whose close fires the exit rule inside
        the window (decided on the close, filled at the next bar's open), else None."""
        lo = self.p_entry if window == "all" else max(self.p_entry, self.p_t1_start)
        hi = self.p_forced - 1                       # last decision bar (fill at <= forced)
        g, run = self.g, self.run
        hmax = -np.inf
        for t in range(lo, hi + 1):
            if np.isfinite(g[t]):
                hmax = max(hmax, g[t])
            if method == "fall":
                x = int(param)
                if run[t] >= x:
                    if prior == "gap>0":
                        g0 = g[t - run[t]]
                        if not (np.isfinite(g0) and g0 > 0):
                            continue
                    return t + 1
            else:  # dd
                if np.isfinite(g[t]) and hmax > 0 and hmax - g[t] >= param:
                    return t + 1
        return None

    def trade(self, exit_pos: int | None, label: str, reason: str | None = None) -> dict:
        x = self.s
        pe, f1e, f2e = self.p_entry, x.F1o.iat[self.p_entry], x.F2o.iat[self.p_entry]
        if exit_pos is None:
            px, f1x, f2x = self.fill(self.p_forced)
            reason = reason or "forced 15:15"
        else:
            px, f1x, f2x = self.fill(exit_pos)
            reason = reason or "signal"
        s_e, s_x = f1e - f2e, f1x - f2x
        gross = s_x - s_e                                         # ₹ per share, long F1 / short F2
        out = dict(ticker=self.t, expiry=self.E, rule=label, peak_b12=self.peak_b12,
                   entry_ts=x.timestamp.iat[pe], exit_ts=x.timestamp.iat[px], exit_reason=reason,
                   hold_h=(px - pe) * 5 / 60, F1_entry=f1e, F2_entry=f2e, F1_exit=f1x, F2_exit=f2x,
                   spread_entry=s_e, spread_exit=s_x,
                   b12_entry=implied_b12(f1e, f2e, x.d1.iat[pe], x.d2.iat[pe]),
                   b12_exit=implied_b12(f1x, f2x, x.d1.iat[px], x.d2.iat[px]),
                   pnl_per_share=gross, pnl_bps=gross / f1e * 1e4, lot=self.lot)
        out["b12_change_pts"] = out["b12_exit"] - out["b12_entry"]
        for c in COSTS_BPS[1:]:
            cost = c / 1e4 * (f1e + f2e + f1x + f2x)
            out[f"net{c}_per_share"] = gross - cost
            out[f"net{c}_bps"] = (gross - cost) / f1e * 1e4
        if self.lot:
            out["pnl_per_lot"] = gross * self.lot
            out["net10_per_lot"] = out["net10_per_share"] * self.lot
        out.update(self.post_exit(px, s_x, out["b12_exit"], f1e))
        return out

    def post_exit(self, px: int, s_x: float, b12_x: float, f1e: float) -> dict:
        """What the spread and b12 did after the exit: to t-1 close and on expiry day."""
        x = self.s
        sc = (x.F1c - x.F2c).where((x.F1c > 0) & (x.F2c > 0))
        after_t1 = slice(px, self.p_t1_end + 1)
        r = dict(post_max_rise_t1_bps=(sc.iloc[after_t1].max() - s_x) / f1e * 1e4,
                 post_to_t1_close_bps=(sc.iloc[:self.p_t1_end + 1].dropna().iloc[-1] - s_x) / f1e * 1e4,
                 post_b12_max_rise_t1_pts=x.b12.iloc[after_t1].max() - b12_x)
        if self.t0 is not None:
            t0 = np.flatnonzero((x.day == self.t0).to_numpy())
            if len(t0):
                s0 = sc.iloc[t0]
                r.update(post_max_rise_t0_bps=(s0.max() - s_x) / f1e * 1e4,
                         post_to_t0_close_bps=(s0.dropna().iloc[-1] - s_x) / f1e * 1e4 if s0.notna().any() else np.nan,
                         post_b12_max_rise_t0_pts=x.b12.iloc[t0].max() - b12_x)
        return r


# ── driver ─────────────────────────────────────────────────────────────────────

def rules():
    out = []
    for w in ("t1", "all"):
        for pr in ("none", "gap>0"):
            for xx in FALL_X:
                out.append(("fall", xx, pr, w, f"fall x={xx} prior={pr} window={w}"))
        for th in DD_THETA:
            out.append(("dd", th, "none", w, f"dd θ={th} window={w}"))
    return out


def summarise(tr: pd.DataFrame) -> dict:
    n = len(tr)
    return dict(n=n, mean_bps=tr.pnl_bps.mean(), median_bps=tr.pnl_bps.median(),
                win=(tr.pnl_bps > 0).mean(), mean_net5=tr.net5_bps.mean(), mean_net10=tr.net10_bps.mean(),
                win_net10=(tr.net10_bps > 0).mean(), std_bps=tr.pnl_bps.std(),
                sharpe=tr.pnl_bps.mean() / tr.pnl_bps.std() if n > 1 and tr.pnl_bps.std() > 0 else np.nan,
                worst_bps=tr.pnl_bps.min(), best_bps=tr.pnl_bps.max(),
                signal_share=(tr.exit_reason == "signal").mean(), hold_h=tr.hold_h.mean(),
                b12_pts=tr.b12_change_pts.mean())


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("tickers", nargs="+")
    ap.add_argument("--threshold", type=float, default=0.30)
    ap.add_argument("--all-cycles", action="store_true", help="also run on every cycle (no HTB filter)")
    ap.add_argument("--out", type=Path, default=OUT, help=f"output folder (default {OUT})")
    args = ap.parse_args(argv)
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    store = ArcticStore()

    cycles, skipped, allc = [], [], []
    for t in (x.upper() for x in args.tickers):
        d = ticker_frame(store, t)
        lots = lot_sizes(store, t)
        sel = select_htb_expiries(d.exp, d.td, d.b12s, args.threshold, 4)
        exps = sorted({e for e in d.exp.unique() if len(e) == 10 and e[4] == "-"})
        known = sorted(lots)
        for E in exps:
            lot = lots.get(E) or (lots[max([k for k in known if k <= E], default=known[0])] if known else None)
            c = Cycle(d, t, E, lot)
            if E in set(sel.expiry):
                (cycles if c.ok else skipped).append(c if c.ok else (t, E, c.why))
            if c.ok:
                allc.append(c)
    print(f"HTB cycles usable: {len(cycles)} | skipped: {len(skipped)}")
    for t, E, why in skipped:
        print(f"   skipped {t} {E}: {why}")

    # grid
    rows, grid = [], []
    for method, p, pr, w, label in rules():
        tr = pd.DataFrame([c.trade(c.trigger(method, p, pr, w), label) for c in cycles])
        rows.append(tr)
        grid.append(dict(rule=label, method=method, param=p, prior=pr, window=w, **summarise(tr)))
    G = pd.DataFrame(grid).sort_values("mean_net10", ascending=False)
    T = pd.concat(rows)
    G.to_csv(out / "grid.csv", index=False)
    T.to_csv(out / "trades_grid.csv", index=False)

    # baselines
    base = {}
    for name, fn in {
        "hold to t-1 15:15": lambda c: c.trade(None, "hold to t-1 15:15", "forced 15:15"),
        "exit t-1 open": lambda c: c.trade(c.p_t1_start, "exit t-1 open", "t-1 open"),
        "exit t-2 close (t-1 open fill)": None,
    }.items():
        if fn:
            base[name] = pd.DataFrame([fn(c) for c in cycles])
    # oracle: best open price between entry and forced exit
    orc = []
    for c in cycles:
        x = c.s
        s_open = (x.F1o - x.F2o).where((x.F1o > 0) & (x.F2o > 0)).iloc[c.p_entry:c.p_forced + 1]
        orc.append(c.trade(int(s_open.idxmax()), "oracle best exit", "oracle"))
    base["oracle best exit (upper bound)"] = pd.DataFrame(orc)
    B = pd.DataFrame([dict(rule=k, **summarise(v)) for k, v in base.items()])
    B.to_csv(out / "baselines.csv", index=False)

    # leave-one-cycle-out choice of rule
    key = T.assign(cy=T.ticker + "|" + T.expiry)
    loco = []
    for cy in key.cy.unique():
        trn = key[key.cy != cy].groupby("rule").net10_bps.mean()
        best = trn.idxmax()
        loco.append(key[(key.cy == cy) & (key.rule == best)].iloc[0].to_dict())
    L = pd.DataFrame(loco)
    L.to_csv(out / "loco.csv", index=False)

    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40)
    print("\n== baselines")
    print(B[["rule", "n", "mean_bps", "median_bps", "win", "mean_net10", "win_net10", "worst_bps", "best_bps", "hold_h", "b12_pts"]].round(2).to_string(index=False))
    print("\n== top 15 rules by mean net (10 bps/leg/side)")
    print(G.head(15)[["rule", "n", "mean_bps", "median_bps", "win", "mean_net10", "win_net10", "worst_bps", "signal_share", "hold_h"]].round(2).to_string(index=False))
    print("\n== bottom 5 rules")
    print(G.tail(5)[["rule", "mean_bps", "win", "mean_net10"]].round(2).to_string(index=False))
    print("\n== leave-one-cycle-out (rule picked on the other cycles, tested on the held-out one)")
    print(summarise(L))
    print(L.rule.value_counts().head(5).to_string())
    if args.all_cycles:
        best = G.iloc[0]
        A = pd.DataFrame([c.trade(c.trigger(best.method, best.param, best.prior, best.window), best.rule) for c in allc])
        H = pd.DataFrame([c.trade(None, "hold", "forced 15:15") for c in allc])
        print(f"\n== ALL cycles (no HTB filter), n={len(allc)}: best rule {summarise(A)['mean_net10']:.1f} bps net10, "
              f"win {summarise(A)['win']:.0%} | hold-to-15:15 {summarise(H)['mean_net10']:.1f} bps net10, win {summarise(H)['win']:.0%}")
        A.to_csv(out / "trades_all_cycles_best_rule.csv", index=False)
        H.to_csv(out / "trades_all_cycles_hold.csv", index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
