"""plot_htb_cycles.py — static per-cycle charts of hard-to-borrow (HTB) expiry cycles.

A contract expiry qualifies when its smoothed b12 (causal EWMA, 0.1-day half-life — the
same smoothing used in the research) reaches ``--threshold`` (default 30%) in the final
``--week-td`` + 1 trading days (default: last 5). One chart per qualifying expiry covers
the whole time that contract was front month: every bar after the previous expiry, up to
and including the expiry day.

Panels (shared trading-time x-axis: one step per bar, so nights, weekends and holidays
take no space and every line is continuous; final week shaded):
  1. b12 (%) — Gaussian-smoothed display line over faint raw b12, 30% line, peak marker,
     red ticks where the F1 or F2 close is missing
  2. basis (%) — spot − F1 and F1 − F2, same display smoothing
  3. open interest — F1 and F2 OI, OI concentration on the right axis
  4. roll flow — feat_flow_intensity and its 1-hour mean; green bands where the
     prototype "active roll" toggle is ON
  5. volume — F1 and F2 volume per hour, F2 volume share on the right axis

Bars where the F1 slot held a different contract (missing-contract problem) are hatched
and left out of the lines; the title then carries a warning.

Writes ``results/charts/htb_cycles/{TICKER}/{TICKER}_{expiry}.png``, one
``{TICKER}.pdf`` per ticker and a ``README.md`` index. Reads ArcticDB only.

Usage
-----
    plot-htb-cycles IREDA RVNL SBICARD --interval 5minute
    (or: python scripts/plot_htb_cycles.py ...)
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

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import transforms  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402

from data.storage.arctic_store import ArcticStore  # noqa: E402
from scripts.ewma_b12_signal import ewma_by_cycle, load  # noqa: E402

OUT_DIR = ROOT / "results" / "charts" / "htb_cycles"
SELECT_HL_DAYS = 0.1          # causal EWMA half-life used to select cycles (research convention)
SMOOTH_STD_BARS = 3.0         # display smoother: Gaussian sigma, in bars
SMOOTH_HALF_WIDTH = 6         # ... window = 2*6+1 = 13 bars (~1 hour at 5minute)
MAX_BRIDGE = 3                # display smoother bridges NaN runs of at most this many bars
# prototype "active roll" toggle
TOGGLE_HL_BARS, Z_ON, Z_OFF, ON_BARS, OFF_BARS, TOGGLE_MIN_BASE = 12, 2.0, 0.5, 3, 6, 500

C = {"b12": "#1f4e79", "raw": "#9db4cc", "basis1": "#7a3e9d", "basis2": "#2e8b57", "f1": "#1f4e79",
     "f2": "#d9822b", "conc": "#555555", "fi": "#2e8b57", "toggle": "#cfe8cf", "week": "#fff2cc",
     "wrong": "#f4c7c3", "miss": "#c0392b", "peak": "#c0392b"}


# ── pure helpers (unit-tested) ─────────────────────────────────────────────────

def display_smooth(x: pd.Series | np.ndarray, std: float = SMOOTH_STD_BARS,
                   half_width: int = SMOOTH_HALF_WIDTH, max_bridge: int = MAX_BRIDGE) -> np.ndarray:
    """Centred Gaussian-weighted mean (no lag). NaNs are skipped by renormalising the
    weights; NaN runs longer than *max_bridge* bars stay NaN, so nothing is drawn where
    data is truly missing."""
    v = np.asarray(x, dtype=float)
    ok = ~np.isnan(v)
    k = np.arange(-half_width, half_width + 1)
    w = np.exp(-0.5 * (k / std) ** 2)
    num = np.convolve(np.where(ok, v, 0.0), w, mode="same")
    den = np.convolve(ok.astype(float), w, mode="same")
    out = np.where(den > 1e-9, num / np.where(den > 1e-9, den, 1.0), np.nan)
    # blank NaN runs longer than max_bridge
    run_id = np.cumsum(ok)                        # constant inside a NaN run
    nan_runs = pd.Series(~ok).groupby(run_id).transform("sum").to_numpy()
    out[(~ok) & (nan_runs > max_bridge)] = np.nan
    return out


def roll_toggle(z: np.ndarray, ro: np.ndarray, cyc: np.ndarray, z_on: float = Z_ON, z_off: float = Z_OFF,
                on_bars: int = ON_BARS, off_bars: int = OFF_BARS) -> np.ndarray:
    """Hysteresis toggle: ON after *on_bars* bars with z >= z_on and ro > 0; OFF after
    *off_bars* bars with z < z_off; always OFF at a new cycle. NaN z keeps the state."""
    on = np.zeros(len(z), bool)
    state, up, down = False, 0, 0
    for i in range(len(z)):
        if i and cyc[i] != cyc[i - 1]:
            state, up, down = False, 0, 0
        zi = z[i]
        if not np.isnan(zi):
            if not state:
                up = up + 1 if (zi >= z_on and ro[i] > 0) else 0
                if up >= on_bars:
                    state, down = True, 0
            else:
                down = down + 1 if zi < z_off else 0
                if down >= off_bars:
                    state, up = False, 0
        on[i] = state
    return on


def expiry_window(index: pd.DatetimeIndex, expiries: list[str], expiry: str) -> np.ndarray:
    """Boolean mask: bars after the previous expiry's date, up to and including *expiry*'s date."""
    days = pd.Index(index.strftime("%Y-%m-%d"))
    prev = [e for e in sorted(expiries) if e < expiry]
    lo = prev[-1] if prev else ""
    return np.asarray((days > lo) & (days <= expiry), dtype=bool)


def select_htb_expiries(f1_expiry: pd.Series, td: pd.Series, b12_ewma: pd.Series,
                        threshold: float, week_td: int) -> pd.DataFrame:
    """Expiries whose smoothed b12 reaches *threshold* on final-week bars (td <= week_td)
    while that contract is in the F1 slot. Cycles split by missing data merge, because
    the grouping is by expiry, not cycle_id."""
    d = pd.DataFrame({"exp": f1_expiry.astype(str).str[:10], "td": td, "b": b12_ewma})
    w = d[(d.td <= week_td) & d.b.notna()]
    if w.empty:
        return pd.DataFrame(columns=["expiry", "peak", "peak_ts", "peak_td"])
    i = w.groupby("exp")["b"].idxmax()
    rows = list(i)                                  # keep tz-aware timestamps
    out = pd.DataFrame({"expiry": i.index, "peak": w.loc[rows, "b"].to_numpy(),
                        "peak_ts": rows, "peak_td": w.loc[rows, "td"].to_numpy()})
    return out[out.peak >= threshold].sort_values("expiry").reset_index(drop=True)


# ── data ───────────────────────────────────────────────────────────────────────

def ticker_frame(store: ArcticStore, ticker: str, interval: str) -> pd.DataFrame:
    br = store.read_borrow_rates(ticker, interval)
    br = br[~br.index.duplicated(keep="last")].sort_index()
    fut = store.read_futures(ticker, interval, with_spot=False, columns=["F1_expiry"])
    br["F1_expiry"] = fut["F1_expiry"][~fut.index.duplicated(keep="last")].reindex(br.index).astype(str).str[:10]
    br["td"] = load(store, ticker, interval, None, None)["td_to_exp"].reindex(br.index)
    cyc = br["cycle_id"]
    bars_per_day = 75 if interval == "5minute" else 25
    br["b12_ewma"] = ewma_by_cycle(br["b12"].astype(float), cyc, SELECT_HL_DAYS * bars_per_day)
    # prototype toggle: 1h EWMA of flow intensity, z vs this ticker's EARLIER cycles
    open_bar = br.index.strftime("%H:%M") == "09:15"            # overnight OI change
    # A leg with no candle has OI stored as 0; the next bar then shows its whole OI as one
    # bar of "flow". Hide flow on that bar and on the empty bar itself.
    oi_gap = (br["F1_oi"].astype(float) <= 0) | (br["F2_oi"].astype(float) <= 0)
    after_gap = oi_gap.groupby(cyc).shift(1, fill_value=False).astype(bool)
    br["flow_masked"] = oi_gap | after_gap
    hide = open_bar | br["flow_masked"].to_numpy()
    fi = br["feat_flow_intensity"].astype(float).where(~hide)
    ro = br["feat_relative_oi_change"].astype(float).where(~hide)
    br["fi_clean"] = fi
    s_fi = ewma_by_cycle(fi, cyc, TOGGLE_HL_BARS).groupby(cyc).ffill()
    s_ro = ewma_by_cycle(ro, cyc, TOGGLE_HL_BARS).groupby(cyc).ffill()
    first_ts = pd.Series(br.index, index=br.index).groupby(cyc).transform("min")
    order = first_ts.rank(method="dense").astype(int)
    med = pd.Series(np.nan, index=br.index)
    mad = pd.Series(np.nan, index=br.index)
    for k in sorted(order.unique()):
        past = s_fi[order < k].dropna()
        if len(past) >= TOGGLE_MIN_BASE:
            m = past.median()
            med[order == k], mad[order == k] = m, (past - m).abs().median() * 1.4826
    z = ((s_fi - med) / mad.replace(0, np.nan)).to_numpy()
    br["roll_on"] = roll_toggle(z, s_ro.fillna(0).to_numpy(), cyc.to_numpy())
    br["fi_1h"] = fi.groupby(cyc).transform(lambda s: s.rolling(12, min_periods=6).mean())
    return br


# ── plotting ───────────────────────────────────────────────────────────────────

def _runs(mask: np.ndarray):
    """(start, end) index pairs of True runs."""
    m = np.asarray(mask, bool)
    if not m.any():
        return []
    starts = np.flatnonzero(m & ~np.r_[False, m[:-1]])
    ends = np.flatnonzero(m & ~np.r_[m[1:], False])
    return list(zip(starts, ends))


def plot_cycle(br: pd.DataFrame, ticker: str, expiry: str, peak_row: pd.Series, expiries: list[str],
               threshold: float, week_td: int):
    d = br[expiry_window(br.index, expiries, expiry)].copy()
    n = len(d)
    x = np.arange(n)
    own = (d["F1_expiry"] == expiry).to_numpy()            # F1 slot holds this contract
    final_week = own & (d["td"] <= week_td).to_numpy()
    miss = d["missing_inputs"].fillna("").astype(str)
    f_missing = (miss.str.contains("F1") | miss.str.contains("F2")).to_numpy() & own
    days = d.index.strftime("%Y-%m-%d")
    sess_start = np.flatnonzero(np.r_[True, days[1:] != days[:-1]])
    wrong_days = len(set(days[~own]))

    fig, axes = plt.subplots(5, 1, figsize=(16, 13), sharex=True,
                             gridspec_kw={"height_ratios": [3.2, 1.6, 1.6, 1.6, 1.4], "hspace": 0.08})
    for ax in axes:
        for a, b in _runs(final_week):
            ax.axvspan(a - 0.5, b + 0.5, color=C["week"], zorder=0, lw=0)
        for a, b in _runs(~own):
            ax.axvspan(a - 0.5, b + 0.5, facecolor=C["wrong"], hatch="///", edgecolor="#d98880", lw=0, zorder=0.5)
        for i, s in enumerate(sess_start):
            new_week = i > 0 and d.index[s].isocalendar().week != d.index[sess_start[i - 1]].isocalendar().week
            ax.axvline(s - 0.5, color="#999999" if new_week else "#dddddd", lw=0.9 if new_week else 0.5, zorder=0.6)
        ax.grid(axis="y", color="#eeeeee", lw=0.6)
        ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)

    # 1. b12
    ax = axes[0]
    b12 = (d["b12"].astype(float) * 100).where(own)
    sm = display_smooth(b12)
    ax.plot(x, b12, color=C["raw"], lw=0.6, label="b12, 5-min raw")
    ax.plot(x, sm, color=C["b12"], lw=2.0, label="b12, smoothed (centred 1-hour Gaussian)")
    ax.axhline(threshold * 100, color="#888888", ls="--", lw=1, label=f"{threshold:.0%} HTB line")
    fw = np.where(final_week, sm, np.nan)
    peak_txt = f"peak b12 {peak_row.peak * 100:.0f}%"
    if np.isfinite(fw).any():
        p = int(np.nanargmax(fw))
        peak_txt = (f"peak b12 {fw[p]:.0f}% on {d.index[p]:%d-%b %H:%M} "
                    f"({int(d['td'].iloc[p])} TD to expiry)")
        ax.plot(p, fw[p], "o", color=C["peak"], ms=7, zorder=5)
        right = p > 0.75 * n
        ax.annotate(f"peak {fw[p]:.0f}%  ·  {d.index[p]:%d-%b %H:%M}  ·  {int(d['td'].iloc[p])} TD to expiry",
                    (p, fw[p]), xytext=(-10 if right else 10, 8), textcoords="offset points", fontsize=10,
                    color=C["peak"], fontweight="bold", ha="right" if right else "left")
    tr = transforms.blended_transform_factory(ax.transData, ax.transAxes)
    if f_missing.any():
        ax.vlines(x[f_missing], 0, 0.035, transform=tr, color=C["miss"], lw=0.8,
                  label="F1 or F2 close missing (b12 not computed)")
    ax.set_ylabel("b12 (% p.a.)")
    lo = np.nanpercentile(sm, 0.5) if np.isfinite(sm).any() else 0
    hi = np.nanmax(sm) if np.isfinite(sm).any() else threshold * 100
    ax.set_ylim(min(0, lo) - 2, max(hi, threshold * 100) * 1.15 + 2)
    ax.legend(loc="upper left", fontsize=8.5, frameon=False, ncol=2)

    # 2. basis
    ax = axes[1]
    ax.plot(x, display_smooth(d["spot_f1_pct"].astype(float).where(own)), color=C["basis1"], lw=1.4,
            label="spot − F1 (% of spot)")
    ax.plot(x, display_smooth(d["f1_f2_pct"].astype(float).where(own)), color=C["basis2"], lw=1.4,
            label="F1 − F2 (% of F1)")
    ax.axhline(0, color="#999999", lw=0.8)
    ax.set_ylabel("basis (%)")
    ax.legend(loc="best", fontsize=8.5, frameon=False, ncol=2)

    # 3. open interest
    ax = axes[2]
    f1oi = d["F1_oi"].astype(float).where(own & (d["F1_oi"] > 0)) / 1e6     # 0 = no print, not zero OI
    f2oi = d["F2_oi"].astype(float).where(own & (d["F2_oi"] > 0)) / 1e6
    ax.plot(x, f1oi, color=C["f1"], lw=1.3, label="F1 OI")
    ax.plot(x, f2oi, color=C["f2"], lw=1.3, label="F2 OI")
    ax.set_ylabel("OI (mn)")
    top = np.nanmax(np.r_[f1oi.to_numpy(), f2oi.to_numpy(), 0])
    ax.set_ylim(0, top * 1.3 if top > 0 else 1)
    ax2 = ax.twinx()
    ax2.plot(x, d["feat_oi_concentration"].astype(float).where(own), color=C["conc"], lw=1, ls=":",
             label="OI concentration F1/(F1+F2)")
    ax2.set_ylim(0, 1.3)
    ax2.set_ylabel("concentration")
    ax2.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax2.spines["top"].set_visible(False)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper left", fontsize=8.5, frameon=False, ncol=3)

    # 4. roll flow + toggle
    ax = axes[3]
    on = d["roll_on"].to_numpy() & own
    for a, b in _runs(on):
        ax.axvspan(a - 0.5, b + 0.5, color=C["toggle"], zorder=0.4, lw=0)
    fi = d["fi_clean"].astype(float).where(own)
    fi1 = d["fi_1h"].astype(float).where(own)
    ax.plot(x, fi, color=C["fi"], lw=0.4, alpha=0.35, label="flow intensity, 5-min")
    ax.plot(x, fi1, color=C["fi"], lw=1.5, label="flow intensity, 1-hour mean")
    ax.axhline(0, color="#999999", lw=0.8)
    if fi1.notna().any():
        q = np.nanpercentile(np.abs(fi.dropna()), 99) if fi.notna().any() else 0.01
        ax.set_ylim(-q, q * 1.3)
    ax.plot([], [], color=C["toggle"], lw=8, label="active-roll toggle ON (prototype)")
    ax.set_ylabel("net roll flow /\ntotal OI")
    ax.legend(loc="upper left", fontsize=8.5, frameon=False, ncol=3)

    # 5. volume per hour + F2 share
    ax = axes[4]
    hr = x // 12
    v1 = d["F1_volume"].astype(float).where(own).groupby(hr).sum(min_count=1)
    v2 = d["F2_volume"].astype(float).where(own).groupby(hr).sum(min_count=1)
    xc = v1.index.to_numpy() * 12 + 5.5
    ax.bar(xc, v1.to_numpy() / 1e6, width=11, color=C["f1"], alpha=0.75, label="F1 volume / hour")
    ax.bar(xc, v2.to_numpy() / 1e6, width=11, bottom=np.nan_to_num(v1.to_numpy() / 1e6), color=C["f2"],
           alpha=0.85, label="F2 volume / hour")
    ax.set_ylabel("volume (mn)")
    ax2 = ax.twinx()
    vs = d["feat_Vshare"].astype(float).where(own)
    ax2.plot(x, vs.rolling(12, min_periods=4).mean(), color="#333333", lw=1, ls="--", label="F2 volume share (1h)")
    ax2.set_ylim(0, 1.3)
    ax2.set_ylabel("F2 share")
    ax2.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax2.spines["top"].set_visible(False)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper left", fontsize=8.5, frameon=False, ncol=3)

    # session ticks: thin out to <= ~26 labels
    step = max(1, int(np.ceil(len(sess_start) / 26)))
    ticks = sess_start[::step]
    axes[-1].set_xticks(ticks)
    axes[-1].set_xticklabels([d.index[t].strftime("%a %d-%b") for t in ticks], rotation=60, ha="right", fontsize=8.5)
    axes[-1].set_xlim(-1, n)
    axes[-1].set_xlabel("trading time (5-minute bars; nights, weekends and holidays removed)  ·  "
                        "yellow = final 5 trading days  ·  hatched red = F1 slot held another contract")

    warn = f"   ⚠ F1 slot held a different contract on {wrong_days} day(s) — missing contract data" if wrong_days else ""
    fig.suptitle(f"{ticker}  ·  expiry {expiry}  ·  {peak_txt}{warn}",
                 x=0.06, ha="left", y=0.995, fontsize=14, fontweight="bold",
                 color="#a93226" if wrong_days else "#222222")
    fig.text(0.06, 0.968, f"{d.index[0]:%d %b %Y} → {d.index[-1]:%d %b %Y}  ·  {n:,} bars  ·  "
             f"{int(f_missing.sum())} bars with F1/F2 missing  ·  roll flow hidden on "
             f"{int((d['flow_masked'].to_numpy() & own).sum())} bars next to a missing OI print",
             fontsize=9.5, color="#555555")
    fig.subplots_adjust(left=0.06, right=0.94, top=0.955, bottom=0.1)
    return fig, wrong_days, int(f_missing.sum())


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("tickers", nargs="+", metavar="TICKER")
    ap.add_argument("--interval", default="5minute", choices=["5minute", "15minute"])
    ap.add_argument("--threshold", type=float, default=0.30, help="peak b12 in the final week (default 0.30)")
    ap.add_argument("--week-td", type=int, default=4, help="final week = td_to_exp <= this (default 4: last 5 sessions)")
    ap.add_argument("--out", type=Path, default=OUT_DIR)
    args = ap.parse_args(argv)

    store = ArcticStore()
    args.out.mkdir(parents=True, exist_ok=True)
    index_rows, none = [], []
    for t in (x.upper() for x in args.tickers):
        br = ticker_frame(store, t, args.interval)
        expiries = sorted({e for e in br["F1_expiry"].unique() if len(e) == 10 and e[4] == "-"})
        sel = select_htb_expiries(br["F1_expiry"], br["td"], br["b12_ewma"], args.threshold, args.week_td)
        if sel.empty:
            none.append(t)
            print(f"{t}: no expiry with final-week b12 >= {args.threshold:.0%} — no charts")
            continue
        (args.out / t).mkdir(exist_ok=True)
        with PdfPages(args.out / f"{t}.pdf") as pdf:
            for _, r in sel.iterrows():
                fig, wrong_days, n_miss = plot_cycle(br, t, r.expiry, r, expiries, args.threshold, args.week_td)
                png = args.out / t / f"{t}_{r.expiry}.png"
                fig.savefig(png, dpi=150)
                pdf.savefig(fig)
                plt.close(fig)
                index_rows.append(dict(ticker=t, expiry=r.expiry, peak=r.peak, peak_ts=r.peak_ts,
                                       peak_td=int(r.peak_td), wrong_days=wrong_days, n_miss=n_miss,
                                       png=f"{t}/{png.name}"))
        print(f"{t}: {len(sel)} chart(s) -> {args.out / t}  +  {t}.pdf")

    # index
    lines = ["# HTB cycle charts", "",
             f"Contract expiries whose smoothed b12 (causal EWMA, {SELECT_HL_DAYS}-day half-life) reached "
             f"**{args.threshold:.0%}** in the final {args.week_td + 1} trading days, {args.interval} data. "
             "One chart per expiry, covering the whole time that contract was front month. "
             f"Built by `plot-htb-cycles`.", "",
             "| Ticker | Expiry | Peak b12 (EWMA, used to select) | EWMA peak time | Days to expiry | Data warnings | Chart |",
             "|---|---|---|---|---|---|---|"]
    for r in index_rows:
        w = []
        if r["wrong_days"]:
            w.append(f"F1 slot held another contract on {r['wrong_days']} day(s)")
        if r["n_miss"]:
            w.append(f"{r['n_miss']} bars with F1/F2 missing")
        lines.append(f"| {r['ticker']} | {r['expiry']} | {r['peak']:.0%} | {pd.Timestamp(r['peak_ts']):%d-%b-%Y %H:%M} | "
                     f"{r['peak_td']} | {'; '.join(w) or '—'} | [png]({r['png']}) |")
    if none:
        lines += ["", f"No qualifying expiries: {', '.join(none)}."]
    lines += ["", "PDFs with every chart of a ticker: " + ", ".join(
        f"[{t}]({t}.pdf)" for t in dict.fromkeys(r["ticker"] for r in index_rows)) + "."]
    (args.out / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\n{len(index_rows)} charts, index -> {args.out / 'README.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
