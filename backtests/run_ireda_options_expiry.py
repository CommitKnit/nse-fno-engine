"""run_ireda_options_expiry.py — the b12 expiry strategy expressed with F1 options (IREDA).

Timing comes from the futures strategy (backtests/run_b12_long_short_final.py):
  CALL leg  : buy at the t−4 open; sell at the long-spread exit (b12 stop / gap_0.05_1 rule
              on t−1 / t−1 15:15).
  PUT leg   : buy at the scheduled long exit (gap rule or 15:15 — not the stop); sell on
              expiry day at 14:45.
Instruments (F1 expiry, strike fixed at entry from that bar's chain):
  ATM = strike nearest the futures price;  OTM = |delta| nearest 0.25 (call +0.25, put −0.25).
  Reported separately and as ATM+OTM (1 lot each).
Fills: buy at synthetic_ask, sell at synthetic_bid (close when missing); if the strike has no
row at the fill bar, the next bar that has one (same day). Costs: ₹20/order + 0.1% STT on sells.

Outputs: results/backtests/ireda_options_expiry/{trades.csv, summary.txt}
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backtests.run_b12_long_expiry import Cycle, ticker_frame  # noqa: E402
from backtests.run_b12_long_short_expiry import fill_upto  # noqa: E402
from backtests.stop_loss_study import leg, stop_exit  # noqa: E402
from data.storage.arctic_store import ArcticStore  # noqa: E402
from scripts.plot_htb_cycles import select_htb_expiries  # noqa: E402

OUT = ROOT / "results" / "backtests" / "ireda_options_expiry"
TICKER, BROKERAGE, STT_SELL = "IREDA", 20.0, 0.001


def pick(chain_bar: pd.DataFrame, kind: str, which: str) -> pd.Series | None:
    c = chain_bar[chain_bar.instrument_type == kind].dropna(subset=["strike_price"]).reset_index(drop=True)
    if c.empty:
        return None
    if which == "ATM":
        return c.loc[(c.strike_price - c.futures_price).abs().idxmin()]
    target = 0.25 if kind == "CE" else -0.25
    c = c.dropna(subset=["delta"])
    c = c[(c.delta > 0) if kind == "CE" else (c.delta < 0)]
    c = c[(c.strike_price > c.futures_price) if kind == "CE" else (c.strike_price < c.futures_price)]
    return None if c.empty else c.loc[(c.delta - target).abs().idxmin()]


def quote(opt: pd.DataFrame, strike: float, kind: str, ts: pd.Timestamp, side: str, until: pd.Timestamp):
    """(timestamp, price, row) of the first row of this strike at/after ts (same day, <= until)."""
    s = opt[(opt.strike_price == strike) & (opt.instrument_type == kind) & (opt.index >= ts) & (opt.index <= until)]
    s = s[s.index.normalize() == ts.normalize()] if len(s) else s
    if s.empty:
        return None
    r = s.iloc[0]
    px = r.synthetic_ask if side == "buy" else r.synthetic_bid
    if not (np.isfinite(px) and px > 0):
        px = r.close
    return s.index[0], float(px), r


def option_trade(opt, kind, which, t_in, t_out, day_end_out, lot_default):
    bar = opt[opt.index == t_in]
    k = 0
    while bar.empty and k < 12:                                 # first chain bar at/after entry
        k += 1
        t_in = t_in + pd.Timedelta(minutes=5)
        bar = opt[opt.index == t_in]
    if bar.empty:
        return None
    sel = pick(bar, kind, which)
    if sel is None:
        return None
    K = float(sel.strike_price)
    q_in = quote(opt, K, kind, t_in, "buy", t_in + pd.Timedelta(hours=1))
    q_out = quote(opt, K, kind, t_out, "sell", day_end_out)
    if q_in is None or q_out is None:
        return None
    (ti, pi, ri), (to, po, ro) = q_in, q_out
    lot = int(ri.lot_size) if np.isfinite(ri.lot_size) else lot_default
    gross = (po - pi) * lot
    cost = 2 * BROKERAGE + STT_SELL * po * lot
    return dict(kind=kind, which=which, strike=K, lot=lot, t_in=ti, t_out=to, prem_in=pi, prem_out=po,
                fut_in=float(ri.futures_price), fut_out=float(ro.futures_price), iv_in=float(ri.iv), iv_out=float(ro.iv),
                delta_in=float(ri.delta), pnl_gross=gross, pnl_net=gross - cost, ret_on_premium=(gross - cost) / (pi * lot))


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    store = ArcticStore()
    d = ticker_frame(store, TICKER)
    htb = select_htb_expiries(d.exp, d.td, d.b12s, 0.30, 4).expiry.tolist()
    chains = set(store.list_option_expiries(TICKER, "5minute"))
    rows, notes = [], []
    for E in htb:
        c = Cycle(d, TICKER, E, None)
        if not c.ok or c.t0 is None:
            notes.append(f"{E}: futures cycle unusable ({c.why or 'no expiry day'})")
            continue
        if E not in chains:
            notes.append(f"{E}: no option chain stored")
            continue
        opt = store.read_options(TICKER, "5minute", E).sort_index()
        x = c.s
        b12s = (x.timestamp.map(d.b12s) * 100).to_numpy()
        pe = c.p_entry
        trig = c.trigger("fall", 3, "gap>0", "t1")
        p_sched = c.fill(c.p_forced if trig is None else trig)[0]
        L = leg(c, pe, p_sched, x.F1o.iat[pe], x.F2o.iat[pe], +1, b12s)
        p_stop = stop_exit(L, "b12", 10, {})
        p_long_out = p_stop if p_stop is not None else p_sched
        t0 = np.flatnonzero((x.day == c.t0).to_numpy())
        le = [p for p in t0 if x.hm.iat[p] <= "14:45"]
        p_sx = fill_upto(c, le[-1], int(t0[-1]))[0]
        ts = x.timestamp
        filt = (b12s[p_sched] >= 30) and (b12s[p_sched] - b12s[pe] >= 10)
        reason = "b12 stop" if p_stop is not None else ("signal" if trig is not None else "t-1 15:15")
        day_end = lambda p: ts.iat[int(np.flatnonzero((x.day == x.day.iat[p]).to_numpy())[-1])]
        for which in ("ATM", "OTM"):
            r = option_trade(opt, "CE", which, ts.iat[pe], ts.iat[p_long_out], day_end(p_long_out), None)
            if r:
                rows.append(dict(expiry=E, leg="long CALL", exit_reason=reason, short_filter=filt, **r))
            r = option_trade(opt, "PE", which, ts.iat[p_sched], ts.iat[p_sx], day_end(p_sx), None)
            if r:
                rows.append(dict(expiry=E, leg="long PUT", exit_reason="t0 14:45", short_filter=filt, **r))
    T = pd.DataFrame(rows)
    T.to_csv(OUT / "trades.csv", index=False)

    out = [f"IREDA HTB cycles with option chains: {T.expiry.nunique()} | skipped: " + "; ".join(notes)]
    pd.set_option("display.width", 250)
    cols = ["expiry", "which", "strike", "t_in", "t_out", "prem_in", "prem_out", "fut_in", "fut_out", "iv_in", "iv_out",
            "delta_in", "pnl_net", "ret_on_premium"]
    for lg in ("long CALL", "long PUT"):
        g = T[T.leg == lg].copy()
        g["t_in"] = pd.to_datetime(g.t_in).dt.strftime("%d-%b %H:%M"); g["t_out"] = pd.to_datetime(g.t_out).dt.strftime("%d-%b %H:%M")
        out.append(f"\n== {lg} trades (₹ per lot, net)")
        out.append(g[cols].round(3).to_string(index=False))
    out.append("\n== summary (₹ per lot, net of ₹20/order + 0.1% STT on sells; fills at synthetic bid/ask)")
    for lg in ("long CALL", "long PUT"):
        for which in ("ATM", "OTM"):
            g = T[(T.leg == lg) & (T.which == which)]
            out.append(f"{lg} {which}: n={len(g)}  total ₹{g.pnl_net.sum():+,.0f}  mean ₹{g.pnl_net.mean():+,.0f}  median ₹{g.pnl_net.median():+,.0f}  "
                       f"win {(g.pnl_net > 0).mean():.0%}  worst ₹{g.pnl_net.min():+,.0f}  mean return on premium {g.ret_on_premium.mean():+.0%}  "
                       f"avg premium paid ₹{(g.prem_in * g.lot).mean():,.0f}")
        g = T[T.leg == lg].groupby("expiry").pnl_net.sum()
        out.append(f"{lg} ATM+OTM: n={len(g)}  total ₹{g.sum():+,.0f}  mean ₹{g.mean():+,.0f}  win {(g > 0).mean():.0%}  worst ₹{g.min():+,.0f}")
    gp = T[(T.leg == "long PUT") & T.short_filter].groupby("expiry").pnl_net.sum()
    out.append(f"long PUT ATM+OTM, only when the t-1 filter passes: n={len(gp)} total ₹{gp.sum():+,.0f} mean ₹{gp.mean():+,.0f}")
    cyc = T.groupby("expiry").pnl_net.sum()
    out.append(f"\nWHOLE STRATEGY (calls + puts, ATM+OTM, 1 lot each = 4 lots per cycle): total ₹{cyc.sum():+,.0f}, "
               f"mean ₹{cyc.mean():+,.0f} per cycle, win {(cyc > 0).mean():.0%}, worst ₹{cyc.min():+,.0f}")
    txt = "\n".join(out)
    (OUT / "summary.txt").write_text(txt, encoding="utf-8")
    print(txt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
