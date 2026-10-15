"""run_roll_put_hedge.py — roll-spread put-hedge backtest on the engine (any symbol).

Long F2 put (ATM, or the nearest OTM strike with OI >= --min-oi), delta-hedged with
the F1 future, held through the final expiry week from the F1-F2 spread trough to
the week's spread maximum (LOOK-AHEAD by design — see
options/strategies/roll_spread_put_hedge.py). Hedge variants x {gross, net of
engine-default costs}.

Data (engine ArcticDB, --symbol / --interval, default IREDA / 5minute):
    futures/{SYMBOL}/{interval}                     F1/F2 wide panel
    options_enriched/{SYMBOL}/{interval}/{expiry}   fetch_options.py chains

Run:
    backtest-roll-put-hedge                                   # IREDA 5minute
    backtest-roll-put-hedge --symbol SBICARD --interval 15minute
    backtest-roll-put-hedge --final-week-td 5 --lots 1
    backtest-roll-put-hedge --strike otm_liquid --hedges relaxed,initial --out results/ireda_roll_put_hedge_otm
    (or: python backtests/run_roll_put_hedge.py ...)

Outputs: <out>/{cycles,paths,summary,windows,skipped}.csv
(default out: results/{symbol}_roll_put_hedge)
"""

from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

import pandas as pd
from loguru import logger

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.engine.backtest_engine import BacktestEngine
from core.orders.execution_simulator import CommissionModel
from data.fetchers.snapshot_builder import LocalSnapshotBuilder
from data.storage.arctic_store import ArcticStore
from options.pricing.dte_utils import nth_trading_day_before
from options.strategies.roll_spread_put_hedge import (
    RollCycle,
    RollSpreadPutHedgeConfig,
    RollSpreadPutHedgeStrategy,
)

DEFAULT_SYMBOL, DEFAULT_INTERVAL = "IREDA", "5minute"
# name -> (band on |net delta| / put shares, rehedge?). "initial" = entry hedge only.
HEDGES = {"tight": (0.02, True), "medium": (0.05, True), "relaxed": (0.10, True),
          "initial": (0.0, False)}
RESULTS = ROOT / "results"
ZERO_COST = CommissionModel(brokerage_per_order=0.0, stt_rate=0.0, exchange_rate=0.0,
                            gst_rate=0.0, stamp_duty_rate=0.0)


def lookahead_windows(store: ArcticStore, symbol: str, interval: str, from_d: str, to_d: str,
                      final_week_td: int) -> pd.DataFrame:
    """Per F1 expiry: entry = spread trough before the final week's max, exit = the max."""
    tz = "Asia/Kolkata"
    w = store.read_futures(symbol, interval, with_spot=False,
                           date_range=(pd.Timestamp(from_d, tz=tz),
                                       pd.Timestamp(to_d, tz=tz) + pd.Timedelta(days=1)))
    w = w.dropna(subset=["F1_close", "F2_close"])
    opt_exps = set(store.list_option_expiries(symbol, interval))
    rows = []
    for e1 in sorted({str(x)[:10] for x in w["F1_expiry"].dropna().unique()}):
        d1 = date.fromisoformat(e1)
        if not (date.fromisoformat(from_d) <= d1 <= date.fromisoformat(to_d)):
            continue
        lo = nth_trading_day_before(d1, final_week_td)
        wk = w[(w["F1_expiry"].astype(str).str[:10] == e1)
               & (w.index.date >= lo) & (w.index.date <= d1)]
        if wk.empty:
            continue
        e2 = str(wk["F2_expiry"].iloc[-1])[:10]
        s = wk["F1_close"] - wk["F2_close"]
        exit_ts = s.idxmax()
        before = s[s.index < exit_ts]
        row = {"f1_expiry": d1, "f2_expiry": date.fromisoformat(e2), "week_start": lo,
               "bars": len(wk), "spread_week_open": s.iloc[0], "spread_week_close": s.iloc[-1],
               "spread_max": s.max(), "exit_ts": exit_ts, "f2_options": e2 in opt_exps}
        if before.empty:
            row.update(entry_ts=pd.NaT, spread_trough=float("nan"),
                       skip="spread max on first bar of week")
        else:
            row.update(entry_ts=before.idxmin(), spread_trough=before.min(), skip="")
        if not row["f2_options"]:
            row["skip"] = "no F2 option chain in Arctic"
        rows.append(row)
    return pd.DataFrame(rows)


def run_band(snaps, cycles, band: float, rehedge: bool, lots: int, costs: bool,
             strike_mode: str, min_oi: int, interval: str) -> tuple[dict, RollSpreadPutHedgeStrategy]:
    strat = RollSpreadPutHedgeStrategy(
        RollSpreadPutHedgeConfig(cycles=cycles, band=band, rehedge=rehedge, lots=lots,
                                 interval=interval, strike_mode=strike_mode, min_oi=min_oi))
    eng = BacktestEngine(
        snaps, strat,
        initial_capital=100_000_000,       # engine NAV books full notional as cash spent;
        max_drawdown_pct=10.0,             # keep the drawdown gate from blocking hedges
        commission_model=None if costs else ZERO_COST,
        slippage_pct=0.001 if costs else 0.0,
        max_position_lots=10**9, max_open_positions=10,
        lot_size=1,                        # strategy emits share quantities
    )
    return eng.run(), strat


def engine_pnl_by_cycle(res: dict, windows: pd.DataFrame) -> pd.DataFrame:
    """Realised P&L and commissions from the engine's closed trades, per cycle."""
    trades = pd.DataFrame(res["closed_trades"])
    out = []
    for _, w in windows.iterrows():
        t = trades[(trades["closed_at"] >= w["entry_ts"]) & (trades["closed_at"] <= w["exit_ts"])]
        out.append({"f1_expiry": w["f1_expiry"],
                    "engine_realized": t["realized_pnl"].sum(),
                    "engine_commission": t["commission"].sum(),
                    "engine_net": t["realized_pnl"].sum() - t["commission"].sum()})
    return pd.DataFrame(out)


def iv_source(store: ArcticStore, symbol: str, interval: str, rec: pd.DataFrame) -> pd.DataFrame:
    """Tag where each put fill's IV came from: the strike's last enriched row at or
    before the fill — traded, put-call-parity repaired ("pcp"), or ffilled ("ffill")."""
    out = []
    for r in rec.drop_duplicates("f1_expiry").itertuples():
        o = store.read_options(symbol, interval, str(r.f2_expiry)).sort_index()
        o = o[(o["instrument_type"] == "PE") & (o["strike_price"] == r.strike)]
        tag = {"f1_expiry": r.f1_expiry}
        for leg, ts in (("entry", r.entry_ts), ("exit", r.exit_ts)):
            q = o[o.index <= ts]
            if q.empty:
                tag[f"iv_src_{leg}"], tag[f"iv_age_min_{leg}"] = "none", float("nan")
                continue
            last = q.iloc[-1]
            tag[f"iv_src_{leg}"] = ("pcp" if last["synthetic_via_putcallparity"]
                                    else "ffill" if last["is_synthetic"] else "traded")
            tag[f"iv_age_min_{leg}"] = (ts - q.index[-1]).total_seconds() / 60
        out.append(tag)
    return pd.DataFrame(out)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--symbol", default=DEFAULT_SYMBOL, help=f"Underlying (default {DEFAULT_SYMBOL}).")
    ap.add_argument("--interval", default=DEFAULT_INTERVAL,
                    help=f"Futures + options interval (default {DEFAULT_INTERVAL}).")
    ap.add_argument("--from", dest="from_d", default="2025-04-01")
    ap.add_argument("--to", dest="to_d", default="2026-06-30")
    ap.add_argument("--final-week-td", type=int, default=5,
                    help="Trading days counted back from (and incl.) F1 expiry (default 5).")
    ap.add_argument("--lots", type=int, default=1)
    ap.add_argument("--strike", choices=["atm", "otm_liquid"], default="atm")
    ap.add_argument("--min-oi", type=int, default=200_000,
                    help="OI floor (shares) for --strike otm_liquid (default 200000).")
    ap.add_argument("--hedges", default="tight,medium,relaxed",
                    help=f"Comma list from {list(HEDGES)} (default tight,medium,relaxed).")
    ap.add_argument("--contract-ffill-days", type=int, default=None,
                    help="Per-contract chain forward-fill window (days); defaults to 5 for "
                         "otm_liquid so every strike's last OI/IV is visible, else off.")
    ap.add_argument("--out", default=None,
                    help="Output folder (default results/{symbol}_roll_put_hedge).")
    args = ap.parse_args(argv)
    symbol, interval = args.symbol.upper(), args.interval
    out = Path(args.out) if args.out else RESULTS / f"{symbol.lower()}_roll_put_hedge"
    hedges = [h.strip() for h in args.hedges.split(",")]
    ffill = args.contract_ffill_days
    if ffill is None and args.strike == "otm_liquid":
        ffill = 5

    logger.remove()
    logger.add(sys.stderr, level="INFO", filter=lambda r: not r["name"].startswith("core."))
    out.mkdir(parents=True, exist_ok=True)

    builder = LocalSnapshotBuilder(symbol=symbol, futures_interval=interval,
                                   options_interval=interval, contract_ffill_days=ffill)
    windows = lookahead_windows(builder._store, symbol, interval, args.from_d, args.to_d,
                                args.final_week_td)
    windows.to_csv(out / "windows.csv", index=False)
    live = windows[windows["skip"] == ""].reset_index(drop=True)
    for _, w in windows[windows["skip"] != ""].iterrows():
        logger.warning(f"skip {w['f1_expiry']}: {w['skip']}")

    snaps = []
    for _, w in live.iterrows():
        s = builder.build(w["week_start"].isoformat(), w["f1_expiry"].isoformat())
        snaps.extend(s)
    cycles = [RollCycle(w.f1_expiry, w.f2_expiry, w.entry_ts.to_pydatetime(), w.exit_ts.to_pydatetime())
              for w in live.itertuples()]
    logger.info(f"{len(cycles)} cycles, {len(snaps)} snapshots")

    frames, paths, summary, skipped = [], [], [], []
    for band_name in hedges:
        band, rehedge = HEDGES[band_name]
        for costs in (False, True):
            res, strat = run_band(snaps, cycles, band, rehedge, args.lots, costs,
                                  args.strike, args.min_oi, interval)
            if not costs:
                skipped.extend({"hedge": band_name, **s} for s in strat.skipped)
            rec = pd.DataFrame(strat.records)
            rec = rec.merge(engine_pnl_by_cycle(res, live), on="f1_expiry", how="left")
            rec.insert(0, "costs", "net" if costs else "gross")
            rec.insert(0, "band", band_name)
            rec.insert(1, "band_delta", band)
            frames.append(rec)
            if not costs:
                p = pd.DataFrame(strat.paths)
                p.insert(0, "band", band_name)
                paths.append(p)
            summary.append({"band": band_name, "band_delta": band,
                            "costs": "net" if costs else "gross", **res["summary"],
                            "fills": res["execution_stats"]["fills"],
                            "rejected": res["risk_stats"]["rejected"]})

    cyc = pd.concat(frames, ignore_index=True)
    cyc = cyc.merge(iv_source(builder._store, symbol, interval, cyc), on="f1_expiry", how="left")
    cyc.to_csv(out / "cycles.csv", index=False)
    pd.concat(paths, ignore_index=True).to_csv(out / "paths.csv", index=False)
    pd.DataFrame(summary).to_csv(out / "summary.csv", index=False)
    pd.DataFrame(skipped).to_csv(out / "skipped.csv", index=False)

    view = cyc.pivot_table(index="f1_expiry", columns=["band", "costs"], values="engine_net")
    pd.set_option("display.width", 200)
    print("\nEngine net P&L per cycle (INR):\n", view.round(0))
    print("\nTotals:\n", view.sum().round(0))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
