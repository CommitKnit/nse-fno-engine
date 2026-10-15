"""build_gap_timeseries.py — per-ticker CSV of b12 and its fast-minus-slow EWMA gaps.

For each ticker it reads ``borrow_rates/{TICKER}/{interval}`` and writes
``results/ewma_b12/gap_timeseries/{ticker}_gap_timeseries.csv`` with one row per bar:

  b12_pts          100 x b12 (NaN when the F1 or F2 close is missing — never filled)
  gap_0.1_2        EWMA(b12_pts, half-life 0.1 trading day) − EWMA(…, 2 days)
  gap_0.1_1        EWMA(0.1 d) − EWMA(1 d)
  gap_0.05_1       EWMA(0.05 d) − EWMA(1 d)
  gap_0.05_1_acc   second difference of gap_0.05_1 within the cycle
  data_missing     True when the F1, F2 or spot close is missing on the bar
  missing_inputs   which ones, e.g. "F2" or "F1,F2,SPOT"
  feat_oi_concentration … feat_roll_deviation
                   the 7 OI roll-flow features, as stored in borrow_rates (build-features)

EWMAs restart at every roll (cycle_id) and are NaN on any bar whose b12 is NaN; the
next valid bar continues from the state before the gap (see ``ewma_by_cycle``).
Floats are written at full precision and dates as ISO ``YYYY-MM-DD``.

Usage
-----
    build-gap-timeseries IREDA RVNL --interval 5minute
    (or: python scripts/build_gap_timeseries.py ...)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # noqa: BLE001
    pass

from data.storage.arctic_store import ArcticStore  # noqa: E402
from scripts.ewma_b12_signal import ewma_by_cycle, halflife_to_alpha, load  # noqa: E402

OUT_DIR = ROOT / "results" / "ewma_b12" / "gap_timeseries"
GAPS = {"gap_0.1_2": (0.1, 2.0), "gap_0.1_1": (0.1, 1.0), "gap_0.05_1": (0.05, 1.0)}
ACC_OF = "gap_0.05_1"
# OI roll-flow features, copied as stored in borrow_rates (built by build-features)
ROLL_FEATURES = ["feat_oi_concentration", "feat_roll_velocity", "feat_roll_acceleration",
                 "feat_relative_oi_change", "feat_net_roll_flow", "feat_flow_intensity",
                 "feat_roll_deviation"]
COLUMNS = (["timestamp", "cycle_id", "F1_expiry", "td_to_exp", "b12_pts"] + list(GAPS)
           + [f"{ACC_OF}_acc", "F1_volume", "F2_volume", "both_traded", "f2_thin_early",
              "data_missing", "missing_inputs"] + ROLL_FEATURES)


def gap_frame(b12_pts: pd.Series, cycle: pd.Series, interval: str) -> pd.DataFrame:
    """The gap columns for one ticker. NaN wherever b12_pts is NaN; nothing crosses a roll."""
    hls = sorted({hl for pair in GAPS.values() for hl in pair})
    ew = {hl: ewma_by_cycle(b12_pts, cycle, halflife_to_alpha(hl, interval)[0]) for hl in hls}
    out = pd.DataFrame({name: ew[fast] - ew[slow] for name, (fast, slow) in GAPS.items()},
                       index=b12_pts.index)
    first = out[ACC_OF].groupby(cycle, sort=False).diff()
    out[f"{ACC_OF}_acc"] = first.groupby(cycle, sort=False).diff()
    return out


def build(store: ArcticStore, ticker: str, interval: str) -> pd.DataFrame:
    df = load(store, ticker, interval, None, None)
    if df.empty:
        return df
    br = store.read_borrow_rates(ticker, interval)
    br = br[~br.index.duplicated(keep="last")].reindex(df.index)
    for c in ("data_missing", "missing_inputs"):
        if c not in br.columns:
            raise SystemExit(f"{ticker}: borrow_rates/{ticker}/{interval} has no '{c}' column — "
                             f"rebuild it first (fetch-ticker {ticker} --history --rebuild-from-stored)")
    lacking = [c for c in ROLL_FEATURES if c not in br.columns]
    if lacking:
        raise SystemExit(f"{ticker}: borrow_rates/{ticker}/{interval} lacks {lacking} — "
                         f"build them first (build-features {ticker} --interval {interval})")
    exp = store.read_futures(ticker, interval, with_spot=False, columns=["F1_expiry"])["F1_expiry"]
    exp = exp[~exp.index.duplicated(keep="last")].reindex(df.index).astype(str).str[:10]

    b12_pts = df["b12"] * 100
    out = pd.DataFrame(index=df.index)
    out["timestamp"] = df.index.astype(str)          # 2025-02-28 09:15:00+05:30
    out["cycle_id"] = br["cycle_id"].astype("int64")
    out["F1_expiry"] = exp
    out["td_to_exp"] = df["td_to_exp"].astype("Int64")
    out["b12_pts"] = b12_pts
    out = out.join(gap_frame(b12_pts, df["cycle"], interval))
    out["F1_volume"] = br["F1_volume"].astype("int64")
    out["F2_volume"] = br["F2_volume"].astype("int64")
    for c in ("both_traded", "f2_thin_early", "data_missing"):
        out[c] = br[c].astype(bool)
    out["missing_inputs"] = br["missing_inputs"].fillna("").astype(str)
    for c in ROLL_FEATURES:
        out[c] = br[c].astype("float64")
    return out[COLUMNS]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("tickers", nargs="+", metavar="TICKER")
    ap.add_argument("--interval", default="5minute", choices=["5minute", "15minute"])
    ap.add_argument("--out", type=Path, default=OUT_DIR, help=f"output folder (default {OUT_DIR})")
    args = ap.parse_args(argv)

    store = ArcticStore()
    args.out.mkdir(parents=True, exist_ok=True)
    for t in (x.upper() for x in args.tickers):
        df = build(store, t, args.interval)
        if df.empty:
            print(f"{t}: no borrow_rates/{t}/{args.interval} — skipped")
            continue
        path = args.out / f"{t.lower()}_gap_timeseries.csv"
        df.to_csv(path, index=False, float_format=None)
        nan_gap = int(df[list(GAPS)].isna().any(axis=1).sum())
        print(f"{t:<11} {len(df):>6} rows  {df['data_missing'].sum():>5} flagged  "
              f"{nan_gap:>5} with NaN gaps  -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
