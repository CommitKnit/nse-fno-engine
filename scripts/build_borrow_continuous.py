"""Build continuous b1/b2/b12 series across cycles and store them in ArcticDB.

Reads ``borrow_rates/{T}/{iv}`` (joined with its futures panel + Kite spot) and
writes ``borrow_rates/{T}/{iv}/continuous`` — see data/borrow_continuous.py for the
roll-stitched and constant-maturity definitions.

    build-borrow-continuous IREDA
    build-borrow-continuous IREDA --intervals 5minute 15minute 1day
    (or: python scripts/build_borrow_continuous.py ...)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.borrow_continuous import NEAR_EXPIRY_DAYS, R, TENORS_DAYS, build_continuous
from data.storage.arctic_store import LIB_BORROW, ArcticStore


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("symbol")
    ap.add_argument("--intervals", nargs="+", default=["15minute", "1day"])
    args = ap.parse_args(argv)
    store = ArcticStore()
    for iv in args.intervals:
        br = store.read_borrow_rates(args.symbol, iv, joined=True)
        out = build_continuous(br)
        sym = f"{store.borrow_symbol(args.symbol, iv)}/continuous"
        store.write(LIB_BORROW, sym, out, metadata={
            "source": f"{LIB_BORROW}/{store.borrow_symbol(args.symbol, iv)} (joined)",
            "builder": "data/borrow_continuous.py", "r": R, "tenors_days": list(TENORS_DAYS),
            "near_expiry_days": NEAR_EXPIRY_DAYS,
            "columns": {
                "b1,b2,b12": "stored front series (clipped >= 0), stitched across cycles",
                "b1_ex_expiry": f"b1 masked where F1_dte_star <= {NEAR_EXPIRY_DAYS}",
                "b1_cm30": "constant-maturity spot->30d borrow (unclipped)",
                "b2_cm60": "constant-maturity spot->60d borrow (unclipped)",
                "b12_cm30_60": "constant-maturity 30d->60d forward borrow (unclipped)",
            }})
        print(f"{LIB_BORROW}/{sym}: {len(out):,} rows  {out.index.min()} -> {out.index.max()}  "
              f"cycles {out['cycle'].min()}-{out['cycle'].max()}  rolls {int(out['roll'].sum())}")
        for c in ("b1", "b1_ex_expiry", "b1_cm30", "b2", "b2_cm60", "b12", "b12_cm30_60"):
            s = out[c]
            print(f"  {c:13s} non-null {s.notna().mean():6.1%}  median {np.nanmedian(s):7.3f}  "
                  f"p5 {np.nanpercentile(s, 5):7.3f}  p95 {np.nanpercentile(s, 95):7.3f}  "
                  f"max {np.nanmax(s):7.3f}  last valid {s.last_valid_index()}")


if __name__ == "__main__":
    main()
