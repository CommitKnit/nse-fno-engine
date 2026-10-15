"""fetch_kite_indices.py — NSE index OHLC history from Kite into the ``indices`` library.

Fetches the broad-market and sectoral Nifty indices at 15minute / 30minute /
60minute / day and writes each to ``indices/{NAME}/{interval}``.

Index history is bounded by NSE's own launch dates, not by the requested span:
Nifty Chemicals and several other recent indices simply do not have five years
behind them. Each symbol is fetched for whatever Kite serves and the actual
first bar is reported, rather than failing the run or silently padding.

Kite's per-request span caps (from the historical API docs) drive the chunking:
15/30minute 200 days, 60minute 400 days, day 2000 days.

Volume is dropped for the index series -- NSE publishes index candles with a
volume field that is either zero or a meaningless sum, and keeping it invites
somebody downstream to treat it as traded quantity.

Usage
-----
    fetch-kite-indices                    # 5 years, all indices
    fetch-kite-indices --years 3
    fetch-kite-indices --index NIFTY50 --index NIFTYBANK
    fetch-kite-indices --interval day --dry-run
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
from loguru import logger

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.settings import settings  # noqa: E402
from data.fetchers.kite_fetcher import KiteFetcher  # noqa: E402
from data.storage.arctic_store import ArcticStore  # noqa: E402

# canonical name -> Kite tradingsymbol in the NSE INDICES segment.
# Resolved to instrument_token at run time from the cached instrument dump so a
# token renumbering on Kite's side does not silently fetch the wrong series.
BROAD = {
    "NIFTY50": "NIFTY 50",
    "NIFTYNEXT50": "NIFTY NEXT 50",
    "NIFTY100": "NIFTY 100",
    "NIFTY200": "NIFTY 200",
    "NIFTY500": "NIFTY 500",
    "NIFTYMIDCAP150": "NIFTY MIDCAP 150",
    "NIFTYSMLCAP250": "NIFTY SMLCAP 250",
}
SECTORAL = {
    "NIFTYBANK": "NIFTY BANK",
    "NIFTYFINSERVICE": "NIFTY FIN SERVICE",
    "NIFTYAUTO": "NIFTY AUTO",
    "NIFTYIT": "NIFTY IT",
    "NIFTYPHARMA": "NIFTY PHARMA",
    "NIFTYFMCG": "NIFTY FMCG",
    "NIFTYMETAL": "NIFTY METAL",
    "NIFTYOILGAS": "NIFTY OIL AND GAS",
    "NIFTYREALTY": "NIFTY REALTY",
    "NIFTYPSUBANK": "NIFTY PSU BANK",
    "NIFTYPVTBANK": "NIFTY PVT BANK",
    "NIFTYCONSRDURBL": "NIFTY CONSR DURBL",
    "NIFTYHEALTHCARE": "NIFTY HEALTHCARE",
    "NIFTYCHEMICALS": "NIFTY CHEMICALS",
    "NIFTYMEDIA": "NIFTY MEDIA",
}
INDICES = {**BROAD, **SECTORAL}

# Requested but NOT carried in Kite's NSE INDICES dump. Reported, never
# substituted: NIFTY CAPITAL MKT is capital *markets* (brokers, exchanges,
# AMCs), a different index from capital goods, and NIFTY ENERGY blends power
# with oil & gas. Quietly swapping either one in would corrupt the series.
UNAVAILABLE = {
    "Nifty Capital Goods": "no NSE equivalent in Kite (NIFTY CAPITAL MKT is a different index)",
    "Nifty Cement": "not present in Kite's NSE INDICES segment",
    "Nifty Power": "not present in Kite's NSE INDICES segment (NIFTY ENERGY is oil & gas + power)",
}

# Kite historical API per-request span caps, in days.
CHUNK_DAYS = {"15minute": 200, "30minute": 200, "60minute": 400, "day": 2000}
INTERVALS = ("15minute", "30minute", "60minute", "day")


def resolve_tokens(names: dict[str, str]) -> dict[str, int]:
    """Map canonical name -> instrument_token via the newest cached NSE dump."""
    cache_dir = Path(settings.KITE_INSTRUMENTS_CACHE_DIR)
    dumps = sorted(cache_dir.glob("NSE_*.json"))
    if not dumps:
        raise SystemExit(f"No cached NSE instrument dump in {cache_dir}")
    rows = json.loads(dumps[-1].read_text())
    idx = {r["tradingsymbol"]: r for r in rows if r.get("segment") == "INDICES"}
    logger.info(f"instrument dump {dumps[-1].name}: {len(idx)} NSE indices")

    tokens, missing = {}, []
    for canon, tsym in names.items():
        if tsym in idx:
            tokens[canon] = int(idx[tsym]["instrument_token"])
        else:
            missing.append(f"{canon} ({tsym})")
    if missing:
        logger.error("not found in dump: " + ", ".join(missing))
    return tokens


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--years", type=int, default=5)
    ap.add_argument("--interval", action="append", dest="intervals",
                    choices=INTERVALS, help="repeatable; default all four")
    ap.add_argument("--index", action="append", dest="only",
                    help="canonical name, repeatable; default all")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--overwrite", action="store_true",
                    help="refetch symbols that already exist")
    args = ap.parse_args(argv)

    intervals = args.intervals or list(INTERVALS)
    wanted = {k: v for k, v in INDICES.items()
              if not args.only or k.upper() in {o.upper() for o in args.only}}
    if not wanted:
        logger.error("no matching index names")
        return 1

    to_date = date.today()
    from_date = to_date - timedelta(days=365 * args.years + 2)
    tokens = resolve_tokens(wanted)
    logger.info(f"{len(tokens)} indices x {len(intervals)} intervals | "
                f"{from_date} -> {to_date}")
    for name, why in UNAVAILABLE.items():
        logger.warning(f"SKIPPED {name}: {why}")

    if args.dry_run:
        for canon, tok in tokens.items():
            logger.info(f"  {canon:<18}{wanted[canon]:<22}{tok}")
        return 0

    store = ArcticStore()
    fetcher = KiteFetcher()
    fetcher.validate_session()

    report: list[dict] = []
    for canon, token in tokens.items():
        for interval in intervals:
            sym = store.indices_symbol(canon, interval)
            if not args.overwrite and store.has_symbol("indices", sym):
                logger.info(f"  {sym}: exists — skipping")
                continue
            df = fetcher.fetch_candles_chunked(
                token, interval, from_date, to_date,
                oi=False, max_days=CHUNK_DAYS[interval],
                instrument_key=f"KITE_NSE|{token}", exchange="NSE",
            )
            if df.empty:
                logger.warning(f"  {sym}: no data returned")
                report.append({"index": canon, "interval": interval, "rows": 0})
                continue
            # Index candles carry a meaningless volume field — drop it.
            df = df[[c for c in ("open", "high", "low", "close") if c in df.columns]]
            store.write_indices(canon, interval, df, prune_previous=True,
                                metadata={"source": "kite", "instrument_token": token,
                                          "tradingsymbol": wanted[canon],
                                          "requested_from": str(from_date)})
            logger.info(f"  {sym}: {len(df)} rows  "
                        f"{df.index[0]:%Y-%m-%d} -> {df.index[-1]:%Y-%m-%d}")
            report.append({"index": canon, "interval": interval, "rows": len(df),
                           "first": df.index[0].date(), "last": df.index[-1].date()})

    rep = pd.DataFrame(report)
    if not rep.empty:
        out = ROOT / "results" / "kite_indices_coverage.csv"
        out.parent.mkdir(exist_ok=True)
        rep.to_csv(out, index=False)
        logger.info(f"{fetcher.request_count} API requests | coverage -> {out}")
        short = rep[(rep.rows > 0) & (pd.to_datetime(rep["first"]).dt.date > from_date +
                                      timedelta(days=90))]
        if not short.empty:
            logger.warning("indices with less than the requested history:")
            for _, r in short.drop_duplicates("index").iterrows():
                logger.warning(f"    {r['index']}: starts {r['first']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
