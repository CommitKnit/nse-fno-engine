"""fetch_kite_equities.py -- generic NSE equity fetch via Kite Connect.

Fills the gap identified in this session's Kite-infra audit: KiteFetcher's
candle-fetching mechanics (``fetch_candles_chunked``, already defaulting to
``continuous=False``) were already equity-ready, but nothing resolved an NSE
ticker to a Kite ``instrument_token`` -- ``resolve_front_month`` is futures-only.
``KiteFetcher.resolve_equity``/``resolve_equities`` (added alongside this
script) close that gap.

Deliberately NOT built on ``fetch_mcx_futures.py``'s scaffolding -- that
script's core complexity (``continuous=1``, roll handling, splicing a Kite
continuous chain onto a separate Upstox live-contract leg) exists solely
because futures expire and roll. Equities never do, so none of that applies;
this is a plain per-symbol chunked fetch.

Output: ``spot/{TICKER}/{interval}`` with symbol metadata ``source='fetch_kite_equities'``
-- that stamp is what makes ``ArcticStore`` accept the series as Kite spot (borrow
rates, SPOT_* joins). Kite interval names are used as-is (``minute``, ``5minute``,
``15minute``, ``day`` ...); ``read_futures`` maps ``1minute``/``1day`` onto them.

Usage
-----
    fetch-kite-equities HDFCBANK --dry-run
    fetch-kite-equities HDFCBANK --intervals day
    fetch-kite-equities HDFCBANK TATASTEEL RELIANCE --intervals day 60minute
    fetch-kite-equities --tickers-file nifty500.txt --intervals day
"""

from __future__ import annotations

import argparse
import sys
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
from arcticdb_ext.version_store import StreamDescriptorMismatch
from kiteconnect.exceptions import PermissionException, TokenException
from loguru import logger

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.fetchers.kite_fetcher import KiteAuthError, KiteFetcher  # noqa: E402
from data.storage.arctic_store import ArcticStore  # noqa: E402

_DEFAULT_OUT_DIR = ROOT / "kite equities"
_DEFAULT_INTERVALS = ["day"]

# Kite's own interval strings (distinct from Upstox's "1minute"/"1hour" naming)
# and their documented per-request day-window ceilings (Kite developer forum,
# cross-checked this session -- NOT independently probed against this account
# the way the MCX day-limit was; see module note below).
KITE_INTERVALS = {
    "minute": 60, "3minute": 100, "5minute": 100, "10minute": 100,
    "15minute": 200, "30minute": 200, "60minute": 400, "day": 2000,
}

_DEFAULT_LOOKBACK_DAYS = 5 * 365


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Fetch NSE equity candles via Kite Connect (plain, non-continuous)."
    )
    ap.add_argument("tickers", nargs="*", help="Trading symbols, e.g. HDFCBANK TATASTEEL")
    ap.add_argument("--tickers-file", type=Path, default=None,
                    help="Path to a newline-separated list of trading symbols (for the full Nifty 500 run)")
    ap.add_argument("--exchange", default="NSE")
    ap.add_argument("--intervals", nargs="+", default=_DEFAULT_INTERVALS,
                    choices=list(KITE_INTERVALS), metavar="INTERVAL")
    ap.add_argument("--from", dest="from_date", type=date.fromisoformat, default=None, metavar="YYYY-MM-DD")
    ap.add_argument("--to", dest="to_date", type=date.fromisoformat, default=date.today(), metavar="YYYY-MM-DD")
    ap.add_argument("--out-dir", type=Path, default=_DEFAULT_OUT_DIR,
                    help="Only used with --dest parquet")
    ap.add_argument("--dest", choices=["arctic", "parquet"], default="arctic",
                    help="Where to write: ArcticDB's existing 'spot' library ({ticker}/{interval} "
                         "symbols, matching its established schema), or flat parquet files.")
    ap.add_argument("--force", action="store_true",
                    help="Arctic dest only: full overwrite (prune_previous) instead of "
                         "incremental append for symbols that already exist")
    ap.add_argument("--dry-run", action="store_true", help="Resolve tickers and print, no candle fetch")
    args = ap.parse_args(argv)

    tickers = list(args.tickers)
    if args.tickers_file:
        tickers += [
            line.strip() for line in args.tickers_file.read_text().splitlines()
            if line.strip() and not line.strip().startswith("#")
        ]
    if not tickers:
        logger.error("No tickers given (positional args or --tickers-file)")
        return 2
    tickers = sorted(set(t.upper() for t in tickers))

    try:
        fetcher = KiteFetcher()
    except KiteAuthError as exc:
        logger.error(str(exc))
        return 1

    logger.info(f"Resolving {len(tickers)} ticker(s) on {args.exchange} ...")
    resolved = fetcher.resolve_equities(tickers, exchange=args.exchange)
    missing = [t for t in tickers if t not in resolved]
    for t in tickers:
        if t in resolved:
            logger.info(f"  {t:12s} -> instrument_token={resolved[t]['instrument_token']}")
    if missing:
        logger.error(f"  Unresolved: {missing}")

    if args.dry_run:
        logger.info("Dry run -- no candle fetch.")
        return 0 if resolved else 2

    if not resolved:
        return 2

    store = ArcticStore() if args.dest == "arctic" else None
    if args.dest == "parquet":
        args.out_dir.mkdir(exist_ok=True)
    from_date_default = args.to_date - timedelta(days=_DEFAULT_LOOKBACK_DAYS)

    summary = []
    for ticker, row in resolved.items():
        token = row["instrument_token"]
        for interval in args.intervals:
            max_days = KITE_INTERVALS[interval]
            from_date = args.from_date or from_date_default
            logger.info(f"  {ticker:12s} {interval:9s} fetching [{from_date} .. {args.to_date}] "
                        f"(chunked <= {max_days}d, exchange={args.exchange}) ...")
            try:
                df = fetcher.fetch_candles_chunked(
                    token, interval, from_date, args.to_date,
                    continuous=False, oi=False, max_days=max_days,
                    instrument_key=f"KITE_{args.exchange}|{token}",
                )
            except (TokenException, PermissionException) as exc:
                # The access token died mid-run (expired, or a fresh session
                # was started elsewhere and invalidated this one -- Kite
                # allows one active session per app+user). Every subsequent
                # request will fail identically, so stop now rather than
                # grinding through the rest of the universe logging the same
                # fatal error hundreds of times.
                logger.error(
                    f"    {ticker} {interval}: {type(exc).__name__}: {exc}\n"
                    "AUTH FAILURE -- stopping. Run `kite-login` and then "
                    "`kite-login --request-token <token>`, and restart this fetch; "
                    "already-written symbols are untouched."
                )
                n_ok = sum(1 for *_, status in summary if status not in ("no data",) and not status.startswith("error"))
                logger.info(f"\n{n_ok} symbol/interval combos written before the auth failure.")
                return 1
            except Exception as exc:
                # One bad ticker (delisted, renamed, a transient fault that
                # outlasted the SDK's own retry budget) must not abandon the
                # remaining several hundred tickers in an unattended run --
                # same posture as fetch_nse_corporate_actions.py's per-ticker
                # isolation. InputException (e.g. a stale/renamed instrument
                # token) is NOT retried inside fetch_candles_chunked (see
                # KiteFetcher._historical_request), so it lands here.
                logger.error(f"    {ticker} {interval}: {type(exc).__name__}: {exc}")
                summary.append((ticker, interval, 0, f"error: {type(exc).__name__}"))
                continue
            if df.empty:
                logger.warning(f"    {ticker} {interval}: 0 rows returned")
                summary.append((ticker, interval, 0, "no data"))
                continue

            if store is not None:
                # Fresh symbols get the cleaner convention (no `oi` -- equities
                # never have open interest; confirmed on ABREL/1hour earlier
                # this session). Pre-existing symbols are NOT uniformly
                # schema'd though (discovered mid-run: ASTRAL/day has an `oi`
                # column of all-zero int64, which ArcticDB itself re-infers as
                # a string type on any attempt to reconstruct it here --
                # BAJAJ-AUTO/day has no `oi` at all). Rather than special-case
                # every possible legacy variant, `append` failing on a schema
                # mismatch falls back to a full `write` below, which never
                # destroys the old data (Arctic keeps it as a prior version;
                # only the "current" pointer moves) -- the right call anyway
                # once new data fully supersedes a small legacy artifact
                # (ASTRAL/day was 58 rows; this fetch is the real 1,242-row
                # 5-year series).
                df = df.drop(columns=["oi"], errors="ignore")
                already = store.has_symbol("spot", f"{ticker}/{interval}")

                meta = {
                    "source": "fetch_kite_equities", "symbol": ticker, "interval": interval,
                    "exchange": args.exchange, "instrument_token": token,
                    "fetched_at": date.today().isoformat(),
                }
                if already and not args.force:
                    try:
                        n = store.append_spot(ticker, interval, df, metadata=meta)
                        logger.info(f"    {ticker} {interval}: appended {n} new row(s) -> spot/{ticker}/{interval}")
                        summary.append((ticker, interval, n, "appended"))
                    except StreamDescriptorMismatch:
                        logger.warning(f"    {ticker} {interval}: schema mismatch vs pre-existing symbol "
                                       f"-- falling back to full write (old version kept in history, not deleted)")
                        store.write_spot(ticker, interval, df, prune_previous=False, metadata=meta)
                        logger.info(f"    {ticker} {interval}: wrote {len(df)} rows "
                                    f"[{df.index[0].date()} .. {df.index[-1].date()}] -> spot/{ticker}/{interval}")
                        summary.append((ticker, interval, len(df), "written (schema fallback)"))
                else:
                    store.write_spot(ticker, interval, df, prune_previous=args.force, metadata=meta)
                    logger.info(f"    {ticker} {interval}: wrote {len(df)} rows "
                                f"[{df.index[0].date()} .. {df.index[-1].date()}] -> spot/{ticker}/{interval}")
                    summary.append((ticker, interval, len(df), "written"))
            else:
                dest = args.out_dir / f"{ticker}_{interval}.parquet"
                df.to_parquet(dest)
                logger.info(f"    {ticker} {interval}: {len(df)} rows "
                            f"[{df.index[0].date()} .. {df.index[-1].date()}] -> {dest.name}")
                summary.append((ticker, interval, len(df), "written"))

    n_ok = sum(1 for *_, status in summary if status != "no data")
    n_empty = len(summary) - n_ok
    logger.info(f"\n{n_ok} symbol/interval combos written or appended, {n_empty} returned no data, "
                f"{len(missing)} tickers unresolved")
    logger.info(f"Total Kite requests used: {fetcher.request_count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
