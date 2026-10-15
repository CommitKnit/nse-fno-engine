"""kite_fetcher.py -- Zerodha Kite Connect fetcher for historical candles.

Exists because Upstox's ``/v2/expired-instruments/*`` API family is NSE-only:
every MCX key format is rejected (live ``MCX_FO|...`` keys return zero expiries;
``MCX_COM|...`` underlying keys return "Invalid Instrument key"). Kite replaces
that whole concept with a single ``continuous=1`` flag on a *live* contract's
token, which is documented to cover NFO and MCX futures.

Deliberately NOT a ``BaseFetcher`` subclass
-------------------------------------------
``BaseFetcher``'s five abstract methods are shaped around Upstox's model:
they take ``instrument_key: str`` (Kite takes ``instrument_token: int``), and
three of them (``fetch_expiries``, ``fetch_option_contracts``,
``fetch_expired_candles``) have no Kite analogue -- Kite's instrument dump only
carries *live* contracts, and ``fetch_expired_candles(expired_key, ...)`` is
precisely the operation ``continuous=1`` replaces, on a *live* token. Subclassing
would mean four ``NotImplementedError`` stubs and a signature that lies. Return
types are kept aligned (``list[Bar]``) so callers can duck-type both fetchers.

Auth (daily)
------------
Kite access tokens expire daily (~06:00 IST). Each trading day:

    kite-login                                   # prints the Kite login URL
    kite-login --request-token <token or URL>    # after logging in in the browser

``exchange_request_token`` turns the manually supplied ``request_token`` into the
day's access token (``generate_session``) and saves it as ``KITE_ACCESS_TOKEN`` in
this repo's ``.env`` (``settings.KITE_ENV_PATH``); ``resolve_kite_access_token``
reads it from there on every later run.
"""

from __future__ import annotations

import json
import time
from datetime import date, datetime, time as dtime, timedelta
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import pandas as pd
from dotenv import dotenv_values, set_key
from loguru import logger
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from config.settings import settings
from data.models.bar import Bar

try:
    from kiteconnect import KiteConnect
    from kiteconnect.exceptions import (
        DataException,
        InputException,
        NetworkException,
        PermissionException,
        TokenException,
    )
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "The `kiteconnect` package is required. Install it with:\n"
        "    pip install kiteconnect"
    ) from exc


IST = "Asia/Kolkata"

# MCX sessions run ~09:00-23:30 IST. Used to build the naive datetimes the SDK
# requires (see _KiteRateLimiter docstring / module notes on the type check).
_SESSION_OPEN = dtime(9, 0, 0)
_SESSION_CLOSE = dtime(23, 59, 59)


class KiteAuthError(RuntimeError):
    """Raised when no usable Kite access token is available (missing or expired)."""


def _login_hint() -> str:
    return (
        "Run:  kite-login   (prints the login URL), then\n"
        "      kite-login --request-token <request_token or redirect URL>\n"
        "  (Kite access tokens expire daily at ~06:00 IST, so this must be "
        "re-run each trading day.)"
    )


def resolve_kite_access_token(access_token: str | None = None) -> str:
    """Resolve a Kite access token, in precedence order.

    1. explicit ``access_token`` argument
    2. ``KITE_ACCESS_TOKEN`` inside ``settings.KITE_ENV_PATH`` -- the repo ``.env``
       that ``kite-login`` writes to (re-read from disk, so a login done after this
       process started is still seen)
    3. ``settings.KITE_ACCESS_TOKEN`` (shell environment / ``.env`` at startup)

    Raises:
        KiteAuthError: if none of the three sources yields a token.
    """
    if access_token:
        return access_token

    env_path = Path(settings.KITE_ENV_PATH)
    if env_path.exists():
        token = (dotenv_values(env_path) or {}).get("KITE_ACCESS_TOKEN")
        if token:
            _warn_if_token_looks_stale(env_path)
            return token

    if settings.KITE_ACCESS_TOKEN:
        return settings.KITE_ACCESS_TOKEN

    raise KiteAuthError(
        "No Kite access token found.\n"
        f"  Checked: explicit arg, settings.KITE_ACCESS_TOKEN, {env_path}\n"
        f"  {_login_hint()}"
    )


def _warn_if_token_looks_stale(env_path: Path) -> None:
    """Cheap offline staleness heuristic -- no network call.

    Kite tokens die at ~06:00 IST. If the env file was last written before the
    most recent 06:00 IST boundary, the token in it is almost certainly dead.
    Warning only; ``KiteFetcher.validate_session()`` is the authoritative check.
    """
    try:
        written = datetime.fromtimestamp(env_path.stat().st_mtime)
    except OSError:
        return

    now = datetime.now()
    boundary = now.replace(hour=6, minute=0, second=0, microsecond=0)
    if now < boundary:
        boundary -= timedelta(days=1)

    if written < boundary:
        age_days = (now - written).days
        logger.warning(
            f"Kite token in {env_path.name} was written {written:%Y-%m-%d %H:%M} "
            f"({age_days}d ago), before the most recent 06:00 IST expiry boundary "
            f"-- it is probably expired.\n  {_login_hint()}"
        )


class _KiteRateLimiter:
    """Monotonic-spacing limiter for Kite's flat 3 req/sec historical cap.

    Deliberately NOT reusing ``api_fetcher.RateLimiter``: its per-second tier is
    dead code. It computes ``sleep_time = 1.0 - (now - cutoff_1s)`` where
    ``cutoff_1s = now - 1.0``, so ``sleep_time`` is identically ``0.0`` and the
    ``if sleep_time > 0`` branch never fires. Its 1-min and 30-min tiers do work,
    but neither is relevant here. Simple spacing is exactly right for a flat
    per-second cap and has no such edge case.
    """

    def __init__(self, rps: float):
        self._min_interval = 1.0 / float(rps)
        self._last = 0.0

    def wait(self) -> None:
        now = time.monotonic()
        earliest = self._last + self._min_interval
        if now < earliest:
            time.sleep(earliest - now)
        self._last = time.monotonic()


def to_session_datetimes(
    from_date: date, to_date: date
) -> tuple[datetime, datetime]:
    """Convert dates to the naive IST datetimes the Kite SDK requires.

    The SDK does ``if type(from_date) == datetime.datetime`` (connect.py:683) --
    a strict type check that a ``datetime.date`` FAILS. A bare ``date`` is then
    passed through raw and stringified without a time component, silently
    changing window semantics for intraday intervals. Always feed it real
    ``datetime`` objects.
    """
    return (
        datetime.combine(from_date, _SESSION_OPEN),
        datetime.combine(to_date, _SESSION_CLOSE),
    )


def date_chunks(
    from_date: date, to_date: date, days: int
) -> list[tuple[datetime, datetime]]:
    """Split [from_date, to_date] into <=`days`-wide naive-IST datetime windows."""
    chunks: list[tuple[datetime, datetime]] = []
    cur = from_date
    while cur <= to_date:
        end = min(cur + timedelta(days=days - 1), to_date)
        chunks.append(to_session_datetimes(cur, end))
        cur = end + timedelta(days=1)
    return chunks


class KiteFetcher:
    """Thin, rate-limited, retrying wrapper over the Kite Connect SDK."""

    def __init__(
        self,
        access_token: str | None = None,
        *,
        api_key: str | None = None,
        rps: float | None = None,
        timeout: int | None = None,
    ):
        self.api_key = api_key or settings.KITE_API_KEY
        if not self.api_key:
            raise KiteAuthError(
                "KITE_API_KEY is not set (checked settings / .env)."
            )
        self.access_token = resolve_kite_access_token(access_token)

        self.kite = KiteConnect(
            api_key=self.api_key,
            timeout=timeout or settings.KITE_TIMEOUT,
        )
        self.kite.set_access_token(self.access_token)
        # Fires on any mid-run 403 TokenException -- a long backfill started just
        # before 06:00 IST would otherwise die with a cryptic error.
        self.kite.set_session_expiry_hook(self._on_session_expiry)

        self.rate_limiter = _KiteRateLimiter(rps or settings.KITE_HISTORICAL_RPS)
        self._request_count = 0

    # -- auth --------------------------------------------------------------

    @staticmethod
    def _on_session_expiry() -> None:
        logger.error(
            "Kite session expired mid-run (403 TokenException).\n"
            f"  {_login_hint()}"
        )

    def validate_session(self) -> dict:
        """Authoritative token check. Raises KiteAuthError if dead."""
        try:
            profile = self.kite.profile()
        except (TokenException, PermissionException) as exc:
            raise KiteAuthError(
                f"Kite rejected the access token ({type(exc).__name__}: {exc}).\n"
                f"  {_login_hint()}"
            ) from exc
        logger.info(
            f"Kite session OK -- {profile.get('user_name')} "
            f"({profile.get('user_id')}, {profile.get('broker')})"
        )
        return profile

    # -- instruments -------------------------------------------------------

    def instruments(self, exchange: str = "MCX", use_cache: bool = True) -> list[dict]:
        """Fetch the instrument master for an exchange, cached per calendar day.

        Kite regenerates the dump once daily, so a same-day cache is free
        accuracy. Note the dump contains ONLY live contracts -- expired ones are
        never listed, which is the whole reason ``continuous=1`` exists.
        """
        cache_dir = Path(settings.KITE_INSTRUMENTS_CACHE_DIR)
        cache_file = cache_dir / f"{exchange}_{date.today().isoformat()}.json"

        if use_cache and cache_file.exists():
            with cache_file.open("r", encoding="utf-8") as fh:
                return json.load(fh)

        self.rate_limiter.wait()
        self._request_count += 1
        rows = self.kite.instruments(exchange)
        logger.info(f"Fetched {len(rows):,} {exchange} instruments from Kite")

        cache_dir.mkdir(parents=True, exist_ok=True)
        with cache_file.open("w", encoding="utf-8") as fh:
            # expiry/last_trade_time come back as date objects -- coerce for JSON
            json.dump(rows, fh, default=str)

        return rows

    def resolve_front_month(
        self,
        name: str,
        exchange: str = "MCX",
        as_of: date | None = None,
        use_cache: bool = True,
    ) -> dict:
        """Return the nearest-expiry LIVE futures contract for a commodity.

        Matches ``name`` EXACTLY. Kite's ``name`` column holds the bare
        commodity ("NICKEL", "LEADMINI"), so exact matching already excludes
        mini contracts -- unlike the Upstox JSON, where "ALUMINIUM" contains the
        substring "MINI" and naive substring exclusion silently drops it.
        """
        as_of = as_of or date.today()
        rows = self.instruments(exchange, use_cache=use_cache)

        hits = []
        for r in rows:
            if r.get("instrument_type") != "FUT":
                continue
            if str(r.get("name", "")).upper() != name.upper():
                continue
            expiry = r.get("expiry")
            if isinstance(expiry, str):
                expiry = date.fromisoformat(expiry[:10]) if expiry else None
            if isinstance(expiry, datetime):
                expiry = expiry.date()
            if not expiry or expiry < as_of:
                continue
            hits.append({**r, "expiry": expiry})

        if not hits:
            available = sorted({
                str(r.get("name")) for r in rows if r.get("instrument_type") == "FUT"
            })
            raise ValueError(
                f"No live {exchange} FUT contract found for name={name!r} "
                f"as of {as_of}. Available names: {available}"
            )

        hits.sort(key=lambda r: r["expiry"])
        return hits[0]

    def resolve_equity(
        self, tradingsymbol: str, exchange: str = "NSE", use_cache: bool = True
    ) -> dict:
        """Return the instrument row for one NSE/BSE equity by trading symbol.

        Exact match on ``tradingsymbol`` (uppercased), restricted to
        ``instrument_type == "EQ"`` -- the equity analogue of
        ``resolve_front_month``.
        """
        rows = self.instruments(exchange, use_cache=use_cache)
        target = tradingsymbol.upper().strip()

        hits = [
            r for r in rows
            if r.get("instrument_type") == "EQ" and str(r.get("tradingsymbol", "")).upper() == target
        ]
        if not hits:
            # Series fallback: a stock in a non-rolling NSE series carries the
            # series as a tradingsymbol suffix (HEG -> HEG-BE in Trade-to-Trade,
            # also -BZ/-SM/-ST), so the exact match above misses it even though
            # the instrument exists. The `name` field stays the bare symbol, so
            # match on that. Hit live on the Nifty 500 fetch: HEG and HFCL were
            # the only 2 of 500 in T2T that day, and index membership means a
            # name can move in and out of BE over time -- so this has to be
            # resolved generically, not by hardcoding the two.
            hits = [
                r for r in rows
                if r.get("instrument_type") == "EQ" and str(r.get("name", "")).upper() == target
            ]
            if hits:
                logger.info(
                    f"resolve_equity: {tradingsymbol!r} matched via name -> "
                    f"tradingsymbol={hits[0].get('tradingsymbol')!r} (non-rolling NSE series)"
                )
        if not hits:
            raise ValueError(f"No {exchange} EQ instrument found for tradingsymbol={tradingsymbol!r}")
        if len(hits) > 1:
            logger.warning(
                f"Multiple {exchange} EQ rows matched {tradingsymbol!r} -- using the first: "
                f"{hits[0].get('instrument_token')}"
            )
        return hits[0]

    def resolve_equities(
        self, tradingsymbols: list[str], exchange: str = "NSE", use_cache: bool = True
    ) -> dict[str, dict]:
        """Batch version of ``resolve_equity`` -- one ``instruments()`` call (cached)
        instead of N, then an in-memory lookup for every symbol in *tradingsymbols*.

        Returns ``{tradingsymbol.upper(): row}``; symbols with no match are
        logged and simply omitted from the returned dict rather than raising,
        since resolving 500 names should not abort on the first miss.
        """
        rows = self.instruments(exchange, use_cache=use_cache)
        by_symbol: dict[str, dict] = {}
        by_name: dict[str, dict] = {}
        for r in rows:
            if r.get("instrument_type") == "EQ":
                by_symbol.setdefault(str(r.get("tradingsymbol", "")).upper(), r)
                # Secondary index for the series fallback -- see resolve_equity.
                by_name.setdefault(str(r.get("name", "")).upper(), r)

        out: dict[str, dict] = {}
        missing: list[str] = []
        for sym in tradingsymbols:
            key = sym.upper().strip()
            row = by_symbol.get(key)
            if row is None:
                row = by_name.get(key)
                if row is not None:
                    logger.info(
                        f"resolve_equities: {sym!r} matched via name -> "
                        f"tradingsymbol={row.get('tradingsymbol')!r} (non-rolling NSE series)"
                    )
            if row is None:
                missing.append(sym)
            else:
                out[key] = row

        if missing:
            logger.warning(f"resolve_equities: {len(missing)} symbol(s) not found on {exchange}: {missing}")
        logger.info(f"resolve_equities: resolved {len(out)}/{len(tradingsymbols)} on {exchange}")
        return out

    # -- candles -----------------------------------------------------------

    @retry(
        retry=retry_if_exception_type((NetworkException, DataException)),
        stop=stop_after_attempt(4),
        wait=wait_exponential(multiplier=1, min=2, max=30),
    )
    def _historical_request(
        self,
        instrument_token: int,
        from_dt: datetime,
        to_dt: datetime,
        interval: str,
        continuous: bool,
        oi: bool,
    ) -> list[dict]:
        """One rate-limited historical call.

        Retries only transient faults. TokenException / PermissionException /
        InputException are permanent for the run -- retrying them just burns 4x
        the time before the identical failure.
        """
        self.rate_limiter.wait()
        self._request_count += 1
        return self.kite.historical_data(
            instrument_token,
            from_date=from_dt,
            to_date=to_dt,
            interval=interval,
            continuous=continuous,
            oi=oi,
        )

    def fetch_candles(
        self,
        instrument_token: int,
        interval: str,
        from_dt: datetime,
        to_dt: datetime,
        *,
        continuous: bool = False,
        oi: bool = True,
        instrument_key: str = "",
        exchange: str = "MCX",
    ) -> list[Bar]:
        """Fetch one window of candles as Bar objects.

        ``exchange`` only feeds the fallback ``instrument_key`` label used when
        the caller doesn't supply one -- it does NOT affect the actual API call
        (Kite resolves purely from ``instrument_token``). Default stays "MCX"
        so existing MCX call sites that don't pass ``instrument_key`` keep
        producing the exact same label as before; equity callers should pass
        ``exchange="NSE"`` (or just pass ``instrument_key`` directly and ignore
        this). Previously this was hardcoded to "MCX" unconditionally, which
        would have mislabeled every equity Bar's ``instrument_key``.
        """
        raw = self._historical_request(
            instrument_token, from_dt, to_dt, interval, continuous, oi
        )
        key = instrument_key or f"KITE_{exchange}|{instrument_token}"
        bars = [Bar.from_kite_candle(c, instrument_key=key) for c in raw]
        bars.sort(key=lambda b: b.timestamp)
        logger.debug(
            f"Kite {instrument_token} {interval} "
            f"[{from_dt:%Y-%m-%d} -> {to_dt:%Y-%m-%d}] continuous={int(continuous)} "
            f"-> {len(bars)} candles"
        )
        return bars

    def fetch_candles_chunked(
        self,
        instrument_token: int,
        interval: str,
        from_date: date,
        to_date: date,
        *,
        continuous: bool = False,
        oi: bool = True,
        max_days: int = 1900,
        instrument_key: str = "",
        exchange: str = "MCX",
    ) -> pd.DataFrame:
        """Fetch a long span in <=`max_days` chunks; return an IST-indexed frame.

        See ``fetch_candles`` for what ``exchange`` does (fallback label only).
        """
        frames: list[pd.DataFrame] = []
        for c_from, c_to in date_chunks(from_date, to_date, max_days):
            bars = self.fetch_candles(
                instrument_token,
                interval,
                c_from,
                c_to,
                continuous=continuous,
                oi=oi,
                instrument_key=instrument_key,
                exchange=exchange,
            )
            if bars:
                frames.append(pd.DataFrame(b.to_dict() for b in bars))

        if not frames:
            return pd.DataFrame()

        df = pd.concat(frames, ignore_index=True)
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True).dt.tz_convert(IST)
        df = df.set_index("timestamp").sort_index()
        df = df[~df.index.duplicated(keep="last")]
        keep = [c for c in ("open", "high", "low", "close", "volume", "oi") if c in df.columns]
        return df[keep]

    @property
    def request_count(self) -> int:
        return self._request_count


# -- daily login: request_token -> access_token --------------------------------

def _parse_request_token(token_or_url: str) -> str:
    """Accept a bare ``request_token`` or the full Kite redirect URL containing it."""
    raw = token_or_url.strip()
    query = parse_qs(urlparse(raw).query).get("request_token") or parse_qs(raw).get("request_token")
    return query[0] if query else raw


def exchange_request_token(token_or_url: str, env_path: Path | None = None) -> str:
    """Exchange a manually supplied Kite ``request_token`` for today's access token.

    Calls ``KiteConnect.generate_session(request_token, api_secret)``, validates the
    new token with ``profile()``, and writes it as ``KITE_ACCESS_TOKEN`` into
    *env_path* (default ``settings.KITE_ENV_PATH`` = this repo's ``.env``), where
    ``resolve_kite_access_token`` picks it up on every later run that day.

    Returns the access token.
    """
    if not settings.KITE_API_KEY or not settings.KITE_API_SECRET:
        raise KiteAuthError("KITE_API_KEY / KITE_API_SECRET are not set in .env.")
    request_token = _parse_request_token(token_or_url)
    kite = KiteConnect(api_key=settings.KITE_API_KEY, timeout=settings.KITE_TIMEOUT)
    try:
        session = kite.generate_session(request_token, api_secret=settings.KITE_API_SECRET)
    except (TokenException, InputException, PermissionException) as exc:
        raise KiteAuthError(
            f"Kite rejected the request token ({type(exc).__name__}: {exc}). Request tokens "
            "are single-use and expire within minutes -- log in again for a fresh one."
        ) from exc
    access_token = session["access_token"]
    kite.set_access_token(access_token)
    profile = kite.profile()

    env_path = Path(env_path or settings.KITE_ENV_PATH)
    env_path.touch(exist_ok=True)
    set_key(str(env_path), "KITE_ACCESS_TOKEN", access_token)
    logger.info(
        f"Kite login OK -- {profile.get('user_name')} ({profile.get('user_id')}). "
        f"KITE_ACCESS_TOKEN saved to {env_path}."
    )
    return access_token


def login_main(argv: list[str] | None = None) -> int:
    """``kite-login`` entry point.

    kite-login                                  print the login URL to open in a browser
    kite-login --request-token <token|URL>      exchange it and save today's access token
    """
    import argparse

    ap = argparse.ArgumentParser(
        prog="kite-login",
        description="Daily Kite Connect login. 1) run without arguments and open the printed "
                    "URL; 2) after logging in, copy the request_token (or the whole redirect "
                    "URL) and run again with --request-token.",
    )
    ap.add_argument("--request-token", metavar="TOKEN_OR_URL",
                    help="request_token from the Kite redirect (or the full redirect URL).")
    args = ap.parse_args(argv)

    if not settings.KITE_API_KEY:
        print("KITE_API_KEY is not set in .env.")
        return 1
    if not args.request_token:
        print("Open this URL, log in, then copy the request_token from the redirect URL:\n")
        print(KiteConnect(api_key=settings.KITE_API_KEY).login_url())
        print("\nThen run:  kite-login --request-token <request_token or full redirect URL>")
        return 0
    try:
        exchange_request_token(args.request_token)
    except KiteAuthError as exc:
        print(exc)
        return 1
    return 0
