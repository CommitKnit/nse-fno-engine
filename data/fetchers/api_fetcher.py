
"""Upstox API fetcher with rate limiting, retries, and all required endpoints.

Handles these Upstox API endpoints:
    1. GET /v2/expired-instruments/expiries          → expiry dates
    2. GET /v2/expired-instruments/option/contract    → option contracts per expiry
    3. GET /v2/expired-instruments/future/contract    → futures contracts per expiry
    4. GET /v2/expired-instruments/historical-candle   → expired F&O OHLCV
    5. GET /v2/historical-candle/{key}/{interval}/... → spot/index OHLCV
    6. GET /v2/option/chain                           → live chain (validation)
    7. GET /v2/instruments/search                     → live instrument resolution
    8. GET /v2/option/contract                        → live option contracts per expiry

Expired vs live:
    Endpoints 1-4 resolve and fetch *expired* F&O history (needs an underlying key
    you already have). Endpoints 7-8 cover the *live* path: search resolves a bare
    equity ticker to live spot / futures / underlying keys, and /v2/option/contract
    returns the live CE+PE chain — both then feed fetch_historical_candles*.

Rate limits: 25/sec, 250/min, 1000/30min (per user, per API).
We stay safely under with 20/sec, 230/min, 950/30min defaults.
"""

from __future__ import annotations

import time
from collections import deque
from datetime import date, datetime, timedelta
from typing import Any, Callable

import requests
from loguru import logger
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from config.settings import settings
from data.fetchers.base_fetcher import BaseFetcher
from data.models.bar import Bar
from data.models.option_contract import OptionContract


class RateLimiter:
    """Token-bucket style rate limiter respecting Upstox's 3-tier limits."""

    def __init__(
        self,
        per_second: int = 20,
        per_minute: int = 230,
        per_30min: int = 950,
    ):
        self.per_second = per_second
        self.per_minute = per_minute
        self.per_30min = per_30min
        self._timestamps: deque[float] = deque()

    def wait_if_needed(self) -> None:
        """Block until it's safe to make the next request."""
        now = time.monotonic()

        # Purge timestamps older than 30 minutes
        cutoff_30m = now - 1800
        while self._timestamps and self._timestamps[0] < cutoff_30m:
            self._timestamps.popleft()

        # Check 30-min window
        if len(self._timestamps) >= self.per_30min:
            wait_until = self._timestamps[0] + 1800
            sleep_time = wait_until - now
            if sleep_time > 0:
                logger.warning(f"Rate limit (30min): sleeping {sleep_time:.1f}s")
                time.sleep(sleep_time)
                return self.wait_if_needed()

        # Check 1-min window
        cutoff_1m = now - 60
        recent_1m = sum(1 for t in self._timestamps if t >= cutoff_1m)
        if recent_1m >= self.per_minute:
            oldest_in_window = next(t for t in self._timestamps if t >= cutoff_1m)
            wait_until = oldest_in_window + 60
            sleep_time = wait_until - now
            if sleep_time > 0:
                logger.warning(f"Rate limit (1min): sleeping {sleep_time:.1f}s")
                time.sleep(sleep_time)
                return self.wait_if_needed()

        # Check 1-sec window
        cutoff_1s = now - 1.0
        recent_1s = sum(1 for t in self._timestamps if t >= cutoff_1s)
        if recent_1s >= self.per_second:
            sleep_time = 1.0 - (now - cutoff_1s)
            if sleep_time > 0:
                time.sleep(sleep_time)

        self._timestamps.append(time.monotonic())

    @property
    def calls_remaining_30min(self) -> int:
        now = time.monotonic()
        cutoff = now - 1800
        used = sum(1 for t in self._timestamps if t >= cutoff)
        return max(0, self.per_30min - used)


class UpstoxFetcher(BaseFetcher):
    """Fetches data from Upstox REST API v2.

    Usage:
        fetcher = UpstoxFetcher(access_token="your_token")
        expiries = fetcher.fetch_expiries("NSE_INDEX|Nifty 50")
        contracts = fetcher.fetch_option_contracts("NSE_INDEX|Nifty 50", expiries[0])
    """

    BASE_URL = "https://api.upstox.com"

    def __init__(self, access_token: str | None = None):
        self.access_token = access_token or settings.UPSTOX_ACCESS_TOKEN
        if not self.access_token:
            raise ValueError(
                "Upstox access token required. Set UPSTOX_ACCESS_TOKEN in .env"
            )
        self.session = requests.Session()
        self.session.headers.update({
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Authorization": f"Bearer {self.access_token}",
        })
        self.rate_limiter = RateLimiter(
            per_second=settings.RATE_LIMIT_PER_SECOND,
            per_minute=settings.RATE_LIMIT_PER_MINUTE,
            per_30min=settings.RATE_LIMIT_PER_30MIN,
        )
        self._request_count = 0

    # ── Core request method ──

    @retry(
        retry=retry_if_exception_type((requests.ConnectionError, requests.Timeout)),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
    )
    def _request(self, method: str, path: str, params: dict | None = None) -> dict:
        """Make a rate-limited, retrying API request.

        Returns the parsed JSON response body.
        Raises ValueError on API errors.
        """
        self.rate_limiter.wait_if_needed()
        url = f"{self.BASE_URL}{path}"

        logger.debug(f"API {method} {path} params={params}")
        response = self.session.request(method, url, params=params, timeout=30)
        self._request_count += 1

        if response.status_code == 429:
            retry_after = int(response.headers.get("Retry-After", 60))
            logger.warning(f"HTTP 429 rate limited. Sleeping {retry_after}s")
            time.sleep(retry_after)
            return self._request(method, path, params)

        if response.status_code != 200:
            logger.error(f"API error {response.status_code}: {response.text[:500]}")
            response.raise_for_status()

        body = response.json()
        if body.get("status") != "success":
            raise ValueError(f"API returned non-success: {body}")

        return body

    # ── Endpoint 1: Get expiry dates ──

    def fetch_expiries(self, instrument_key: str) -> list[date]:
        """Get all available expiry dates for an underlying.

        Upstox returns up to 6 months of historical expiries.

        Args:
            instrument_key: e.g. "NSE_INDEX|Nifty 50"

        Returns:
            Sorted list of expiry dates (oldest first).
        """
        body = self._request(
            "GET",
            "/v2/expired-instruments/expiries",
            params={"instrument_key": instrument_key},
        )
        dates = [date.fromisoformat(d) for d in body.get("data", [])]
        dates.sort()
        logger.info(
            f"Fetched {len(dates)} expiries for {instrument_key} "
            f"({dates[0]} to {dates[-1]})" if dates else "no expiries found"
        )
        return dates

    # ── Endpoint 2: Get expired option contracts ──

    def fetch_option_contracts(
        self, instrument_key: str, expiry_date: date
    ) -> list[OptionContract]:
        """Get all CE+PE contracts for a specific expiry.

        Args:
            instrument_key: Underlying key e.g. "NSE_INDEX|Nifty 50"
            expiry_date: Expiry date to query.

        Returns:
            List of OptionContract with metadata (no OHLCV yet).
        """
        body = self._request(
            "GET",
            "/v2/expired-instruments/option/contract",
            params={
                "instrument_key": instrument_key,
                "expiry_date": expiry_date.isoformat(),
            },
        )
        contracts = [
            OptionContract.from_upstox_contract(c) for c in body.get("data", [])
        ]
        n_calls = sum(1 for c in contracts if c.is_call)
        n_puts = len(contracts) - n_calls
        logger.info(
            f"Fetched {len(contracts)} contracts for {expiry_date} "
            f"({n_calls} CE + {n_puts} PE)"
        )
        return contracts

    # ── Endpoint 3: Get expired futures contracts ──

    def fetch_future_contracts(
        self, instrument_key: str, expiry_date: date
    ) -> list[dict]:
        """Get futures contracts for a specific expiry.

        Args:
            instrument_key: Underlying key e.g. "NSE_INDEX|Nifty 50"
            expiry_date: Expiry date to query.

        Returns:
            List of dicts with instrument_key, trading_symbol, etc.
        """
        body = self._request(
            "GET",
            "/v2/expired-instruments/future/contract",
            params={
                "instrument_key": instrument_key,
                "expiry_date": expiry_date.isoformat(),
            },
        )
        contracts = body.get("data", [])
        logger.info(
            f"Fetched {len(contracts)} future contracts for {expiry_date}"
        )
        return contracts

    # ── Endpoint 4: Get expired instrument candles ──

    def fetch_expired_candles(
        self,
        expired_instrument_key: str,
        interval: str,
        from_date: date,
        to_date: date,
    ) -> list[Bar]:
        """Get OHLCV + OI candles for an expired F&O contract.

        Args:
            expired_instrument_key: Contract key e.g. "NSE_FO|37590"
                or "NSE_FO|NIFTY22D0117800CE" (symbol format also works)
            interval: "1minute", "3minute", "5minute", "15minute", "30minute", "day"
            from_date: Start date.
            to_date: End date.

        Returns:
            List of Bar objects sorted by timestamp ascending.
        """
        path = (
            f"/v2/expired-instruments/historical-candle"
            f"/{expired_instrument_key}/{interval}"
            f"/{to_date.isoformat()}/{from_date.isoformat()}"
        )
        body = self._request("GET", path)
        candles = body.get("data", {}).get("candles", [])
        bars = [
            Bar.from_upstox_candle(c, instrument_key=expired_instrument_key)
            for c in candles
        ]
        bars.sort(key=lambda b: b.timestamp)
        logger.debug(
            f"Fetched {len(bars)} candles for {expired_instrument_key} "
            f"({interval}, {from_date} to {to_date})"
        )
        return bars

    # ── Endpoint 5: Get spot/index historical candles ──

    def fetch_historical_candles(
        self,
        instrument_key: str,
        interval: str,
        from_date: date,
        to_date: date,
    ) -> list[Bar]:
        """Get OHLCV candles for a live/active instrument (spot, index, equity).

        Note: For expired F&O contracts, use fetch_expired_candles() instead.

        Lookback limits:
            1minute  → 1 month
            30minute → 1 year
            day      → 1 year

        Args:
            instrument_key: e.g. "NSE_INDEX|Nifty 50" or "NSE_EQ|INE009A01021"
            interval: "1minute", "30minute", "day", "week", "month"
            from_date: Start date.
            to_date: End date.

        Returns:
            List of Bar objects sorted by timestamp ascending.
        """
        path = (
            f"/v2/historical-candle/{instrument_key}/{interval}"
            f"/{to_date.isoformat()}/{from_date.isoformat()}"
        )
        body = self._request("GET", path)
        candles = body.get("data", {}).get("candles", [])
        bars = [
            Bar.from_upstox_candle(c, instrument_key=instrument_key)
            for c in candles
        ]
        bars.sort(key=lambda b: b.timestamp)
        logger.debug(
            f"Fetched {len(bars)} candles for {instrument_key} "
            f"({interval}, {from_date} to {to_date})"
        )
        return bars

    # ── Endpoint 5b: v3 historical candles (arbitrary unit/interval) ──

    def fetch_historical_candles_v3(
        self,
        instrument_key: str,
        unit: str,
        interval: int,
        from_date: date,
        to_date: date,
    ) -> list[Bar]:
        """Get OHLCV candles via the v3 endpoint (spot, index, equity, futures, options).

        Unlike the v2 endpoint (fixed to 1minute/30minute/day/week/month), v3 takes an
        arbitrary unit+interval pair, e.g. minutes/5, minutes/15, hours/1, days/1.
        Useful for intervals v2 no longer supports (e.g. 15minute was dropped from v2).

        Args:
            instrument_key: e.g. "NSE_EQ|INE009A01021"
            unit: "minutes", "hours", "days", "weeks", or "months"
            interval: multiplier for unit, e.g. 5 for minutes/5
            from_date: Start date.
            to_date: End date.

        Returns:
            List of Bar objects sorted by timestamp ascending.
        """
        path = (
            f"/v3/historical-candle/{instrument_key}/{unit}/{interval}"
            f"/{to_date.isoformat()}/{from_date.isoformat()}"
        )
        body = self._request("GET", path)
        candles = body.get("data", {}).get("candles", [])
        bars = [
            Bar.from_upstox_candle(c, instrument_key=instrument_key)
            for c in candles
        ]
        bars.sort(key=lambda b: b.timestamp)
        logger.debug(
            f"Fetched {len(bars)} candles for {instrument_key} "
            f"({unit}/{interval}, {from_date} to {to_date})"
        )
        return bars

    # ── Endpoint 6: Live option chain (validation only) ──

    def fetch_live_option_chain(
        self, instrument_key: str, expiry_date: date
    ) -> list[dict]:
        """Get live option chain with Greeks (for active contracts only).

        Useful for validating your IV solver against Upstox's computed Greeks.
        Does NOT work for expired contracts.

        Returns:
            Raw list of strike data dicts with call_options/put_options.
        """
        body = self._request(
            "GET",
            "/v2/option/chain",
            params={
                "instrument_key": instrument_key,
                "expiry_date": expiry_date.isoformat(),
            },
        )
        data = body.get("data", [])
        logger.info(
            f"Fetched live chain: {len(data)} strikes for {expiry_date}"
        )
        return data

    # ── Endpoint 7: Search live instruments (resolve keys from a ticker) ──

    @staticmethod
    def _expiry_param(expiry: date | str | None) -> str | None:
        """Normalise an expiry arg for query params.

        Accepts a ``date`` (→ ISO ``yyyy-MM-dd``), a raw string (ISO date or a
        relative keyword like ``current_month`` — passed through), or ``None``.
        """
        if expiry is None:
            return None
        if isinstance(expiry, date):
            return expiry.isoformat()
        return str(expiry)

    def search_instruments(
        self,
        query: str,
        *,
        exchanges: str = "NSE",
        segments: str | None = None,
        instrument_types: str | None = None,
        expiry: date | str | None = None,
        atm_offset: int | None = None,
        records: int = 30,
        max_pages: int | None = None,
    ) -> list[dict]:
        """Search the live instrument master via ``GET /v2/instruments/search``.

        This is Upstox's "instrument search" feature: resolve a free-text ``query``
        (symbol name, strike, or ISIN — partial, case-insensitive) into live
        instrument records. Used to resolve a bare equity ticker into live spot /
        futures / underlying keys, which then feed ``fetch_historical_candles*``.

        Args:
            query: Symbol / name / strike / ISIN (max 50 chars).
            exchanges: ``ALL``, ``NSE``, ``BSE`` or ``MCX`` (default ``NSE``).
            segments: ``ALL|EQ|FO|CURR|COMM|INDEX|OPT|FUT`` (default: all).
            instrument_types: Comma-separated, e.g. ``FUT`` or ``CE,PE``.
            expiry: Date, ISO string, or relative keyword (``current_month`` …).
            atm_offset: Distance from ATM (0=ATM, +above, -below).
            records: Per-page limit (API max 30; clamped).
            max_pages: Stop after this many pages (default: all pages).

        Returns:
            The concatenated ``data`` list of raw instrument dicts across all pages.
            NOTE: ``query`` is a *partial* match — callers must filter to an exact
            symbol before trusting a single result.
        """
        params: dict[str, Any] = {
            "query": query,
            "exchanges": exchanges,
            "segments": segments,
            "instrument_types": instrument_types,
            "expiry": self._expiry_param(expiry),
            "atm_offset": atm_offset,
            "records": min(records, 30),
        }
        params = {k: v for k, v in params.items() if v is not None}

        results: list[dict] = []
        page_number = 1
        while True:
            params["page_number"] = page_number
            body = self._request("GET", "/v2/instruments/search", params=params)
            results.extend(body.get("data", []))

            page_meta = body.get("meta_data", {}).get("page", {})
            total_pages = int(page_meta.get("total_pages", page_number))
            if page_number >= total_pages:
                break
            if max_pages is not None and page_number >= max_pages:
                break
            page_number += 1

        logger.info(
            f"Instrument search '{query}' "
            f"(exchanges={exchanges}, segments={segments}, "
            f"types={instrument_types}): {len(results)} records"
        )
        return results

    def resolve_spot_key(self, symbol: str, exchange: str = "NSE") -> str | None:
        """Resolve a ticker to its live spot/underlying ``instrument_key``.

        Tries the equity segment first (e.g. ``resolve_spot_key("TECHM")`` →
        ``"NSE_EQ|INE669C01036"``), then falls back to the index segment for
        tickers with no equity listing (e.g. ``resolve_spot_key("NIFTY")`` →
        ``"NSE_INDEX|Nifty 50"``). This is both the ticker→key entry step and the
        spot key used by ``fetch_historical_candles*``.

        Index matching via free-text search is best-effort — an index's
        ``trading_symbol`` may not equal the bare ticker exactly (e.g. "Nifty 50"
        vs "NIFTY"), unlike equities where it does. For known indices, prefer
        ``config.constants.get_instrument(symbol).instrument_key`` (exact, no API
        call) over this fallback; this exists for tickers not yet registered there.

        Returns the exact-match instrument_key, or ``None`` if not found in either
        segment.
        """
        target = symbol.strip().upper()

        eq_results = self.search_instruments(symbol, exchanges=exchange, segments="EQ")
        for r in eq_results:
            seg = str(r.get("segment", ""))
            ts = str(r.get("trading_symbol", "")).strip().upper()
            if seg.endswith("_EQ") and ts == target:
                return r.get("instrument_key")

        idx_results = self.search_instruments(symbol, exchanges=exchange, segments="INDEX")
        for r in idx_results:
            seg = str(r.get("segment", ""))
            ts = str(r.get("trading_symbol", "")).strip().upper()
            name = str(r.get("name", "")).strip().upper()
            if seg.endswith("_INDEX") and (ts == target or name == target):
                return r.get("instrument_key")

        logger.warning(
            f"resolve_spot_key: no exact EQ or INDEX match for '{symbol}' on {exchange}"
        )
        return None

    # ── Convenience: fetch live futures contracts (search has no expired analog) ──

    def fetch_live_future_contracts(
        self, symbol: str, exchange: str = "NSE"
    ) -> list[dict]:
        """Resolve the live (active) futures contracts for an underlying *symbol*.

        There is no dedicated live "future/contract" REST endpoint — the search API
        is the only way to enumerate live futures. This is the LIVE counterpart to
        ``fetch_future_contracts`` (expired): it returns dicts with the same keys
        downstream reads (``instrument_key``, ``trading_symbol``, ``expiry`` …), so
        callers can feed ``instrument_key`` into ``fetch_historical_candles_v3``.

        Args:
            symbol: Underlying ticker, e.g. ``TECHM`` (equity) or ``NIFTY`` (index).
            exchange: ``NSE`` (default), ``BSE`` or ``MCX``.

        Returns:
            Live FUT contract dicts, sorted by expiry ascending.
        """
        target = symbol.strip().upper()
        results = self.search_instruments(
            symbol, exchanges=exchange, segments="FO", instrument_types="FUT"
        )
        # Index futures sometimes report the ticker only in `name`, not
        # `underlying_symbol` (both fields equal the short ticker for equities, but
        # not always for indices) — match either, mirroring
        # scripts/fetch_ticker.py::resolve_futures_contracts's proven fallback.
        futures = [
            r for r in results
            if str(r.get("instrument_type", "")).upper() == "FUT"
            and (
                str(r.get("underlying_symbol", "")).strip().upper() == target
                or str(r.get("name", "")).strip().upper() == target
            )
        ]
        futures.sort(key=lambda r: str(r.get("expiry", "")))
        logger.info(f"Resolved {len(futures)} live futures for {symbol}")
        return futures

    # ── Endpoint 8: Get live option contracts (full CE+PE chain per expiry) ──

    def fetch_live_option_contracts(
        self, underlying: str, expiry_date: date | str | None = None
    ) -> list[OptionContract]:
        """Get the live CE+PE option chain for an underlying via ``/v2/option/contract``.

        The LIVE counterpart to ``fetch_option_contracts`` (expired): returns the full
        chain for an expiry in a single call and reuses ``OptionContract.from_upstox_contract``.

        Args:
            underlying: Underlying instrument key (``NSE_EQ|…`` / ``NSE_INDEX|…``) OR a
                bare ticker (e.g. ``TECHM``) which is resolved via ``resolve_spot_key``.
            expiry_date: Optional — a ``date``, ISO string, or relative keyword
                (``current_month``, ``next_month`` …). Omit for all live expiries.

        Returns:
            List of OptionContract (metadata only; feed keys to fetch_historical_candles*).
        """
        instrument_key = underlying if "|" in underlying else self.resolve_spot_key(underlying)
        if not instrument_key:
            raise ValueError(
                f"fetch_live_option_contracts: could not resolve underlying '{underlying}'"
            )

        params: dict[str, Any] = {"instrument_key": instrument_key}
        expiry_param = self._expiry_param(expiry_date)
        if expiry_param is not None:
            params["expiry_date"] = expiry_param

        body = self._request("GET", "/v2/option/contract", params=params)
        contracts = [
            OptionContract.from_upstox_contract(c) for c in body.get("data", [])
        ]
        n_calls = sum(1 for c in contracts if c.is_call)
        n_puts = len(contracts) - n_calls
        logger.info(
            f"Fetched {len(contracts)} live contracts for {instrument_key} "
            f"expiry={expiry_param or 'all'} ({n_calls} CE + {n_puts} PE)"
        )
        return contracts

    def resolve_underlying(self, symbol: str, exchange: str = "NSE") -> dict:
        """Resolve everything the live path needs from a bare ticker in one call.

        Returns a dict with the spot key, the underlying key (as reported by the live
        futures records; falls back to the spot key), the live futures contracts, and
        their sorted unique expiries.
        """
        spot_key = self.resolve_spot_key(symbol, exchange=exchange)
        futures = self.fetch_live_future_contracts(symbol, exchange=exchange)
        underlying_key = (futures[0].get("underlying_key") if futures else None) or spot_key
        expiries = sorted({str(f.get("expiry", "")) for f in futures if f.get("expiry")})
        return {
            "symbol": symbol,
            "spot_key": spot_key,
            "underlying_key": underlying_key,
            "futures": futures,
            "expiries": expiries,
        }

    # ── Convenience: fetch near-month futures OHLCV ──

    def fetch_near_month_futures(
        self,
        instrument_key: str,
        target_date: date,
        interval: str = "day",
    ) -> tuple[dict, list[Bar]]:
        """Fetch the near-month futures contract and its OHLCV.

        Finds the futures contract whose expiry is >= target_date
        (i.e., the contract that was the "current month" on that date),
        then fetches its historical candles.

        Args:
            instrument_key: Underlying key e.g. "NSE_INDEX|Nifty 50"
            target_date: The date you're interested in.
            interval: Candle interval.

        Returns:
            Tuple of (futures_contract_dict, list_of_bars).
        """
        # Get all expiries, find the nearest one >= target_date
        expiries = self.fetch_expiries(instrument_key)
        future_expiries = [e for e in expiries if e >= target_date]
        if not future_expiries:
            # All expiries are in the past — use the most recent
            future_expiries = expiries[-1:]

        near_expiry = future_expiries[0] if future_expiries else expiries[-1]

        # Get the futures contract for that expiry
        fut_contracts = self.fetch_future_contracts(instrument_key, near_expiry)
        if not fut_contracts:
            raise ValueError(
                f"No futures contract found for {instrument_key} expiry {near_expiry}"
            )

        fut_contract = fut_contracts[0]  # should be exactly one
        fut_key = fut_contract["instrument_key"]

        # Fetch candles
        # For expired futures, we need to go back from expiry
        from_date = date(target_date.year, target_date.month, 1)
        bars = self.fetch_expired_candles(
            fut_key, interval, from_date, near_expiry
        )

        logger.info(
            f"Near-month futures: {fut_contract.get('trading_symbol', fut_key)} "
            f"expiry={near_expiry}, {len(bars)} bars fetched"
        )
        return fut_contract, bars

    # ── Stats ──

    @property
    def request_count(self) -> int:
        return self._request_count

    @property
    def rate_limit_remaining(self) -> int:
        return self.rate_limiter.calls_remaining_30min

