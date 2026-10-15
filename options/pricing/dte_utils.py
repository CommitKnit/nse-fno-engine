"""Trading-day calendar + intraday DTE utilities.

This module is the single source of truth for:

* the NSE trading-day calendar (weekends + holidays), year-aware,
* candle-interval arithmetic (interval name -> minutes, candles per session),
* the **fractional, intraday-decrementing DTE** used across the pipeline, and
* the year-fraction ``T`` conversion for Black-76.

DTE convention (trading days, not calendar days)
------------------------------------------------
``day_open_dte(date)`` = number of trading days in the inclusive window
``[date, expiry]``. So on the expiry day itself the day-open value is ``1.0``
(one trading day of life left) and it decrements to ``0.0`` at the 15:30 close
candle. A bar 30 trading days out opens at ``30`` and closes that day at ``29``.

Within a trading day the value decrements once per candle by
``interval_minutes / SESSION_MINUTES``. For a 15-minute candle that is
``15 / 375 = 0.04`` (25 candles per session), so the first candle of the day
reads ``day_open - 0.04`` and the last reads ``day_open - 1.0``.

The year fraction fed to Black-76 is ``dte / 252`` (trading-day year).
"""

from __future__ import annotations

import math
import re
from collections.abc import Iterable
from datetime import date, datetime, time, timedelta

from config.constants import ALL_TRADING_HOLIDAYS

_ONE_DAY = timedelta(days=1)

# Default per-expiry lookback window (trading days) for the "is this bar recent
# enough to keep" flag — see in_last_n_trading_days / last_n_td_column below.
DEFAULT_LOOKBACK_TD = 40

# Safety bound for nth_trading_day_before's backward walk: even a fully
# holiday-free run of DEFAULT_LOOKBACK_TD*3 calendar days is generous headroom
# for any realistic N passed by callers (fetch_options.py's --lookback-td).
_MAX_LOOKBACK_CALENDAR_DAYS = 400

# ── Session geometry (NSE equity/F&O, IST) ─────────────────────────────────────
SESSION_OPEN = time(9, 15)
SESSION_CLOSE = time(15, 30)
SESSION_MINUTES = 375  # 09:15 -> 15:30 inclusive of the closing candle
TRADING_DAYS_PER_YEAR = 252

# interval name -> minutes per candle. Daily/weekly collapse to one session.
_INTERVAL_MINUTES: dict[str, int] = {
    "1minute": 1,
    "3minute": 3,
    "5minute": 5,
    "10minute": 10,
    "15minute": 15,
    "30minute": 30,
    "60minute": 60,
    "1hour": 60,
    "day": SESSION_MINUTES,
    "1day": SESSION_MINUTES,
}

# Holidays parsed once into a set[date].
_HOLIDAYS: set[date] = {
    date.fromisoformat(h) if isinstance(h, str) else h for h in ALL_TRADING_HOLIDAYS
}


def interval_to_minutes(interval: str) -> int:
    """Map a candle-interval name (e.g. ``"15minute"``) to minutes per candle.

    Daily/weekly intervals collapse to a full session (``SESSION_MINUTES``), so
    their fractional DTE decrements by exactly 1.0 per trading day.
    """
    key = str(interval).strip().lower()
    if key not in _INTERVAL_MINUTES:
        raise ValueError(
            f"Unknown interval {interval!r}; known: {sorted(_INTERVAL_MINUTES)}"
        )
    return _INTERVAL_MINUTES[key]


def candles_per_day(interval: str) -> int:
    """Number of candles in one trading session for *interval*."""
    return max(1, SESSION_MINUTES // interval_to_minutes(interval))


def is_trading_day(d: date) -> bool:
    """True if *d* is a weekday and not an NSE holiday."""
    return d.weekday() < 5 and d not in _HOLIDAYS


def trading_days_inclusive(start: date, expiry: date) -> int:
    """Count trading days in the inclusive window ``[start, expiry]``.

    Returns 0 when ``start > expiry``. This powers both the day-open DTE and the
    last-N-trading-days flag, so on the expiry day itself this is 1.
    """
    if start > expiry:
        return 0
    count = 0
    current = start
    while current <= expiry:
        if is_trading_day(current):
            count += 1
        current += _ONE_DAY
    return count


def _candle_index(ts: datetime, interval_minutes: int) -> int:
    """1-based candle ordinal within the trading session for timestamp *ts*.

    The 09:15 candle is index 1; the final 15:15 candle of a 15-minute session is
    index 25. Values are clamped into ``[1, candles_per_day]`` so pre/post-session
    timestamps don't push DTE outside the day's band.
    """
    minutes_since_open = (ts.hour - SESSION_OPEN.hour) * 60 + (ts.minute - SESSION_OPEN.minute)
    n_candles = max(1, SESSION_MINUTES // interval_minutes)
    idx = math.floor(minutes_since_open / interval_minutes) + 1
    return max(1, min(n_candles, idx))


def fractional_dte(ts: datetime, expiry: date, interval: str) -> float:
    """Fractional, intraday-decrementing trading-day DTE for a candle at *ts*.

    ``day_open = trading_days_inclusive(ts.date(), expiry)`` then subtract
    ``candle_index * (interval_minutes / SESSION_MINUTES)``. Never negative:
    the expiry-day close candle lands on exactly 0.0.
    """
    m = interval_to_minutes(interval)
    day_open = trading_days_inclusive(ts.date(), expiry)
    if day_open <= 0:
        return 0.0
    k = _candle_index(ts, m)
    return max(0.0, day_open - k * (m / SESSION_MINUTES))


def year_fraction(dte: float) -> float:
    """Convert a trading-day DTE into a Black-76 year fraction (``dte / 252``)."""
    return dte / TRADING_DAYS_PER_YEAR


def in_last_n_trading_days(bar_date: date, expiry: date, n: int = DEFAULT_LOOKBACK_TD) -> bool:
    """True when *bar_date* is within the last *n* trading days up to *expiry*.

    Uses the inclusive trading-day count, so it is stable across a trading day
    (does not flip mid-session).
    """
    tdays = trading_days_inclusive(bar_date, expiry)
    return 0 < tdays <= n


def last_n_td_column(n: int) -> str:
    """Column name for the "last *n* trading days" flag, e.g. ``in_last_40_td``.

    Single definition of this naming convention — shared by whatever writes the
    flag (``options/chain/build_enriched_chains.py::_attach_in_last_n_td``) and every
    downstream reader that needs to find it (``fetch_options.py``,
    ``analyze_enriched_options.py``, ``fit_ssvi_surface.py``), so a non-default
    ``n`` can never desync the writer's column name from a reader's lookup.
    """
    return f"in_last_{int(n)}_td"


_LAST_N_TD_RE = re.compile(r"^in_last_(\d+)_td$")


def resolve_lookback_td_column(columns: Iterable[str]) -> tuple[str, int] | None:
    """Find the last-N-trading-days flag among *columns*, whatever N it was written with.

    Inverse of :func:`last_n_td_column`: scans for the first name matching
    ``in_last_(\\d+)_td`` and returns ``(column_name, n)``, or ``None`` if none is
    present. Lets a reader (``analyze_enriched_options.py``, ``fit_ssvi_surface.py``)
    resolve whatever ``--lookback-td`` a given ``fetch_options.py`` /
    ``build_enriched_chains.py`` run used, instead of assuming the default
    ``in_last_40_td`` and silently analyzing unfiltered data when it doesn't match.

    Takes column *names* (e.g. ``df.columns``) rather than a DataFrame directly,
    so this stays dependency-free of pandas.
    """
    for col in columns:
        m = _LAST_N_TD_RE.match(str(col))
        if m:
            return str(col), int(m.group(1))
    return None


def nth_trading_day_before(expiry: date, n: int) -> date:
    """Calendar date of the *n*-th trading day counting back from *expiry*.

    This is the earliest date ``d`` for which
    ``in_last_n_trading_days(d, expiry, n)`` is ``True`` — i.e. the left edge of
    the last-*n*-trading-days window, inclusive of both ends. Used to turn a
    trading-day lookback into a concrete fetch window
    (``fetch_options.py::_enumerate_windows``).

    Walks backward one calendar day at a time counting trading days via
    ``is_trading_day`` (reusing the same weekend/holiday calendar as everything
    else in this module) until *n* trading days have been seen, including
    *expiry* itself if it is a trading day. Bounded by
    ``_MAX_LOOKBACK_CALENDAR_DAYS`` so a degenerate holiday table can't spin
    forever.

    Raises:
        ValueError: if *n* is not positive, or the walk exceeds the bound
            (should not happen for any realistic *n*).
    """
    if n <= 0:
        raise ValueError(f"nth_trading_day_before: n must be positive, got {n}")

    remaining = n
    current = expiry
    for _ in range(_MAX_LOOKBACK_CALENDAR_DAYS):
        if is_trading_day(current):
            remaining -= 1
            if remaining == 0:
                return current
        current -= _ONE_DAY
    raise ValueError(
        f"nth_trading_day_before: did not find {n} trading days before "
        f"{expiry} within {_MAX_LOOKBACK_CALENDAR_DAYS} calendar days — "
        "check the holiday calendar / n."
    )
