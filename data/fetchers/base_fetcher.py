
"""Abstract base class for data fetchers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import date

import pandas as pd

from data.models.bar import Bar
from data.models.option_contract import OptionContract


class BaseFetcher(ABC):
    """Interface that all data fetchers must implement.

    Concrete implementations: UpstoxFetcher (API), CsvFetcher (local files).
    """

    @abstractmethod
    def fetch_expiries(self, instrument_key: str) -> list[date]:
        """Get all available expiry dates for an underlying."""
        ...

    @abstractmethod
    def fetch_option_contracts(
        self, instrument_key: str, expiry_date: date
    ) -> list[OptionContract]:
        """Get all option contracts (CE+PE) for one expiry."""
        ...

    @abstractmethod
    def fetch_future_contracts(
        self, instrument_key: str, expiry_date: date
    ) -> list[dict]:
        """Get futures contracts for one expiry."""
        ...

    @abstractmethod
    def fetch_historical_candles(
        self,
        instrument_key: str,
        interval: str,
        from_date: date,
        to_date: date,
    ) -> list[Bar]:
        """Get OHLCV candles for any instrument (spot, futures, options)."""
        ...

    @abstractmethod
    def fetch_expired_candles(
        self,
        expired_instrument_key: str,
        interval: str,
        from_date: date,
        to_date: date,
    ) -> list[Bar]:
        """Get OHLCV candles for an expired F&O contract."""
        ...
