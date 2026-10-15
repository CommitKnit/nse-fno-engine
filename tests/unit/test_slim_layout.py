"""Slim layout: futures panels and borrow_rates never store copied columns,
and reads join spot / panel inputs back on the fly."""

import numpy as np
import pandas as pd
import pytest

from data.storage.arctic_store import (
    BORROW_COPIED_COLUMNS,
    FUTURES_SPOT_COLUMNS,
    LIB_BORROW,
    LIB_FUTURES,
    ArcticStore,
)

# Metadata the Kite fetchers stamp on spot symbols; only these are used for SPOT_* joins.
KITE = {"source": "fetch_kite_equities"}


@pytest.fixture(scope="module")
def store(tmp_path_factory):
    root = tmp_path_factory.mktemp("arctic_slim")
    return ArcticStore(uri=f"lmdb://{root}?map_size=20MB")


def _idx(n=6):
    return pd.date_range("2025-01-01 09:15", periods=n, freq="15min", tz="Asia/Kolkata")


@pytest.fixture(scope="module")
def loaded(store):
    idx = _idx()
    panel = pd.DataFrame({
        "F1_close": np.arange(100.0, 106.0), "F2_close": np.arange(101.0, 107.0),
        "F3_close": np.arange(102.0, 108.0), "F1_oi": 10.0, "F2_oi": 20.0, "F3_oi": 30.0,
        "F1_dte": 5, "F2_dte": 33, "F3_dte": 61, "F1_volume": 7.0, "F2_volume": 8.0,
        "cycle_id": [1, 1, 1, 2, 2, 2], "is_expiry_day": False,
        # copies that must not be stored:
        "SPOT_close": 1.0, "SPOT_open": 1.0, "SPOT_ticker": "X", "spot_missing": False,
    }, index=idx)
    store.write_futures("TST", "15minute", panel)
    spot = pd.DataFrame({"open": 99.0, "high": 99.5, "low": 98.5,
                         "close": np.arange(99.0, 104.0), "volume": 1}, index=idx[:-1])
    store.write_spot("TST", "15minute", spot, metadata=KITE)
    borrow = panel.assign(b12=0.1, b1=0.05)            # full frame, as old code writes it
    store.write_borrow_rates("TST", "15minute", borrow)
    return panel, spot


def test_writes_drop_copied_columns(store, loaded):
    fut_cols = set(store.read(LIB_FUTURES, "TST/15minute").columns)
    br_cols = set(store.read(LIB_BORROW, "TST/15minute").columns)
    assert not fut_cols & set(FUTURES_SPOT_COLUMNS)
    assert not br_cols & set(BORROW_COPIED_COLUMNS)
    assert br_cols == {"b12", "b1"}


def test_read_futures_joins_kite_spot(store, loaded):
    _, spot = loaded
    df = store.read_futures("TST", "15minute")
    np.testing.assert_allclose(df["SPOT_close"].iloc[:-1], spot["close"])
    assert df["spot_missing"].tolist() == [False] * 5 + [True]
    assert np.isnan(df["SPOT_close"].iloc[-1])
    assert set(store.read_futures("TST", "15minute", with_spot=False).columns).isdisjoint(FUTURES_SPOT_COLUMNS)


def test_read_futures_column_projection(store, loaded):
    df = store.read_futures("TST", "15minute", columns=["F1_close", "SPOT_close"])
    assert list(df.columns) == ["F1_close", "SPOT_close"]


def test_read_borrow_rates_joined_and_raw(store, loaded):
    panel, spot = loaded
    full = store.read_borrow_rates("TST", "15minute")
    for c in ("F1_close", "F2_oi", "cycle_id", "SPOT_close", "b12"):
        assert c in full.columns
    np.testing.assert_allclose(full["F1_close"], panel["F1_close"])
    np.testing.assert_allclose(full["SPOT_close"].iloc[:-1], spot["close"])
    raw = store.read_borrow_rates("TST", "15minute", joined=False)
    assert set(raw.columns) == {"b12", "b1"}


def test_daily_alias_reads_kite_day_spot(store):
    idx = pd.date_range("2025-02-03", periods=3, freq="D", tz="Asia/Kolkata")
    store.write_futures("TST", "1day", pd.DataFrame({"F1_close": [1.0, 2.0, 3.0]}, index=idx))
    store.write_spot("TST", "day", pd.DataFrame({"open": 1.0, "high": 1.0, "low": 1.0,
                                                 "close": [9.0, 8.0, 7.0], "volume": 1}, index=idx),
                    metadata=KITE)
    assert store.spot_symbol_for("TST", "1day") == "TST/day"
    assert store.read_futures("TST", "1day")["SPOT_close"].tolist() == [9.0, 8.0, 7.0]


def test_kite_minute_preferred_over_short_upstox_1minute(store):
    idx = pd.date_range("2025-03-03 09:15", periods=4, freq="1min", tz="Asia/Kolkata")
    store.write_futures("TST", "1minute", pd.DataFrame({"F1_close": [1.0, 2.0, 3.0, 4.0]}, index=idx))
    store.write_spot("TST", "minute", pd.DataFrame({"open": 1.0, "high": 1.0, "low": 1.0,
                                                    "close": [5.0, 6.0, 7.0, 8.0], "volume": 1}, index=idx),
                    metadata=KITE)
    store.write_spot("TST", "1minute", pd.DataFrame({"open": 1.0, "high": 1.0, "low": 1.0,
                                                     "close": [0.5], "volume": 1}, index=idx[-1:]))
    assert store.spot_symbol_for("TST", "1minute") == "TST/minute"
    assert store.read_futures("TST", "1minute")["SPOT_close"].tolist() == [5.0, 6.0, 7.0, 8.0]
    # A ticker with ONLY a non-Kite (Upstox, no `source` metadata) series gets no spot.
    store.write_futures("UPX", "1minute", pd.DataFrame({"F1_close": [1.0]}, index=idx[-1:]))
    store.write_spot("UPX", "1minute", pd.DataFrame({"open": 1.0, "high": 1.0, "low": 1.0,
                                                     "close": [0.5], "volume": 1}, index=idx[-1:]))
    assert store.spot_symbol_for("UPX", "1minute") is None
    assert store.read_futures("UPX", "1minute")["spot_missing"].tolist() == [True]


def test_cycle_summary_untouched(store):
    summ = pd.DataFrame({"cycle_id": [1], "max_b12": [0.2], "F1_close": [5.0]})
    store.write(LIB_BORROW, "TST/15minute/cycle_summary", summ)
    assert "F1_close" in store.read(LIB_BORROW, "TST/15minute/cycle_summary").columns
