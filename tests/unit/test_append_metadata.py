"""Incremental appends/upserts keep symbol metadata.

ArcticDB stamps each new version with only the metadata passed to that call, so
a bare ``lib.append`` used to drop the Kite ``source`` stamp on spot symbols and
``spot_symbol_for`` (hence every borrow rate) silently went to None/NaN.
"""

import numpy as np
import pandas as pd
import pytest

from data.storage.arctic_store import LIB_FUTURES, LIB_SPOT, ArcticStore

KITE = {"source": "fetch_kite_equities", "symbol": "TST", "interval": "day",
        "fetched_at": "2026-10-01"}


@pytest.fixture
def uri(tmp_path):
    return f"lmdb://{tmp_path}?map_size=20MB"


def _bars(start, n):
    idx = pd.date_range(start, periods=n, freq="D", tz="Asia/Kolkata")
    return pd.DataFrame({"open": 1.0, "high": 2.0, "low": 0.5,
                         "close": np.arange(n, dtype=float), "volume": 10}, index=idx)


def test_append_spot_carries_metadata_forward(uri):
    store = ArcticStore(uri=uri)
    store.write_spot("TST", "day", _bars("2026-10-01", 5), metadata=KITE)

    assert store.append_spot("TST", "day", _bars("2026-10-06", 3)) == 3

    assert store.read_metadata(LIB_SPOT, "TST/day") == KITE
    assert len(store.read_spot("TST", "day")) == 8
    # Fresh instance: _is_kite_spot's cache must not mask a lost stamp.
    assert ArcticStore(uri=uri).spot_symbol_for("TST", "1day") == "TST/day"


def test_append_spot_explicit_metadata_replaces(uri):
    store = ArcticStore(uri=uri)
    store.write_spot("TST", "day", _bars("2026-10-01", 5), metadata=KITE)
    new_meta = {**KITE, "fetched_at": "2026-10-12"}

    store.append_spot("TST", "day", _bars("2026-10-06", 3), metadata=new_meta)

    assert store.read_metadata(LIB_SPOT, "TST/day") == new_meta
    assert ArcticStore(uri=uri).spot_symbol_for("TST", "1day") == "TST/day"


def test_append_futures_and_update_carry_metadata_forward(uri):
    store = ArcticStore(uri=uri)
    meta = {"source": "fetch_ticker", "symbol": "TST"}
    store.write_futures("TST", "1day", _bars("2026-10-01", 5), metadata=meta)

    store.append_futures("TST", "1day", _bars("2026-10-06", 2))
    assert store.read_metadata(LIB_FUTURES, "TST/1day") == meta

    store.update(LIB_FUTURES, "TST/1day", _bars("2026-10-07", 2))
    assert store.read_metadata(LIB_FUTURES, "TST/1day") == meta
