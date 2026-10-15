"""fetch_ticker.py --history: candle parsing, slot assignment and borrow formulas."""

from datetime import date, datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from data.models.bar import Bar
from data.storage.arctic_store import ArcticStore
from scripts.fetch_ticker import (
    BORROW_COLUMNS,
    PANEL_COLUMNS,
    add_missing_borrow_columns,
    bars_to_contract_frame,
    build_panel_from_contracts,
    compute_borrow,
    fill_missing,
    write_borrow,
)

IST = timezone(timedelta(hours=5, minutes=30))


def _bar(ts, px, oi=100):
    return Bar(timestamp=ts, open=px, high=px, low=px, close=px, volume=10, oi=oi, instrument_key="K")


def _contract(expiry: date, days: list[date], px: float, interval="1day"):
    bars = [_bar(datetime(d.year, d.month, d.day, 9, 15, tzinfo=IST), px + i) for i, d in enumerate(days)]
    return bars_to_contract_frame(bars, "TEST", expiry, interval, "NSE_FO|1")


def test_intraday_bars_are_floored_and_kept_in_session():
    ts = [datetime(2026, 1, 5, 9, 16, tzinfo=IST),    # offset bar -> 09:15
          datetime(2026, 1, 5, 3, 30, tzinfo=IST),    # pre-market summary -> dropped
          datetime(2026, 1, 5, 15, 25, tzinfo=IST)]
    df = bars_to_contract_frame([_bar(t, 10.0) for t in ts], "TEST", date(2026, 1, 27), "5minute", "NSE_FO|1")
    assert [t.strftime("%H:%M") for t in df.index] == ["09:15", "15:25"]
    assert df["dte"].tolist() == [22, 22]


def test_daily_bars_normalised_to_midnight():
    df = _contract(date(2026, 1, 27), [date(2026, 1, 5)], 10.0)
    assert df.index[0].strftime("%H:%M") == "00:00"


def test_slots_roll_after_expiry_and_cycle_id_counts_f1_changes():
    d = [date(2026, 1, 26), date(2026, 1, 27), date(2026, 1, 28)]
    c1 = _contract(date(2026, 1, 27), d[:2], 100.0)
    c2 = _contract(date(2026, 2, 24), d, 200.0)
    c3 = _contract(date(2026, 3, 30), d, 300.0)
    p = build_panel_from_contracts([c1, c2, c3], "TEST", "1day")
    assert list(p.columns) == PANEL_COLUMNS
    assert p["F1_expiry"].tolist() == ["2026-01-27", "2026-01-27", "2026-02-24"]
    assert p["F2_expiry"].tolist() == ["2026-02-24", "2026-02-24", "2026-03-30"]
    assert p["F3_expiry"].tolist()[2] is None            # empty slot stays missing
    assert p["cycle_id"].tolist() == [1, 1, 2]
    assert p["is_expiry_day"].tolist() == [False, True, False]


def test_zero_oi_on_priced_bars_carries_previous_oi_within_contract():
    ts = [datetime(2026, 1, 5, 9, 15 + 5 * i, tzinfo=IST) for i in range(5)]
    bars = [_bar(t, 100.0, oi=o) for t, o in zip(ts, [0, 500, 0, 0, 700])]   # leading 0 has nothing to carry
    c1 = bars_to_contract_frame(bars, "TEST", date(2026, 1, 27), "5minute", "NSE_FO|1")
    c2 = bars_to_contract_frame([_bar(ts[2], 90.0, oi=0)], "TEST", date(2026, 2, 24), "5minute", "NSE_FO|2")
    p = build_panel_from_contracts([c1, c2], "TEST", "5minute")
    assert list(p.columns) == PANEL_COLUMNS
    assert p["F1_oi"].tolist() == [0, 500, 500, 500, 700]
    assert p["F1_oi_filled"].tolist() == [False, False, True, True, False]
    # F2 contract: zero with no earlier print stays 0; slots with no candle are not filled
    assert p["F2_oi"].tolist() == [0, 0, 0, 0, 0]
    assert not p["F2_oi_filled"].any() and not p["F3_oi_filled"].any()


def test_missing_closes_are_flagged_and_never_filled():
    idx = pd.date_range("2026-01-05 09:15", periods=4, freq="5min", tz="Asia/Kolkata")
    panel = pd.DataFrame({
        "F1_close": [99.0, 99.0, np.nan, 99.0], "F2_close": [98.0, np.nan, np.nan, 98.0],
        "F3_close": np.nan, "SPOT_close": [100.0, 100.0, np.nan, np.nan],
        "F1_dte": 22, "F2_dte": 50, "F3_dte": 0, "F1_oi": 300, "F2_oi": 100, "is_expiry_day": False,
    }, index=idx)
    br = compute_borrow(panel, "5minute")
    assert br["missing_inputs"].tolist() == ["", "F2", "F1,F2,SPOT", "SPOT"]
    assert br["data_missing"].tolist() == [False, True, True, True]
    assert br["b12"].notna().tolist() == [True, False, False, True]   # NaN wherever F1 or F2 is missing
    assert br["b1"].notna().tolist() == [True, True, False, False]


def test_borrow_formulas_match_hand_calculation():
    idx = pd.DatetimeIndex([pd.Timestamp("2026-01-05 09:15", tz="Asia/Kolkata")])
    panel = pd.DataFrame({
        "F1_close": [99.0], "F2_close": [98.0], "F3_close": [np.nan], "SPOT_close": [100.0],
        "F1_dte": [22], "F2_dte": [50], "F3_dte": [0], "F1_oi": [300], "F2_oi": [100],
        "is_expiry_day": [False],
    }, index=idx)
    br = compute_borrow(panel, "5minute")
    t1, t2 = (22 + 1 - 1 / 75) / 365, (50 + 1 - 1 / 75) / 365
    assert br["b1"].iloc[0] == pytest.approx(0.0625 - np.log(99 / 100) / t1)
    assert br["b12"].iloc[0] == pytest.approx(0.0625 - np.log(98 / 99) / (t2 - t1))
    assert br["OI_ratio"].iloc[0] == pytest.approx(3.0)
    assert br["U_t"].iloc[0] == pytest.approx(3.0 / (22 + 1 - 1 / 75))
    assert br["spread_f1_f2"].iloc[0] == pytest.approx(1.0)
    assert br["spot_f1_pct"].iloc[0] == pytest.approx((100 - 99) / 100 * 100)   # 1.0 %
    assert br["f1_f2_pct"].iloc[0] == pytest.approx((99 - 98) / 99 * 100)       # 1.0101 %
    assert list(br.columns) == BORROW_COLUMNS
    assert bool(br["in_backwardation"].iloc[0])
    assert np.isnan(br["b3"].iloc[0])


# ── stored series: adding the *_pct columns without recomputing anything else ──

@pytest.fixture
def store(tmp_path):
    return ArcticStore(uri=f"lmdb://{tmp_path}?map_size=20MB")


def _seed(store, days=("2026-01-05", "2026-01-06")):
    """Panel + Kite spot + an OLD-layout borrow series (no *_pct columns, with a feat_ col)."""
    idx = pd.DatetimeIndex([pd.Timestamp(f"{d} {t}", tz="Asia/Kolkata")
                            for d in days for t in ("09:15", "09:20")])
    n = len(idx)
    panel = pd.DataFrame({
        "F1_open": 99.0, "F1_high": 99.0, "F1_low": 99.0, "F1_close": np.linspace(99, 100, n),
        "F2_close": np.linspace(98, 99, n), "F3_close": np.linspace(97, 98, n),
        "F1_volume": 1, "F2_volume": 1, "F3_volume": 1, "F1_oi": 300, "F2_oi": 100, "F3_oi": 10,
        "F1_dte": 22, "F2_dte": 50, "F3_dte": 78, "F1_expiry": "2026-01-27", "F2_expiry": "2026-02-24",
        "F3_expiry": "2026-03-30", "cycle_id": 1, "is_expiry_day": False,
    }, index=idx)
    store.write_futures("TST", "5minute", panel)
    store.write_spot("TST", "5minute", pd.DataFrame({"open": 100.0, "high": 100.0, "low": 100.0,
                     "close": np.linspace(100, 101, n), "volume": 1}, index=idx),
                     metadata={"source": "fetch_kite_equities"})
    old = compute_borrow(store.read_futures("TST", "5minute"), "5minute").drop(columns=["spot_f1_pct", "f1_f2_pct"])
    old["feat_x"] = np.arange(n, dtype=float)
    store.write("borrow_rates", "TST/5minute", old)
    return old


def test_add_missing_columns_keeps_every_stored_value(store):
    old = _seed(store)
    assert add_missing_borrow_columns(store, "TST", "5minute") == "added ['spot_f1_pct', 'f1_f2_pct']"
    new = store.read("borrow_rates", "TST/5minute")
    pd.testing.assert_frame_equal(new[old.columns], old, check_freq=False)
    full = store.read_borrow_rates("TST", "5minute")
    np.testing.assert_allclose(new["spot_f1_pct"], (full["SPOT_close"] - full["F1_close"]) / full["SPOT_close"] * 100)
    np.testing.assert_allclose(new["f1_f2_pct"], (full["F1_close"] - full["F2_close"]) / full["F1_close"] * 100)
    assert add_missing_borrow_columns(store, "TST", "5minute") == "up to date"


def test_trade_flags_mark_stale_legs_and_thin_early_f2():
    tz = "Asia/Kolkata"
    idx = pd.DatetimeIndex([pd.Timestamp(t, tz=tz) for t in (
        "2026-01-05 09:15", "2026-01-05 09:20", "2026-01-05 09:25", "2026-01-05 09:30", "2026-01-23 09:15")])
    panel = pd.DataFrame({
        "F1_close": 99.0, "F2_close": 98.0, "F3_close": np.nan, "SPOT_close": 100.0,
        "F1_dte": 22, "F2_dte": 50, "F3_dte": 0, "F1_oi": 300, "F2_oi": 100, "is_expiry_day": False,
        "F1_volume": [1000, 1000, 1000, 0, 1000],
        "F2_volume": [500, 0, 50, 50, 50],          # traded | stale F2 | thin | stale F1 | thin but late
        "F1_expiry": "2026-01-27",
    }, index=idx)
    br = compute_borrow(panel, "5minute")
    assert br["both_traded"].tolist() == [True, False, True, False, True]
    # 2026-01-05 is 15 trading days before the 27th (early); 2026-01-23 is 2 (inside the last 10)
    assert br["f2_thin_early"].tolist() == [False, False, True, False, False]


def test_add_missing_columns_adds_trade_flags_only(store):
    old = _seed(store)
    old = old.drop(columns=["both_traded", "f2_thin_early"])
    store.write("borrow_rates", "TST/5minute", old)
    assert add_missing_borrow_columns(store, "TST", "5minute") == \
        "added ['spot_f1_pct', 'f1_f2_pct', 'both_traded', 'f2_thin_early']"
    new = store.read("borrow_rates", "TST/5minute")
    pd.testing.assert_frame_equal(new[old.columns], old, check_freq=False)
    assert new["both_traded"].all() and not new["f2_thin_early"].any()   # seed: 1 lot each leg
    assert add_missing_borrow_columns(store, "TST", "5minute") == "up to date"


def test_tail_update_on_old_series_upgrades_schema_first(store):
    old = _seed(store)
    res = write_borrow(store, "TST", "5minute", pd.Timestamp("2026-01-06", tz="Asia/Kolkata"))
    assert res.startswith("tail")
    new = store.read("borrow_rates", "TST/5minute")
    assert {"spot_f1_pct", "f1_f2_pct", "feat_x"} <= set(new.columns)
    assert new["spot_f1_pct"].notna().all() and len(new) == len(old)


def test_full_rewrite_keeps_new_columns_and_feat(store):
    _seed(store)
    write_borrow(store, "TST", "5minute", None)
    new = store.read("borrow_rates", "TST/5minute")
    assert set(BORROW_COLUMNS) <= set(new.columns) and "feat_x" in new.columns


def test_fill_missing_computes_series_when_absent(store):
    _seed(store)
    store.delete("borrow_rates", "TST/5minute")
    assert fill_missing(store, "TST", "5minute", dry_run=True) == "would compute full borrow series"
    assert fill_missing(store, "TST", "5minute").startswith("computed full")
    assert set(BORROW_COLUMNS) <= set(store.read("borrow_rates", "TST/5minute").columns)
    assert fill_missing(store, "NOPE", "5minute") == "no panel - needs fetch-ticker --history"
