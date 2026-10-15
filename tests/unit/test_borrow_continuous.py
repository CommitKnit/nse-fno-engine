"""Constant-maturity borrow interpolation (data/borrow_continuous.py)."""

import numpy as np
import pandas as pd
import pytest

from data.borrow_continuous import R, build_continuous, cm_cum_borrow


def _frame(b: float, t1_days: list[float], cycles: list[int], S: float = 100.0) -> pd.DataFrame:
    """Panel whose futures all price a flat borrow b: F = S*exp((r-b)*t)."""
    t1 = np.array(t1_days, dtype=float)
    t2, t3 = t1 + 28, t1 + 56
    idx = pd.date_range("2026-01-01", periods=len(t1), freq="D", tz="Asia/Kolkata")
    F = {k: S * np.exp((R - b) * t / 365) for k, t in ((1, t1), (2, t2), (3, t3))}
    return pd.DataFrame({
        "cycle_id": cycles, "SPOT_close": S,
        "F1_close": F[1], "F2_close": F[2], "F3_close": F[3],
        "F1_dte_star": t1, "F2_dte_star": t2, "F3_dte_star": t3,
        "b1": b, "b2": b, "b12": b,
    }, index=idx)


def test_flat_borrow_is_recovered_at_every_tenor_and_across_a_roll():
    df = _frame(0.20, [3.0, 2.0, 1.0, 0.1, 30.0, 29.0], [1, 1, 1, 1, 2, 2])
    out = build_continuous(df)
    for c in ("b1_cm30", "b2_cm60", "b12_cm30_60"):
        assert out[c].to_numpy() == pytest.approx(0.20, abs=1e-9)
    assert out["roll"].tolist() == [False, False, False, False, True, False]


def test_near_expiry_mask_and_negative_borrow_unclipped():
    df = _frame(-0.05, [5.0, 1.5], [1, 1])
    out = build_continuous(df)
    assert out["b1_ex_expiry"].isna().tolist() == [False, True]
    assert out["b1_cm30"].to_numpy() == pytest.approx(-0.05, abs=1e-9)


def test_cm_reads_the_point_when_tenor_equals_a_contract_dte():
    t = np.array([[30 / 365, 58 / 365, 86 / 365]])
    B = np.array([[0.01, 0.03, 0.04]])
    assert cm_cum_borrow(t, B, 30 / 365)[0] == pytest.approx(0.01)
    assert cm_cum_borrow(t, B, 44 / 365)[0] == pytest.approx(0.02)


def test_no_extrapolation_beyond_limit():
    t = np.array([[20 / 365, 40 / 365, np.nan]])
    B = np.array([[0.01, 0.02, np.nan]])
    assert np.isnan(cm_cum_borrow(t, B, 60 / 365)[0])          # 20 days past last point
    assert not np.isnan(cm_cum_borrow(t, B, 45 / 365)[0])      # 5 days -> flat forward
