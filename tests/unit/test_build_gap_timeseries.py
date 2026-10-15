"""scripts/build_gap_timeseries.py gap columns on synthetic data (no ArcticDB needed)."""

import numpy as np
import pandas as pd

from scripts.build_gap_timeseries import gap_frame


def _series(values_per_cycle):
    vals, cyc = [], []
    for i, v in enumerate(values_per_cycle, 1):
        vals += list(v)
        cyc += [f"T|{i}"] * len(v)
    idx = pd.date_range("2026-01-05 09:15", periods=len(vals), freq="5min", tz="Asia/Kolkata")
    return pd.Series(np.asarray(vals, float), index=idx), pd.Series(cyc, index=idx)


def test_gaps_are_nan_wherever_b12_is_nan():
    b12, cyc = _series([[10, 12, np.nan, 15, 14, 13]])
    g = gap_frame(b12, cyc, "5minute")
    assert g.loc[b12.isna(), ["gap_0.1_2", "gap_0.1_1", "gap_0.05_1"]].isna().all().all()
    assert g.loc[b12.notna(), "gap_0.1_2"].notna().all()


def test_acc_is_second_difference_within_cycle():
    b12, cyc = _series([[10, 12, 11, 15, 14, 13], [5, 6, 7, 8]])
    g = gap_frame(b12, cyc, "5minute")
    gm = g["gap_0.05_1"]
    expect = gm.groupby(cyc).diff().groupby(cyc).diff()
    pd.testing.assert_series_equal(g["gap_0.05_1_acc"], expect, check_names=False)
    # first two bars of each cycle have no second difference: nothing crosses the roll
    assert g["gap_0.05_1_acc"].iloc[[0, 1, 6, 7]].isna().all()


def test_ewma_restarts_at_roll():
    b12, cyc = _series([[1.0] * 5, [9.0] * 5])
    g = gap_frame(b12, cyc, "5minute")
    assert (g["gap_0.1_2"].abs() < 1e-12).all()        # constant inside each cycle -> gap 0
