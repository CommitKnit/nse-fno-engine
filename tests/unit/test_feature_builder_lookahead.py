"""feat_roll_deviation / feat_b12_percentile must only use EARLIER cycles."""

import numpy as np
import pandas as pd
import pytest

from scripts.feature_builder import (
    compute_derived_features,
    compute_oi_features,
)


def _panel(n_cycles: int = 4, bars_per_dte: int = 3, seed: int = 0) -> pd.DataFrame:
    """Synthetic 15-min-style borrow panel: each cycle counts F1_dte 3 → 0."""
    rng = np.random.default_rng(seed)
    rows, ts = [], pd.Timestamp("2025-01-01 09:15", tz="Asia/Kolkata")
    for cid in range(1, n_cycles + 1):
        for dte in (3, 2, 1, 0):
            for _ in range(bars_per_dte):
                rows.append({
                    "timestamp": ts,
                    "cycle_id": cid,
                    "F1_dte": dte,
                    "F1_oi": float(rng.integers(100, 1000)),
                    "F2_oi": float(rng.integers(100, 1000)),
                    "b12": float(rng.normal()),
                })
                ts += pd.Timedelta(minutes=15)
    return pd.DataFrame(rows).set_index("timestamp")


def _features(df: pd.DataFrame) -> pd.DataFrame:
    return compute_derived_features(compute_oi_features(df))


@pytest.mark.parametrize("col", ["feat_roll_deviation", "feat_b12_percentile"])
def test_first_cycle_is_nan(col):
    out = _features(_panel())
    assert out.loc[out["cycle_id"] == 1, col].isna().all()
    assert out.loc[out["cycle_id"] > 1, col].notna().all()


@pytest.mark.parametrize("col", ["feat_roll_deviation", "feat_b12_percentile"])
def test_changing_last_cycle_does_not_move_earlier_rows(col):
    base = _panel()
    shocked = base.copy()
    last = shocked["cycle_id"] == shocked["cycle_id"].max()
    shocked.loc[last, ["F1_oi", "b12"]] = [1e9, 1e6]

    a, b = _features(base), _features(shocked)
    earlier = base["cycle_id"] < base["cycle_id"].max()
    pd.testing.assert_series_equal(a.loc[earlier, col], b.loc[earlier, col])


def test_roll_deviation_matches_prior_cycle_mean():
    out = _features(_panel())
    c = out["feat_oi_concentration"]
    for dte in (3, 0):
        prior = c[(out["cycle_id"] < 3) & (out["F1_dte"] == dte)].mean()
        row = out[(out["cycle_id"] == 3) & (out["F1_dte"] == dte)].iloc[0]
        assert row["feat_roll_deviation"] == pytest.approx(row["feat_oi_concentration"] - prior)


def test_b12_percentile_against_prior_cycles():
    out = _features(_panel())
    row = out[(out["cycle_id"] == 4) & (out["F1_dte"] == 2)].iloc[0]
    hist = out.loc[(out["cycle_id"] < 4) & (out["F1_dte"] == 2), "b12"].to_numpy()
    expected = ((hist < row["b12"]).sum() + 0.5 * (hist == row["b12"]).sum()) / hist.size * 100
    assert row["feat_b12_percentile"] == pytest.approx(expected)
