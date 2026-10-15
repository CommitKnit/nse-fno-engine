"""feat_roll_fraction is dropped; feat_flow_intensity = net roll flow / total OI."""

import numpy as np
import pandas as pd

from scripts.feature_builder import ALL_FEATURES, compute_oi_features
from tests.unit.test_feature_builder_lookahead import _panel


def test_roll_fraction_not_built():
    out = compute_oi_features(_panel())
    assert "feat_roll_fraction" not in ALL_FEATURES
    assert "feat_roll_fraction" not in out.columns
    assert "feat_flow_intensity" in out.columns


def test_flow_intensity_definition():
    df = _panel()
    out = compute_oi_features(df)

    g = df.groupby("cycle_id", sort=False)
    nrf = g["F2_oi"].diff() - g["F1_oi"].diff()
    valid = (df["F1_oi"] > 0) & (df["F2_oi"] > 0)
    expected = (nrf / (df["F1_oi"] + df["F2_oi"])).where(valid).rename("feat_flow_intensity")

    pd.testing.assert_series_equal(out["feat_flow_intensity"], expected, check_dtype=False)
    assert out["feat_flow_intensity"].notna().any()
    assert np.isfinite(out["feat_flow_intensity"].dropna()).all()


def test_flow_intensity_ignores_days_to_expiry():
    df = _panel()
    base = compute_oi_features(df)["feat_flow_intensity"]
    df["F1_dte_star"] = 0.01                      # near expiry: used to blow the value up
    df["F1_dte"] = 0
    pd.testing.assert_series_equal(compute_oi_features(df)["feat_flow_intensity"], base)
