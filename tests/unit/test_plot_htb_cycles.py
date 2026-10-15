"""scripts/plot_htb_cycles.py helpers on synthetic data (no ArcticDB, no plotting)."""

import numpy as np
import pandas as pd

from scripts.plot_htb_cycles import display_smooth, expiry_window, roll_toggle, select_htb_expiries


def test_display_smooth_has_no_lag_and_bridges_only_short_gaps():
    x = np.exp(-0.5 * ((np.arange(101) - 50) / 8.0) ** 2)            # symmetric bump at bar 50
    assert int(np.nanargmax(display_smooth(x))) == 50                  # centred: peak stays put
    y = np.ones(60)
    y[10:13] = np.nan                                                 # 3-bar gap -> bridged
    y[30:40] = np.nan                                                 # 10-bar gap -> stays missing
    s = display_smooth(y)
    assert np.isfinite(s[10:13]).all()
    assert np.isnan(s[30:40]).all()
    assert np.allclose(s[np.isfinite(s)], 1.0)                        # constant stays constant


def test_toggle_needs_persistence_and_has_hysteresis():
    z = np.array([3, 3, 0, 3, 3, 3, 1, 0.2, 0.2, 1, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2], float)
    ro = np.ones_like(z)
    on = roll_toggle(z, ro, np.zeros(len(z)))
    assert not on[:5].any() and on[5]                 # 2 hits then a miss: needs 3 in a row
    assert on[5:15].all()                             # dip at bar 9 resets the off-count
    assert not on[15]                                 # 6 bars below the off-level -> OFF
    assert not roll_toggle(z, -ro, np.zeros(len(z))).any()   # relative OI must agree


def test_toggle_resets_at_a_roll():
    z = np.full(8, 3.0)
    on = roll_toggle(z, np.ones(8), np.array([1, 1, 1, 1, 2, 2, 2, 2]))
    assert on[3] and not on[4] and not on[5] and on[6]


def test_expiry_window_runs_from_previous_expiry_to_expiry():
    idx = pd.DatetimeIndex(pd.date_range("2026-01-26 09:15", "2026-02-25 15:25", freq="1D"), tz="Asia/Kolkata")
    m = expiry_window(idx, ["2026-01-27", "2026-02-24"], "2026-02-24")
    days = idx[m].strftime("%Y-%m-%d")
    assert days[0] == "2026-01-28" and days[-1] == "2026-02-24"


def test_selection_uses_final_week_only_and_merges_split_cycles():
    idx = pd.date_range("2026-08-01", periods=8, freq="D", tz="Asia/Kolkata")
    exp = pd.Series(["2026-08-25"] * 4 + ["2026-09-29"] * 2 + ["2026-08-25"] * 2, index=idx)
    td = pd.Series([10, 9, 4, 3, 30, 29, 1, 0], index=idx)
    b = pd.Series([0.9, 0.9, 0.2, 0.25, 0.8, 0.8, 0.35, 0.1], index=idx)   # 0.9 is early: ignored
    sel = select_htb_expiries(exp, td, b, threshold=0.30, week_td=4)
    assert sel.expiry.tolist() == ["2026-08-25"]                      # one row despite the split
    assert sel.peak.iloc[0] == 0.35 and sel.peak_td.iloc[0] == 1
