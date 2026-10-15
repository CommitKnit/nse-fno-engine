"""scripts/ewma_b12_signal.py on synthetic data (no ArcticDB needed)."""

import numpy as np
import pandas as pd
import pytest

from scripts.ewma_b12_signal import (
    ewma_by_cycle,
    future_change,
    halflife_to_alpha,
    score,
    split_cycles,
)


def _frame(values_per_cycle):
    """One row per bar; consecutive 5-minute timestamps; cycle labels T|1, T|2, ..."""
    vals, cyc = [], []
    for i, v in enumerate(values_per_cycle, 1):
        vals += list(v)
        cyc += [f"T|{i}"] * len(v)
    idx = pd.date_range("2026-01-05 09:15", periods=len(vals), freq="5min", tz="Asia/Kolkata")
    return pd.DataFrame({"b12": np.asarray(vals, float), "cycle": cyc}, index=idx)


def test_halflife_to_alpha_uses_bars_per_day():
    bars, alpha = halflife_to_alpha(1.0, "5minute")
    assert bars == 75
    assert alpha == pytest.approx(1 - 0.5 ** (1 / 75))
    assert halflife_to_alpha(1.0, "15minute")[0] == 25
    # the weight really halves after `bars` bars
    assert (1 - alpha) ** bars == pytest.approx(0.5)


def test_ewma_restarts_at_each_cycle():
    df = _frame([[1.0] * 10, [5.0] * 10])
    e = ewma_by_cycle(df.b12, df.cycle, hl_bars=5)
    assert e.iloc[9] == pytest.approx(1.0)
    assert e.iloc[10] == pytest.approx(5.0)        # first bar of cycle 2 does not remember cycle 1


def test_ewma_is_nan_on_missing_bars_and_resumes_from_pre_gap_state():
    df = _frame([[1.0, 2.0, np.nan, np.nan, 3.0]])
    e = ewma_by_cycle(df.b12, df.cycle, hl_bars=5)
    assert e.iloc[2:4].isna().all()                    # no value carried through the gap
    full = df.b12.dropna().ewm(halflife=5, adjust=True).mean()
    assert e.iloc[4] == pytest.approx(full.iloc[-1])   # state continues as if the gap were absent


def test_future_change_never_crosses_a_roll():
    df = _frame([[0, 1, 2, 3], [10, 11, 12, 13]])
    fc = future_change(df.b12, df.cycle, 2)
    assert fc.tolist()[:2] == [2.0, 2.0]
    assert np.isnan(fc.iloc[2]) and np.isnan(fc.iloc[3])   # t+2 would be in cycle 2
    assert fc.iloc[4] == 2.0


def test_divergence_predicts_a_smooth_trend_but_not_noise():
    rng = np.random.default_rng(0)
    n = 600
    trend = [0.2 + 0.1 * np.sin(np.arange(n) / 40 + k) for k in range(6)]    # smooth swings
    noise = [0.2 + rng.normal(0, 0.01, n) for _ in range(6)]                   # no structure
    for series, expect_pos in ((trend, True), (noise, False)):
        df = _frame(series)
        div = ewma_by_cycle(df.b12, df.cycle, 3) - ewma_by_cycle(df.b12, df.cycle, 30)
        fut = future_change(df.b12, df.cycle, 10)
        r = score(div, fut, df.cycle)
        if expect_pos:
            assert r["ic_mean"] > 0.3 and r["hit_rate"] > 0.6
        else:
            assert r["ic_mean"] < 0.15                               # noise: no positive edge


def test_split_puts_latest_cycles_in_test():
    df = _frame([[0.1] * 5 for _ in range(10)])
    sp = split_cycles(df, 0.3)
    test_cycles = sorted(df.cycle[sp == "test"].unique())
    assert test_cycles == ["T|10", "T|8", "T|9"]
    assert df.cycle[sp == "train"].nunique() == 7
