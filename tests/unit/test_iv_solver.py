"""Tests for IV solver — Newton-Raphson + Brent hybrid."""

import math
import numpy as np
import pytest

from options.pricing.black76 import black76_price
from options.pricing.iv_solver import solve_iv, iv_newton, iv_brent, solve_iv_chain


class TestSolveIV:
    """Test round-trip: price → IV → price should match."""

    def test_atm_call_roundtrip(self):
        """Known IV → price → solve IV → should recover original IV."""
        F, K, T, r = 23500, 23500, 30 / 365, 0.065
        true_iv = 0.15
        price = black76_price(F, K, T, true_iv, r, is_call=True)
        solved = solve_iv(price, F, K, T, r, is_call=True)
        assert solved is not None
        assert abs(solved - true_iv) < 1e-4

    def test_atm_put_roundtrip(self):
        F, K, T, r = 23500, 23500, 30 / 365, 0.065
        true_iv = 0.18
        price = black76_price(F, K, T, true_iv, r, is_call=False)
        solved = solve_iv(price, F, K, T, r, is_call=False)
        assert solved is not None
        assert abs(solved - true_iv) < 1e-4

    def test_otm_call_roundtrip(self):
        """OTM options should also round-trip correctly."""
        F, K, T, r = 23500, 24000, 30 / 365, 0.065
        true_iv = 0.16
        price = black76_price(F, K, T, true_iv, r, is_call=True)
        solved = solve_iv(price, F, K, T, r, is_call=True)
        assert solved is not None
        assert abs(solved - true_iv) < 1e-3

    def test_otm_put_roundtrip(self):
        F, K, T, r = 23500, 23000, 30 / 365, 0.065
        true_iv = 0.17
        price = black76_price(F, K, T, true_iv, r, is_call=False)
        solved = solve_iv(price, F, K, T, r, is_call=False)
        assert solved is not None
        assert abs(solved - true_iv) < 1e-3

    def test_high_vol_roundtrip(self):
        """High IV (50%) should still converge."""
        F, K, T, r = 23500, 23500, 30 / 365, 0.065
        true_iv = 0.50
        price = black76_price(F, K, T, true_iv, r, is_call=True)
        solved = solve_iv(price, F, K, T, r, is_call=True)
        assert solved is not None
        assert abs(solved - true_iv) < 1e-3

    def test_short_dte_roundtrip(self):
        """Near-expiry (3 days) should still solve."""
        F, K, T, r = 23500, 23500, 3 / 365, 0.065
        true_iv = 0.20
        price = black76_price(F, K, T, true_iv, r, is_call=True)
        solved = solve_iv(price, F, K, T, r, is_call=True)
        assert solved is not None
        assert abs(solved - true_iv) < 1e-3

    def test_banknifty_scale(self):
        """Should work for BANKNIFTY price levels too."""
        F, K, T, r = 50000, 50000, 14 / 365, 0.065
        true_iv = 0.14
        price = black76_price(F, K, T, true_iv, r, is_call=True)
        solved = solve_iv(price, F, K, T, r, is_call=True)
        assert solved is not None
        assert abs(solved - true_iv) < 1e-4


class TestEdgeCases:
    """IV solver edge cases — bad data, boundary conditions."""

    def test_zero_price_returns_none(self):
        solved = solve_iv(0.0, 23500, 23500, 30 / 365, 0.065, is_call=True)
        assert solved is None

    def test_negative_price_returns_none(self):
        solved = solve_iv(-10.0, 23500, 23500, 30 / 365, 0.065, is_call=True)
        assert solved is None

    def test_zero_time_returns_none(self):
        solved = solve_iv(100.0, 23500, 23500, 0, 0.065, is_call=True)
        assert solved is None

    def test_price_below_intrinsic_returns_none(self):
        """Price below intrinsic → arbitrage, no valid IV."""
        # ITM call: intrinsic = 23500 - 23000 = 500
        # Market price of 400 is below intrinsic
        solved = solve_iv(400.0, 23500, 23000, 30 / 365, 0.065, is_call=True)
        assert solved is None

    def test_deep_otm_near_zero_price(self):
        """Very small OTM price — might return None or very high IV."""
        solved = solve_iv(0.10, 23500, 26000, 7 / 365, 0.065, is_call=True)
        # Either returns a very high IV or None — both acceptable
        if solved is not None:
            assert solved > 0.05  # should be non-trivial vol


class TestNewtonVsBrent:
    """Verify both methods agree when both converge."""

    def test_both_methods_agree(self):
        F, K, T, r = 23500, 23500, 30 / 365, 0.065
        true_iv = 0.15
        price = black76_price(F, K, T, true_iv, r, is_call=True)

        iv_n = iv_newton(price, F, K, T, r, is_call=True)
        iv_b = iv_brent(price, F, K, T, r, is_call=True)

        assert iv_n is not None
        assert iv_b is not None
        assert abs(iv_n - iv_b) < 1e-4


class TestChainSolver:
    """Test vectorized chain IV solver."""

    def test_solve_chain(self):
        """Solve IV for a small chain of 5 strikes."""
        F, T, r = 23500.0, 30 / 365, 0.065
        strikes = np.array([23000, 23250, 23500, 23750, 24000], dtype=float)
        true_ivs = np.array([0.16, 0.155, 0.15, 0.155, 0.16])
        is_call = np.array([True, True, True, True, True])

        prices = np.array([
            black76_price(F, K, T, iv, r, is_call=True)
            for K, iv in zip(strikes, true_ivs)
        ])

        solved = solve_iv_chain(prices, F, strikes, T, r, is_call)

        assert np.all(~np.isnan(solved))
        np.testing.assert_allclose(solved, true_ivs, atol=1e-3)

    def test_chain_with_bad_data(self):
        """Chain with some zero-price entries → NaN for those."""
        F, T, r = 23500.0, 30 / 365, 0.065
        strikes = np.array([23000, 23500, 24000], dtype=float)
        prices = np.array([0.0, 300.0, 0.0])  # first and last are bad
        is_call = np.array([True, True, True])

        solved = solve_iv_chain(prices, F, strikes, T, r, is_call)

        assert np.isnan(solved[0])
        assert not np.isnan(solved[1])
        assert np.isnan(solved[2])

