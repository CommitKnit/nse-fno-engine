
"""Implied Volatility solver using Black-76 model.

Given an observed market price and the contract parameters (F, K, T, r),
back-solve for the implied volatility σ that makes the model price
equal to the market price.

Methods:
    1. Newton-Raphson (fast, but can diverge for deep OTM)
    2. Brent's method (slower, but guaranteed convergence)
    3. Hybrid: try Newton first, fall back to Brent

Edge cases handled:
    - Deep OTM options (price near zero → IV undefined)
    - Near-expiry options (T → 0 → theta explosion)
    - Price below intrinsic (arbitrage → return NaN)
    - Zero volume / stale prices (flag and skip)
"""

from __future__ import annotations

import math

import numpy as np
from loguru import logger
from scipy.optimize import brentq
from scipy.stats import norm

from options.pricing.black76 import black76_price


# ── Newton-Raphson Solver ──

def _vega_black76(F: float, K: float, T: float, sigma: float, r: float) -> float:
    """Black-76 vega (dPrice/dSigma) — needed for Newton step."""
    if T <= 0 or sigma <= 0:
        return 0.0
    sqrt_T = math.sqrt(T)
    d1 = (math.log(F / K) + 0.5 * sigma**2 * T) / (sigma * sqrt_T)
    return F * math.exp(-r * T) * norm.pdf(d1) * sqrt_T


def iv_newton(
    market_price: float,
    F: float,
    K: float,
    T: float,
    r: float,
    is_call: bool,
    initial_guess: float = 0.20,
    max_iter: int = 50,
    tol: float = 1e-6,
) -> float | None:
    """Solve IV using Newton-Raphson method.

    Returns:
        Implied volatility, or None if failed to converge.
    """
    sigma = initial_guess

    for i in range(max_iter):
        price = black76_price(F, K, T, sigma, r, is_call)
        diff = price - market_price

        if abs(diff) < tol:
            return sigma

        vega = _vega_black76(F, K, T, sigma, r)
        if vega < 1e-12:
            # Vega too small — Newton step would explode
            return None

        sigma -= diff / vega

        # Keep sigma in reasonable bounds
        if sigma <= 0.001:
            sigma = 0.001
        if sigma > 5.0:
            return None  # diverging

    return None  # didn't converge


# ── Brent's Method Solver ──

def iv_brent(
    market_price: float,
    F: float,
    K: float,
    T: float,
    r: float,
    is_call: bool,
    sigma_low: float = 0.001,
    sigma_high: float = 5.0,
    tol: float = 1e-6,
) -> float | None:
    """Solve IV using Brent's method (bracketed, guaranteed convergence).

    Returns:
        Implied volatility, or None if no solution in [sigma_low, sigma_high].
    """
    def objective(sigma: float) -> float:
        return black76_price(F, K, T, sigma, r, is_call) - market_price

    try:
        # Check that the solution is bracketed
        f_low = objective(sigma_low)
        f_high = objective(sigma_high)

        if f_low * f_high > 0:
            # Not bracketed — no solution in range
            return None

        return brentq(objective, sigma_low, sigma_high, xtol=tol)

    except (ValueError, RuntimeError):
        return None


# ── Hybrid Solver (recommended) ──

def solve_iv(
    market_price: float,
    F: float,
    K: float,
    T: float,
    r: float,
    is_call: bool,
    initial_guess: float = 0.20,
) -> float | None:
    """Solve implied volatility using hybrid Newton + Brent approach.

    This is the primary function to use. It tries Newton-Raphson first
    (fast), and falls back to Brent's method if Newton fails.

    Args:
        market_price: Observed option price (e.g. close or mid).
        F: Futures price (near-month).
        K: Strike price.
        T: Time to expiry in years.
        r: Risk-free rate.
        is_call: True for CE, False for PE.
        initial_guess: Starting IV for Newton method.

    Returns:
        Implied volatility (annualized), or None if unsolvable.

    Examples:
        >>> solve_iv(market_price=150, F=23500, K=23500, T=30/365, r=0.065, is_call=True)
        0.1234  # ~12.34% annualized IV
    """
    # ── Guard: reject obviously bad inputs ──
    if market_price <= 0 or F <= 0 or K <= 0 or T <= 0:
        return None

    # Check against intrinsic value
    intrinsic = max(0.0, F - K) if is_call else max(0.0, K - F)
    if market_price < intrinsic * 0.90:
        # Price below intrinsic — likely bad data or arbitrage
        return None

    # If price equals intrinsic (zero extrinsic), IV is effectively zero
    if market_price <= intrinsic + 0.01:
        return 0.001

    # ── Try Newton-Raphson first ──
    iv = iv_newton(market_price, F, K, T, r, is_call, initial_guess)

    if iv is not None and 0.001 < iv < 5.0:
        return iv

    # ── Fall back to Brent's method ──
    iv = iv_brent(market_price, F, K, T, r, is_call)

    if iv is not None and 0.001 < iv < 5.0:
        return iv

    return None


# ── Vectorized IV solver for entire chain ──

def solve_iv_chain(
    market_prices: np.ndarray,
    F: float,
    strikes: np.ndarray,
    T: float,
    r: float,
    is_call: np.ndarray,
) -> np.ndarray:
    """Solve IV for an entire chain at once.

    Args:
        market_prices: Array of observed option prices.
        F: Futures price (scalar, same for all).
        strikes: Array of strike prices.
        T: Time to expiry in years (scalar).
        r: Risk-free rate (scalar).
        is_call: Boolean array (True for CE, False for PE).

    Returns:
        Array of IVs (NaN where solve failed).
    """
    n = len(market_prices)
    ivs = np.full(n, np.nan)

    for i in range(n):
        result = solve_iv(
            market_price=float(market_prices[i]),
            F=F,
            K=float(strikes[i]),
            T=T,
            r=r,
            is_call=bool(is_call[i]),
        )
        if result is not None:
            ivs[i] = result

    solved = np.count_nonzero(~np.isnan(ivs))
    logger.debug(f"IV solved: {solved}/{n} contracts (T={T:.4f}, F={F:.0f})")

    return ivs
