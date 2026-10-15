
"""Black-76 pricing model for options on futures.

Black-76 is the standard model for pricing European options on futures.
It replaces the spot price S×e^(rT) in Black-Scholes with the futures
price F directly, eliminating the need to estimate dividend yield and
cost of carry — the market has already priced those into the future.

Formulas:
    Call = e^(-rT) × [F × N(d1) - K × N(d2)]
    Put  = e^(-rT) × [K × N(-d2) - F × N(-d1)]

    d1 = [ln(F/K) + (σ²/2)T] / (σ√T)
    d2 = d1 - σ√T

Where:
    F = futures price
    K = strike price
    T = time to expiry in years
    r = risk-free rate
    σ = implied volatility
    N = cumulative standard normal distribution
"""

from __future__ import annotations

import math

import numpy as np
from scipy.stats import norm


# ── Black-76 Price ──

def black76_price(
    F: float,
    K: float,
    T: float,
    sigma: float,
    r: float,
    is_call: bool,
) -> float:
    """Calculate option price using Black-76 model.

    Args:
        F: Futures price.
        K: Strike price.
        T: Time to expiry in years (e.g. 30/365 for 30 days).
        sigma: Implied volatility (annualized, e.g. 0.15 for 15%).
        r: Risk-free rate (annualized, e.g. 0.065 for 6.5%).
        is_call: True for call, False for put.

    Returns:
        Theoretical option price.

    Raises:
        ValueError: If inputs are invalid (negative, zero where inappropriate).
    """
    if T <= 0:
        # At or past expiry — return intrinsic value
        if is_call:
            return max(0.0, F - K)
        return max(0.0, K - F)

    if sigma <= 0 or F <= 0 or K <= 0:
        raise ValueError(f"Invalid inputs: F={F}, K={K}, sigma={sigma}")

    sqrt_T = math.sqrt(T)
    d1 = (math.log(F / K) + 0.5 * sigma**2 * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T

    discount = math.exp(-r * T)

    if is_call:
        return discount * (F * norm.cdf(d1) - K * norm.cdf(d2))
    else:
        return discount * (K * norm.cdf(-d2) - F * norm.cdf(-d1))


# ── Greeks ──

def black76_delta(
    F: float, K: float, T: float, sigma: float, r: float, is_call: bool
) -> float:
    """Delta — rate of change of option price with respect to futures price."""
    if T <= 0:
        if is_call:
            return 1.0 if F > K else 0.0
        return -1.0 if F < K else 0.0

    sqrt_T = math.sqrt(T)
    d1 = (math.log(F / K) + 0.5 * sigma**2 * T) / (sigma * sqrt_T)
    discount = math.exp(-r * T)

    if is_call:
        return discount * norm.cdf(d1)
    else:
        return discount * (norm.cdf(d1) - 1.0)


def black76_gamma(
    F: float, K: float, T: float, sigma: float, r: float
) -> float:
    """Gamma — rate of change of delta w.r.t. futures price.

    Same for both calls and puts.
    """
    if T <= 0 or sigma <= 0:
        return 0.0

    sqrt_T = math.sqrt(T)
    d1 = (math.log(F / K) + 0.5 * sigma**2 * T) / (sigma * sqrt_T)
    discount = math.exp(-r * T)

    return discount * norm.pdf(d1) / (F * sigma * sqrt_T)


def black76_theta(
    F: float, K: float, T: float, sigma: float, r: float, is_call: bool
) -> float:
    """Theta — time decay per calendar day (negative = losing value).

    Returns theta in price units per day (divide annual theta by 365).
    """
    if T <= 0:
        return 0.0

    sqrt_T = math.sqrt(T)
    d1 = (math.log(F / K) + 0.5 * sigma**2 * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T
    discount = math.exp(-r * T)

    # Annual theta
    term1 = -(F * norm.pdf(d1) * sigma) / (2 * sqrt_T)

    if is_call:
        term2 = r * discount * (F * norm.cdf(d1) - K * norm.cdf(d2))
    else:
        term2 = r * discount * (K * norm.cdf(-d2) - F * norm.cdf(-d1))

    annual_theta = discount * term1 - term2

    # Convert to per-day
    return annual_theta / 365.0


def black76_vega(
    F: float, K: float, T: float, sigma: float, r: float
) -> float:
    """Vega — sensitivity to 1% change in IV.

    Same for both calls and puts.
    Returns vega per 1 percentage point of vol (i.e., multiply by 0.01).
    """
    if T <= 0 or sigma <= 0:
        return 0.0

    sqrt_T = math.sqrt(T)
    d1 = (math.log(F / K) + 0.5 * sigma**2 * T) / (sigma * sqrt_T)
    discount = math.exp(-r * T)

    # Vega per 1% vol change
    return discount * F * norm.pdf(d1) * sqrt_T * 0.01


def black76_rho(
    F: float, K: float, T: float, sigma: float, r: float, is_call: bool
) -> float:
    """Rho — sensitivity to interest rate change."""
    if T <= 0:
        return 0.0

    price = black76_price(F, K, T, sigma, r, is_call)
    return -T * price


def black76_greeks(
    F: float, K: float, T: float, sigma: float, r: float, is_call: bool
) -> dict[str, float]:
    """Compute all Greeks at once.

    Returns:
        Dict with keys: price, delta, gamma, theta, vega, rho.
    """
    return {
        "price": black76_price(F, K, T, sigma, r, is_call),
        "delta": black76_delta(F, K, T, sigma, r, is_call),
        "gamma": black76_gamma(F, K, T, sigma, r),
        "theta": black76_theta(F, K, T, sigma, r, is_call),
        "vega": black76_vega(F, K, T, sigma, r),
        "rho": black76_rho(F, K, T, sigma, r, is_call),
    }


# ── Vectorized versions for DataFrames ──

def black76_price_vec(
    F: np.ndarray,
    K: np.ndarray,
    T: np.ndarray,
    sigma: np.ndarray,
    r: float,
    is_call: np.ndarray,
) -> np.ndarray:
    """Vectorized Black-76 price for entire chain at once.

    All inputs should be numpy arrays of the same length,
    except r which is a scalar.
    """
    sqrt_T = np.sqrt(np.maximum(T, 1e-10))
    d1 = (np.log(F / K) + 0.5 * sigma**2 * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T
    discount = np.exp(-r * T)

    call_price = discount * (F * norm.cdf(d1) - K * norm.cdf(d2))
    put_price = discount * (K * norm.cdf(-d2) - F * norm.cdf(-d1))

    # Handle at/past-expiry
    expired = T <= 0
    call_intrinsic = np.maximum(F - K, 0)
    put_intrinsic = np.maximum(K - F, 0)

    prices = np.where(is_call, call_price, put_price)
    intrinsic = np.where(is_call, call_intrinsic, put_intrinsic)

    return np.where(expired, intrinsic, prices)


def black76_greeks_vec(
    F: np.ndarray,
    K: np.ndarray,
    T: np.ndarray,
    sigma: np.ndarray,
    r: float,
    is_call: np.ndarray,
) -> dict[str, np.ndarray]:
    """Vectorized Black-76 Greeks for an entire chain at once.

    All inputs should be numpy arrays of the same length, except ``r`` which is
    a scalar. Mirrors :func:`black76_price_vec`'s expiry handling; ``T <= 0``
    rows get the same closed-form expiry values the scalar ``black76_*``
    functions return (delta: the moneyness step function; gamma/theta/vega/
    rho: 0.0). Rows with ``sigma <= 0`` (degenerate — shouldn't occur once
    callers only pass solver-bounded IV) get NaN delta and 0.0 for the rest,
    rather than dividing by zero. Same scaling as the scalars: vega per 1 vol
    point (x0.01), theta per calendar day (annual theta / 365).

    Returns:
        Dict with keys: delta, gamma, theta, vega, rho (no ``price`` — use
        :func:`black76_price_vec` for that).
    """
    F = np.asarray(F, dtype=float)
    K = np.asarray(K, dtype=float)
    T = np.asarray(T, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    is_call = np.asarray(is_call, dtype=bool)

    expired = T <= 0
    no_vol = (~expired) & (sigma <= 0)
    unsafe = expired | no_vol

    # Safe substitutes so the shared d1/d2 computation never divides by zero
    # or logs a non-positive number — every unsafe row is overwritten below.
    T_s = np.where(unsafe, 1.0, T)
    sigma_s = np.where(unsafe, 1.0, sigma)

    sqrt_T = np.sqrt(T_s)
    d1 = (np.log(F / K) + 0.5 * sigma_s**2 * T_s) / (sigma_s * sqrt_T)
    d2 = d1 - sigma_s * sqrt_T
    discount = np.exp(-r * T_s)

    pdf_d1 = norm.pdf(d1)
    cdf_d1, cdf_d2 = norm.cdf(d1), norm.cdf(d2)
    cdf_neg_d1, cdf_neg_d2 = norm.cdf(-d1), norm.cdf(-d2)

    # ── Delta — expiry: moneyness step function; no_vol: undefined (NaN) ──
    delta_live = np.where(is_call, discount * cdf_d1, discount * (cdf_d1 - 1.0))
    delta_exp = np.where(is_call, np.where(F > K, 1.0, 0.0), np.where(F < K, -1.0, 0.0))
    delta = np.where(expired, delta_exp, np.where(no_vol, np.nan, delta_live))

    # ── Gamma (same for call/put) ──
    gamma = np.where(unsafe, 0.0, discount * pdf_d1 / (F * sigma_s * sqrt_T))

    # ── Theta (per calendar day) ──
    term1 = -(F * pdf_d1 * sigma_s) / (2 * sqrt_T)
    term2 = np.where(
        is_call,
        r * discount * (F * cdf_d1 - K * cdf_d2),
        r * discount * (K * cdf_neg_d2 - F * cdf_neg_d1),
    )
    theta = np.where(unsafe, 0.0, (discount * term1 - term2) / 365.0)

    # ── Vega (per 1 vol point, same for call/put) ──
    vega = np.where(unsafe, 0.0, discount * F * pdf_d1 * sqrt_T * 0.01)

    # ── Rho ──
    price_live = np.where(
        is_call,
        discount * (F * cdf_d1 - K * cdf_d2),
        discount * (K * cdf_neg_d2 - F * cdf_neg_d1),
    )
    rho = np.where(unsafe, 0.0, -T_s * price_live)

    return {"delta": delta, "gamma": gamma, "theta": theta, "vega": vega, "rho": rho}
