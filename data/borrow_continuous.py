"""Continuous (cross-cycle) borrow-rate series built from the stored b1/b2/b12 inputs.

Two families, side by side, one row per bar:

Roll-stitched front series — the stored ``b1``/``b2``/``b12`` (clipped >= 0, see
src/update_intraday.py::calc_borrow_rates) as one series across cycles, plus
``cycle``, a ``roll`` flag on the first bar of each new F1 contract, and
``b1_ex_expiry`` = ``b1`` masked inside the last ``NEAR_EXPIRY_DAYS`` of the cycle,
where ``t1 -> 0`` makes b1 blow up and then clip to 0. Roll jumps are left in:
they are real contract changes.

Constant-maturity series — no roll jumps by construction. Each bar's term
structure is the cumulative borrow ``B(t) = r*t - ln(F(t)/S)`` at the points
(0, 0), (t1, B1), (t2, B2), (t3, B3); B is linearly interpolated in t (i.e.
piecewise-flat forward borrow) and read off at fixed tenors:

    b1_cm30      = B(30d) / 30d          spot -> 30 calendar days   (continuous b1)
    b2_cm60      = B(60d) / 60d          spot -> 60 calendar days   (continuous b2)
    b12_cm30_60  = (B(60d) - B(30d)) / 30d   30 -> 60-day forward   (continuous b12)

These are NOT clipped (a negative value means the futures trade rich to carry).
Interpolating B rather than b keeps the series stable as t1 -> 0. A tenor past the
last valid point is extrapolated flat-forward for at most ``MAX_EXTRAP_DAYS``,
otherwise NaN. Rows with no spot are NaN (all three need S).

Time is ``F*_dte_star / 365`` and ``r = 0.0625``, matching the stored b-series.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

R = 0.0625
TENORS_DAYS = (30, 60)
NEAR_EXPIRY_DAYS = 2.0
MAX_EXTRAP_DAYS = 10.0


def _cum_borrow(F, S, t, r: float = R) -> np.ndarray:
    F, S, t = (np.asarray(x, dtype=float) for x in (F, S, t))
    with np.errstate(divide="ignore", invalid="ignore"):
        out = r * t - np.log(F / S)
    return np.where((F > 0) & (S > 0) & (t > 0), out, np.nan)


def cm_cum_borrow(t_pts: np.ndarray, b_pts: np.ndarray, tau: float) -> np.ndarray:
    """Cumulative borrow at tenor *tau* (years) per row.

    ``t_pts``/``b_pts``: (n, k) arrays of curve points per row, NaN = missing.
    A (0, 0) anchor is prepended. Linear in t between points; flat-forward
    extrapolation past the last point for at most MAX_EXTRAP_DAYS.
    """
    n = t_pts.shape[0]
    out = np.full(n, np.nan)
    max_ext = MAX_EXTRAP_DAYS / 365.0
    for i in range(n):
        ok = ~(np.isnan(t_pts[i]) | np.isnan(b_pts[i]))
        if not ok.any():
            continue
        t = np.concatenate(([0.0], t_pts[i][ok]))
        b = np.concatenate(([0.0], b_pts[i][ok]))
        order = np.argsort(t, kind="stable")
        t, b = t[order], b[order]
        if tau <= t[-1]:
            out[i] = np.interp(tau, t, b)
        elif tau - t[-1] <= max_ext and len(t) >= 2:
            fwd = (b[-1] - b[-2]) / (t[-1] - t[-2])
            out[i] = b[-1] + fwd * (tau - t[-1])
    return out


def build_continuous(br: pd.DataFrame, r: float = R) -> pd.DataFrame:
    """Continuous series from a ``read_borrow_rates(..., joined=True)`` frame.

    Needs columns: b1, b2, b12, cycle_id, F1/F2/F3_close, SPOT_close, F1/F2/F3_dte_star.
    """
    df = br.sort_index()
    out = pd.DataFrame(index=df.index)

    # ── roll-stitched front series ───────────────────────────────────────────
    out["cycle"] = df["cycle_id"].astype("int64")
    out["roll"] = out["cycle"].ne(out["cycle"].shift()) & out["cycle"].shift().notna()
    out["f1_dte_star"] = df["F1_dte_star"]
    out["f2_dte_star"] = df["F2_dte_star"]
    for c in ("b1", "b2", "b12"):
        out[c] = df[c]
    out["b1_near_expiry"] = df["F1_dte_star"] <= NEAR_EXPIRY_DAYS
    out["b1_ex_expiry"] = df["b1"].where(~out["b1_near_expiry"])

    # ── constant maturity ────────────────────────────────────────────────────
    S = df["SPOT_close"].to_numpy(float)
    t_pts = np.column_stack([df[f"F{k}_dte_star"].to_numpy(float) / 365.0 for k in (1, 2, 3)])
    b_pts = np.column_stack([_cum_borrow(df[f"F{k}_close"], S, t_pts[:, k - 1], r) for k in (1, 2, 3)])
    t_pts = np.where(np.isnan(b_pts), np.nan, t_pts)
    lo, hi = (d / 365.0 for d in TENORS_DAYS)
    B_lo, B_hi = cm_cum_borrow(t_pts, b_pts, lo), cm_cum_borrow(t_pts, b_pts, hi)
    out["b1_cm30"] = B_lo / lo
    out["b2_cm60"] = B_hi / hi
    out["b12_cm30_60"] = (B_hi - B_lo) / (hi - lo)
    return out
