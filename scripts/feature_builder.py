"""
feature_builder.py
==================
Build ML feature columns and write them back into the ``borrow_rates`` ArcticDB
series ``borrow_rates/{TICKER}/{interval}`` — for ANY ticker at ANY interval,
every cycle, every timestamp.

Prerequisite: ``fetch-ticker TICKER --history --intervals <interval>`` (it writes the
futures panel and the b1/b2/b12 borrow-rate columns these features are built on).

All feature columns are numeric (float64) and named ``feat_*`` (plus the raw
``F1_volume``/``F2_volume`` pass-throughs). The script is idempotent: every run
drops the previously built feature columns, recomputes, and overwrites the
series (``prune_previous=True``). The separate ``cycle_summary`` symbol is untouched.

Interval handling: every rolling window is defined in TRADING TIME and converted
to bars with ``candles_per_day(interval)`` (see ``bar_windows``), so a 1-day b12
slope is 25 bars at 15minute, 75 at 5minute and (floored) 5 at 1day. At 15minute
the windows are exactly the original 250 / 25 / 10 / 5 bars.

IV features are built ONLY when option chains exist for the ticker at that interval
(``options_enriched/{TICKER}/{interval}/<expiry>``, from ``fetch-options``). Without
chains the IV group and the IV-derived features (``feat_25d_risk_reversal``,
``feat_residual_atm_iv``) are skipped and not written. Both stored chain schemas are
read (fetch_options.py ``strike_price``/``instrument_type``/``futures_price`` and the
older ``strike``/``opt_type``/``futures_ref_price``/``slot``).

Features
--------
Volume pass-through group — joined directly from the futures ArcticDB panel
(same ticker/interval, 1:1 timestamp-aligned with borrow_rates), no
derivation:

  0a. F1_volume   front-month traded volume, every cycle, every timestamp
  0b. F2_volume   next-month traded volume, every cycle, every timestamp

Volume-weighted roll-flow group — depends on F1_volume/F2_volume above and
feat_net_roll_flow (OI group below); emitted only where volume is positive
on both legs:

  0c. feat_Vshare    F2_volume / (F1_volume + F2_volume)            (in [0,1])
  0d. feat_volflow   feat_net_roll_flow * feat_Vshare

OI roll-flow group — derived purely from borrow_rates (F1_oi/F2_oi/F1_dte_star),
all diffs computed WITHIN each cycle so boundaries never bleed across:

  1.  feat_oi_concentration   C   = F1_oi / (F1_oi + F2_oi)      (in [0,1])
  2.  feat_roll_velocity      RV  = ΔC within cycle
  3.  feat_roll_acceleration  RA  = ΔRV within cycle
  4.  feat_relative_oi_change Flow= g2 − g1,  g = ΔOI / OI_prev
  5.  feat_net_roll_flow      NRF = ΔF2_oi − ΔF1_oi
  7.  feat_flow_intensity     NRF / (F1_oi + F2_oi)
  8.  feat_roll_deviation     C − E[C|DTE]   (E = mean C at the same int DTE over
                              EARLIER cycles only; NaN for the first cycle)

IV smile group (only when chains exist) — from the options_enriched chain of each
cycle's F1_expiry (the front month), evaluated at each timestamp:

  9.  feat_atm_iv             mean(iv) of ATM call & put (ATM = strike nearest
                              the chain's futures price)
  10. feat_otm_25d_call_iv    iv of CE whose delta nearest +0.25
  11. feat_otm_15d_call_iv    iv of CE whose delta nearest +0.15
  12. feat_otm_25d_put_iv     iv of PE whose delta nearest −0.25
  13. feat_otm_15d_put_iv     iv of PE whose delta nearest −0.15

NaN policy: leave NaN where an input is genuinely missing (F2_oi absent,
no options chain for a cycle's expiry, first-bar-of-cycle diffs).

Usage
-----
  build-features IREDA                                  # 15minute (default)
  build-features IREDA SBICARD --interval 5minute
  build-features POWERGRID --interval 5minute --dry-run # compute + summarise, no write
  build-features IREDA --interval 5minute --features feat_oi_concentration feat_Vshare
                                                        # only these; other feat_* kept
  (or: python scripts/feature_builder.py ...)
"""

from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from data.storage.arctic_store import ArcticStore
from options.pricing.dte_utils import candles_per_day

DEFAULT_INTERVAL = "15minute"
INTERVALS = ["1minute", "5minute", "15minute", "1day"]

# Delta targets for the OTM IV features (calls positive, puts negative)
DELTA_TARGETS = {
    "feat_otm_25d_call_iv": ("CE", +0.25),
    "feat_otm_15d_call_iv": ("CE", +0.15),
    "feat_otm_25d_put_iv":  ("PE", -0.25),
    "feat_otm_15d_put_iv":  ("PE", -0.15),
}

F1_VOLUME_FEATURE = "F1_volume"
F2_VOLUME_FEATURE = "F2_volume"
OI_FEATURES = [
    "feat_oi_concentration", "feat_roll_velocity", "feat_roll_acceleration",
    "feat_relative_oi_change", "feat_net_roll_flow",
    "feat_flow_intensity", "feat_roll_deviation",
]
VSHARE_FEATURE  = "feat_Vshare"    # F2_volume / (F1_volume + F2_volume)
VOLFLOW_FEATURE = "feat_volflow"   # feat_net_roll_flow * feat_Vshare
VOLUME_FLOW_FEATURES = [VSHARE_FEATURE, VOLFLOW_FEATURE]
IV_FEATURES = ["feat_atm_iv"] + list(DELTA_TARGETS.keys())
IV_DERIVED_FEATURES = ["feat_25d_risk_reversal", "feat_residual_atm_iv"]

# Batch-2 derived features — depend on b12 / IV cols computed above
DERIVED_FEATURES = [
    "feat_b12_percentile",      # percentile of b12 vs earlier cycles at same int DTE
    "feat_25d_risk_reversal",   # 25d call IV − 25d put IV
    "feat_beta_regression",     # β of within-cycle expanding OLS: log(F1_close) ~ F1_dte_star
    "feat_r2_regression",       # R² of the same within-cycle expanding OLS
    "feat_beta_rolling",        # β of 250-bar rolling OLS (≈10 trading days)
    "feat_r2_rolling",          # R² of the same rolling OLS
    "feat_beta_tsat",           # t-stat of β (within-cycle expanding OLS): β / SE(β)
    "feat_beta_reversal",       # feat_beta_rolling − feat_beta_regression
    "feat_b12_slope_1d",        # 25-bar rolling OLS slope of b12 (+ = building, − = collapsing)
    "feat_b12_accel",           # 5-bar change in slope (2nd derivative — inflection detector)
    "feat_residual_atm_iv",     # current atm_iv − median atm_iv of all prior cycles at same int-DTE
]
ALL_FEATURES = ([F1_VOLUME_FEATURE, F2_VOLUME_FEATURE] + OI_FEATURES + VOLUME_FLOW_FEATURES
                + IV_FEATURES + DERIVED_FEATURES)
# What a run without option chains builds (no IV group, no IV-derived features).
NON_IV_FEATURES = [f for f in ALL_FEATURES if f not in IV_FEATURES + IV_DERIVED_FEATURES]


# Rolling windows in TRADING DAYS; converted to bars per interval by bar_windows().
ROLLING_DAYS       = 10    # rolling OLS of log(F1) on DTE*
ROLLING_MIN_PTS    = 20    # require ≥20 valid pairs before emitting a rolling OLS value
B12_SLOPE_DAYS     = 1     # b12 slope window
B12_SLOPE_MIN_FRAC = 0.4   # min valid bars before emitting slope, as a share of the window
B12_ACCEL_FRAC     = 0.2   # acceleration lag = 1/5 trading day (75 min)


def bar_windows(interval: str) -> dict[str, int]:
    """Bar counts for every rolling window at *interval*.

    15minute (25 bars/day) gives the original fixed values: rolling 250,
    slope 25 (min 10), accel lag 5. Floors keep coarse intervals usable
    (slope window ≥ 5 bars, its min-points ≥ 3, lag ≥ 1 bar).
    """
    bpd = candles_per_day(interval)
    slope = max(5, B12_SLOPE_DAYS * bpd)
    return {
        "rolling": max(ROLLING_MIN_PTS, ROLLING_DAYS * bpd),
        "rolling_min": ROLLING_MIN_PTS,
        "slope": slope,
        "slope_min": max(3, round(B12_SLOPE_MIN_FRAC * slope)),
        "accel_lag": max(1, round(B12_ACCEL_FRAC * bpd)),
    }


# ── Prior-cycle (look-ahead-free) DTE statistics ──────────────────────────────

def _cycle_order(df: pd.DataFrame) -> pd.Series:
    """Chronological rank of each row's cycle (0 = earliest), by the cycle's first bar.

    Ordering by first timestamp rather than trusting cycle_id numbering keeps the
    "earlier cycles only" guarantee even if cycle_ids were ever renumbered.
    """
    first_ts = pd.Series(df.index, index=df.index).groupby(df["cycle_id"]).min()
    rank = first_ts.rank(method="first").astype(int) - 1
    return df["cycle_id"].map(rank)


def _prior_cycle_dte_mean(df: pd.DataFrame, col: str) -> pd.Series:
    """Mean of ``col`` over all bars of EARLIER cycles at the same integer F1_dte.

    Bar-weighted (same weighting as a pooled groupby mean), but only cycles that
    started before the row's own cycle contribute. Rows with no earlier-cycle
    observation at their DTE get NaN (always the case for the first cycle).
    """
    tmp = pd.DataFrame({
        "v": df[col].astype("float64"),
        "c": _cycle_order(df),
        "d": df["F1_dte"].round(0),
    }, index=df.index).dropna(subset=["v", "d"])
    agg = tmp.groupby(["d", "c"])["v"].agg(["sum", "count"])       # sorted by d, then c
    prior = agg.groupby(level="d").cumsum() - agg                   # exclude own cycle
    prior_mean = prior["sum"] / prior["count"].replace(0, np.nan)

    keys = pd.MultiIndex.from_arrays([df["F1_dte"].round(0), _cycle_order(df)])
    return pd.Series(prior_mean.reindex(keys).to_numpy(), index=df.index, dtype="float64")


def _prior_cycle_dte_percentile(df: pd.DataFrame, col: str) -> pd.Series:
    """Percentile (0–100) of each ``col`` value within the pooled distribution of
    EARLIER cycles' values at the same integer F1_dte.

    Ties count half (scipy ``percentileofscore(kind="mean")``), which matches the
    average-rank convention of the old pooled ``rank(pct=True)``. NaN when no
    earlier cycle has an observation at that DTE.
    """
    out = pd.Series(np.nan, index=df.index, dtype="float64")
    tmp = pd.DataFrame({
        "v": df[col].astype("float64"),
        "c": _cycle_order(df),
        "d": df["F1_dte"].round(0),
    }, index=df.index).dropna(subset=["v", "d"])

    for _d, grp in tmp.groupby("d", sort=False):
        hist = np.empty(0)
        for _c, cyc in grp.groupby("c", sort=True):                 # chronological
            vals = cyc["v"].to_numpy()
            if hist.size:
                lo = np.searchsorted(hist, vals, side="left")
                hi = np.searchsorted(hist, vals, side="right")
                out.loc[cyc.index] = (lo + hi) / 2.0 / hist.size * 100.0
            hist = np.sort(np.concatenate([hist, vals]))
    return out


# ── Volume pass-through (0a, 0b) ──────────────────────────────────────────────

def add_f1_volume(df: pd.DataFrame, fut: pd.DataFrame) -> pd.DataFrame:
    """Join F1_volume (front-month traded volume) from the futures panel.

    futures and borrow_rates share an identical timestamp index per
    (ticker, interval) — this is a direct 1:1 reindex, not a derivation.
    """
    df = df.copy()
    if "F1_volume" in fut.columns:
        df[F1_VOLUME_FEATURE] = fut["F1_volume"].reindex(df.index).astype("float64")
    else:
        df[F1_VOLUME_FEATURE] = np.nan
    return df


def add_f2_volume(df: pd.DataFrame, fut: pd.DataFrame) -> pd.DataFrame:
    """Join F2_volume (next-month traded volume) from the futures panel.

    futures and borrow_rates share an identical timestamp index per
    (ticker, interval) — this is a direct 1:1 reindex, not a derivation.
    """
    df = df.copy()
    if "F2_volume" in fut.columns:
        df[F2_VOLUME_FEATURE] = fut["F2_volume"].reindex(df.index).astype("float64")
    else:
        df[F2_VOLUME_FEATURE] = np.nan
    return df


# ── OI roll-flow features (1–8) ───────────────────────────────────────────────

def compute_oi_features(br: pd.DataFrame) -> pd.DataFrame:
    """Add feat_* OI-flow columns to a borrow_rates dataframe (in place copy)."""
    df = br.sort_index().copy()
    g = df.groupby("cycle_id", sort=False)

    f1 = df["F1_oi"].astype("float64")
    f2 = df["F2_oi"].astype("float64")

    # Validity guard — both legs must have positive OI for a meaningful split.
    valid = (f1 > 0) & (f2 > 0)

    # 1. OI concentration  C = F1 / (F1 + F2)
    C = (f1 / (f1 + f2)).where(valid)
    df["feat_oi_concentration"] = C

    # 2. Roll velocity  ΔC within cycle
    df["feat_roll_velocity"] = df.groupby("cycle_id", sort=False)["feat_oi_concentration"].diff()

    # 3. Roll acceleration  ΔRV within cycle
    df["feat_roll_acceleration"] = df.groupby("cycle_id", sort=False)["feat_roll_velocity"].diff()

    # 4. Relative OI change  g2 − g1,  g = ΔOI / OI_prev
    d_f1 = g["F1_oi"].diff()
    d_f2 = g["F2_oi"].diff()
    f1_prev = g["F1_oi"].shift(1).astype("float64")
    f2_prev = g["F2_oi"].shift(1).astype("float64")
    g1 = (d_f1 / f1_prev.replace(0, np.nan))
    g2 = (d_f2 / f2_prev.replace(0, np.nan))
    df["feat_relative_oi_change"] = (g2 - g1).where(valid)

    # 5. Net roll flow  ΔF2 − ΔF1
    nrf = (d_f2 - d_f1)
    df["feat_net_roll_flow"] = nrf.where(valid)

    # 7. Flow intensity  NRF / (F1 + F2) — net roll flow as a share of total OI
    #    (not scaled by days to expiry)
    df["feat_flow_intensity"] = (nrf / (f1 + f2).replace(0, np.nan)).where(valid)

    # 8. Roll deviation  C − E[C|DTE]   E = mean C at this int DTE over EARLIER cycles
    #    only (a pooled all-history mean would leak future cycles into every row).
    if "F1_dte" in df.columns:
        expected = _prior_cycle_dte_mean(df, "feat_oi_concentration")
        df["feat_roll_deviation"] = df["feat_oi_concentration"] - expected
    else:
        df["feat_roll_deviation"] = np.nan

    return df


# ── Volume-weighted roll-flow features ────────────────────────────────────────

def compute_volume_flow_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add feat_Vshare and feat_volflow.

    feat_Vshare  = F2_volume / (F1_volume + F2_volume)   (in [0,1])
    feat_volflow = feat_net_roll_flow * feat_Vshare

    Both require F1_volume/F2_volume (add_f1_volume/add_f2_volume) and
    feat_net_roll_flow (compute_oi_features) to already be present, and are
    only emitted where volume is positive on BOTH legs — rows with zero,
    missing, or negative volume are left NaN.
    """
    df = df.copy()

    f1v = df["F1_volume"].astype("float64") if "F1_volume" in df.columns else None
    f2v = df["F2_volume"].astype("float64") if "F2_volume" in df.columns else None

    if f1v is None or f2v is None:
        df[VSHARE_FEATURE] = np.nan
        df[VOLFLOW_FEATURE] = np.nan
        return df

    # Validity guard — both legs must have POSITIVE volume.
    valid_vol = (f1v > 0) & (f2v > 0)

    # feat_Vshare = F2_volume / (F1_volume + F2_volume)
    vshare = (f2v / (f1v + f2v)).where(valid_vol)
    df[VSHARE_FEATURE] = vshare

    # feat_volflow = feat_net_roll_flow * feat_Vshare
    if "feat_net_roll_flow" in df.columns:
        df[VOLFLOW_FEATURE] = (df["feat_net_roll_flow"] * vshare).where(valid_vol)
    else:
        df[VOLFLOW_FEATURE] = np.nan

    return df


# ── IV smile features (9–13) ──────────────────────────────────────────────────

def _normalise_chain(opt: pd.DataFrame) -> pd.DataFrame:
    """Map either stored chain schema onto ``strike`` / ``opt_type`` / ``futures_ref_price``.

    fetch_options.py writes ``strike_price`` / ``instrument_type`` / ``futures_price``;
    the older chains use ``strike`` / ``opt_type`` / ``futures_ref_price``.
    """
    rename = {"strike_price": "strike", "instrument_type": "opt_type",
              "futures_price": "futures_ref_price"}
    return opt.rename(columns={k: v for k, v in rename.items()
                               if k in opt.columns and v not in opt.columns})


def _atm_iv_per_ts(opt: pd.DataFrame) -> pd.Series:
    """Mean of ATM call & put IV per timestamp (ATM = nearest strike to fut ref).

    Returns a Series indexed by timestamp.  The option index has duplicate
    timestamps (one row per strike×type), so we reset to a unique RangeIndex
    before any groupby-idxmin to avoid ambiguous label lookups.
    """
    o = opt.dropna(subset=["iv", "strike", "futures_ref_price"]).reset_index()
    if o.empty:
        return pd.Series(dtype="float64")
    tscol = o.columns[0]                                # former index ("timestamp")
    o["sdist"] = (o["strike"] - o["futures_ref_price"]).abs()
    # ATM strike per timestamp = strike with min |strike - fut_ref|
    atm_rows = o.loc[o.groupby(tscol)["sdist"].idxmin()]
    atm_strike = atm_rows.set_index(tscol)["strike"]
    o = o.merge(atm_strike.rename("atm_strike"), left_on=tscol, right_index=True)
    sel = o[np.isclose(o["strike"], o["atm_strike"])]
    return sel.groupby(tscol)["iv"].mean()


def _delta_iv_per_ts(opt: pd.DataFrame, opt_type: str, target: float) -> pd.Series:
    """IV of the option (of given type) whose delta is nearest `target`, per ts.

    Returns a Series indexed by timestamp.
    """
    o = opt[opt["opt_type"] == opt_type].dropna(subset=["delta", "iv"]).reset_index()
    if o.empty:
        return pd.Series(dtype="float64")
    tscol = o.columns[0]
    o["ddist"] = (o["delta"] - target).abs()
    sel = o.loc[o.groupby(tscol)["ddist"].idxmin()]
    return sel.set_index(tscol)["iv"]


def compute_iv_features(
    store: ArcticStore, ticker: str, interval: str, df: pd.DataFrame,
    cycle_to_expiry: pd.Series,
) -> pd.DataFrame:
    """Fill the IV feature columns per cycle from the front-month option chain.

    Only called when ``options_enriched/{ticker}/{interval}`` holds chains. A cycle
    whose F1 expiry has no chain keeps NaN IV columns.
    """
    for col in IV_FEATURES:
        df[col] = np.nan

    available = set(store.list_option_expiries(ticker, interval))

    for cid, expiry in cycle_to_expiry.items():
        if expiry is None or (isinstance(expiry, float) and np.isnan(expiry)):
            continue
        expiry = str(expiry)
        if expiry not in available:
            continue   # no options chain for this cycle → IV cols stay NaN

        try:
            opt = store.read_options(ticker, interval, expiry)
        except Exception:
            continue
        if opt.empty:
            continue
        opt = _normalise_chain(opt)

        # front-month rows only (avoids same expiry when it was F2/F3 earlier)
        if "slot" in opt.columns:
            opt = opt[opt["slot"] == "F1"]
        if opt.empty:
            continue

        # restrict to this cycle's timestamps in the borrow panel
        cyc_idx = df.index[df["cycle_id"] == cid]
        opt = opt[opt.index.isin(cyc_idx)]
        if opt.empty:
            continue

        atm = _atm_iv_per_ts(opt)
        df.loc[atm.index, "feat_atm_iv"] = atm.values

        for col, (otype, target) in DELTA_TARGETS.items():
            s = _delta_iv_per_ts(opt, otype, target)
            if not s.empty:
                df.loc[s.index, col] = s.values

    return df


# ── b12 slope + acceleration helper ──────────────────────────────────────────

def _compute_b12_dynamics(df: pd.DataFrame, win: dict[str, int]) -> tuple[pd.Series, pd.Series]:
    """feat_b12_slope_1d  — 1-trading-day rolling OLS slope of b12 on integer time
    index (``win["slope"]`` bars), reset at every cycle boundary (avoids
    cross-cycle discontinuity artefacts).

    feat_b12_accel  — change in slope over 1/5 trading day (``win["accel_lag"]``
    bars; 2nd derivative of b12).
    Positive accel near a peak => still accelerating upward.
    Negative accel near a peak => momentum reversing => inflection / peak detected.

    Implementation: rolling.apply with np.polyfit(t, b12, 1)[0].
    x = [0, 1, ..., len(window)-1] (integer time within window).
    Cycle-aware: each cycle is processed independently so the reset at expiry
    (b12 collapses to ~0) cannot contaminate the early bars of the next cycle.
    """
    if "b12" not in df.columns or "cycle_id" not in df.columns:
        nan = pd.Series(np.nan, index=df.index, dtype="float64")
        return nan, nan

    slope_parts: list[pd.Series] = []
    accel_parts: list[pd.Series] = []

    for _cid, cyc in df.groupby("cycle_id", sort=False):
        b12_cyc = cyc["b12"]

        # rolling OLS slope: polyfit returns [slope, intercept]
        slope_cyc = b12_cyc.rolling(
            win["slope"], min_periods=win["slope_min"]
        ).apply(
            lambda v: np.polyfit(np.arange(len(v)), v, 1)[0], raw=True
        ).astype("float64")

        # acceleration: 5-bar change in slope (within cycle, so no cross-cycle diff)
        accel_cyc = slope_cyc.diff(win["accel_lag"]).astype("float64")

        slope_parts.append(slope_cyc)
        accel_parts.append(accel_cyc)

    slope_out = pd.concat(slope_parts).reindex(df.index)
    accel_out  = pd.concat(accel_parts).reindex(df.index)
    return slope_out, accel_out


# ── Residual ATM IV helper ────────────────────────────────────────────────────

def _compute_residual_atm_iv(df: pd.DataFrame) -> pd.Series:
    """feat_residual_atm_iv = feat_atm_iv − median(feat_atm_iv over ALL PRIOR cycles
    at the same integer DTE bucket).

    Algorithm (look-ahead-free):
      1. Round F1_dte to the nearest integer → dte_int bucket.
      2. For each unique dte_int, iterate cycles in ascending order.
         Maintain a running list of every feat_atm_iv observation seen so far
         from completed cycles.  The historical median for cycle C at DTE d is
         np.nanmedian(observations from cycles 1 … C-1 at DTE d).
      3. Subtract that median from the row's own feat_atm_iv.

    First cycle always gets NaN (no prior history).
    Rows where feat_atm_iv is NaN propagate NaN in the residual.
    """
    if "feat_atm_iv" not in df.columns or "F1_dte" not in df.columns:
        return pd.Series(np.nan, index=df.index, dtype="float64")

    dte_int = df["F1_dte"].round(0).astype(int)
    cid_int = df["cycle_id"].astype(int)

    # per-(cycle_id, dte_int) → list of atm_iv observations (NaN-dropped)
    tmp = df.assign(_dte=dte_int).dropna(subset=["feat_atm_iv"])
    cyc_dte_ivs = (
        tmp.groupby(["cycle_id", "_dte"])["feat_atm_iv"]
        .agg(list)
    )  # MultiIndex: (cycle_id, _dte)

    # Build (cycle_id, dte_int) → historical_median mapping
    hist_map: dict[tuple[int, int], float] = {}

    for dte_val in sorted(dte_int.unique()):
        # Slice all cycles that have observations at this DTE
        try:
            dte_series = cyc_dte_ivs.xs(int(dte_val), level="_dte")
        except KeyError:
            continue                    # no data at this DTE bucket at all
        dte_series = dte_series.sort_index()   # ascending cycle_id order

        running: list[float] = []
        for cyc_id, iv_list in dte_series.items():
            # historical median = median of PREVIOUS cycles' observations only
            hist_map[(int(cyc_id), int(dte_val))] = (
                float(np.nanmedian(running)) if running else np.nan
            )
            # after recording the lookup, accumulate this cycle into history
            running.extend(v for v in iv_list if not np.isnan(float(v)))

    # Vectorised lookup: build hist_median series aligned with df.index
    keys = list(zip(cid_int.values, dte_int.values))
    hist_median = pd.Series(
        [hist_map.get((int(k[0]), int(k[1])), np.nan) for k in keys],
        index=df.index,
        dtype="float64",
    )

    return (df["feat_atm_iv"] - hist_median).astype("float64")


# ── Derived / batch-2 features (14–19) ───────────────────────────────────────

def compute_derived_features(
    df: pd.DataFrame, interval: str = DEFAULT_INTERVAL, with_iv: bool | None = None,
    with_b12_dynamics: bool = True,
) -> pd.DataFrame:
    """Compute the derived features that depend on b12, F1 and (optionally) IV.

    Must be called AFTER compute_oi_features() (and compute_iv_features() when
    chains exist). *interval* sets the bar counts of the rolling windows
    (``bar_windows``). *with_iv* (default: auto — True when ``feat_atm_iv`` is
    present) controls whether the IV-derived ``feat_25d_risk_reversal`` and
    ``feat_residual_atm_iv`` are built; when False they are not added at all.
    *with_b12_dynamics=False* skips ``feat_b12_slope_1d`` / ``feat_b12_accel``
    (a per-cycle rolling polyfit — the slowest step) and does not add them.

    Features added
    --------------
    feat_b12_percentile    — percentile (0–100) of b12 against EARLIER cycles' b12 at
                             the same integer DTE (per ticker).  Look-ahead-free; NaN
                             for the first cycle.  Same prior-cycle pooling as
                             feat_roll_deviation.

    feat_25d_risk_reversal — feat_otm_25d_call_iv − feat_otm_25d_put_iv.  Positive =
                             call skew (contango skew), negative = put skew.

    feat_beta_regression   — β of within-cycle expanding OLS: log(F1_close) ~ F1_dte_star.
                             Resets at the start of every cycle (groupby cycle_id cumsum),
                             so beta/R² reflect only the current cycle's price-vs-DTE slope.
                             Implemented via vectorised cumulative sums (O(n), no loops).

    feat_r2_regression     — R² of the same within-cycle expanding OLS.

    feat_beta_rolling      — β of a 10-trading-day rolling OLS (250 bars at 15minute).
                             Cross-cycle sliding window, vectorised via rolling().sum().

    feat_r2_rolling        — R² of the same rolling OLS.
    """
    win = bar_windows(interval)
    if with_iv is None:
        with_iv = "feat_atm_iv" in df.columns

    # ── 14. b12 percentile (vs EARLIER cycles at the same int DTE only) ─────
    if "b12" in df.columns and "F1_dte" in df.columns:
        df["feat_b12_percentile"] = _prior_cycle_dte_percentile(df, "b12")
    else:
        df["feat_b12_percentile"] = np.nan

    # ── 15. 25-delta risk reversal (IV runs only) ─────────────────────────────
    if with_iv:
        if "feat_otm_25d_call_iv" in df.columns and "feat_otm_25d_put_iv" in df.columns:
            df["feat_25d_risk_reversal"] = (
                df["feat_otm_25d_call_iv"] - df["feat_otm_25d_put_iv"]
            ).astype("float64")
        else:
            df["feat_25d_risk_reversal"] = np.nan

    # ── 16–17 & 18–19: OLS helpers ───────────────────────────────────────────
    # Inputs for both regressions
    _log_price = np.log(df["F1_close"].astype("float64")) if "F1_close" in df.columns else None
    _dte_star  = df["F1_dte_star"].astype("float64") if "F1_dte_star" in df.columns else None

    if _log_price is not None and _dte_star is not None:
        y = _log_price
        x = _dte_star
        valid = y.notna() & x.notna()

        # Fill NaN with 0 so they contribute nothing to any sum
        xf = x.where(valid, 0.0)
        yf = y.where(valid, 0.0)
        vf = valid.astype("float64")

        # ── Vectorised OLS helper ─────────────────────────────────────────────
        def _ols_from_sums(
            n_s: pd.Series,
            sx: pd.Series,
            sy: pd.Series,
            sxx: pd.Series,
            sxy: pd.Series,
            syy: pd.Series,
            min_n: int,
        ) -> tuple[np.ndarray, np.ndarray]:
            """Return (beta, r2) arrays from running OLS moment sums."""
            denom_b = n_s * sxx - sx ** 2
            numer_b = n_s * sxy - sx * sy
            denom_r = denom_b * (n_s * syy - sy ** 2)
            numer_r = numer_b ** 2

            enough = (n_s >= min_n).values
            ok_b   = enough & (np.abs(denom_b.values) > 1e-10)
            ok_r   = ok_b   & (np.abs(denom_r.values) > 1e-10)

            # np.where evaluates both branches before masking; suppress expected 0/0
            with np.errstate(invalid="ignore", divide="ignore"):
                beta = np.where(ok_b, numer_b.values / denom_b.values, np.nan)
                r2   = np.where(ok_r, numer_r.values / denom_r.values, np.nan)
            return beta, r2

        # ── Vectorised slope t-stat helper ────────────────────────────────────
        def _ols_tstat_from_sums(
            n_s: pd.Series,
            sx: pd.Series,
            sy: pd.Series,
            sxx: pd.Series,
            sxy: pd.Series,
            syy: pd.Series,
            min_n: int,
        ) -> np.ndarray:
            """Return the slope t-statistic β/SE(β) from running OLS moment sums.

            t = Nb·sqrt(n−2) / sqrt(Db·Dy − Nb²) where
              Db = n·Σxx − (Σx)²,  Nb = n·Σxy − Σx·Σy,  Dy = n·Σyy − (Σy)².
            Needs n ≥ 3; the residual term (Db·Dy − Nb²) must be > 0 (R² < 1).
            """
            Db = n_s * sxx - sx ** 2
            Nb = n_s * sxy - sx * sy
            Dy = n_s * syy - sy ** 2
            resid = Db * Dy - Nb ** 2          # >= 0; ~0 when R² → 1

            enough = (n_s >= max(min_n, 3)).values
            ok = (enough
                  & (np.abs(Db.values) > 1e-10)
                  & (resid.values > 1e-12))

            with np.errstate(invalid="ignore", divide="ignore"):
                t = np.where(
                    ok,
                    Nb.values * np.sqrt(np.maximum(n_s.values - 2.0, 0.0))
                    / np.sqrt(resid.values),
                    np.nan,
                )
            return t

        # ── 16–17. Expanding OLS — resets at each cycle boundary ────────────
        # groupby(cycle_id).cumsum() restarts the running sums at the first
        # bar of every cycle, so beta/R² reflect only within-cycle history.
        cg = df["cycle_id"]
        def _ccumsum(s: pd.Series) -> pd.Series:
            return s.groupby(cg, sort=False).cumsum()

        n_e   = _ccumsum(vf)
        sx_e  = _ccumsum(xf)
        sy_e  = _ccumsum(yf)
        sxx_e = _ccumsum(xf ** 2)
        sxy_e = _ccumsum(xf * yf)
        syy_e = _ccumsum(yf ** 2)

        beta_e, r2_e = _ols_from_sums(n_e, sx_e, sy_e, sxx_e, sxy_e, syy_e, min_n=3)
        df["feat_beta_regression"] = beta_e.astype("float64")
        df["feat_r2_regression"]   = r2_e.astype("float64")

        # ── 20. β t-stat of the within-cycle expanding OLS (β / SE(β)) ───────
        tstat_e = _ols_tstat_from_sums(n_e, sx_e, sy_e, sxx_e, sxy_e, syy_e,
                                       min_n=3)
        df["feat_beta_tsat"] = tstat_e.astype("float64")

        # ── 18–19. Rolling OLS (10-trading-day cross-cycle sliding window) ───
        rw = dict(window=win["rolling"], min_periods=1)
        n_r   = vf.rolling(**rw).sum()
        sx_r  = xf.rolling(**rw).sum()
        sy_r  = yf.rolling(**rw).sum()
        sxx_r = (xf ** 2).rolling(**rw).sum()
        sxy_r = (xf * yf).rolling(**rw).sum()
        syy_r = (yf ** 2).rolling(**rw).sum()

        beta_r, r2_r = _ols_from_sums(n_r, sx_r, sy_r, sxx_r, sxy_r, syy_r,
                                       min_n=win["rolling_min"])
        df["feat_beta_rolling"] = beta_r.astype("float64")
        df["feat_r2_rolling"]   = r2_r.astype("float64")

        # ── 21. β reversal: rolling slope − within-cycle slope ──────────────
        df["feat_beta_reversal"] = (
            df["feat_beta_rolling"] - df["feat_beta_regression"]
        ).astype("float64")
    else:
        for col in ["feat_beta_regression", "feat_r2_regression",
                    "feat_beta_rolling", "feat_r2_rolling",
                    "feat_beta_tsat", "feat_beta_reversal"]:
            df[col] = np.nan

    # ── 23–24. b12 slope (1 trading day) and acceleration ────────────────────
    if with_b12_dynamics:
        slope_1d, accel = _compute_b12_dynamics(df, win)
        df["feat_b12_slope_1d"] = slope_1d
        df["feat_b12_accel"]    = accel

    # ── 25. Residual ATM IV vs prior-cycle historical median at same DTE ─────
    if with_iv:
        df["feat_residual_atm_iv"] = _compute_residual_atm_iv(df)

    return df


# ── Per-ticker driver ─────────────────────────────────────────────────────────

def build_for_ticker(store: ArcticStore, ticker: str, interval: str, dry_run: bool,
                     features: list[str] | None = None) -> None:
    """Build features for one ticker. *features* (default: all) limits what is
    computed and written; stored feature columns outside it are kept as they are."""
    print("\n" + "=" * 78)
    print(f"  {ticker}  —  building features on borrow_rates/{ticker}/{interval}")
    print("=" * 78)

    try:
        br = store.read_borrow_rates(ticker, interval)
    except Exception as e:
        print(f"  LOAD ERROR (borrow_rates): {e}")
        return
    if br is None or br.empty:
        print(f"  no borrow_rates data — run `fetch-ticker {ticker} --history "
              f"--intervals {interval}` first; skipping")
        return

    # Drop previously built feature columns so a group skipped this run (e.g. IV
    # when chains are gone) can never leave stale values behind. With --features,
    # only the requested columns are dropped; the rest are carried over untouched.
    stored_order = list(br.columns)
    stale = [c for c in br.columns if c.startswith("feat_")
             or c in (F1_VOLUME_FEATURE, F2_VOLUME_FEATURE)]
    # (F1/F2_volume are never stored here — the read joins them from futures)
    kept = (br[[c for c in stale if c.startswith("feat_") and c not in features]]
            if features is not None else None)
    br = br.drop(columns=stale)
    wanted = set(features) if features is not None else set(ALL_FEATURES)

    # cycle_id → F1_expiry map (borrow_rates lacks F1_expiry; futures has it)
    fut = pd.DataFrame()
    try:
        fut = store.read_futures(ticker, interval, with_spot=False)
        cycle_to_expiry = fut.groupby("cycle_id")["F1_expiry"].first()
    except Exception as e:
        print(f"  WARN: could not build cycle→expiry map ({e}); IV features skipped")
        cycle_to_expiry = pd.Series(dtype="object")

    n0 = len(br)
    if not fut.empty:
        df = add_f1_volume(br, fut)
        df = add_f2_volume(df, fut)
    else:
        df = br.copy()
        df[F1_VOLUME_FEATURE] = np.nan
        df[F2_VOLUME_FEATURE] = np.nan
        print(f"  WARN: no futures data ({ticker}); F1_volume/F2_volume left NaN")
    df = compute_oi_features(df)
    df = compute_volume_flow_features(df)

    if wanted & set(IV_FEATURES + IV_DERIVED_FEATURES):
        n_chains = len(store.list_option_expiries(ticker, interval))
        with_iv = n_chains > 0 and not cycle_to_expiry.empty
        if with_iv:
            print(f"  option chains: {n_chains} expiries in options_enriched/{ticker}/{interval} "
                  "— building IV features")
            df = compute_iv_features(store, ticker, interval, df, cycle_to_expiry)
        else:
            print(f"  option chains: none in options_enriched/{ticker}/{interval} "
                  "— IV features skipped")
    else:
        with_iv = False
    df = compute_derived_features(
        df, interval, with_iv=with_iv,
        with_b12_dynamics=bool(wanted & {"feat_b12_slope_1d", "feat_b12_accel"}),
    )
    built = [f for f in (ALL_FEATURES if with_iv else NON_IV_FEATURES) if f in wanted]
    if features is not None:
        # keep the requested features plus the carried-over ones; drop by-products
        df = df.drop(columns=[c for c in ALL_FEATURES if c in df.columns and c not in built])
        df = df.join(kept)
        # keep the stored column order; columns new to the series go at the end
        df = df[[c for c in stored_order if c in df.columns]
                + [c for c in df.columns if c not in stored_order]]

    # ── summary ───────────────────────────────────────────────────────────────
    w = bar_windows(interval)
    print(f"  rows: {n0}  |  cycles: {df['cycle_id'].nunique()}  |  "
          f"{df.index[0].date()} → {df.index[-1].date()}  |  windows (bars): {w}")
    print(f"\n  {'feature':<26}{'non-null%':>10}{'mean':>12}{'min':>12}{'max':>12}")
    print("  " + "-" * 70)
    for col in built:
        s = df[col]
        nn = s.notna().mean() * 100
        if s.notna().any():
            print(f"  {col:<26}{nn:>9.1f}%{s.mean():>12.4f}{s.min():>12.4f}{s.max():>12.4f}")
        else:
            print(f"  {col:<26}{nn:>9.1f}%{'—':>12}{'—':>12}{'—':>12}")

    if dry_run:
        print("\n  DRY RUN — not writing back to ArcticDB")
        return

    stored = built + ([c for c in kept.columns if c not in built] if features is not None else [])
    store.write_borrow_rates(
        ticker, interval, df, prune_previous=True,
        metadata={"features_added": stored, "interval": interval, "iv_features": with_iv,
                  "windows_bars": w, "built": date.today().isoformat()},
    )
    print(f"\n  WROTE {len(df.columns)} columns "
          f"({len(built)} features) -> borrow_rates/{ticker}/{interval}")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("tickers", nargs="+", metavar="TICKER",
                    help="One or more tickers, e.g. IREDA SBICARD.")
    ap.add_argument("--interval", default=DEFAULT_INTERVAL, choices=INTERVALS,
                    help=f"borrow_rates interval to build on (default {DEFAULT_INTERVAL}).")
    ap.add_argument("--dry-run", action="store_true",
                    help="Compute and summarise but do NOT write back.")
    ap.add_argument("--features", nargs="+", metavar="FEATURE",
                    help="Build only these features (default: all). Stored feature "
                         "columns not listed are kept as they are.")
    args = ap.parse_args(argv)

    if args.features is not None:
        unknown = [f for f in args.features if f not in ALL_FEATURES]
        if unknown:
            ap.error(f"unknown feature(s): {', '.join(unknown)}. "
                     f"Choose from: {', '.join(ALL_FEATURES)}")

    tickers = [t.upper() for t in args.tickers]
    store = ArcticStore()
    for t in tickers:
        build_for_ticker(store, t, args.interval, dry_run=args.dry_run,
                         features=args.features)

    print("\n" + "=" * 78)
    print(f"  feature_builder complete — {len(tickers)} ticker(s) @ {args.interval}"
          f"{' (dry run)' if args.dry_run else ''}")
    print("=" * 78)


if __name__ == "__main__":
    main()
