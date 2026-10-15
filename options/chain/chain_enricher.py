"""Chain enricher — attach IV, Greeks, and synthetic bid-ask to option chains.

Takes a raw OptionChain (contracts with OHLCV only) and enriches each
contract with: implied volatility, all Greeks (delta, gamma, theta, vega),
moneyness classification, and synthetic bid-ask spread.

This is the bridge between raw data and strategy-ready data.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
from loguru import logger

from config.settings import settings
from data.models.option_chain import OptionChain
from data.models.option_contract import OptionContract
from options.pricing.black76 import black76_greeks, black76_greeks_vec, black76_price_vec
from options.pricing.dte_utils import year_fraction
from options.pricing.iv_solver import solve_iv
from options.execution.slippage_model import SlippageModel


class ChainEnricher:
    """Enrich option chains with IV, Greeks, and synthetic spreads.

    Usage:
        enricher = ChainEnricher()
        enriched_chain = enricher.enrich(raw_chain)
        # Now every contract in the chain has .iv, .delta, .gamma, etc.
    """

    def __init__(
        self,
        risk_free_rate: float | None = None,
        slippage_model: SlippageModel | None = None,
        min_volume_for_iv: int = 0,
        min_oi_for_iv: int = 0,
    ):
        """
        Args:
            risk_free_rate: Annualized risk-free rate. Defaults to settings value.
            slippage_model: Model for synthetic bid-ask estimation.
            min_volume_for_iv: Skip IV computation for contracts with volume below this.
            min_oi_for_iv: Skip IV computation for contracts with OI below this.
        """
        self.r = risk_free_rate or settings.RISK_FREE_RATE
        self.slippage = slippage_model or SlippageModel()
        self.min_volume = min_volume_for_iv
        self.min_oi = min_oi_for_iv

    def enrich(self, chain: OptionChain) -> OptionChain:
        """Enrich all contracts in a chain with IV, Greeks, and spreads.

        Modifies contracts in-place and returns the same chain.

        Args:
            chain: OptionChain with OHLCV data but no IV/Greeks.

        Returns:
            The same chain with IV, Greeks, and synthetic bid-ask populated.
        """
        if not chain.contracts:
            return chain

        F = chain.futures_price
        T = year_fraction(chain.dte)  # trading-day DTE -> year fraction (/252)

        if T <= 0 or F <= 0:
            logger.warning(
                f"Cannot enrich chain: T={T:.4f}, F={F:.0f} "
                f"(expiry={chain.expiry}, ts={chain.timestamp})"
            )
            return chain

        solved_count = 0
        failed_count = 0

        for contract in chain.contracts:
            # Skip stale/illiquid contracts
            if contract.volume < self.min_volume or contract.oi < self.min_oi:
                continue

            # Skip contracts with zero or negative close
            if contract.close <= 0:
                continue

            # Solve IV
            iv = solve_iv(
                market_price=contract.close,
                F=F,
                K=contract.strike,
                T=T,
                r=self.r,
                is_call=contract.is_call,
            )

            if iv is None:
                failed_count += 1
                continue

            # Compute Greeks using solved IV
            greeks = black76_greeks(
                F=F,
                K=contract.strike,
                T=T,
                sigma=iv,
                r=self.r,
                is_call=contract.is_call,
            )

            # Attach to contract
            contract.iv = iv
            contract.delta = greeks["delta"]
            contract.gamma = greeks["gamma"]
            contract.theta = greeks["theta"]
            contract.vega = greeks["vega"]
            contract.rho = greeks["rho"]

            # Compute synthetic bid-ask
            bid, ask = self.slippage.estimate_spread(contract, F)
            contract.synthetic_bid = bid
            contract.synthetic_ask = ask

            solved_count += 1

        total = len(chain.contracts)
        logger.debug(
            f"Enriched chain {chain.underlying_symbol} {chain.expiry} "
            f"DTE={chain.dte}: {solved_count}/{total} solved, "
            f"{failed_count} failed"
        )

        return chain

    def enrich_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Single-pass IV solve, stale-price repair, and vectorized Greeks.

        Classifies every row by how "real" its price is, in priority order,
        and solves IV exactly once from whichever price that classification
        settles on. Replaces the old three-touch flow (per-chain
        :meth:`enrich` on possibly-stale closes -> ``OptionsDataCleaner.
        repair_stale_prices`` Tier 1/Tier 2 -> a per-row re-enrich loop) that
        could solve IV against a stale price and then silently propagate
        that pre-correction value if a both-stale row's forward-fill
        happened to read it before the re-enrich loop got around to fixing it.

        Classification per row (mutually exclusive):

        * **traded**            — this leg AND its opposite leg at the same
          ``(timestamp, expiry, strike)`` both traded (``volume > 0``). IV
          solved directly on the real close.
        * **put-call parity**   — this leg is stale (``volume == 0``) but the
          opposite leg traded. ``close`` is repriced via put-call parity on
          the opposite leg's live close, *then* IV is solved on that
          repaired close — in the same pass, before anything reads it.
        * **forward-fill**      — both legs stale. IV is forward-filled from
          this instrument's own most recent solved IV (which, for the row
          immediately after a put-call-parity repair, is now that row's
          *corrected* IV — this is the ordering fix), then ``close`` is
          repriced via Black-76 from the carried IV.
        * **dropped**           — both legs stale and no prior IV exists for
          this instrument (nothing to carry forward).

        Sets ``is_synthetic`` (forward-fill) and
        ``synthetic_via_putcallparity`` (put-call parity) with their existing
        meaning; a traded row has neither. A row whose classification calls
        for a solve but where :func:`solve_iv` itself fails keeps NaN
        iv/greeks (kept, not dropped — matches ``post_solver_cleanup``/
        ``filter_delta_band``'s existing NaN-keeping philosophy) and is
        never forward-filled *into* — only genuinely both-stale rows are.

        Greeks are computed once, vectorized, from the final iv for every
        row. ``synthetic_bid``/``synthetic_ask`` are (re)computed once from
        the final close via the existing slippage model, for every row with
        a solved iv.

        Args:
            df: Flat per-(timestamp, expiry, strike, type) DataFrame — the
                output of ``_chains_to_df`` on *unenriched* chains
                (``iv``/``delta``/... columns present but not populated).

        Returns:
            Enriched copy; both-stale rows with no prior IV are dropped,
            index reset. Output schema is identical to the previous
            enrich + repair_stale_prices pipeline.
        """
        if df.empty:
            return df.copy()

        df = df.copy()
        n_initial = len(df)

        df["is_synthetic"] = False
        df["synthetic_via_putcallparity"] = False

        itype = df["instrument_type"].astype(str).str.upper().str.strip()
        is_call = itype.isin({"CE", "CALL"}).to_numpy()
        is_put = itype.isin({"PE", "PUT"}).to_numpy()
        stale_self = (pd.to_numeric(df["volume"], errors="coerce").fillna(0) == 0).to_numpy()

        # ── Pair legs on (timestamp, expiry, strike) so PCP can look sideways ──
        pair_keys = ["timestamp", "expiry", "strike_price"]
        calls = (
            df.loc[is_call, pair_keys + ["close"]]
            .assign(_stale=stale_self[is_call])
            .groupby(pair_keys, sort=False).first().reset_index()
            .rename(columns={"close": "_ce_close", "_stale": "_ce_stale"})
        )
        puts = (
            df.loc[is_put, pair_keys + ["close"]]
            .assign(_stale=stale_self[is_put])
            .groupby(pair_keys, sort=False).first().reset_index()
            .rename(columns={"close": "_pe_close", "_stale": "_pe_stale"})
        )
        df = df.merge(pd.merge(calls, puts, on=pair_keys, how="outer"), on=pair_keys, how="left")

        # ── Sort by instrument then time so the forward-fill below
        #    propagates correctly; every array from here on is derived
        #    AFTER this sort (row order just changed). ────────────────────
        group_col = "instrument_key" if "instrument_key" in df.columns else None
        if group_col is None:
            df["_grp"] = "all"
            group_col = "_grp"
        df = df.sort_values([group_col, "timestamp"]).reset_index(drop=True)

        itype = df["instrument_type"].astype(str).str.upper().str.strip()
        is_call = itype.isin({"CE", "CALL"}).to_numpy()
        is_put = itype.isin({"PE", "PUT"}).to_numpy()
        stale_self = (pd.to_numeric(df["volume"], errors="coerce").fillna(0) == 0).to_numpy()

        F = pd.to_numeric(df["futures_price"], errors="coerce").to_numpy(dtype=float)
        K = pd.to_numeric(df["strike_price"], errors="coerce").to_numpy(dtype=float)
        dte = pd.to_numeric(df["dte"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        T = np.maximum(year_fraction(dte), 0.0)
        disc = np.exp(-self.r * T)

        pe_live = df["_pe_stale"].eq(False).to_numpy() & df["_pe_close"].notna().to_numpy()
        ce_live = df["_ce_stale"].eq(False).to_numpy() & df["_ce_close"].notna().to_numpy()
        call_pcp = is_call & stale_self & pe_live
        put_pcp = is_put & stale_self & ce_live
        pcp_mask = call_pcp | put_pcp

        # ── Put-call-parity reprice ──────────────────────────────────────
        close = df["close"].to_numpy(dtype=float).copy()
        pe_close = pd.to_numeric(df["_pe_close"], errors="coerce").to_numpy(dtype=float)
        ce_close = pd.to_numeric(df["_ce_close"], errors="coerce").to_numpy(dtype=float)
        close[call_pcp] = np.maximum(pe_close[call_pcp] + disc[call_pcp] * (F[call_pcp] - K[call_pcp]), 0.0)
        close[put_pcp] = np.maximum(ce_close[put_pcp] - disc[put_pcp] * (F[put_pcp] - K[put_pcp]), 0.0)
        df["close"] = close
        df.loc[pcp_mask, "synthetic_via_putcallparity"] = True
        n_pcp = int(pcp_mask.sum())
        n_traded = int((~stale_self).sum())

        # ── Solve IV once, for traded ∪ PCP rows only (real-priced rows) ──
        volume = pd.to_numeric(df["volume"], errors="coerce").fillna(0).to_numpy()
        oi = (pd.to_numeric(df["oi"], errors="coerce").fillna(0).to_numpy()
              if "oi" in df.columns else np.zeros(len(df)))
        eligible = (volume >= self.min_volume) & (oi >= self.min_oi)
        solve_mask = ((~stale_self) | pcp_mask) & eligible

        iv = np.full(len(df), np.nan)
        close_now = df["close"].to_numpy(dtype=float)
        for i in np.flatnonzero(solve_mask):
            result = solve_iv(
                market_price=float(close_now[i]), F=float(F[i]), K=float(K[i]),
                T=float(T[i]), r=self.r, is_call=bool(is_call[i]),
            )
            if result is not None:
                iv[i] = result
        n_solve_failed = int(solve_mask.sum() - np.count_nonzero(~np.isnan(iv[solve_mask])))
        df["iv"] = iv

        # ── Forward-fill both-stale rows from this instrument's own
        #    history. A scratch ffill is computed for every NaN gap, but
        #    the write-back is gated to both_stale positions only — a
        #    traded/PCP row whose own solve failed keeps its honest NaN
        #    rather than silently borrowing a neighbour's IV. This is the
        #    ordering fix: because PCP rows were solved above, in this same
        #    pass, a both-stale row that ffills from a PCP neighbour reads
        #    that neighbour's *corrected* IV, never a pre-correction one. ──
        both_stale = stale_self & ~pcp_mask
        ffilled = df.groupby(group_col, sort=False)["iv"].ffill().to_numpy(dtype=float)
        iv_final = df["iv"].to_numpy(dtype=float).copy()
        apply_ffill = both_stale & ~np.isnan(ffilled)
        iv_final[apply_ffill] = ffilled[apply_ffill]
        df["iv"] = iv_final

        ffill_ok = apply_ffill
        ffill_drop = both_stale & np.isnan(ffilled)
        n_ffill = int(ffill_ok.sum())
        n_dropped = int(ffill_drop.sum())

        if ffill_ok.any():
            synth = black76_price_vec(
                F=F[ffill_ok], K=K[ffill_ok], T=T[ffill_ok],
                sigma=iv_final[ffill_ok], r=self.r, is_call=is_call[ffill_ok],
            )
            close_arr = df["close"].to_numpy(dtype=float)
            close_arr[ffill_ok] = np.maximum(synth, 0.0)
            df["close"] = close_arr
            df.loc[ffill_ok, "is_synthetic"] = True

        if n_dropped:
            df = df[~ffill_drop].reset_index(drop=True)

        # ── Greeks once, vectorized, from the final iv (NaN iv -> NaN
        #    greeks, matching the existing keep-not-drop philosophy). ─────
        F = pd.to_numeric(df["futures_price"], errors="coerce").to_numpy(dtype=float)
        K = pd.to_numeric(df["strike_price"], errors="coerce").to_numpy(dtype=float)
        dte = pd.to_numeric(df["dte"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        T = np.maximum(year_fraction(dte), 0.0)
        itype = df["instrument_type"].astype(str).str.upper().str.strip()
        is_call = itype.isin({"CE", "CALL"}).to_numpy()
        iv_final = df["iv"].to_numpy(dtype=float)

        greeks = black76_greeks_vec(F=F, K=K, T=T, sigma=iv_final, r=self.r, is_call=is_call)
        df["delta"] = greeks["delta"]
        df["gamma"] = greeks["gamma"]
        df["theta"] = greeks["theta"]
        df["vega"] = greeks["vega"]

        # ── Synthetic bid/ask via the existing slippage model, from the
        #    final close, for every row with a solved iv (matches: skipped
        #    when IV didn't solve, same as the old per-contract loop). ────
        have_iv = ~np.isnan(iv_final)
        close_final = df["close"].to_numpy(dtype=float)
        volume = pd.to_numeric(df["volume"], errors="coerce").fillna(0).to_numpy()
        oi = (pd.to_numeric(df["oi"], errors="coerce").fillna(0).to_numpy()
              if "oi" in df.columns else np.zeros(len(df)))
        dte_arr = pd.to_numeric(df["dte"], errors="coerce").to_numpy()
        bid = np.full(len(df), np.nan)
        ask = np.full(len(df), np.nan)
        for i in np.flatnonzero(have_iv):
            stub = SimpleNamespace(
                close=float(close_final[i]), strike=float(K[i]), is_call=bool(is_call[i]),
                oi=int(oi[i]), volume=int(volume[i]), dte=float(dte_arr[i]),
            )
            b, a = self.slippage.estimate_spread(stub, float(F[i]))
            bid[i], ask[i] = b, a
        df["synthetic_bid"] = bid
        df["synthetic_ask"] = ask

        df = df.drop(
            columns=["_ce_close", "_ce_stale", "_pe_close", "_pe_stale", "_grp"],
            errors="ignore",
        )

        logger.info(
            f"enrich_dataframe | {n_initial} rows: {n_traded} traded, {n_pcp} put-call-parity, "
            f"{n_ffill} forward-filled, {n_dropped} dropped (no prior IV), "
            f"{n_solve_failed} solve failures kept as NaN -> {len(df)} final rows."
        )
        return df.reset_index(drop=True)

    def enrich_multiple(
        self, chains: dict, progress_interval: int = 10
    ) -> dict:
        """Enrich multiple chains (e.g., from ChainBuilder output).

        Args:
            chains: Dict mapping timestamp -> OptionChain (or date -> dict of chains).
            progress_interval: Log progress every N chains.

        Returns:
            Same dict with all chains enriched.
        """
        total = len(chains)
        for i, (key, chain) in enumerate(chains.items()):
            if isinstance(chain, OptionChain):
                self.enrich(chain)
            elif isinstance(chain, dict):
                # Handle nested structure (e.g., from build_market_snapshots)
                for sub_chain in chain.values():
                    if isinstance(sub_chain, OptionChain):
                        self.enrich(sub_chain)

            if (i + 1) % progress_interval == 0:
                logger.info(f"  Enriched {i + 1}/{total} chains")

        logger.info(f"Enriched {total} chains total")
        return chains

    def compute_chain_iv_summary(self, chain: OptionChain) -> dict:
        """Compute summary IV statistics for a chain.

        Returns:
            Dict with atm_iv, call_skew, put_skew, avg_iv, iv_range.
        """
        calls_with_iv = [c for c in chain.calls if c.iv is not None and c.iv > 0]
        puts_with_iv = [c for c in chain.puts if c.iv is not None and c.iv > 0]

        if not calls_with_iv and not puts_with_iv:
            return {"atm_iv": None, "call_skew": None, "put_skew": None}

        # ATM IV: average of ATM call and put IV
        atm_strike = chain.atm_strike
        atm_call = next((c for c in calls_with_iv if c.strike == atm_strike), None)
        atm_put = next((c for c in puts_with_iv if c.strike == atm_strike), None)

        atm_iv = None
        if atm_call and atm_put:
            atm_iv = (atm_call.iv + atm_put.iv) / 2
        elif atm_call:
            atm_iv = atm_call.iv
        elif atm_put:
            atm_iv = atm_put.iv

        # Skew: difference between OTM and ATM IV
        # Call skew: OTM call IV - ATM IV (usually negative = normal skew)
        call_skew = None
        if atm_iv and len(calls_with_iv) >= 3:
            otm_calls = sorted(
                [c for c in calls_with_iv if c.strike > atm_strike],
                key=lambda c: c.strike,
            )
            if otm_calls:
                # Take the 2nd OTM strike
                target = otm_calls[min(1, len(otm_calls) - 1)]
                call_skew = target.iv - atm_iv

        # Put skew: OTM put IV - ATM IV (usually positive = crash risk premium)
        put_skew = None
        if atm_iv and len(puts_with_iv) >= 3:
            otm_puts = sorted(
                [c for c in puts_with_iv if c.strike < atm_strike],
                key=lambda c: c.strike,
                reverse=True,
            )
            if otm_puts:
                target = otm_puts[min(1, len(otm_puts) - 1)]
                put_skew = target.iv - atm_iv

        # Overall stats
        all_ivs = [c.iv for c in chain.contracts if c.iv is not None and c.iv > 0]

        return {
            "atm_iv": atm_iv,
            "call_skew": call_skew,
            "put_skew": put_skew,
            "avg_iv": np.mean(all_ivs) if all_ivs else None,
            "min_iv": min(all_ivs) if all_ivs else None,
            "max_iv": max(all_ivs) if all_ivs else None,
            "contracts_with_iv": len(all_ivs),
            "contracts_total": len(chain.contracts),
        }

