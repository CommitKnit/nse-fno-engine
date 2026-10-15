"""Tests for chain enricher — IV, Greeks, and spread computation."""

import pytest
from datetime import date, datetime
from config.constants import OptionType
from data.models.option_chain import OptionChain
from data.models.option_contract import OptionContract
from options.pricing.black76 import black76_price
from options.pricing.dte_utils import year_fraction
from options.chain.chain_enricher import ChainEnricher


def _make_chain(futures_price=23500.0, dte=15):
    """Helper to create a test chain with realistic prices."""
    expiry = date(2025, 1, 30)
    ts = datetime(2025, 1, 15, 15, 30)
    # Must match chain_enricher's own T convention (trading-day DTE / 252),
    # or the price seeded here and the price re-solved by ChainEnricher.enrich()
    # disagree on time basis and recovered IV won't match the input sigma.
    T = year_fraction(dte)
    r = 0.065
    sigma = 0.15
    F = futures_price

    strikes = [23200, 23300, 23400, 23500, 23600, 23700, 23800]
    contracts = []

    for strike in strikes:
        for opt_type in [OptionType.CE, OptionType.PE]:
            is_call = opt_type == OptionType.CE
            price = black76_price(F, strike, T, sigma, r, is_call)

            contracts.append(OptionContract(
                instrument_key=f"NSE_FO|{opt_type.value}_{strike}",
                trading_symbol=f"NIFTY {strike} {opt_type.value}",
                underlying_key="NSE_INDEX|Nifty 50",
                strike=float(strike),
                option_type=opt_type,
                expiry=expiry,
                lot_size=75,
                timestamp=ts,
                close=round(price, 2),
                volume=100000,
                oi=500000,
                moneyness=strike / F,
                dte=dte,
            ))

    return OptionChain(
        underlying_symbol="NIFTY",
        expiry=expiry,
        timestamp=ts,
        futures_price=F,
        contracts=contracts,
    )


class TestChainEnricher:

    def test_enrich_solves_iv_for_all_contracts(self):
        chain = _make_chain()
        enricher = ChainEnricher()
        enricher.enrich(chain)

        solved = [c for c in chain.contracts if c.iv is not None]
        assert len(solved) == len(chain.contracts)

    def test_iv_close_to_input_vol(self):
        """If chain was priced at 15% vol, solved IV should be ~15%."""
        chain = _make_chain()
        enricher = ChainEnricher()
        enricher.enrich(chain)

        atm_call = chain.get_contract(23500, OptionType.CE)
        assert atm_call is not None
        assert atm_call.iv is not None
        assert abs(atm_call.iv - 0.15) < 0.005

    def test_delta_signs_correct(self):
        chain = _make_chain()
        enricher = ChainEnricher()
        enricher.enrich(chain)

        for c in chain.calls:
            if c.delta is not None:
                assert c.delta > 0, f"Call delta should be positive: {c.strike}"

        for c in chain.puts:
            if c.delta is not None:
                assert c.delta < 0, f"Put delta should be negative: {c.strike}"

    def test_atm_delta_near_half(self):
        chain = _make_chain()
        enricher = ChainEnricher()
        enricher.enrich(chain)

        atm_call = chain.get_contract(23500, OptionType.CE)
        assert atm_call is not None and atm_call.delta is not None
        assert 0.40 < atm_call.delta < 0.60

        atm_put = chain.get_contract(23500, OptionType.PE)
        assert atm_put is not None and atm_put.delta is not None
        assert -0.60 < atm_put.delta < -0.40

    def test_gamma_positive_everywhere(self):
        chain = _make_chain()
        enricher = ChainEnricher()
        enricher.enrich(chain)

        for c in chain.contracts:
            if c.gamma is not None:
                assert c.gamma >= 0

    def test_theta_negative_for_all(self):
        chain = _make_chain()
        enricher = ChainEnricher()
        enricher.enrich(chain)

        for c in chain.contracts:
            if c.theta is not None:
                assert c.theta <= 0

    def test_vega_positive_everywhere(self):
        chain = _make_chain()
        enricher = ChainEnricher()
        enricher.enrich(chain)

        for c in chain.contracts:
            if c.vega is not None:
                assert c.vega >= 0

    def test_synthetic_spread_populated(self):
        chain = _make_chain()
        enricher = ChainEnricher()
        enricher.enrich(chain)

        for c in chain.contracts:
            if c.iv is not None:
                assert c.synthetic_bid is not None
                assert c.synthetic_ask is not None
                assert c.synthetic_bid < c.synthetic_ask
                assert c.synthetic_bid <= c.close
                assert c.synthetic_ask >= c.close

    def test_skips_zero_price_contracts(self):
        chain = _make_chain()
        chain.contracts[0].close = 0.0
        enricher = ChainEnricher()
        enricher.enrich(chain)

        assert chain.contracts[0].iv is None

    def test_iv_summary(self):
        chain = _make_chain()
        enricher = ChainEnricher()
        enricher.enrich(chain)

        summary = enricher.compute_chain_iv_summary(chain)
        assert summary["atm_iv"] is not None
        assert abs(summary["atm_iv"] - 0.15) < 0.01
        assert summary["contracts_with_iv"] > 0
