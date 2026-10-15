
"""Shared test fixtures for the backtesting engine."""

import pytest
from datetime import date, datetime
from data.models.bar import Bar
from data.models.option_contract import OptionContract
from data.models.option_chain import OptionChain
from config.constants import OptionType


@pytest.fixture
def sample_futures_bar():
    """A single NIFTY futures bar."""
    return Bar(
        timestamp=datetime(2025, 1, 15, 15, 30),
        open=23480.0,
        high=23550.0,
        low=23420.0,
        close=23500.0,
        volume=1_200_000,
        oi=8_500_000,
        instrument_key="NSE_FO|NIFTY25JANFUT",
    )


@pytest.fixture
def sample_option_contract():
    """A single NIFTY 23500 CE option contract."""
    return OptionContract(
        instrument_key="NSE_FO|37590",
        trading_symbol="NIFTY 23500 CE 30 JAN 25",
        underlying_key="NSE_INDEX|Nifty 50",
        strike=23500.0,
        option_type=OptionType.CE,
        expiry=date(2025, 1, 30),
        lot_size=75,
        timestamp=datetime(2025, 1, 15, 15, 30),
        open=310.0,
        high=350.0,
        low=290.0,
        close=320.0,
        volume=500_000,
        oi=2_000_000,
    )


@pytest.fixture
def sample_chain():
    """A small option chain with 5 strikes (CE + PE)."""
    expiry = date(2025, 1, 30)
    ts = datetime(2025, 1, 15, 15, 30)
    contracts = []

    strikes = [23300, 23400, 23500, 23600, 23700]
    # Approximate prices for ~15% IV, F=23500, ~15 DTE
    call_prices = [280, 210, 150, 100, 65]
    put_prices = [80, 110, 150, 200, 265]

    for i, strike in enumerate(strikes):
        contracts.append(OptionContract(
            instrument_key=f"NSE_FO|CE_{strike}",
            trading_symbol=f"NIFTY {strike} CE 30 JAN 25",
            underlying_key="NSE_INDEX|Nifty 50",
            strike=float(strike),
            option_type=OptionType.CE,
            expiry=expiry,
            lot_size=75,
            timestamp=ts,
            close=float(call_prices[i]),
            volume=100_000 * (5 - abs(i - 2)),  # ATM has highest volume
            oi=500_000 * (5 - abs(i - 2)),
        ))
        contracts.append(OptionContract(
            instrument_key=f"NSE_FO|PE_{strike}",
            trading_symbol=f"NIFTY {strike} PE 30 JAN 25",
            underlying_key="NSE_INDEX|Nifty 50",
            strike=float(strike),
            option_type=OptionType.PE,
            expiry=expiry,
            lot_size=75,
            timestamp=ts,
            close=float(put_prices[i]),
            volume=100_000 * (5 - abs(i - 2)),
            oi=500_000 * (5 - abs(i - 2)),
        ))

    return OptionChain(
        underlying_symbol="NIFTY",
        expiry=expiry,
        timestamp=ts,
        futures_price=23500.0,
        contracts=contracts,
    )
