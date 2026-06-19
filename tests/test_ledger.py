"""Tests for the double-entry ledger invariants."""

import pytest

from vnquant.execution.ledger import Ledger, LedgerError


def test_buy_sell_balances():
    led = Ledger(cash=100_000_000)
    led.buy("VCB", 100, 90_000, fee=1_000)
    # cash = 100,000,000 - (100*90,000) - 1,000
    assert led.cash == 100_000_000 - 9_000_000 - 1_000
    led.check_invariant()


def test_buy_then_sell_realizes_pnl():
    led = Ledger(cash=100_000_000)
    led.buy("HPG", 1000, 25_000, fee=0)
    led.sell("HPG", 1000, 27_000, fee=0)
    assert led.realized_pnl == pytest.approx((27_000 - 25_000) * 1000)
    assert "HPG" not in led.positions
    led.check_invariant()


def test_insufficient_cash_raises():
    led = Ledger(cash=1000)
    with pytest.raises(LedgerError):
        led.buy("VNM", 100, 90_000, fee=0)


def test_oversell_raises():
    led = Ledger(cash=100_000_000)
    led.buy("FPT", 100, 100_000, fee=0)
    with pytest.raises(LedgerError):
        led.sell("FPT", 200, 110_000, fee=0)


def test_avg_price_on_add():
    led = Ledger(cash=100_000_000)
    led.buy("SSI", 100, 20_000, fee=0)
    led.buy("SSI", 100, 30_000, fee=0)
    assert led.positions["SSI"].avg_price == pytest.approx(25_000)
    assert led.positions["SSI"].quantity == 200


def test_unbalanced_transaction_detected():
    led = Ledger(cash=1_000_000)
    with pytest.raises(LedgerError):
        led._post(None, [("Cash", 100, 0), ("Positions", 0, 50)], "broken")
