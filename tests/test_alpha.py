"""Tests for the alpha research stack (IC, neutralization, multiple testing)."""

import numpy as np
import pandas as pd

from vnquant.alpha.ic import evaluate, forward_returns
from vnquant.alpha.multiple_testing import (
    benjamini_hochberg,
    deflated_sharpe_ratio,
)
from vnquant.alpha.neutralizer import neutralize


def test_forward_returns_shifts_backwards():
    close = pd.DataFrame({"A": [10, 11, 12, 13, 14]})
    fwd = forward_returns(close, horizon=1)
    # fwd return at t should equal (close[t+1]/close[t] - 1)
    assert np.isclose(fwd["A"].iloc[0], 11 / 10 - 1)
    assert np.isnan(fwd["A"].iloc[-1])  # no future for last row


def test_neutralize_removes_market_mean():
    sig = pd.DataFrame(
        {"A": [1.0, 2.0], "B": [3.0, 4.0], "C": [5.0, 6.0]},
        index=pd.to_datetime(["2024-01-01", "2024-01-02"]),
    )
    out = neutralize(sig, neutralize_market=True, neutralize_sector=False)
    # market-neutral -> each row should sum (approx) to zero
    assert abs(out.iloc[0].sum()) < 1e-9


def test_benjamini_hochberg_basic():
    # one tiny p-value among large ones should be discovered
    p = [0.001, 0.4, 0.6, 0.8, 0.9]
    res = benjamini_hochberg(p, alpha=0.10)
    assert res.rejected[0] is True
    assert res.n_significant >= 1


def test_bh_rejects_nothing_when_all_large():
    p = [0.5, 0.6, 0.7, 0.8]
    res = benjamini_hochberg(p, alpha=0.10)
    assert res.n_significant == 0


def test_deflated_sharpe_more_trials_lower_prob():
    base = deflated_sharpe_ratio(0.1, n_obs=252, n_trials=1)
    many = deflated_sharpe_ratio(0.1, n_obs=252, n_trials=200)
    assert many < base  # more trials -> harder to be significant


def test_evaluate_returns_result():
    rng = np.random.default_rng(0)
    dates = pd.bdate_range("2024-01-01", periods=120)
    cols = list("ABCDE")
    close = pd.DataFrame(
        100 * np.cumprod(1 + rng.normal(0, 0.01, (120, 5)), axis=0),
        index=dates, columns=cols,
    )
    signal = close.pct_change(5)
    res = evaluate("mom", signal, close, horizon=5, delay=1)
    assert res.n_periods > 0
    assert -1.0 <= res.mean_ic <= 1.0
