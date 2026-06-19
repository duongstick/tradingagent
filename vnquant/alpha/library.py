"""Example alpha factors (TEXTBOOK ONLY).

These are classic, publicly-documented signals — momentum, short-term reversal,
low-volatility, volume-trend. They exist to demonstrate the research pipeline. The
proprietary mined factors that constitute the real edge are NOT included.

Each factor is a callable: (close, volume, high, low) wide frames -> raw signal frame.
"""

from __future__ import annotations

import pandas as pd

from . import operators as op


def momentum_12_1(close: pd.DataFrame, volume=None, high=None, low=None) -> pd.DataFrame:
    """12-day momentum skipping the most recent day (classic continuation)."""
    return op.cs_rank(close.shift(1) / close.shift(12) - 1.0)


def short_term_reversal(close: pd.DataFrame, volume=None, high=None, low=None) -> pd.DataFrame:
    """5-day reversal: recent losers tend to bounce (mean reversion)."""
    return op.cs_rank(-(close / close.shift(5) - 1.0))


def low_volatility(close: pd.DataFrame, volume=None, high=None, low=None) -> pd.DataFrame:
    """Low-volatility anomaly: lower 20-day vol -> higher expected risk-adj return."""
    vol = op.ts_std(op.ts_returns(close, 1), 20)
    return op.cs_rank(-vol)


def volume_trend(close: pd.DataFrame, volume=None, high=None, low=None) -> pd.DataFrame:
    """Rising volume on rising price (very rough accumulation proxy)."""
    ret = op.ts_returns(close, 1)
    vchg = op.ts_returns(volume, 5)
    return op.cs_rank(ret.rolling(5).mean() * vchg)


def high_low_range(close: pd.DataFrame, volume=None, high=None, low=None) -> pd.DataFrame:
    """Normalized 10-day range — a crude volatility/illiquidity proxy."""
    rng = (high.rolling(10).max() - low.rolling(10).min()) / close
    return op.cs_rank(-rng)


# Registry used by the CLI / research orchestrator.
FACTOR_LIBRARY = {
    "momentum_12_1": momentum_12_1,
    "short_term_reversal": short_term_reversal,
    "low_volatility": low_volatility,
    "volume_trend": volume_trend,
    "high_low_range": high_low_range,
}
