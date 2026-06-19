"""Information Coefficient (IC) computation and factor evaluation.

IC = cross-sectional Spearman rank correlation between a (delayed) factor and forward
returns. We report mean IC, ICIR (IC / std(IC)) and a t-statistic. The ``delay``
parameter is critical: signals are lagged so that today's factor only predicts FUTURE
returns — a guard against look-ahead bias.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class ICResult:
    name: str
    mean_ic: float
    icir: float
    t_stat: float
    p_value: float
    hit_rate: float
    n_periods: int


def forward_returns(close: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Forward return over ``horizon`` days (close-to-close), aligned to signal date."""
    return close.shift(-horizon) / close - 1.0


def compute_ic_series(
    signal: pd.DataFrame, fwd_ret: pd.DataFrame, delay: int = 1
) -> pd.Series:
    """Per-date Spearman IC between delayed signal and forward returns."""
    sig = signal.shift(delay)
    common = sig.index.intersection(fwd_ret.index)
    ics = {}
    for dt in common:
        a = sig.loc[dt]
        b = fwd_ret.loc[dt]
        df = pd.concat([a, b], axis=1).dropna()
        if len(df) < 5:
            continue
        # Spearman = Pearson on ranks (avoids a scipy dependency).
        r0 = df.iloc[:, 0].rank()
        r1 = df.iloc[:, 1].rank()
        ics[dt] = r0.corr(r1)
    return pd.Series(ics).dropna()


def evaluate(
    name: str,
    signal: pd.DataFrame,
    close: pd.DataFrame,
    horizon: int = 5,
    delay: int = 1,
) -> ICResult:
    fwd = forward_returns(close, horizon)
    ic = compute_ic_series(signal, fwd, delay=delay)
    n = len(ic)
    if n < 2:
        return ICResult(name, 0.0, 0.0, 0.0, 1.0, 0.0, n)
    mean_ic = float(ic.mean())
    std_ic = float(ic.std(ddof=1)) or np.nan
    icir = mean_ic / std_ic if std_ic and not np.isnan(std_ic) else 0.0
    t_stat = icir * np.sqrt(n)
    # two-sided p-value from a normal approximation
    from math import erfc, sqrt

    p_value = erfc(abs(t_stat) / sqrt(2))
    hit_rate = float((np.sign(ic) == np.sign(mean_ic)).mean())
    return ICResult(name, mean_ic, icir, t_stat, p_value, hit_rate, n)
