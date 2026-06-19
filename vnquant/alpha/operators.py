"""Cross-sectional alpha operators.

A small library of vectorized operators over wide (date x symbol) frames, modelled on
the WorldQuant-style alpha expression vocabulary. These are the building blocks of the
alpha DSL — generic and well-known; the *combinations* that work are the edge and are
not shipped here.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# --- time-series operators (operate down the date axis, per symbol) ---


def ts_delay(df: pd.DataFrame, n: int) -> pd.DataFrame:
    return df.shift(n)


def ts_delta(df: pd.DataFrame, n: int) -> pd.DataFrame:
    return df - df.shift(n)


def ts_returns(df: pd.DataFrame, n: int = 1) -> pd.DataFrame:
    return df.pct_change(n)


def ts_mean(df: pd.DataFrame, n: int) -> pd.DataFrame:
    return df.rolling(n).mean()


def ts_std(df: pd.DataFrame, n: int) -> pd.DataFrame:
    return df.rolling(n).std()


def ts_sum(df: pd.DataFrame, n: int) -> pd.DataFrame:
    return df.rolling(n).sum()


def ts_rank(df: pd.DataFrame, n: int) -> pd.DataFrame:
    return df.rolling(n).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False)


def ts_argmax(df: pd.DataFrame, n: int) -> pd.DataFrame:
    return df.rolling(n).apply(lambda x: float(np.argmax(x)), raw=True)


def ts_argmin(df: pd.DataFrame, n: int) -> pd.DataFrame:
    return df.rolling(n).apply(lambda x: float(np.argmin(x)), raw=True)


def ts_corr(a: pd.DataFrame, b: pd.DataFrame, n: int) -> pd.DataFrame:
    return a.rolling(n).corr(b)


# --- cross-sectional operators (operate across symbols, per date) ---


def cs_rank(df: pd.DataFrame) -> pd.DataFrame:
    return df.rank(axis=1, pct=True)


def cs_zscore(df: pd.DataFrame) -> pd.DataFrame:
    mu = df.mean(axis=1)
    sd = df.std(axis=1).replace(0, np.nan)
    return df.sub(mu, axis=0).div(sd, axis=0)


def cs_demean(df: pd.DataFrame) -> pd.DataFrame:
    return df.sub(df.mean(axis=1), axis=0)


# --- element-wise helpers ---


def winsorize(df: pd.DataFrame, pct: float = 0.02) -> pd.DataFrame:
    lo = df.quantile(pct, axis=1)
    hi = df.quantile(1 - pct, axis=1)
    return df.clip(lower=lo, upper=hi, axis=0)


def sign(df: pd.DataFrame) -> pd.DataFrame:
    return np.sign(df)


def log(df: pd.DataFrame) -> pd.DataFrame:
    return np.log(df.where(df > 0))
