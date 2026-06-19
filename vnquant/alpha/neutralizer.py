"""Neutralization: strip unwanted exposures from a raw alpha signal.

Removes market (constant) and optional sector exposure via cross-sectional OLS at each
date, leaving the residual as the "neutralized" alpha. This is standard practice so a
factor isn't just re-expressing beta or a sector tilt.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def neutralize(
    signal: pd.DataFrame,
    sym2sector: dict[str, str] | None = None,
    neutralize_market: bool = True,
    neutralize_sector: bool = True,
) -> pd.DataFrame:
    """Cross-sectionally neutralize ``signal`` (date x symbol) against market/sector.

    For each date we regress the signal on a design matrix of [intercept, sector
    dummies] and keep the residual. With no regressors requested, returns the signal
    cross-sectionally de-meaned (or unchanged).
    """
    if not neutralize_market and not neutralize_sector:
        return signal

    symbols = list(signal.columns)
    sectors = None
    if neutralize_sector and sym2sector:
        sec_labels = [sym2sector.get(s, "NA") for s in symbols]
        uniq = sorted(set(sec_labels))
        # sector dummy matrix (n_sym x n_sec)
        sectors = np.zeros((len(symbols), len(uniq)))
        for i, lab in enumerate(sec_labels):
            sectors[i, uniq.index(lab)] = 1.0

    out = signal.copy()
    for dt, row in signal.iterrows():
        y = row.values.astype(float)
        mask = ~np.isnan(y)
        if mask.sum() < 3:
            continue
        cols = []
        if neutralize_market:
            cols.append(np.ones(mask.sum()))
        if sectors is not None:
            cols.append(sectors[mask])
        if not cols:
            out.loc[dt, signal.columns[mask]] = y[mask] - y[mask].mean()
            continue
        X = np.column_stack(cols)
        yv = y[mask]
        # least squares residual
        beta, *_ = np.linalg.lstsq(X, yv, rcond=None)
        resid = yv - X @ beta
        out.loc[dt, signal.columns[mask]] = resid
    return out
