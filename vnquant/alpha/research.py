"""Alpha research orchestrator.

Ties the pieces together into one disciplined research loop:

    raw factor -> neutralize -> IC evaluation -> multiple-testing correction (BH-FDR
    + Deflated Sharpe) -> surviving factors.

This is the part that separates "I backtested a signal" from "I controlled for
data-snooping across N signals".
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from ..default_config import AlphaConfig
from . import library
from .ic import ICResult, evaluate
from .multiple_testing import (
    benjamini_hochberg,
    deflated_sharpe_ratio,
)
from .neutralizer import neutralize


@dataclass
class FactorReport:
    ic: ICResult
    survives_fdr: bool
    deflated_sr_prob: float


def research(
    close: pd.DataFrame,
    volume: pd.DataFrame,
    high: pd.DataFrame,
    low: pd.DataFrame,
    sym2sector: dict[str, str],
    cfg: AlphaConfig | None = None,
    factors: dict | None = None,
) -> dict[str, FactorReport]:
    """Run the full research loop over a set of factors."""
    cfg = cfg or AlphaConfig()
    factors = factors or library.FACTOR_LIBRARY

    results: dict[str, ICResult] = {}
    for name, fn in factors.items():
        raw = fn(close, volume, high, low)
        neut = neutralize(
            raw,
            sym2sector,
            neutralize_market=cfg.neutralize_market,
            neutralize_sector=cfg.neutralize_sector,
        )
        results[name] = evaluate(name, neut, close, horizon=cfg.horizon, delay=cfg.delay)

    names = list(results)
    p_values = [results[n].p_value for n in names]
    fdr = benjamini_hochberg(p_values, alpha=cfg.fdr_alpha)

    reports: dict[str, FactorReport] = {}
    for i, n in enumerate(names):
        r = results[n]
        # Convert per-period IC into an approximate per-period Sharpe of an IC-weighted
        # long-short book (illustrative): SR ~ ICIR.
        dsr = deflated_sharpe_ratio(
            observed_sr=r.icir,
            n_obs=r.n_periods,
            n_trials=len(names),
        )
        reports[n] = FactorReport(
            ic=r,
            survives_fdr=fdr.rejected[i],
            deflated_sr_prob=dsr,
        )
    return reports
