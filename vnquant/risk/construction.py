"""Portfolio construction from an alpha signal + factor risk model.

Implements a simple, robust long-only construction:
  - rank assets by the combined alpha score,
  - select the top-N,
  - size by inverse-volatility (risk-based weighting) using the factor model's diagonal,
  - cap per-name weight and scale the book to a target volatility.

This is intentionally classic and explainable rather than a black-box optimizer.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..default_config import RiskConfig
from .factor_model import FactorRiskModel


def construct_portfolio(
    alpha_score: pd.Series,
    risk_model: FactorRiskModel,
    cfg: RiskConfig | None = None,
) -> pd.Series:
    """Return target weights (sum <= 1) for the latest cross-section.

    alpha_score : Series indexed by symbol (higher = more attractive).
    """
    cfg = cfg or RiskConfig()
    score = alpha_score.dropna()
    if score.empty:
        return pd.Series(dtype=float)

    top = score.sort_values(ascending=False).head(cfg.max_positions)
    syms = list(top.index)

    sigma = risk_model.covariance()
    vols = pd.Series(np.sqrt(np.diag(sigma.loc[syms, syms].values)), index=syms)
    inv_vol = (1.0 / vols.replace(0, np.nan)).fillna(0.0)
    if inv_vol.sum() == 0:
        weights = pd.Series(1.0 / len(syms), index=syms)
    else:
        weights = inv_vol / inv_vol.sum()

    # Per-name cap, then renormalize.
    weights = weights.clip(upper=cfg.max_weight)
    if weights.sum() > 0:
        weights = weights / weights.sum()

    # Scale to target volatility using the portfolio variance from the risk model.
    w = weights.values
    cov = sigma.loc[syms, syms].values
    port_var = float(w @ cov @ w)
    port_vol_annual = np.sqrt(max(port_var, 1e-12)) * np.sqrt(252)
    if port_vol_annual > 0:
        scale = min(1.0, cfg.target_vol / port_vol_annual)
        weights = weights * scale

    return weights
