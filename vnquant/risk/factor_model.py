"""Structural factor risk model.

Decomposes the asset covariance matrix as

    Sigma = B * F * B^T + D

where B is the factor-loading matrix (assets x factors), F is the factor covariance,
and D is the diagonal of idiosyncratic (specific) variances. Estimating Sigma this way
is far more stable than a raw sample covariance when the number of assets is large
relative to the history — a classic small-sample problem in equity portfolios.

Factor returns here are estimated via cross-sectional regression of asset returns on
loadings (a fundamental-factor-model style fit). Ledoit-Wolf shrinkage is applied to
the factor covariance for extra robustness.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


def ledoit_wolf_shrinkage(returns: np.ndarray) -> tuple[np.ndarray, float]:
    """Ledoit-Wolf shrinkage of a sample covariance toward a scaled identity.

    Parameters
    ----------
    returns : array (T x N) of demeaned returns.

    Returns
    -------
    (sigma_hat, shrinkage_intensity)
    """
    t, n = returns.shape
    x = returns - returns.mean(axis=0, keepdims=True)
    sample = (x.T @ x) / t
    mu = np.trace(sample) / n
    target = mu * np.eye(n)

    # Pi: sum of asymptotic variances of sample cov entries.
    x2 = x ** 2
    phi_mat = (x2.T @ x2) / t - sample ** 2
    phi = phi_mat.sum()
    gamma = np.linalg.norm(sample - target, "fro") ** 2
    kappa = phi / gamma if gamma > 0 else 0.0
    shrinkage = max(0.0, min(1.0, kappa / t))
    sigma_hat = shrinkage * target + (1 - shrinkage) * sample
    return sigma_hat, float(shrinkage)


@dataclass
class FactorRiskModel:
    loadings: pd.DataFrame        # B: assets x factors
    factor_cov: np.ndarray        # F: factors x factors
    specific_var: pd.Series       # diag(D): per-asset
    shrinkage: float

    def covariance(self) -> pd.DataFrame:
        """Reconstruct the asset covariance Sigma = B F B^T + D."""
        b = self.loadings.values
        sigma = b @ self.factor_cov @ b.T + np.diag(self.specific_var.values)
        return pd.DataFrame(sigma, index=self.loadings.index, columns=self.loadings.index)


def build_statistical_factors(returns: pd.DataFrame, n_factors: int) -> pd.DataFrame:
    """Derive factor loadings via PCA of the return covariance (statistical factors)."""
    x = returns.dropna(how="all").fillna(0.0)
    cov = np.cov(x.values, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1][:n_factors]
    loadings = eigvecs[:, order]
    return pd.DataFrame(
        loadings, index=returns.columns, columns=[f"F{i+1}" for i in range(n_factors)]
    )


def fit_factor_risk_model(
    returns: pd.DataFrame, n_factors: int = 5
) -> FactorRiskModel:
    """Fit Sigma = B F B^T + D from a (T x N) return frame."""
    rets = returns.dropna(how="all").fillna(0.0)
    loadings = build_statistical_factors(rets, n_factors)
    b = loadings.values

    # Factor returns by cross-sectional regression each period: f_t = (B'B)^-1 B' r_t
    bt_b_inv = np.linalg.pinv(b.T @ b)
    factor_returns = (rets.values @ b) @ bt_b_inv.T  # (T x k)

    factor_cov, shrink = ledoit_wolf_shrinkage(factor_returns)

    # Specific returns = residuals of r_t - B f_t
    fitted = factor_returns @ b.T
    resid = rets.values - fitted
    specific_var = pd.Series(resid.var(axis=0), index=rets.columns)

    return FactorRiskModel(
        loadings=loadings,
        factor_cov=factor_cov,
        specific_var=specific_var,
        shrinkage=shrink,
    )
