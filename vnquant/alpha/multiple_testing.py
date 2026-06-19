"""Multiple-testing correction for alpha research.

When you mine N candidate factors, some will look profitable by pure chance. Reporting
the best one's Sharpe or IC without correction is the classic backtest-overfitting trap.
This module implements two standard institutional defenses:

1. Benjamini-Hochberg False Discovery Rate (BH-FDR) control over the candidates' p-values.
2. The Deflated Sharpe Ratio (Bailey & Lopez de Prado, 2014), which discounts an
   observed Sharpe for the number of trials, plus skew/kurtosis of returns.

Both are textbook methods — the value is in actually wiring them into the research loop.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import erf, log, sqrt

import numpy as np


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))


def _norm_ppf(p: float) -> float:
    """Inverse normal CDF via Acklam's rational approximation."""
    if p <= 0.0:
        return -np.inf
    if p >= 1.0:
        return np.inf
    a = [-3.969683028665376e01, 2.209460984245205e02, -2.759285104469687e02,
         1.383577518672690e02, -3.066479806614716e01, 2.506628277459239e00]
    b = [-5.447609879822406e01, 1.615858368580409e02, -1.556989798598866e02,
         6.680131188771972e01, -1.328068155288572e01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e00,
         -2.549732539343734e00, 4.374664141464968e00, 2.938163982698783e00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e00,
         3.754408661907416e00]
    plow, phigh = 0.02425, 1 - 0.02425
    if p < plow:
        q = sqrt(-2 * log(p))
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
               ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    if p > phigh:
        q = sqrt(-2 * log(1 - p))
        return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
               ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    q = p - 0.5
    r = q * q
    return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5]) * q / \
           (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)


@dataclass
class FDRResult:
    rejected: list[bool]       # True -> discovery survives FDR
    threshold: float           # the BH p-value cutoff
    n_significant: int


def benjamini_hochberg(p_values: list[float], alpha: float = 0.10) -> FDRResult:
    """Control the FDR at level ``alpha`` over a list of p-values."""
    n = len(p_values)
    if n == 0:
        return FDRResult([], 0.0, 0)
    order = sorted(range(n), key=lambda i: p_values[i])
    thresh = 0.0
    cutoff_rank = -1
    for rank, idx in enumerate(order, start=1):
        if p_values[idx] <= (rank / n) * alpha:
            cutoff_rank = rank
            thresh = (rank / n) * alpha
    rejected = [False] * n
    if cutoff_rank > 0:
        for rank, idx in enumerate(order, start=1):
            if rank <= cutoff_rank:
                rejected[idx] = True
    return FDRResult(rejected, thresh, sum(rejected))


def deflated_sharpe_ratio(
    observed_sr: float,
    n_obs: int,
    n_trials: int,
    skew: float = 0.0,
    kurtosis: float = 3.0,
    sr_variance_across_trials: float | None = None,
) -> float:
    """Probability that the true Sharpe is > 0 given multiple trials (DSR).

    observed_sr / sr_variance are expressed per-observation (e.g. daily). Returns a
    probability in [0, 1]; values near 1 mean the result survives the multiple-testing
    deflation, near 0.5 or below means it is likely luck.
    """
    if n_obs < 2:
        return 0.0
    # Expected maximum Sharpe under the null across n_trials independent trials.
    if sr_variance_across_trials is None:
        sr_variance_across_trials = 1.0  # standardized null
    e = 0.5772156649  # Euler-Mascheroni
    z1 = _norm_ppf(1 - 1.0 / max(n_trials, 1))
    z2 = _norm_ppf(1 - 1.0 / (max(n_trials, 1) * e))
    expected_max_sr = sqrt(sr_variance_across_trials) * ((1 - e) * z1 + e * z2)

    # Standard error of the Sharpe estimator with non-normal adjustments.
    denom = sqrt(
        max(
            1e-12,
            1.0 - skew * observed_sr + (kurtosis - 1.0) / 4.0 * observed_sr ** 2,
        )
    )
    dsr_stat = (observed_sr - expected_max_sr) * sqrt(n_obs - 1) / denom
    return _norm_cdf(dsr_stat)
