"""Data-leakage and bias audit framework.

A research result is only trustworthy if the data feeding it is free of forward-looking
information and common biases. These checkers encode the hard-won lessons of a real
system: the most dangerous bugs don't crash — they silently inflate your backtest.

Checkers return an AuditResult with a severity. CRITICAL findings should block a model
from being promoted.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np
import pandas as pd


class Severity(str, Enum):
    OK = "OK"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"


@dataclass
class AuditResult:
    check: str
    severity: Severity
    message: str


def check_lookahead_target(features: pd.DataFrame, target_col: str = "target") -> AuditResult:
    """A forward-looking target must have NaNs at the TAIL (no future price yet).

    War story: a blanket ``df.dropna()`` once deleted exactly those tail rows, silently
    truncating the most recent sessions so inference read stale prices. The fix was to
    drop NaNs on FEATURE columns only and keep target NaNs at the tail.
    """
    if target_col not in features.columns:
        return AuditResult("lookahead_target", Severity.WARNING,
                           f"no '{target_col}' column to check")
    tail = features[target_col].tail(5)
    if not tail.isna().any():
        return AuditResult(
            "lookahead_target", Severity.CRITICAL,
            "target has no NaN at the tail — forward returns may be leaking from the future",
        )
    return AuditResult("lookahead_target", Severity.OK,
                       "target NaNs present at tail as expected")


def check_feature_completeness(features: pd.DataFrame, feature_cols: list[str]) -> AuditResult:
    """Rows with partial features must not be silently mislabeled.

    War story: legacy rows missing a key feature got a bogus default classification,
    producing a hallucinated 34% win-rate. Always stratify metrics by completeness.
    """
    if not feature_cols:
        return AuditResult("feature_completeness", Severity.WARNING, "no feature columns given")
    present = [c for c in feature_cols if c in features.columns]
    sub = features[present]
    incomplete = sub.isna().any(axis=1).mean()
    if incomplete > 0.5:
        return AuditResult(
            "feature_completeness", Severity.WARNING,
            f"{incomplete:.0%} of rows have incomplete features — stratify metrics before trusting them",
        )
    return AuditResult("feature_completeness", Severity.OK,
                       f"{incomplete:.0%} rows incomplete")


def check_temporal_order(index: pd.Index) -> AuditResult:
    """Time index must be strictly increasing — out-of-order data breaks causality."""
    idx = pd.Index(index)
    if not idx.is_monotonic_increasing:
        return AuditResult("temporal_order", Severity.CRITICAL,
                           "time index is not monotonic increasing")
    if idx.has_duplicates:
        return AuditResult("temporal_order", Severity.CRITICAL, "duplicate timestamps present")
    return AuditResult("temporal_order", Severity.OK, "time index strictly increasing")


def check_survivorship(panel_symbols: set[str], universe_symbols: set[str]) -> AuditResult:
    """Backtest universe should include delisted names, not just survivors."""
    missing = universe_symbols - panel_symbols
    if not missing and panel_symbols == universe_symbols:
        return AuditResult(
            "survivorship", Severity.WARNING,
            "panel exactly equals current universe — verify delisted names are included",
        )
    return AuditResult("survivorship", Severity.OK, "panel differs from current universe")


def check_constant_signal(signal: pd.Series, std_threshold: float = 0.02) -> AuditResult:
    """Detect model/signal collapse (near-constant output).

    War story: a collapsed neural model emitted near-identical scores (std < 0.02). The
    probe below flags it before its garbage poisons the training set.
    """
    s = signal.dropna()
    if len(s) < 5:
        return AuditResult("signal_collapse", Severity.WARNING, "too few points to assess")
    if float(np.std(s)) < std_threshold:
        return AuditResult("signal_collapse", Severity.CRITICAL,
                           f"signal std {np.std(s):.4f} < {std_threshold} — model likely collapsed")
    return AuditResult("signal_collapse", Severity.OK, f"signal std {np.std(s):.4f}")


def run_all(results: list[AuditResult]) -> Severity:
    """Aggregate severity across checks (worst wins)."""
    if any(r.severity == Severity.CRITICAL for r in results):
        return Severity.CRITICAL
    if any(r.severity == Severity.WARNING for r in results):
        return Severity.WARNING
    return Severity.OK
