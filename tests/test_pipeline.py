"""Tests for the SOR, audit checkers, factor risk model, and full pipeline."""

import numpy as np
import pandas as pd

from vnquant.audit import (
    Severity,
    check_constant_signal,
    check_lookahead_target,
    check_temporal_order,
)
from vnquant.default_config import Config
from vnquant.execution.sor import route
from vnquant.pipeline import run_pipeline
from vnquant.risk.factor_model import fit_factor_risk_model, ledoit_wolf_shrinkage


def test_sor_twap_conserves_size_roughly():
    cfg = Config().execution
    orders = route("twap", 10_000, 50_000, adv=2_000_000, cfg=cfg)
    assert len(orders) == cfg.n_slices
    assert all(o.quantity % 100 == 0 for o in orders)


def test_sor_pov_respects_participation():
    cfg = Config().execution
    orders = route("pov", 50_000, 50_000, adv=1_000_000, cfg=cfg)
    assert sum(o.quantity for o in orders) <= 50_000 + 100


def test_sor_unknown_algo_raises():
    import pytest
    with pytest.raises(ValueError):
        route("midnight", 100, 1.0, 1.0)


def test_audit_lookahead_flags_no_tail_nan():
    df = pd.DataFrame({"target": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]})
    res = check_lookahead_target(df, "target")
    assert res.severity == Severity.CRITICAL


def test_audit_lookahead_ok_with_tail_nan():
    df = pd.DataFrame({"target": [0.1, 0.2, 0.3, np.nan, np.nan, np.nan]})
    res = check_lookahead_target(df, "target")
    assert res.severity == Severity.OK


def test_audit_temporal_order_detects_unsorted():
    idx = pd.to_datetime(["2024-01-03", "2024-01-01", "2024-01-02"])
    res = check_temporal_order(idx)
    assert res.severity == Severity.CRITICAL


def test_audit_signal_collapse():
    s = pd.Series([0.5] * 50)
    res = check_constant_signal(s)
    assert res.severity == Severity.CRITICAL


def test_ledoit_wolf_in_unit_interval():
    rng = np.random.default_rng(1)
    rets = rng.normal(0, 0.01, (200, 8))
    _, shrink = ledoit_wolf_shrinkage(rets)
    assert 0.0 <= shrink <= 1.0


def test_factor_model_reconstruction_psd():
    rng = np.random.default_rng(2)
    dates = pd.bdate_range("2023-01-01", periods=200)
    rets = pd.DataFrame(rng.normal(0, 0.01, (200, 10)), index=dates,
                        columns=[f"S{i}" for i in range(10)])
    rm = fit_factor_risk_model(rets, n_factors=3)
    cov = rm.covariance().values
    eig = np.linalg.eigvalsh(cov)
    assert eig.min() > -1e-8  # positive semi-definite


def test_full_pipeline_runs_and_balances():
    cfg = Config()
    cfg.data.n_symbols = 20
    cfg.data.n_days = 300
    out = run_pipeline(cfg)
    assert out.audit_severity in (Severity.OK, Severity.WARNING)
    assert out.backtest.final_equity > 0
    assert len(out.factor_reports) > 0
    out.backtest.ledger.check_invariant()
