"""End-to-end research & backtest pipeline orchestrator.

Stitches the stages into one reproducible run:

    synthetic data -> audit -> alpha research (IC + FDR + DSR) -> factor risk model
    -> portfolio construction -> event-driven backtest.

Mirrors the multi-step pipeline of the production system (regime -> signal -> risk ->
execution) but on synthetic data with textbook factors.
"""

from __future__ import annotations

from dataclasses import dataclass

from .alpha.research import FactorReport, research
from .audit import (
    Severity,
    check_lookahead_target,
    check_temporal_order,
    run_all,
)
from .backtest.engine import BacktestResult, run_backtest
from .data.synthetic import generate_panel, to_wide
from .default_config import Config
from .risk.factor_model import fit_factor_risk_model


@dataclass
class PipelineOutput:
    audit_severity: Severity
    factor_reports: dict[str, FactorReport]
    risk_shrinkage: float
    backtest: BacktestResult


def run_pipeline(cfg: Config | None = None) -> PipelineOutput:
    cfg = cfg or Config()

    # 1. Data
    panel, sym2sector = generate_panel(cfg.data)
    close = to_wide(panel, "close")
    volume = to_wide(panel, "volume")
    high = to_wide(panel, "high")
    low = to_wide(panel, "low")

    # 2. Audit (fail-fast on structural data problems)
    audit_results = [
        check_temporal_order(close.index),
    ]
    # build a target column to demonstrate the lookahead checker
    demo = close[[close.columns[0]]].copy()
    demo.columns = ["price"]
    demo["target"] = demo["price"].shift(-cfg.alpha.horizon) / demo["price"] - 1.0
    audit_results.append(check_lookahead_target(demo, "target"))
    severity = run_all(audit_results)

    # 3. Alpha research with multiple-testing correction
    reports = research(close, volume, high, low, sym2sector, cfg.alpha)

    # 4. Risk model (report shrinkage as a health signal)
    rets = close.pct_change()
    rm = fit_factor_risk_model(rets.iloc[-cfg.data.n_days // 2:], n_factors=cfg.risk.n_factors)

    # 5. Backtest
    bt = run_backtest(panel, sym2sector, cfg)

    return PipelineOutput(
        audit_severity=severity,
        factor_reports=reports,
        risk_shrinkage=rm.shrinkage,
        backtest=bt,
    )
