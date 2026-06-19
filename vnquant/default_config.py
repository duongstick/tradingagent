"""Default configuration for the vnquant showcase engine.

All values here are GENERIC defaults for demonstration. In a production system these
would be tuned per-market and kept private. Anything you would calibrate against live
performance is marked with `# TUNE THIS`.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class DataConfig:
    """Synthetic data generation parameters (showcase uses generated data only)."""

    n_symbols: int = 30
    n_days: int = 750
    seed: int = 42
    start_price: float = 20_000.0  # VND, illustrative
    annual_drift: float = 0.08
    annual_vol: float = 0.35
    # Vietnam market microstructure constraints
    lot_size: int = 100
    price_limit_pct: float = 0.07  # HOSE daily band +/-7%


@dataclass
class AlphaConfig:
    delay: int = 1                 # signal delay to avoid look-ahead
    horizon: int = 5               # forward return horizon (days)
    neutralize_market: bool = True
    neutralize_sector: bool = True
    fdr_alpha: float = 0.10        # Benjamini-Hochberg FDR level  # TUNE THIS
    min_ic: float = 0.02           # minimum |IC| to keep a factor  # TUNE THIS


@dataclass
class RiskConfig:
    n_factors: int = 5
    shrinkage: float = 0.0         # 0 -> auto Ledoit-Wolf
    target_vol: float = 0.15       # annualized portfolio vol target  # TUNE THIS
    max_weight: float = 0.10       # per-name cap  # TUNE THIS
    max_positions: int = 15


@dataclass
class ExecutionConfig:
    participation_rate: float = 0.10   # POV cap  # TUNE THIS
    n_slices: int = 6
    spread_bps: float = 10.0           # assumed half-spread cost
    impact_coef: float = 0.1           # square-root impact coefficient  # TUNE THIS


@dataclass
class BacktestConfig:
    initial_cash: float = 1_000_000_000.0  # 1B VND, illustrative
    commission_bps: float = 15.0           # round-trip-ish per side
    rebalance_every: int = 5               # trading days


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    alpha: AlphaConfig = field(default_factory=AlphaConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)


DEFAULT_CONFIG = Config()
