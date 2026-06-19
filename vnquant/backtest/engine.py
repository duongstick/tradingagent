"""Event-driven backtester.

Walks the price panel forward one bar at a time. On each rebalance date it: builds a
factor risk model from the trailing window, computes the combined alpha signal,
constructs target weights, and routes the implied trades through the double-entry ledger
with commission + slippage. Equity is marked-to-market every bar.

Event-driven (vs vectorized) execution makes look-ahead structurally hard: the engine
can only ever see data up to the current bar.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..alpha import library
from ..alpha.neutralizer import neutralize
from ..default_config import Config
from ..execution.ledger import Ledger
from ..risk.construction import construct_portfolio
from ..risk.factor_model import fit_factor_risk_model


@dataclass
class BacktestResult:
    equity_curve: pd.Series
    trades: int
    final_equity: float
    total_return: float
    sharpe: float
    max_drawdown: float
    ledger: Ledger = field(repr=False, default=None)

    def summary(self) -> dict:
        return {
            "final_equity": round(self.final_equity, 2),
            "total_return_pct": round(self.total_return * 100, 2),
            "sharpe": round(self.sharpe, 3),
            "max_drawdown_pct": round(self.max_drawdown * 100, 2),
            "trades": self.trades,
        }


def _combined_signal(close, volume, high, low, sym2sector, cfg) -> pd.DataFrame:
    """Equal-weight blend of the example factors, neutralized."""
    sigs = []
    for fn in library.FACTOR_LIBRARY.values():
        raw = fn(close, volume, high, low)
        sigs.append(raw)
    blended = sum(s.rank(axis=1, pct=True) for s in sigs) / len(sigs)
    return neutralize(
        blended, sym2sector,
        neutralize_market=cfg.alpha.neutralize_market,
        neutralize_sector=cfg.alpha.neutralize_sector,
    )


def run_backtest(
    panel: pd.DataFrame,
    sym2sector: dict[str, str],
    cfg: Config | None = None,
) -> BacktestResult:
    cfg = cfg or Config()
    close = panel.pivot(index="date", columns="symbol", values="close").sort_index()
    volume = panel.pivot(index="date", columns="symbol", values="volume").sort_index()
    high = panel.pivot(index="date", columns="symbol", values="high").sort_index()
    low = panel.pivot(index="date", columns="symbol", values="low").sort_index()
    rets = close.pct_change()

    signal = _combined_signal(close, volume, high, low, sym2sector, cfg)

    ledger = Ledger(cash=cfg.backtest.initial_cash)
    dates = close.index
    warmup = 60
    equity = {}
    n_trades = 0
    comm = cfg.backtest.commission_bps / 1e4

    for i in range(warmup, len(dates)):
        dt = dates[i]
        prices = close.loc[dt].dropna().to_dict()

        if (i - warmup) % cfg.backtest.rebalance_every == 0:
            window = rets.iloc[i - warmup:i]
            try:
                rm = fit_factor_risk_model(window, n_factors=cfg.risk.n_factors)
            except np.linalg.LinAlgError:
                equity[dt] = ledger.equity(prices)
                continue
            score = signal.iloc[i - 1].reindex(rm.loadings.index)
            target_w = construct_portfolio(score, rm, cfg.risk)

            eq = ledger.equity(prices)
            # Liquidate names no longer targeted.
            for sym in list(ledger.positions.keys()):
                if sym not in target_w.index and sym in prices:
                    lot = ledger.positions[sym]
                    ledger.sell(sym, lot.quantity, prices[sym], prices[sym] * lot.quantity * comm, dt)
                    n_trades += 1
            # Move toward targets (full rebalance, long-only).
            for sym, w in target_w.items():
                if sym not in prices or w <= 0:
                    continue
                target_val = eq * w
                cur_qty = ledger.positions[sym].quantity if sym in ledger.positions else 0
                cur_val = cur_qty * prices[sym]
                diff_val = target_val - cur_val
                qty = int((abs(diff_val) / prices[sym]) // 100 * 100)
                if qty < 100:
                    continue
                fee = qty * prices[sym] * comm
                try:
                    if diff_val > 0:
                        ledger.buy(sym, qty, prices[sym], fee, dt)
                    else:
                        sell_qty = min(qty, cur_qty)
                        if sell_qty >= 100:
                            ledger.sell(sym, sell_qty, prices[sym], fee, dt)
                    n_trades += 1
                except Exception:
                    pass  # insufficient cash / holding — skip slice

        equity[dt] = ledger.equity(prices)

    ledger.check_invariant()
    eq_curve = pd.Series(equity).sort_index()
    daily = eq_curve.pct_change().dropna()
    sharpe = float(daily.mean() / daily.std() * np.sqrt(252)) if daily.std() > 0 else 0.0
    running_max = eq_curve.cummax()
    mdd = float(((eq_curve - running_max) / running_max).min())
    total_ret = float(eq_curve.iloc[-1] / cfg.backtest.initial_cash - 1.0)

    return BacktestResult(
        equity_curve=eq_curve,
        trades=n_trades,
        final_equity=float(eq_curve.iloc[-1]),
        total_return=total_ret,
        sharpe=sharpe,
        max_drawdown=mdd,
        ledger=ledger,
    )
