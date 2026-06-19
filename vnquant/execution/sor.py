"""Smart Order Router (SOR) — execution scheduling algorithms.

Splits a parent order into child slices to reduce market impact in a thin market. Three
classic schedules are provided:

  - TWAP : equal slices across the window (time-weighted).
  - VWAP : slices proportional to a volume profile (volume-weighted).
  - POV  : participate at a fixed % of expected volume (participation-of-volume).

A square-root market-impact model estimates the cost of trading a given size relative
to average daily volume. Vietnam lot-size (100) rounding is applied to every slice.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..default_config import ExecutionConfig


@dataclass
class ChildOrder:
    slice_idx: int
    quantity: int
    expected_price: float
    impact_bps: float


def square_root_impact_bps(qty: float, adv: float, coef: float, spread_bps: float) -> float:
    """Square-root impact in basis points: spread + coef * sqrt(qty/ADV)."""
    if adv <= 0:
        return spread_bps
    return spread_bps + coef * 1e4 * np.sqrt(max(qty, 0.0) / adv)


def _round_lot(q: float, lot: int = 100) -> int:
    return int((q // lot) * lot)


def twap(total_qty: int, ref_price: float, adv: float, cfg: ExecutionConfig) -> list[ChildOrder]:
    n = cfg.n_slices
    per = total_qty / n
    out = []
    for i in range(n):
        q = _round_lot(per)
        imp = square_root_impact_bps(q, adv, cfg.impact_coef, cfg.spread_bps)
        out.append(ChildOrder(i, q, ref_price * (1 + imp / 1e4), imp))
    return out


def vwap(total_qty: int, ref_price: float, adv: float, volume_profile: list[float],
         cfg: ExecutionConfig) -> list[ChildOrder]:
    prof = np.array(volume_profile, dtype=float)
    prof = prof / prof.sum() if prof.sum() > 0 else np.ones(len(prof)) / len(prof)
    out = []
    for i, frac in enumerate(prof):
        q = _round_lot(total_qty * frac)
        imp = square_root_impact_bps(q, adv, cfg.impact_coef, cfg.spread_bps)
        out.append(ChildOrder(i, q, ref_price * (1 + imp / 1e4), imp))
    return out


def pov(total_qty: int, ref_price: float, adv: float, cfg: ExecutionConfig) -> list[ChildOrder]:
    """Participate at ``participation_rate`` of per-slice expected volume."""
    slice_vol = adv / cfg.n_slices
    max_per_slice = max(_round_lot(slice_vol * cfg.participation_rate), 100)
    out = []
    remaining = total_qty
    i = 0
    while remaining > 0 and i < 1000:
        q = _round_lot(min(remaining, max_per_slice))
        if q <= 0:
            break
        imp = square_root_impact_bps(q, adv, cfg.impact_coef, cfg.spread_bps)
        out.append(ChildOrder(i, q, ref_price * (1 + imp / 1e4), imp))
        remaining -= q
        i += 1
    return out


def route(algo: str, total_qty: int, ref_price: float, adv: float,
          cfg: ExecutionConfig | None = None, volume_profile: list[float] | None = None
          ) -> list[ChildOrder]:
    """Dispatch to a scheduling algorithm by name."""
    cfg = cfg or ExecutionConfig()
    algo = algo.lower()
    if algo == "twap":
        return twap(total_qty, ref_price, adv, cfg)
    if algo == "vwap":
        prof = volume_profile or [1.0] * cfg.n_slices
        return vwap(total_qty, ref_price, adv, prof, cfg)
    if algo == "pov":
        return pov(total_qty, ref_price, adv, cfg)
    raise ValueError(f"unknown algo: {algo!r} (use twap|vwap|pov)")
