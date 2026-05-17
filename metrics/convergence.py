"""Convergence of market mids toward latent fair values."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class MidSnapshot:
    tick: int
    mids: np.ndarray


@dataclass
class ConvergenceResult:
    normalized_rmse_log: float
    normalized_max_rel_price_error: float
    convergence_tick: Optional[int]


def max_relative_price_error(mids: np.ndarray, fair_prices: np.ndarray) -> float:
    """Max over markets of |mid - fair| / fair."""
    fair_prices = np.maximum(np.asarray(fair_prices, dtype=float), 1e-12)
    mids = np.asarray(mids, dtype=float)
    return float(np.max(np.abs(mids - fair_prices) / fair_prices))


def rmse_log_mid_vs_truth(mids: np.ndarray, log_truth: np.ndarray) -> float:
    return float(
        np.sqrt(np.mean((np.log(np.maximum(mids, 1e-12)) - log_truth) ** 2))
    )


def build_mid_trajectory_from_trades(
    trade_timestamps: list[int],
    trade_market_ids: list[int],
    trade_mid_after: list[float],
    n_markets: int,
    initial_mids: np.ndarray,
) -> list[MidSnapshot]:
    """Piecewise-constant mids; update only the traded market on each event."""
    events = sorted(
        zip(trade_timestamps, trade_market_ids, trade_mid_after),
        key=lambda x: (x[0], x[1]),
    )
    mids = np.asarray(initial_mids, dtype=float).copy()
    out: list[MidSnapshot] = [MidSnapshot(tick=0, mids=mids.copy())]
    for t, m, mid_a in events:
        mids[m] = mid_a
        out.append(MidSnapshot(tick=int(t), mids=mids.copy()))
    return out


def time_to_within_relative_band(
    trajectory: list[MidSnapshot],
    fair_prices: np.ndarray,
    rel_band: float,
) -> Optional[int]:
    """
    First tick (snapshot index after that event) where
    max_m |mid_m - fair_m| / fair_m <= rel_band.
    """
    fair_prices = np.asarray(fair_prices, dtype=float)
    for snap in trajectory:
        if max_relative_price_error(snap.mids, fair_prices) <= rel_band:
            return snap.tick
    return None


def convergence_metrics(
    *,
    final_mids: np.ndarray,
    fair_prices: np.ndarray,
    log_truths: np.ndarray,
    trajectory: list[MidSnapshot],
    rel_band: float = 0.02,
) -> ConvergenceResult:
    normalized_rmse_log = rmse_log_mid_vs_truth(final_mids, log_truths)
    normalized_max_rel_price_error = max_relative_price_error(
        final_mids, fair_prices
    )
    conv_tick = time_to_within_relative_band(trajectory, fair_prices, rel_band)
    return ConvergenceResult(
        normalized_rmse_log=normalized_rmse_log,
        normalized_max_rel_price_error=normalized_max_rel_price_error,
        convergence_tick=conv_tick,
    )


__all__ = [
    "ConvergenceResult",
    "MidSnapshot",
    "build_mid_trajectory_from_trades",
    "convergence_metrics",
    "max_relative_price_error",
    "rmse_log_mid_vs_truth",
    "time_to_within_relative_band",
]
