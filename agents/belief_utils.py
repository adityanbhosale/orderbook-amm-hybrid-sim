"""Conjugate Gaussian updates and trade-vs-mid logic in log-price space."""
from __future__ import annotations

import math
from typing import TYPE_CHECKING, Tuple

import numpy as np

from environment.margin import MarginSpec, committed_capital
from environment.trade_records import TradeIntent
from environment.trading_utils import confidence_weighted_size, downsize_quantity_for_capital

if TYPE_CHECKING:
    from environment.market_environment import MarketEnvironment


def gaussian_scalar_nif_update(
    mean: float,
    precision: float,
    obs: float,
    obs_precision: float,
) -> Tuple[float, float]:
    """One conjugate scalar Normal-Normal update (information form).

    τ_new = τ_old + τ_obs
    μ_new = (τ_old μ_old + τ_obs x) / τ_new
    """
    if precision <= 0:
        raise ValueError("precision must be positive")
    if obs_precision <= 0:
        raise ValueError("obs_precision must be positive")
    new_prec = precision + obs_precision
    new_mean = (precision * mean + obs_precision * obs) / new_prec
    return new_mean, new_prec


def log_mid_reference(mid_price: float) -> float:
    if mid_price <= 0:
        raise ValueError("mid_price must be positive for log-space beliefs")
    return math.log(mid_price)


def consider_trade_log_space(
    *,
    agent_id: int,
    market_id: int,
    posterior_mean_log: float,
    posterior_precision: float,
    market_env: "MarketEnvironment",
    disagreement_threshold_log: float,
    target_quantity: float,
    safety_margin: float,
    available: float,
    min_quantity: float = 1e-6,
    confidence_weighted: bool,
    prior_precision_for_sizing: float,
    confidence_floor: float = 0.25,
    confidence_ceiling: float = 4.0,
) -> tuple[TradeIntent | None, float]:
    """
    Compare posterior mean of log-fair-value to log(mid); emit market buy/sell
    sized by impact-based capital (pre-trade ``committed_capital``).
    """
    if disagreement_threshold_log < 0:
        raise ValueError("disagreement_threshold_log must be non-negative")

    mid_log = log_mid_reference(market_env.mid_price(market_id))
    edge = posterior_mean_log - mid_log
    if abs(edge) < disagreement_threshold_log:
        return None, 0.0

    side = "buy" if edge > 0 else "sell"
    venue = market_env.venue(market_id)
    margin: MarginSpec = market_env.margin

    q0 = confidence_weighted_size(
        target_quantity,
        confidence_weighted=confidence_weighted,
        posterior_precision=posterior_precision,
        prior_precision=prior_precision_for_sizing,
        floor=confidence_floor,
        ceiling=confidence_ceiling,
    )
    q_adj = downsize_quantity_for_capital(
        venue,
        side,
        q0,
        available,
        margin,
        safety_margin=safety_margin,
        min_quantity=min_quantity,
    )
    if q_adj < min_quantity:
        return None, 0.0

    cost_est = committed_capital(
        venue, side, q_adj, margin, safety_margin=safety_margin
    )
    intent = TradeIntent(
        market_id=market_id,
        agent_id=agent_id,
        side=side,
        quantity=q_adj,
        order_type="market",
    )
    return intent, cost_est


def alphas_log_linear_offset(
    initial_mid_by_market: dict[int, float],
    loadings: dict[int, np.ndarray],
    f_prior_mean: np.ndarray,
) -> dict[int, float]:
    """α_m = log(mid_m(0)) − β_m^⊤ f̄ (substrate-consistent offsets)."""
    f_bar = np.asarray(f_prior_mean, dtype=float).ravel()
    out: dict[int, float] = {}
    for m, mid in initial_mid_by_market.items():
        if mid <= 0:
            raise ValueError("initial mids must be positive")
        b = np.asarray(loadings[m], dtype=float).ravel()
        if b.shape != f_bar.shape:
            raise ValueError(
                f"loading dim for market {m} {b.shape} != f_prior {f_bar.shape}"
            )
        out[m] = float(np.log(mid) - b @ f_bar)
    return out


__all__ = [
    "gaussian_scalar_nif_update",
    "log_mid_reference",
    "consider_trade_log_space",
    "alphas_log_linear_offset",
]
