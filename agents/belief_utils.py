"""Conjugate Gaussian updates and trade-vs-mid logic in log-price space."""
from __future__ import annotations

import math
from typing import TYPE_CHECKING, Literal, Tuple

import numpy as np

from environment.margin import MarginSpec, committed_capital, committed_capital_limit
from environment.trade_records import TradeIntent
from environment.trading_utils import (
    confidence_weighted_size,
    downsize_limit_quantity_for_capital,
    downsize_quantity_for_capital,
)
from venues.base import VenueState
from venues.clob import CLOB, EmptyBookError
from venues.constant_product import ConstantProductAMM
from venues.fba import FBAVenue
from venues.hybrid import HybridVenue

if TYPE_CHECKING:
    from environment.market_environment import MarketEnvironment


def gaussian_scalar_nif_update(
    mean: float,
    precision: float,
    obs: float,
    obs_precision: float,
    *,
    q: float = 0.0,
    dt: float = 0.0,
) -> Tuple[float, float]:
    """One conjugate scalar Normal-Normal update (information form).

    Default (``q == 0``) — STATIONARY target, precision only accumulates:

        τ_new = τ_old + τ_obs
        μ_new = (τ_old μ_old + τ_obs x) / τ_new

    Time-aware (``q > 0``, process variance ``q`` per unit time, elapsed ``dt``)
    — the prior is treated as a random walk, so its precision DECAYS for the
    time since this belief was last updated before absorbing the new signal:

        τ_decayed = 1 / (1/τ_old + q·dt)
        τ_new     = τ_decayed + τ_obs
        μ_new     = (τ_decayed μ_old + τ_obs x) / τ_new

    so steady-state gain stays bounded away from 0 (the belief tracks a moving
    target instead of freezing). With ``q == 0`` OR ``dt == 0`` the decay branch
    is SKIPPED entirely and the exact stationary arithmetic above runs
    byte-for-byte — the time-aware path is a gated-off no-op by default.
    """
    if precision <= 0:
        raise ValueError("precision must be positive")
    if obs_precision <= 0:
        raise ValueError("obs_precision must be positive")
    if q < 0.0:
        raise ValueError("process variance q must be non-negative")
    if dt < 0.0:
        raise ValueError("dt must be non-negative")
    if q > 0.0 and dt > 0.0:
        # Inflate prior variance by q·dt (random-walk prior), i.e. decay prior
        # precision. Only on the q>0 path — never perturbs the q=0 baseline.
        precision = 1.0 / (1.0 / precision + q * dt)
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

    mid_mp = market_env.mid_price(market_id)
    if mid_mp is None:
        return None, 0.0
    mid_log = log_mid_reference(mid_mp)
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


def maker_taker_decision(
    venue: CLOB | HybridVenue | FBAVenue,
    posterior_mean_log: float,
    venue_state: VenueState,
    signal_strength: float,
    taker_threshold: float,
    limit_offset_log: float,
    *,
    disagreement_threshold_log: float,
    depth_probe_qty: float = 1e-6,
) -> Literal["market", "limit_buy", "limit_sell", "skip"]:
    """CLOB/hybrid: taker when |edge|/spread_log is large; else passive limit."""
    if venue_state.mid_price is None:
        return "skip"
    if (
        venue_state.best_bid is None
        or venue_state.best_ask is None
        or venue_state.spread is None
    ):
        return "skip"

    mid = float(venue_state.mid_price)
    log_mid = math.log(mid)
    signed_edge = posterior_mean_log - log_mid
    if abs(signed_edge) < disagreement_threshold_log:
        return "skip"

    spread_log = math.log(float(venue_state.best_ask) / float(venue_state.best_bid))
    if spread_log <= 0 or not math.isfinite(spread_log):
        return "skip"

    want_taker = abs(signed_edge) / spread_log >= taker_threshold
    side = "buy" if signed_edge > 0 else "sell"

    if want_taker:
        dq = max(depth_probe_qty, 1e-12)
        try:
            vwap = venue.estimate_impact(side, dq)
        except EmptyBookError:
            want_taker = False
        else:
            if side == "buy" and (
                math.isinf(vwap) or (isinstance(vwap, float) and vwap <= 0)
            ):
                want_taker = False
            if side == "sell" and vwap == 0.0:
                want_taker = False

    if want_taker:
        _ = signal_strength  # callers may pass precomputed |edge|; we use signed_edge
        return "market"

    if signed_edge > 0:
        return "limit_buy"
    return "limit_sell"


def consider_clob_hybrid_trade(
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
    taker_threshold: float = 2.0,
    limit_offset_log: float | None = None,
) -> tuple[TradeIntent | None, float]:
    """Size and emit market or limit intent on a :class:`CLOB` or :class:`HybridVenue`."""
    from environment.market_environment import MarketEnvironment as _ME

    if not isinstance(market_env, _ME):
        raise TypeError("market_env must be MarketEnvironment")

    venue = market_env.venue(market_id)
    margin: MarginSpec = market_env.margin
    st = venue.get_state()

    if (
        st.mid_price is None
        or st.best_bid is None
        or st.best_ask is None
        or st.spread is None
    ):
        return None, 0.0

    spr_log = math.log(float(st.best_ask) / float(st.best_bid))
    lo = 0.5 * spr_log if limit_offset_log is None else limit_offset_log
    sig_strength = abs(posterior_mean_log - math.log(float(st.mid_price)))

    action = maker_taker_decision(
        venue,
        posterior_mean_log,
        st,
        sig_strength,
        taker_threshold,
        lo,
        disagreement_threshold_log=disagreement_threshold_log,
    )
    if action == "skip":
        return None, 0.0

    q0 = confidence_weighted_size(
        target_quantity,
        confidence_weighted=confidence_weighted,
        posterior_precision=posterior_precision,
        prior_precision=prior_precision_for_sizing,
        floor=confidence_floor,
        ceiling=confidence_ceiling,
    )

    if st.mid_price is None:
        return None, 0.0
    log_mid = math.log(float(st.mid_price))

    if action == "market":
        side = "buy" if posterior_mean_log > log_mid else "sell"
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
            capital_safety_margin=safety_margin,
        )
        return intent, cost_est

    if action == "limit_buy":
        side = "buy"
        limit_px = math.exp(log_mid + lo)
    else:
        side = "sell"
        limit_px = math.exp(log_mid - lo)

    q_adj = downsize_limit_quantity_for_capital(
        side,
        limit_px,
        q0,
        available,
        margin,
        safety_margin=safety_margin,
        min_quantity=min_quantity,
    )
    if q_adj < min_quantity:
        return None, 0.0
    cost_est = committed_capital_limit(
        side, q_adj, limit_px, margin, safety_margin=safety_margin
    )
    intent = TradeIntent(
        market_id=market_id,
        agent_id=agent_id,
        side=side,
        quantity=q_adj,
        order_type="limit",
        limit_price=float(limit_px),
        capital_safety_margin=safety_margin,
    )
    return intent, cost_est


def route_log_space_trade(
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
    """Dispatch to AMM (market-only) or CLOB/hybrid (maker/taker) sizing."""
    from environment.market_environment import MarketEnvironment as _ME

    if not isinstance(market_env, _ME):
        raise TypeError("market_env must be MarketEnvironment")
    v = market_env.venue(market_id)
    if isinstance(v, ConstantProductAMM):
        return consider_trade_log_space(
            agent_id=agent_id,
            market_id=market_id,
            posterior_mean_log=posterior_mean_log,
            posterior_precision=posterior_precision,
            market_env=market_env,
            disagreement_threshold_log=disagreement_threshold_log,
            target_quantity=target_quantity,
            safety_margin=safety_margin,
            available=available,
            min_quantity=min_quantity,
            confidence_weighted=confidence_weighted,
            prior_precision_for_sizing=prior_precision_for_sizing,
            confidence_floor=confidence_floor,
            confidence_ceiling=confidence_ceiling,
        )
    if isinstance(v, (CLOB, HybridVenue, FBAVenue)):
        return consider_clob_hybrid_trade(
            agent_id=agent_id,
            market_id=market_id,
            posterior_mean_log=posterior_mean_log,
            posterior_precision=posterior_precision,
            market_env=market_env,
            disagreement_threshold_log=disagreement_threshold_log,
            target_quantity=target_quantity,
            safety_margin=safety_margin,
            available=available,
            min_quantity=min_quantity,
            confidence_weighted=confidence_weighted,
            prior_precision_for_sizing=prior_precision_for_sizing,
            confidence_floor=confidence_floor,
            confidence_ceiling=confidence_ceiling,
        )
    raise TypeError(f"unsupported venue type: {type(v).__name__}")


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
    "maker_taker_decision",
    "consider_clob_hybrid_trade",
    "route_log_space_trade",
    "alphas_log_linear_offset",
]
