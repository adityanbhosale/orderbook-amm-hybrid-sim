"""Discounted cross-market Gaussian updates on primary markets only."""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

from environment.market_environment import MarketEnvironment
from environment.signals import Signal
from environment.simulator import Simulator
from environment.trade_records import TradeIntent

from agents.belief_utils import gaussian_scalar_nif_update, route_log_space_trade


@dataclass
class AggregatedEvidenceAgent:
    """
    Observes signals on ``observed_markets``; maintains posteriors only on
    ``market_ids`` (primaries). Cross-market updates follow the discounted
    precision construction from AggregationDepthAgent in reference
    ``sim/agentic.py``.
    """

    agent_id: int
    budget: float
    market_ids: tuple[int, ...]
    observed_markets: tuple[int, ...]
    cross_weights: dict[tuple[int, int], float]
    initial_log_fair_mean: dict[int, float]
    observation_delay: int = 0
    review_interval: int = 500
    prior_precision: float = 2.0
    signal_precision_assumed: float = 1.0
    disagreement_threshold_log: float = 0.005
    trade_size: float = 1.0
    confidence_weighted: bool = False
    confidence_floor: float = 0.25
    confidence_ceiling: float = 4.0
    safety_margin: float = 1.2
    min_cross_weight: float = 0.05
    min_trade_quantity: float = 1e-6

    deployed: float = field(default=0.0, init=False)
    pending_cost: float = field(default=0.0, init=False)
    _mean_log: dict[int, float] = field(default_factory=dict, init=False)
    _prec: dict[int, float] = field(default_factory=dict, init=False)
    arrival_rate_per_unit: float = field(default=0.0, init=False)

    def __post_init__(self) -> None:
        if not self.market_ids:
            raise ValueError("market_ids must be non-empty")
        if not self.observed_markets:
            raise ValueError("observed_markets must be non-empty")
        if self.observation_delay < 0:
            raise ValueError("observation_delay must be non-negative")
        if self.review_interval < 0:
            raise ValueError("review_interval must be non-negative")
        if self.prior_precision <= 0:
            raise ValueError("prior_precision must be positive")
        if self.signal_precision_assumed <= 0:
            raise ValueError("signal_precision_assumed must be positive")
        if self.disagreement_threshold_log < 0:
            raise ValueError("disagreement_threshold_log must be non-negative")
        if self.trade_size <= 0:
            raise ValueError("trade_size must be positive")
        if self.safety_margin < 1.0:
            raise ValueError("safety_margin must be >= 1.0")
        if not (0 <= self.min_cross_weight <= 1):
            raise ValueError("min_cross_weight must be in [0, 1]")
        self.market_ids = tuple(self.market_ids)
        self.observed_markets = tuple(self.observed_markets)
        obs_set = set(self.observed_markets)
        for m in self.market_ids:
            if m not in obs_set:
                raise ValueError(f"primary market {m} must be in observed_markets")
            if m not in self.initial_log_fair_mean:
                raise ValueError(f"initial_log_fair_mean missing market {m}")
            self._mean_log[m] = float(self.initial_log_fair_mean[m])
            self._prec[m] = float(self.prior_precision)

    @property
    def available(self) -> float:
        return self.budget - self.deployed - self.pending_cost

    def observes(self, market_id: int) -> bool:
        return market_id in self.observed_markets

    def posterior(self, market_id: int) -> tuple[float, float]:
        return self._mean_log[market_id], self._prec[market_id]

    def update_posterior(self, signal: Signal) -> None:
        sig_m = signal.market_id
        for primary_m in self.market_ids:
            if primary_m == sig_m:
                weight = 1.0
            else:
                weight = self.cross_weights.get((primary_m, sig_m), 0.0)
            if abs(weight) < self.min_cross_weight:
                continue
            tau_eff = self.signal_precision_assumed * weight * weight
            val_eff = weight * signal.value
            mu, pr = gaussian_scalar_nif_update(
                self._mean_log[primary_m],
                self._prec[primary_m],
                val_eff,
                tau_eff,
            )
            self._mean_log[primary_m] = mu
            self._prec[primary_m] = pr

    def decide(
        self,
        sim: Simulator,
        signal: Signal,
        market_env: MarketEnvironment,
    ) -> TradeIntent | None:
        self.update_posterior(signal)
        if signal.market_id in self._mean_log:
            return self._consider_trade(signal.market_id, market_env)
        return None

    def review(
        self,
        sim: Simulator,
        market_env: MarketEnvironment,
    ) -> list[TradeIntent]:
        out: list[TradeIntent] = []
        for m_id in self.market_ids:
            req = self._consider_trade(m_id, market_env)
            if req is not None:
                out.append(req)
        return out

    def fire_noise(
        self, sim: Simulator, market_env: MarketEnvironment
    ) -> TradeIntent | None:
        return None

    def _consider_trade(
        self,
        market_id: int,
        market_env: MarketEnvironment,
    ) -> TradeIntent | None:
        req, cost = route_log_space_trade(
            agent_id=self.agent_id,
            market_id=market_id,
            posterior_mean_log=self._mean_log[market_id],
            posterior_precision=self._prec[market_id],
            market_env=market_env,
            disagreement_threshold_log=self.disagreement_threshold_log,
            target_quantity=self.trade_size,
            safety_margin=self.safety_margin,
            available=self.available,
            min_quantity=self.min_trade_quantity,
            confidence_weighted=self.confidence_weighted,
            prior_precision_for_sizing=self.signal_precision_assumed,
            confidence_floor=self.confidence_floor,
            confidence_ceiling=self.confidence_ceiling,
        )
        if req is not None:
            self.pending_cost += cost
        return req


def make_aggregation_depth_pool(
    n_agents: int,
    base_id: int,
    budget: float,
    primary_markets: tuple[int, ...],
    observed_markets: tuple[int, ...],
    cross_weights: dict[tuple[int, int], float],
    prior_precisions: Optional[list[float]] = None,
    **kwargs,
) -> list[AggregatedEvidenceAgent]:
    """Pool with varying ``prior_precision``.

    Required in ``kwargs``: ``initial_log_fair_mean`` (per-primary log anchors),
    and any other ``AggregatedEvidenceAgent`` fields not covered above.
    """
    if n_agents <= 0:
        raise ValueError("n_agents must be positive")
    if prior_precisions is None:
        if n_agents == 1:
            prior_precisions = [2.0]
        else:
            log_lo, log_hi = math.log(0.5), math.log(10.0)
            prior_precisions = [
                math.exp(log_lo + (log_hi - log_lo) * i / (n_agents - 1))
                for i in range(n_agents)
            ]
    if len(prior_precisions) != n_agents:
        raise ValueError(
            f"len(prior_precisions) = {len(prior_precisions)} != n_agents = {n_agents}"
        )
    return [
        AggregatedEvidenceAgent(
            agent_id=base_id + i,
            budget=budget,
            market_ids=primary_markets,
            observed_markets=observed_markets,
            cross_weights=cross_weights,
            prior_precision=prior_precisions[i],
            **kwargs,
        )
        for i in range(n_agents)
    ]


__all__ = ["AggregatedEvidenceAgent", "make_aggregation_depth_pool"]
