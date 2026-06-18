"""Modal-prior Gaussian belief in log-price space with misspecified signal precision."""
from __future__ import annotations

from dataclasses import dataclass, field

from environment.market_environment import MarketEnvironment
from environment.signals import Signal
from environment.simulator import Simulator
from environment.trade_records import TradeIntent

from agents.belief_utils import gaussian_scalar_nif_update, route_log_space_trade


@dataclass
class NaiveGaussianBeliefAgent:
    """
    Per-market conjugate Gaussian on log-fair-value L_m.

    Updates use a fixed ``signal_precision_assumed`` for every signal (modal
    bias), not the true ``Signal.noise_std`` — port of NaiveCredentialedAgent
    from ``sim/agentic.py`` in the reference repo.
    """

    agent_id: int
    budget: float
    market_ids: tuple[int, ...]
    initial_log_fair_mean: dict[int, float]
    observation_delay: int = 0
    review_interval: int = 5000
    prior_precision: float = 5.0
    signal_precision_assumed: float = 1.0
    disagreement_threshold_log: float = 0.005
    trade_size: float = 1.0
    confidence_weighted: bool = False
    confidence_floor: float = 0.25
    confidence_ceiling: float = 4.0
    safety_margin: float = 1.2
    min_trade_quantity: float = 1e-6
    #: Belief process variance per unit time. 0 = stationary target (default,
    #: byte-identical to the pre-Kalman filter). >0 decays prior precision by
    #: q·Δt between updates so the belief tracks a moving target instead of
    #: freezing. Δt is ticks since this agent last updated this market.
    q: float = 0.0

    deployed: float = field(default=0.0, init=False)
    pending_cost: float = field(default=0.0, init=False)
    _mean_log: dict[int, float] = field(default_factory=dict, init=False)
    _prec: dict[int, float] = field(default_factory=dict, init=False)
    _last_update_tick: dict[int, int] = field(default_factory=dict, init=False)
    arrival_rate_per_unit: float = field(default=0.0, init=False)

    def __post_init__(self) -> None:
        if not self.market_ids:
            raise ValueError("market_ids must be non-empty")
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
        self.market_ids = tuple(self.market_ids)
        for m in self.market_ids:
            if m not in self.initial_log_fair_mean:
                raise ValueError(f"initial_log_fair_mean missing market {m}")
            self._mean_log[m] = float(self.initial_log_fair_mean[m])
            self._prec[m] = float(self.prior_precision)

    @property
    def available(self) -> float:
        return self.budget - self.deployed - self.pending_cost

    def observes(self, market_id: int) -> bool:
        return market_id in self._mean_log

    def posterior(self, market_id: int) -> tuple[float, float]:
        return self._mean_log[market_id], self._prec[market_id]

    def update_posterior(self, signal: Signal, now: int) -> None:
        m = signal.market_id
        if m not in self._mean_log:
            return
        last = self._last_update_tick.get(m)
        dt = 0.0 if last is None else float(now - last)
        self._last_update_tick[m] = now
        obs_prec = self.signal_precision_assumed
        mu, pr = gaussian_scalar_nif_update(
            self._mean_log[m], self._prec[m], signal.value, obs_prec,
            q=self.q, dt=dt,
        )
        self._mean_log[m] = mu
        self._prec[m] = pr

    def decide(
        self,
        sim: Simulator,
        signal: Signal,
        market_env: MarketEnvironment,
    ) -> TradeIntent | None:
        self.update_posterior(signal, sim.now)
        return self._consider_trade(signal.market_id, market_env)

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
            prior_precision_for_sizing=self.prior_precision,
            confidence_floor=self.confidence_floor,
            confidence_ceiling=self.confidence_ceiling,
        )
        if req is not None:
            self.pending_cost += cost
        return req


__all__ = ["NaiveGaussianBeliefAgent"]
