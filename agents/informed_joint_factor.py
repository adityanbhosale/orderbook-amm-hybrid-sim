"""Joint Gaussian posterior over latent factors driving log-prices (misspecified, no ε_m)."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from environment.market_environment import MarketEnvironment
from environment.signals import Signal
from environment.simulator import Simulator
from environment.trade_records import TradeIntent

from agents.belief_utils import (
    log_mid_reference,
    route_log_space_trade,
)


@dataclass
class JointFactorFairValueAgent:
    """
    Information-form posterior (Λ, η) on latent factors f.

    Belief model: L_m = α_m + β_m^T f, ignoring idiosyncratic ε_m in inference.
    Signals are noisy observations of full log-fair-value; updates use
    y = z − α_j as the informative residual for the factor likelihood
    (continuous analog of CrossMarketConsistencyAgent in ``sim/agentic.py``).
    """

    agent_id: int
    budget: float
    market_ids: tuple[int, ...]
    observed_markets: tuple[int, ...]
    loadings: dict[int, np.ndarray]
    alpha_by_market: dict[int, float]
    observation_delay: int = 0
    review_interval: int = 1000
    prior_precision_scale: float = 1.0
    signal_noise_inflation: float = 1.0
    disagreement_threshold_log: float = 0.005
    trade_size: float = 1.0
    confidence_weighted: bool = False
    confidence_floor: float = 0.25
    confidence_ceiling: float = 4.0
    safety_margin: float = 1.2
    min_trade_quantity: float = 1e-6
    #: Belief process variance per unit time (matrix analog of the scalar q).
    #: 0 = stationary target (default, byte-identical to the precision-only
    #: matrix update). >0 inflates the factor covariance by q·Δt before each
    #: update so the joint posterior tracks a moving target instead of freezing.
    q: float = 0.0

    deployed: float = field(default=0.0, init=False)
    pending_cost: float = field(default=0.0, init=False)
    _k: int = field(default=0, init=False)
    _Lambda: np.ndarray = field(init=False)
    _eta: np.ndarray = field(init=False)
    _observed_set: set[int] = field(default_factory=set, init=False)
    _primary_set: set[int] = field(default_factory=set, init=False)
    #: tick of the last posterior update (single, shared factor posterior);
    #: -1 = never updated. Δt = now - this drives the q>0 covariance inflation.
    _last_update_tick: int = field(default=-1, init=False)
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
        if self.prior_precision_scale <= 0:
            raise ValueError("prior_precision_scale must be positive")
        if self.signal_noise_inflation <= 0:
            raise ValueError("signal_noise_inflation must be positive")
        if self.disagreement_threshold_log < 0:
            raise ValueError("disagreement_threshold_log must be non-negative")
        if self.trade_size <= 0:
            raise ValueError("trade_size must be positive")
        if self.safety_margin < 1.0:
            raise ValueError("safety_margin must be >= 1.0")
        if self.q < 0.0:
            raise ValueError("process variance q must be non-negative")

        self.market_ids = tuple(self.market_ids)
        self.observed_markets = tuple(self.observed_markets)
        self._observed_set = set(self.observed_markets)
        self._primary_set = set(self.market_ids)

        for m in self.market_ids:
            if m not in self._observed_set:
                raise ValueError(f"primary market {m} must be in observed_markets")
        for m in self.observed_markets:
            if m not in self.loadings:
                raise ValueError(f"loadings missing market {m}")
            if m not in self.alpha_by_market:
                raise ValueError(f"alpha_by_market missing market {m}")

        first = np.asarray(self.loadings[self.observed_markets[0]], dtype=float)
        if first.ndim != 1 or first.shape[0] < 1:
            raise ValueError("loadings must be 1-D non-empty arrays")
        self._k = int(first.shape[0])

        norm_load: dict[int, np.ndarray] = {}
        for m, beta in self.loadings.items():
            arr = np.asarray(beta, dtype=float).ravel()
            if arr.shape != (self._k,):
                raise ValueError(
                    f"loading dim mismatch for market {m}: "
                    f"expected ({self._k},), got {arr.shape}"
                )
            norm_load[m] = arr
        self.loadings = norm_load

        self._Lambda = self.prior_precision_scale * np.eye(self._k)
        self._eta = np.zeros(self._k)

    @property
    def available(self) -> float:
        return self.budget - self.deployed - self.pending_cost

    @property
    def k(self) -> int:
        return self._k

    def observes(self, market_id: int) -> bool:
        return market_id in self._observed_set

    def posterior_factor_mean(self) -> np.ndarray:
        return np.linalg.solve(self._Lambda, self._eta)

    def implied_log_fair(self, market_id: int) -> float:
        mu_f = self.posterior_factor_mean()
        return float(self.alpha_by_market[market_id] + self.loadings[market_id] @ mu_f)

    def _heuristic_linear_precision(self, market_id: int) -> float:
        """Reference heuristic: β_m^T Λ β_m (see CrossMarketConsistencyAgent)."""
        b = self.loadings[market_id]
        return float(b @ self._Lambda @ b)

    def _prior_precision_sizing(self) -> float:
        return self.prior_precision_scale * max(self._k, 1)

    def _decay_information(self, qdt: float) -> None:
        """Random-walk predict step on the factor posterior (mean preserved).

        Works in COVARIANCE form, which is PSD-preserving by construction:
        Σ = Λ⁻¹ (PD, since Λ is PD), inflate Σ ← Σ + qdt·I (adding a PD diagonal
        keeps it PD), then Λ ← Σ⁻¹ (inverse of PD is PD). The mean μ = Σ η is
        held fixed and η rebuilt as Λ·μ. Symmetrized for fp hygiene. Only ever
        called on the q>0 path, so it never perturbs the q=0 baseline.
        """
        sigma = np.linalg.inv(self._Lambda)
        mu = sigma @ self._eta
        sigma = sigma + qdt * np.eye(self._k)
        lam = np.linalg.inv(sigma)
        lam = 0.5 * (lam + lam.T)
        self._Lambda = lam
        self._eta = lam @ mu

    def update_posterior(self, signal: Signal, now: int) -> None:
        j = signal.market_id
        if j not in self._observed_set:
            return
        if signal.noise_std <= 0:
            return
        dt = 0.0 if self._last_update_tick < 0 else float(now - self._last_update_tick)
        self._last_update_tick = now
        if self.q > 0.0 and dt > 0.0:
            # Inflate factor covariance by q·Δt for elapsed time, before
            # absorbing this signal. Skipped entirely at q=0 (no inversion
            # round-trip) so the matrix arithmetic below is byte-identical.
            self._decay_information(self.q * dt)
        beta = self.loadings[j]
        sigma_eff = signal.noise_std * self.signal_noise_inflation
        tau_j = 1.0 / (sigma_eff * sigma_eff)
        y = float(signal.value - self.alpha_by_market[j])
        self._Lambda = self._Lambda + tau_j * np.outer(beta, beta)
        self._eta = self._eta + tau_j * y * beta

    def decide(
        self,
        sim: Simulator,
        signal: Signal,
        market_env: MarketEnvironment,
    ) -> TradeIntent | None:
        self.update_posterior(signal, sim.now)
        if signal.market_id not in self._primary_set:
            return None
        impl = self.implied_log_fair(signal.market_id)
        return self._consider_trade(signal.market_id, impl, market_env)

    def review(
        self,
        sim: Simulator,
        market_env: MarketEnvironment,
    ) -> list[TradeIntent]:
        implied = {
            m: self.implied_log_fair(m) for m in self.market_ids
        }
        opportunities: list[tuple[float, int, float]] = []
        for m_id in self.market_ids:
            ml = implied[m_id]
            mid_px = market_env.mid_price(m_id)
            if mid_px is None:
                continue
            mid_l = log_mid_reference(mid_px)
            diff_mag = abs(ml - mid_l)
            if diff_mag >= self.disagreement_threshold_log:
                opportunities.append((diff_mag, m_id, ml))
        opportunities.sort(key=lambda t: -t[0])

        trades: list[TradeIntent] = []
        for _, m_id, impl in opportunities:
            req = self._consider_trade(m_id, impl, market_env)
            if req is not None:
                trades.append(req)
        return trades

    def fire_noise(
        self, sim: Simulator, market_env: MarketEnvironment
    ) -> TradeIntent | None:
        return None

    def _consider_trade(
        self,
        market_id: int,
        implied_log: float,
        market_env: MarketEnvironment,
    ) -> TradeIntent | None:
        post_prec = self._heuristic_linear_precision(market_id)
        req, cost = route_log_space_trade(
            agent_id=self.agent_id,
            market_id=market_id,
            posterior_mean_log=implied_log,
            posterior_precision=post_prec,
            market_env=market_env,
            disagreement_threshold_log=self.disagreement_threshold_log,
            target_quantity=self.trade_size,
            safety_margin=self.safety_margin,
            available=self.available,
            min_quantity=self.min_trade_quantity,
            confidence_weighted=self.confidence_weighted,
            prior_precision_for_sizing=self._prior_precision_sizing(),
            confidence_floor=self.confidence_floor,
            confidence_ceiling=self.confidence_ceiling,
        )
        if req is not None:
            self.pending_cost += cost
        return req


def make_joint_factor_agent(
    agent_id: int,
    budget: float,
    primary_markets: tuple[int, ...],
    observed_markets: tuple[int, ...],
    loadings_matrix: np.ndarray,
    initial_mid_by_market: dict[int, float],
    f_prior_mean: np.ndarray | None = None,
    **kwargs,
) -> JointFactorFairValueAgent:
    """
    Build loadings dict and α offsets from a (n_markets, k) ``loadings_matrix``
    and opening mids; α_m = log(mid_m) − β_m^T f̄.
    """
    fm = f_prior_mean
    if fm is None:
        fm = np.zeros(loadings_matrix.shape[1])
    fm_a = np.asarray(fm, dtype=float).ravel()
    loadings: dict[int, np.ndarray] = {}
    alpha: dict[int, float] = {}
    for m in observed_markets:
        b = np.asarray(loadings_matrix[m], dtype=float).ravel()
        loadings[m] = b
        mid = initial_mid_by_market[m]
        alpha[m] = float(np.log(mid) - b @ fm_a)
    return JointFactorFairValueAgent(
        agent_id=agent_id,
        budget=budget,
        market_ids=primary_markets,
        observed_markets=observed_markets,
        loadings=loadings,
        alpha_by_market=alpha,
        **kwargs,
    )


__all__ = ["JointFactorFairValueAgent", "make_joint_factor_agent"]
