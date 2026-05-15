"""
Log-linear latent-factor information process and signal scheduling.

Rework of:
  /Users/adityabhosale/Downloads/Projects/lmsr-preclinical-markets/sim/information.py

Truth for market ``m`` uses log-price :math:`L^*_m = \\alpha_m + \\beta_m^\\top f + \\varepsilon_m`
with :math:`f \\sim \\mathcal N(0, I_k)`. Offsets are anchored so that **before** idiosyncratic noise,
log-price equals the declared initial mid: ``alpha_m = log(mid_m(0)) - beta_m^T f``, hence
:math:`L^*_m = \\log(\\text{mid}_m(0)) + \\varepsilon_m`.

Signals observe :math:`z = L^*_m + \\mathcal N(0, \\sigma^2)` with :math:`\\sigma` in log-price units.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from environment.events import EventPriority
from environment.signals import Signal
from environment.simulator import Simulator, schedule_poisson


@dataclass(frozen=True)
class ClusterSpec:
    primary_factor: int
    market_count: int
    primary_loading_mean: float = 1.5
    primary_loading_std: float = 0.2
    secondary_loading_std: float = 0.15

    def __post_init__(self) -> None:
        if self.market_count <= 0:
            raise ValueError("market_count must be positive")
        if self.primary_factor < 0:
            raise ValueError("primary_factor must be non-negative")
        if self.primary_loading_std < 0 or self.secondary_loading_std < 0:
            raise ValueError("loading stds must be non-negative")


@dataclass
class InformationConfig:
    k: int = 5
    clusters: list[ClusterSpec] = field(default_factory=list)
    n_independent_markets: int = 0
    independent_loading_std: float = 0.4
    idiosyncratic_std: float = 0.5

    signal_noise_std: float = 0.02
    tail_noise_std: float = 0.008

    routine_rate_per_market: float = 1.0
    tail_rate_per_market: float = 0.05
    tail_mode: str = "separate"

    #: Length ``n_markets`` — price level :math:`\\text{mid}_m` at simulation start (t=0).
    initial_mid_prices: Optional[np.ndarray] = None

    @property
    def n_markets(self) -> int:
        return sum(c.market_count for c in self.clusters) + self.n_independent_markets

    def validate(self) -> None:
        if self.k < 1:
            raise ValueError("k must be >= 1")
        if self.tail_mode not in {"separate", "marked"}:
            raise ValueError(
                f"tail_mode must be 'separate' or 'marked', got {self.tail_mode!r}"
            )
        for c in self.clusters:
            if c.primary_factor >= self.k:
                raise ValueError(
                    f"cluster primary_factor {c.primary_factor} exceeds k={self.k}"
                )
        if self.n_markets == 0:
            raise ValueError("config defines zero markets")
        for name in (
            "idiosyncratic_std",
            "signal_noise_std",
            "tail_noise_std",
            "independent_loading_std",
        ):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be non-negative")
        for name in ("routine_rate_per_market", "tail_rate_per_market"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be non-negative")
        if self.initial_mid_prices is not None:
            arr = np.asarray(self.initial_mid_prices, dtype=float)
            if arr.shape != (self.n_markets,):
                raise ValueError(
                    f"initial_mid_prices must shape ({self.n_markets},), got {arr.shape}"
                )
            if np.any(arr <= 0):
                raise ValueError("initial_mid_prices must be positive")


@dataclass(frozen=True, eq=False)
class MarketTruth:
    market_id: int
    cluster_id: Optional[int]
    loadings: np.ndarray
    idiosyncratic: float
    alpha: float
    log_fair_value: float
    fair_price: float


class LatentFactorModel:
    def __init__(self, config: InformationConfig, rng: np.random.Generator):
        config.validate()
        self.config = config
        self.f: np.ndarray = rng.standard_normal(config.k)

        if config.initial_mid_prices is not None:
            mids = np.asarray(config.initial_mid_prices, dtype=float)
        else:
            mids = np.ones(config.n_markets, dtype=float)

        truths: list[MarketTruth] = []
        market_id = 0

        for cluster_id, cluster in enumerate(config.clusters):
            for _ in range(cluster.market_count):
                loadings = rng.normal(0.0, cluster.secondary_loading_std, size=config.k)
                loadings[cluster.primary_factor] = rng.normal(
                    cluster.primary_loading_mean, cluster.primary_loading_std
                )
                idio = float(rng.normal(0.0, config.idiosyncratic_std))
                beta_f = float(loadings @ self.f)
                log_mid = float(np.log(mids[market_id]))
                alpha = float(log_mid - beta_f)
                log_fv = alpha + beta_f + idio
                truths.append(
                    MarketTruth(
                        market_id=market_id,
                        cluster_id=cluster_id,
                        loadings=loadings,
                        idiosyncratic=idio,
                        alpha=alpha,
                        log_fair_value=log_fv,
                        fair_price=float(np.exp(log_fv)),
                    )
                )
                market_id += 1

        for _ in range(config.n_independent_markets):
            loadings = rng.normal(0.0, config.independent_loading_std, size=config.k)
            idio = float(rng.normal(0.0, config.idiosyncratic_std))
            beta_f = float(loadings @ self.f)
            log_mid = float(np.log(mids[market_id]))
            alpha = float(log_mid - beta_f)
            log_fv = alpha + beta_f + idio
            truths.append(
                MarketTruth(
                    market_id=market_id,
                    cluster_id=None,
                    loadings=loadings,
                    idiosyncratic=idio,
                    alpha=alpha,
                    log_fair_value=log_fv,
                    fair_price=float(np.exp(log_fv)),
                )
            )
            market_id += 1

        self.truths: list[MarketTruth] = truths
        self.n_markets: int = len(truths)

    def truth(self, market_id: int) -> MarketTruth:
        return self.truths[market_id]

    @property
    def loadings_matrix(self) -> np.ndarray:
        return np.stack([t.loadings for t in self.truths], axis=0)


class InformationEnvironment:
    """Owns ``LatentFactorModel`` and pre-schedules ``Signal`` events."""

    SIGNAL_EVENT = "signal"

    def __init__(self, config: InformationConfig, rng: np.random.Generator):
        self.config = config
        self.rng = rng
        self.world = LatentFactorModel(config, rng)
        self._scheduled = False

    @property
    def truths(self) -> list[MarketTruth]:
        return self.world.truths

    @property
    def n_markets(self) -> int:
        return self.world.n_markets

    def schedule_signals(self, sim: Simulator, until_ts: int) -> dict[str, int]:
        if self._scheduled:
            raise RuntimeError(
                "schedule_signals already called; create a fresh env for a new run"
            )

        cfg = self.config
        routine_count = [0]
        tail_count = [0]

        if cfg.tail_mode == "separate":
            for m_id in range(self.world.n_markets):
                log_fv = self.world.truths[m_id].log_fair_value

                def make_routine(
                    s,
                    mid=m_id,
                    lg=log_fv,
                    rc=routine_count,
                ):
                    rc[0] += 1
                    return Signal(
                        market_id=mid,
                        value=float(
                            lg + s.rng.normal(0.0, cfg.signal_noise_std)
                        ),
                        is_tail=False,
                        noise_std=cfg.signal_noise_std,
                    )

                def make_tail(s, mid=m_id, lg=log_fv, tc=tail_count):
                    tc[0] += 1
                    return Signal(
                        market_id=mid,
                        value=float(lg + s.rng.normal(0.0, cfg.tail_noise_std)),
                        is_tail=True,
                        noise_std=cfg.tail_noise_std,
                    )

                schedule_poisson(
                    sim,
                    rate_per_unit_time=cfg.routine_rate_per_market,
                    event_type=self.SIGNAL_EVENT,
                    until_ts=until_ts,
                    payload_fn=make_routine,
                    priority=EventPriority.SIGNAL,
                )
                schedule_poisson(
                    sim,
                    rate_per_unit_time=cfg.tail_rate_per_market,
                    event_type=self.SIGNAL_EVENT,
                    until_ts=until_ts,
                    payload_fn=make_tail,
                    priority=EventPriority.SIGNAL,
                )

        else:
            total_rate = cfg.routine_rate_per_market + cfg.tail_rate_per_market
            if total_rate > 0:
                p_tail = cfg.tail_rate_per_market / total_rate
            else:
                p_tail = 0.0

            for m_id in range(self.world.n_markets):
                log_fv = self.world.truths[m_id].log_fair_value

                def make_signal(
                    s,
                    mid=m_id,
                    lg=log_fv,
                    pt=p_tail,
                    rc=routine_count,
                    tc=tail_count,
                ):
                    is_tail = bool(s.rng.random() < pt)
                    sigma = cfg.tail_noise_std if is_tail else cfg.signal_noise_std
                    if is_tail:
                        tc[0] += 1
                    else:
                        rc[0] += 1
                    return Signal(
                        market_id=mid,
                        value=float(lg + s.rng.normal(0.0, sigma)),
                        is_tail=is_tail,
                        noise_std=sigma,
                    )

                schedule_poisson(
                    sim,
                    rate_per_unit_time=total_rate,
                    event_type=self.SIGNAL_EVENT,
                    until_ts=until_ts,
                    payload_fn=make_signal,
                    priority=EventPriority.SIGNAL,
                )

        self._scheduled = True
        return {
            "routine": routine_count[0],
            "tail": tail_count[0],
            "total": routine_count[0] + tail_count[0],
        }
