"""Background Poisson-style noise trader for event-driven simulations."""
from __future__ import annotations

from dataclasses import dataclass, field

from environment.margin import committed_capital
from environment.market_environment import MarketEnvironment
from environment.signals import Signal
from environment.simulator import Simulator
from environment.trade_records import TradeIntent
from environment.trading_utils import downsize_quantity_for_capital
from venues.clob import EmptyBookError


@dataclass
class EventDrivenNoiseAgent:
    """
    Random market/side/size flow; uses impact-based capital checks like
    ``NoiseTrader`` in reference ``sim/agentic.py``.
    """

    agent_id: int
    budget: float
    market_ids: tuple[int, ...]
    arrival_rate_per_unit: float = 1.5
    mean_trade_size: float = 2.0
    size_jitter: float = 0.4
    safety_margin: float = 1.15
    observation_delay: int = field(default=0, init=False)
    review_interval: int = field(default=0, init=False)

    deployed: float = field(default=0.0, init=False)
    pending_cost: float = field(default=0.0, init=False)

    def __post_init__(self) -> None:
        if not self.market_ids:
            raise ValueError("market_ids must be non-empty")
        if self.arrival_rate_per_unit < 0:
            raise ValueError("arrival_rate_per_unit must be non-negative")
        if self.mean_trade_size <= 0:
            raise ValueError("mean_trade_size must be positive")
        if not (0 <= self.size_jitter < 1):
            raise ValueError("size_jitter must be in [0, 1)")
        if self.safety_margin < 1.0:
            raise ValueError("safety_margin must be >= 1.0")
        self.market_ids = tuple(self.market_ids)

    @property
    def available(self) -> float:
        return self.budget - self.deployed - self.pending_cost

    def observes(self, market_id: int) -> bool:
        return False

    def decide(
        self, sim: Simulator, signal: Signal, market_env: MarketEnvironment
    ) -> TradeIntent | None:
        return None

    def review(
        self, sim: Simulator, market_env: MarketEnvironment
    ) -> list[TradeIntent]:
        return []

    def fire_noise(
        self, sim: Simulator, market_env: MarketEnvironment
    ) -> TradeIntent | None:
        m_id = int(sim.rng.choice(self.market_ids))
        side = "buy" if bool(sim.rng.integers(0, 2)) else "sell"
        mult = float(
            sim.rng.uniform(1.0 - self.size_jitter, 1.0 + self.size_jitter)
        )
        target = mult * self.mean_trade_size
        venue = market_env.venue(m_id)
        margin = market_env.margin
        q = downsize_quantity_for_capital(
            venue,
            side,
            target,
            self.available,
            margin,
            safety_margin=self.safety_margin,
        )
        if q < 1e-8:
            return None
        try:
            cost = committed_capital(
                venue, side, q, margin, safety_margin=self.safety_margin
            )
        except EmptyBookError:
            return None
        self.pending_cost += cost
        return TradeIntent(
            market_id=m_id,
            agent_id=self.agent_id,
            side=side,
            quantity=q,
            order_type="market",
        )


__all__ = ["EventDrivenNoiseAgent"]
