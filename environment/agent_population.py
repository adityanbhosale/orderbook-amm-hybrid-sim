"""
Multi-agent orchestration: signal fan-out, delayed decisions, reviews, trades.

Ported structure from:
  /Users/adityabhosale/Downloads/Projects/lmsr-preclinical-markets/sim/agentic.py
  (class AgentPopulation)
"""
from __future__ import annotations

from typing import Optional, Protocol

from environment.events import Event, EventPriority
from environment.information import InformationEnvironment
from environment.market_environment import MarketEnvironment
from environment.signals import Signal
from environment.simulator import Simulator
from environment.trade_records import TradeIntent


class PopulationAgent(Protocol):
    """Agents participating in ``AgentPopulation`` must expose this surface."""

    agent_id: int
    budget: float
    deployed: float
    pending_cost: float
    observation_delay: int
    review_interval: int
    arrival_rate_per_unit: float

    def observes(self, market_id: int) -> bool:
        ...

    def decide(
        self, sim: Simulator, signal: Signal, market_env: MarketEnvironment
    ) -> TradeIntent | None:
        ...

    def review(
        self, sim: Simulator, market_env: MarketEnvironment
    ) -> list[TradeIntent]:
        ...

    def fire_noise(
        self, sim: Simulator, market_env: MarketEnvironment
    ) -> TradeIntent | None:
        ...


class AgentPopulation:
    """
    Registers ``signal``, ``agent_decision``, ``agent_review``, ``noise_trade``
    handlers. Capital is reconciled from ``MarketEnvironment.trade_log``:
    each agent's ``deployed`` increases by ``capital_committed + fees_paid``.
    """

    SIGNAL_EVENT: str = InformationEnvironment.SIGNAL_EVENT
    DECISION_EVENT: str = "agent_decision"
    REVIEW_EVENT: str = "agent_review"
    NOISE_EVENT: str = "noise_trade"

    def __init__(self, agents: list[PopulationAgent]) -> None:
        if len({a.agent_id for a in agents}) != len(agents):
            raise ValueError("agent_ids must be unique across the population")
        self.agents: list[PopulationAgent] = list(agents)
        self.agent_by_id: dict[int, PopulationAgent] = {a.agent_id: a for a in self.agents}
        self._sim: Optional[Simulator] = None
        self._market_env: Optional[MarketEnvironment] = None
        self._until_ts: Optional[int] = None
        self._registered: bool = False
        self._log_cursor: int = 0

    @property
    def n_agents(self) -> int:
        return len(self.agents)

    def register(
        self,
        sim: Simulator,
        market_env: MarketEnvironment,
        until_ts: int,
    ) -> None:
        if self._registered:
            raise RuntimeError("AgentPopulation.register called twice")
        sim.register_handler(self.SIGNAL_EVENT, self._on_signal)
        sim.register_handler(self.DECISION_EVENT, self._on_decision)
        sim.register_handler(self.REVIEW_EVENT, self._on_review)
        sim.register_handler(self.NOISE_EVENT, self._on_noise)
        self._sim = sim
        self._market_env = market_env
        self._until_ts = until_ts
        for agent in self.agents:
            if agent.review_interval > 0:
                first = sim.now + agent.review_interval
                if first <= until_ts:
                    sim.schedule_at(
                        timestamp=first,
                        event_type=self.REVIEW_EVENT,
                        payload=agent.agent_id,
                        priority=EventPriority.DECISION,
                    )
            if agent.arrival_rate_per_unit > 0:
                self._schedule_next_noise(agent)
        self._registered = True

    def sync_costs(self) -> None:
        """Public wrapper for tests / end-of-run bookkeeping."""
        self._sync_costs()

    def _sync_costs(self) -> None:
        log = self._market_env.trade_log
        if self._log_cursor == len(log):
            return
        affected: set[int] = set()
        for r in log[self._log_cursor:]:
            a = self.agent_by_id.get(r.agent_id)
            if a is not None:
                cash = r.capital_committed + r.fees_paid
                a.deployed += cash
                affected.add(r.agent_id)
        for agent_id in affected:
            self.agent_by_id[agent_id].pending_cost = 0.0
        self._log_cursor = len(log)

    def _schedule_next_noise(self, agent: PopulationAgent) -> None:
        sim = self._sim
        if sim is None:
            return
        rate_per_tick = agent.arrival_rate_per_unit / sim.time_resolution
        if rate_per_tick <= 0:
            return
        gap = sim.rng.exponential(1.0 / rate_per_tick)
        t = sim.now + max(1, int(round(gap)))
        if self._until_ts is not None and t > self._until_ts:
            return
        sim.schedule_at(
            timestamp=t,
            event_type=self.NOISE_EVENT,
            payload=agent.agent_id,
            priority=EventPriority.DECISION,
        )

    def _on_signal(self, sim: Simulator, event: Event) -> None:
        signal: Signal = event.payload
        until = self._until_ts if self._until_ts is not None else sim.now
        for agent in self.agents:
            if not agent.observes(signal.market_id):
                continue
            delay = agent.observation_delay
            if sim.now + delay > until:
                continue
            sim.schedule(
                delay=delay,
                event_type=self.DECISION_EVENT,
                payload=(agent.agent_id, signal),
                priority=EventPriority.DECISION,
            )

    def _on_decision(self, sim: Simulator, event: Event) -> None:
        self._sync_costs()
        agent_id, signal = event.payload
        agent = self.agent_by_id[agent_id]
        req = agent.decide(sim, signal, self._market_env)
        if req is not None:
            sim.schedule(
                delay=0,
                event_type=MarketEnvironment.TRADE_EVENT,
                payload=req,
                priority=EventPriority.TRADE,
            )

    def _on_review(self, sim: Simulator, event: Event) -> None:
        self._sync_costs()
        agent_id: int = event.payload
        agent = self.agent_by_id[agent_id]
        for req in agent.review(sim, self._market_env):
            sim.schedule(
                delay=0,
                event_type=MarketEnvironment.TRADE_EVENT,
                payload=req,
                priority=EventPriority.TRADE,
            )
        if self._until_ts is not None and agent.review_interval > 0:
            next_t = sim.now + agent.review_interval
            if next_t <= self._until_ts:
                sim.schedule_at(
                    timestamp=next_t,
                    event_type=self.REVIEW_EVENT,
                    payload=agent.agent_id,
                    priority=EventPriority.DECISION,
                )

    def _on_noise(self, sim: Simulator, event: Event) -> None:
        self._sync_costs()
        agent_id: int = event.payload
        agent = self.agent_by_id[agent_id]
        req = agent.fire_noise(sim, self._market_env)
        if req is not None:
            sim.schedule(
                delay=0,
                event_type=MarketEnvironment.TRADE_EVENT,
                payload=req,
                priority=EventPriority.TRADE,
            )
        if agent.arrival_rate_per_unit > 0:
            self._schedule_next_noise(agent)


__all__ = ["AgentPopulation", "PopulationAgent"]
