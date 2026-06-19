"""
Discrete-event simulator core.

Ported from:
  /Users/adityabhosale/Downloads/Projects/lmsr-preclinical-markets/sim/simulator.py
"""
from __future__ import annotations

from typing import Any, Callable, Optional

import numpy as np

from environment.events import Event, EventQueue, EventPriority

EventHandler = Callable[["Simulator", Event], None]


class Simulator:
    def __init__(self, rng: np.random.Generator, time_resolution: int = 1) -> None:
        if not isinstance(time_resolution, int) or time_resolution < 1:
            raise ValueError("time_resolution must be a positive integer")
        self.rng = rng
        self.time_resolution = time_resolution
        self.queue = EventQueue()
        self._now: int = 0
        self._handlers: dict[str, EventHandler] = {}
        self._processed: int = 0

    @property
    def now(self) -> int:
        return self._now

    @property
    def now_time(self) -> float:
        return self._now / self.time_resolution

    def register_handler(self, event_type: str, handler: EventHandler) -> None:
        if event_type in self._handlers:
            raise ValueError(f"handler already registered for {event_type!r}")
        self._handlers[event_type] = handler

    def schedule(
        self,
        delay: int,
        event_type: str,
        payload: Any = None,
        priority: int = EventPriority.DECISION,
    ) -> Event:
        if delay < 0:
            raise ValueError(f"cannot schedule in the past (delay={delay})")
        return self.queue.push(self._now + delay, event_type, payload, priority)

    def schedule_at(
        self,
        timestamp: int,
        event_type: str,
        payload: Any = None,
        priority: int = EventPriority.DECISION,
    ) -> Event:
        if timestamp < self._now:
            raise ValueError(
                f"cannot schedule in the past (timestamp={timestamp} < now={self._now})"
            )
        return self.queue.push(timestamp, event_type, payload, priority)

    def run_until(self, until_ts: int) -> int:
        n = 0
        while self.queue and self.queue.peek().timestamp <= until_ts:
            event = self.queue.pop()
            self._dispatch(event)
            n += 1
        if until_ts > self._now:
            self._now = until_ts
        return n

    def run_count(self, max_events: int) -> int:
        if max_events < 0:
            raise ValueError("max_events must be >= 0")
        n = 0
        while n < max_events and self.queue:
            event = self.queue.pop()
            self._dispatch(event)
            n += 1
        return n

    def _dispatch(self, event: Event) -> None:
        self._now = event.timestamp
        handler = self._handlers.get(event.event_type)
        if handler is None:
            raise KeyError(f"no handler registered for event_type {event.event_type!r}")
        handler(self, event)
        self._processed += 1

    @property
    def events_processed(self) -> int:
        return self._processed


def schedule_poisson(
    sim: Simulator,
    rate_per_unit_time: float,
    event_type: str,
    until_ts: int,
    payload_fn: Optional[Callable[["Simulator", int], Any]] = None,
    priority: int = EventPriority.SIGNAL,
) -> int:
    if rate_per_unit_time <= 0:
        return 0

    start_tick = sim.now
    if until_ts <= start_tick:
        return 0

    rate_per_tick = rate_per_unit_time / sim.time_resolution
    if rate_per_tick <= 0:
        return 0

    n_scheduled = 0
    t_cont = float(start_tick)
    while True:
        gap = sim.rng.exponential(1.0 / rate_per_tick)
        t_cont += gap
        if t_cont > until_ts:
            break
        t_tick = int(round(t_cont))
        if t_tick < sim.now:
            t_tick = sim.now
        if t_tick > until_ts:
            break
        # payload_fn receives the scheduled tick so emission-time values (e.g. a
        # moving truth read at t_tick) are snapshotted into the event at schedule
        # time. Signals are the only caller; the value is still fixed at emission.
        payload = payload_fn(sim, t_tick) if payload_fn is not None else None
        sim.schedule_at(t_tick, event_type, payload, priority)
        n_scheduled += 1

    return n_scheduled
