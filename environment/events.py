"""
Event queue and priorities for the discrete-event simulator.

Ported from:
  /Users/adityabhosale/Downloads/Projects/lmsr-preclinical-markets/sim/events.py
"""
from __future__ import annotations

from dataclasses import dataclass
import heapq
from typing import Any, Optional


class EventPriority:
    """Lower integer fires first within the same timestamp."""

    SIGNAL = 0
    DECISION = 100
    TRADE = 200
    BOOKKEEPING = 1000


@dataclass(eq=False)
class Event:
    timestamp: int
    priority: int
    insertion_order: int
    event_type: str
    payload: Any = None

    def __lt__(self, other: "Event") -> bool:
        return (
            (self.timestamp, self.priority, self.insertion_order)
            < (other.timestamp, other.priority, other.insertion_order)
        )

    def __repr__(self) -> str:
        return (
            f"Event(t={self.timestamp}, prio={self.priority}, "
            f"seq={self.insertion_order}, type={self.event_type!r})"
        )


class EventQueue:
    def __init__(self) -> None:
        self._heap: list[Event] = []
        self._counter: int = 0

    def push(
        self,
        timestamp: int,
        event_type: str,
        payload: Any = None,
        priority: int = EventPriority.DECISION,
    ) -> Event:
        if not isinstance(timestamp, int) or isinstance(timestamp, bool):
            raise TypeError(
                f"timestamp must be int, got {type(timestamp).__name__}={timestamp!r}"
            )
        event = Event(
            timestamp=timestamp,
            priority=priority,
            insertion_order=self._counter,
            event_type=event_type,
            payload=payload,
        )
        self._counter += 1
        heapq.heappush(self._heap, event)
        return event

    def pop(self) -> Event:
        if not self._heap:
            raise IndexError("pop from empty EventQueue")
        return heapq.heappop(self._heap)

    def peek(self) -> Optional[Event]:
        return self._heap[0] if self._heap else None

    def __len__(self) -> int:
        return len(self._heap)

    def __bool__(self) -> bool:
        return bool(self._heap)

    @property
    def total_pushed(self) -> int:
        return self._counter
