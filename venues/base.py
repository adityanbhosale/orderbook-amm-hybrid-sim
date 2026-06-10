"""Abstract base class for trading venues.

A Venue is any matching environment that accepts orders from agents and
produces a price. AMMs, CLOBs, and hybrid mechanisms all implement this
interface, allowing the same agent populations to trade across regimes.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional


@dataclass
class OrderResult:
    """Result of submitting an order to a venue."""
    filled_quantity: float       # Quantity actually executed
    avg_fill_price: float        # Volume-weighted average fill price
    remaining_quantity: float    # Unfilled remainder (zero for market orders on AMMs)
    order_id: Optional[str]      # For tracking resting limit orders; None for market orders
    fees_paid: float


@dataclass(frozen=True)
class MakerFill:
    """One deferred fill leg, surfaced outside the synchronous submit path.

    Produced inside the venue's matching loops and buffered until
    ``drain_maker_fills`` is called, so the environment can record fills that
    do not reach ``_on_trade`` via the synchronous ``OrderResult``.

    Continuous venues (CLOB/hybrid) emit only maker legs (the taker side is
    recorded synchronously in ``_on_trade``), so ``liquidity`` defaults to
    ``"maker"`` and ``fees_paid`` to ``0.0`` — existing call sites are
    unchanged. Batch venues (FBA) produce BOTH legs at clear time and tag
    each with ``liquidity`` ("maker" = limit order, "taker" = market order).
    """

    agent_id: str
    side: str                    # this leg's side
    quantity: float              # quantity executed on this leg
    price: float                 # execution price (limit price / batch p*)
    order_id: str
    mid_before: Optional[float]  # venue mid immediately before execution
    mid_after: Optional[float]   # venue mid immediately after execution
    liquidity: str = "maker"     # "maker" | "taker"
    fees_paid: float = 0.0


@dataclass
class VenueState:
    """Snapshot of venue state, returned to agents on observation."""

    mid_price: float | None
    best_bid: Optional[float]
    best_ask: Optional[float]
    spread: Optional[float]
    tick: int


class Venue(ABC):
    """Abstract base class for trading venues."""

    @abstractmethod
    def submit_market_order(
        self, agent_id: str, side: str, quantity: float
    ) -> OrderResult:
        """Submit a market order. side is 'buy' or 'sell'."""
        ...

    @abstractmethod
    def submit_limit_order(
        self, agent_id: str, side: str, quantity: float, price: float
    ) -> OrderResult:
        """Submit a limit order. May raise NotImplementedError for pure AMM venues."""
        ...

    @abstractmethod
    def cancel_order(self, agent_id: str, order_id: str) -> bool:
        """Cancel an outstanding limit order. May be no-op for AMMs."""
        ...

    @abstractmethod
    def get_state(self) -> VenueState:
        """Return current venue state for agent observation."""
        ...

    @abstractmethod
    def estimate_impact(self, side: str, quantity: float) -> float:
        """Estimate the volume-weighted average fill price for a market order
        of the given side and quantity, without executing it.

        AMM: compute from cost function.
        CLOB: walk the book.
        Hybrid: walk the combined book.

        Returns the expected average fill price. Agents use this to size
        trades against a target impact rather than against mechanism-specific
        liquidity parameters.
        """
        ...

    @abstractmethod
    def tick(self) -> None:
        """Advance venue by one timestep. Used for time-dependent state."""
        ...

    def drain_maker_fills(self) -> list[MakerFill]:
        """Return and clear deferred fills produced since the last drain.

        Venues without resting orders (pure AMMs) have no deferred fills and
        use this default. Continuous order-book venues override it to surface
        maker executions; batch venues (FBA) surface BOTH legs of every
        clear-time execution through this channel, tagged via
        ``MakerFill.liquidity``.
        """
        return []