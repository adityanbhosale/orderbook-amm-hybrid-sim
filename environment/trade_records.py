"""Trade intents and execution log rows."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TradeIntent:
    """Market or limit instruction scheduled as a TRADE event payload."""

    market_id: int
    agent_id: int
    side: str  # "buy" | "sell"
    quantity: float
    order_type: str = "market"  # "market" | "limit"
    limit_price: float | None = None
    capital_safety_margin: float = 1.0

    def __post_init__(self) -> None:
        if self.market_id < 0:
            raise ValueError(f"market_id must be non-negative, got {self.market_id}")
        if self.quantity <= 0:
            raise ValueError(f"quantity must be positive, got {self.quantity}")
        if self.side not in ("buy", "sell"):
            raise ValueError("side must be 'buy' or 'sell'")
        if self.order_type not in ("market", "limit"):
            raise ValueError("order_type must be 'market' or 'limit'")
        if self.order_type == "limit" and self.limit_price is None:
            raise ValueError("limit_price required for limit orders")
        if self.capital_safety_margin < 1.0:
            raise ValueError("capital_safety_margin must be >= 1.0")


@dataclass(frozen=True)
class TradeRecord:
    """One executed trade (actual fills).

    Every execution produces one "taker" record (the incoming order) and, on
    order-book venues, one "maker" record per resting order consumed. Maker
    records carry ``capital_committed=0.0`` — the maker's capital was already
    committed when the resting limit order was placed (see ``CostEntry``).
    """

    timestamp: int
    market_id: int
    agent_id: int
    side: str
    quantity: float
    avg_fill_price: float
    fees_paid: float
    capital_committed: float
    mid_price_before: float
    mid_price_after: float
    liquidity: str = "taker"  # "taker" | "maker"


@dataclass(frozen=True)
class CostEntry:
    """Capital/fee cash flow for one processed ``TradeIntent``.

    Exactly one entry per intent (even if nothing filled, e.g. a fully
    resting limit order). ``AgentPopulation._sync_costs`` reconciles agent
    budgets from this log; ``trade_log`` holds only actual fills.
    """

    timestamp: int
    market_id: int
    agent_id: int
    capital_committed: float
    fees_paid: float
