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


@dataclass(frozen=True)
class TradeRecord:
    """One executed trade (actual fills)."""

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
