"""Minimal xy = k AMM for tests (mid = y / x)."""
from __future__ import annotations

from venues.base import OrderResult, Venue, VenueState


class ConstantProductAMM(Venue):
    """Constant-product pool; base reserve ``x``, quote reserve ``y``.

    Mid price is quote per base: ``y / x``. Market buy removes base from the
    pool; market sell adds base. ``estimate_impact`` matches executed VWAP.
    """

    def __init__(
        self,
        reserve_x: float,
        reserve_y: float,
        *,
        tick: int = 0,
    ) -> None:
        if reserve_x <= 0 or reserve_y <= 0:
            raise ValueError("reserves must be positive")
        self.reserve_x = reserve_x
        self.reserve_y = reserve_y
        self._tick = tick

    def _k(self) -> float:
        return self.reserve_x * self.reserve_y

    def _mid(self) -> float:
        return self.reserve_y / self.reserve_x

    def get_state(self) -> VenueState:
        return VenueState(
            mid_price=self._mid(),
            best_bid=None,
            best_ask=None,
            spread=None,
            tick=self._tick,
        )

    def tick(self) -> None:
        self._tick += 1

    def estimate_impact(self, side: str, quantity: float) -> float:
        """VWAP in quote currency per unit base."""
        if quantity <= 0:
            raise ValueError("quantity must be positive")
        x, y = self.reserve_x, self.reserve_y
        k = x * y
        if side == "buy":
            if quantity >= x:
                raise ValueError("buy size exceeds pool base liquidity")
            x1 = x - quantity
            y1 = k / x1
            return float((y1 - y) / quantity)
        if side == "sell":
            x1 = x + quantity
            y1 = k / x1
            return float((y - y1) / quantity)
        raise ValueError("side must be 'buy' or 'sell'")

    def submit_market_order(
        self, agent_id: str, side: str, quantity: float
    ) -> OrderResult:
        if quantity <= 0:
            raise ValueError("quantity must be positive")
        x, y = self.reserve_x, self.reserve_y
        k = x * y
        if side == "buy":
            if quantity >= x:
                raise ValueError("buy size exceeds pool base liquidity")
            x1 = x - quantity
            y1 = k / x1
            quote_paid = y1 - y
            avg = quote_paid / quantity
            self.reserve_x, self.reserve_y = x1, y1
        elif side == "sell":
            x1 = x + quantity
            y1 = k / x1
            quote_recv = y - y1
            avg = quote_recv / quantity
            self.reserve_x, self.reserve_y = x1, y1
        else:
            raise ValueError("side must be 'buy' or 'sell'")
        return OrderResult(
            filled_quantity=quantity,
            avg_fill_price=float(avg),
            remaining_quantity=0.0,
            order_id=None,
            fees_paid=0.0,
        )

    def submit_limit_order(
        self, agent_id: str, side: str, quantity: float, price: float
    ) -> OrderResult:
        raise NotImplementedError("ConstantProductAMM is quote-only via market orders")

    def cancel_order(self, agent_id: str, order_id: str) -> bool:
        return False


__all__ = ["ConstantProductAMM"]
