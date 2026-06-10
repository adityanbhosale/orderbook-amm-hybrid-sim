"""Price-time priority central limit order book (minimal)."""
from __future__ import annotations

import math
from bisect import insort
from collections import deque
from typing import Optional

from venues.base import MakerFill, OrderResult, Venue, VenueState


class EmptyBookError(RuntimeError):
    """No resting liquidity on the side needed for pricing or impact."""

    pass


BOOTSTRAP_AGENT_ID = "-1"
DEFAULT_BOOTSTRAP_LEVELS = 5
DEFAULT_BOOTSTRAP_SPREAD_STEP = 0.001


def _pkey(p: float) -> float:
    return round(float(p), 12)


class CLOB(Venue):
    """
    Bids high-first for incoming sells; asks low-first for incoming buys.
    ``estimate_impact`` walks the book without mutating state.
    """

    def __init__(self) -> None:
        self._bid_prices: list[float] = []
        self._ask_prices: list[float] = []
        self._bids: dict[float, deque] = {}
        self._asks: dict[float, deque] = {}
        self._order_counter = 0
        self._tick = 0
        self._maker_fills: list[MakerFill] = []

    def _gen_order_id(self) -> str:
        self._order_counter += 1
        return f"o{self._order_counter}"

    def _best_bid(self) -> Optional[float]:
        return self._bid_prices[-1] if self._bid_prices else None

    def _best_ask(self) -> Optional[float]:
        return self._ask_prices[0] if self._ask_prices else None

    def _mid(self) -> Optional[float]:
        bb, ba = self._best_bid(), self._best_ask()
        if bb is None or ba is None:
            return None
        return (bb + ba) / 2.0

    def drain_maker_fills(self) -> list[MakerFill]:
        out = self._maker_fills
        self._maker_fills = []
        return out

    def _record_maker_fill(
        self,
        agent_id: str,
        maker_side: str,
        quantity: float,
        price: float,
        order_id: str,
        mid_before: Optional[float],
    ) -> None:
        self._maker_fills.append(
            MakerFill(
                agent_id=agent_id,
                side=maker_side,
                quantity=quantity,
                price=price,
                order_id=order_id,
                mid_before=mid_before,
                mid_after=self._mid(),
            )
        )

    def get_state(self) -> VenueState:
        bb, ba = self._best_bid(), self._best_ask()
        if bb is None or ba is None:
            return VenueState(None, bb, ba, None, self._tick)
        spread = ba - bb
        mid = (bb + ba) / 2.0
        return VenueState(mid, bb, ba, spread, self._tick)

    def tick(self) -> None:
        self._tick += 1

    def seed_initial_book(
        self,
        anchor_price: float,
        depth_per_level: float,
        *,
        levels: int = DEFAULT_BOOTSTRAP_LEVELS,
        spread_step: float = DEFAULT_BOOTSTRAP_SPREAD_STEP,
    ) -> None:
        """Post persistent bid/ask ladder so ``mid_price`` exists at cold start."""
        if anchor_price <= 0:
            raise ValueError("anchor_price must be positive")
        if depth_per_level <= 0:
            raise ValueError("depth_per_level must be positive")
        if levels < 1:
            raise ValueError("levels must be >= 1")

        aid = BOOTSTRAP_AGENT_ID
        for k in range(1, levels + 1):
            bid_p = anchor_price * (1.0 - spread_step * k)
            ask_p = anchor_price * (1.0 + spread_step * k)
            self.submit_limit_order(aid, "buy", depth_per_level, bid_p)
            self.submit_limit_order(aid, "sell", depth_per_level, ask_p)

    def cancel_order(self, agent_id: str, order_id: str) -> bool:
        aid = str(agent_id)
        for plist, book in ((self._bid_prices, self._bids), (self._ask_prices, self._asks)):
            for px in list(plist):
                dq = book.get(px)
                if not dq:
                    continue
                new_dq: deque = deque()
                found = False
                while dq:
                    ag, q, oid = dq.popleft()
                    if ag == aid and oid == order_id:
                        found = True
                        continue
                    new_dq.append((ag, q, oid))
                if found:
                    if new_dq:
                        book[px] = new_dq
                    else:
                        del book[px]
                        plist.remove(px)
                    return True
        return False

    def estimate_impact(self, side: str, quantity: float) -> float:
        if quantity < 0:
            raise ValueError("quantity must be non-negative")
        st = self.get_state()
        if quantity == 0.0:
            if st.mid_price is None:
                raise EmptyBookError("empty book: no mid for zero-size quote")
            return float(st.mid_price)

        if side == "buy":
            if not self._ask_prices:
                raise EmptyBookError("no asks")
            tot_cost, got = self._simulate_buy(quantity)
            if got < quantity - 1e-12:
                return float("inf")
            return tot_cost / got
        if side == "sell":
            if not self._bid_prices:
                raise EmptyBookError("no bids")
            tot_recv, got = self._simulate_sell(quantity)
            if got < quantity - 1e-12:
                return 0.0
            return tot_recv / got
        raise ValueError("side must be 'buy' or 'sell'")

    def _simulate_buy(self, qty: float) -> tuple[float, float]:
        remaining = qty
        cost = 0.0
        filled = 0.0
        for price in list(self._ask_prices):
            if remaining <= 0:
                break
            dq = self._asks.get(price)
            if not dq:
                continue
            for ag, q, oid in list(dq):
                if remaining <= 0:
                    break
                take = min(remaining, q)
                cost += take * price
                filled += take
                remaining -= take
        return cost, filled

    def _simulate_sell(self, qty: float) -> tuple[float, float]:
        remaining = qty
        recv = 0.0
        filled = 0.0
        for price in reversed(self._bid_prices):
            if remaining <= 0:
                break
            dq = self._bids.get(price)
            if not dq:
                continue
            for ag, q, oid in list(dq):
                if remaining <= 0:
                    break
                take = min(remaining, q)
                recv += take * price
                filled += take
                remaining -= take
        return recv, filled

    def submit_market_order(
        self, agent_id: str, side: str, quantity: float
    ) -> OrderResult:
        if quantity <= 0:
            raise ValueError("quantity must be positive")
        aid = str(agent_id)
        if side == "buy":
            return self._execute_buy(aid, quantity, max_price=math.inf)
        if side == "sell":
            return self._execute_sell(aid, quantity, min_price=0.0)
        raise ValueError("side must be 'buy' or 'sell'")

    def submit_limit_order(
        self, agent_id: str, side: str, quantity: float, price: float
    ) -> OrderResult:
        if quantity <= 0:
            raise ValueError("quantity must be positive")
        aid = str(agent_id)
        px = _pkey(price)
        if side == "buy":
            return self._limit_buy(aid, quantity, px)
        if side == "sell":
            return self._limit_sell(aid, quantity, px)
        raise ValueError("side must be 'buy' or 'sell'")

    def _ensure_level(self, book: str, price: float) -> None:
        k = _pkey(price)
        if book == "bid":
            if k not in self._bids:
                self._bids[k] = deque()
                insort(self._bid_prices, k)
        else:
            if k not in self._asks:
                self._asks[k] = deque()
                insort(self._ask_prices, k)

    def _execute_buy(
        self, agent_id: str, qty: float, max_price: float
    ) -> OrderResult:
        remaining = qty
        tot_cost = 0.0
        filled = 0.0
        while remaining > 1e-15 and self._ask_prices:
            ap = self._ask_prices[0]
            if ap > max_price + 1e-12:
                break
            dq = self._asks[ap]
            if not dq:
                self._ask_prices.pop(0)
                del self._asks[ap]
                continue
            ag, q, oid = dq[0]
            take = min(remaining, q)
            mid_before = self._mid()
            tot_cost += take * ap
            filled += take
            remaining -= take
            if take >= q - 1e-15:
                dq.popleft()
            else:
                dq[0] = (ag, q - take, oid)
            if not dq:
                self._ask_prices.pop(0)
                del self._asks[ap]
            self._record_maker_fill(ag, "sell", take, ap, oid, mid_before)
        unfilled = qty - filled
        return OrderResult(
            filled_quantity=filled,
            avg_fill_price=tot_cost / filled if filled > 0 else 0.0,
            remaining_quantity=unfilled,
            order_id=None,
            fees_paid=0.0,
        )

    def _execute_sell(
        self, agent_id: str, qty: float, min_price: float
    ) -> OrderResult:
        remaining = qty
        tot_recv = 0.0
        filled = 0.0
        while remaining > 1e-15 and self._bid_prices:
            bp = self._bid_prices[-1]
            if bp < min_price - 1e-12:
                break
            dq = self._bids[bp]
            if not dq:
                self._bid_prices.pop()
                del self._bids[bp]
                continue
            ag, q, oid = dq[0]
            take = min(remaining, q)
            mid_before = self._mid()
            tot_recv += take * bp
            filled += take
            remaining -= take
            if take >= q - 1e-15:
                dq.popleft()
            else:
                dq[0] = (ag, q - take, oid)
            if not dq:
                self._bid_prices.pop()
                del self._bids[bp]
            self._record_maker_fill(ag, "buy", take, bp, oid, mid_before)
        unfilled = qty - filled
        return OrderResult(
            filled_quantity=filled,
            avg_fill_price=tot_recv / filled if filled > 0 else 0.0,
            remaining_quantity=unfilled,
            order_id=None,
            fees_paid=0.0,
        )

    def _limit_buy(self, agent_id: str, qty: float, limit_px: float) -> OrderResult:
        remaining = qty
        tot_cost = 0.0
        filled = 0.0
        while (
            remaining > 1e-15
            and self._ask_prices
            and self._ask_prices[0] <= limit_px + 1e-12
        ):
            ap = self._ask_prices[0]
            dq = self._asks[ap]
            if not dq:
                self._ask_prices.pop(0)
                del self._asks[ap]
                continue
            ag, q, oid = dq[0]
            take = min(remaining, q)
            mid_before = self._mid()
            tot_cost += take * ap
            filled += take
            remaining -= take
            if take >= q - 1e-15:
                dq.popleft()
            else:
                dq[0] = (ag, q - take, oid)
            if not dq:
                self._ask_prices.pop(0)
                del self._asks[ap]
            self._record_maker_fill(ag, "sell", take, ap, oid, mid_before)
        resting_oid: Optional[str] = None
        rest_qty = remaining
        if rest_qty > 1e-12:
            resting_oid = self._gen_order_id()
            self._ensure_level("bid", limit_px)
            self._bids[limit_px].append((agent_id, rest_qty, resting_oid))
        return OrderResult(
            filled_quantity=filled,
            avg_fill_price=tot_cost / filled if filled > 0 else 0.0,
            remaining_quantity=rest_qty,
            order_id=resting_oid,
            fees_paid=0.0,
        )

    def _limit_sell(self, agent_id: str, qty: float, limit_px: float) -> OrderResult:
        remaining = qty
        tot_recv = 0.0
        filled = 0.0
        while (
            remaining > 1e-15
            and self._bid_prices
            and self._bid_prices[-1] >= limit_px - 1e-12
        ):
            bp = self._bid_prices[-1]
            dq = self._bids[bp]
            if not dq:
                self._bid_prices.pop()
                del self._bids[bp]
                continue
            ag, q, oid = dq[0]
            take = min(remaining, q)
            mid_before = self._mid()
            tot_recv += take * bp
            filled += take
            remaining -= take
            if take >= q - 1e-15:
                dq.popleft()
            else:
                dq[0] = (ag, q - take, oid)
            if not dq:
                self._bid_prices.pop()
                del self._bids[bp]
            self._record_maker_fill(ag, "buy", take, bp, oid, mid_before)
        resting_oid: Optional[str] = None
        rest_qty = remaining
        if rest_qty > 1e-12:
            resting_oid = self._gen_order_id()
            self._ensure_level("ask", limit_px)
            self._asks[limit_px].append((agent_id, rest_qty, resting_oid))
        return OrderResult(
            filled_quantity=filled,
            avg_fill_price=tot_recv / filled if filled > 0 else 0.0,
            remaining_quantity=rest_qty,
            order_id=resting_oid,
            fees_paid=0.0,
        )


__all__ = [
    "BOOTSTRAP_AGENT_ID",
    "CLOB",
    "DEFAULT_BOOTSTRAP_LEVELS",
    "DEFAULT_BOOTSTRAP_SPREAD_STEP",
    "EmptyBookError",
]
