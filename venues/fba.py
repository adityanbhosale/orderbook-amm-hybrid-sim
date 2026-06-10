"""Frequent batch auction (FBA) venue: periodic uniform-price call auction.

Orders accumulate between clears; every ``tau_ticks`` calls to ``tick()`` the
venue runs a uniform-price call auction over the resting limit book plus any
market orders queued since the last clear. ``submit_*`` never fills
synchronously — both legs of every execution are surfaced at clear time
through the ``drain_maker_fills`` channel (``MakerFill`` with a ``liquidity``
tag), which the environment routes into ``TradeRecord``s.

Clearing algorithm is a native float reimplementation of the uniform-price
call auction in
``kalshi-polymarket-microstructure/batch_counterfactual/auction.py``
(max-volume objective, midpoint tie-break over the optimal price interval,
pro-rata rationing at the margin with largest-remainder rounding — both
tie-breaks ported verbatim, see ASSUMPTION-1/2 there). Quantities are
quantized to an integer grid (``QUANTITY_QUANTUM``) so rationing is exact
integer arithmetic: deterministic and conserving, no RNG anywhere.

Price/quantity conventions mirror ``venues/clob.py``: float prices rounded
through ``_pkey`` (no new price type), market buys are treated as bids at the
top of the sim's price domain (``+inf``) and market sells as asks at the
bottom (``0.0``) — infinitely elastic at any clearing price.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

from venues.base import MakerFill, OrderResult, Venue, VenueState
from venues.clob import EmptyBookError, _pkey

#: Quantity quantum: all quantities are converted to integer multiples of
#: 1/QUANTITY_SCALE units so pro-rata + largest-remainder rationing runs in
#: exact integer arithmetic (auction.py operates on integral contract counts;
#: this is the float-world equivalent).
QUANTITY_SCALE = 10**9
QUANTITY_QUANTUM = 1.0 / QUANTITY_SCALE

#: Price domain bounds: market orders participate at any clearing price.
MARKET_BUY_PRICE = math.inf
MARKET_SELL_PRICE = 0.0


def _to_q(qty: float) -> int:
    return int(round(qty * QUANTITY_SCALE))


def _from_q(qty_q: int) -> float:
    return qty_q / QUANTITY_SCALE


@dataclass
class _BatchOrder:
    """One participant in the next clear (resting limit or queued market)."""

    order_id: str
    agent_id: str
    side: str          # "buy" | "sell"
    price: float       # _pkey-rounded limit price; +inf / 0.0 for market orders
    qty_q: int         # remaining quantity in integer quanta
    seq: int           # submission sequence: deterministic processing order
    is_market: bool    # market orders expire if unfilled at the next clear


def _largest_remainder(weights: list[int], total: int) -> list[int]:
    """Deterministic pro-rata allocation of ``total`` quanta across ``weights``.

    Integer port of ``auction.py::_largest_remainder`` (ASSUMPTION-2): floor
    the exact pro-rata shares, then hand out the residue one quantum at a
    time by descending fractional remainder, ties broken by list index.
    """
    if total <= 0 or not weights:
        return [0] * len(weights)
    wsum = sum(weights)
    if wsum <= 0:
        return [0] * len(weights)
    floors = [total * w // wsum for w in weights]
    residue = total - sum(floors)
    # fractional remainder of total*w/wsum is (total*w mod wsum)/wsum
    order = sorted(range(len(weights)), key=lambda i: (-(total * weights[i] % wsum), i))
    out = list(floors)
    for k in range(residue):
        out[order[k]] += 1
    return out


class FBAVenue(Venue):
    """Periodic uniform-price call auction over a resting limit book.

    ``tick()`` increments an internal counter; when ``counter % tau_ticks == 0``
    the batch clears. Orders submitted during ``[last_clear, this_clear)``
    participate in this clear (the sweep fires ``tick()`` at priority -50
    before trades, so a submit at step t lands in t+1's clear).

    Submit semantics are fully deferred: ``submit_limit_order`` and
    ``submit_market_order`` return the "accepted, pending" ``OrderResult``
    (``filled_quantity=0, remaining_quantity=qty``, real ``order_id``).
    Resting limits persist across clears until filled/cancelled; market
    orders that cannot fill at the next clear expire silently.
    """

    def __init__(self, tau_ticks: int, *, fee_rate: float = 0.0) -> None:
        if not isinstance(tau_ticks, int) or tau_ticks < 1:
            raise ValueError("tau_ticks must be a positive integer")
        if fee_rate < 0.0:
            raise ValueError("fee_rate must be non-negative")
        self.tau_ticks = tau_ticks
        #: Taker fee as fraction of notional, charged on market-order legs at
        #: clear time. Maker (limit) legs pay no fee (Entry-2 convention).
        self.fee_rate = fee_rate
        self._tick = 0
        self._order_counter = 0
        self._seq = 0
        #: Resting limit book: order_id -> _BatchOrder (insertion-ordered).
        self._resting: dict[str, _BatchOrder] = {}
        #: Market orders queued for the next clear only.
        self._market_queue: list[_BatchOrder] = []
        #: Clear-time fills (both legs) buffered until drained.
        self._fills: list[MakerFill] = []

    # ------------------------------------------------------------------ #
    # Book state                                                         #
    # ------------------------------------------------------------------ #

    def _gen_order_id(self) -> str:
        self._order_counter += 1
        return f"o{self._order_counter}"

    def _next_seq(self) -> int:
        self._seq += 1
        return self._seq

    def _best_bid(self) -> Optional[float]:
        bids = [o.price for o in self._resting.values() if o.side == "buy"]
        return max(bids) if bids else None

    def _best_ask(self) -> Optional[float]:
        asks = [o.price for o in self._resting.values() if o.side == "sell"]
        return min(asks) if asks else None

    def _mid(self) -> Optional[float]:
        bb, ba = self._best_bid(), self._best_ask()
        if bb is None or ba is None:
            return None
        return (bb + ba) / 2.0

    def get_state(self) -> VenueState:
        """Resting-book best-bid/ask midpoint between clears.

        Note: an FBA book may be crossed between clears (no continuous
        matching), in which case ``spread`` is negative; the midpoint is
        still the natural inter-clear price observation.
        """
        bb, ba = self._best_bid(), self._best_ask()
        if bb is None or ba is None:
            return VenueState(None, bb, ba, None, self._tick)
        return VenueState((bb + ba) / 2.0, bb, ba, ba - bb, self._tick)

    # ------------------------------------------------------------------ #
    # Submission (deferred — never fills synchronously)                  #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _pending(quantity: float, order_id: str) -> OrderResult:
        return OrderResult(
            filled_quantity=0.0,
            avg_fill_price=0.0,
            remaining_quantity=quantity,
            order_id=order_id,
            fees_paid=0.0,
        )

    def submit_limit_order(
        self, agent_id: str, side: str, quantity: float, price: float
    ) -> OrderResult:
        if quantity <= 0:
            raise ValueError("quantity must be positive")
        if side not in ("buy", "sell"):
            raise ValueError("side must be 'buy' or 'sell'")
        if not math.isfinite(price) or price <= 0:
            raise ValueError("price must be positive and finite")
        oid = self._gen_order_id()
        self._resting[oid] = _BatchOrder(
            order_id=oid,
            agent_id=str(agent_id),
            side=side,
            price=_pkey(price),
            qty_q=_to_q(quantity),
            seq=self._next_seq(),
            is_market=False,
        )
        return self._pending(quantity, oid)

    def submit_market_order(
        self, agent_id: str, side: str, quantity: float
    ) -> OrderResult:
        if quantity <= 0:
            raise ValueError("quantity must be positive")
        if side not in ("buy", "sell"):
            raise ValueError("side must be 'buy' or 'sell'")
        oid = self._gen_order_id()
        self._market_queue.append(
            _BatchOrder(
                order_id=oid,
                agent_id=str(agent_id),
                side=side,
                price=MARKET_BUY_PRICE if side == "buy" else MARKET_SELL_PRICE,
                qty_q=_to_q(quantity),
                seq=self._next_seq(),
                is_market=True,
            )
        )
        # Same pending shape as limits; the order_id is real so clear-time
        # fills remain attributable (deviates from the CLOB's "None for
        # market orders" because FBA market orders live until the clear).
        return self._pending(quantity, oid)

    def cancel_order(self, agent_id: str, order_id: str) -> bool:
        o = self._resting.get(order_id)
        if o is None or o.agent_id != str(agent_id):
            return False
        del self._resting[order_id]
        return True

    # ------------------------------------------------------------------ #
    # Clearing                                                           #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _solve_clear(
        orders: list[_BatchOrder],
    ) -> Optional[tuple[float, list[tuple[_BatchOrder, int]], list[tuple[_BatchOrder, int]]]]:
        """Uniform-price clear of ``orders``; pure (no mutation).

        Returns ``(p_star, buy_allocs, sell_allocs)`` with allocations in
        quanta, or ``None`` if no positive volume clears. Mirrors
        ``auction.py``: candidate prices are the distinct limit prices
        (executable volume is maximized at a limit price; market orders
        participate at every price but define no candidates), the objective
        is max executable volume, and a flat max-volume interval clears at
        its MIDPOINT (ASSUMPTION-1). Allocation recomputes participation at
        p* and rations the long side pro-rata with largest remainder
        (ASSUMPTION-2), in submission order.
        """
        candidates = sorted({o.price for o in orders if not o.is_market})
        if not candidates:
            return None

        def demand_q(p: float) -> int:
            return sum(
                o.qty_q for o in orders
                if o.side == "buy" and (o.is_market or o.price >= p)
            )

        def supply_q(p: float) -> int:
            return sum(
                o.qty_q for o in orders
                if o.side == "sell" and (o.is_market or o.price <= p)
            )

        best_vol = 0
        best_prices: list[float] = []
        for p in candidates:
            vol = min(demand_q(p), supply_q(p))
            if vol > best_vol:
                best_vol = vol
                best_prices = [p]
            elif vol == best_vol and best_vol > 0:
                best_prices.append(p)
        if best_vol <= 0:
            return None

        p_star = (best_prices[0] + best_prices[-1]) / 2.0

        buys = sorted(
            (o for o in orders
             if o.side == "buy" and (o.is_market or o.price >= p_star)),
            key=lambda o: o.seq,
        )
        sells = sorted(
            (o for o in orders
             if o.side == "sell" and (o.is_market or o.price <= p_star)),
            key=lambda o: o.seq,
        )
        dem = sum(o.qty_q for o in buys)
        sup = sum(o.qty_q for o in sells)
        trade = min(dem, sup)
        if trade <= 0:
            return None

        if dem <= sup:
            buy_q = [o.qty_q for o in buys]
            sell_q = _largest_remainder([o.qty_q for o in sells], trade)
        else:
            sell_q = [o.qty_q for o in sells]
            buy_q = _largest_remainder([o.qty_q for o in buys], trade)

        return p_star, list(zip(buys, buy_q)), list(zip(sells, sell_q))

    def _run_clear(self) -> None:
        mid_before = self._mid()
        orders = list(self._resting.values()) + self._market_queue
        solved = self._solve_clear(orders)
        # Unfilled market orders expire regardless of outcome (record nothing).
        self._market_queue = []
        if solved is None:
            return
        p_star, buy_allocs, sell_allocs = solved

        # Apply book mutations first so every fill in the batch can be
        # stamped with the same pre-clear and post-clear mids.
        executed: list[tuple[_BatchOrder, int]] = []
        for o, alloc_q in buy_allocs + sell_allocs:
            if alloc_q <= 0:
                continue
            executed.append((o, alloc_q))
            if not o.is_market:
                o.qty_q -= alloc_q
                if o.qty_q <= 0:
                    del self._resting[o.order_id]
        mid_after = self._mid()

        executed.sort(key=lambda pair: pair[0].seq)
        for o, alloc_q in executed:
            qty = _from_q(alloc_q)
            fee = self.fee_rate * qty * p_star if o.is_market else 0.0
            self._fills.append(
                MakerFill(
                    agent_id=o.agent_id,
                    side=o.side,
                    quantity=qty,
                    price=p_star,
                    order_id=o.order_id,
                    mid_before=mid_before,
                    mid_after=mid_after,
                    liquidity="taker" if o.is_market else "maker",
                    fees_paid=fee,
                )
            )

    def tick(self) -> None:
        self._tick += 1
        if self._tick % self.tau_ticks == 0:
            self._run_clear()

    def drain_maker_fills(self) -> list[MakerFill]:
        """Return and clear ALL clear-time fills (maker AND taker legs)."""
        out = self._fills
        self._fills = []
        return out

    # ------------------------------------------------------------------ #
    # Impact estimation                                                  #
    # ------------------------------------------------------------------ #

    def estimate_impact(self, side: str, quantity: float) -> float:
        """Expected clearing price if ``quantity`` joined the next batch.

        Simulates adding a market order of the given side to the current
        pending state (resting limits + queued market orders) and returns
        the hypothetical uniform clearing price p* — every FBA fill executes
        at p*, so it is also the VWAP. Mirrors CLOB conventions at the
        edges: raises ``EmptyBookError`` with no opposite resting limits;
        returns ``inf`` (buy) / ``0.0`` (sell) if the order cannot fully
        fill at the hypothetical clear.
        """
        if quantity < 0:
            raise ValueError("quantity must be non-negative")
        if side not in ("buy", "sell"):
            raise ValueError("side must be 'buy' or 'sell'")
        if quantity == 0.0:
            mid = self._mid()
            if mid is None:
                raise EmptyBookError("empty book: no mid for zero-size quote")
            return float(mid)

        opposite = "sell" if side == "buy" else "buy"
        if not any(o.side == opposite for o in self._resting.values()):
            raise EmptyBookError("no asks" if side == "buy" else "no bids")

        probe = _BatchOrder(
            order_id="__probe__",
            agent_id="__probe__",
            side=side,
            price=MARKET_BUY_PRICE if side == "buy" else MARKET_SELL_PRICE,
            qty_q=_to_q(quantity),
            seq=self._seq + 1,
            is_market=True,
        )
        orders = list(self._resting.values()) + self._market_queue + [probe]
        solved = self._solve_clear(orders)
        if solved is None:
            return float("inf") if side == "buy" else 0.0
        p_star, buy_allocs, sell_allocs = solved
        allocs = buy_allocs if side == "buy" else sell_allocs
        probe_fill = next(
            (q for o, q in allocs if o.order_id == "__probe__"), 0
        )
        if probe_fill < probe.qty_q:
            return float("inf") if side == "buy" else 0.0
        return float(p_star)

    # ------------------------------------------------------------------ #
    # Bootstrap                                                          #
    # ------------------------------------------------------------------ #

    def seed_initial_book(
        self,
        anchor_price: float,
        depth_per_level: float,
        *,
        levels: int = 5,
        spread_step: float = 0.001,
    ) -> None:
        """Post a persistent bid/ask ladder so ``mid_price`` exists at cold start.

        Same shape as ``CLOB.seed_initial_book``; the ladder rests in the
        batch book and participates in clears like any other limit orders.
        """
        if anchor_price <= 0:
            raise ValueError("anchor_price must be positive")
        if depth_per_level <= 0:
            raise ValueError("depth_per_level must be positive")
        if levels < 1:
            raise ValueError("levels must be >= 1")
        from venues.clob import BOOTSTRAP_AGENT_ID

        for k in range(1, levels + 1):
            self.submit_limit_order(
                BOOTSTRAP_AGENT_ID, "buy", depth_per_level,
                anchor_price * (1.0 - spread_step * k),
            )
            self.submit_limit_order(
                BOOTSTRAP_AGENT_ID, "sell", depth_per_level,
                anchor_price * (1.0 + spread_step * k),
            )


__all__ = [
    "FBAVenue",
    "MARKET_BUY_PRICE",
    "MARKET_SELL_PRICE",
    "QUANTITY_QUANTUM",
]
