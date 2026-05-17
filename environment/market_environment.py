"""
Multi-market routing over ``Venue`` instances.

Replaces LMSR-specific ``sim/market_env.py`` from the reference repo with a
thin orchestration layer that owns the trade log and TRADE_EVENT handler.
"""
from __future__ import annotations

import math

from environment.events import Event
from environment.margin import MarginSpec, committed_capital_limit
from environment.simulator import Simulator
from environment.trade_records import TradeIntent, TradeRecord
from venues.base import Venue, VenueState


class MarketEnvironment:
    """Maps ``market_id`` to a ``Venue`` and records executed trades."""

    TRADE_EVENT = "trade"

    def __init__(
        self,
        venues: dict[int, Venue],
        *,
        margin: MarginSpec | None = None,
    ) -> None:
        if not venues:
            raise ValueError("venues must be non-empty")
        self.venues = dict(venues)
        self.trade_log: list[TradeRecord] = []
        self.margin = margin or MarginSpec()
        self._registered = False

    @property
    def n_markets(self) -> int:
        return len(self.venues)

    def venue(self, market_id: int) -> Venue:
        return self.venues[market_id]

    def mid_price(self, market_id: int) -> float | None:
        return self.venues[market_id].get_state().mid_price

    @staticmethod
    def _float_mid(mp: float | None) -> float:
        return float(mp) if mp is not None else math.nan

    def get_state(self, market_id: int) -> VenueState:
        return self.venues[market_id].get_state()

    def estimate_impact(self, market_id: int, side: str, quantity: float) -> float:
        return self.venues[market_id].estimate_impact(side, quantity)

    def register(self, sim: Simulator) -> None:
        if self._registered:
            raise RuntimeError("MarketEnvironment.register called twice")
        sim.register_handler(self.TRADE_EVENT, self._on_trade)
        self._registered = True

    def _capital_from_fill(
        self, side: str, quantity: float, avg_fill: float
    ) -> float:
        notional = quantity * avg_fill
        if side == "buy":
            return float(notional * self.margin.long_margin_fraction)
        return float(notional * self.margin.short_margin_fraction)

    def execute_market_order(
        self,
        sim: Simulator,
        market_id: int,
        agent_id: int,
        side: str,
        quantity: float,
        *,
        agent_id_str: str | None = None,
    ) -> TradeRecord:
        """Immediate execution path (no TRADE_EVENT) for tests and simple runners."""
        if market_id not in self.venues:
            raise KeyError(f"unknown market_id={market_id}")
        venue = self.venues[market_id]
        pre_mid = venue.get_state().mid_price
        aid = agent_id_str if agent_id_str is not None else str(agent_id)
        res = venue.submit_market_order(aid, side, quantity)
        post_mid = venue.get_state().mid_price
        cap = self._capital_from_fill(side, res.filled_quantity, res.avg_fill_price)
        rec = TradeRecord(
            timestamp=sim.now,
            market_id=market_id,
            agent_id=agent_id,
            side=side,
            quantity=res.filled_quantity,
            avg_fill_price=res.avg_fill_price,
            fees_paid=res.fees_paid,
            capital_committed=cap,
            mid_price_before=self._float_mid(pre_mid),
            mid_price_after=self._float_mid(post_mid),
        )
        self.trade_log.append(rec)
        return rec

    def _on_trade(self, sim: Simulator, event: Event) -> None:
        payload = event.payload
        if not isinstance(payload, TradeIntent):
            raise TypeError(
                f"trade event payload must be TradeIntent, got {type(payload).__name__}"
            )
        mid = payload.market_id
        if mid not in self.venues:
            raise KeyError(f"unknown market_id={mid}")
        venue = self.venues[mid]
        pre_mid = venue.get_state().mid_price

        aid = str(payload.agent_id)
        lim_px = payload.limit_price
        if payload.order_type == "market":
            res = venue.submit_market_order(aid, payload.side, payload.quantity)
            cap = self._capital_from_fill(
                payload.side, res.filled_quantity, res.avg_fill_price
            )
        elif payload.order_type == "limit":
            if lim_px is None:
                raise ValueError("limit_price required for limit orders")
            res = venue.submit_limit_order(
                aid, payload.side, payload.quantity, float(lim_px)
            )
            cap = committed_capital_limit(
                payload.side,
                payload.quantity,
                float(lim_px),
                self.margin,
                safety_margin=payload.capital_safety_margin,
            )
        else:
            raise ValueError(f"unsupported order_type {payload.order_type!r}")

        post_mid = venue.get_state().mid_price
        self.trade_log.append(
            TradeRecord(
                timestamp=sim.now,
                market_id=mid,
                agent_id=payload.agent_id,
                side=payload.side,
                quantity=res.filled_quantity,
                avg_fill_price=res.avg_fill_price,
                fees_paid=res.fees_paid,
                capital_committed=cap,
                mid_price_before=self._float_mid(pre_mid),
                mid_price_after=self._float_mid(post_mid),
            )
        )
