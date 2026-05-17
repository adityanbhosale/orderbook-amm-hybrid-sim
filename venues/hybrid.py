"""CLOB plus per-tick refreshed passive LP quotes."""
from __future__ import annotations

from dataclasses import dataclass

from venues.base import OrderResult, Venue, VenueState
from venues.clob import CLOB

LP_AGENT_ID = "-2"


@dataclass
class HybridLpConfig:
    lp_spread_pct: float = 0.005
    lp_base_qty: float = 1.0
    lp_decay_factor: float = 0.7
    lp_anchor_price: float = 100.0
    lp_levels: int = 3


class HybridVenue(Venue):
    """Composition: inner :class:`CLOB` + algorithmic LP (``agent_id=-2``)."""

    def __init__(self, clob: CLOB | None = None, *, lp: HybridLpConfig | None = None):
        self._clob = clob if clob is not None else CLOB()
        self._lp_cfg = lp or HybridLpConfig()
        self._lp_order_ids: list[str] = []

    @property
    def inner_clob(self) -> CLOB:
        return self._clob

    def tick(self) -> None:
        for oid in self._lp_order_ids:
            self._clob.cancel_order(LP_AGENT_ID, oid)
        self._lp_order_ids.clear()
        self._clob.tick()
        st = self._clob.get_state()
        mid = st.mid_price if st.mid_price is not None else self._lp_cfg.lp_anchor_price
        cfg = self._lp_cfg
        for k in range(1, cfg.lp_levels + 1):
            q = cfg.lp_base_qty * (cfg.lp_decay_factor ** (k - 1))
            bid_p = mid * (1.0 - cfg.lp_spread_pct * k)
            ask_p = mid * (1.0 + cfg.lp_spread_pct * k)
            rb = self._clob.submit_limit_order(LP_AGENT_ID, "buy", q, bid_p)
            if rb.order_id:
                self._lp_order_ids.append(rb.order_id)
            ra = self._clob.submit_limit_order(LP_AGENT_ID, "sell", q, ask_p)
            if ra.order_id:
                self._lp_order_ids.append(ra.order_id)

    def get_state(self) -> VenueState:
        return self._clob.get_state()

    def submit_market_order(
        self, agent_id: str, side: str, quantity: float
    ) -> OrderResult:
        return self._clob.submit_market_order(agent_id, side, quantity)

    def submit_limit_order(
        self, agent_id: str, side: str, quantity: float, price: float
    ) -> OrderResult:
        return self._clob.submit_limit_order(agent_id, side, quantity, price)

    def cancel_order(self, agent_id: str, order_id: str) -> bool:
        return self._clob.cancel_order(agent_id, order_id)

    def estimate_impact(self, side: str, quantity: float) -> float:
        return self._clob.estimate_impact(side, quantity)


__all__ = ["HybridLpConfig", "HybridVenue", "LP_AGENT_ID"]
