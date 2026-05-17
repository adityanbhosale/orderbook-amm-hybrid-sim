"""Smoke tests for hybrid venue (CLOB + passive LP)."""
from __future__ import annotations

import pytest

from venues.clob import CLOB
from venues.hybrid import LP_AGENT_ID, HybridLpConfig, HybridVenue


def _count_lp_orders(h: HybridVenue) -> int:
    c: CLOB = h.inner_clob
    n = 0
    for px in c._bid_prices:
        for ag, _q, _oid in c._bids.get(px, ()):
            if ag == LP_AGENT_ID:
                n += 1
    for px in c._ask_prices:
        for ag, _q, _oid in c._asks.get(px, ()):
            if ag == LP_AGENT_ID:
                n += 1
    return n


def _best_ask_price(h: HybridVenue) -> float:
    c = h.inner_clob
    assert c._ask_prices
    return float(c._ask_prices[0])


def test_smoke_hybrid_lp_tick_and_refresh() -> None:
    h = HybridVenue(lp=HybridLpConfig(lp_anchor_price=100.0, lp_base_qty=1.0))
    assert _count_lp_orders(h) == 0

    h.tick()
    assert _count_lp_orders(h) == 6
    lp_ids_after_post = list(h._lp_order_ids)

    mid = h.get_state().mid_price
    assert mid is not None
    assert mid == pytest.approx(100.0, rel=0, abs=0.05)

    ba0 = _best_ask_price(h)
    res = h.submit_market_order("9", "buy", 1.0)
    assert res.filled_quantity == pytest.approx(1.0)
    assert res.avg_fill_price == pytest.approx(ba0, rel=0, abs=1e-6)

    h.tick()
    assert _count_lp_orders(h) == 6
    assert h._lp_order_ids != lp_ids_after_post
