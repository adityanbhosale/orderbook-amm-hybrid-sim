"""Smoke tests for stylized CLOB."""
from __future__ import annotations

import pytest

from venues.clob import CLOB, EmptyBookError


def _count_resting_orders(clob: CLOB) -> int:
    n = 0
    for px in clob._bid_prices:
        n += len(clob._bids.get(px, ()))
    for px in clob._ask_prices:
        n += len(clob._asks.get(px, ()))
    return n


def test_smoke_clob_empty_limit_market_partial_cancel() -> None:
    clob = CLOB()
    st0 = clob.get_state()
    assert st0.mid_price is None
    assert st0.best_bid is None

    with pytest.raises(EmptyBookError):
        clob.estimate_impact("buy", 1.0)

    clob.submit_limit_order("1", "buy", 10.0, 99.0)
    clob.submit_limit_order("shallow", "sell", 1.0, 100.5)
    clob.submit_limit_order("2", "sell", 10.0, 101.0)
    st1 = clob.get_state()
    assert st1.mid_price is not None
    assert st1.best_bid == pytest.approx(99.0)
    assert st1.best_ask == pytest.approx(100.5)
    assert _count_resting_orders(clob) == 3

    mid_before = float(st1.mid_price)
    res_m = clob.submit_market_order("3", "buy", 1.0)
    assert res_m.filled_quantity == pytest.approx(1.0)
    st2 = clob.get_state()
    assert st2.mid_price is not None
    assert st2.mid_price > mid_before
    assert st2.best_ask == pytest.approx(101.0)

    res_p = clob.submit_limit_order("4", "buy", 15.0, 101.0)
    assert res_p.filled_quantity == pytest.approx(10.0)
    assert res_p.remaining_quantity == pytest.approx(5.0)
    assert res_p.order_id is not None
    assert _count_resting_orders(clob) >= 1

    oid = res_p.order_id
    assert clob.cancel_order("4", oid) is True
    found = False
    for px in clob._bid_prices:
        for ag, _q, o in clob._bids.get(px, ()):
            if o == oid and ag == "4":
                found = True
    assert not found
