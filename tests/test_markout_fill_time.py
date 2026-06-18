"""Markout-at-fill-time plumbing (measurement-only; truth stays frozen).

These tests pin the time-indexed marking introduced in the markout rework:
a fill is marked against ``fair_at(market_id, rec.timestamp)``, NOT against a
single terminal scalar. The ARTIFICIAL two-point fair series here is a test
fixture to exercise the plumbing — real runs still use a frozen (static) truth,
for which fill-time marking is byte-identical to terminal marking.
"""
from __future__ import annotations

from environment.trade_records import TradeRecord
from metrics.rent import frozen_fair_value, pnl_by_role


def _fill(ts: int, side: str, price: float, qty: float = 1.0) -> TradeRecord:
    return TradeRecord(
        timestamp=ts,
        market_id=0,
        agent_id=1,
        side=side,
        quantity=qty,
        avg_fill_price=price,
        fees_paid=0.0,
        capital_committed=0.0,
        mid_price_before=price,
        mid_price_after=price,
    )


def test_fill_marks_against_fair_at_its_own_timestamp() -> None:
    """Artificial series: fair=100 before t=4000, 110 from t=4000 on."""
    roles = {1: "x"}
    fair_at = lambda m, t: 100.0 if t < 4000 else 110.0  # noqa: E731

    early = _fill(0, "buy", 100.0)      # marked @100 -> 1*(100-100) = 0
    late = _fill(4000, "buy", 100.0)    # marked @110 -> 1*(110-100) = +10

    out = pnl_by_role([early, late], role_by_agent_id=roles, fair_at=fair_at)

    # Time-indexed: 0 + 10 = 10. If it (wrongly) marked BOTH at terminal 110 it
    # would be 20; if it marked both at the t=0 value 100 it would be 0.
    assert out["x"] == 10.0

    # And each fill individually resolves at its own ts.
    assert pnl_by_role([early], role_by_agent_id=roles, fair_at=fair_at)["x"] == 0.0
    assert pnl_by_role([late], role_by_agent_id=roles, fair_at=fair_at)["x"] == 10.0


def test_frozen_accessor_is_a_noop_vs_scalar_map() -> None:
    """A static-truth accessor marks every ts at the same scalar."""
    roles = {1: "x"}
    fills = [_fill(0, "buy", 100.0), _fill(4000, "buy", 100.0)]

    via_scalar = pnl_by_role(
        fills, role_by_agent_id=roles, fair_prices_by_market={0: 105.0}
    )
    via_frozen = pnl_by_role(
        fills, role_by_agent_id=roles, fair_at=frozen_fair_value({0: 105.0})
    )
    # Both mark every fill at 105 regardless of ts: 2 * (105 - 100) = 10.
    assert via_scalar["x"] == via_frozen["x"] == 10.0
