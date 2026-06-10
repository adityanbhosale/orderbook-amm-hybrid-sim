"""FBA venue: deferred submits, uniform-price batch clears, drain wiring.

Covers (a)-(i) from the Entry-3 spec plus the MarketEnvironment routing of
clear-time fills (both legs) through the ``drain_maker_fills`` channel.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from environment import (
    EventPriority,
    MarginSpec,
    MarketEnvironment,
    Simulator,
    TradeIntent,
)
from venues.base import MakerFill
from venues.clob import EmptyBookError
from venues.fba import FBAVenue


def _clear_once(venue: FBAVenue) -> list[MakerFill]:
    """Tick exactly tau times (one full batch interval) and drain."""
    for _ in range(venue.tau_ticks):
        venue.tick()
    return venue.drain_maker_fills()


# --------------------------------------------------------------------- #
# (a) hand-computed clear: exact p* and exact matched quantities        #
# --------------------------------------------------------------------- #

def test_hand_computed_clear_exact() -> None:
    # Demand:  10 @ 101 (agent 1), 5 @ 100 (agent 2)   -> D(99)=15 D(100)=15 D(101)=10
    # Supply:   8 @ 99  (agent 3), 4 @ 100 (agent 4)   -> S(99)=8  S(100)=12 S(101)=12
    # vol: 8 @ 99, 12 @ 100, 10 @ 101  =>  unique max 12 at p* = 100.
    # At p*: dem=15 > sup=12 -> sells fill fully, buys pro-rata:
    #   agent 1: 12*10/15 = 8, agent 2: 12*5/15 = 4 (exact, no residue).
    v = FBAVenue(tau_ticks=1)
    v.submit_limit_order("1", "buy", 10.0, 101.0)
    v.submit_limit_order("2", "buy", 5.0, 100.0)
    v.submit_limit_order("3", "sell", 8.0, 99.0)
    v.submit_limit_order("4", "sell", 4.0, 100.0)

    fills = _clear_once(v)
    assert len(fills) == 4

    by_agent = {f.agent_id: f for f in fills}
    assert by_agent["1"].quantity == 8.0
    assert by_agent["2"].quantity == 4.0
    assert by_agent["3"].quantity == 8.0
    assert by_agent["4"].quantity == 4.0
    assert all(f.price == 100.0 for f in fills)
    assert by_agent["1"].side == "buy" and by_agent["3"].side == "sell"

    # rationed buys leave the unfilled remainder resting
    assert sum(o.qty_q for o in v._resting.values() if o.side == "buy") == 3 * 10**9


# --------------------------------------------------------------------- #
# (b) uniform price: every fill in a batch trades at exactly p*         #
# --------------------------------------------------------------------- #

def test_uniform_price_single_p_star_per_batch() -> None:
    v = FBAVenue(tau_ticks=1)
    v.submit_limit_order("1", "buy", 10.0, 101.0)
    v.submit_limit_order("2", "buy", 5.0, 100.0)
    v.submit_limit_order("3", "sell", 8.0, 99.0)
    v.submit_limit_order("4", "sell", 4.0, 100.0)
    fills = _clear_once(v)
    assert len({f.price for f in fills}) == 1
    assert fills[0].price == 100.0


def test_uniform_price_midpoint_tie_break() -> None:
    # Flat max-volume interval [98, 102] -> clear at the MIDPOINT 100.
    v = FBAVenue(tau_ticks=1)
    v.submit_limit_order("1", "buy", 5.0, 102.0)
    v.submit_limit_order("2", "sell", 5.0, 98.0)
    fills = _clear_once(v)
    assert len(fills) == 2
    assert all(f.price == 100.0 for f in fills)
    assert all(f.quantity == 5.0 for f in fills)


# --------------------------------------------------------------------- #
# (c) conservation: buy qty == sell qty within each clear               #
# --------------------------------------------------------------------- #

def test_conservation_per_clear_with_rationing() -> None:
    v = FBAVenue(tau_ticks=1)
    # Clear 1: market buys 3.3 + 1.4 + 2.6 = 7.3 rationed against 5 @ 100.
    v.submit_limit_order("1", "sell", 5.0, 100.0)
    v.submit_market_order("2", "buy", 3.3)
    v.submit_market_order("3", "buy", 1.4)
    v.submit_market_order("4", "buy", 2.6)
    fills1 = _clear_once(v)
    buys1 = sum(f.quantity for f in fills1 if f.side == "buy")
    sells1 = sum(f.quantity for f in fills1 if f.side == "sell")
    assert buys1 == pytest.approx(sells1, abs=1e-9)
    assert sells1 == pytest.approx(5.0, abs=1e-9)

    # Clear 2: fresh crossing limits with awkward quantities.
    v.submit_limit_order("5", "buy", 2.0001, 103.0)
    v.submit_limit_order("6", "sell", 0.7, 101.0)
    v.submit_limit_order("7", "sell", 0.9003, 102.0)
    fills2 = _clear_once(v)
    buys2 = sum(f.quantity for f in fills2 if f.side == "buy")
    sells2 = sum(f.quantity for f in fills2 if f.side == "sell")
    assert buys2 == pytest.approx(sells2, abs=1e-9)
    assert buys2 > 0.0


# --------------------------------------------------------------------- #
# (d) pending semantics: no fill before the clear tick                  #
# --------------------------------------------------------------------- #

def test_pending_order_results_and_fill_timing() -> None:
    v = FBAVenue(tau_ticks=3)

    res_l = v.submit_limit_order("1", "sell", 5.0, 100.0)
    assert res_l.filled_quantity == 0.0
    assert res_l.remaining_quantity == 5.0
    assert res_l.avg_fill_price == 0.0
    assert res_l.order_id is not None

    res_m = v.submit_market_order("2", "buy", 5.0)
    assert res_m.filled_quantity == 0.0
    assert res_m.remaining_quantity == 5.0
    assert res_m.order_id is not None

    # counter 1, 2: no clear, no fills
    v.tick()
    assert v.drain_maker_fills() == []
    v.tick()
    assert v.drain_maker_fills() == []

    # counter 3: counter % tau == 0 -> clear fires, both legs appear
    v.tick()
    fills = v.drain_maker_fills()
    assert len(fills) == 2
    assert all(f.price == 100.0 for f in fills)
    assert {f.order_id for f in fills} == {res_l.order_id, res_m.order_id}


# --------------------------------------------------------------------- #
# (e) resting persistence and partial remainders                        #
# --------------------------------------------------------------------- #

def test_resting_limit_persists_across_clears_and_partial_remainder() -> None:
    v = FBAVenue(tau_ticks=1)
    res = v.submit_limit_order("1", "sell", 10.0, 100.0)

    # clear with no opposite interest: nothing trades, the limit survives
    assert _clear_once(v) == []
    assert v.get_state().best_ask == 100.0

    # partial fill: 4 of 10; remainder 6 stays resting
    v.submit_market_order("2", "buy", 4.0)
    fills = _clear_once(v)
    assert sum(f.quantity for f in fills if f.side == "sell") == 4.0
    assert v._resting[res.order_id].qty_q == 6 * 10**9
    assert v.get_state().best_ask == 100.0

    # the remainder fills at a later clear
    v.submit_market_order("3", "buy", 6.0)
    fills = _clear_once(v)
    assert sum(f.quantity for f in fills if f.side == "sell") == 6.0
    assert res.order_id not in v._resting
    assert v.get_state().best_ask is None


# --------------------------------------------------------------------- #
# (f) market orders expire if they cannot fill at the next clear        #
# --------------------------------------------------------------------- #

def test_market_order_expires_without_opposite_interest() -> None:
    v = FBAVenue(tau_ticks=1)
    v.submit_market_order("9", "buy", 3.0)
    assert _clear_once(v) == []          # records nothing

    # the expired market order must not haunt later clears
    v.submit_limit_order("1", "sell", 5.0, 100.0)
    assert _clear_once(v) == []          # still no buy interest

    v.submit_market_order("2", "buy", 2.0)
    fills = _clear_once(v)
    assert sum(f.quantity for f in fills if f.side == "buy") == 2.0
    assert all(f.agent_id != "9" for f in fills)


# --------------------------------------------------------------------- #
# (g) determinism: same order stream -> identical clears                #
# --------------------------------------------------------------------- #

def _run_seeded_stream(seed: int) -> list[MakerFill]:
    rng = np.random.default_rng(seed)
    v = FBAVenue(tau_ticks=2)
    fills: list[MakerFill] = []
    for _ in range(40):
        agent = str(rng.integers(1, 6))
        side = "buy" if rng.random() < 0.5 else "sell"
        qty = float(np.round(rng.uniform(0.1, 5.0), 3))
        if rng.random() < 0.6:
            px = float(np.round(rng.uniform(95.0, 105.0), 2))
            v.submit_limit_order(agent, side, qty, px)
        else:
            v.submit_market_order(agent, side, qty)
        if rng.random() < 0.4:
            v.tick()
            fills.extend(v.drain_maker_fills())
    for _ in range(2 * v.tau_ticks):
        v.tick()
    fills.extend(v.drain_maker_fills())
    return fills


def test_deterministic_clears_for_identical_streams() -> None:
    fills_a = _run_seeded_stream(123)
    fills_b = _run_seeded_stream(123)
    assert fills_a == fills_b            # MakerFill is frozen -> field equality
    assert len(fills_a) > 0              # the stream actually produced clears


# --------------------------------------------------------------------- #
# (h) pre/post-clear mids stamped on fills, distinct from submit bracket#
# --------------------------------------------------------------------- #

def test_clear_time_mids_differ_from_submit_time_bracket() -> None:
    v = FBAVenue(tau_ticks=2)
    # backstop book: bid 99 / ask 105 -> mid 102
    v.submit_limit_order("90", "buy", 5.0, 99.0)
    v.submit_limit_order("91", "sell", 5.0, 105.0)
    submit_mid_before = v.get_state().mid_price
    assert submit_mid_before == 102.0

    # crossing pair arrives before the clear
    v.submit_limit_order("1", "buy", 6.0, 102.0)
    mid_after_buy_submit = v.get_state().mid_price   # (102+105)/2 = 103.5
    v.submit_limit_order("2", "sell", 6.0, 100.0)

    # clear: candidates 99/100/102/105, max vol 6 on [100,102] -> p* = 101
    fills = _clear_once(v)
    assert len(fills) == 2
    assert all(f.price == 101.0 for f in fills)

    # pre-clear mid = crossed book (102+100)/2 = 101; post-clear = (99+105)/2 = 102
    assert all(f.mid_before == 101.0 for f in fills)
    assert all(f.mid_after == 102.0 for f in fills)

    # neither stamped mid equals the submit-time bracket -> markout is not a no-op
    assert fills[0].mid_before != submit_mid_before
    assert fills[0].mid_before != mid_after_buy_submit


# --------------------------------------------------------------------- #
# (i) tau_ticks = 1: clears every tick, well-formed records             #
# --------------------------------------------------------------------- #

def test_tau_one_batches_every_tick_with_well_formed_records() -> None:
    v = FBAVenue(tau_ticks=1)
    res_l = v.submit_limit_order("1", "sell", 2.0, 100.0)
    res_m = v.submit_market_order("2", "buy", 2.0)

    v.tick()                              # first tick already clears
    fills = v.drain_maker_fills()
    assert len(fills) == 2

    maker = next(f for f in fills if f.liquidity == "maker")
    taker = next(f for f in fills if f.liquidity == "taker")
    assert maker.order_id == res_l.order_id
    assert taker.order_id == res_m.order_id
    assert maker.side == "sell" and taker.side == "buy"
    assert maker.price == taker.price == 100.0
    assert maker.quantity == taker.quantity == 2.0
    assert maker.fees_paid == 0.0 and taker.fees_paid == 0.0
    assert v.drain_maker_fills() == []    # drained exactly once


# --------------------------------------------------------------------- #
# estimate_impact: hypothetical clearing price                          #
# --------------------------------------------------------------------- #

def test_estimate_impact_returns_hypothetical_p_star() -> None:
    v = FBAVenue(tau_ticks=1)
    v.submit_limit_order("1", "sell", 5.0, 100.0)
    assert v.estimate_impact("buy", 3.0) == 100.0
    assert v.estimate_impact("buy", 10.0) == math.inf   # cannot fully fill
    with pytest.raises(EmptyBookError):
        v.estimate_impact("sell", 1.0)                  # no bids at all


# --------------------------------------------------------------------- #
# environment wiring: drain right after the clear tick                  #
# --------------------------------------------------------------------- #

def test_environment_drain_routes_both_legs_with_clear_timestamp() -> None:
    venue = FBAVenue(tau_ticks=1)
    sim = Simulator(rng=np.random.default_rng(0), time_resolution=1000)
    env = MarketEnvironment({0: venue}, margin=MarginSpec(1.0, 1.0))
    env.register(sim)

    # mirrors the sweep's venue clock (priority -50), plus the drain
    def _pulse(sim_: Simulator, _event) -> None:
        for v in env.venues.values():
            v.tick()
        env.drain_venue_fills(sim_)

    sim.register_handler("venue_clock", _pulse)
    for t in range(0, 4):
        sim.schedule_at(t, "venue_clock", priority=-50)

    sim.schedule_at(
        1, MarketEnvironment.TRADE_EVENT,
        payload=TradeIntent(0, 10, "sell", 5.0, order_type="limit", limit_price=100.0),
        priority=EventPriority.TRADE,
    )
    sim.schedule_at(
        1, MarketEnvironment.TRADE_EVENT,
        payload=TradeIntent(0, 20, "buy", 5.0, order_type="market"),
        priority=EventPriority.TRADE,
    )
    sim.run_until(3)

    # submits at t=1 happen AFTER t=1's pulse -> they clear at the t=2 pulse
    assert len(env.trade_log) == 2
    maker = next(r for r in env.trade_log if r.liquidity == "maker")
    taker = next(r for r in env.trade_log if r.liquidity == "taker")

    assert maker.timestamp == taker.timestamp == 2     # clear tick, not submit tick
    assert maker.agent_id == 10 and taker.agent_id == 20
    assert maker.avg_fill_price == taker.avg_fill_price == 100.0
    assert maker.quantity == taker.quantity == 5.0
    # maker capital was committed at submit; taker (market) charged at clear
    assert maker.capital_committed == 0.0
    assert taker.capital_committed == pytest.approx(5.0 * 100.0)

    # cost_log: limit + market submit entries at t=1, taker clear entry at t=2
    assert [(c.timestamp, c.agent_id) for c in env.cost_log] == [(1, 10), (1, 20), (2, 20)]
    limit_cost, market_submit_cost, taker_clear_cost = env.cost_log
    assert limit_cost.capital_committed == pytest.approx(5.0 * 100.0)
    assert market_submit_cost.capital_committed == 0.0   # nothing filled at submit
    assert taker_clear_cost.capital_committed == pytest.approx(5.0 * 100.0)
