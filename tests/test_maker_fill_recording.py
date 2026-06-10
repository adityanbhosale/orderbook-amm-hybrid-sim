"""Maker-side fills are recorded symmetrically with taker fills.

Covers the recording fix for: resting limit orders consumed by incoming
orders previously left no maker TradeRecord (only the taker side reached
``MarketEnvironment._on_trade``).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pytest

from environment import (
    AgentPopulation,
    EventPriority,
    MarginSpec,
    MarketEnvironment,
    Simulator,
    TradeIntent,
)
from venues.clob import CLOB
from venues.hybrid import HybridVenue, HybridLpConfig, LP_AGENT_ID


def _make_env() -> tuple[Simulator, MarketEnvironment, CLOB]:
    clob = CLOB()
    sim = Simulator(rng=np.random.default_rng(0), time_resolution=1000)
    env = MarketEnvironment({0: clob}, margin=MarginSpec(1.0, 1.0))
    env.register(sim)
    return sim, env, clob


def _send(sim: Simulator, at: int, intent: TradeIntent) -> None:
    sim.schedule_at(
        at, MarketEnvironment.TRADE_EVENT, payload=intent, priority=EventPriority.TRADE
    )


# (a) one resting limit fully consumed -> exactly TWO records, mirror image
def test_market_order_hitting_resting_limit_produces_maker_and_taker_records() -> None:
    sim, env, clob = _make_env()
    # two-sided book so mids exist
    clob.submit_limit_order("30", "buy", 5.0, 99.0)

    _send(sim, 1, TradeIntent(0, 10, "sell", 5.0, order_type="limit", limit_price=101.0))
    _send(sim, 2, TradeIntent(0, 20, "buy", 5.0, order_type="market"))
    sim.run_until(2)

    # the fully-resting limit at t=1 must NOT create a quantity=0 record
    assert all(r.quantity > 0 for r in env.trade_log)
    assert len(env.trade_log) == 2

    maker, taker = env.trade_log
    assert maker.liquidity == "maker"
    assert taker.liquidity == "taker"
    assert maker.agent_id == 10
    assert taker.agent_id == 20
    assert maker.side == "sell"
    assert taker.side == "buy"
    assert maker.quantity == pytest.approx(taker.quantity) == pytest.approx(5.0)
    assert maker.avg_fill_price == pytest.approx(101.0)
    assert taker.avg_fill_price == pytest.approx(101.0)
    assert maker.timestamp == taker.timestamp == 2
    # maker capital was committed at rest time; the fill itself charges nothing
    assert maker.capital_committed == 0.0
    assert taker.capital_committed == pytest.approx(5.0 * 101.0)


# (b) conservation: total recorded buy qty == total recorded sell qty
def test_recorded_buy_volume_equals_recorded_sell_volume() -> None:
    sim, env, clob = _make_env()
    clob.seed_initial_book(100.0, 5.0)  # bootstrap maker, agent "-1"

    _send(sim, 1, TradeIntent(0, 20, "buy", 7.0, order_type="market"))
    _send(sim, 2, TradeIntent(0, 21, "sell", 3.0, order_type="market"))
    _send(sim, 3, TradeIntent(0, 22, "buy", 4.0, order_type="limit", limit_price=100.3))
    _send(sim, 4, TradeIntent(0, 23, "sell", 2.0, order_type="limit", limit_price=99.0))
    _send(sim, 5, TradeIntent(0, 24, "buy", 1.0, order_type="limit", limit_price=50.0))
    sim.run_until(5)

    assert all(r.quantity > 0 for r in env.trade_log)
    buys = sum(r.quantity for r in env.trade_log if r.side == "buy")
    sells = sum(r.quantity for r in env.trade_log if r.side == "sell")
    assert buys == pytest.approx(sells)
    assert buys > 0
    # bootstrap maker fills are recorded under its integer id
    assert any(r.agent_id == -1 and r.liquidity == "maker" for r in env.trade_log)


# (c) partial fill: both sides record the partial; remainder stays correct
def test_partial_fill_recorded_both_sides_with_correct_remainder() -> None:
    sim, env, clob = _make_env()
    clob.submit_limit_order("30", "buy", 5.0, 99.0)

    _send(sim, 1, TradeIntent(0, 10, "sell", 10.0, order_type="limit", limit_price=101.0))
    _send(sim, 2, TradeIntent(0, 20, "buy", 4.0, order_type="market"))
    sim.run_until(2)

    assert len(env.trade_log) == 2
    maker, taker = env.trade_log
    assert maker.liquidity == "maker" and maker.agent_id == 10
    assert taker.liquidity == "taker" and taker.agent_id == 20
    assert maker.quantity == pytest.approx(4.0)
    assert taker.quantity == pytest.approx(4.0)

    # the resting order keeps the unfilled 6.0 on the book
    (ag, q, _oid), = clob._asks[101.0]
    assert ag == "10"
    assert q == pytest.approx(6.0)

    # venue-level remaining_quantity contracts unchanged
    clob2 = CLOB()
    clob2.submit_limit_order("10", "sell", 10.0, 101.0)
    res = clob2.submit_market_order("20", "buy", 4.0)
    assert res.filled_quantity == pytest.approx(4.0)
    assert res.remaining_quantity == pytest.approx(0.0)
    fills = clob2.drain_maker_fills()
    assert len(fills) == 1
    assert fills[0].agent_id == "10"
    assert fills[0].quantity == pytest.approx(4.0)
    assert fills[0].price == pytest.approx(101.0)

    res_l = clob2.submit_limit_order("21", "buy", 15.0, 101.0)
    assert res_l.filled_quantity == pytest.approx(6.0)
    assert res_l.remaining_quantity == pytest.approx(9.0)
    assert res_l.order_id is not None


# (d) maker capital charged exactly once (at rest time, not again at fill)
@dataclass
class _StubAgent:
    agent_id: int
    budget: float
    deployed: float = field(default=0.0)
    pending_cost: float = field(default=0.0)
    observation_delay: int = 0
    review_interval: int = 0
    arrival_rate_per_unit: float = 0.0

    def observes(self, market_id: int) -> bool:
        return False

    def decide(self, sim, signal, market_env):
        return None

    def review(self, sim, market_env):
        return []

    def fire_noise(self, sim, market_env):
        return None


def test_maker_capital_charged_once_and_pending_cost_cleared() -> None:
    sim, env, clob = _make_env()
    clob.submit_limit_order("30", "buy", 5.0, 99.0)

    maker = _StubAgent(agent_id=10, budget=1000.0)
    taker = _StubAgent(agent_id=20, budget=1000.0)
    population = AgentPopulation([maker, taker])
    population.register(sim, env, until_ts=10)

    cap_at_rest = 5.0 * 101.0
    maker.pending_cost = cap_at_rest  # as an agent would set when emitting the intent
    _send(sim, 1, TradeIntent(0, 10, "sell", 5.0, order_type="limit", limit_price=101.0))
    sim.run_until(1)
    population.sync_costs()
    assert maker.deployed == pytest.approx(cap_at_rest)
    assert maker.pending_cost == pytest.approx(0.0)

    _send(sim, 3, TradeIntent(0, 20, "buy", 5.0, order_type="market"))
    sim.run_until(3)
    population.sync_costs()

    # maker fill recorded, but capital NOT charged a second time
    assert any(r.agent_id == 10 and r.liquidity == "maker" for r in env.trade_log)
    assert maker.deployed == pytest.approx(cap_at_rest)
    assert maker.pending_cost == pytest.approx(0.0)
    assert taker.deployed == pytest.approx(5.0 * 101.0)

    # exactly one cost entry per intent
    assert len(env.cost_log) == 2
    maker_costs = [c for c in env.cost_log if c.agent_id == 10]
    assert len(maker_costs) == 1
    assert maker_costs[0].capital_committed == pytest.approx(cap_at_rest)


# hybrid: LP quotes consumed by a taker also produce maker records
def test_hybrid_lp_maker_fills_recorded() -> None:
    hybrid = HybridVenue(
        lp=HybridLpConfig(lp_spread_pct=0.005, lp_base_qty=5.0, lp_anchor_price=100.0)
    )
    sim = Simulator(rng=np.random.default_rng(0), time_resolution=1000)
    env = MarketEnvironment({0: hybrid}, margin=MarginSpec(1.0, 1.0))
    env.register(sim)
    hybrid.tick()  # post LP ladder

    _send(sim, 1, TradeIntent(0, 20, "buy", 3.0, order_type="market"))
    sim.run_until(1)

    makers = [r for r in env.trade_log if r.liquidity == "maker"]
    takers = [r for r in env.trade_log if r.liquidity == "taker"]
    assert len(takers) == 1
    assert takers[0].quantity == pytest.approx(3.0)
    assert makers, "LP maker fills must be recorded"
    assert all(r.agent_id == int(LP_AGENT_ID) for r in makers)
    assert sum(r.quantity for r in makers) == pytest.approx(3.0)
    assert all(r.side == "sell" for r in makers)
