"""Integration: AgentPopulation + four belief agents on 3 CPAMM markets."""
from __future__ import annotations

import math

import numpy as np
import pytest

from agents.informed_aggregated import AggregatedEvidenceAgent
from agents.informed_joint_factor import make_joint_factor_agent
from agents.informed_naive import NaiveGaussianBeliefAgent
from agents.informed_tail import TailAwareGaussianBeliefAgent
from environment import (
    AgentPopulation,
    EventPriority,
    MarginSpec,
    MarketEnvironment,
    Signal,
    Simulator,
    cross_weights_from_loadings,
)
from environment.information import InformationEnvironment
from venues.constant_product import ConstantProductAMM


def test_multi_agent_all_types_trade_and_budgets_respected() -> None:
    rng = np.random.default_rng(0)
    until_ts = 100
    sim = Simulator(rng=rng, time_resolution=1000)

    margin = MarginSpec(long_margin_fraction=1.0, short_margin_fraction=1.0)
    venues = {
        0: ConstantProductAMM(10_000.0, 1_000_000.0),
        1: ConstantProductAMM(10_000.0, 1_000_000.0),
        2: ConstantProductAMM(10_000.0, 1_000_000.0),
    }
    market_env = MarketEnvironment(venues, margin=margin)
    market_env.register(sim)

    log_mid0 = math.log(100.0)
    for m in range(3):
        assert market_env.mid_price(m) == pytest.approx(100.0)

    loadings_matrix = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [0.6, 0.6],
        ],
        dtype=float,
    )
    cw = cross_weights_from_loadings(
        loadings_matrix,
        primary_markets=(0, 1, 2),
        observed_markets=(0, 1, 2),
        min_weight=0.05,
    )

    budget = 100_000.0
    thresh = 0.002
    pri = 3.0
    trade_sz = 4.0
    review = 8
    safety = 1.05

    init_log = {0: log_mid0, 1: log_mid0, 2: log_mid0}

    naive = NaiveGaussianBeliefAgent(
        agent_id=1,
        budget=budget,
        market_ids=(0, 1, 2),
        initial_log_fair_mean=init_log.copy(),
        observation_delay=0,
        review_interval=review,
        prior_precision=pri,
        signal_precision_assumed=0.6,
        disagreement_threshold_log=thresh,
        trade_size=trade_sz,
        safety_margin=safety,
    )
    tail = TailAwareGaussianBeliefAgent(
        agent_id=2,
        budget=budget,
        market_ids=(0, 1, 2),
        base_log_levels=init_log.copy(),
        observation_delay=0,
        review_interval=review,
        prior_precision=1.5,
        disagreement_threshold_log=thresh,
        trade_size=trade_sz,
        safety_margin=safety,
    )
    agg = AggregatedEvidenceAgent(
        agent_id=3,
        budget=budget,
        market_ids=(0, 1, 2),
        observed_markets=(0, 1, 2),
        cross_weights=cw,
        initial_log_fair_mean=init_log.copy(),
        observation_delay=0,
        review_interval=review,
        prior_precision=2.0,
        signal_precision_assumed=0.9,
        disagreement_threshold_log=thresh,
        trade_size=trade_sz,
        safety_margin=safety,
    )
    joint = make_joint_factor_agent(
        agent_id=4,
        budget=budget,
        primary_markets=(0, 1, 2),
        observed_markets=(0, 1, 2),
        loadings_matrix=loadings_matrix,
        initial_mid_by_market={0: 100.0, 1: 100.0, 2: 100.0},
        observation_delay=0,
        review_interval=review,
        prior_precision_scale=1.0,
        signal_noise_inflation=1.0,
        disagreement_threshold_log=thresh,
        trade_size=trade_sz,
        safety_margin=safety,
    )

    population = AgentPopulation([naive, tail, agg, joint])
    population.register(sim, market_env, until_ts=until_ts)

    shocks = [
        0.045,
        -0.038,
        0.042,
        -0.04,
        0.035,
        -0.033,
        0.041,
        -0.036,
        0.039,
        -0.034,
    ]
    spec_ticks = [6, 11, 16, 21, 28, 35, 43, 51, 62, 74]
    signal_skew = {0: 0.0, 1: 0.0, 2: 0.0}
    markets_cycle = [0, 1, 2, 0, 1, 2, 0, 1, 2, 0]
    for t, mid, sh in zip(spec_ticks, markets_cycle, shocks):
        z = log_mid0 + sh
        signal_skew[mid] += sh
        sim.schedule_at(
            t,
            InformationEnvironment.SIGNAL_EVENT,
            payload=Signal(
                market_id=mid,
                value=float(z),
                is_tail=abs(sh) > 0.04,
                noise_std=0.025 if abs(sh) > 0.04 else 0.055,
            ),
            priority=EventPriority.SIGNAL,
        )

    sim.run_until(until_ts)
    population.sync_costs()

    for agent in population.agents:
        assert agent.deployed <= agent.budget + 1e-6
        assert agent.pending_cost == pytest.approx(0.0)

    by_agent: dict[int, int] = {a.agent_id: 0 for a in population.agents}
    for rec in market_env.trade_log:
        by_agent[rec.agent_id] = by_agent.get(rec.agent_id, 0) + 1

    for aid in (1, 2, 3, 4):
        assert by_agent[aid] >= 1, f"agent {aid} never traded"

    for m in range(3):
        mid_i = 100.0
        mid_f = market_env.mid_price(m)
        delta = mid_f - mid_i
        skew = signal_skew[m]
        if abs(skew) < 1e-9:
            continue
        assert delta * skew > 0, (
            f"market {m} mid moved opposite net signal skew (delta={delta}, skew={skew})"
        )
