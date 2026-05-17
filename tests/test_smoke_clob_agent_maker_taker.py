"""Naive-style maker vs taker intents on a seeded CLOB."""
from __future__ import annotations

import math

from agents.belief_utils import consider_clob_hybrid_trade
from environment import MarginSpec, MarketEnvironment
from venues.clob import CLOB


def test_smoke_clob_maker_taker_intents() -> None:
    clob = CLOB()
    clob.submit_limit_order("seed_a", "buy", 100.0, 99.5)
    clob.submit_limit_order("seed_b", "sell", 100.0, 100.5)

    env = MarketEnvironment({0: clob}, margin=MarginSpec(1.0, 1.0))

    weak, _ = consider_clob_hybrid_trade(
        agent_id=1,
        market_id=0,
        posterior_mean_log=math.log(100.0) + 0.01,
        posterior_precision=5.0,
        market_env=env,
        disagreement_threshold_log=0.001,
        target_quantity=5.0,
        safety_margin=1.05,
        available=10_000.0,
        confidence_weighted=False,
        prior_precision_for_sizing=5.0,
        taker_threshold=10.0,
    )
    assert weak is not None
    assert weak.order_type == "limit"

    strong, _ = consider_clob_hybrid_trade(
        agent_id=1,
        market_id=0,
        posterior_mean_log=math.log(100.0) + 0.15,
        posterior_precision=5.0,
        market_env=env,
        disagreement_threshold_log=0.001,
        target_quantity=5.0,
        safety_margin=1.05,
        available=10_000.0,
        confidence_weighted=False,
        prior_precision_for_sizing=5.0,
        taker_threshold=0.5,
    )
    assert strong is not None
    assert strong.order_type == "market"
