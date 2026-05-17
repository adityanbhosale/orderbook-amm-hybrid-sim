"""Gate: one market, one CPAMM, naive agent, one signal → posterior + sized trade."""
from __future__ import annotations

import math

import numpy as np
import pytest

from agents.informed_naive import NaiveGaussianBeliefAgent
from environment import (
    EventPriority,
    MarginSpec,
    MarketEnvironment,
    Signal,
    Simulator,
    committed_capital,
    downsize_quantity_for_capital,
)
from venues.base import OrderResult
from venues.constant_product import ConstantProductAMM


class SpyAMM(ConstantProductAMM):
    """Records ``submit_market_order`` for assertions."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.submit_log: list[tuple[str, str, float]] = []

    def submit_market_order(
        self, agent_id: str, side: str, quantity: float
    ) -> OrderResult:
        self.submit_log.append((agent_id, side, quantity))
        return super().submit_market_order(agent_id, side, quantity)


def test_smoke_naive_agent_signal_trade_and_posterior() -> None:
    """Posterior conjugate update, trade when |μ−log mid|>δ, impact-bounded size."""
    rng = np.random.default_rng(42)
    sim = Simulator(rng=rng, time_resolution=1000)

    spy = SpyAMM(2000.0, 200000.0)
    margin = MarginSpec(long_margin_fraction=1.0, short_margin_fraction=1.0)
    market_env = MarketEnvironment({0: spy}, margin=margin)

    mid = spy.get_state().mid_price
    log_mid = math.log(mid)
    assert mid == pytest.approx(100.0)

    tau0 = 5.0
    tau_assumed = 1.0
    z_signal = 4.65
    mu_expected = (tau0 * log_mid + tau_assumed * z_signal) / (tau0 + tau_assumed)
    assert abs(mu_expected - log_mid) > 0.005

    agent = NaiveGaussianBeliefAgent(
        agent_id=7,
        budget=400.0,
        market_ids=(0,),
        initial_log_fair_mean={0: log_mid},
        prior_precision=tau0,
        signal_precision_assumed=tau_assumed,
        disagreement_threshold_log=0.005,
        trade_size=50.0,
        safety_margin=1.2,
        confidence_weighted=False,
    )

    sig = Signal(
        market_id=0,
        value=float(z_signal),
        is_tail=False,
        noise_std=0.5,
    )

    captured: dict[str, object] = {}

    def on_signal(s: Simulator, event) -> None:
        payload = event.payload
        assert isinstance(payload, Signal)
        intent = agent.decide(s, payload, market_env)
        captured["intent"] = intent
        if intent is not None:
            exp_q = downsize_quantity_for_capital(
                spy,
                intent.side,
                agent.trade_size,
                agent.budget,
                margin,
                safety_margin=agent.safety_margin,
                min_quantity=agent.min_trade_quantity,
            )
            captured["expected_qty"] = exp_q
            cap_pre = committed_capital(
                spy,
                intent.side,
                intent.quantity,
                margin,
                safety_margin=agent.safety_margin,
            )
            captured["cap_pre_execute"] = cap_pre
            market_env.execute_market_order(
                s,
                intent.market_id,
                intent.agent_id,
                intent.side,
                intent.quantity,
            )

    sim.register_handler("signal", on_signal)
    sim.schedule_at(10, "signal", payload=sig, priority=EventPriority.SIGNAL)
    sim.run_until(100)

    mu_post, tau_post = agent.posterior(0)
    assert tau_post == pytest.approx(tau0 + tau_assumed)
    assert mu_post == pytest.approx(mu_expected)

    intent = captured.get("intent")
    assert intent is not None
    assert abs(mu_post - log_mid) > agent.disagreement_threshold_log
    assert intent.side == "buy"
    assert intent.market_id == 0
    assert intent.quantity > 0

    assert len(spy.submit_log) == 1
    aid, side, qty = spy.submit_log[0]
    assert aid == str(agent.agent_id)
    assert side == "buy"
    assert qty == pytest.approx(intent.quantity)

    expected_qty = captured["expected_qty"]
    assert isinstance(expected_qty, float)
    assert intent.quantity == pytest.approx(expected_qty)

    cap_pre = captured["cap_pre_execute"]
    assert isinstance(cap_pre, float)
    assert cap_pre <= agent.budget + 1e-6

    assert len(market_env.trade_log) == 1
