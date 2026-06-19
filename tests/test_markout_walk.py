"""Markout re-pointed to the moving-truth walk path (fair-at-fill-time).

Reuses the FairValueAt machinery from the markout rework: when walk_var>0 the
accessor returns the true fair PRICE at each fill's own tick (exp of the walk
path), so per-fill markout reflects fair-at-fill-time rather than the t=0 /
terminal truth. The marking choice MATTERS under a walk — it separates adverse
selection (mispricing vs fair-at-fill) from inventory drift (truth moving while a
position is held), which the t=0 marking conflates.

These are constructed-fill demonstrations on the REAL path + REAL accessors (not
a full LP run — that experiment is the next step).
"""
from __future__ import annotations

import math

import numpy as np

from environment.information import InformationEnvironment
from environment.trade_records import TradeRecord
from metrics.rent import frozen_fair_value, pnl_by_role
from simulation.sweep import _default_information_config, _walk_path_fair_value

UNTIL = 8000


def _walk_env(walk_var: float = 1e-4):
    env = InformationEnvironment(
        _default_information_config(
            signal_noise_std=0.012, tail_noise_std=0.005, walk_var=walk_var
        ),
        np.random.default_rng(123),
    )
    # mimic schedule_signals materializing the path at t=0
    env._log_fv_path = env._materialize_factor_walk_path(
        UNTIL, np.random.default_rng(999)
    )
    return env


def _fill(t: int, side: str, price: float) -> TradeRecord:
    return TradeRecord(
        timestamp=t, market_id=0, agent_id=1, side=side, quantity=1.0,
        avg_fill_price=price, fees_paid=0.0, capital_committed=0.0,
        mid_price_before=price, mid_price_after=price,
    )


# G-MARK-PATH ----------------------------------------------------------- #
def test_gmark_path_marks_against_path_at_fill_tick() -> None:
    env = _walk_env()
    fair_at = _walk_path_fair_value(env, UNTIL)
    t0_price = math.exp(env.world.truths[0].log_fair_value)

    # a late fill is marked against the PATH price at its own tick
    assert fair_at(0, 4000) == math.exp(env._log_fv_path[0, 4000])
    # at t=0 the path equals the static t=0 fair
    assert abs(fair_at(0, 0) - t0_price) < 1e-9
    # and a late fill's fair DIFFERS from the t=0 fair (truth moved)
    assert abs(fair_at(0, 4000) - t0_price) > 1e-3
    # beyond the materialized horizon is clamped
    assert fair_at(0, UNTIL + 500) == fair_at(0, UNTIL)


# G-INV-vs-AS ----------------------------------------------------------- #
def test_ginv_vs_as_marking_choice_changes_markout() -> None:
    env = _walk_env()
    path = env._log_fv_path
    fair_at = _walk_path_fair_value(env, UNTIL)
    fair_t0 = {0: math.exp(env.world.truths[0].log_fair_value)}
    roles = {1: "trader"}

    # Fills that are FAIR at their own tick: price == fair-at-fill. These carry
    # NO adverse selection by construction.
    fills = [_fill(t, "buy", math.exp(path[0, t])) for t in (2000, 4000, 6000)]

    pnl_new = pnl_by_role(fills, role_by_agent_id=roles, fair_at=fair_at)["trader"]
    pnl_old = pnl_by_role(
        fills, role_by_agent_id=roles, fair_prices_by_market=fair_t0
    )["trader"]

    print(
        f"\n[G-INV-vs-AS] fair-at-fill markout (new) = {pnl_new:+.6f}   "
        f"t=0 markout (old) = {pnl_old:+.6f}   "
        f"inventory-drift component = {pnl_old - pnl_new:+.6f}"
    )
    # New marking: fair trades -> ~0 markout (no adverse selection). Correct.
    assert abs(pnl_new) < 1e-9
    # Old (t=0) marking: materially nonzero -> this is PURE inventory drift
    # (truth moved t0->t_fill), which the t=0 marking wrongly attributes to the
    # trader. Proves the marking choice changes the number, hence why
    # fair-at-fill is required before the LP bleed experiment.
    assert abs(pnl_old) > 1e-2
    assert abs(pnl_old - pnl_new) > 1e-2
