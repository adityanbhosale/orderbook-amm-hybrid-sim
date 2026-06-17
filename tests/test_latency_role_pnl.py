"""Role-differentiated observation_delay + per-role PnL split (§5.2 precondition).

G1 reproducibility: default (all-zero) delay config reproduces the pre-latency
   baseline — identical to an explicit zero config and deterministic.
G2 extraction signature: fast<slow delays on a CLOB run shift PnL toward the
   fast role and away from the slow role vs the all-zero run on the same seed.
G3 determinism: a differentiated-delay run is byte-identical across two
   invocations with the same seed (1c tie-break introduced no nondeterminism).

Smoke scale only (until_ts=8000, single seed, diverse mix). NOT the grid.
"""
from __future__ import annotations

import pytest

from simulation.sweep import RoleDelayConfig, run_single_simulation

SEED = 0
SMOKE_TS = 8000
# Placeholder delay pair: config-driven, not load-bearing yet.
FAST_DELAY = 1
SLOW_DELAY = 6

# Summary keys that capture execution outcome (for reproducibility comparison).
_BASELINE_KEYS = (
    "n_trades",
    "normalized_rmse_log",
    "max_relative_price_error",
    "convergence_tick",
    "informed_pnl_total",
    "noise_pnl",
    "noise_loss",
    "lp_rent_total",
    "rent_efficiency_stable",
    "frac_informed_exhausted_before_convergence",
)


def _run(delays: RoleDelayConfig | None):
    return run_single_simulation(
        seed=SEED,
        mechanism="clob",
        mix="diverse",
        capital_band="mid",
        signal_regime="low",
        until_ts=SMOKE_TS,
        observation_delays=delays,
    )["summary"]


# G1 -------------------------------------------------------------------- #
def test_g1_default_reproduces_zero_config_baseline() -> None:
    default = _run(None)
    explicit_zero = _run(RoleDelayConfig(fast=0, slow=0))

    # default path == explicit all-zero == pre-latency behavior (delay still 0)
    for k in _BASELINE_KEYS:
        assert default[k] == explicit_zero[k], f"mismatch on {k}"
    assert default["delay_fast"] == 0 and default["delay_slow"] == 0

    # determinism of the baseline path itself
    again = _run(None)
    for k in _BASELINE_KEYS:
        assert default[k] == again[k], f"nondeterministic baseline on {k}"


# G2 -------------------------------------------------------------------- #
# FINDING (smoke, seed 0, clob/diverse/mid/low, ts=8000): differentiated
# delays produce ZERO change. Every maker fill is the bootstrap ladder
# (agent -1); informed agents act only as takers, so no informed resting
# quote is ever picked off and reordering fast-vs-slow takers against a deep
# static book changes no fills. Confirmed NOT a wiring bug: delays propagate
# (slow=large drops n_trades 60->40 and zeroes slow PnL), yet fast PnL is
# invariant to slow-role presence. Extraction needs a fillable-quote
# mechanism first (the §5.4 quoting role). The positive extraction signature
# is therefore xfailed until that lands.
def test_g2_no_extraction_without_fillable_quotes() -> None:
    """Document the current (no-extraction) reality and its root cause."""
    base = _run(RoleDelayConfig(fast=0, slow=0))
    diff = _run(RoleDelayConfig(fast=FAST_DELAY, slow=SLOW_DELAY))
    slow_gone = _run(RoleDelayConfig(fast=2, slow=10_000))  # slow effectively absent

    print(
        "\n[G2] all-zero:      "
        f"fast={base['pnl_fast_informed']:+.6f}  slow={base['pnl_slow_informed']:+.6f}  "
        f"informed_total={base['informed_pnl_total']:+.6f}  n_trades={base['n_trades']}"
    )
    print(
        f"[G2] fast={FAST_DELAY}/slow={SLOW_DELAY}: "
        f"fast={diff['pnl_fast_informed']:+.6f}  slow={diff['pnl_slow_informed']:+.6f}  "
        f"informed_total={diff['informed_pnl_total']:+.6f}  n_trades={diff['n_trades']}"
    )
    print(
        "[G2] slow~absent:   "
        f"fast={slow_gone['pnl_fast_informed']:+.6f}  slow={slow_gone['pnl_slow_informed']:+.6f}  "
        f"n_trades={slow_gone['n_trades']}"
    )

    # Small differentiated delays change nothing (delta == 0 exactly).
    assert diff["pnl_fast_informed"] == base["pnl_fast_informed"]
    assert diff["pnl_slow_informed"] == base["pnl_slow_informed"]
    # Root cause: fast PnL does not depend on the slow role at all -> no
    # extraction channel (fast agents are not filling slow resting quotes).
    assert slow_gone["pnl_fast_informed"] == base["pnl_fast_informed"]
    # ...while delays ARE applied (removing the slow role drops trade count).
    assert slow_gone["n_trades"] < base["n_trades"]


@pytest.mark.xfail(
    reason="No fillable resting quotes yet: all maker fills are the bootstrap "
    "ladder; informed limits are never picked off. Needs the §5.4 quoting "
    "role before a latency edge can extract. Remove when that lands.",
    strict=True,
)
def test_g2_extraction_signature_fast_gains_slow_loses() -> None:
    base = _run(RoleDelayConfig(fast=0, slow=0))
    diff = _run(RoleDelayConfig(fast=FAST_DELAY, slow=SLOW_DELAY))
    assert diff["pnl_fast_informed"] > base["pnl_fast_informed"]
    assert diff["pnl_slow_informed"] < base["pnl_slow_informed"]


# G3 -------------------------------------------------------------------- #
def test_g3_differentiated_run_is_deterministic() -> None:
    a = _run(RoleDelayConfig(fast=FAST_DELAY, slow=SLOW_DELAY))
    b = _run(RoleDelayConfig(fast=FAST_DELAY, slow=SLOW_DELAY))
    for k in (*_BASELINE_KEYS, "pnl_fast_informed", "pnl_slow_informed", "pnl_noise_role"):
        assert a[k] == b[k], f"nondeterministic differentiated run on {k}"
