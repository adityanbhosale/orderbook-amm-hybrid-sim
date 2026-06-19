"""FBA wired as a runnable sweep mechanism (handoff §5.6).

The FBA venue was unit-tested standalone but not runnable in a sweep: the
mechanism literal excluded it, there was no builder, and clear-time fills were
never drained. These tests pin the integration:
- FBA runs end-to-end and its clear-time fills reach trade_log;
- FBA is deterministic;
- FROZEN-world consistency with CLOB: with no moving truth there is no latency
  extraction, so the LP just earns its spread under BOTH mechanisms — they land
  in the same ballpark. (NB: this is the right internal-validity check; under a
  walk, batch-at-τ=1 is NOT ≈ CLOB — uniform-price simultaneous clearing already
  collapses the within-tick speed race. That divergence is the result the τ-curve
  measures, not a wiring error.)

Smoke scale (single seed, ts=8000, lp_vs_informed). NOT the τ-curve.
"""
from __future__ import annotations

from simulation.sweep import RoleDelayConfig, run_single_simulation

DELAYS = RoleDelayConfig(fast=1, slow=6)


def _run(mech: str, *, tau: int = 1, walk_var: float = 1e-6, seed: int = 0):
    return run_single_simulation(
        seed=seed, mechanism=mech, mix="lp_vs_informed", capital_band="mid",
        signal_regime="low", until_ts=8000, observation_delays=DELAYS,
        lp_observation_delay=50, walk_var=walk_var, tau_ticks=tau,
    )["summary"]


# G-FBA-RUNS ------------------------------------------------------------ #
def test_fba_runs_and_tapes_clear_time_fills() -> None:
    s = _run("fba", tau=10)
    assert s["n_trades"] > 0, "FBA produced no trade records — drain not wired"
    assert s["n_lp_fills"] > 0, "LP never filled on FBA — clear/quoting mismatch"


# G-DET ----------------------------------------------------------------- #
def test_fba_deterministic() -> None:
    a = _run("fba", tau=10)
    b = _run("fba", tau=10)
    for k in ("pnl_lp", "pnl_fast_informed", "n_lp_fills", "n_trades",
              "lp_frac_fills_second_half"):
        assert a[k] == b[k], f"FBA nondeterministic on {k}"


# G-CONSISTENCY (frozen world) ------------------------------------------ #
def test_fba_frozen_consistency_with_clob() -> None:
    """No walk => no extraction => LP ~earns spread under both mechanisms."""
    clob = _run("clob", walk_var=0.0)
    fba = _run("fba", tau=1, walk_var=0.0)
    # Both: LP modestly profits (earns the spread off noise flow), same ballpark.
    assert clob["pnl_lp"] > 0.0 and fba["pnl_lp"] > 0.0
    assert abs(fba["pnl_lp"] - clob["pnl_lp"]) < 1.5, (
        f"frozen LP PnL diverges: clob={clob['pnl_lp']}, fba={fba['pnl_lp']}"
    )
    assert fba["n_lp_fills"] > 0
