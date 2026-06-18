"""§5.4 LP / market-maker smoke gates — the inter-agent fill channel.

G2 channel exists: a CLOB ``lp_vs_informed`` run with FAST informed delay < LP
   delay produces (a) LP maker fills > 0 (informed-as-maker is no longer ~0 —
   someone is being picked off) and (b) NEGATIVE LP PnL (the LP is adversely
   selected). This is the result 4a could not produce.
G3 solvency: the LP does NOT go silent — net-inventory accounting lets it
   requote across the full run, so a material fraction of its fills land in the
   second half (the gross monotonic-deployed model would have exhausted it
   early).
G4 determinism: the lp_vs_informed run is byte-identical across two same-seed
   invocations.

(G1 — zero-delay diverse/clob reproduces the committed baseline byte-identical —
is verified out-of-band against the pre-change tree, since it asserts equality to
committed numbers this file cannot import. See the build report.)

Smoke scale only (until_ts=8000, single seed). NOT the grid. Per recon
UNCERTAINTY #2: informed-as-maker is ~0 economically, not identically zero; the
point here is the LP makes the maker channel MATERIAL and its PnL negative.
"""
from __future__ import annotations

from simulation.sweep import RoleDelayConfig, run_single_simulation

SEED = 0
SMOKE_TS = 8000
FAST_DELAY = 1
SLOW_DELAY = 6
LP_DELAY = 50  # staler than FAST (§5.2)


def _run() -> dict:
    return run_single_simulation(
        seed=SEED,
        mechanism="clob",
        mix="lp_vs_informed",
        capital_band="mid",
        signal_regime="low",
        until_ts=SMOKE_TS,
        observation_delays=RoleDelayConfig(fast=FAST_DELAY, slow=SLOW_DELAY),
        lp_observation_delay=LP_DELAY,
    )["summary"]


# G2 -------------------------------------------------------------------- #
def test_g2_channel_exists_lp_filled_and_bleeding() -> None:
    s = _run()
    print(
        f"\n[G2] n_lp_fills={s['n_lp_fills']}  pnl_lp={s['pnl_lp']:+.6f}  "
        f"fast={s['pnl_fast_informed']:+.6f}  slow={s['pnl_slow_informed']:+.6f}  "
        f"n_trades={s['n_trades']}"
    )
    # (a) the LP is actually getting filled -> inter-agent maker channel exists.
    assert s["n_lp_fills"] > 0, "LP never filled — quoting/priority bug, not a result"
    # (b) the LP is adversely selected -> negative markout/PnL.
    assert s["pnl_lp"] < 0.0, "LP PnL non-negative — not being picked off"


# G3 -------------------------------------------------------------------- #
def test_g3_lp_stays_solvent_does_not_go_silent() -> None:
    s = _run()
    print(
        f"\n[G3] n_lp_fills={s['n_lp_fills']}  "
        f"frac_second_half={s['lp_frac_fills_second_half']:.3f}"
    )
    # Net-inventory accounting -> the LP keeps quoting to the end; a meaningful
    # share of fills land in the second half (a gross-deployed LP would be
    # exhausted and silent by then).
    assert s["lp_frac_fills_second_half"] > 0.2, "LP went quiet — solvency/capital bug"


# G4 -------------------------------------------------------------------- #
def test_g4_lp_run_is_deterministic() -> None:
    a = _run()
    b = _run()
    for k in (
        "n_trades",
        "pnl_lp",
        "n_lp_fills",
        "lp_frac_fills_second_half",
        "pnl_fast_informed",
        "pnl_slow_informed",
        "informed_pnl_total",
    ):
        assert a[k] == b[k], f"nondeterministic lp_vs_informed run on {k}"
