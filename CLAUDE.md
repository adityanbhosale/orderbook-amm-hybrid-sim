# CLAUDE.md — orderbook-amm-hybrid-sim

Standing rules for this repo. Task-specific state lives in `handoff_fba_sim.md` (the source of truth); read it before any FBA / latency / markout work. This file is the *discipline*, not the plan.

## What this is

An event-driven agent-based simulator comparing market mechanisms (AMM / CLOB / hybrid, with a frequent-batch-auction venue mid-integration) against Bayesian trader populations. Outputs per-run metrics to a sweep summary. The research question is whether batch clearing reduces information-asymmetry extraction (the rent paid to faster actors) relative to continuous matching. Numbers from this sim are pitched and published, so correctness and honesty outrank speed.

## Non-negotiable workflow

- **Stop before committing.** Never `git commit` or `git push` on your own. End a unit of work by showing the diff and the test output, then stop. I decide when it commits.
- **Never fabricate a result to satisfy a gate.** If a smoke test can't run (environment, disk, missing dep) or a check fails, *report the blocker and stop* — do not invent passing output, do not skip the check and proceed. A correct "I'm blocked, here's why" is the desired outcome, not a failure.
- **Audit-first, phase-gated.** Every phase of a build ends on a smoke test that re-derives a *known* result (a published finding, a prior baseline). Don't move to the next phase until the current one re-derives ground truth. This is how this repo has always been built (see `build_log*.md`).
- **Recon before building new components.** Before adding a new agent class, venue, or anything that plugs into the core loop, do a read-only pass (Plan Mode) on how the existing pieces interact and report findings. Do not edit during recon. If something contradicts the notes you were given, STOP and report rather than guessing.
- **Follow the EXP-N convention and update `build_log*.md`** as work lands, matching the existing entry style.

## Locked modeling decisions — do not "improve"

These are settled. If a task seems to require changing one, STOP and flag it; do not silently revise.

- **Signal semantics (M1):** a delayed agent acts *later on the same signal draw*. The signal `value` is snapshotted at emission; `observation_delay` defers the combined observe-and-act. **Never re-draw or re-noise the signal at act time** — the latency edge is timing-of-belief-update only, not information staleness. Signal *quality* differentiation is separate and already modeled via per-class precision.
- **Role map:**
  - FAST = `TailAwareGaussianBeliefAgent`, `JointFactorFairValueAgent`
  - SLOW = `NaiveGaussianBeliefAgent`, `AggregatedEvidenceAgent`
  - NOISE = `EventDrivenNoiseAgent` — untouched; no `observation_delay`, acts via the Poisson `arrival_rate_per_unit` path.
- **Delays are config-driven and default to 0**, which must reproduce the pre-latency baseline byte-for-byte. Differentiated delays are an opt-in axis, never the default.
- **Maker/taker is class-independent**, decided in `route_log_space_trade → consider_clob_hybrid_trade → maker_taker_decision` by `|edge|/spread_log` vs `taker_threshold`. Don't special-case it per class.

## Known structural fact (don't re-discover it the hard way)

Latency differentiation is **inert until agents trade with each other**. Today informed/noise agents trade as takers against the deep static bootstrap ladder (agent `-1`); informed-as-maker fills = 0, so there is no extraction channel and reordering fast-vs-slow changes nothing. Creating an inter-agent fill channel (thinned book or a dedicated LP/market-maker agent) is the precondition for any latency/markout result. See `handoff_fba_sim.md` §5.4.

## Reproducibility & data safety

- **All-zero-delay config must reproduce the committed baseline byte-identical** (final mids, deployed, pending_cost, informed_pnl_total, convergence). Treat any drift as a regression to explain, not accept.
- **Determinism:** same-seed runs must be byte-identical across invocations. The event heap tie-break is `(timestamp, priority, insertion_order)`; flag if any change introduces nondeterminism.
- **Frozen datasets are content-hashed and `data/` is gitignored.** Never delete, move, or regenerate anything under `data/`, `results/`, or recordings without confirming it's regenerable-and-archived. A disk-space cleanup must not be the thing that destroys a frozen artifact.

## How to talk to me

Direct and unvarnished. Flag objective errors (wrong names, broken reproducibility, factual overclaims) plainly. You can flag a strategic concern twice; past a second flag, if I've decided, build it my way. I make the final call on sends and commits. Prefer concise structured prose over bullet-dumps.

## Key files (orientation, not exhaustive — confirm against the tree)

- `simulation/sweep.py` — run orchestration, `run_single_simulation`, population builders, mechanism literal/guard
- `metrics/rent.py` — `rent_and_pnl`, `_trade_mtm_pnl`, per-role PnL split (`pnl_by_role`)
- `metrics/convergence.py` — `build_mid_trajectory_from_trades`
- `agent_population.py` — signal fan-out, decision/review/noise event handlers
- `venues/fba.py` — frequent-batch-auction venue (mid-integration; not yet a runnable sweep mechanism)
- `trade_records.py`, `market_environment.py` — `TradeRecord`, fill recording / generalized drain
- `events.py` — event ordering and priorities
