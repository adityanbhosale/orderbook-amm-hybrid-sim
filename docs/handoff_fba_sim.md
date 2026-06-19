# Handoff — FBA Venue Simulation Build (`orderbook-amm-hybrid-sim`)

> Paste this into a new chat to resume. The task is to finish the **frequent-batch-auction (FBA) venue integration** in my agent-based market simulator: wire the latency/information structure that makes batching *matter*, add markout metrics, run the τ-sweep, and produce the headline result. Everything below is the context you need; the engineering state sections (4–5) are the load-bearing part. Sections 1–3 are the "why," section 7 is the strategic frame, section 8 is how I want you to work.

---

## STATUS (2026-06-19) — τ-curve DONE; headline result produced

The full §5 build landed (each phase gated, committed): LP fill channel
(`1401c97`), time-indexed markout (`9b2b8d5`), time-aware Kalman belief
scalar+matrix (`b113966`, `ad8a327`), moving-truth common-factor walk
(`6c17346`), markout re-pointed to fair-at-fill on the path (`57dab02`), FBA
wired as a runnable sweep mechanism (`cbcccbc`). Full detail + the result table
are in `build_log_fba.md` Entry 4.

**Headline (paired-by-seed, N=25, `walk_var=1e-6`, `q=walk_var`,
`lp_vs_informed` mid/low, FAST delay 1 < LP delay 50):** switching continuous
CLOB → FBA uniform-price batch clearing reduces the LP's latency-driven bleed
from `pnl_lp ≈ −92` to `≈ −5` — **paired reduction +86.6, SEM 6.7, ≈13σ, 95% of
seeds**. Genuine per-fill protection (~−4.2/fill CLOB vs ~−0.4/fill FBA, ~10×).
**Flat in τ** (≈+87 at τ=1,10,50) — the protection is the *mechanism switch*, not
the batch interval. Transfer holds directionally (informed Σ +182 → ~+132;
prevention, not 1:1 conservation).

**Caveats (load-bearing — do not drop):** one cell only (mid/low, ts=8000, single
delay gap); τ=200 contaminated by the LP trading less (`n_fills` 12→7 — clean
comparison is τ=1–50); 96% robust not 100% (1/25 seeds flipped); whole result
rests on the faithful-environment chain above. Defensible claim is the **paired
~95% reduction in this regime, protection from the mechanism switch**, NOT
"batching eliminates extraction" and NOT generality beyond the tested cell.

**Open threads (not done):** walk-aware convergence metric + AMM `lp_rent` still
t=0-referenced (Phase C); regime/delay-gap sweep of the τ-result; endogenous LP
spread (§5.4); corrected incumbent baselines on the new recording (§5.6). The §5
sections below are the original plan; the §9 "first move" is historical.

---

## 0. Who I am / how to read this

I'm Aditya — rising senior at Penn (Econ/Bio/CS, grad May 2027), based in NYC for summer 2026. I do independent research on prediction-market microstructure and mechanism design. Public surface: `adityanb.com`, Substack (`adibhosale.substack.com`), GitHub (`github.com/adityanbhosale`). Stack: Python, Solidity, TypeScript, SQL, React/Next, Foundry, Node. I use a `handoff.md` pattern for cross-session continuity (this doc).

I implement in **Cursor**; I want you to **architect and review** — write detailed specs and Cursor prompts, read back what the agent reports, catch errors. Don't write production code directly unless I ask; do write precise prompts I can paste. Recon before building against interfaces you're guessing at (see §8).

---

## 1. The through-line (the one question)

**When should continuous matching give way to batch / optimization-based clearing in prediction markets — and what does that imply as institutional and AI-agent capital arrives?**

I attack this two ways that triangulate:
- a **synthetic agent-based simulator** (`orderbook-amm-hybrid-sim`) where behavior is endogenous but the world is invented;
- **real cross-venue prediction-market data** (`kalshi-polymarket-microstructure`) where the world is real but behavior is fixed.

Neither alone is sufficient; the program is the interplay. Theoretical anchor: Budish–Cramton–Shim (continuous limit order books are a flawed design that pays sniping rents to the fastest; a frequent batch auction removes the speed race). Frontier bet: event/on-chain markets differ structurally from equities (prices bounded [0,1], discrete resolution, cross-market correlation, MEV adversary), so the optimal mechanism is an open question, not a clean import.

---

## 2. Project history & artifacts

**2.1 `orderbook-amm-hybrid-sim`** — the simulator. Agent-based comparison of AMM vs price-time CLOB vs hybrid venue designs on synthetic order flow; ~900-run sweep across capital/signal regimes. Measures price discovery, LP rent, rent efficiency per mechanism. Headline takeaway from the original work: matching-mechanism choice has measurable distributional consequences (who captures rent, how fast price converges). **This is the repo for the current build.**

**2.2 `kalshi-polymarket-microstructure`** — the empirical work (two published parts).
- *Part 1*: frozen ~1.06M-row cross-venue L2 dataset (Kalshi/Polymarket), fee/execution model calibrated to each venue's real schedule. Finding (negative result): **0 of 15 markets takeable at any retail fee tier** despite persistent visible crosses — edge survives only at an institutional tier neither venue offers; crosses are adverse-selection-paid (negative post-fill markout on LP-edge markets); ~100ms latency floor traceable to venue cloud regions (Kalshi us-east, Polymarket eu-west). Conceptual hinge: short-resolution binaries discretize *resolution time*, not *matching*, so the HFT race survives.
- *Part 2* (`batch_counterfactual/` module): a deterministic uniform-price call-auction engine on the frozen set. See §3.

**2.3 Published essays** (Substack): Part 1 ("Are Prediction Markets Evolving Towards Batch Auctions?") and Part 2 ("Institutionalization of Prediction Markets & Anticipated Mechanism Failures").

**2.4 Earlier context (not load-bearing here):** a dual-layer on-chain liquidity protocol (ERC-3643 SPV on Ethereum Sepolia + LS-LMSR AMM on Base Sepolia; live at `dual-layerbiotechliquidity.vercel.app`, repo `lmsr-preclinical-markets`); a moat-decay analysis. Mentioned so you have the full shape of my background; ignore for this build.

---

## 3. What the counterfactual batching study proved (Part 2 / `batch_counterfactual`)

The deterministic uniform-price call-auction engine clears real crossed episodes the way a batch auction would. Key results:
- **Fee cliff survives at episode level:** gross ~100% of crossed episode-starts clearable, **retail ~6%**, **institutional ~77%** (0.30/0.20% tier). The edge is real and nearly always present gross; the fee schedule reserves it for fast institutional capital.
- **Flagship:** a 15.15h cross (1,787 cycles) that a single batch call collapses into one print — ~132k contracts, **~$468 price improvement** to the resting side (blended 0.354¢/contract gross; institutional $239 / 0.181¢).
- **Objective disagreement:** max-executable-volume and max-price-improvement objectives select **different clearing prices in 1,864 sized cases** → *which* objective a venue's auction optimizes is itself a design parameter with distributional consequences. (This is the deepest mechanism-design result I have.)
- **Integrity beat (kelce reconciliation):** the engine contradicted Part 1's published "0 of 15" headline — one deep-tail market *was* retail-clearable. Traced to ~85% data vintage (wide-cross regime emerged ~June 2; retail-clearable fraction 21%→88% around June 3), ~0% instrument, ~15% fee-structure (median clearing C≈0.03 vs NYK 0.36 — kelce sits in the low-fee tail). Published the reconciliation rather than burying it.

**Critical limitation that motivates the sim:** this is a **mechanical counterfactual — order flow held fixed**. It shows the rent *exists*, not that batching *removes it under endogenous behavior*. That's exactly the gap the FBA sim closes.

---

## 4. Current engineering state of `orderbook-amm-hybrid-sim` (DETAILED)

Three reconnaissance/build entries are logged in `build_log_fba.md`. Summary of the real interfaces and what's done:

### 4.1 Architecture (recon, Entry 1)
- **Venue ABC** (`venues/base.py`), six methods: `submit_market_order(agent_id, side, qty)`, `submit_limit_order(..., price)`, `cancel_order`, `get_state() -> VenueState`, `estimate_impact(side, qty)`, `tick()`. AMM, CLOB, hybrid all implement it.
- **Clock/cadence:** the sweep fires `venue.tick()` once per integer timestamp via a `venue_clock` event at **priority −50**, *before* signals (0) → decisions (100) → trades (200). So submits at step `t` land **after** t's clear and **before** t+1's → a clean "accumulate during [t, t+1), clear at t+1" batch cadence, for free. **τ = clear every N ticks** is just a counter in the venue, not a scheduler.
- **Agents:** no base class; they satisfy a `PopulationAgent` Protocol (`observes`/`decide`/`review`/`fire_noise` + fields `observation_delay`, `review_interval`, `arrival_rate_per_unit`). Five classes: `NaiveGaussianBeliefAgent`, `TailAwareGaussianBeliefAgent`, `AggregatedEvidenceAgent` (cross-market), `JointFactorFairValueAgent` (joint factor posterior), `EventDrivenNoiseAgent`.
- **Latency plumbing exists but is OFF:** `observation_delay` schedules a decision at `now + delay`, but the sweep sets it to **0 for everyone**, so same-timestamp ordering falls through to heap insertion order (agent-list order). **Turning delays on is the main thing that makes FBA have something to neutralize.**
- **Fair value — NO price process (important):** truth is a **static** log-linear latent-factor draw at t=0 (`L*_m = α_m + β_m·f + ε_m`), anchored to opening mids, **never moves, unbounded, no jumps**. "News" = Poisson streams of **noisy signals of the fixed truth**, in two tiers (routine vs more-precise "tail" signals). Venue prices start displaced from truth by N(0, 1%) shocks. **Implication:** the extraction channel here is **information asymmetry** (better-informed agents picking off worse-informed LPs; speed = who acts on a signal-round first), **NOT** latency-sniping-on-news (there are no jumps to snipe). Frame any writeup honestly as information-asymmetry extraction. (Adding a jumping truth is a possible *second* arm, not this one — it touches every agent's belief update.)
- **Metrics:** per run — convergence (`normalized_rmse_log`, max rel price error, `convergence_tick`), rent/PnL (`lp_rent_total`, `informed_pnl_total`, `noise_pnl/loss`, `rent_efficiency[_stable]`), capital-exhaustion fraction. **NO markout, NO effective spread.** PnL is horizon MTM at terminal fair, split only informed-vs-noise. But `TradeRecord` carries `mid_price_before/after` per fill → markout-at-Δ and effective spread are **post-hoc computable**.
- **Sweep** (`simulation/sweep.py`): 3 mechanisms × 2 mixes × 3 capital bands × 2 signal regimes × 25 seeds = **900 runs**, fully serial, seeded via `SeedSequence.spawn(4)`. Output: `analysis/results/sweep_summary.parquet` + `sweep_timeseries.parquet`. `rerun_clob_and_merge` is an existing precedent for an additive single-mechanism run — template for an additive `fba`-only run.
- **Clearing:** only continuous matching today. `batch_counterfactual/auction.py` (sibling repo) is the algorithm reference (uniform-price call, midpoint tie-break, pro-rata + largest-remainder) but **not importable as-is** (loose module, hard sibling imports, Decimal math, [0,1] prices) — vendored natively instead (done, §4.4).

### 4.2 The deferred-fill path (recon, Entry 1 cont.)
- `_on_trade` (in `MarketEnvironment`) handles `TRADE_EVENT`, fed `TradeIntent` payloads at delay 0; it **builds the `TradeRecord` itself** from the synchronous `OrderResult` returned by `submit_*`.
- `OrderResult`/`TradeRecord` have **no status field**; the "accepted, no fill yet" convention is `filled_quantity=0, remaining_quantity=qty, order_id=<id>`.
- **Pre-existing bug found:** when a resting limit is hit later, the **maker's fill was never recorded** — the CLOB just shrank the `(agent_id, qty, oid)` tuple in the deque (`venues/clob.py` `_execute_buy/_execute_sell`). No `TradeRecord`, no callback. So maker-side PnL/markout was invisible. (Fixed — §4.3.)
- `mid_price_before/after` were captured inside `_on_trade` bracketing the synchronous submit → meaningless for a fill produced at clear time. The batch clear must capture pre/post-clear mids itself.
- `tick()` was declared `-> None`, every call site discarded its return, env never called it; env learned of executions **only** via `submit_*` return values.

### 4.3 Maker-fill recording fix — DONE, committed
- New `MakerFill` dataclass + `Venue.drain_maker_fills()` (default `[]`, so AMMs untouched, six abstract methods unchanged). CLOB's four crossing loops buffer one `MakerFill` per resting order consumed (agent, maker side, qty, resting limit price, order id, venue mid bracketing that consumption incl. level cleanup). `HybridVenue` delegates the drain.
- `_on_trade` (and `execute_market_order`) drains maker fills into `TradeRecord`s with `liquidity="maker"`, `fees_paid=0`, `capital_committed=0` (maker capital committed at rest time). `TradeRecord` gained `liquidity: str = "taker"`.
- **Zero-qty/capital decoupling** (the subtle bit): capital reconciliation moved to a new `cost_log` (one `CostEntry` per intent, identical amounts/timing); `_sync_costs` + exhaustion metric read that; `trade_log` holds fills only; zero-qty rows dropped. `pending_cost` verified bit-identical.
- **Verification:** 10 tests pass (5 pre-existing + 5 new, incl. conservation Σbuy==Σsell). Behavior byte-identical (final mids, `deployed`/`pending_cost`, taker records). **Recorded volume exactly 2×** (both legs now taped). `informed_pnl_total` on hybrid moved −6.878 → −7.614 (informed resting limits that got hit are now marked — previously invisible); CLOB unchanged on that seed (no informed resting order hit). Convergence/`frac_informed_exhausted` unchanged.
- **Consequence:** prior 900-cell results undercounted maker PnL → **incumbent baselines must be re-run before any FBA-vs-incumbent comparison.**

### 4.4 The FBA venue — DONE, built + tested, committed
- `venues/fba.py` (~408 lines). Canonical Budish–Cramton–Shim: resting limit book cleared by a periodic uniform-price call every `tau_ticks`.
- **Deferred submits:** `submit_limit_order`/`submit_market_order` return an "accepted, pending" `OrderResult` (`filled=0, remaining=qty`, real `order_id`), no synchronous fill. Limits rest across batches until filled/cancelled. Market orders queue (treated as buy@max / sell@min), **expire** if no opposite interest at the next clear (record nothing).
- **`_solve_clear`:** candidates from limit prices, max-volume objective, **midpoint tie-break** on the flat max-volume interval (`p_star = (best_prices[0]+best_prices[-1])/2`), short side fills fully, other side rationed pro-rata + largest-remainder. Quantities quantized to a **1e-9 grid** so rationing is exact integer math (deterministic, conserving).
- **`_run_clear`:** capture `mid_before`, solve, apply book mutations, capture `mid_after`, stamp the **same pre/post-clear mids on every fill** in the batch, tag `liquidity` ("maker"=limit leg, "taker"=market leg).
- **Drain wiring:** extended the Entry-2 channel rather than a parallel one — `MakerFill` gained `liquidity` and `fees_paid` (defaults keep CLOB/hybrid byte-identical); env drain generalized to `drain_venue_fills(sim)` → `_record_drained_fills`. Maker legs capital 0 (committed at submit); taker legs charge capital at clear with a matching `CostEntry`. **Entry-4 sweep integration is one line in `_pulse`:** tick all venues, then `market_env.drain_venue_fills(sim)`.
- **Reuse vs mirror:** CLOB's price-time deque is for continuous priority FBA doesn't have (pro-rata, no time priority in a batch) → book **mirrored** as a flat insertion-ordered order map; no new price type (uses CLOB `_pkey`); `EmptyBookError` imported for `estimate_impact` parity.
- **`get_state()`** mid = resting-book best-bid/ask midpoint between clears. **`estimate_impact`** = approximate clearing-price move from adding qty.
- **Verification:** 12 FBA tests pass (hand-computed clear, uniform price, midpoint tie-break, conservation w/ rationing, pending semantics, resting persistence + partial remainder, market expiry, determinism, clear-time-vs-submit-time mids, τ=1 batches every tick, estimate_impact, env-drain both legs w/ clear timestamp). Full suite **22 pass**.
- **Documented deviation:** FBA market orders return a real `order_id` (the `OrderResult` docstring says "None for market orders") because they live until the clear and their fills need attribution. **Grep downstream for `order_id is None` used as a market-order sentinel** before relying on it.

**Status:** venue is correct and cleanly recorded, but **inert** — with `observation_delay=0` there's no speed/info asymmetry for batching to neutralize, so an FBA-vs-CLOB run right now would show ~null difference by construction. Wiring that asymmetry is the next task.

---

## 5. WHAT TO BUILD NEXT (the task)

### 5.1 Goal & headline deliverable
Turn the sim into an apparatus that takes (venue mechanism, agent population w/ latency/info structure) and outputs **LP markout, taker markout, adverse-selection cost, and price-discovery lag per mechanism**. The headline figure is the **τ-curve**: extraction (LP markout / sniper-equivalent PnL) falling as the batch interval τ grows, plotted against the **immediacy cost** of waiting, yielding an **optimal τ\***. The claim to test: *a batched venue reduces information-asymmetry extraction vs CLOB and AMM on event-contract-shaped flow, at a quantifiable immediacy cost.*

### 5.2 The critical design point — latency/information differentiation (do this first)
Turn `observation_delay` **on**, differentiated by agent role: better-informed agents (tail-signal recipients) act on fresh signal rounds at short delay; LP-style agents quote at longer delay so their resting quotes reflect staler beliefs. Wire the **signal-tier structure** (routine vs tail signals) together with the delay so that fast access to precise tail signals is the edge a batch erases (everyone acting on the same signal-round clears together at one price). Without this, the result is null by construction.

### 5.3 Markout metrics (post-hoc)
Compute markout-at-Δ and effective spread from `TradeRecord.mid_price_before/after` + the `liquidity` tag, **outside the hot loop**. Markout is the headline metric because it maps to the bleeder we care about (the adversely-selected LP). Make sure FBA fills (clear-time mids) and CLOB/hybrid fills (now symmetric after the maker fix) are computed on the same definition so cross-venue comparison is honest.

### 5.4 Endogenous LP spread (the sim's edge over the frozen-data study)
Let LP spread **respond to expected markout**: an LP sniped less under batching should quote tighter, so the welfare gain shows up as a narrower spread for everyone, not just redistributed PnL. This is what lets the sim answer the question the mechanical counterfactual couldn't ("what would participants *do* under batching"). It's also the more honest result.

### 5.5 The honesty beat (bake in from the start)
Batching is not free — it trades immediacy for protection (orders wait up to τ). Report the **tradeoff**, not just the dividend: if FBA cuts extraction but reduces volume or delays informed price discovery, that cost goes in the headline next to the benefit. The credible, defensible result is "extraction falls faster than immediacy cost rises, up to τ\*" — a real optimum, far stronger than "batching strictly wins."

### 5.6 Sweep integration + corrected baselines
- Wire `drain_venue_fills` into `_pulse` (one line, per §4.4) and add `fba` (with a τ grid) to the sweep, using `rerun_clob_and_merge` as the additive-run template.
- **Re-run the incumbent baselines** (CLOB/hybrid/AMM) on the corrected recording before comparing — the maker fix changed their PnL. **Sequencing:** do the latency wiring + a single-seed FBA-vs-CLOB sanity check **before** spending compute on the full corrected grid (no point regenerating 900+ cells for a world where batching can't matter yet).

### 5.7 Framing nuance (don't get this wrong)
Because the truth is static (§4.1), the demonstrated claim is **"batching reduces information-asymmetry extraction among differentially-informed agents,"** NOT "batching stops latency arbitrage on news." Keep that distinction honest in any writeup — it's the same discipline that made the kelce reconciliation land. (A jumping-truth arm could add the canonical news-sniping channel later; it's out of scope for this build.)

### 5.8 Out of scope for this build
CoW-style solver competition / coincidence-of-wants (canonical FBA first; solver layer is a later arm). A jumping fair-value process. Any of the other three one-pager primitives (§7) — those are theses, not summer builds.

---

## 6. Engineering discipline / conventions
- Deterministic seeds; config-driven sweep grid; unit tests (FBA clearing must reproduce a hand-computed uniform price); smoke tests; append design decisions to `build_log_fba.md` (entries are numbered — next is Entry 4).
- Content-hash / freeze datasets where reproducibility matters (mirrors the frozen-set discipline from the empirical repo).
- Read-only recon before building against interfaces; report before implementing; stop before commit and show diffs + test output; I make the commit decision.
- Conservation invariants as guardrails (Σbuy == Σsell per clear) — the single best check that fills aren't invented or dropped.

---

## 7. Strategic context — why this build is load-bearing (the market mapping)

Over the past week I mapped the decentralized-infrastructure market against **Brad Holden / PL Capital Crypto's** thesis (I'm in conversation with them). His frame: *crypto isn't an asset class, it's programmable infrastructure for capital, coordination, and computation; resting on three properties — **contracts verifiable, markets neutral, systems trustless**.* The first and third already have primitive stacks and funded portfolios (FHE/Zama, ZK, decentralized compute like Gensyn/Prime Intellect, crypto-native identity like Privy). **"Markets neutral" is the under-built leg** — the property is named but lacks an equivalent primitive stack.

I wrote a one-pager (RFS-style) claiming that white space, with four candidate primitives, each framed on the FHE-bet template ("looks early, but the demand — institutions now, AI agents next — is arriving"):
1. **Neutral matching for AI-agent markets** (coordination) — batch/optimization clearing as the coordination primitive for machine actors; a speed race among agents has no equilibrium but maximal infra spend.
2. **Neutral settlement rails for on-chain event markets** (capital + coordination) — a CoW-style batch-settlement layer specialized for event-contract microstructure; my home turf, with empirical receipts.
3. **Verifiable execution-quality infrastructure** (computation + verifiability) — open, auditable "what's my realized execution quality on this venue" — the tooling I had to hand-build.
4. **Pre-trade privacy for agent order flow** (privacy + coordination) — sealed-bid batch auctions + encrypted submission (threshold/MPC/FHE); flagged as the speculative swing.

**Why the FBA sim is the load-bearing next step:** Gaps 1 and 2 both rest on a single unproven claim — *that batch clearing measurably reduces extraction on event-contract flow under endogenous behavior.* The frozen-data study showed the rent **exists** (mechanical, behavior fixed); it did **not** show batching **removes** it when agents adapt. The FBA sim is the experiment that converts the central claim from **asserted** to **demonstrated**. If the τ-curve comes back flat, the matching primitive isn't worth building — which is itself enormously valuable to know. So the sim is the proof that earns the right to build the primitive; it is not a detour from the primitives, it *is* the first one, done right.

**Decision already made:** build the FBA sim first (proof), then build the *one* primitive the proof says matters — do **not** build thin versions of all four (breadth here is the "not-ready-to-found" failure mode flagged by EF; depth on the measured thing is the stronger path). The applied target, if the proof holds, is a CoW-style batch-settlement layer where the **customer is the flow originator** (frontend/wallet/aggregator whose users are bleeding to extraction), not "be the venue" — which dissolves the two-sided cold-start.

---

## 8. How I want you to work
- **Direct, unvarnished feedback.** Push back when I'm wrong; name tradeoffs; don't cheerlead. If something I propose is a mistake, say so and why.
- **Concise, structured prose.** Not heavy bullets for simple things; structure when genuinely multifaceted.
- **I make final calls.** Flag send-blockers / real errors clearly; flag a given concern at most twice, then defer to me (the "second-flag" rule).
- **Integrity beat.** Surface contradictions in my own work proactively (the kelce-reconciliation ethos) — catching a confound before it becomes a published artifact is the highest-value thing you do.
- **Workflow:** spec + Cursor prompts; I implement; you review the agent's reported output (read the hand-computed test and conservation invariant first). Recon (read-only) before specs when interfaces are uncertain.
- **Sequencing honesty:** tell me when something is a banked win vs. an open thread, and don't let me cram a build against a hard deadline when a rested pass would be better.

---

## 9. Immediate first move for the new chat
Start with the **latency/information differentiation wiring** (§5.2) — it's the precondition for everything else. Before writing the spec, do a short **read-only recon** of how `observation_delay`, the signal-tier emission, and the agent `decide`/`review` cycle actually interact in the current code (the sweep zeroes the delay, so confirm exactly where it's set and how a non-zero delay propagates), then write the Cursor prompt to (a) enable role-differentiated delays, (b) tie tail-signal access to the fast role, and (c) add the post-hoc markout metric. Then a single-seed FBA-vs-CLOB sanity check to confirm the markout difference is non-null **before** any full-grid re-run.
