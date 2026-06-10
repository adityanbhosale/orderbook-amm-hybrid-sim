# Build Log — FBA (Frequent Batch Auction) Venue

## Entry 0 — Architecture survey (read-only, no FBA code yet)

Date: 2026-06-09. Survey of the existing repo prior to adding a fourth venue
(`fba`). No code modified.

---

## 1. Venue abstraction

All three venues (`ConstantProductAMM`, `CLOB`, `HybridVenue`) implement the
abstract base class `Venue` in `venues/base.py`. A fourth venue must implement
exactly these six methods. Pasted verbatim:

```python
@dataclass
class OrderResult:
    """Result of submitting an order to a venue."""
    filled_quantity: float       # Quantity actually executed
    avg_fill_price: float        # Volume-weighted average fill price
    remaining_quantity: float    # Unfilled remainder (zero for market orders on AMMs)
    order_id: Optional[str]      # For tracking resting limit orders; None for market orders
    fees_paid: float


@dataclass
class VenueState:
    """Snapshot of venue state, returned to agents on observation."""

    mid_price: float | None
    best_bid: Optional[float]
    best_ask: Optional[float]
    spread: Optional[float]
    tick: int


class Venue(ABC):
    """Abstract base class for trading venues."""

    @abstractmethod
    def submit_market_order(
        self, agent_id: str, side: str, quantity: float
    ) -> OrderResult:
        """Submit a market order. side is 'buy' or 'sell'."""
        ...

    @abstractmethod
    def submit_limit_order(
        self, agent_id: str, side: str, quantity: float, price: float
    ) -> OrderResult:
        """Submit a limit order. May raise NotImplementedError for pure AMM venues."""
        ...

    @abstractmethod
    def cancel_order(self, agent_id: str, order_id: str) -> bool:
        """Cancel an outstanding limit order. May be no-op for AMMs."""
        ...

    @abstractmethod
    def get_state(self) -> VenueState:
        """Return current venue state for agent observation."""
        ...

    @abstractmethod
    def estimate_impact(self, side: str, quantity: float) -> float:
        """Estimate the volume-weighted average fill price for a market order
        of the given side and quantity, without executing it.

        AMM: compute from cost function.
        CLOB: walk the book.
        Hybrid: walk the combined book.

        Returns the expected average fill price. Agents use this to size
        trades against a target impact rather than against mechanism-specific
        liquidity parameters.
        """
        ...

    @abstractmethod
    def tick(self) -> None:
        """Advance venue by one timestep. Used for time-dependent state."""
        ...
```

Key observations for FBA:

- **`tick()` is the natural batch-clear hook.** The sweep harness registers a
  `venue_clock` event at every integer timestamp (`_register_venue_clock` in
  `simulation/sweep.py`, priority −50, fires before SIGNAL=0 / DECISION=100 /
  TRADE=200) and calls `v.tick()` on every venue. `HybridVenue.tick()` already
  uses this to cancel-and-replace LP quotes each tick, so an FBA venue can
  accumulate orders between ticks and clear the batch inside `tick()`.
- **Synchronous fill assumption.** `OrderResult` is returned synchronously
  from `submit_*`. In an FBA, fills happen at the next batch clear, so either
  (a) the venue reports `filled_quantity=0, remaining_quantity=qty, order_id=…`
  at submit time and the trade log / capital accounting needs a path for
  deferred fills, or (b) the batch interval is aligned so submits within a
  tick are cleared by the venue clock before the next decision — note
  `MarketEnvironment._on_trade` builds the `TradeRecord` from the synchronous
  `OrderResult`, so deferred fills will NOT appear in `trade_log` without
  changes to `MarketEnvironment` or an FBA-internal fill log.
- `estimate_impact` must answer "what price would I get" — for FBA the honest
  answer is the expected clearing price; agents size trades with it
  (`route_log_space_trade` in `agents/belief_utils.py` dispatches on
  `isinstance(v, ConstantProductAMM)` — an FBA venue would fall into the
  CLOB/hybrid maker-taker branch by default; check `_route` logic there).
- `MarketEnvironment` (`environment/market_environment.py`) maps
  `market_id -> Venue`, owns `trade_log: list[TradeRecord]`, and handles
  `TRADE_EVENT`. Adding a venue requires no changes there, only a new
  `_fba_venues_from_truth(...)` builder + mechanism literal in the sweep.

## 2. Agent model

There is **no agent base class** — agents are plain `@dataclass`es satisfying
the `PopulationAgent` `Protocol` in `environment/agent_population.py`. Pasted
verbatim:

```python
class PopulationAgent(Protocol):
    """Agents participating in ``AgentPopulation`` must expose this surface."""

    agent_id: int
    budget: float
    deployed: float
    pending_cost: float
    observation_delay: int
    review_interval: int
    arrival_rate_per_unit: float

    def observes(self, market_id: int) -> bool:
        ...

    def decide(
        self, sim: Simulator, signal: Signal, market_env: MarketEnvironment
    ) -> TradeIntent | None:
        ...

    def review(
        self, sim: Simulator, market_env: MarketEnvironment
    ) -> list[TradeIntent]:
        ...

    def fire_noise(
        self, sim: Simulator, market_env: MarketEnvironment
    ) -> TradeIntent | None:
        ...
```

### Agent classes (all in `agents/`)

| Class | File | Observes | Belief model |
|---|---|---|---|
| `NaiveGaussianBeliefAgent` | `informed_naive.py` | signals on its `market_ids` | per-market conjugate Gaussian on log-fair-value; uses a fixed (misspecified) `signal_precision_assumed` for every signal |
| `TailAwareGaussianBeliefAgent` | `informed_tail.py` | signals on its `market_ids` | same, but priors anchored to noisy truth (`base_log_levels`) and weights each signal by its true precision 1/noise_std² (tail signals carry more weight) |
| `AggregatedEvidenceAgent` | `informed_aggregated.py` | signals on `observed_markets` (superset); posteriors only on primary `market_ids` | discounted cross-market Gaussian updates via `cross_weights` |
| `JointFactorFairValueAgent` | `informed_joint_factor.py` | signals on `observed_markets` | joint information-form Gaussian posterior (Λ, η) on the latent factor vector f; L_m = α_m + β_mᵀf, ignoring idiosyncratic ε |
| `EventDrivenNoiseAgent` | `event_noise.py` | nothing (no signals) | Poisson arrivals (`arrival_rate_per_unit`), random market/side/size market orders |

### Latency model — CRITICAL FOR FBA

The infrastructure for latency differentiation **exists but is unused**:

- Each agent has `observation_delay: int`. `AgentPopulation._on_signal` fans
  out each `Signal` to all observing agents and schedules each agent's
  `agent_decision` event at `now + agent.observation_delay`
  (`agent_population.py` lines 143–157). **In the sweep, every agent is
  constructed with `observation_delay=0`** — so all informed agents react to
  the same signal at the same timestamp.
- Within a timestamp, ordering is deterministic by `(timestamp, priority,
  insertion_order)` (heap in `environment/events.py`): venue_clock (−50) →
  SIGNAL (0) → DECISION (100) → TRADE (200) → BOOKKEEPING (1000). Same-priority
  ties break by **insertion order**, i.e. agent list order in the fan-out loop.
  So agents act "simultaneously" in wall-clock terms but sequentially and
  deterministically in execution — agent 1 always trades before agent 2 on the
  same signal under continuous matching. This is exactly the speed-race
  artifact an FBA is meant to neutralize, and it's currently decided by list
  order rather than an explicit latency draw.
- Other timing knobs: `review_interval` (periodic re-evaluation, 4500–5000
  ticks in the sweep) and `arrival_rate_per_unit` (noise Poisson intensity).
- Time: discrete-event `Simulator` (`environment/simulator.py`) with integer
  ticks; `time_resolution=1000` ticks per unit time; horizon `until_ts=40_000`
  (i.e. 40 time units).

**Implication:** to study FBA vs continuous properly, per-agent
`observation_delay` (or a randomized within-batch arrival jitter) should be
turned on; otherwise FBA's batching neutralizes an ordering artifact that is
purely insertion-order, not a modeled latency advantage.

## 3. Fair-value / price process

**The truth is static — there is no price process at all.** Each run draws a
single log-linear latent-factor fair value at t=0 and it never moves: no
random walk, no GBM, no mean reversion, no jumps, no news shocks. What arrives
over time are noisy *signals* of the fixed truth (so "news" = progressively
informative observations, not value changes). From `environment/information.py`:

Truth construction (`LatentFactorModel.__init__`, abridged to the core):

```python
self.f: np.ndarray = rng.standard_normal(config.k)   # latent factors ~ N(0, I_k)

for cluster_id, cluster in enumerate(config.clusters):
    for _ in range(cluster.market_count):
        loadings = rng.normal(0.0, cluster.secondary_loading_std, size=config.k)
        loadings[cluster.primary_factor] = rng.normal(
            cluster.primary_loading_mean, cluster.primary_loading_std
        )
        idio = float(rng.normal(0.0, config.idiosyncratic_std))
        beta_f = float(loadings @ self.f)
        log_mid = float(np.log(mids[market_id]))
        alpha = float(log_mid - beta_f)          # anchored so pre-noise log fv = log(mid_0)
        log_fv = alpha + beta_f + idio
        truths.append(MarketTruth(..., log_fair_value=log_fv,
                                  fair_price=float(np.exp(log_fv))))
```

So `L*_m = α_m + β_mᵀ f + ε_m`, with α anchored such that
`L*_m = log(mid_m(0)) + ε_m`. Unbounded in log space (lognormal price > 0).

Signals (`InformationEnvironment.schedule_signals`): Poisson streams per
market, two tiers —

- routine: rate `routine_rate_per_market` (0.75/unit in sweep), noise std
  `signal_noise_std` (0.012 low-regime / 0.055 high-regime, log units)
- tail: rate `tail_rate_per_market` (0.035/unit), noise std `tail_noise_std`
  (0.005 / 0.022) — tail signals are *more precise*, not jumps

`tail_mode="marked"` in the sweep: one merged Poisson stream; each draw is
tail w.p. `tail_rate / (routine_rate + tail_rate)`. Payload:
`Signal(market_id, value = log_fv + N(0, σ), is_tail, noise_std)`
(`environment/signals.py`).

Initial venue prices are displaced from truth by per-market
`N(0, ANCHOR_DISPLACEMENT_STD=0.01)` shocks (`_anchor_displacements` in
`simulation/sweep.py`), so there is something to converge toward.

**Implication for FBA:** with a static truth, post-fill markout against
"future fair value" is degenerate (fair never moves); markout would have to be
measured against future *mids*. If FBA results should speak to sniping/
adverse-selection, consider whether the information process needs jumps or a
moving truth — none exists today.

## 4. Metrics

Modules: `metrics/convergence.py`, `metrics/rent.py`, `metrics/capital.py`.

Per-run summary columns (assembled in `run_single_simulation`, sweep.py):

- `n_trades`
- Convergence (`ConvergenceResult`): `normalized_rmse_log` (RMSE of log final
  mids vs log truth), `max_relative_price_error`, `convergence_tick` (first
  tick all markets within `rel_band=0.02` of fair; from a piecewise-constant
  mid trajectory rebuilt from `TradeRecord.mid_price_after`)
- Rent/PnL (`RentPnlResult`): `lp_rent_total` (CP-AMM LP mark-to-market change
  at terminal fair), `informed_pnl_total`, `noise_pnl`, `noise_loss`,
  `rent_efficiency` (= informed_pnl / |noise_loss|),
  `rent_efficiency_stable` (denominator floored at 10.0)
- Capital (`CapitalSaturationResult`):
  `frac_informed_exhausted_before_convergence`, `n_informed_agents`,
  `n_informed_exhausted`
- Plus a long-format per-(run, tick, market) mid timeseries.

**Markout: NOT measured anywhere.** PnL is horizon mark-to-market at terminal
fair (`_trade_mtm_pnl`: `qty * (fair_end − fill)` for buys), valid only
because truth is static. No post-fill markout at t+Δ, no effective spread, no
realized spread, no price-impact decomposition. The raw material exists:
`TradeRecord` carries `timestamp, market_id, agent_id, side, quantity,
avg_fill_price, fees_paid, capital_committed, mid_price_before,
mid_price_after`, and `build_mid_trajectory_from_trades` gives mids over time
— so markout-at-Δ and effective spread (`2·|fill − mid_before|`) are
computable post-hoc from existing logs without touching venues.

**PnL by agent type:** only two buckets (informed vs noise), set-membership by
`agent_id` (`informed_agent_ids={1,2,3,4}` for diverse mix; noise id 99).
Per-class PnL (e.g. naive vs joint-factor) is not broken out, but trivially
derivable from `trade_log` since every record has `agent_id`.

## 5. Sweep harness (`simulation/sweep.py`)

- Grid (`SweepConfig` defaults): `mechanisms=("amm","clob","hybrid")` ×
  `mixes=("diverse","naive_dominated")` × `capital_bands=("low","mid","high")
  = (100, 1000, 10000)` × `signal_regimes=("low","high")` × `n_seeds=25`
  → **3 × 2 × 3 × 2 × 25 = 900 runs**. Adding `fba` → 1200.
- Mechanism is a `Literal["amm","clob","hybrid"]` (`MechanismName`); the only
  branch points are the venue-builder if/elif in `run_single_simulation`
  (lines 377–382) and the pool-start/end bookkeeping (which for non-AMM venues
  synthesizes notional reserves `(reserve_x, reserve_x·mid)`).
- **No parallelization** — five nested `for` loops, fully serial, single
  process (`run_sweep`). No joblib/multiprocessing. Reproducibility via
  `np.random.SeedSequence(seed).spawn(4)` → independent sim/world/agent/anchor
  RNG streams per run.
- Persistence: **Parquet** under `analysis/results/` —
  `sweep_summary.parquet` (900 rows) and `sweep_timeseries.parquet`
  (long mids). `run_id = f"{mechanism}__{mix}__{capital_band}__{signal_regime}__seed{seed}"`.
- There is a precedent for incremental mechanism re-runs:
  `rerun_clob_and_merge` re-runs one mechanism and merges into existing
  Parquet by filtering `mechanism != "clob"` / `run_id.startswith("clob__")`.
  An `fba`-only additive run can follow the same pattern without recomputing
  the existing 900.
- CLI: `python -m simulation.sweep` (full) or `... clob-only`.

## 6. Clearing code — continuous only; auction.py is vendorable

**This repo has only continuous matching.** `CLOB` (`venues/clob.py`) is
price-time priority with immediate execution on submit (market orders walk the
book; marketable limits cross then rest the remainder). `HybridVenue` wraps
the same CLOB and adds per-tick cancel/replace LP ladders. The AMM clears
trade-by-trade on the constant-product curve. Grep for
auction/batch/uniform-price finds nothing in this repo.

**However**, a uniform-price call auction engine exists in a sibling project:

- Path: `/Users/adityabhosale/Downloads/Projects/kalshi-polymarket-microstructure/batch_counterfactual/auction.py`
  ("Uniform-price call auction engine for the batch-auction counterfactual").
- Relevant pieces: an `Order(order_id, owner_id, venue, side, price, qty)`
  dataclass, `clear` / `clear_joint` full order-book clearance, midpoint
  tie-break over the optimal clearing interval, pro-rata rationing at the
  margin with largest-remainder rounding (deterministic, no RNG).
- **Not importable as-is**: it is a loose top-level module (no package), with
  hard imports of its siblings `book` (BookState, venue ticks) and `fees`
  (Kalshi/Polymarket fee schedules), and it works in `Decimal` with
  prediction-market prices in [0,1] and venue tick grids. **Vendorable** with
  moderate surgery: the core uniform-price crossing + pro-rata rationing logic
  is self-contained once `BookState`/fee/tick dependencies are stripped and
  Decimal is swapped for float to match this repo's conventions.
- Alternative: a from-scratch FBA venue here is small — accumulate
  `submit_*` orders into a batch, and in `tick()` compute the
  max-volume uniform clearing price from the aggregated supply/demand step
  functions, pro-rata at the margin. The auction.py tie-break and rationing
  rules are the part worth porting verbatim.

---

### Summary of integration points for the FBA venue (for the next entry)

1. New `venues/fba.py` implementing `Venue`; batch accumulation in
   `submit_market_order`/`submit_limit_order`, uniform-price clear in `tick()`
   (already driven once per integer timestamp by `_register_venue_clock`).
2. Decide how deferred fills reach `MarketEnvironment.trade_log` (synchronous
   `OrderResult` is assumed today).
3. `estimate_impact` semantics for batch clearing (agents size with it).
4. Add `"fba"` to `MechanismName`, a `_fba_venues_from_truth` builder, and the
   pool bookkeeping branch in `run_single_simulation`; extend `SweepConfig`.
5. Latency: enable nonzero per-agent `observation_delay` if the FBA comparison
   is meant to say anything about speed competition (currently all zero;
   same-timestamp ordering is insertion-order).
6. Metrics: add markout-at-Δ (vs future mid, since truth is static) and
   effective spread — both computable from existing `TradeRecord` fields.

---

## Entry 1 — Deferred-fill path (read-only survey for tick-time FBA fills)

Date: 2026-06-09. Question: how does trade recording handle fills produced at
`venue.tick()` time instead of at `submit_*` time? Short answer: **it doesn't.
There is no deferred-fill path anywhere in the repo, and no existing template
to copy.** Details per question below.

### 1. `MarketEnvironment._on_trade` — full body (verbatim)

`environment/market_environment.py` lines 104–153:

```python
    def _on_trade(self, sim: Simulator, event: Event) -> None:
        payload = event.payload
        if not isinstance(payload, TradeIntent):
            raise TypeError(
                f"trade event payload must be TradeIntent, got {type(payload).__name__}"
            )
        mid = payload.market_id
        if mid not in self.venues:
            raise KeyError(f"unknown market_id={mid}")
        venue = self.venues[mid]
        pre_mid = venue.get_state().mid_price

        aid = str(payload.agent_id)
        lim_px = payload.limit_price
        if payload.order_type == "market":
            res = venue.submit_market_order(aid, payload.side, payload.quantity)
            cap = self._capital_from_fill(
                payload.side, res.filled_quantity, res.avg_fill_price
            )
        elif payload.order_type == "limit":
            if lim_px is None:
                raise ValueError("limit_price required for limit orders")
            res = venue.submit_limit_order(
                aid, payload.side, payload.quantity, float(lim_px)
            )
            cap = committed_capital_limit(
                payload.side,
                payload.quantity,
                float(lim_px),
                self.margin,
                safety_margin=payload.capital_safety_margin,
            )
        else:
            raise ValueError(f"unsupported order_type {payload.order_type!r}")

        post_mid = venue.get_state().mid_price
        self.trade_log.append(
            TradeRecord(
                timestamp=sim.now,
                market_id=mid,
                agent_id=payload.agent_id,
                side=payload.side,
                quantity=res.filled_quantity,
                avg_fill_price=res.avg_fill_price,
                fees_paid=res.fees_paid,
                capital_committed=cap,
                mid_price_before=self._float_mid(pre_mid),
                mid_price_after=self._float_mid(post_mid),
            )
        )
```

**Who calls it, with what, when:**

- It is the registered handler for `MarketEnvironment.TRADE_EVENT = "trade"`
  (`register()`, line 59). Events of that type are scheduled exclusively by
  `AgentPopulation._on_decision` / `_on_review` / `_on_noise`
  (`environment/agent_population.py`), always with `delay=0` and
  `priority=EventPriority.TRADE` (=200), payload = a `TradeIntent`.
- **Order within a timestamp t:** `venue_clock` (priority −50, calls
  `tick()` on every venue) → SIGNAL (0) → DECISION (100) → TRADE (200).
  So `_on_trade` always runs *after* that timestamp's `tick()` and *before*
  the next one. For FBA semantics: every order submitted at t lands in the
  book after the t-clear; the earliest it can clear is `tick()` at t+1. This
  gives a natural "orders accumulate during [t, t+1), clear at t+1" batch
  cadence with zero scheduling changes.
- **It builds the `TradeRecord` itself from the synchronous `OrderResult`**
  returned by `submit_market_order` / `submit_limit_order`. It never receives
  a `TradeRecord`, and one record is appended per intent unconditionally —
  including `quantity=0.0, avg_fill_price=0.0` rows when a limit order rests
  entirely (no fill). There is a second, parallel path
  `execute_market_order(...)` (lines 70–102) used by tests/simple runners;
  identical recording logic, no TRADE_EVENT.

### 2. `OrderResult` and `TradeRecord` — full fields (verbatim)

`venues/base.py`:

```python
@dataclass
class OrderResult:
    """Result of submitting an order to a venue."""
    filled_quantity: float       # Quantity actually executed
    avg_fill_price: float        # Volume-weighted average fill price
    remaining_quantity: float    # Unfilled remainder (zero for market orders on AMMs)
    order_id: Optional[str]      # For tracking resting limit orders; None for market orders
    fees_paid: float
```

`environment/trade_records.py`:

```python
@dataclass(frozen=True)
class TradeRecord:
    """One executed trade (actual fills)."""

    timestamp: int
    market_id: int
    agent_id: int
    side: str
    quantity: float
    avg_fill_price: float
    fees_paid: float
    capital_committed: float
    mid_price_before: float
    mid_price_after: float
```

**No status field on either.** `OrderResult` has no pending/queued/accepted
enum. The only way to express "accepted, no fill yet" with the current shape
is the limit-order convention: `filled_quantity=0, remaining_quantity=qty,
order_id=<id>`. An FBA `submit_market_order` returning that shape is
*shape-legal* but semantically off (`order_id` is documented "None for market
orders"), and `_on_trade` will still log a zero-quantity TradeRecord with
`avg_fill_price=0.0` and (for market intents) `capital_committed=0.0` — so
the agent's `pending_cost` would be cleared without any capital being charged
(see `AgentPopulation._sync_costs`, which credits `capital_committed +
fees_paid` per record and zeroes `pending_cost`).

### 3. CLOB fill → TradeRecord trace; is there ANY deferred-fill precedent?

Trace for a market order today (all synchronous, single call stack):

1. Agent `decide/review/fire_noise` returns a `TradeIntent` →
   `AgentPopulation` schedules TRADE event at delay 0.
2. `_on_trade` pops it, captures `pre_mid`, calls
   `CLOB.submit_market_order(aid, side, qty)`.
3. `CLOB._execute_buy/_execute_sell` walks the opposite book immediately,
   mutating resting queues in place (`venues/clob.py` lines 208–278) and
   returns an `OrderResult` with the aggregate fill.
4. `_on_trade` captures `post_mid`, builds the one `TradeRecord` (taker side
   only), appends to `trade_log`.

**Is there any path where a fill is recorded outside the submit call? No —
and worse, the maker side of every fill is never recorded at all.** When an
incoming order consumes a resting limit, the CLOB just shrinks/pops the
`(agent_id, qty, order_id)` tuple in the deque:

```python
            ag, q, oid = dq[0]
            take = min(remaining, q)
            tot_cost += take * ap
            filled += take
            remaining -= take
            if take >= q - 1e-15:
                dq.popleft()
            else:
                dq[0] = (ag, q - take, oid)
```

The maker (`ag`) gets no `TradeRecord`, no callback, no notification — its
fill exists only implicitly as book-state mutation. The maker's TradeRecord
was written once, at limit-submit time, with whatever crossed immediately
(possibly 0) and `capital_committed` for the full resting quantity. Later
executions of that resting order are invisible to `trade_log`, to
`AgentPopulation._sync_costs`, and to `metrics/rent.py` PnL.

**Consequence: there is no existing template for deferred FBA fills.** The
hoped-for mechanism (resting-limit fills routed back later) does not exist —
maker fills are silently absorbed. Deferred fill recording is new plumbing,
not a copy job. (It also means the existing CLOB/hybrid PnL metrics already
undercount maker-side executions for the informed agents that post limits via
`route_log_space_trade`'s passive branch — a pre-existing caveat worth noting
when comparing FBA numbers against the incumbent mechanisms.)

### 4. `mid_price_before` / `mid_price_after` — capture points

Both are captured inside `_on_trade`, bracketing the synchronous submit:
`pre_mid = venue.get_state().mid_price` immediately before `submit_*`,
`post_mid` immediately after, same timestamp, same call stack (lines 114 and
139). So today "before" is simultaneously submit-time and fill-time because
the two coincide.

For an FBA fill created at `tick()` time this breaks both ways:

- If the TradeRecord keeps being written at submit time, `mid_before/after`
  bracket a no-op (order queued, book/quote unchanged) — markout off these
  rows is meaningless, and `mid_price_after` no longer reflects an execution.
- If fills are recorded at clear time, nothing currently captures a pre-clear
  mid: `tick()` is invoked by the sweep's `_pulse` handler (below), which
  takes no snapshots. Correct markout for FBA wants `mid_price_before` = the
  pre-clear quote at the batch timestamp and `mid_price_after` = post-clear
  (or the clearing price itself); both must be captured by whatever new code
  runs the clear — they cannot be reconstructed from the submit-time path.
- Also note `convergence.build_mid_trajectory_from_trades` keys the mid
  trajectory off `TradeRecord.mid_price_after` — zero-fill submit-time
  records with a stale/None-derived mid would pollute convergence metrics, so
  FBA should suppress submit-time records or ensure they never carry
  misleading `mid_price_after`.

### 5. Can `tick()` emit trades back to the environment? No.

`Venue.tick()` is declared `-> None` and every call site discards any return.
The only caller in the sweep is `_register_venue_clock` in
`simulation/sweep.py`:

```python
    def _pulse(_sim: Simulator, _event: Event) -> None:
        for v in market_env.venues.values():
            v.tick()
```

`MarketEnvironment` never calls `tick()` and has no handler for anything a
venue might produce; the environment learns about executions exclusively via
the `OrderResult` returned by `submit_*` inside `_on_trade` /
`execute_market_order`. Precedent: `HybridVenue.tick()` does real work
(cancels and re-posts LP ladders) but its effects are consumed only
implicitly, as changed book state on the next `get_state()` /
`estimate_impact()` — never as records.

**Options for surfacing tick-time FBA fills (for the next entry, not built):**

- (a) Fill buffer on the venue: FBA `tick()` clears the batch and stores
  per-agent fills (incl. maker/taker, clearing price, pre/post-clear mids);
  the environment (or a replacement `venue_clock` handler that knows about
  `market_env.trade_log`) drains the buffer into `TradeRecord`s each pulse.
  Keeps the `Venue` ABC unchanged; recording logic lives beside the existing
  `_on_trade`.
- (b) Change `tick()` (or add an FBA-specific `clear()`) to return fills, and
  register the venue clock through `MarketEnvironment` instead of the sweep's
  bare `_pulse`, so the env owns record construction (timestamp, capital,
  mids) exactly as it does in `_on_trade`.
- Either way, the deferred path must also reproduce the capital bookkeeping
  side of `_on_trade` (`_capital_from_fill` / `committed_capital_limit`) so
  `AgentPopulation._sync_costs` and `metrics/capital.py` see FBA fills, and
  must assign `timestamp = clear tick`, not submit tick.

---

## Entry 2 — maker-fill recording fix (implemented, not committed)

Date: 2026-06-09. Fixes the asymmetry found in Entry 1: resting limit orders
consumed by incoming orders left no maker-side `TradeRecord`. Purely a
recording change — matching and economic behavior verified unchanged.

### Design

- **`venues/base.py`** — new frozen dataclass `MakerFill(agent_id, side,
  quantity, price, order_id, mid_before, mid_after)` and a non-abstract
  `Venue.drain_maker_fills() -> list[MakerFill]` defaulting to `[]` (AMMs have
  no maker side; the `Venue` ABC's six abstract methods are unchanged).
- **`venues/clob.py`** — all four crossing loops (`_execute_buy`,
  `_execute_sell`, and the marketable portions of `_limit_buy`/`_limit_sell`)
  now append a `MakerFill` per resting order consumed, bracketing each
  individual consumption with the venue mid before/after (incl. level
  cleanup). Matching arithmetic is untouched — only bookkeeping lines added.
  Fills buffer in `self._maker_fills` until drained.
- **`venues/hybrid.py`** — delegates `drain_maker_fills` to the inner CLOB, so
  algorithmic-LP (`agent_id=-2`) executions are now recorded too.
- **`environment/trade_records.py`** — `TradeRecord` gains
  `liquidity: str = "taker"` (`"taker" | "maker"`; default keeps old
  constructors valid). New `CostEntry(timestamp, market_id, agent_id,
  capital_committed, fees_paid)`: exactly one per processed intent.
- **`environment/market_environment.py`** — gains `cost_log: list[CostEntry]`.
  `_on_trade` (and `execute_market_order`) now: (1) drain maker fills →
  one maker `TradeRecord` each, `fees_paid=0`, **`capital_committed=0`**
  (capital was committed at rest time — no double charge), maker legs appended
  before the aggregate taker record so the mid trajectory ends on the
  post-execution mid; (2) append a `CostEntry` for the intent (always, even
  zero-fill — preserves capital semantics exactly); (3) append the taker
  `TradeRecord` **only if `filled_quantity > 0`** — the unconditional
  zero-quantity rows are gone from `trade_log`.
- **`environment/agent_population.py`** — `_sync_costs` reconciles from
  `cost_log` instead of `trade_log`. Per intent it sees the same
  `capital_committed + fees_paid` amounts at the same times as before, so
  `deployed`/`pending_cost` trajectories are bit-identical; maker records
  cannot double-charge or spuriously zero `pending_cost`.
- **`metrics/capital.py` / `simulation/sweep.py`** — exhaustion metric now
  consumes `cost_log` (same capital-flow semantics as the old `trade_log`,
  including fully-resting limit commitments that no longer appear in
  `trade_log`). `rent_and_pnl` still consumes `trade_log` and therefore now
  sees maker-side PnL — the intended correction.

### Tests (`tests/test_maker_fill_recording.py`, all pass; suite 10/10)

- (a) market order hitting one resting limit → exactly TWO records:
  maker+taker, equal qty, same price (the resting limit price), opposite
  sides, same timestamp; maker `capital_committed == 0`; no zero-qty rows.
- (b) conservation: Σ recorded buy qty == Σ recorded sell qty across a mixed
  run (markets, crossing limits, resting limits, bootstrap maker `-1`).
- (c) partial fill: 4 of 10 filled → both sides record 4.0; resting remainder
  6.0 stays on the book; venue-level `remaining_quantity` contracts unchanged.
- (d) maker capital charged once: `deployed` rises by the rest-time commitment
  and does not move when the resting order is later filled; `pending_cost`
  cleared correctly; exactly one `CostEntry` per intent.
- Hybrid: LP ladder consumed by a taker → maker records under `agent_id=-2`,
  quantities summing to the taker fill.

### Before/after verification (seed 0, diverse/mid/low, until_ts=8000)

Same harness run on pre-change and post-change code (verbatim port of
`run_single_simulation`'s clob/hybrid branches, raw `trade_log` dumped):

| | clob | hybrid |
|---|---|---|
| final mids identical | yes (all 3 markets) | yes |
| agent `deployed`/`pending_cost` identical | yes | yes |
| taker records (qty>0) byte-identical | 30/30 | 13/13 |
| zero-qty rows | 1 → 0 (suppressed) | 3 → 0 |
| maker records added | +30 | +13 |
| recorded volume | 55.39 → 110.79 (+100.0%) | 32.50 → 65.01 (+100.0%) |
| buy qty == sell qty after | yes | yes |

Metric shifts (recording-driven only, execution identical):

- `n_trades`: 31 → 60 (clob), 16 → 26 (hybrid) — maker rows added, zero rows
  dropped.
- `informed_pnl_total` (hybrid): −6.878 → −7.614, `rent_efficiency_stable`
  −0.688 → −0.761 — informed agents' resting limits that got hit are now
  marked to terminal fair (previously invisible). On clob this seed, no
  informed resting order was hit, so PnL was unchanged.
- Convergence metrics unchanged on both mechanisms this seed (maker rows add
  intermediate mid snapshots at the same ticks; values can differ on other
  seeds — semantics, not behavior).
- `frac_informed_exhausted` unchanged (now fed from `cost_log`, which equals
  the old per-intent capital stream exactly).

Recorded volume exactly doubling is the expected signature: every execution
now has both sides on the tape. **FBA relevance:** the venue→environment fill
channel (`drain_maker_fills` + `_record_maker_fills`) is precisely the shape a
batch venue needs — FBA `tick()` can buffer batch-clear fills (maker AND
taker) and the environment drains them into `trade_log` the same way; and
`liquidity` tagging plus per-leg mid bracketing makes maker/taker markout
well-defined on all venues. Not committed.

---

## Entry 3 — FBA venue (implemented, not committed; sweep untouched)

Date: 2026-06-10. New `venues/fba.py` (`FBAVenue`) implementing the `Venue`
ABC as a periodic uniform-price call auction over a resting limit book, plus
`tests/test_fba_venue.py` (12 tests). The incumbent venues' matching code is
untouched; the only shared-file changes extend the Entry-2 deferred-fill
channel (details below). The sweep was NOT run and gains no `fba` mechanism
yet — that's Entry 4 work.

### Cadence and submit semantics

- `FBAVenue(tau_ticks: int, *, fee_rate: float = 0.0)`. `tick()` increments a
  counter; when `counter % tau_ticks == 0` the batch clears inside `tick()`.
  No scheduler added: the sweep's venue clock (priority −50, before
  SIGNAL/DECISION/TRADE) already gives "submits at t land in t+1's clear" for
  free (verified in the env-wiring test: submit at t=1, records stamped t=2).
- `submit_limit_order` / `submit_market_order` NEVER fill synchronously. Both
  return the Entry-1 "accepted, pending" shape:
  `OrderResult(filled_quantity=0, avg_fill_price=0, remaining_quantity=qty,
  order_id=<real id>, fees_paid=0)`. **Deliberate deviation:** market orders
  get a real `order_id` (the `OrderResult` docstring says "None for market
  orders") because FBA market orders live until the next clear and their
  clear-time fills need attribution. Shape-legal; `_on_trade` ignores the id
  for market intents.
- Resting limits persist across clears until filled/cancelled; partial fills
  shrink `qty` in place and keep resting. Market orders queue for exactly one
  clear at `+inf` (buy) / `0.0` (sell) — the sim's open price domain, so they
  are infinitely elastic — and expire silently if unfilled (nothing recorded,
  nothing persists). `cancel_order` removes a resting limit; standard.

### Clearing (native reimpl of batch_counterfactual/auction.py)

- Candidate prices = distinct resting/arriving **limit** prices (market
  orders participate at every price but define no candidates — with both
  D and S step functions breaking only at limit prices, interior volumes
  never exceed candidate volumes, so scanning candidates finds the max).
- Objective: max executable volume `min(D(p), S(p))`; flat max-volume set →
  clear at the **midpoint** of `[min, max]` of the optimal prices
  (auction.py ASSUMPTION-1; no tick quantization here — prices are floats,
  so the raw midpoint is used). All fills in a batch execute at exactly p*.
- Rationing: participation recomputed at p*; short side fills fully, long
  side pro-rata with **largest-remainder** rounding, ties by submission
  order (auction.py ASSUMPTION-2, `_largest_remainder` ported as exact
  integer arithmetic). To make that exact in float-land, all quantities are
  quantized to integer multiples of 1e-9 units (`QUANTITY_SCALE = 10**9`) at
  submit; rationing is pure `int` math — deterministic (no RNG anywhere) and
  conserving: per-clear Σbuy_qty == Σsell_qty to the quantum. This quantum
  grid is the one place the float-quantity convention forced a change vs
  auction.py's integral `Decimal` contracts.

### Resting-book reuse vs mirror

`CLOB`'s book is price-level `dict[float, deque[(agent_id, qty, order_id)]]`
plus sorted price lists — shaped for price-time-priority *continuous*
crossing, which FBA doesn't do (no time priority inside a batch; pro-rata at
the margin). Reusing it wholesale would have meant carrying machinery the
auction never uses, so the book is **mirrored, not imported**: a flat
insertion-ordered `dict[order_id -> _BatchOrder(agent_id, side, price, qty_q,
seq, is_market)]`. No new price type was invented: prices go through the
CLOB's own `_pkey` (imported from `venues/clob.py`), and `EmptyBookError` is
imported for `estimate_impact` parity. One consequence of the flat layout:
best-bid/ask are computed by scan rather than kept sorted (fine at sim scale;
the CLOB's sorted-level structure would be the optimization if ever needed).
A `seed_initial_book` with the CLOB's exact ladder shape is provided for the
Entry-4 sweep builder.

### Deferred-fill recording — drain wiring chosen

Chose **extending the Entry-2 channel** over a new `drain_fills()` method:

- `venues/base.py`: `MakerFill` gains `liquidity: str = "maker"` and
  `fees_paid: float = 0.0`. Defaults keep every existing CLOB/hybrid
  construction site and consumer valid — for continuous venues nothing
  changes byte-wise. FBA emits BOTH legs of every clear through
  `drain_maker_fills()`, tagged `"maker"` (came from a limit) or `"taker"`
  (market order); the name now reads as "drain deferred fills" but was kept
  to avoid churning the ABC and hybrid's delegation.
- `environment/market_environment.py`:
  `_record_maker_fills` → `_record_drained_fills`, generalized: maker legs
  keep the Entry-2 convention (`capital_committed=0`, capital was charged by
  the submit-time `CostEntry` via `committed_capital_limit`); **taker legs
  charge capital at clear** — `_capital_from_fill(side, qty, p*)` plus a
  `CostEntry` at the clear tick, since the submit-time market-intent
  `CostEntry` was computed off `filled_quantity=0` and charged nothing.
  Fees: taker legs pay `fee_rate × qty × p*` (default 0, matching the
  fee-free CLOB); maker legs pay 0 — keeps `trade_log` fees consistent with
  `cost_log` (which only taker legs touch at clear).
- New public `MarketEnvironment.drain_venue_fills(sim)`: loops venues and
  records buffered fills. This is what must run **right after the venue
  clock pulse** so records get `timestamp = clear tick`. Entry-4 wiring is a
  one-line change to the sweep's `_pulse`:
  `v.tick()` for all venues, then `market_env.drain_venue_fills(sim)`.
  (`_on_trade` also still drains after each submit — harmless for FBA since
  submits buffer nothing, and it keeps CLOB/hybrid maker recording exactly
  where it was.)

### Mids / markout validity

`_on_trade`'s `mid_price_before/after` bracket a synchronous submit, which
for FBA brackets a no-op. The venue therefore captures the **pre-clear book
mid** (resting best-bid/ask midpoint before fills are applied — note an FBA
book may be legitimately crossed between clears, giving a well-defined mid
with negative spread) and the **post-clear mid** (after fills/removals), and
stamps both on every `MakerFill` in the batch. Test (h) constructs a case
where submit-time mid = 102, pre-clear mid = 101 (crossed book), post-clear
mid = 102, and asserts the stamped values are the clear-time ones — post-hoc
markout off FBA rows is not a no-op. Since FBA submits never fill,
`_on_trade` appends no taker row for them (`filled_quantity=0` suppression
from Entry 2), so no submit-time rows pollute
`build_mid_trajectory_from_trades`.

### ABC completeness

- `get_state()`: mid/best-bid/best-ask/spread from the resting book between
  clears (agents observe this; spread can be negative when crossed).
- `estimate_impact(side, qty)`: simulates adding a market order of `qty` to
  the current pending state (resting limits + queued market orders) and
  returns the hypothetical clearing price p* — the honest "what price would
  I get", and the VWAP, since every fill executes at p*. Edge conventions
  mirror the CLOB: `EmptyBookError` when no opposite resting limits;
  `inf` (buy) / `0.0` (sell) when the probe can't fully fill.

### Tests (tests/test_fba_venue.py — 12, all pass; suite 22/22)

- (a) hand-computed clear: D = {10@101, 5@100}, S = {8@99, 4@100} → unique
  max volume 12 at p* = 100; buys rationed 8/4, sells full — asserted with
  `==`, no tolerance.
- (b) uniform price: single price per batch; plus the midpoint tie-break
  case (buy 5@102 vs sell 5@98 → flat interval [98,102] → p* = 100 exactly).
- (c) conservation: Σbuy == Σsell within each clear, including a pro-rata
  rationing clear with awkward decimals (3.3/1.4/2.6 vs 5.0) and a
  multi-price clear.
- (d) pending semantics: both submit shapes return filled=0/remaining=qty
  with real order ids; drains empty at counters 1,2 (tau=3); both legs
  appear exactly at counter % tau == 0 with matching order ids.
- (e) resting persistence: unfilled limit survives an empty clear; a 4-of-10
  partial leaves exactly 6 resting (checked in quanta) and fills at a later
  clear.
- (f) market expiry: market buy with no opposite interest records nothing
  and does not haunt later clears.
- (g) determinism: identical seeded order streams (40 orders, mixed
  limit/market, interleaved clears) → field-identical fill lists.
- (h) mids: as above — clear-time pre/post mids stamped, distinct from the
  submit-time bracket.
- (i) tau sanity: tau_ticks=1 clears on the first tick with well-formed
  maker/taker records, drained exactly once.
- Plus: `estimate_impact` returns the hypothetical p* / inf / EmptyBookError;
  and the env-wiring test mirrors the sweep's priority −50 pulse + drain,
  asserting timestamp = clear tick (submit t=1 → records t=2), maker
  capital 0 at clear, taker capital `qty·p*` at clear, and the exact
  `cost_log` sequence [(1,maker-limit), (1,market-submit, 0 capital),
  (2,taker-clear)].

Existing suite unchanged and green (the `MakerFill`/environment extensions
are default-compatible: CLOB/hybrid records are byte-identical). Sweep not
run; nothing committed.
