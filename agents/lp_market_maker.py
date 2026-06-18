"""Two-sided liquidity provider that rests pickable quotes and is adversely selected.

The §5.4 quoting role: the inter-agent fill channel that latency/information
differentiation (§5.2) needs. Today informed/noise agents trade almost entirely
as takers against the static bootstrap ladder (agent ``-1``), so informed-as-maker
fills are ~0 and reordering fast-vs-slow changes nothing. This agent makes the
bleeding LP an explicit object: it quotes a two-sided spread *inside* the
bootstrap (so it owns price priority and gets hit first), updates its fair-value
belief on a deliberately LONG ``observation_delay`` (staler than the FAST
informed role), and is picked off by faster, better-informed takers.

Mechanism (Option C, confirmed): the LP manages its quotes **directly on the
venue inside** ``review()`` — mirroring how the bootstrap ladder posts via
``CLOB.seed_initial_book`` — because the ``TradeIntent`` path never returns
``order_id``s, so cancel-and-replace is impossible through it. ``review()``
returns ``[]``; ``decide()`` only updates belief (no trade).

Capital is **net-inventory margin, isolated to this class**: the LP reconstructs
its signed net position from its own fills in ``MarketEnvironment.trade_log``
(read-only) and margins the NET position, not gross legs. Cancel-and-requote
therefore does not exhaust it (the failure mode of the shared monotonic-``deployed``
model). It never writes to ``cost_log`` / the shared ``_sync_costs`` path — its
maker fills carry ``capital_committed=0`` — so zero-delay incumbent baselines stay
byte-identical.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

from environment.market_environment import MarketEnvironment
from environment.signals import Signal
from environment.simulator import Simulator
from environment.trade_records import TradeIntent

from agents.belief_utils import gaussian_scalar_nif_update


@dataclass
class LpMarketMakerAgent:
    """Two-sided market maker quoting inside the bootstrap on a stale belief.

    Satisfies the ``PopulationAgent`` Protocol. It is neither fast- nor
    slow-informed nor noise: it gets its own ``ROLE_LP`` PnL bucket so its
    adverse-selection losses report separately.
    """

    agent_id: int
    budget: float
    market_ids: tuple[int, ...]
    initial_log_fair_mean: dict[int, float]
    #: Deliberately LONGER than the FAST informed delay (§5.2): the LP's belief
    #: updates late, so its resting quotes reflect staler information.
    observation_delay: int = 50
    review_interval: int = 500
    #: Half-spread δ as a fraction of belief fair value. Must be < the innermost
    #: bootstrap step (0.001) so the quotes sit *inside* the ladder and own
    #: price priority. The §5.4 endogenous-spread arm will later make this knob
    #: respond to realized markout; for now it is fixed config.
    half_spread_pct: float = 0.0005
    quote_size: float = 4.0
    prior_precision: float = 1.0
    signal_precision_assumed: float = 0.7

    # Protocol fields the LP does not use through the belief/noise paths.
    arrival_rate_per_unit: float = field(default=0.0, init=False)
    deployed: float = field(default=0.0, init=False)
    pending_cost: float = field(default=0.0, init=False)

    # Internal state.
    _mean_log: dict[int, float] = field(default_factory=dict, init=False)
    _prec: dict[int, float] = field(default_factory=dict, init=False)
    _net_pos: dict[int, float] = field(default_factory=dict, init=False)
    #: market_id -> (bid_order_id, ask_order_id) currently resting on the venue.
    _resting: dict[int, tuple[str | None, str | None]] = field(
        default_factory=dict, init=False
    )
    #: cursor into MarketEnvironment.trade_log for incremental inventory updates.
    _tl_cursor: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        if not self.market_ids:
            raise ValueError("market_ids must be non-empty")
        if self.observation_delay < 0:
            raise ValueError("observation_delay must be non-negative")
        if self.review_interval <= 0:
            raise ValueError("review_interval must be positive (LP must requote)")
        if not (0.0 < self.half_spread_pct < 0.001):
            raise ValueError(
                "half_spread_pct must be in (0, 0.001) to quote inside the bootstrap"
            )
        if self.quote_size <= 0:
            raise ValueError("quote_size must be positive")
        if self.prior_precision <= 0:
            raise ValueError("prior_precision must be positive")
        if self.signal_precision_assumed <= 0:
            raise ValueError("signal_precision_assumed must be positive")
        self.market_ids = tuple(self.market_ids)
        for m in self.market_ids:
            if m not in self.initial_log_fair_mean:
                raise ValueError(f"initial_log_fair_mean missing market {m}")
            self._mean_log[m] = float(self.initial_log_fair_mean[m])
            self._prec[m] = float(self.prior_precision)
            self._net_pos[m] = 0.0

    @property
    def available(self) -> float:
        return self.budget - self.deployed - self.pending_cost

    # --- belief path (decide only updates; the LP never takes) ----------- #
    def observes(self, market_id: int) -> bool:
        return market_id in self._mean_log

    def _update_posterior(self, signal: Signal) -> None:
        m = signal.market_id
        if m not in self._mean_log:
            return
        mu, pr = gaussian_scalar_nif_update(
            self._mean_log[m], self._prec[m], signal.value, self.signal_precision_assumed
        )
        self._mean_log[m] = mu
        self._prec[m] = pr

    def decide(
        self, sim: Simulator, signal: Signal, market_env: MarketEnvironment
    ) -> TradeIntent | None:
        self._update_posterior(signal)
        return None

    def fire_noise(
        self, sim: Simulator, market_env: MarketEnvironment
    ) -> TradeIntent | None:
        return None

    # --- net-inventory capital (isolated; reads trade_log read-only) ----- #
    def _update_inventory(self, market_env: MarketEnvironment) -> None:
        """Fold this agent's new fills from the shared trade_log into net_pos.

        Maker-side semantics: a resting buy that is hit tapes ``side="buy"``
        (LP bought); a resting sell that is hit tapes ``side="sell"`` (LP sold).
        """
        tl = market_env.trade_log
        for r in tl[self._tl_cursor :]:
            if r.agent_id != self.agent_id:
                continue
            signed = r.quantity if r.side == "buy" else -r.quantity
            self._net_pos[r.market_id] = self._net_pos.get(r.market_id, 0.0) + signed
        self._tl_cursor = len(tl)

    def _capital_used(self, market_env: MarketEnvironment) -> float:
        """Collateral on the NET position only (not gross legs)."""
        margin = market_env.margin
        total = 0.0
        for m, pos in self._net_pos.items():
            if pos == 0.0:
                continue
            fair = math.exp(self._mean_log[m])
            frac = (
                margin.long_margin_fraction
                if pos > 0
                else margin.short_margin_fraction
            )
            total += abs(pos) * fair * frac
        return total

    # --- quoting (direct venue management; mirrors the bootstrap) -------- #
    def review(
        self, sim: Simulator, market_env: MarketEnvironment
    ) -> list[TradeIntent]:
        self._update_inventory(market_env)
        aid = str(self.agent_id)

        for m in self.market_ids:
            venue = market_env.venue(m)

            # Cancel prior quotes first so net-inventory accounting holds and
            # the book is not littered with stale orders.
            prev = self._resting.get(m)
            if prev is not None:
                bid_oid, ask_oid = prev
                if bid_oid is not None:
                    venue.cancel_order(aid, bid_oid)
                if ask_oid is not None:
                    venue.cancel_order(aid, ask_oid)
                self._resting[m] = None

            # Net-inventory solvency gate: no room to take on more inventory
            # risk -> sit out this market this cycle (can resume when the net
            # position mean-reverts). This is what keeps the LP from going
            # silent the way the gross monotonic model would force.
            if self._capital_used(market_env) >= self.budget:
                continue

            fair = math.exp(self._mean_log[m])
            bid_px = fair * (1.0 - self.half_spread_pct)
            ask_px = fair * (1.0 + self.half_spread_pct)

            # Safety: keep quotes resting (non-marketable) and uncrossed. A
            # marketable LP limit would fill as a taker, and the taker leg is
            # invisible to trade_log (only the discarded OrderResult) -> would
            # corrupt inventory. Clamp just inside the current best opposite.
            st = venue.get_state()
            if st.best_ask is not None:
                bid_px = min(bid_px, float(st.best_ask) * (1.0 - 1e-6))
            if st.best_bid is not None:
                ask_px = max(ask_px, float(st.best_bid) * (1.0 + 1e-6))
            if bid_px <= 0.0 or ask_px <= bid_px:
                continue

            bres = venue.submit_limit_order(aid, "buy", self.quote_size, bid_px)
            ares = venue.submit_limit_order(aid, "sell", self.quote_size, ask_px)
            self._resting[m] = (bres.order_id, ares.order_id)

        self.deployed = self._capital_used(market_env)
        return []


__all__ = ["LpMarketMakerAgent"]
