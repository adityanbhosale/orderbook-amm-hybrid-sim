"""
Parameter sweep harness for multi-agent × multi-market simulations.

Mechanism axis: ``amm`` | ``clob`` | ``hybrid`` (constant-product AMM, plain
CLOB, or CLOB plus passive hybrid LP quotes).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Sequence

import numpy as np
import pandas as pd

from agents.event_noise import EventDrivenNoiseAgent
from agents.informed_aggregated import AggregatedEvidenceAgent
from agents.informed_joint_factor import (
    JointFactorFairValueAgent,
    make_joint_factor_agent,
)
from agents.informed_naive import NaiveGaussianBeliefAgent
from agents.informed_tail import TailAwareGaussianBeliefAgent
from agents.lp_market_maker import LpMarketMakerAgent
from environment import (
    AgentPopulation,
    InformationConfig,
    InformationEnvironment,
    MarginSpec,
    MarketEnvironment,
    Simulator,
)
from environment.events import Event
from environment.agent_population import PopulationAgent
from environment.cross_market import cross_weights_from_loadings
from environment.information import ClusterSpec
from environment.information_helpers import base_log_levels_from_truth
from metrics.capital import fraction_exhausted_before_convergence
from metrics.convergence import build_mid_trajectory_from_trades, convergence_metrics
from metrics.rent import frozen_fair_value, pnl_by_role, rent_and_pnl
from venues.clob import CLOB
from venues.constant_product import ConstantProductAMM
from venues.hybrid import HybridLpConfig, HybridVenue

MechanismName = Literal["amm", "clob", "hybrid"]
MixName = Literal["diverse", "naive_dominated", "lp_vs_informed"]
SignalRegimeName = Literal["low", "high"]
CapitalBandName = Literal["low", "mid", "high"]

NOISE_AGENT_ID = 99
LP_AGENT_ID = 50

CAPITAL_BANDS: dict[CapitalBandName, float] = {
    "low": 100.0,
    "mid": 1000.0,
    "high": 10_000.0,
}

SIGNAL_REGIMES: dict[SignalRegimeName, dict[str, float]] = {
    "low": {"signal_noise_std": 0.012, "tail_noise_std": 0.005},
    "high": {"signal_noise_std": 0.055, "tail_noise_std": 0.022},
}

ANCHOR_DISPLACEMENT_STD = 0.01

#: Latency role map. FAST agents update beliefs (and act) on a shorter
#: observation_delay than SLOW agents; NOISE is untouched (Poisson path, no
#: observation_delay). Used both for delay assignment and PnL bucketing.
FAST_ROLE_CLASSES = (TailAwareGaussianBeliefAgent, JointFactorFairValueAgent)
SLOW_ROLE_CLASSES = (NaiveGaussianBeliefAgent, AggregatedEvidenceAgent)

ROLE_FAST = "fast_informed"
ROLE_SLOW = "slow_informed"
ROLE_NOISE = "noise"
#: The §5.4 liquidity provider. Neither fast/slow-informed nor noise: it quotes
#: two-sided and is adversely selected, so its PnL is bucketed separately.
ROLE_LP = "lp"


@dataclass(frozen=True)
class RoleDelayConfig:
    """Per-role ``observation_delay`` in integer ticks.

    Default ``fast=slow=0`` reproduces the pre-latency baseline exactly (every
    informed agent already constructed with ``observation_delay=0``).
    Differentiated delays are an opt-in axis: set ``fast < slow`` to give the
    fast role a timing-of-belief-update edge. ``EventDrivenNoiseAgent`` never
    receives a delay (it has no ``observation_delay`` field).
    """

    fast: int = 0
    slow: int = 0

    def __post_init__(self) -> None:
        if self.fast < 0 or self.slow < 0:
            raise ValueError("role delays must be non-negative")

    def delay_for_class(self, agent_obj: object) -> int:
        if isinstance(agent_obj, FAST_ROLE_CLASSES):
            return self.fast
        if isinstance(agent_obj, SLOW_ROLE_CLASSES):
            return self.slow
        return 0


def _role_by_agent_id(agents: Sequence[PopulationAgent]) -> dict[int, str]:
    """Map each agent_id to its latency role label for PnL bucketing."""
    roles: dict[int, str] = {}
    for a in agents:
        if isinstance(a, FAST_ROLE_CLASSES):
            roles[a.agent_id] = ROLE_FAST
        elif isinstance(a, SLOW_ROLE_CLASSES):
            roles[a.agent_id] = ROLE_SLOW
        elif isinstance(a, EventDrivenNoiseAgent):
            roles[a.agent_id] = ROLE_NOISE
        elif isinstance(a, LpMarketMakerAgent):
            roles[a.agent_id] = ROLE_LP
    return roles


def _anchor_displacements(
    anchor_rng: np.random.Generator, n_markets: int
) -> np.ndarray:
    """Per-market N(0, std) shocks; drawn from seed-dedicated RNG before agents."""
    return anchor_rng.normal(0.0, ANCHOR_DISPLACEMENT_STD, size=n_markets)


def _anchor_prices_by_market(
    fair_arr: np.ndarray, displacements: np.ndarray
) -> dict[int, float]:
    return {
        m: float(fair_arr[m] * (1.0 + displacements[m])) for m in range(len(fair_arr))
    }


def _default_information_config(
    *,
    signal_noise_std: float,
    tail_noise_std: float,
    n_markets: int = 3,
    k: int = 3,
) -> InformationConfig:
    assert n_markets == 3 and k == 3
    clusters = [
        ClusterSpec(primary_factor=0, market_count=1),
        ClusterSpec(primary_factor=1, market_count=1),
        ClusterSpec(primary_factor=2, market_count=1),
    ]
    mids = np.full(n_markets, 100.0, dtype=float)
    return InformationConfig(
        k=k,
        clusters=clusters,
        n_independent_markets=0,
        independent_loading_std=0.35,
        idiosyncratic_std=0.12,
        signal_noise_std=signal_noise_std,
        tail_noise_std=tail_noise_std,
        routine_rate_per_market=0.75,
        tail_rate_per_market=0.035,
        tail_mode="marked",
        initial_mid_prices=mids,
    )


def _venues_from_truth(
    info_env: InformationEnvironment,
    reserve_x: float,
    anchor_prices: dict[int, float],
) -> dict[int, ConstantProductAMM]:
    venues: dict[int, ConstantProductAMM] = {}
    for t in info_env.world.truths:
        m = t.market_id
        anchor = anchor_prices[m]
        x = reserve_x
        y = reserve_x * anchor
        venues[m] = ConstantProductAMM(float(x), float(y))
    return venues


def _clob_venues_from_truth(
    info_env: InformationEnvironment,
    reserve_x: float,
    anchor_prices: dict[int, float],
) -> dict[int, CLOB]:
    depth = 0.005 * float(reserve_x)
    out: dict[int, CLOB] = {}
    for t in info_env.world.truths:
        m = t.market_id
        clob = CLOB()
        clob.seed_initial_book(anchor_prices[m], depth)
        out[m] = clob
    return out


def _hybrid_venues_from_truth(
    info_env: InformationEnvironment,
    reserve_x: float,
    anchor_prices: dict[int, float],
    *,
    lp_spread_pct: float = 0.005,
    lp_decay_factor: float = 0.7,
    lp_levels: int = 3,
) -> dict[int, HybridVenue]:
    out: dict[int, HybridVenue] = {}
    for t in info_env.world.truths:
        m = t.market_id
        lp = HybridLpConfig(
            lp_spread_pct=lp_spread_pct,
            lp_base_qty=0.01 * float(reserve_x),
            lp_decay_factor=lp_decay_factor,
            lp_anchor_price=anchor_prices[m],
            lp_levels=lp_levels,
        )
        out[m] = HybridVenue(lp=lp)
    return out


def _safe_mid(market_env: MarketEnvironment, m: int, fallback: float) -> float:
    mp = market_env.mid_price(m)
    if mp is None:
        return float(fallback)
    return float(mp)


def _opening_log_means(
    market_env: MarketEnvironment, info_env: InformationEnvironment, n_markets: int
) -> dict[int, float]:
    out: dict[int, float] = {}
    for m in range(n_markets):
        mp = market_env.mid_price(m)
        if mp is not None:
            out[m] = math.log(mp)
        else:
            out[m] = math.log(float(info_env.world.truths[m].fair_price))
    return out


def _register_venue_clock(
    sim: Simulator, market_env: MarketEnvironment, until_ts: int
) -> None:
    """Advance every venue once per integer timestamp (hybrid LP refresh)."""

    def _pulse(_sim: Simulator, _event: Event) -> None:
        for v in market_env.venues.values():
            v.tick()

    sim.register_handler("venue_clock", _pulse)
    for t in range(0, until_ts + 1):
        sim.schedule_at(t, "venue_clock", priority=-50)



def _trade_size_for_budget(budget: float) -> float:
    return float(max(0.4, min(12.0, budget / 120.0)))


def build_agents_diverse(
    budget: float,
    market_env: MarketEnvironment,
    info_env: InformationEnvironment,
    loadings_matrix: np.ndarray,
    cross_weights: dict[tuple[int, int], float],
    agent_rng: np.random.Generator,
    n_markets: int,
    *,
    delays: RoleDelayConfig | None = None,
) -> list[PopulationAgent]:
    delays = delays or RoleDelayConfig()
    init_log = _opening_log_means(market_env, info_env, n_markets)
    ts = _trade_size_for_budget(budget)
    rev = 5000
    safety = 1.08
    thresh = 0.0035

    tail_bases = base_log_levels_from_truth(
        info_env, tuple(range(n_markets)), agent_rng, noise_std=0.025
    )
    mids_map = {
        m: float(market_env.mid_price(m) or info_env.world.truths[m].fair_price)
        for m in range(n_markets)
    }

    naive = NaiveGaussianBeliefAgent(
        agent_id=1,
        budget=budget,
        market_ids=tuple(range(n_markets)),
        initial_log_fair_mean=init_log.copy(),
        observation_delay=delays.slow,
        review_interval=rev,
        prior_precision=3.0,
        signal_precision_assumed=0.55,
        disagreement_threshold_log=thresh,
        trade_size=ts,
        safety_margin=safety,
    )
    tail = TailAwareGaussianBeliefAgent(
        agent_id=2,
        budget=budget,
        market_ids=tuple(range(n_markets)),
        base_log_levels=tail_bases,
        observation_delay=delays.fast,
        review_interval=rev,
        prior_precision=1.2,
        disagreement_threshold_log=thresh,
        trade_size=ts,
        safety_margin=safety,
    )
    agg = AggregatedEvidenceAgent(
        agent_id=3,
        budget=budget,
        market_ids=tuple(range(n_markets)),
        observed_markets=tuple(range(n_markets)),
        cross_weights=cross_weights,
        initial_log_fair_mean=init_log.copy(),
        observation_delay=delays.slow,
        review_interval=rev,
        prior_precision=2.0,
        signal_precision_assumed=0.85,
        disagreement_threshold_log=thresh,
        trade_size=ts,
        safety_margin=safety,
    )
    joint = make_joint_factor_agent(
        agent_id=4,
        budget=budget,
        primary_markets=tuple(range(n_markets)),
        observed_markets=tuple(range(n_markets)),
        loadings_matrix=loadings_matrix,
        initial_mid_by_market=mids_map,
        observation_delay=delays.fast,
        review_interval=rev,
        prior_precision_scale=1.0,
        signal_noise_inflation=1.0,
        disagreement_threshold_log=thresh,
        trade_size=ts,
        safety_margin=safety,
    )
    noise = EventDrivenNoiseAgent(
        agent_id=NOISE_AGENT_ID,
        budget=budget,
        market_ids=tuple(range(n_markets)),
        arrival_rate_per_unit=1.2,
        mean_trade_size=max(0.8, ts * 0.6),
        size_jitter=0.45,
        safety_margin=safety,
    )
    return [naive, tail, agg, joint, noise]


def build_agents_naive_dominated(
    budget: float,
    market_env: MarketEnvironment,
    info_env: InformationEnvironment,
    n_markets: int,
    n_naive: int = 5,
    *,
    delays: RoleDelayConfig | None = None,
) -> list[PopulationAgent]:
    delays = delays or RoleDelayConfig()
    init_log = _opening_log_means(market_env, info_env, n_markets)
    ts = _trade_size_for_budget(budget)
    rev = 4500
    safety = 1.08
    thresh = 0.0035
    agents: list[PopulationAgent] = []
    for i in range(1, n_naive + 1):
        agents.append(
            NaiveGaussianBeliefAgent(
                agent_id=i,
                budget=budget,
                market_ids=tuple(range(n_markets)),
                initial_log_fair_mean=init_log.copy(),
                observation_delay=delays.slow,
                review_interval=rev,
                prior_precision=2.5 + 0.1 * i,
                signal_precision_assumed=0.5,
                disagreement_threshold_log=thresh,
                trade_size=ts,
                safety_margin=safety,
            )
        )
    agents.append(
        EventDrivenNoiseAgent(
            agent_id=NOISE_AGENT_ID,
            budget=budget,
            market_ids=tuple(range(n_markets)),
            arrival_rate_per_unit=1.3,
            mean_trade_size=max(0.8, ts * 0.55),
            size_jitter=0.5,
            safety_margin=safety,
        )
    )
    return agents


def build_agents_lp_vs_informed(
    budget: float,
    market_env: MarketEnvironment,
    info_env: InformationEnvironment,
    loadings_matrix: np.ndarray,
    cross_weights: dict[tuple[int, int], float],
    agent_rng: np.random.Generator,
    n_markets: int,
    *,
    delays: RoleDelayConfig | None = None,
    lp_observation_delay: int = 50,
    lp_half_spread_pct: float = 0.0005,
    lp_quote_size: float | None = None,
    lp_budget: float = 20_000.0,
) -> list[PopulationAgent]:
    """The known diverse informed population PLUS one two-sided LP.

    Reuses ``build_agents_diverse`` verbatim for the informed/noise agents (ids
    1-4, 99) so this is a clean "diverse + a bleeding LP" world for the τ-curve,
    then adds ``LpMarketMakerAgent`` at ``LP_AGENT_ID``. The LP's
    ``observation_delay`` is its own knob (NOT part of ``RoleDelayConfig``) and
    defaults LONGER than the FAST informed delay so its quotes are staler (§5.2).
    """
    delays = delays or RoleDelayConfig()
    base = build_agents_diverse(
        budget,
        market_env,
        info_env,
        loadings_matrix,
        cross_weights,
        agent_rng,
        n_markets,
        delays=delays,
    )
    init_log = _opening_log_means(market_env, info_env, n_markets)
    ts = _trade_size_for_budget(budget)
    quote = lp_quote_size if lp_quote_size is not None else max(2.0, ts)
    lp = LpMarketMakerAgent(
        agent_id=LP_AGENT_ID,
        budget=lp_budget,
        market_ids=tuple(range(n_markets)),
        initial_log_fair_mean=init_log.copy(),
        observation_delay=lp_observation_delay,
        review_interval=500,
        half_spread_pct=lp_half_spread_pct,
        quote_size=quote,
    )
    return [*base, lp]


def _informed_ids_for_mix(mix: MixName, n_naive: int = 5) -> set[int]:
    if mix in ("diverse", "lp_vs_informed"):
        return {1, 2, 3, 4}
    return set(range(1, n_naive + 1))


def run_single_simulation(
    *,
    seed: int,
    mechanism: MechanismName,
    mix: MixName,
    capital_band: CapitalBandName,
    signal_regime: SignalRegimeName,
    until_ts: int = 40_000,
    time_resolution: int = 1000,
    reserve_x: float = 8000.0,
    rel_convergence_band: float = 0.02,
    naive_dominated_count: int = 5,
    observation_delays: RoleDelayConfig | None = None,
    lp_observation_delay: int = 50,
    lp_half_spread_pct: float = 0.0005,
    lp_quote_size: float | None = None,
    lp_budget: float = 20_000.0,
) -> dict[str, Any]:
    if mechanism not in ("amm", "clob", "hybrid"):
        raise ValueError(f"unknown mechanism {mechanism!r}")
    delays = observation_delays or RoleDelayConfig()
    budget = CAPITAL_BANDS[capital_band]
    sig = SIGNAL_REGIMES[signal_regime]

    seq = np.random.SeedSequence(seed)
    child_sim, child_world, child_agent, child_anchor = seq.spawn(4)
    sim_rng = np.random.default_rng(child_sim)
    world_rng = np.random.default_rng(child_world)
    agent_rng = np.random.default_rng(child_agent)
    anchor_rng = np.random.default_rng(child_anchor)

    cfg = _default_information_config(
        signal_noise_std=sig["signal_noise_std"],
        tail_noise_std=sig["tail_noise_std"],
    )
    info_env = InformationEnvironment(cfg, world_rng)
    sim = Simulator(rng=sim_rng, time_resolution=time_resolution)
    n_markets = info_env.n_markets
    fair_arr = np.array(
        [float(info_env.world.truths[m].fair_price) for m in range(n_markets)],
        dtype=float,
    )
    displacements = _anchor_displacements(anchor_rng, n_markets)
    anchor_prices = _anchor_prices_by_market(fair_arr, displacements)

    if mechanism == "amm":
        venues_any = _venues_from_truth(info_env, reserve_x, anchor_prices)
    elif mechanism == "clob":
        venues_any = _clob_venues_from_truth(info_env, reserve_x, anchor_prices)
    else:
        venues_any = _hybrid_venues_from_truth(info_env, reserve_x, anchor_prices)

    margin = MarginSpec(long_margin_fraction=1.0, short_margin_fraction=1.0)
    market_env = MarketEnvironment(venues_any, margin=margin)
    market_env.register(sim)
    info_env.schedule_signals(sim, until_ts)

    loadings_matrix = info_env.world.loadings_matrix
    cw = cross_weights_from_loadings(
        loadings_matrix,
        primary_markets=tuple(range(n_markets)),
        observed_markets=tuple(range(n_markets)),
        min_weight=0.04,
    )

    if mix == "diverse":
        agents = build_agents_diverse(
            budget,
            market_env,
            info_env,
            loadings_matrix,
            cw,
            agent_rng,
            n_markets,
            delays=delays,
        )
    elif mix == "lp_vs_informed":
        agents = build_agents_lp_vs_informed(
            budget,
            market_env,
            info_env,
            loadings_matrix,
            cw,
            agent_rng,
            n_markets,
            delays=delays,
            lp_observation_delay=lp_observation_delay,
            lp_half_spread_pct=lp_half_spread_pct,
            lp_quote_size=lp_quote_size,
            lp_budget=lp_budget,
        )
    else:
        agents = build_agents_naive_dominated(
            budget, market_env, info_env, n_markets, naive_dominated_count,
            delays=delays,
        )

    population = AgentPopulation(agents)
    population.register(sim, market_env, until_ts=until_ts)
    _register_venue_clock(sim, market_env, until_ts)

    if mechanism == "amm":
        pool_start = {
            m: (float(venues_any[m].reserve_x), float(venues_any[m].reserve_y))
            for m in range(n_markets)
        }
    else:
        pool_start = {
            m: (float(reserve_x), float(reserve_x * anchor_prices[m]))
            for m in range(n_markets)
        }

    if mechanism == "amm":
        initial_mids = np.array(
            [pool_start[m][1] / pool_start[m][0] for m in range(n_markets)],
            dtype=float,
        )
    else:
        initial_mids = np.array(
            [
                float(market_env.mid_price(m) or fair_arr[m])
                for m in range(n_markets)
            ],
            dtype=float,
        )

    sim.run_until(until_ts)
    population.sync_costs()

    if mechanism == "amm":
        pool_end = {
            m: (float(venues_any[m].reserve_x), float(venues_any[m].reserve_y))
            for m in range(n_markets)
        }
    else:
        pool_end = {
            m: (
                float(reserve_x),
                float(reserve_x * _safe_mid(market_env, m, fair_arr[m])),
            )
            for m in range(n_markets)
        }

    log_truth = np.log(fair_arr)

    records = list(market_env.trade_log)
    if mechanism == "amm":
        final_mids = np.array(
            [float(market_env.mid_price(m)) for m in range(n_markets)],
            dtype=float,
        )
    else:
        final_mids = np.array(
            [_safe_mid(market_env, m, float(fair_arr[m])) for m in range(n_markets)],
            dtype=float,
        )

    ts_list = [r.timestamp for r in records]
    mid_list = [r.mid_price_after for r in records]
    mkt_list = [r.market_id for r in records]

    traj = build_mid_trajectory_from_trades(
        ts_list, mkt_list, mid_list, n_markets, initial_mids
    )
    conv = convergence_metrics(
        final_mids=final_mids,
        fair_prices=fair_arr,
        log_truths=log_truth,
        trajectory=traj,
        rel_band=rel_convergence_band,
    )

    informed_ids = _informed_ids_for_mix(mix, naive_dominated_count)
    noise_ids = {NOISE_AGENT_ID}
    fair_map = {m: float(fair_arr[m]) for m in range(n_markets)}
    budgets_map = {a.agent_id: a.budget for a in agents}

    # Per-fill markout against fair-at-fill-time. Truth is the static scalar this
    # phase, so the accessor returns the same value at every timestamp and this
    # is byte-identical to terminal-fair marking — the time-indexing is plumbing
    # for a later phase, not a behavior change now.
    fair_at = frozen_fair_value(fair_map)

    rpnl = rent_and_pnl(
        records,
        fair_prices_by_market=fair_map,
        informed_agent_ids=informed_ids,
        noise_agent_ids=noise_ids,
        pool_reserves_start=pool_start,
        pool_reserves_end=pool_end,
        fair_at=fair_at,
    )

    role_map = _role_by_agent_id(agents)
    role_pnl = pnl_by_role(
        records,
        role_by_agent_id=role_map,
        fair_at=fair_at,
    )

    # LP observability (0 / 0.0 for mixes without an LP). n_lp_fills is the
    # inter-agent fill channel count; the second-half fraction is the G3
    # solvency proxy (an LP that exhausted early would have ~0 here).
    lp_records = [r for r in records if r.agent_id == LP_AGENT_ID]
    n_lp_fills = len(lp_records)
    _half = until_ts / 2.0
    lp_frac_fills_second_half = (
        sum(1 for r in lp_records if r.timestamp >= _half) / n_lp_fills
        if n_lp_fills
        else 0.0
    )

    cap_sat = fraction_exhausted_before_convergence(
        list(market_env.cost_log),
        informed_agent_ids=informed_ids,
        budgets=budgets_map,
        convergence_tick=conv.convergence_tick,
    )

    run_id = (
        f"{mechanism}__{mix}__{capital_band}__{signal_regime}__seed{seed}"
    )

    out: dict[str, Any] = {
        "run_id": run_id,
        "seed": seed,
        "mechanism": mechanism,
        "mix": mix,
        "capital_band": capital_band,
        "budget": budget,
        "signal_regime": signal_regime,
        "signal_noise_std": sig["signal_noise_std"],
        "tail_noise_std": sig["tail_noise_std"],
        "until_ts": until_ts,
        "n_trades": len(records),
        "normalized_rmse_log": conv.normalized_rmse_log,
        "max_relative_price_error": conv.normalized_max_rel_price_error,
        "convergence_tick": conv.convergence_tick,
        "lp_rent_total": rpnl.lp_rent_total,
        "informed_pnl_total": rpnl.informed_pnl_total,
        "noise_pnl": rpnl.noise_pnl,
        "noise_loss": rpnl.noise_loss,
        "rent_efficiency": rpnl.rent_efficiency,
        "rent_efficiency_stable": rpnl.rent_efficiency_stable,
        "frac_informed_exhausted_before_convergence": cap_sat.fraction_informed_exhausted_before_convergence,
        "n_informed_agents": cap_sat.n_informed,
        "n_informed_exhausted": cap_sat.n_exhausted,
        "delay_fast": delays.fast,
        "delay_slow": delays.slow,
        "pnl_fast_informed": role_pnl.get(ROLE_FAST, 0.0),
        "pnl_slow_informed": role_pnl.get(ROLE_SLOW, 0.0),
        "pnl_noise_role": role_pnl.get(ROLE_NOISE, 0.0),
        "pnl_lp": role_pnl.get(ROLE_LP, 0.0),
        "n_lp_fills": n_lp_fills,
        "lp_frac_fills_second_half": lp_frac_fills_second_half,
    }
    return {
        "summary": out,
        "timeseries": _timeseries_long(run_id, traj, n_markets),
    }


def _timeseries_long(
    run_id: str, trajectory: list, n_markets: int
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for snap in trajectory:
        t = snap.tick
        for m in range(n_markets):
            rows.append(
                {
                    "run_id": run_id,
                    "tick": t,
                    "market_id": m,
                    "mid": float(snap.mids[m]),
                }
            )
    return rows


@dataclass
class SweepConfig:
    mechanisms: Sequence[MechanismName] = ("amm", "clob", "hybrid")
    mixes: Sequence[MixName] = ("diverse", "naive_dominated")
    capital_bands: Sequence[CapitalBandName] = ("low", "mid", "high")
    signal_regimes: Sequence[SignalRegimeName] = ("low", "high")
    n_seeds: int = 25
    seed_offset: int = 0
    until_ts: int = 40_000
    time_resolution: int = 1000
    reserve_x: float = 8000.0
    rel_convergence_band: float = 0.02
    naive_dominated_count: int = 5
    results_dir: Path = field(default_factory=lambda: Path("analysis/results"))
    save_timeseries: bool = True


def run_sweep(
    config: SweepConfig | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """
    Execute full factorial over config axes; write Parquet under ``results_dir``.

    Returns (summary_df, timeseries_df or None if save_timeseries=False).
    """
    if config is None:
        config = SweepConfig()

    config.results_dir.mkdir(parents=True, exist_ok=True)
    summary_rows: list[dict[str, Any]] = []
    ts_rows: list[dict[str, Any]] = []

    for mechanism in config.mechanisms:
        for mix in config.mixes:
            for cap_band in config.capital_bands:
                for sig_reg in config.signal_regimes:
                    for i in range(config.n_seeds):
                        seed = config.seed_offset + i
                        pack = run_single_simulation(
                            seed=seed,
                            mechanism=mechanism,
                            mix=mix,
                            capital_band=cap_band,
                            signal_regime=sig_reg,
                            until_ts=config.until_ts,
                            time_resolution=config.time_resolution,
                            reserve_x=config.reserve_x,
                            rel_convergence_band=config.rel_convergence_band,
                            naive_dominated_count=config.naive_dominated_count,
                        )
                        summary_rows.append(pack["summary"])
                        if config.save_timeseries:
                            ts_rows.extend(pack["timeseries"])

    summary_df = pd.DataFrame(summary_rows)
    summary_path = config.results_dir / "sweep_summary.parquet"
    summary_df.to_parquet(summary_path, index=False)

    ts_df: pd.DataFrame | None = None
    if config.save_timeseries and ts_rows:
        ts_df = pd.DataFrame(ts_rows)
        ts_df.to_parquet(config.results_dir / "sweep_timeseries.parquet", index=False)

    return summary_df, ts_df


def rerun_clob_and_merge(config: SweepConfig | None = None) -> pd.DataFrame:
    """
    Re-run only ``clob`` cells and merge into existing Parquet under ``results_dir``.
    """
    if config is None:
        config = SweepConfig()

    summary_path = config.results_dir / "sweep_summary.parquet"
    ts_path = config.results_dir / "sweep_timeseries.parquet"
    if not summary_path.exists():
        raise FileNotFoundError(f"missing {summary_path} — run full sweep first")

    existing = pd.read_parquet(summary_path)
    kept = existing[existing["mechanism"] != "clob"].copy()

    summary_rows: list[dict[str, Any]] = []
    ts_rows: list[dict[str, Any]] = []
    for mix in config.mixes:
        for cap_band in config.capital_bands:
            for sig_reg in config.signal_regimes:
                for i in range(config.n_seeds):
                    seed = config.seed_offset + i
                    pack = run_single_simulation(
                        seed=seed,
                        mechanism="clob",
                        mix=mix,
                        capital_band=cap_band,
                        signal_regime=sig_reg,
                        until_ts=config.until_ts,
                        time_resolution=config.time_resolution,
                        reserve_x=config.reserve_x,
                        rel_convergence_band=config.rel_convergence_band,
                        naive_dominated_count=config.naive_dominated_count,
                    )
                    summary_rows.append(pack["summary"])
                    if config.save_timeseries:
                        ts_rows.extend(pack["timeseries"])

    clob_summary = pd.DataFrame(summary_rows)
    merged = pd.concat([kept, clob_summary], ignore_index=True)
    merged = merged.sort_values("run_id").reset_index(drop=True)
    merged.to_parquet(summary_path, index=False)

    if config.save_timeseries and ts_path.exists():
        ts_kept = pd.read_parquet(ts_path)
        ts_kept = ts_kept[~ts_kept["run_id"].str.startswith("clob__")]
        ts_merged = pd.concat([ts_kept, pd.DataFrame(ts_rows)], ignore_index=True)
        ts_merged = ts_merged.sort_values(["run_id", "tick", "market_id"]).reset_index(
            drop=True
        )
        ts_merged.to_parquet(ts_path, index=False)

    return merged


def main() -> None:
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "clob-only":
        rerun_clob_and_merge(SweepConfig())
        return
    run_sweep(SweepConfig())


if __name__ == "__main__":
    main()
