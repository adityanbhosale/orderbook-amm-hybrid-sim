"""Phase B — random-walk true_val (moving-truth world).

Truth rides a common-factor Gaussian walk (f_t = f_{t-1} + N(0, walk_var)), so
per-market log-fair-value = alpha_m + beta_m·f_t + idio_m. Validates:
- walk_var=0 => no path (static world, byte-identical handled by the run-level
  G-ID check);
- the path is deterministic from its dedicated seed;
- (G-WALK-MOVES) truth actually moves and markets CO-MOVE via the loadings (the
  common-factor signature), not independently;
- (G-TRACK) with q=walk_var a correctly-specified agent tracks the moving truth
  with bounded lag, where the old precision-only filter (q=0) freezes.
"""
from __future__ import annotations

import numpy as np

from agents.informed_naive import NaiveGaussianBeliefAgent
from environment.information import InformationEnvironment
from environment.signals import Signal
from simulation.sweep import _default_information_config

UNTIL = 8000


def _env(walk_var: float, world_seed: int = 123):
    cfg = _default_information_config(
        signal_noise_std=0.012, tail_noise_std=0.005, walk_var=walk_var
    )
    return InformationEnvironment(cfg, np.random.default_rng(world_seed))


def test_walk_var_zero_has_no_path_and_static_truth() -> None:
    env = _env(0.0)
    assert env._materialize_factor_walk_path(UNTIL, np.random.default_rng(1)) is None
    # accessor returns the constant static truth at any t
    assert env.log_fair_value_at(0, 0) == env.world.truths[0].log_fair_value
    assert env.log_fair_value_at(0, UNTIL) == env.world.truths[0].log_fair_value


def test_path_is_deterministic_from_seed() -> None:
    env = _env(1e-4)
    p1 = env._materialize_factor_walk_path(UNTIL, np.random.default_rng(999))
    p2 = env._materialize_factor_walk_path(UNTIL, np.random.default_rng(999))
    p3 = env._materialize_factor_walk_path(UNTIL, np.random.default_rng(7))
    assert np.array_equal(p1, p2)
    assert not np.array_equal(p1, p3)
    # t=0 column equals the static truth exactly (walk starts at f0)
    static = np.array([t.log_fair_value for t in env.world.truths])
    assert np.allclose(p1[:, 0], static, atol=1e-12)


def test_gwalk_moves_and_comoves_via_loadings() -> None:
    walk_var = 1e-4
    env = _env(walk_var)
    path = env._materialize_factor_walk_path(UNTIL, np.random.default_rng(999))

    # (1) truth ACTUALLY moves over the horizon.
    move = np.abs(path[:, -1] - path[:, 0])
    assert move.min() > 1e-3, f"truth barely moved: {move}"
    assert path.std(axis=1).min() > 1e-3

    # (2) co-movement is the COMMON-FACTOR signature: Cov(Δ) = walk_var·LLᵀ.
    # Independent per-market shocks would give a DIAGONAL covariance; the shared
    # factor produces nonzero off-diagonals matching β_m·β_n.
    d = np.diff(path, axis=1)              # (M, T) increments
    emp_cov = (d @ d.T) / d.shape[1]
    L = env.world.loadings_matrix
    theory = walk_var * (L @ L.T)
    # structural alignment (incl. off-diagonals) is near-perfect
    corr = np.corrcoef(emp_cov.ravel(), theory.ravel())[0, 1]
    assert corr > 0.99, f"increment cov not common-factor shaped (corr={corr})"
    # and at least one cross-market term is materially nonzero (true co-movement)
    off = emp_cov[~np.eye(emp_cov.shape[0], dtype=bool)]
    assert np.max(np.abs(off)) > 1e-6


def _naive(m0: float, q: float, sp: float = 100.0):
    return NaiveGaussianBeliefAgent(
        agent_id=1,
        budget=1e9,
        market_ids=(0,),
        initial_log_fair_mean={0: m0},
        prior_precision=1.0,
        signal_precision_assumed=sp,
        q=q,
    )


def test_gtrack_qmatched_tracks_bounded_where_q0_freezes() -> None:
    walk_var = 1e-4
    env = _env(walk_var)
    path = env._materialize_factor_walk_path(UNTIL, np.random.default_rng(999))
    m0 = float(path[0, 0])

    track = _naive(m0, q=walk_var)   # correctly specified (q == walk_var)
    frozen = _naive(m0, q=0.0)       # old precision-only behaviour

    err_track, err_frozen = [], []
    for t in range(0, UNTIL, 50):
        truth_t = float(path[0, t])
        sig = Signal(market_id=0, value=truth_t, is_tail=False, noise_std=0.01)
        track.update_posterior(sig, now=t)
        frozen.update_posterior(sig, now=t)
        err_track.append(abs(track._mean_log[0] - truth_t))
        err_frozen.append(abs(frozen._mean_log[0] - truth_t))

    et, ef = np.array(err_track), np.array(err_frozen)
    n = len(et)
    b = n // 4
    track_early, track_late = et[b:2 * b].mean(), et[-b:].mean()
    frozen_early, frozen_late = ef[b:2 * b].mean(), ef[-b:].mean()
    total_move = float(abs(path[0, -1] - path[0, 0]))

    # q=walk_var: BOUNDED — error stays small vs how far truth moved and does
    # not grow over the run (stationary steady state).
    assert et.max() < 0.5 * total_move, f"track error not bounded: {et.max()} / {total_move}"
    assert track_late < 1.5 * track_early, f"track error growing: {track_early}->{track_late}"
    # q=0: FREEZES — error GROWS as truth wanders, and is far worse than the tracker.
    assert frozen_late > 1.25 * frozen_early, f"frozen not growing: {frozen_early}->{frozen_late}"
    assert frozen_late > 3.0 * track_late, f"frozen not worse: frozen={frozen_late}, track={track_late}"
