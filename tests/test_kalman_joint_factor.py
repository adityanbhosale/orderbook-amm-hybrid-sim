"""Phase-A2 time-aware process noise for joint-factor's MATRIX belief update.

Proves the matrix analog of the scalar Kalman step on the (Λ, η) factor posterior
without any moving truth:
- q=0 (or dt=0) is byte-identical to the precision-only matrix update;
- q>0 with dt>0 inflates the factor covariance before absorbing each signal, so
  the posterior decays toward the prior (tracks rather than freezes); and
- the decayed information matrix stays positive-semidefinite (the new risk vs the
  scalar case). Truth stays frozen in real runs.
"""
from __future__ import annotations

import math

import numpy as np

from agents.informed_joint_factor import make_joint_factor_agent
from environment.signals import Signal

LOADINGS = np.array([[1.0, 0.0], [0.0, 1.0]])  # 2 markets, k=2 factors
MIDS = {0: 100.0, 1: 100.0}


def _agent(q: float):
    return make_joint_factor_agent(
        agent_id=4,
        budget=1000.0,
        primary_markets=(0, 1),
        observed_markets=(0, 1),
        loadings_matrix=LOADINGS,
        initial_mid_by_market=MIDS,
        signal_noise_inflation=1.0,
        q=q,
    )


def _sig(m: int, px: float) -> Signal:
    return Signal(market_id=m, value=math.log(px), is_tail=False, noise_std=0.01)


# q=0 / dt=0 short-circuit ---------------------------------------------- #
def test_q0_matches_precision_only_matrix_update() -> None:
    a = _agent(0.0)
    for t in (0, 100, 200):
        a.update_posterior(_sig(0, 101.0), now=t)
        a.update_posterior(_sig(1, 99.0), now=t)
    # Manual precision-only reconstruction (Λ += τ ββᵀ; η += τ y β), same order.
    Lam = 1.0 * np.eye(2)
    eta = np.zeros(2)
    tau = 1.0 / (0.01 * 0.01)
    for _ in range(3):
        for m, px in ((0, 101.0), (1, 99.0)):
            beta = LOADINGS[m]
            y = math.log(px) - a.alpha_by_market[m]
            Lam = Lam + tau * np.outer(beta, beta)
            eta = eta + tau * y * beta
    assert np.array_equal(a._Lambda, Lam)
    assert np.array_equal(a._eta, eta)


def test_qpos_but_all_same_tick_is_identical_to_q0() -> None:
    """dt=0 everywhere (same tick) => decay never fires => == q=0."""
    a0, aq = _agent(0.0), _agent(0.5)
    for m, px in ((0, 101.0), (1, 99.0), (0, 102.0)):
        a0.update_posterior(_sig(m, px), now=7)
        aq.update_posterior(_sig(m, px), now=7)
    assert np.array_equal(a0._Lambda, aq._Lambda)
    assert np.array_equal(a0._eta, aq._eta)


# G-PSD: decayed information matrix stays PSD ---------------------------- #
def test_gpsd_lambda_stays_psd_across_decayed_updates() -> None:
    a = _agent(0.02)
    now = 0
    for m, px in ((0, 101.0), (1, 98.0), (0, 103.0), (1, 99.5), (0, 100.5)):
        now += 50  # dt>0 each step -> decay fires
        a.update_posterior(_sig(m, px), now=now)
        lam_eig = np.linalg.eigvalsh(a._Lambda)
        sig_eig = np.linalg.eigvalsh(np.linalg.inv(a._Lambda))
        assert lam_eig.min() >= -1e-9, f"Lambda non-PSD: {lam_eig}"
        assert sig_eig.min() >= -1e-9, f"Sigma non-PSD: {sig_eig}"
        # symmetry preserved (fp hygiene)
        assert np.allclose(a._Lambda, a._Lambda.T, atol=1e-12)


# G-PROC: decay fires (belief tracks instead of freezing) --------------- #
def test_gproc_decay_lowers_precision_vs_no_decay() -> None:
    a0, aq = _agent(0.0), _agent(0.05)
    now = 0
    for m, px in ((0, 101.0), (1, 99.0), (0, 101.5), (1, 98.5)):
        now += 100  # time gaps so q>0 agent decays between updates
        a0.update_posterior(_sig(m, px), now=now)
        aq.update_posterior(_sig(m, px), now=now)
    # Same evidence, but the q>0 agent shed information to the random walk
    # between updates -> strictly lower accumulated precision (not frozen).
    assert np.trace(aq._Lambda) < np.trace(a0._Lambda)
