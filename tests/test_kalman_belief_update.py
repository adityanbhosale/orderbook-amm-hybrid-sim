"""Phase-A time-aware (Kalman) scalar belief update — gated process noise.

Proves the decay MECHANISM on the shared scalar helper without any moving truth:
- q=0 (or dt=0) is byte-identical to the stationary precision-only update;
- q>0 with dt>0 decays prior precision before absorbing the signal, so the gain
  is strictly higher (the posterior moves further toward the observation) — i.e.
  the belief tracks rather than freezes. Truth stays frozen in real runs.
"""
from __future__ import annotations

import pytest

from agents.belief_utils import gaussian_scalar_nif_update

# Fixed inputs: a confident prior (high precision -> low stationary gain).
MEAN, PREC, OBS, OBS_PREC = 0.0, 10.0, 1.0, 1.0


def _manual_stationary(mean, prec, obs, obs_prec):
    new_prec = prec + obs_prec
    new_mean = (prec * mean + obs_prec * obs) / new_prec
    return new_mean, new_prec


# q=0 byte-identical ----------------------------------------------------- #
def test_q0_matches_manual_stationary_exactly() -> None:
    assert gaussian_scalar_nif_update(MEAN, PREC, OBS, OBS_PREC) == _manual_stationary(
        MEAN, PREC, OBS, OBS_PREC
    )


def test_q0_ignores_dt_byte_identical() -> None:
    """q=0 must SKIP the decay entirely — no float reciprocal round-trip."""
    base = gaussian_scalar_nif_update(MEAN, PREC, OBS, OBS_PREC)
    assert gaussian_scalar_nif_update(MEAN, PREC, OBS, OBS_PREC, q=0.0, dt=999.0) == base


def test_qpos_dt0_short_circuits_to_stationary() -> None:
    """No elapsed time => no decay, even with q>0."""
    base = gaussian_scalar_nif_update(MEAN, PREC, OBS, OBS_PREC)
    assert gaussian_scalar_nif_update(MEAN, PREC, OBS, OBS_PREC, q=0.05, dt=0.0) == base


# q>0 decay fires -------------------------------------------------------- #
def test_qpos_dt_pos_decays_prior_and_raises_gain() -> None:
    stat_mean, stat_prec = gaussian_scalar_nif_update(MEAN, PREC, OBS, OBS_PREC)
    kal_mean, kal_prec = gaussian_scalar_nif_update(
        MEAN, PREC, OBS, OBS_PREC, q=0.01, dt=100.0
    )
    # Prior precision was decayed: 1/(1/10 + 0.01*100) = 1/1.1, so posterior
    # precision is far lower and the posterior mean moves much further toward
    # the observation (1.0) than the frozen-prior case.
    assert kal_prec < stat_prec
    assert kal_mean > stat_mean
    expected_decayed = 1.0 / (1.0 / PREC + 0.01 * 100.0)
    assert kal_prec == pytest.approx(expected_decayed + OBS_PREC)
    assert kal_mean == pytest.approx(
        (expected_decayed * MEAN + OBS_PREC * OBS) / (expected_decayed + OBS_PREC)
    )


def test_more_elapsed_time_means_more_decay() -> None:
    """Monotone: longer gap since last update -> staler prior -> higher gain."""
    _, p_short = gaussian_scalar_nif_update(MEAN, PREC, OBS, OBS_PREC, q=0.01, dt=10.0)
    _, p_long = gaussian_scalar_nif_update(MEAN, PREC, OBS, OBS_PREC, q=0.01, dt=1000.0)
    assert p_long < p_short  # more decay -> lower posterior precision


def test_negative_q_or_dt_rejected() -> None:
    with pytest.raises(ValueError):
        gaussian_scalar_nif_update(MEAN, PREC, OBS, OBS_PREC, q=-1.0, dt=1.0)
    with pytest.raises(ValueError):
        gaussian_scalar_nif_update(MEAN, PREC, OBS, OBS_PREC, q=0.01, dt=-1.0)
