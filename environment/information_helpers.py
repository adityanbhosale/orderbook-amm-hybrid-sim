"""Helpers for constructing agent-level priors from simulator truth.

Ported idea from:
  ``base_rates_from_truth`` in
  /Users/adityabhosale/Downloads/Projects/lmsr-preclinical-markets/sim/agentic.py
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from environment.information import InformationEnvironment


def base_log_levels_from_truth(
    info_env: "InformationEnvironment",
    market_ids: tuple[int, ...],
    rng: np.random.Generator,
    *,
    noise_std: float = 0.0,
    clip_log_width: float = 6.0,
) -> dict[int, float]:
    """
    Noisy prior means over log-fair-value for each market.

    If ``noise_std == 0``, returns true log-fair-value. Otherwise
    ``prior_log = log_fv + N(0, noise_std^2)`` clamped to
    ``[log_fv - clip, log_fv + clip]`` (relative to each truth).
    """
    if noise_std < 0:
        raise ValueError("noise_std must be non-negative")
    out: dict[int, float] = {}
    for m in market_ids:
        true_log = info_env.world.truths[m].log_fair_value
        if noise_std == 0:
            noisy = true_log
        else:
            noisy = true_log + float(rng.normal(0.0, noise_std))
        lo, hi = true_log - clip_log_width, true_log + clip_log_width
        out[m] = float(max(lo, min(hi, noisy)))
    return out
