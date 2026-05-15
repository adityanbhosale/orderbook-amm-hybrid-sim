"""Sizing helpers: confidence scaling and capital-feasible quantity search."""
from __future__ import annotations

import math

import numpy as np

from environment.margin import MarginSpec, committed_capital
from venues.base import Venue


def confidence_weighted_size(
    base_size: float,
    *,
    confidence_weighted: bool,
    posterior_precision: float,
    prior_precision: float,
    floor: float = 0.25,
    ceiling: float = 4.0,
) -> float:
    """Port of ``_compute_trade_size`` from reference ``sim/agentic.py``."""
    if not confidence_weighted:
        return base_size
    relative = posterior_precision / max(prior_precision, 1e-9)
    mult = float(np.clip(math.sqrt(max(relative, 0.0)), floor, ceiling))
    return base_size * mult


def downsize_quantity_for_capital(
    venue: Venue,
    side: str,
    target_quantity: float,
    available: float,
    margin: MarginSpec,
    *,
    safety_margin: float = 1.2,
    min_quantity: float = 1e-6,
    search_iters: int = 24,
) -> float:
    """Largest q in [0, target_quantity] with committed_capital(q) <= available."""
    if target_quantity <= 0:
        return 0.0
    if available <= 0:
        return 0.0

    if (
        committed_capital(
            venue, side, target_quantity, margin, safety_margin=safety_margin
        )
        <= available
    ):
        return target_quantity

    lo, hi = 0.0, target_quantity
    for _ in range(search_iters):
        mid = 0.5 * (lo + hi)
        if mid < min_quantity:
            break
        c = committed_capital(venue, side, mid, margin, safety_margin=safety_margin)
        if c <= available:
            lo = mid
        else:
            hi = mid
    if lo < min_quantity:
        return 0.0
    return lo
