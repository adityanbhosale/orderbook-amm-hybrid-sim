"""Margin and notional helpers for impact-based capital checks."""
from __future__ import annotations

from dataclasses import dataclass

from venues.base import Venue


@dataclass(frozen=True)
class MarginSpec:
    """Per-side initial margin as *fraction of notional* (positive, dimensionless).

    long_margin_fraction: collateral required per dollar notional for buys.
    short_margin_fraction: collateral required per dollar notional for sells.

    Use ``1.0`` for full cash payment on the long leg; use ``< 1`` for typical
    linear perp-style initial margin on either side; values ``> 1`` are allowed
    (extra-conservative haircuts).
    """

    long_margin_fraction: float = 1.0
    short_margin_fraction: float = 1.0

    def __post_init__(self) -> None:
        if self.long_margin_fraction <= 0:
            raise ValueError("long_margin_fraction must be positive")
        if self.short_margin_fraction <= 0:
            raise ValueError("short_margin_fraction must be positive")


def committed_capital(
    venue: Venue,
    side: str,
    quantity: float,
    margin: MarginSpec,
    *,
    safety_margin: float = 1.0,
) -> float:
    """Upper-bound capital locked using *estimated* VWAP (pre-trade).

    notional = quantity * estimate_impact(side, quantity)
    Commitment = notional * margin_fraction(side) * safety_margin
    """
    if quantity <= 0:
        raise ValueError("quantity must be positive")
    if safety_margin < 1.0:
        raise ValueError("safety_margin must be >= 1.0")
    if side not in ("buy", "sell"):
        raise ValueError("side must be 'buy' or 'sell'")

    vwap = venue.estimate_impact(side, quantity)
    notional = quantity * vwap
    frac = margin.long_margin_fraction if side == "buy" else margin.short_margin_fraction
    return float(notional * frac * safety_margin)
