"""Signal payloads for the information process."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Signal:
    """Information arrival delivered to agents via the event queue.

    In the log-linear world, ``value`` is a noisy observation of the true
    log-fair-value :math:`L^*_m` (natural log of the full fair price),
    and ``noise_std`` is the observation noise *in log-price units*.
    """

    market_id: int
    value: float
    is_tail: bool
    noise_std: float
