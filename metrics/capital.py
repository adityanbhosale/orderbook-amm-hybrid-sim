"""Budget exhaustion relative to convergence time."""
from __future__ import annotations

from dataclasses import dataclass

from environment.trade_records import TradeRecord


@dataclass
class CapitalSaturationResult:
    fraction_informed_exhausted_before_convergence: float
    n_informed: int
    n_exhausted: int


def fraction_exhausted_before_convergence(
    records: list[TradeRecord],
    *,
    informed_agent_ids: set[int],
    budgets: dict[int, float],
    convergence_tick: int | None,
    exhaustion_frac: float = 0.99,
) -> CapitalSaturationResult:
    """
    An agent is exhausted the first time cumulative
    ``capital_committed + fees_paid`` reaches ``exhaustion_frac`` of budget.
    Count informed agents exhausted strictly before ``convergence_tick``.
    If ``convergence_tick`` is None, compare exhaustion tick to final tick only
    (no mid-convergence benchmark).
    """
    # chronological
    sorted_recs = sorted(records, key=lambda r: (r.timestamp, r.market_id))
    cum: dict[int, float] = {a: 0.0 for a in informed_agent_ids}
    exhaust_tick: dict[int, int] = {}

    for r in sorted_recs:
        if r.agent_id not in informed_agent_ids:
            continue
        cum[r.agent_id] += r.capital_committed + r.fees_paid
        b = budgets.get(r.agent_id, 0.0)
        if (
            r.agent_id not in exhaust_tick
            and b > 0
            and cum[r.agent_id] >= exhaustion_frac * b
        ):
            exhaust_tick[r.agent_id] = r.timestamp

    n_inf = len(informed_agent_ids)
    if n_inf == 0:
        return CapitalSaturationResult(0.0, 0, 0)

    n_ex = 0
    if convergence_tick is None:
        return CapitalSaturationResult(float("nan"), n_inf, 0)

    for a in informed_agent_ids:
        if a not in exhaust_tick:
            continue
        if exhaust_tick[a] < convergence_tick:
            n_ex += 1

    return CapitalSaturationResult(
        fraction_informed_exhausted_before_convergence=float(n_ex) / float(n_inf),
        n_informed=n_inf,
        n_exhausted=n_ex,
    )


__all__ = [
    "CapitalSaturationResult",
    "fraction_exhausted_before_convergence",
]
