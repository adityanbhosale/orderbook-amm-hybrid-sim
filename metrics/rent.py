"""AMM pool rent and agent mark-to-market PnL from trade logs."""
from __future__ import annotations

from dataclasses import dataclass

from environment.trade_records import TradeRecord


@dataclass
class RentPnlResult:
    lp_rent_total: float
    informed_pnl_total: float
    noise_pnl: float
    noise_loss: float
    rent_efficiency: float
    rent_efficiency_stable: float


def rent_efficiency_stable(
    informed_pnl: float,
    noise_loss: float,
    epsilon_floor: float = 10.0,
) -> float:
    """Floored denominator to prevent ratio explosion when noise loss is small."""
    return float(informed_pnl / max(abs(noise_loss), epsilon_floor))


def _trade_mtm_pnl(rec: TradeRecord, fair_end: float) -> float:
    if rec.side == "buy":
        return float(rec.quantity * (fair_end - rec.avg_fill_price))
    return float(rec.quantity * (rec.avg_fill_price - fair_end))


def pnl_by_role(
    records: list[TradeRecord],
    *,
    fair_prices_by_market: dict[int, float],
    role_by_agent_id: dict[int, str],
) -> dict[str, float]:
    """Bucket per-agent terminal mark-to-market PnL by role.

    Uses the same per-fill ``_trade_mtm_pnl`` as :func:`rent_and_pnl` (terminal
    fair, static log-truth), so role buckets are consistent with the existing
    informed-vs-noise outputs. ``role_by_agent_id`` maps each agent_id to a
    role label; agents absent from the map are ignored. Returns a dict keyed
    by every role label present in ``role_by_agent_id`` (zero if no fills).
    """
    out: dict[str, float] = {role: 0.0 for role in set(role_by_agent_id.values())}
    for r in records:
        role = role_by_agent_id.get(r.agent_id)
        if role is None:
            continue
        out[role] += _trade_mtm_pnl(r, fair_prices_by_market[r.market_id])
    return out


def lp_rent_cp_amm_per_pool(
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    fair_price: float,
) -> float:
    """Change in LP mark-to-market at the terminal fair price (quote numeraire)."""
    v0 = float(y0 + x0 * fair_price)
    v1 = float(y1 + x1 * fair_price)
    return v1 - v0


def rent_and_pnl(
    records: list[TradeRecord],
    *,
    fair_prices_by_market: dict[int, float],
    informed_agent_ids: set[int],
    noise_agent_ids: set[int],
    pool_reserves_start: dict[int, tuple[float, float]],
    pool_reserves_end: dict[int, tuple[float, float]],
) -> RentPnlResult:
    """
    LP rent summed over parallel pools; PnL is horizon mark-to-market at
    terminal fair (log-truth is static in the current world model).
    """
    lp_rent = 0.0
    for m, (x0, y0) in pool_reserves_start.items():
        x1, y1 = pool_reserves_end[m]
        fair = fair_prices_by_market[m]
        lp_rent += lp_rent_cp_amm_per_pool(x0, y0, x1, y1, fair)

    inf_pnl = 0.0
    noise_pnl = 0.0
    for r in records:
        fair = fair_prices_by_market[r.market_id]
        pnl = _trade_mtm_pnl(r, fair)
        if r.agent_id in informed_agent_ids:
            inf_pnl += pnl
        elif r.agent_id in noise_agent_ids:
            noise_pnl += pnl

    noise_loss = float(-noise_pnl)
    denom = abs(noise_loss) + 1e-12
    rent_eff = float(inf_pnl / denom)
    rent_eff_stable = rent_efficiency_stable(inf_pnl, noise_loss)

    return RentPnlResult(
        lp_rent_total=float(lp_rent),
        informed_pnl_total=float(inf_pnl),
        noise_pnl=float(noise_pnl),
        noise_loss=noise_loss,
        rent_efficiency=rent_eff,
        rent_efficiency_stable=rent_eff_stable,
    )


__all__ = [
    "RentPnlResult",
    "lp_rent_cp_amm_per_pool",
    "pnl_by_role",
    "rent_and_pnl",
    "rent_efficiency_stable",
]
