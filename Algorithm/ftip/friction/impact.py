from __future__ import annotations

import math

from .costs import bps_to_cash, clamp, safe_div
from .models import CostModel, MarketStateInputs


def compute_adv_notional(market: MarketStateInputs) -> float:
    adv_20 = market.adv_20 if market.adv_20 is not None else market.volume
    return max(0.0, adv_20) * market.close


def compute_impact_bps(
    notional: float, cost_model: CostModel, market: MarketStateInputs
) -> float:
    adv_notional = compute_adv_notional(market)
    participation = max(0.0, safe_div(notional, adv_notional, default=0.0))
    rv = market.rv_20 if market.rv_20 is not None else 0.02
    vol_scale = max(0.25, rv / cost_model.impact.median_rv_reference)
    bps = cost_model.impact.impact_k * math.sqrt(participation) * math.sqrt(vol_scale)
    return clamp(bps, 0.0, cost_model.impact.max_impact_bps)


def compute_impact_paid(
    notional: float, cost_model: CostModel, market: MarketStateInputs
) -> float:
    return bps_to_cash(notional, compute_impact_bps(notional, cost_model, market))
