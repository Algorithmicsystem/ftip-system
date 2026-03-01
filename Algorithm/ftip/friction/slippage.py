from __future__ import annotations

from .costs import bps_to_cash, clamp
from .models import CostModel, MarketStateInputs


def compute_spread_proxy_bps(market: MarketStateInputs, k: float = 0.1) -> float:
    if market.spread_proxy_bps is not None:
        return float(market.spread_proxy_bps)
    range_bps = ((market.high - market.low) / market.close) * 10000.0
    return clamp(range_bps * k, 1.0, 80.0)


def compute_slippage_bps(cost_model: CostModel, market: MarketStateInputs) -> float:
    rv = market.rv_20 if market.rv_20 is not None else 0.02
    spread_proxy = compute_spread_proxy_bps(market)
    rv_scale = max(0.5, rv / 0.02)
    spread_scale = max(0.5, spread_proxy / max(1.0, cost_model.spread_bps))
    return cost_model.slippage_bps * (0.6 + 0.2 * rv_scale + 0.2 * spread_scale)


def compute_slippage_paid(
    notional: float, cost_model: CostModel, market: MarketStateInputs
) -> float:
    return bps_to_cash(notional, compute_slippage_bps(cost_model, market))


def compute_spread_paid(notional: float, market: MarketStateInputs) -> float:
    return bps_to_cash(notional, compute_spread_proxy_bps(market))
