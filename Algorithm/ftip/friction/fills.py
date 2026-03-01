from __future__ import annotations

import hashlib
import random
from typing import Tuple

from .costs import clamp
from .impact import compute_adv_notional
from .models import CostModel, ExecutionPlan, MarketStateInputs


def stable_hash(*parts: str) -> int:
    joined = "|".join(parts)
    digest = hashlib.sha256(joined.encode("utf-8")).hexdigest()
    return int(digest[:16], 16)


def _seeded_random(cost_model: CostModel, plan: ExecutionPlan) -> random.Random:
    mixed_seed = cost_model.seed + stable_hash(
        plan.symbol,
        plan.date.isoformat(),
        plan.side,
        plan.order_type,
        str(plan.limit_price or ""),
    )
    return random.Random(mixed_seed)


def compute_fill_fraction(
    cost_model: CostModel,
    market: MarketStateInputs,
    plan: ExecutionPlan,
) -> Tuple[float, str]:
    adv_notional = compute_adv_notional(market)
    max_notional = adv_notional * cost_model.max_adv_pct
    if max_notional <= 0:
        return 0.0, "NO_LIQUIDITY"

    base_fill = 1.0
    if plan.order_type == "MARKET":
        if not cost_model.allow_market_orders:
            return 0.0, "MARKET_ORDERS_DISABLED"
    else:
        if not cost_model.allow_limit_orders:
            return 0.0, "LIMIT_ORDERS_DISABLED"
        if plan.limit_price is None:
            return 0.0, "LIMIT_PRICE_REQUIRED"

        if plan.side == "BUY" and plan.limit_price < market.low:
            return 0.0, "LIMIT_NOT_REACHED"
        if plan.side == "SELL" and plan.limit_price > market.high:
            return 0.0, "LIMIT_NOT_REACHED"

        if plan.side == "BUY" and plan.limit_price >= market.high:
            base_fill = 1.0
        elif plan.side == "SELL" and plan.limit_price <= market.low:
            base_fill = 1.0
        else:
            distance_to_close = abs(plan.limit_price - market.close) / market.close
            distance_penalty = clamp(1.0 - distance_to_close * 20.0, 0.0, 1.0)
            volume_regime = clamp(
                (market.volume / max(1.0, market.adv_20 or market.volume)), 0.5, 2.0
            )
            probability = clamp(
                cost_model.limit_fill_probability
                * distance_penalty
                * (0.5 + 0.5 * volume_regime),
                0.0,
                1.0,
            )
            rng = _seeded_random(cost_model, plan)
            base_fill = probability if rng.random() <= probability else 0.0

    fill_fraction = min(base_fill, max_notional / plan.notional)
    if fill_fraction <= 0:
        return 0.0, "ADV_CONSTRAINT"
    if fill_fraction < 1:
        return fill_fraction, "PARTIAL_ADV_CONSTRAINT"
    return 1.0, "FILLED"
