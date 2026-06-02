from __future__ import annotations
from api.assistant.phase3.common import clamp

def compute_ias(flow_data: dict) -> float:
    """Institutional Accumulation Score (0-100). Higher = more bullish institutional flow."""
    dark_pool_buy_ratio = flow_data.get("dark_pool_buy_ratio")   # 0-1, > 0.6 = accumulation
    block_trade_direction = flow_data.get("block_trade_direction")  # -100 to 100
    short_interest_change = flow_data.get("short_interest_change")  # -0.30 to 0.30, negative = bullish

    components = {}

    # Component 1: dark_pool_buy_ratio bounded [0.2, 0.8]
    if dark_pool_buy_ratio is not None:
        dp = clamp(float(dark_pool_buy_ratio), 0.2, 0.8)
        components["dark_pool_buy_ratio"] = ((dp - 0.2) / 0.6) * 100.0

    # Component 2: block_trade_direction [-100, 100] → normalized 0-100
    if block_trade_direction is not None:
        btd = clamp(float(block_trade_direction), -100.0, 100.0)
        components["block_trade_direction"] = (btd + 100.0) / 2.0

    # Component 3: short_interest_change inverted [-0.30, 0.30], falling short = bullish = higher
    if short_interest_change is not None:
        sic = clamp(float(short_interest_change), -0.30, 0.30)
        # invert: falling (negative) = bullish = high score
        components["short_interest_change_inv"] = ((0.30 - sic) / 0.60) * 100.0

    if not components:
        return 50.0

    canonical_weights = {
        "dark_pool_buy_ratio": 0.40,
        "block_trade_direction": 0.35,
        "short_interest_change_inv": 0.25,
    }
    total_w = sum(canonical_weights[k] for k in components)
    if total_w <= 0:
        return 50.0
    ias = sum(components[k] * canonical_weights[k] for k in components) / total_w
    return round(clamp(ias, 0.0, 100.0), 2)
