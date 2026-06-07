"""Phase 15.1: SMB Pricing Intelligence Engine."""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from api.assistant.phase3.common import clamp


SECTOR_PRICING_POWER_BASE: Dict[str, float] = {
    "Technology": 70.0,
    "Healthcare": 65.0,
    "Professional Services": 60.0,
    "Manufacturing": 45.0,
    "Retail": 40.0,
    "Restaurant": 35.0,
    "Commodity": 25.0,
}

_ACTION_RATIONALES = {
    "raise_prices": "High pricing power with elevated input cost pressure — you can raise prices and need to protect margins.",
    "hold": "Stable pricing environment — hold prices to preserve volume while margins are healthy.",
    "defend_volume": "Insufficient pricing power to offset cost pressure — focus on cost reduction and volume protection.",
    "discount": "Low pricing power with low costs — selective discounting may drive volume growth.",
}


@dataclass
class PricingIntelligence:
    entity_id: str
    as_of_date: dt.date
    pricing_power_score: float
    recommended_action: str
    price_increase_potential_pct: float
    margin_trend: str
    input_cost_pressure_score: float
    competitive_position: str
    regime_pricing_context: str


# ---------------------------------------------------------------------------
# Core computations
# ---------------------------------------------------------------------------

def compute_pricing_power_score(
    financials_history: List[Dict[str, Any]],
    sector: str,
    macro_data: Optional[Dict] = None,
) -> float:
    """4-component pricing power score (0–100)."""
    sector_base = SECTOR_PRICING_POWER_BASE.get(sector, 50.0)

    if not financials_history:
        return round(clamp(sector_base, 0.0, 100.0), 2)

    most_recent = financials_history[0]

    # Component 1: gross_margin_trend (weight 0.35)
    gm_recent = float(most_recent.get("gross_margin") or 0.40)
    if len(financials_history) >= 5:
        prior = financials_history[1:5]
    elif len(financials_history) >= 2:
        prior = financials_history[1:]
    else:
        prior = []

    if prior:
        gm_prior_avg = sum(float(p.get("gross_margin") or gm_recent) for p in prior) / len(prior)
    else:
        gm_prior_avg = gm_recent

    margin_delta = gm_recent - gm_prior_avg
    margin_trend_score = clamp((margin_delta / 0.05 * 50.0) + 50.0, 0.0, 100.0)

    # Component 2: revenue_per_unit_trend (weight 0.25)
    rev_growth = float(most_recent.get("revenue_growth_yoy") or 0.0)
    if rev_growth > 0.08:
        rev_score = clamp(50.0 + (rev_growth - 0.08) / 0.12 * 50.0, 50.0, 100.0)
    elif rev_growth >= 0.0:
        rev_score = 50.0
    else:
        rev_score = clamp(50.0 + rev_growth * 300.0, 0.0, 50.0)

    # Component 3: customer_stickiness (weight 0.20)
    ar_vals = [
        float(p["accounts_receivable"])
        for p in financials_history
        if p.get("accounts_receivable") is not None
    ]
    if len(ar_vals) >= 2:
        ar_delta_pct = (ar_vals[0] - ar_vals[-1]) / (ar_vals[-1] + 1.0)
        # AR declining or stable = sticky customers
        stickiness_score = clamp(65.0 - ar_delta_pct * 80.0, 20.0, 90.0)
    else:
        stickiness_score = 60.0

    # Component 4: market_position / sector base (weight 0.20)
    market_score = sector_base

    score = (
        margin_trend_score * 0.35
        + rev_score * 0.25
        + stickiness_score * 0.20
        + market_score * 0.20
    )
    return round(clamp(score, 0.0, 100.0), 2)


def compute_input_cost_pressure(
    financials_history: List[Dict[str, Any]],
    macro_data: Optional[Dict] = None,
) -> float:
    """Input cost pressure score (0–100, higher = more pressure)."""
    if not financials_history:
        return 50.0

    most_recent = financials_history[0]

    # Primary: cogs_growth vs revenue_growth
    cogs_growth = float(most_recent.get("cogs_growth_yoy") or 0.0)
    rev_growth = float(most_recent.get("revenue_growth_yoy") or 0.0)
    spread = cogs_growth - rev_growth

    pressure_score = clamp((spread / 0.10 * 50.0) + 50.0, 0.0, 100.0)

    # Secondary: cogs_pct trend
    cogs_pct = float(most_recent.get("cogs_pct_revenue") or 0.60)
    if len(financials_history) >= 2:
        prior_cogs_pct = float(financials_history[1].get("cogs_pct_revenue") or cogs_pct)
        cogs_pct_delta = cogs_pct - prior_cogs_pct
        cogs_pct_pressure = clamp((cogs_pct_delta / 0.03 * 20.0) + 50.0, 20.0, 80.0)
    else:
        cogs_pct_pressure = 50.0

    # Blend: primary 0.70, secondary 0.30
    return round(clamp(pressure_score * 0.70 + cogs_pct_pressure * 0.30, 0.0, 100.0), 2)


def _classify_margin_trend(financials_history: List[Dict]) -> str:
    if not financials_history or len(financials_history) < 2:
        return "stable"
    gm_now = float(financials_history[0].get("gross_margin") or 0.40)
    gm_then = float(financials_history[-1].get("gross_margin") or gm_now)
    delta = gm_now - gm_then
    if delta > 0.02:
        return "expanding"
    if delta < -0.02:
        return "compressing"
    return "stable"


def _classify_competitive_position(pricing_power_score: float) -> str:
    if pricing_power_score >= 70:
        return "premium"
    if pricing_power_score >= 55:
        return "market"
    if pricing_power_score >= 40:
        return "value"
    return "discount"


def _regime_pricing_context(regime_label: str) -> str:
    regime = regime_label.upper()
    contexts = {
        "HIGH_VOL": "High volatility regime — consumers are price-sensitive; price increases carry higher churn risk.",
        "RECOVERY": "Recovery regime — demand is rebounding; pricing power window is opening.",
        "TRENDING": "Trending regime — stable growth; pricing power follows fundamentals.",
        "CHOPPY": "Choppy regime — mixed signals; proceed with targeted price adjustments.",
        "LIQUIDITY_FRACTURE": "Liquidity fracture — focus on survival; avoid price increases.",
        "COMPENSATION_CAPTURE": "Fed tightening — cost of capital rising; pass through costs where possible.",
    }
    return contexts.get(regime, "Monitor regime transitions before adjusting pricing strategy.")


def generate_pricing_recommendation(
    pricing_power_score: float,
    input_cost_pressure: float,
    margin_trend: str,
    regime_label: str = "CHOPPY",
) -> Dict[str, Any]:
    """Matrix-based pricing recommendation with regime overlay."""
    high_power = pricing_power_score > 65
    low_power = pricing_power_score < 40
    high_pressure = input_cost_pressure > 60
    low_pressure = input_cost_pressure < 40

    if high_power and high_pressure:
        base_action = "raise_prices"
    elif high_power and not high_pressure:
        base_action = "hold"
    elif low_power and high_pressure:
        base_action = "defend_volume"
    else:
        base_action = "hold"

    # Regime overlay
    regime = regime_label.upper()
    action = base_action
    if regime == "HIGH_VOL":
        if action == "raise_prices":
            action = "hold"
    elif regime == "RECOVERY":
        if action == "hold" and high_power:
            action = "raise_prices"

    # Price increase potential (capped at 30%)
    if action == "raise_prices":
        potential = clamp(pricing_power_score / 100.0 * 0.25, 0.0, 0.30)
    elif action == "hold":
        potential = clamp(pricing_power_score / 100.0 * 0.05, 0.0, 0.10)
    else:
        potential = 0.0

    timing = "immediate" if action == "raise_prices" else "next_quarter" if action == "hold" else "monitor"

    # Action text — plain English
    if action == "raise_prices" and input_cost_pressure > 60:
        action_text = (
            "Raise prices now — your market position supports it and your "
            "input costs demand it. Target 6-8% at next opportunity."
        )
    elif action == "raise_prices":
        action_text = (
            "Your pricing power is strong and costs are well-controlled. "
            "Consider a 4-6% price increase at your next contract renewal."
        )
    elif pricing_power_score >= 40:
        action_text = (
            "Selective pricing: raise by 2-3% for new customers, hold for "
            "existing relationships where churn risk is higher."
        )
    else:
        action_text = (
            "Do not raise prices — focus on cost reduction. Identify the "
            "2 largest cost categories and target 10% reduction each."
        )

    # Supporting analysis
    trend_phrase = {
        "expanding": "margins are expanding — signal of growing pricing leverage",
        "stable": "margins are stable — pricing discipline is adequate",
        "compressing": "margins are compressing — cost or volume pressure detected",
    }.get(margin_trend, "margin trend is unknown")
    supporting_analysis = (
        f"Pricing power score of {pricing_power_score:.0f}/100 with {trend_phrase}. "
        f"Input cost pressure at {input_cost_pressure:.0f}/100 "
        f"({'elevated — pass through costs where possible' if input_cost_pressure > 60 else 'manageable'})."
    )

    return {
        "action": action,
        "rationale": _ACTION_RATIONALES.get(action, "Monitor pricing environment."),
        "price_increase_potential_pct": round(clamp(potential, 0.0, 0.30), 4),
        "timing": timing,
        "action_text": action_text,
        "supporting_analysis": supporting_analysis,
    }


def build_pricing_intelligence(
    entity_id: str,
    financials_history: List[Dict[str, Any]],
    sector: str = "Unknown",
    regime_label: str = "CHOPPY",
) -> PricingIntelligence:
    power = compute_pricing_power_score(financials_history, sector)
    pressure = compute_input_cost_pressure(financials_history)
    margin_trend = _classify_margin_trend(financials_history)
    rec = generate_pricing_recommendation(power, pressure, margin_trend, regime_label)
    return PricingIntelligence(
        entity_id=entity_id,
        as_of_date=dt.date.today(),
        pricing_power_score=power,
        recommended_action=rec["action"],
        price_increase_potential_pct=rec["price_increase_potential_pct"],
        margin_trend=margin_trend,
        input_cost_pressure_score=pressure,
        competitive_position=_classify_competitive_position(power),
        regime_pricing_context=_regime_pricing_context(regime_label),
    )
