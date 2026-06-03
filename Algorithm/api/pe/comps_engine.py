"""Phase 13.3: Comparable Transaction Analysis Engine."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

from api.assistant.phase3.common import clamp

# Sector median trading multiples (update quarterly)
SECTOR_TRADING_MULTIPLES: Dict[str, Dict[str, float]] = {
    "Technology":             {"ev_ebitda": 18.0, "ev_revenue": 4.5,  "pe": 28.0},
    "Healthcare":             {"ev_ebitda": 14.0, "ev_revenue": 3.2,  "pe": 22.0},
    "Finance":                {"ev_ebitda": 10.0, "ev_revenue": 2.8,  "pe": 14.0},
    "Consumer Staples":       {"ev_ebitda": 13.0, "ev_revenue": 1.8,  "pe": 20.0},
    "Consumer Discretionary": {"ev_ebitda": 11.0, "ev_revenue": 1.5,  "pe": 18.0},
    "Energy":                 {"ev_ebitda":  7.0, "ev_revenue": 1.2,  "pe": 12.0},
    "Industrials":            {"ev_ebitda": 12.0, "ev_revenue": 1.8,  "pe": 18.0},
    "Materials":              {"ev_ebitda":  9.0, "ev_revenue": 1.4,  "pe": 15.0},
    "Utilities":              {"ev_ebitda": 11.0, "ev_revenue": 2.5,  "pe": 17.0},
    "Real Estate":            {"ev_ebitda": 15.0, "ev_revenue": 5.0,  "pe": 35.0},
    "Unknown":                {"ev_ebitda": 12.0, "ev_revenue": 2.5,  "pe": 18.0},
}

# PE buyout premium over public trading comps
PE_PREMIUM = 0.35


@dataclass
class CompAnalysis:
    target_symbol: str
    comps_methodology: str         # "sector_trading_comps" | "transaction_comps"
    ev_ebitda_range: Dict[str, float]
    ev_revenue_range: Dict[str, float]
    pe_ratio_range: Dict[str, float]
    quality_adjusted_multiple: float
    implied_valuation_range: Dict[str, float]  # in $M
    upside_to_consensus: float
    confidence: str                # "high" | "medium" | "low"


def compute_quality_adjusted_multiple(
    base_multiple: float,
    eis_score: float,
    caps_score: float,
    schilit_score: float = 100.0,
) -> float:
    """Apply AXIOM quality premium/discount to a base valuation multiple."""
    quality_adjustment = (
        (eis_score - 50.0) / 50.0 * 0.15
        + (caps_score - 50.0) / 50.0 * 0.20
        + (schilit_score - 50.0) / 50.0 * 0.10
    )
    return round(base_multiple * (1.0 + quality_adjustment), 4)


def run_comps_analysis(
    entity_id: str,
    financials: Dict,
    sector: str,
    eis_score: float,
    caps_score: float,
    schilit_score: float = 100.0,
) -> CompAnalysis:
    """Build full comparable transaction analysis for a PE entity."""
    sector_multiples = SECTOR_TRADING_MULTIPLES.get(sector, SECTOR_TRADING_MULTIPLES["Unknown"])

    # Step 2: Apply PE premium
    transaction_ev_ebitda = sector_multiples["ev_ebitda"] * (1.0 + PE_PREMIUM)
    transaction_ev_revenue = sector_multiples["ev_revenue"] * (1.0 + PE_PREMIUM)
    transaction_pe = sector_multiples["pe"] * (1.0 + PE_PREMIUM)

    # Step 3: Quality-adjusted multiple
    quality_ev_ebitda = compute_quality_adjusted_multiple(
        transaction_ev_ebitda, eis_score, caps_score, schilit_score
    )
    quality_ev_revenue = compute_quality_adjusted_multiple(
        transaction_ev_revenue, eis_score, caps_score, schilit_score
    )
    quality_pe = compute_quality_adjusted_multiple(
        transaction_pe, eis_score, caps_score, schilit_score
    )

    # Step 4: Build ranges
    def _range(median: float) -> Dict[str, float]:
        return {
            "low": round(median * 0.85, 2),
            "median": round(median, 2),
            "high": round(median * 1.15, 2),
        }

    ev_ebitda_range = _range(quality_ev_ebitda)
    ev_revenue_range = _range(quality_ev_revenue)
    pe_ratio_range = _range(quality_pe)

    # Step 5: Implied valuation from EBITDA
    ebitda = float(financials.get("ebitda") or 0.0)
    revenue = float(financials.get("revenue") or 0.0)
    current_market_cap = float(financials.get("market_cap") or 0.0)

    if ebitda > 0:
        implied_low = ebitda * ev_ebitda_range["low"]
        implied_median = ebitda * ev_ebitda_range["median"]
        implied_high = ebitda * ev_ebitda_range["high"]
    elif revenue > 0:
        implied_low = revenue * ev_revenue_range["low"]
        implied_median = revenue * ev_revenue_range["median"]
        implied_high = revenue * ev_revenue_range["high"]
    else:
        implied_low = implied_median = implied_high = 0.0

    implied_valuation_range = {
        "low": round(implied_low, 2),
        "median": round(implied_median, 2),
        "high": round(implied_high, 2),
    }

    # Step 6: Upside
    if current_market_cap > 0 and implied_median > 0:
        upside = (implied_median - current_market_cap) / current_market_cap
    else:
        upside = 0.0

    # Confidence
    if eis_score > 65 and schilit_score > 75:
        confidence = "high"
    elif schilit_score < 50:
        confidence = "low"
    else:
        confidence = "medium"

    return CompAnalysis(
        target_symbol=entity_id,
        comps_methodology="transaction_comps",
        ev_ebitda_range=ev_ebitda_range,
        ev_revenue_range=ev_revenue_range,
        pe_ratio_range=pe_ratio_range,
        quality_adjusted_multiple=quality_ev_ebitda,
        implied_valuation_range=implied_valuation_range,
        upside_to_consensus=round(upside, 4),
        confidence=confidence,
    )
