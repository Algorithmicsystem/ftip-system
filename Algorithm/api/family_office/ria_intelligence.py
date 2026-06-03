"""Phase 14.5: RIA Client Intelligence."""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from api.assistant.phase3.common import clamp
from api.family_office.multi_asset import PortfolioSnapshot


@dataclass
class RIAClientProfile:
    client_id: str
    client_name: str
    portfolio_value_usd: float
    risk_tolerance: str             # conservative, moderate, aggressive
    time_horizon_years: float
    income_need_annual: float
    tax_bracket: float              # e.g. 0.37
    esg_preference: bool
    axiom_score: Optional[float] = None
    last_review_date: Optional[dt.date] = None
    sri_score: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Review triggers
# ---------------------------------------------------------------------------

def compute_client_review_trigger(
    client: RIAClientProfile,
    axiom_regime: str,
    sri_score: float,
) -> Dict[str, Any]:
    triggers: List[str] = []

    if sri_score > 70:
        triggers.append("systemic_risk_elevated")
    if axiom_regime.upper() in ("HIGH_VOL", "compensation_capture"):
        triggers.append("adverse_market_regime")
    if (client.axiom_score or 100.0) < 40:
        triggers.append("low_axiom_score")
    if client.last_review_date is not None:
        days_since = (dt.date.today() - client.last_review_date).days
        if days_since > 90:
            triggers.append("review_overdue")
    else:
        triggers.append("review_overdue")

    if sri_score > 80:
        urgency = "urgent"
    elif triggers:
        urgency = "elevated"
    else:
        urgency = "routine"

    return {
        "client_id": client.client_id,
        "triggers": triggers,
        "urgency": urgency,
        "review_recommended": len(triggers) > 0 or urgency != "routine",
        "sri_score": round(sri_score, 2),
        "axiom_regime": axiom_regime,
    }


# ---------------------------------------------------------------------------
# Client brief
# ---------------------------------------------------------------------------

def generate_client_brief(
    client: RIAClientProfile,
    portfolio: PortfolioSnapshot,
    axiom_context: Dict[str, Any],
) -> Dict[str, Any]:
    axiom_score = axiom_context.get("weighted_axiom_score", 50.0)
    regime = axiom_context.get("regime_label", "UNKNOWN")
    sri = axiom_context.get("sri_score", 50.0)

    review_info = compute_client_review_trigger(client, regime, float(sri))

    income_yield = client.income_need_annual / portfolio.total_value_usd if portfolio.total_value_usd > 0 else 0.0
    income_coverage = (
        (portfolio.fixed_income_weight + portfolio.cash_weight) * portfolio.total_value_usd
        * 0.04  # proxy 4% yield on income assets
    ) / client.income_need_annual if client.income_need_annual > 0 else 999.0

    if axiom_score >= 70:
        market_outlook = "Favorable — AXIOM signals support current positioning"
    elif axiom_score >= 50:
        market_outlook = "Neutral — monitor for regime transitions"
    else:
        market_outlook = "Cautious — AXIOM signals elevated risk; review allocation"

    key_messages: List[str] = []
    if review_info["urgency"] == "urgent":
        key_messages.append(f"Urgent review needed: {', '.join(review_info['triggers'])}")
    if income_coverage < 1.0:
        key_messages.append(
            f"Income gap: current yield covers {income_coverage*100:.0f}% of annual need "
            f"(${client.income_need_annual:,.0f})"
        )
    if client.esg_preference:
        key_messages.append("ESG screening active — verify new positions align with mandate")
    if not key_messages:
        key_messages.append("Portfolio on track — no immediate action required")

    return {
        "client_id": client.client_id,
        "client_name": client.client_name,
        "portfolio_value_usd": round(portfolio.total_value_usd, 2),
        "axiom_score": round(axiom_score, 2),
        "market_outlook": market_outlook,
        "review_urgency": review_info["urgency"],
        "review_triggers": review_info["triggers"],
        "income_coverage_ratio": round(min(income_coverage, 99.0), 4),
        "equity_weight": round(portfolio.equity_weight, 4),
        "fixed_income_weight": round(portfolio.fixed_income_weight, 4),
        "cash_weight": round(portfolio.cash_weight, 4),
        "key_messages": key_messages,
        "regime": regime,
    }


# ---------------------------------------------------------------------------
# Book analytics
# ---------------------------------------------------------------------------

def compute_ria_book_analytics(
    clients: List[RIAClientProfile],
    portfolios: Optional[List[PortfolioSnapshot]] = None,
) -> Dict[str, Any]:
    if not clients:
        return {
            "total_aum": 0.0,
            "avg_axiom_score": 50.0,
            "clients_needing_review": 0,
            "risk_distribution": {},
            "regime_alignment": {},
            "themes": [],
        }

    total_aum = sum(c.portfolio_value_usd for c in clients)
    scores = [c.axiom_score for c in clients if c.axiom_score is not None]
    avg_score = sum(scores) / len(scores) if scores else 50.0

    risk_distribution: Dict[str, int] = {"conservative": 0, "moderate": 0, "aggressive": 0}
    for c in clients:
        bucket = c.risk_tolerance if c.risk_tolerance in risk_distribution else "moderate"
        risk_distribution[bucket] += 1

    clients_needing_review = sum(
        1 for c in clients
        if (c.last_review_date is None or (dt.date.today() - c.last_review_date).days > 90)
        or (c.sri_score or 0.0) > 70
        or (c.axiom_score or 100.0) < 40
    )

    esg_count = sum(1 for c in clients if c.esg_preference)
    themes: List[str] = []
    if esg_count / len(clients) > 0.30:
        themes.append("esg_integration")
    if sum(c.portfolio_value_usd for c in clients if c.risk_tolerance == "conservative") / total_aum > 0.40:
        themes.append("capital_preservation_focus")
    if avg_score < 45:
        themes.append("defensive_repositioning_needed")
    elif avg_score > 70:
        themes.append("growth_tilted_book")

    return {
        "total_aum": round(total_aum, 2),
        "avg_axiom_score": round(avg_score, 2),
        "clients_needing_review": clients_needing_review,
        "total_clients": len(clients),
        "risk_distribution": risk_distribution,
        "themes": themes,
        "esg_client_count": esg_count,
    }
