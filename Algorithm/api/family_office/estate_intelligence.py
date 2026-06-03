"""Phase 14.4: Estate and Succession Intelligence."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from api.assistant.phase3.common import clamp
from api.family_office.multi_asset import PortfolioSnapshot


# ---------------------------------------------------------------------------
# Estate tax exposure
# ---------------------------------------------------------------------------

def compute_estate_tax_exposure(
    portfolio_value_usd: float,
    estate_exemption_usd: float = 13_610_000,
    state_exemption_usd: float = 0.0,
    family_discount_pct: float = 0.35,
) -> Dict[str, Any]:
    """Federal + state estate tax estimate with family discount."""
    taxable_value = portfolio_value_usd * (1.0 - family_discount_pct)
    federal_taxable = max(0.0, taxable_value - estate_exemption_usd)
    federal_tax = federal_taxable * 0.40

    state_taxable = max(0.0, taxable_value - state_exemption_usd) if state_exemption_usd > 0 else 0.0
    state_tax = state_taxable * 0.16

    total_tax = federal_tax + state_tax
    effective_rate = total_tax / portfolio_value_usd if portfolio_value_usd > 0 else 0.0
    net_to_heirs = portfolio_value_usd - total_tax

    return {
        "gross_estate_value_usd": round(portfolio_value_usd, 2),
        "taxable_estate_usd": round(taxable_value, 2),
        "federal_estate_tax_usd": round(federal_tax, 2),
        "state_estate_tax_usd": round(state_tax, 2),
        "total_estate_tax_usd": round(total_tax, 2),
        "effective_estate_tax_rate": round(effective_rate, 4),
        "net_to_heirs_usd": round(net_to_heirs, 2),
        "family_discount_applied_pct": round(family_discount_pct, 4),
    }


# ---------------------------------------------------------------------------
# Gifting intelligence
# ---------------------------------------------------------------------------

def compute_gifting_intelligence(
    portfolio_value_usd: float,
    annual_gift_exclusion: float = 18_000,
    n_recipients: int = 4,
    years: int = 10,
) -> Dict[str, Any]:
    """Annual exclusion gift capacity and estate tax savings."""
    annual_capacity = annual_gift_exclusion * n_recipients
    total_capacity = annual_capacity * years
    # Each dollar gifted removes it from estate at 40% marginal rate
    estate_tax_savings = total_capacity * 0.40
    estate_after_gifting = max(0.0, portfolio_value_usd - total_capacity)

    return {
        "annual_gift_capacity_usd": round(annual_capacity, 2),
        "total_gift_capacity_10yr_usd": round(total_capacity, 2),
        "estimated_estate_tax_savings_usd": round(estate_tax_savings, 2),
        "estate_value_after_gifting_usd": round(estate_after_gifting, 2),
        "n_recipients": n_recipients,
        "years": years,
        "annual_exclusion_per_recipient": annual_gift_exclusion,
    }


# ---------------------------------------------------------------------------
# Dynasty trust analysis
# ---------------------------------------------------------------------------

def compute_dynasty_trust_analysis(
    portfolio_value_usd: float,
    projected_growth_rate: float,
    trust_horizon_years: int = 100,
) -> Dict[str, Any]:
    """Compare dynasty trust vs traditional multi-generational transfer."""
    # Dynasty: assets grow untaxed for full horizon
    dynasty_terminal_value = portfolio_value_usd * ((1.0 + projected_growth_rate) ** trust_horizon_years)

    # Traditional: 3 estate tax events (each generation pays 40%, modeled as surviving 60%)
    traditional_terminal_value = portfolio_value_usd * (0.60 ** 3)

    dynasty_advantage = dynasty_terminal_value - traditional_terminal_value
    dynasty_advantage_pct = (
        dynasty_advantage / traditional_terminal_value if traditional_terminal_value > 0 else 0.0
    )

    # Breakeven: if growth = 0, dynasty still avoids 3× estate tax
    generation_intervals = [25, 50, 75]
    generation_values = {
        f"generation_{i+1}_traditional_usd": round(portfolio_value_usd * (0.60 ** (i + 1)), 2)
        for i in range(3)
    }

    return {
        "dynasty_terminal_value_usd": round(dynasty_terminal_value, 2),
        "traditional_terminal_value_usd": round(traditional_terminal_value, 2),
        "dynasty_advantage_usd": round(dynasty_advantage, 2),
        "dynasty_advantage_pct": round(dynasty_advantage_pct, 4),
        "trust_horizon_years": trust_horizon_years,
        "projected_growth_rate": round(projected_growth_rate, 4),
        "generation_values": generation_values,
    }


# ---------------------------------------------------------------------------
# Full estate report
# ---------------------------------------------------------------------------

def build_estate_intelligence_report(
    portfolio_value_usd: float,
    estate_exemption_usd: float = 13_610_000,
    state_exemption_usd: float = 0.0,
    family_discount_pct: float = 0.35,
    annual_gift_exclusion: float = 18_000,
    n_gift_recipients: int = 4,
    projected_growth_rate: float = 0.07,
    trust_horizon_years: int = 100,
) -> Dict[str, Any]:
    tax_exposure = compute_estate_tax_exposure(
        portfolio_value_usd, estate_exemption_usd, state_exemption_usd, family_discount_pct
    )
    gifting = compute_gifting_intelligence(
        portfolio_value_usd, annual_gift_exclusion, n_gift_recipients
    )
    dynasty = compute_dynasty_trust_analysis(
        portfolio_value_usd, projected_growth_rate, trust_horizon_years
    )

    strategies: List[str] = []
    if tax_exposure["total_estate_tax_usd"] > 0:
        strategies.append("annual_exclusion_gifting")
        strategies.append("irrevocable_life_insurance_trust")
        if tax_exposure["total_estate_tax_usd"] > 5_000_000:
            strategies.append("dynasty_trust")
            strategies.append("grantor_retained_annuity_trust")

    return {
        "estate_tax_exposure": tax_exposure,
        "gifting_intelligence": gifting,
        "dynasty_trust_analysis": dynasty,
        "recommended_strategies": strategies,
    }
