"""Phase 26: SMB intelligence from public market data (yfinance + AXIOM)."""
from __future__ import annotations

import datetime as dt
import logging
from typing import Any, Dict, List, Optional

from api.assistant.phase3.common import clamp

logger = logging.getLogger(__name__)


def compute_smb_intelligence(symbol: str) -> Dict[str, Any]:
    """Compute SMB-style intelligence for a public ticker using real fundamentals."""
    from api.pe.fundamental_loader import load_company_fundamentals

    sym = symbol.upper()
    fund = load_company_fundamentals(sym)

    gross_margin = float(fund.get("gross_margin") or 0.30)
    op_margin = float(fund.get("op_margin") or 0.10)
    fcf_margin = float(fund.get("fcf_margin") or 0.05)
    rev_growth = float(fund.get("revenue_growth_yoy") or 0.03)
    revenue_ttm = float(fund.get("revenue_ttm") or 0.0)
    dte = float(fund.get("debt_to_equity") or 0.0)
    axiom_dau = fund.get("axiom_dau")
    axiom_signal = fund.get("axiom_signal") or "HOLD"

    # DSCR approximation: EBITDA / estimated debt service
    ebitda_annual = revenue_ttm * op_margin if revenue_ttm > 0 else 0.0
    # Estimate annual debt service from D/E and assumed 6% rate on debt
    estimated_debt = (dte / 100.0) * (revenue_ttm * 0.5) if dte > 0 and revenue_ttm > 0 else 0.0
    annual_debt_service = estimated_debt * 0.06 if estimated_debt > 0 else 1.0
    if ebitda_annual > 0 and annual_debt_service > 0:
        dscr = round(ebitda_annual / annual_debt_service, 2)
    else:
        dscr = 3.0  # no debt = strong

    # Credit score (0-100)
    credit_base = 50.0
    if dscr >= 2.0:
        credit_base += 20.0
    elif dscr >= 1.25:
        credit_base += 10.0
    if gross_margin >= 0.40:
        credit_base += 15.0
    elif gross_margin >= 0.25:
        credit_base += 8.0
    if rev_growth >= 0.05:
        credit_base += 10.0
    if dte < 50:
        credit_base += 5.0
    elif dte > 200:
        credit_base -= 15.0
    credit_score = round(clamp(credit_base, 0.0, 100.0), 1)

    # Max additional debt
    if ebitda_annual > 0:
        max_leverage_ebitda = 3.5 if credit_score >= 60 else 2.5
        max_debt_capacity = ebitda_annual * max_leverage_ebitda
        max_additional_debt = max(0.0, max_debt_capacity - estimated_debt)
    else:
        max_additional_debt = 0.0

    # Pricing power score
    pricing_power = 50.0
    if gross_margin >= 0.50:
        pricing_power += 20.0
    elif gross_margin >= 0.35:
        pricing_power += 10.0
    if rev_growth >= 0.10:
        pricing_power += 15.0
    elif rev_growth >= 0.05:
        pricing_power += 8.0
    if op_margin >= 0.15:
        pricing_power += 10.0
    pricing_power_score = round(clamp(pricing_power, 0.0, 100.0), 1)

    # Revenue growth pct
    revenue_growth_pct = round(rev_growth * 100.0, 2)

    # Cash flow forecast (simplified 3-period)
    monthly_fcf = revenue_ttm * fcf_margin / 12.0 if revenue_ttm > 0 else 0.0
    cash_flow_forecast: List[Dict[str, Any]] = []
    for i in range(1, 4):
        projected = monthly_fcf * (1.0 + rev_growth / 12.0) ** i
        cash_flow_forecast.append({
            "month": i,
            "projected_fcf": round(projected, 0),
        })

    # Recommendation
    if credit_score >= 70 and pricing_power_score >= 65:
        recommendation = "strong_buy_signals"
    elif credit_score >= 55 and pricing_power_score >= 50:
        recommendation = "monitor_and_hold"
    else:
        recommendation = "caution_review_needed"

    # Pricing intelligence
    if pricing_power_score >= 70:
        pricing_action = "raise_prices"
        price_increase_pct = 5.0 if rev_growth >= 0.05 else 3.0
    elif pricing_power_score >= 50:
        pricing_action = "maintain_prices"
        price_increase_pct = 2.0
    else:
        pricing_action = "protect_volume"
        price_increase_pct = 0.0

    return {
        "symbol": sym,
        "as_of_date": dt.date.today().isoformat(),
        "dscr": dscr,
        "credit_score": credit_score,
        "max_additional_debt_usd": round(max_additional_debt, 0),
        "pricing_power_score": pricing_power_score,
        "gross_margin_pct": round(gross_margin * 100.0, 2),
        "revenue_growth_pct": revenue_growth_pct,
        "recommendation": recommendation,
        "axiom_dau": axiom_dau,
        "axiom_signal": axiom_signal,
        "cash_flow_forecast": cash_flow_forecast,
        "pricing_intelligence": {
            "pricing_action": pricing_action,
            "price_increase_potential_pct": price_increase_pct,
            "gross_margin_pct": round(gross_margin * 100.0, 2),
            "revenue_growth_pct": revenue_growth_pct,
        },
        "credit": {
            "dscr": dscr,
            "credit_score": credit_score,
            "max_additional_debt_usd": round(max_additional_debt, 0),
        },
        "sector": fund.get("sector") or "Unknown",
        "market_cap": fund.get("market_cap"),
        "revenue_ttm": revenue_ttm,
    }
