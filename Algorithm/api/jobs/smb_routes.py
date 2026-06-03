"""Phase 4 + Phase 15: SMB Intelligence API endpoints."""
from __future__ import annotations

import datetime as dt
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel

from api.jobs.tenant_auth import require_tier

router = APIRouter(
    prefix="/smb",
    tags=["smb"],
    dependencies=[Depends(require_tier("enterprise"))],
)


class SMBFinancialsIn(BaseModel):
    entity_id: str
    month_end: str       # ISO date string (last day of month)
    revenue: Optional[float] = None
    cogs: Optional[float] = None
    operating_expenses: Optional[float] = None
    net_income: Optional[float] = None
    cash_balance: Optional[float] = None
    accounts_receivable: Optional[float] = None
    accounts_payable: Optional[float] = None
    inventory: Optional[float] = None
    payroll: Optional[float] = None
    top_supplier_concentration: Optional[float] = None


@router.post("/entity/financials")
def post_smb_financials(payload: SMBFinancialsIn) -> Dict[str, Any]:
    """Upsert monthly financials for an SMB entity."""
    from api.jobs.smb_intelligence import store_smb_financials
    month_end = dt.date.fromisoformat(payload.month_end)
    financials = {
        k: v for k, v in payload.model_dump().items()
        if k not in ("entity_id", "month_end")
    }
    ok = store_smb_financials(payload.entity_id, month_end, financials)
    return {
        "status": "stored" if ok else "failed",
        "entity_id": payload.entity_id,
        "month_end": payload.month_end,
    }


@router.get("/entity/{entity_id}/cash-flow-forecast")
def get_cash_flow_forecast(
    entity_id: str,
    horizon: int = Query(default=12, ge=1, le=24),
) -> Dict[str, Any]:
    """Return 12-month cash flow forecast for an SMB entity."""
    from api.jobs.smb_intelligence import forecast_cash_flow
    return forecast_cash_flow(entity_id, horizon_months=horizon)


@router.get("/entity/{entity_id}/supplier-risks")
def get_supplier_risks(entity_id: str) -> Dict[str, Any]:
    """Return supplier concentration and cash buffer risk analysis."""
    from api.jobs.smb_intelligence import compute_supplier_risks
    return compute_supplier_risks(entity_id)


# ---------------------------------------------------------------------------
# Phase 15 endpoints
# ---------------------------------------------------------------------------

@router.get("/entity/{entity_id}/pricing-intelligence")
def get_pricing_intelligence(
    entity_id: str,
    sector: str = Query(default="Unknown"),
    regime: str = Query(default="CHOPPY"),
) -> Dict[str, Any]:
    """Pricing power score and recommendations for an SMB entity."""
    from api import db
    from api.jobs.smb_intelligence import _load_monthly_history
    from api.smb.pricing_intelligence import build_pricing_intelligence

    history = _load_monthly_history(entity_id, n=12)
    # Normalize monthly rows to pricing_intelligence expected format
    hist_fmt = []
    for i, row in enumerate(history):
        rev = row.get("revenue") or 0.0
        cogs = row.get("cogs") or 0.0
        gm = (1.0 - cogs / rev) if rev > 0 else 0.40
        entry: Dict[str, Any] = {
            "gross_margin": gm,
            "accounts_receivable": row.get("accounts_receivable"),
        }
        if i > 0:
            prev = history[i - 1]
            prev_rev = prev.get("revenue") or rev
            entry["revenue_growth_yoy"] = (rev - prev_rev) / prev_rev if prev_rev > 0 else 0.0
            prev_cogs = prev.get("cogs") or cogs
            entry["cogs_growth_yoy"] = (cogs - prev_cogs) / prev_cogs if prev_cogs > 0 else 0.0
        hist_fmt.append(entry)

    intel = build_pricing_intelligence(entity_id, hist_fmt, sector, regime)
    return {
        "entity_id": intel.entity_id,
        "as_of_date": intel.as_of_date.isoformat(),
        "pricing_power_score": intel.pricing_power_score,
        "recommended_action": intel.recommended_action,
        "price_increase_potential_pct": intel.price_increase_potential_pct,
        "margin_trend": intel.margin_trend,
        "input_cost_pressure_score": intel.input_cost_pressure_score,
        "competitive_position": intel.competitive_position,
        "regime_pricing_context": intel.regime_pricing_context,
    }


@router.get("/entity/{entity_id}/customer-concentration")
def get_customer_concentration(
    entity_id: str,
    customer_revenues: str = Query(default="", description="customer_id:amount pairs, comma-separated"),
    customer_tickers: str = Query(default="", description="customer_id:ticker pairs, comma-separated"),
) -> Dict[str, Any]:
    """Customer concentration risk for an SMB entity."""
    from api.smb.customer_concentration import build_customer_concentration_report
    from api import db

    # Parse customer_revenues: "c1:50000,c2:30000"
    revenue_breakdown: Dict[str, float] = {}
    if customer_revenues:
        for pair in customer_revenues.split(","):
            parts = pair.strip().split(":")
            if len(parts) == 2:
                try:
                    revenue_breakdown[parts[0].strip()] = float(parts[1].strip())
                except ValueError:
                    pass

    # Fallback: load annual revenue from DB
    annual_revenue = 0.0
    if db.db_read_enabled():
        try:
            row = db.safe_fetchone(
                "SELECT SUM(revenue) FROM smb_monthly_financials WHERE entity_id = %s",
                (entity_id,),
            )
            if row and row[0]:
                annual_revenue = float(row[0])
        except Exception:
            pass

    # Parse customer_tickers: "c1:AAPL,c2:MSFT"
    ticker_map: Dict[str, str] = {}
    if customer_tickers:
        for pair in customer_tickers.split(","):
            parts = pair.strip().split(":")
            if len(parts) == 2:
                ticker_map[parts[0].strip()] = parts[1].strip()

    report = build_customer_concentration_report(
        entity_id, revenue_breakdown, annual_revenue, ticker_map
    )
    return {
        "entity_id": report.entity_id,
        "concentration_risk_score": report.concentration_risk_score,
        "concentration_label": report.concentration_label,
        "top_customer_revenue_pct": report.top_customer_revenue_pct,
        "top_3_customer_revenue_pct": report.top_3_customer_revenue_pct,
        "estimated_revenue_at_risk": report.estimated_revenue_at_risk,
        "revenue_diversification_score": report.revenue_diversification_score,
        "customer_axiom_scores": report.customer_axiom_scores,
        "alerts": report.alerts,
    }


@router.get("/entity/{entity_id}/working-capital")
def get_working_capital(
    entity_id: str,
    sector: str = Query(default="Unknown"),
) -> Dict[str, Any]:
    """Working capital optimization analysis for an SMB entity."""
    from api.jobs.smb_intelligence import _load_monthly_history
    from api.smb.working_capital import build_working_capital_analysis

    history = _load_monthly_history(entity_id, n=3)
    if not history:
        return {"status": "no_data", "entity_id": entity_id}

    # Use most recent month
    financials = history[0]
    analysis = build_working_capital_analysis(entity_id, financials, sector)
    return {
        "entity_id": analysis.entity_id,
        "current_cash_conversion_cycle": analysis.current_cash_conversion_cycle,
        "optimal_cash_conversion_cycle": analysis.optimal_cash_conversion_cycle,
        "working_capital_gap_days": analysis.working_capital_gap_days,
        "trapped_working_capital_usd": analysis.trapped_working_capital_usd,
        "ar_days": analysis.ar_days,
        "ap_days": analysis.ap_days,
        "inventory_days": analysis.inventory_days,
        "potential_cash_release_usd": analysis.potential_cash_release_usd,
        "recommendations": analysis.recommendations,
    }


@router.get("/entity/{entity_id}/credit-intelligence")
def get_credit_intelligence(
    entity_id: str,
    regime: str = Query(default="CHOPPY"),
) -> Dict[str, Any]:
    """Credit capacity and timing recommendation for an SMB entity."""
    from api.jobs.smb_intelligence import _load_monthly_history
    from api.smb.credit_intelligence import build_credit_intelligence

    history = _load_monthly_history(entity_id, n=3)
    if not history:
        return {"status": "no_data", "entity_id": entity_id}

    # Approximate annual EBITDA from monthly data
    recent = history[0]
    rev = float(recent.get("revenue") or 0.0)
    cogs = float(recent.get("cogs") or 0.0)
    opex = float(recent.get("operating_expenses") or 0.0)
    payroll = float(recent.get("payroll") or 0.0)
    ebitda_monthly = rev - cogs - opex
    ebitda_annual = ebitda_monthly * 12
    revenue_annual = rev * 12
    ebitda_margin = ebitda_annual / revenue_annual if revenue_annual > 0 else 0.10

    financials = {
        "revenue": revenue_annual,
        "ebitda": ebitda_annual,
        "ebitda_margin": ebitda_margin,
        "revenue_growth_yoy": 0.05,  # conservative default
        "total_debt": 0.0,
        "annual_debt_service": 0.0,
    }

    intel = build_credit_intelligence(entity_id, financials, regime)
    return {
        "entity_id": intel.entity_id,
        "borrowing_capacity_score": intel.borrowing_capacity_score,
        "estimated_max_loan_usd": intel.estimated_max_loan_usd,
        "optimal_loan_structure": intel.optimal_loan_structure,
        "rate_environment": intel.rate_environment,
        "rate_environment_score": intel.rate_environment_score,
        "credit_timing_recommendation": intel.credit_timing_recommendation,
        "dscr": intel.dscr,
        "ltv_estimate": intel.ltv_estimate,
    }


@router.get("/entity/{entity_id}/sector-intelligence")
def get_sector_intelligence(
    entity_id: str,
    sector: str = Query(default=""),
) -> Dict[str, Any]:
    """Sector-specific SMB intelligence module."""
    from api.jobs.smb_intelligence import _load_monthly_history
    from api.smb.sector_modules import get_sector_module

    history = _load_monthly_history(entity_id, n=3)
    financials = history[0] if history else {}
    financials["entity_id"] = entity_id

    if not sector:
        return {"status": "no_sector_specified", "entity_id": entity_id}

    return get_sector_module(sector, financials)


@router.get("/entity/{entity_id}/intelligence-dashboard")
def get_intelligence_dashboard(
    entity_id: str,
    sector: str = Query(default="Unknown"),
    regime: str = Query(default="CHOPPY"),
) -> Dict[str, Any]:
    """Unified SMB intelligence dashboard."""
    from api.jobs.smb_intelligence import (
        _load_monthly_history, forecast_cash_flow, compute_supplier_risks
    )
    from api.smb.pricing_intelligence import build_pricing_intelligence
    from api.smb.working_capital import build_working_capital_analysis
    from api.smb.credit_intelligence import build_credit_intelligence
    from api.smb.sector_modules import get_sector_module

    history = _load_monthly_history(entity_id, n=12)
    if not history:
        return {"status": "no_data", "entity_id": entity_id}

    recent = history[0]
    rev = float(recent.get("revenue") or 0.0)
    cogs = float(recent.get("cogs") or 0.0)
    ebitda_monthly = rev - cogs - float(recent.get("operating_expenses") or 0.0)

    # Pricing
    hist_fmt = []
    for i, row in enumerate(history):
        r = row.get("revenue") or 0.0
        c = row.get("cogs") or 0.0
        gm = (1.0 - c / r) if r > 0 else 0.40
        entry: Dict[str, Any] = {"gross_margin": gm, "accounts_receivable": row.get("accounts_receivable")}
        if i > 0:
            prev_r = history[i - 1].get("revenue") or r
            entry["revenue_growth_yoy"] = (r - prev_r) / prev_r if prev_r > 0 else 0.0
        hist_fmt.append(entry)

    pricing = build_pricing_intelligence(entity_id, hist_fmt, sector, regime)

    # Working capital
    wc = build_working_capital_analysis(entity_id, recent, sector)

    # Credit
    financials_credit = {
        "revenue": rev * 12,
        "ebitda": ebitda_monthly * 12,
        "ebitda_margin": ebitda_monthly / rev if rev > 0 else 0.10,
        "revenue_growth_yoy": 0.05,
        "total_debt": 0.0,
        "annual_debt_service": 0.0,
    }
    credit = build_credit_intelligence(entity_id, financials_credit, regime)

    # Cash flow forecast
    forecast = forecast_cash_flow(entity_id)
    cf_status = forecast.get("runway_status", "unknown")

    # Sector module
    sector_data = get_sector_module(sector, {**recent, "entity_id": entity_id})
    sector_score = 0.0
    if sector_data.get("status") == "ok":
        for key in ("restaurant_intelligence_score", "manufacturing_intelligence_score", "profservices_intelligence_score"):
            if key in sector_data:
                sector_score = sector_data[key]
                break

    # Overall health score
    scores = [
        pricing.pricing_power_score,
        credit.borrowing_capacity_score,
        credit.rate_environment_score,
    ]
    if sector_score > 0:
        scores.append(sector_score)
    overall_health = round(sum(scores) / len(scores), 2)

    # Top risk and opportunity
    risks = []
    if pricing.input_cost_pressure_score > 65:
        risks.append("Input cost inflation squeezing margins")
    if wc.working_capital_gap_days > 30:
        risks.append(f"Working capital gap of {wc.working_capital_gap_days:.0f} days")
    if credit.borrowing_capacity_score < 40:
        risks.append("Weak credit capacity limits growth financing")
    top_risk = risks[0] if risks else "No critical risks identified"

    opportunities = []
    if pricing.recommended_action == "raise_prices":
        opportunities.append(f"Raise prices {pricing.price_increase_potential_pct*100:.1f}% — pricing power supports it")
    if wc.potential_cash_release_usd > 0:
        opportunities.append(f"Release ${wc.potential_cash_release_usd:,.0f} from working capital optimization")
    if credit.credit_timing_recommendation == "borrow_now":
        opportunities.append("Favorable credit conditions — opportunistic borrowing window open")
    top_opportunity = opportunities[0] if opportunities else "Maintain current strategy"

    return {
        "entity_id": entity_id,
        "as_of_date": dt.date.today().isoformat(),
        "overall_health_score": overall_health,
        "cash_flow_status": cf_status,
        "pricing_action": pricing.recommended_action,
        "top_risk": top_risk,
        "top_opportunity": top_opportunity,
        "modules": {
            "pricing": {
                "score": pricing.pricing_power_score,
                "action": pricing.recommended_action,
            },
            "working_capital": {
                "trapped_usd": wc.trapped_working_capital_usd,
                "recommendation_count": len(wc.recommendations),
            },
            "credit": {
                "capacity_score": credit.borrowing_capacity_score,
                "timing": credit.credit_timing_recommendation,
            },
            "sector_module": {
                "score": sector_score,
                "sector": sector,
            },
        },
    }
