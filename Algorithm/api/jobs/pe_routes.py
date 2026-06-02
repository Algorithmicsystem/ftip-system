"""Phase 3: PE and Corporate Intelligence API endpoints."""
from __future__ import annotations

import datetime as dt
from typing import Any, Dict, Optional

from fastapi import APIRouter, Query
from pydantic import BaseModel

router = APIRouter(prefix="/pe", tags=["pe"])


class EntityFinancialsIn(BaseModel):
    entity_id: str
    period_end: str          # ISO date string
    period_type: str = "quarterly"
    revenue: Optional[float] = None
    ebitda: Optional[float] = None
    net_income: Optional[float] = None
    total_debt: Optional[float] = None
    cash: Optional[float] = None
    capex: Optional[float] = None
    free_cash_flow: Optional[float] = None
    headcount: Optional[int] = None
    arr: Optional[float] = None


@router.post("/entity/financials")
def post_entity_financials(payload: EntityFinancialsIn) -> Dict[str, Any]:
    """Upsert periodic financials for a portfolio company."""
    from api.jobs.pe_intelligence import store_entity_financials
    period_end = dt.date.fromisoformat(payload.period_end)
    financials = {
        k: v for k, v in payload.model_dump().items()
        if k not in ("entity_id", "period_end", "period_type")
    }
    ok = store_entity_financials(
        payload.entity_id, period_end, financials, payload.period_type
    )
    return {
        "status": "stored" if ok else "failed",
        "entity_id": payload.entity_id,
        "period_end": payload.period_end,
    }


@router.get("/entity/{entity_id}/health")
def get_entity_health(entity_id: str) -> Dict[str, Any]:
    """Return health score and component breakdown for a portfolio company."""
    from api.jobs.pe_intelligence import compute_entity_health
    return compute_entity_health(entity_id)


@router.get("/entity/{entity_id}/exit-timing")
def get_exit_timing(entity_id: str) -> Dict[str, Any]:
    """Return exit readiness score and recommendation for a portfolio company."""
    from api.jobs.pe_intelligence import compute_exit_timing
    return compute_exit_timing(entity_id)


@router.get("/portfolio/{org_id}/overview")
def get_portfolio_overview(org_id: str) -> Dict[str, Any]:
    """Return all active portfolio companies with health scores."""
    from api.jobs.pe_intelligence import get_portfolio_overview
    return get_portfolio_overview(org_id)


@router.get("/portfolio/{org_id}/stress-alerts")
def get_stress_alerts(org_id: str) -> Dict[str, Any]:
    """Return distressed portfolio companies (health < 40)."""
    from api.jobs.pe_intelligence import get_portfolio_stress_alerts
    return get_portfolio_stress_alerts(org_id)
