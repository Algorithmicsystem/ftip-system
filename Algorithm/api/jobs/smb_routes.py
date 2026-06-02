"""Phase 4: SMB Intelligence API endpoints."""
from __future__ import annotations

import datetime as dt
from typing import Any, Dict, Optional

from fastapi import APIRouter, Query
from pydantic import BaseModel

router = APIRouter(prefix="/smb", tags=["smb"])


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
