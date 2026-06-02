"""Trial tenant onboarding — 14-day trial creation."""
from __future__ import annotations
import datetime as dt
import secrets
import string
from typing import Any, Dict

from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter(prefix="/onboarding", tags=["onboarding"])

_SECTOR_ENDPOINTS = {
    "investment": [
        {"path": "/prosperity/latest/signal?symbol=AAPL", "description": "Latest trading signal"},
        {"path": "/axiom/scores?symbol=AAPL", "description": "7-engine AXIOM composite score"},
        {"path": "/ops/intelligence", "description": "Daily CIO intelligence digest"},
    ],
    "pe": [
        {"path": "/pe/entity/financials (POST)", "description": "Upload portfolio company financials"},
        {"path": "/pe/entity/{id}/health", "description": "PE company health score"},
        {"path": "/pe/portfolio/{org}/overview", "description": "Portfolio overview"},
    ],
    "smb": [
        {"path": "/smb/entity/financials (POST)", "description": "Upload SMB monthly financials"},
        {"path": "/smb/entity/{id}/cash-flow-forecast", "description": "12-month cash flow forecast"},
        {"path": "/smb/entity/{id}/supplier-risks", "description": "Supplier risk analysis"},
    ],
    "all": [
        {"path": "/prosperity/latest/signal", "description": "Trading signal"},
        {"path": "/pe/entity/{id}/health", "description": "PE health score"},
        {"path": "/smb/entity/{id}/cash-flow-forecast", "description": "SMB forecast"},
    ],
}


class TrialRequest(BaseModel):
    org_name: str
    contact_email: str
    sector_preference: str = "investment"  # investment | pe | smb | all


@router.post("/trial")
def create_trial(payload: TrialRequest) -> Dict[str, Any]:
    """Create a 14-day trial tenant and return API key + getting-started guide."""
    from api.jobs.tenant_auth import register_tenant, _hash_key
    from api import db

    sector = payload.sector_preference if payload.sector_preference in _SECTOR_ENDPOINTS else "investment"
    tenant_id = f"trial_{secrets.token_hex(8)}"
    raw_key = "ftip_trial_" + "".join(secrets.choice(string.ascii_letters + string.digits) for _ in range(32))
    expires_at = dt.date.today() + dt.timedelta(days=14)

    ok = register_tenant(
        tenant_id=tenant_id,
        org_name=payload.org_name,
        raw_api_key=raw_key,
        tier="pro",
    )

    # Update expires_at
    if ok and db.db_write_enabled():
        try:
            db.safe_execute(
                "UPDATE api_tenants SET expires_at = %s WHERE tenant_id = %s",
                (expires_at, tenant_id),
            )
        except Exception:
            pass

    return {
        "status": "created" if ok else "db_disabled",
        "tenant_id": tenant_id,
        "api_key": raw_key,
        "tier": "pro",
        "expires_at": expires_at.isoformat(),
        "org_name": payload.org_name,
        "getting_started": {
            "header": "X-API-Key",
            "example": f"curl -H 'X-API-Key: {raw_key}' http://api.ftip.io/prosperity/latest/signal?symbol=AAPL",
            "recommended_endpoints": _SECTOR_ENDPOINTS[sector],
        },
    }
