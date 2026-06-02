"""Phase 6: Commercial readiness endpoints — tenant info and API documentation."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel

router = APIRouter(tags=["platform"])


# ---------------------------------------------------------------------------
# GET /auth/tenant — current tenant info from API key
# ---------------------------------------------------------------------------

@router.get("/auth/tenant")
def get_tenant_info(request: Request) -> Dict[str, Any]:
    """Return tier and access details for the current API key."""
    from api.jobs.tenant_auth import _get_raw_api_key, resolve_tenant, get_tier_limits
    raw_key = _get_raw_api_key(request)
    if not raw_key:
        return {
            "status": "no_key",
            "message": "No API key provided. Pass X-API-Key header or Authorization: Bearer <key>.",
        }
    tenant = resolve_tenant(raw_key)
    if tenant is None:
        return {"status": "invalid_key", "message": "API key not recognised or expired."}

    limits = get_tier_limits(tenant["tier"])
    return {
        "status": "ok",
        "tenant_id": tenant["tenant_id"],
        "org_name": tenant["org_name"],
        "tier": tenant["tier"],
        "allowed_sectors": tenant["allowed_sectors"],
        "rpm_limit": tenant["rpm_limit"],
        "tier_description": limits["description"],
        "allowed_endpoint_prefixes": limits["allowed_prefixes"],
        "expires_at": tenant["expires_at"],
    }


# ---------------------------------------------------------------------------
# GET /api-docs — structured endpoint catalogue
# ---------------------------------------------------------------------------

_ENDPOINT_CATALOGUE: List[Dict[str, Any]] = [
    # Signal / Prosperity
    {
        "path": "/prosperity/latest/signal",
        "method": "GET",
        "tier": "free",
        "description": "Latest trading signal for a symbol (official v1 surface)",
        "params": [{"name": "symbol", "required": True}, {"name": "lookback", "required": False}],
    },
    {
        "path": "/prosperity/latest/features",
        "method": "GET",
        "tier": "free",
        "description": "Latest feature bundle for a symbol",
        "params": [{"name": "symbol", "required": True}],
    },
    # AXIOM
    {
        "path": "/axiom/scores",
        "method": "GET",
        "tier": "free",
        "description": "AXIOM 7-engine composite score for a symbol",
        "params": [{"name": "symbol", "required": True}, {"name": "as_of_date", "required": False}],
    },
    # Ops / Regime
    {
        "path": "/ops/intelligence",
        "method": "GET",
        "tier": "pro",
        "description": "CIO daily intelligence digest: top opportunities, sector rotation, IC health",
        "params": [{"name": "as_of_date", "required": False}, {"name": "top_n", "required": False}],
    },
    {
        "path": "/ops/regime/analogs",
        "method": "GET",
        "tier": "pro",
        "description": "Closest historical regime analogs with sector return attribution",
        "params": [
            {"name": "regime_label", "required": True},
            {"name": "vix", "required": False},
            {"name": "cape", "required": False},
            {"name": "limit", "required": False},
        ],
    },
    # Linkage
    {
        "path": "/linkage/peers/{symbol}",
        "method": "GET",
        "tier": "pro",
        "description": "Active sector peers with current AXIOM scores",
        "params": [{"name": "symbol", "required": True}, {"name": "as_of_date", "required": False}],
    },
    {
        "path": "/linkage/stress-propagation/{symbol}",
        "method": "GET",
        "tier": "pro",
        "description": "Fragility stress transmission to linked peers",
        "params": [{"name": "symbol", "required": True}, {"name": "as_of_date", "required": False}],
    },
    # PE
    {
        "path": "/pe/entity/financials",
        "method": "POST",
        "tier": "enterprise",
        "description": "Upsert periodic financials for a PE portfolio company",
        "params": [{"name": "body", "required": True, "type": "EntityFinancialsIn"}],
    },
    {
        "path": "/pe/entity/{entity_id}/health",
        "method": "GET",
        "tier": "enterprise",
        "description": "Health score with EBITDA, leverage, FCF, and growth components",
        "params": [{"name": "entity_id", "required": True}],
    },
    {
        "path": "/pe/entity/{entity_id}/exit-timing",
        "method": "GET",
        "tier": "enterprise",
        "description": "Exit readiness score and recommendation (hold / monitor / exit)",
        "params": [{"name": "entity_id", "required": True}],
    },
    {
        "path": "/pe/portfolio/{org_id}/overview",
        "method": "GET",
        "tier": "enterprise",
        "description": "All active portfolio companies with health scores for an org",
        "params": [{"name": "org_id", "required": True}],
    },
    {
        "path": "/pe/portfolio/{org_id}/stress-alerts",
        "method": "GET",
        "tier": "enterprise",
        "description": "Distressed portfolio companies (health score < 40)",
        "params": [{"name": "org_id", "required": True}],
    },
    # SMB
    {
        "path": "/smb/entity/financials",
        "method": "POST",
        "tier": "enterprise",
        "description": "Upsert monthly financials for an SMB entity",
        "params": [{"name": "body", "required": True, "type": "SMBFinancialsIn"}],
    },
    {
        "path": "/smb/entity/{entity_id}/cash-flow-forecast",
        "method": "GET",
        "tier": "enterprise",
        "description": "12-month cash flow forecast with runway detection",
        "params": [{"name": "entity_id", "required": True}, {"name": "horizon", "required": False}],
    },
    {
        "path": "/smb/entity/{entity_id}/supplier-risks",
        "method": "GET",
        "tier": "enterprise",
        "description": "Supplier concentration, AP growth, and cash buffer risk analysis",
        "params": [{"name": "entity_id", "required": True}],
    },
]


@router.get("/api-docs")
def api_documentation() -> Dict[str, Any]:
    """Return structured documentation for all platform endpoints."""
    from api.jobs.tenant_auth import get_tier_limits, _TIER_ORDER
    tiers = {
        tier: get_tier_limits(tier)
        for tier in sorted(_TIER_ORDER, key=lambda t: _TIER_ORDER[t])
    }
    return {
        "version": "1.0",
        "authentication": {
            "header": "X-API-Key: <your-api-key>",
            "bearer": "Authorization: Bearer <your-api-key>",
            "note": "API keys are provisioned per tenant. Contact support to upgrade tier.",
        },
        "tiers": {
            tier: {
                "description": info["description"],
                "rpm_limit": info["rpm"],
                "endpoint_count": sum(1 for e in _ENDPOINT_CATALOGUE if e["tier"] == tier),
            }
            for tier, info in tiers.items()
        },
        "endpoints": _ENDPOINT_CATALOGUE,
        "endpoint_count": len(_ENDPOINT_CATALOGUE),
    }
