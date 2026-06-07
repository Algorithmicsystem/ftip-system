"""Phase 19.6: Developer Platform Routes."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel

from api.jobs.tenant_auth import require_tier

router = APIRouter(prefix="/developer", tags=["developer"])


# ---------------------------------------------------------------------------
# API Versioning / Documentation
# ---------------------------------------------------------------------------

@router.get("/api-docs")
def get_api_docs() -> Dict[str, Any]:
    from api.developer.api_versioning import get_api_documentation
    return get_api_documentation()


@router.get("/api-docs/{category}")
def get_api_docs_by_category(category: str) -> Dict[str, Any]:
    from api.developer.api_versioning import get_endpoints_by_category
    endpoints = get_endpoints_by_category(category)
    return {"category": category, "endpoints": endpoints, "count": len(endpoints)}


@router.get("/endpoints")
def get_endpoints(
    category: Optional[str] = Query(default=None),
    tier: Optional[str] = Query(default=None),
) -> Dict[str, Any]:
    from api.developer.api_versioning import V1_ENDPOINTS
    result = V1_ENDPOINTS
    if category:
        result = [ep for ep in result if ep["category"] == category]
    if tier:
        result = [ep for ep in result if ep["tier_required"] == tier]
    return {"endpoints": result, "count": len(result)}


# ---------------------------------------------------------------------------
# SDK Downloads
# ---------------------------------------------------------------------------

@router.get("/sdk/python")
def get_python_sdk(
    base_url: str = Query(default="https://api.axiom.ftip.io"),
    version: str = Query(default="v1"),
) -> Dict[str, Any]:
    from api.developer.sdk_generator import generate_python_sdk
    code = generate_python_sdk(base_url=base_url, version=version)
    return {"language": "python", "version": version, "sdk_code": code}


@router.get("/sdk/javascript")
def get_javascript_sdk(
    base_url: str = Query(default="https://api.axiom.ftip.io"),
) -> Dict[str, Any]:
    from api.developer.sdk_generator import generate_javascript_sdk
    code = generate_javascript_sdk(base_url=base_url)
    return {"language": "javascript", "version": "1.0.0", "sdk_code": code}


@router.get("/sdk/r")
def get_r_sdk(
    base_url: str = Query(default="https://api.axiom.ftip.io"),
) -> Dict[str, Any]:
    from api.developer.sdk_generator import generate_r_sdk
    code = generate_r_sdk(base_url=base_url)
    return {"language": "r", "version": "1.0.0", "sdk_code": code}


# ---------------------------------------------------------------------------
# Webhooks
# ---------------------------------------------------------------------------

class WebhookSubscribeRequest(BaseModel):
    event_type: str
    callback_url: str
    secret: str
    filter: Dict[str, Any] = {}


@router.post("/webhooks", dependencies=[Depends(require_tier("pro"))])
def subscribe_webhook(
    body: WebhookSubscribeRequest,
    _tenant=Depends(require_tier("pro")),
) -> Dict[str, Any]:
    from api.developer.webhooks import create_subscription, WEBHOOK_EVENTS
    if body.event_type not in WEBHOOK_EVENTS:
        raise HTTPException(status_code=400, detail=f"Unknown event_type. Valid: {WEBHOOK_EVENTS}")
    tenant_id = getattr(_tenant, "tenant_id", None) if _tenant else "anon"
    if isinstance(_tenant, dict):
        tenant_id = _tenant.get("tenant_id", "anon")
    sub = create_subscription(
        tenant_id=tenant_id or "anon",
        event_type=body.event_type,
        callback_url=body.callback_url,
        secret=body.secret,
        filter_config=body.filter,
    )
    return {
        "subscription_id": sub.subscription_id,
        "tenant_id": sub.tenant_id,
        "event_type": sub.event_type,
        "callback_url": sub.callback_url,
        "is_active": sub.is_active,
        "created_at": sub.created_at.isoformat() if sub.created_at else None,
    }


@router.get("/webhooks", dependencies=[Depends(require_tier("pro"))])
def list_webhooks(_tenant=Depends(require_tier("pro"))) -> Dict[str, Any]:
    from api.developer.webhooks import list_subscriptions
    tenant_id = "anon"
    if isinstance(_tenant, dict):
        tenant_id = _tenant.get("tenant_id", "anon")
    subs = list_subscriptions(tenant_id)
    return {
        "tenant_id": tenant_id,
        "subscriptions": [
            {
                "subscription_id": s.subscription_id,
                "event_type": s.event_type,
                "callback_url": s.callback_url,
                "is_active": s.is_active,
                "retry_count": s.retry_count,
            }
            for s in subs
        ],
    }


@router.delete("/webhooks/{subscription_id}", dependencies=[Depends(require_tier("pro"))])
def delete_webhook(
    subscription_id: str,
    _tenant=Depends(require_tier("pro")),
) -> Dict[str, Any]:
    from api.developer.webhooks import delete_subscription
    tenant_id = "anon"
    if isinstance(_tenant, dict):
        tenant_id = _tenant.get("tenant_id", "anon")
    success = delete_subscription(subscription_id, tenant_id)
    return {"deleted": success, "subscription_id": subscription_id}


@router.get("/webhooks/events")
def list_webhook_events() -> Dict[str, Any]:
    from api.developer.webhooks import WEBHOOK_EVENTS
    return {"events": WEBHOOK_EVENTS, "count": len(WEBHOOK_EVENTS)}


# ---------------------------------------------------------------------------
# Usage Analytics
# ---------------------------------------------------------------------------

@router.get("/usage", dependencies=[Depends(require_tier("pro"))])
def get_usage(
    period: str = Query(default="7d"),
    _tenant=Depends(require_tier("pro")),
) -> Dict[str, Any]:
    from api.developer.usage_analytics import compute_usage_metrics
    import dataclasses
    tenant_id = "anon"
    if isinstance(_tenant, dict):
        tenant_id = _tenant.get("tenant_id", "anon")
    metrics = compute_usage_metrics(tenant_id, period)
    return dataclasses.asdict(metrics)


@router.get("/usage/history", dependencies=[Depends(require_tier("pro"))])
def get_usage_history(
    period: str = Query(default="30d"),
    _tenant=Depends(require_tier("pro")),
) -> Dict[str, Any]:
    from api.developer.usage_analytics import compute_usage_history
    tenant_id = "anon"
    if isinstance(_tenant, dict):
        tenant_id = _tenant.get("tenant_id", "anon")
    history = compute_usage_history(tenant_id, period)
    return {"tenant_id": tenant_id, "period": period, "history": history}


@router.get("/analytics", dependencies=[Depends(require_tier("enterprise"))])
def get_platform_analytics(period: str = Query(default="7d")) -> Dict[str, Any]:
    from api.developer.usage_analytics import compute_platform_analytics
    return compute_platform_analytics(period)


# ---------------------------------------------------------------------------
# Partner Program
# ---------------------------------------------------------------------------

class PartnerRegisterRequest(BaseModel):
    org_name: str
    partner_tier: str
    contact_email: str


@router.post("/partners", dependencies=[Depends(require_tier("enterprise"))])
def register_partner(body: PartnerRegisterRequest) -> Dict[str, Any]:
    from api.developer.partner_program import register_partner as _register, PARTNER_TIERS
    if body.partner_tier not in PARTNER_TIERS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown partner_tier. Valid: {list(PARTNER_TIERS)}",
        )
    import dataclasses
    partner = _register(body.org_name, body.partner_tier, body.contact_email)
    return dataclasses.asdict(partner)


@router.get("/partners/{partner_id}", dependencies=[Depends(require_tier("enterprise"))])
def get_partner_profile(partner_id: str) -> Dict[str, Any]:
    from api.developer.partner_program import get_partner
    import dataclasses
    partner = get_partner(partner_id)
    if not partner:
        raise HTTPException(status_code=404, detail="Partner not found")
    return dataclasses.asdict(partner)


@router.get("/partners/{partner_id}/revenue-share", dependencies=[Depends(require_tier("enterprise"))])
def get_partner_revenue_share(
    partner_id: str,
    period: str = Query(default="30d"),
) -> Dict[str, Any]:
    from api.developer.partner_program import compute_partner_revenue_share
    return compute_partner_revenue_share(partner_id, period)


# ---------------------------------------------------------------------------
# API Key Management
# ---------------------------------------------------------------------------

@router.get("/keys")
def list_api_keys(_tenant=Depends(require_tier("free"))) -> Dict[str, Any]:
    """Return metadata about the caller's active API key."""
    from api.security import get_allowed_api_keys
    keys = get_allowed_api_keys()
    masked = [f"{k[:4]}{'*' * (len(k) - 8)}{k[-4:]}" if len(k) > 8 else "****" for k in keys]
    return {
        "key_count": len(masked),
        "keys": masked,
        "note": "Keys are masked for security. Use /developer/keys/rotate to issue a new key.",
    }


@router.post("/keys/rotate")
def rotate_api_key(_tenant=Depends(require_tier("free"))) -> Dict[str, Any]:
    """Generate a new API key identifier (actual provisioning is ops-side)."""
    import uuid
    new_key_id = f"ax_{uuid.uuid4().hex[:24]}"
    return {
        "status": "rotation_requested",
        "new_key_id": new_key_id,
        "note": "Contact ops@axiom.ftip.io to activate the new key. Old key remains valid for 48h.",
    }


# ---------------------------------------------------------------------------
# Billing Usage
# ---------------------------------------------------------------------------

@router.get("/billing/usage")
def get_billing_usage(
    tier: str = Query(default="free"),
    _tenant=Depends(require_tier("free")),
) -> Dict[str, Any]:
    """Return billing tier config and usage summary for the current key."""
    from api.developer.billing import BILLING_TIERS, quota_enforcer
    from api.security import get_provided_api_key
    cfg = BILLING_TIERS.get(tier, BILLING_TIERS["free"])
    return {
        "tier": tier,
        "price_usd": cfg["price_usd"],
        "monthly_calls": cfg["monthly_calls"],
        "rpm": cfg["rpm"],
        "features": cfg["features"],
        "description": cfg["description"],
    }


@router.get("/billing/tiers")
def list_billing_tiers() -> Dict[str, Any]:
    from api.developer.billing import BILLING_TIERS
    return {"tiers": BILLING_TIERS}


# ---------------------------------------------------------------------------
# Revenue Share
# ---------------------------------------------------------------------------

@router.get("/revenue-share/{partner_type}")
def get_revenue_share_config(partner_type: str) -> Dict[str, Any]:
    from api.developer.revenue_share import PARTNER_REVENUE_SHARE
    cfg = PARTNER_REVENUE_SHARE.get(partner_type)
    if not cfg:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown partner_type. Valid: {list(PARTNER_REVENUE_SHARE)}",
        )
    return {"partner_type": partner_type, **cfg}


@router.post("/revenue-share/compute")
def compute_revenue_share_endpoint(
    partner_id: str = Query(...),
    partner_type: str = Query(...),
    period: str = Query(default="30d"),
    gross_revenue_usd: Optional[float] = Query(default=None),
) -> Dict[str, Any]:
    from api.developer.revenue_share import compute_partner_revenue_share
    return compute_partner_revenue_share(partner_id, partner_type, period, gross_revenue_usd)


# ---------------------------------------------------------------------------
# Webhook Test
# ---------------------------------------------------------------------------

@router.post("/webhooks/test")
def test_webhook(
    event_type: str = Query(default="signal.buy"),
) -> Dict[str, Any]:
    """Fire a test webhook event without triggering actual pipeline logic."""
    from api.developer.webhooks import WEBHOOK_EVENTS, check_and_fire_webhooks
    if event_type not in WEBHOOK_EVENTS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown event_type. Valid: {WEBHOOK_EVENTS}",
        )
    test_payload = {
        "test": True,
        "event_type": event_type,
        "symbol": "AAPL",
        "dau": 72.5,
        "signal_label": "BUY",
        "note": "This is a test delivery from POST /developer/webhooks/test",
    }
    deliveries = check_and_fire_webhooks(event_type, test_payload)
    return {
        "event_type": event_type,
        "test_payload": test_payload,
        "deliveries_attempted": len(deliveries),
        "deliveries": [
            {
                "delivery_id": d.delivery_id,
                "subscription_id": d.subscription_id,
                "status": d.status,
                "http_status_code": d.http_status_code,
            }
            for d in deliveries
        ],
    }
