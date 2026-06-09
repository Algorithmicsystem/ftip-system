"""Phase 6: Multi-tenant authentication and tier enforcement.

Provides:
  resolve_tenant       — look up tenant by raw API key (hash-matched)
  require_tier         — FastAPI Depends factory for tier gating
  check_sector_access  — validate sector against tenant's allowed_sectors
  get_tier_limits      — RPM and endpoint access rules per tier
  register_tenant      — insert/update a tenant record
"""
from __future__ import annotations

import hashlib
import logging
from typing import Any, Dict, List, Optional

from fastapi import Depends, HTTPException, Request

from api import db

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tier definitions
# ---------------------------------------------------------------------------

_TIER_ORDER = {"free": 0, "pro": 1, "enterprise": 2}

_TIER_LIMITS: Dict[str, Dict[str, Any]] = {
    "free": {
        "rpm": 30,
        "allowed_prefixes": ["/prosperity", "/axiom", "/health", "/version"],
        "description": "Basic signal access — /prosperity and /axiom endpoints",
    },
    "pro": {
        "rpm": 120,
        "allowed_prefixes": ["/prosperity", "/axiom", "/linkage", "/ops", "/health", "/version"],
        "description": "Institutional access — adds /linkage and /ops endpoints",
    },
    "enterprise": {
        "rpm": 0,  # 0 = unlimited
        "allowed_prefixes": ["/prosperity", "/axiom", "/linkage", "/ops", "/pe", "/smb", "/health", "/version"],
        "description": "Full platform — adds /pe and /smb endpoints",
    },
}


def get_tier_limits(tier: str) -> Dict[str, Any]:
    """Return RPM limit and allowed endpoint prefixes for a tier."""
    return _TIER_LIMITS.get(tier, _TIER_LIMITS["free"])


def _hash_key(raw_key: str) -> str:
    """SHA-256 hex digest of an API key."""
    return hashlib.sha256(raw_key.encode()).hexdigest()


# ---------------------------------------------------------------------------
# Tenant resolution
# ---------------------------------------------------------------------------

def resolve_tenant(raw_api_key: str) -> Optional[Dict[str, Any]]:
    """Resolve a raw API key to tenant metadata (hash-matched, active only)."""
    if not raw_api_key or not db.db_read_enabled():
        return None

    key_hash = _hash_key(raw_api_key)
    try:
        row = db.safe_fetchone(
            """
            SELECT tenant_id, org_name, tier, allowed_sectors, rpm_limit, expires_at
            FROM api_tenants
            WHERE api_key_hash = %s
              AND is_active = TRUE
              AND (expires_at IS NULL OR expires_at > now())
            """,
            (key_hash,),
        )
    except Exception as exc:
        logger.warning("tenant_auth.resolve_failed error=%s", exc)
        return None

    if not row:
        return None

    tenant_id, org_name, tier, allowed_sectors, rpm_limit, expires_at = row
    return {
        "tenant_id": tenant_id,
        "org_name": org_name,
        "tier": tier,
        "allowed_sectors": allowed_sectors if isinstance(allowed_sectors, list) else None,
        "rpm_limit": int(rpm_limit) if rpm_limit is not None else get_tier_limits(tier)["rpm"],
        "expires_at": expires_at.isoformat() if expires_at else None,
    }


def check_sector_access(tenant: Dict[str, Any], sector: str) -> bool:
    """Return True if the tenant has access to the given sector.

    A tenant with allowed_sectors=None has unrestricted sector access.
    """
    allowed = tenant.get("allowed_sectors")
    if allowed is None:
        return True
    return sector in allowed


def tier_has_access(tenant_tier: str, required_tier: str) -> bool:
    """Return True if tenant_tier meets or exceeds required_tier."""
    return _TIER_ORDER.get(tenant_tier, 0) >= _TIER_ORDER.get(required_tier, 0)


# ---------------------------------------------------------------------------
# FastAPI dependency
# ---------------------------------------------------------------------------

def _get_raw_api_key(request: Request) -> Optional[str]:
    """Extract raw API key from X-API-Key header, Authorization Bearer, or legacy headers."""
    for header in ("x-api-key", "x-ftip-api-key"):
        val = request.headers.get(header)
        if val:
            return val.strip()
    auth = request.headers.get("authorization", "")
    if auth.lower().startswith("bearer "):
        return auth[7:].strip()
    return None


def require_tier(min_tier: str = "free"):
    """FastAPI Depends factory that gates endpoints by minimum tier.

    Usage::

        @router.get("/pe/entity/{entity_id}/health")
        def get_health(entity_id: str, _: None = Depends(require_tier("enterprise"))):
            ...
    """
    async def _dependency(request: Request) -> Optional[Dict[str, Any]]:
        if not db.db_read_enabled():
            return None  # DB offline: allow through (dev mode)

        raw_key = _get_raw_api_key(request)
        if not raw_key:
            return None  # No key provided: fall through to existing auth layer

        # Primary API key (from FTIP_API_KEYS env var) is always enterprise tier —
        # this ensures the dashboard can always trigger the pipeline with its own key.
        try:
            from api.config import get_api_key as _primary_key_fn
            primary_key = _primary_key_fn()
            if primary_key and raw_key == primary_key:
                tenant_primary: Dict[str, Any] = {
                    "tenant_id": "primary",
                    "org_name": "Primary",
                    "tier": "enterprise",
                    "allowed_sectors": None,
                    "rpm_limit": 0,
                    "expires_at": None,
                }
                request.state.tenant = tenant_primary
                return tenant_primary
        except Exception:
            pass

        tenant = resolve_tenant(raw_key)
        if tenant is None:
            raise HTTPException(status_code=401, detail="Invalid or expired API key")

        if not tier_has_access(tenant.get("tier", "free"), min_tier):
            raise HTTPException(
                status_code=403,
                detail=f"This endpoint requires '{min_tier}' tier or above. "
                       f"Current tier: '{tenant.get('tier', 'free')}'",
            )

        request.state.tenant = tenant
        return tenant

    return _dependency


# ---------------------------------------------------------------------------
# Tenant management
# ---------------------------------------------------------------------------

def register_tenant(
    tenant_id: str,
    org_name: str,
    raw_api_key: str,
    tier: str = "free",
    allowed_sectors: Optional[List[str]] = None,
) -> bool:
    """Insert or update a tenant record (upsert by tenant_id)."""
    if not db.db_write_enabled():
        return False
    import json
    sectors_json = json.dumps(allowed_sectors) if allowed_sectors is not None else None
    try:
        db.safe_execute(
            """
            INSERT INTO api_tenants
                (tenant_id, org_name, api_key_hash, tier, allowed_sectors, rpm_limit)
            VALUES (%s, %s, %s, %s, %s::jsonb, %s)
            ON CONFLICT (tenant_id) DO UPDATE SET
                org_name        = EXCLUDED.org_name,
                api_key_hash    = EXCLUDED.api_key_hash,
                tier            = EXCLUDED.tier,
                allowed_sectors = EXCLUDED.allowed_sectors,
                rpm_limit       = EXCLUDED.rpm_limit
            """,
            (
                tenant_id,
                org_name,
                _hash_key(raw_api_key),
                tier,
                sectors_json,
                get_tier_limits(tier)["rpm"],
            ),
        )
        return True
    except Exception as exc:
        logger.warning("tenant_auth.register_failed tenant=%s error=%s", tenant_id, exc)
        return False
