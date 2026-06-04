"""Phase 19.5: Partner program management."""
from __future__ import annotations

import datetime as dt
import json
import logging
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from api import db

logger = logging.getLogger(__name__)

PARTNER_TIERS: Dict[str, Dict[str, Any]] = {
    "reseller": {
        "revenue_share_pct": 20.0,
        "rate_limit_multiplier": 2.0,
        "api_key_prefix": "ax_rs_",
        "description": "Resell AXIOM to clients; 20% revenue share.",
    },
    "white_label": {
        "revenue_share_pct": 30.0,
        "rate_limit_multiplier": 5.0,
        "api_key_prefix": "ax_wl_",
        "description": "White-label AXIOM under your brand; 30% revenue share.",
    },
    "academic": {
        "revenue_share_pct": 0.0,
        "rate_limit_multiplier": 1.0,
        "api_key_prefix": "ax_ac_",
        "description": "Academic research access; no revenue share.",
    },
    "integration_partner": {
        "revenue_share_pct": 0.0,
        "rate_limit_multiplier": 3.0,
        "api_key_prefix": "ax_ip_",
        "description": "Technology integration partner; no revenue share.",
    },
}


@dataclass
class PartnerProfile:
    partner_id: str
    org_name: str
    partner_tier: str
    contact_email: str
    agreement_signed: bool = False
    revenue_share_pct: float = 0.0
    custom_branding: Dict[str, Any] = field(default_factory=dict)
    rate_limit_multiplier: float = 1.0
    endpoints_allowed: List[str] = field(default_factory=list)
    api_key_prefix: str = "ax_"


def register_partner(
    org_name: str,
    partner_tier: str,
    contact_email: str,
) -> PartnerProfile:
    if partner_tier not in PARTNER_TIERS:
        raise ValueError(f"Unknown partner_tier: {partner_tier}. Valid: {list(PARTNER_TIERS)}")

    tier_cfg = PARTNER_TIERS[partner_tier]
    partner = PartnerProfile(
        partner_id=str(uuid.uuid4()),
        org_name=org_name,
        partner_tier=partner_tier,
        contact_email=contact_email,
        agreement_signed=False,
        revenue_share_pct=tier_cfg["revenue_share_pct"],
        custom_branding={},
        rate_limit_multiplier=tier_cfg["rate_limit_multiplier"],
        endpoints_allowed=[],
        api_key_prefix=tier_cfg["api_key_prefix"],
    )

    if db.db_write_enabled():
        try:
            db.safe_execute(
                """
                INSERT INTO partner_profiles
                    (partner_id, org_name, partner_tier, contact_email, agreement_signed,
                     revenue_share_pct, custom_branding, rate_limit_multiplier,
                     endpoints_allowed, api_key_prefix, created_at, updated_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s::jsonb, %s, %s::jsonb, %s, now(), now())
                """,
                (
                    partner.partner_id, partner.org_name, partner.partner_tier,
                    partner.contact_email, partner.agreement_signed,
                    partner.revenue_share_pct,
                    json.dumps(partner.custom_branding),
                    partner.rate_limit_multiplier,
                    json.dumps(partner.endpoints_allowed),
                    partner.api_key_prefix,
                ),
            )
        except Exception as exc:
            logger.warning("partner.register_failed error=%s", exc)

    return partner


def get_partner(partner_id: str) -> Optional[PartnerProfile]:
    if not db.db_read_enabled():
        return None
    try:
        row = db.safe_fetchone(
            """
            SELECT partner_id, org_name, partner_tier, contact_email, agreement_signed,
                   revenue_share_pct, custom_branding, rate_limit_multiplier,
                   endpoints_allowed, api_key_prefix
              FROM partner_profiles WHERE partner_id = %s
            """,
            (partner_id,),
        )
        if not row:
            return None
        return PartnerProfile(
            partner_id=row[0], org_name=row[1], partner_tier=row[2],
            contact_email=row[3], agreement_signed=bool(row[4]),
            revenue_share_pct=float(row[5] or 0),
            custom_branding=row[6] if isinstance(row[6], dict) else {},
            rate_limit_multiplier=float(row[7] or 1.0),
            endpoints_allowed=row[8] if isinstance(row[8], list) else [],
            api_key_prefix=row[9] or "ax_",
        )
    except Exception as exc:
        logger.warning("partner.get_failed error=%s", exc)
        return None


def generate_white_label_config(partner: PartnerProfile, branding: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "partner_id": partner.partner_id,
        "org_name": partner.org_name,
        "partner_tier": partner.partner_tier,
        "branding": {
            "logo_url": branding.get("logo_url", ""),
            "primary_color": branding.get("primary_color", "#000000"),
            "secondary_color": branding.get("secondary_color", "#FFFFFF"),
            "product_name": branding.get("product_name", partner.org_name + " Intelligence"),
            "support_email": branding.get("support_email", partner.contact_email),
        },
        "api_key_prefix": partner.api_key_prefix,
        "rate_limit_multiplier": partner.rate_limit_multiplier,
        "endpoints_allowed": partner.endpoints_allowed or ["*"],
        "revenue_share_pct": partner.revenue_share_pct,
    }


def compute_partner_revenue_share(partner_id: str, period: str = "30d") -> Dict[str, Any]:
    """Compute estimated revenue share for a partner based on API usage."""
    if not db.db_read_enabled():
        return {"partner_id": partner_id, "period": period, "estimated_revenue_share": 0.0,
                "total_api_calls": 0}

    days = {"1d": 1, "7d": 7, "30d": 30, "90d": 90}.get(period, 30)
    since = dt.datetime.utcnow() - dt.timedelta(days=days)

    partner = get_partner(partner_id)
    if not partner:
        return {"partner_id": partner_id, "period": period, "estimated_revenue_share": 0.0,
                "total_api_calls": 0}

    try:
        row = db.safe_fetchone(
            "SELECT COUNT(*) FROM api_usage_log WHERE tenant_id LIKE %s AND created_at >= %s",
            (f"%{partner.api_key_prefix}%", since),
        )
        total_calls = int(row[0]) if row and row[0] else 0
        rev_per_call = 0.001
        estimated = round(total_calls * rev_per_call * partner.revenue_share_pct / 100.0, 4)
        return {
            "partner_id": partner_id,
            "org_name": partner.org_name,
            "period": period,
            "total_api_calls": total_calls,
            "revenue_share_pct": partner.revenue_share_pct,
            "estimated_revenue_share": estimated,
        }
    except Exception as exc:
        logger.warning("partner.revenue_share_failed error=%s", exc)
        return {"partner_id": partner_id, "period": period, "estimated_revenue_share": 0.0,
                "total_api_calls": 0}
