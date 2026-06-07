"""Phase 19.8: Partner revenue share computation."""
from __future__ import annotations

import datetime as dt
import logging
from typing import Any, Dict, Optional

from api import db

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Partner revenue share tiers
# ---------------------------------------------------------------------------

PARTNER_REVENUE_SHARE: Dict[str, Dict[str, Any]] = {
    "referral": {
        "revenue_share_pct": 10.0,
        "description": "Referral partner — 10% of referred customer revenue",
        "payout_trigger": "subscription",
        "min_payout_usd": 50.0,
    },
    "reseller": {
        "revenue_share_pct": 25.0,
        "description": "Reseller — 25% of resold subscription revenue",
        "payout_trigger": "invoice",
        "min_payout_usd": 100.0,
    },
    "oem": {
        "revenue_share_pct": 30.0,
        "description": "OEM integration — 30% of embedded product revenue",
        "payout_trigger": "usage",
        "min_payout_usd": 200.0,
    },
    "white_label": {
        "revenue_share_pct": 40.0,
        "description": "White-label — 40% of branded platform revenue",
        "payout_trigger": "invoice",
        "min_payout_usd": 500.0,
    },
}


def compute_partner_revenue_share(
    partner_id: str,
    partner_type: str,
    period: str = "30d",
    gross_revenue_usd: Optional[float] = None,
) -> Dict[str, Any]:
    """Compute revenue share for a partner. Uses gross_revenue_usd if supplied, else estimates from API usage."""
    cfg = PARTNER_REVENUE_SHARE.get(partner_type, PARTNER_REVENUE_SHARE["referral"])
    pct = cfg["revenue_share_pct"]

    if gross_revenue_usd is None:
        gross_revenue_usd = _estimate_revenue_from_usage(partner_id, period)

    share = round(gross_revenue_usd * pct / 100.0, 2)
    payout_eligible = share >= cfg["min_payout_usd"]

    return {
        "partner_id": partner_id,
        "partner_type": partner_type,
        "period": period,
        "gross_revenue_usd": gross_revenue_usd,
        "revenue_share_pct": pct,
        "revenue_share_usd": share,
        "payout_eligible": payout_eligible,
        "min_payout_usd": cfg["min_payout_usd"],
        "payout_trigger": cfg["payout_trigger"],
        "computed_at": dt.datetime.utcnow().isoformat(),
    }


def _estimate_revenue_from_usage(partner_id: str, period: str) -> float:
    """Estimate gross revenue from API usage log (proxy: calls × $0.01)."""
    if not db.db_read_enabled():
        return 0.0
    days = {"1d": 1, "7d": 7, "30d": 30, "90d": 90}.get(period, 30)
    since = dt.datetime.utcnow() - dt.timedelta(days=days)
    try:
        row = db.safe_fetchone(
            "SELECT COUNT(*) FROM api_usage_log WHERE tenant_id LIKE %s AND created_at >= %s",
            (f"%{partner_id}%", since),
        )
        calls = int(row[0]) if row and row[0] else 0
        return round(calls * 0.01, 2)
    except Exception as exc:
        logger.warning("revenue_share.estimate_failed partner=%s err=%s", partner_id, exc)
        return 0.0
