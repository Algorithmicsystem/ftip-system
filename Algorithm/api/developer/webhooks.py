"""Phase 19.3: Webhook subscription and delivery system."""
from __future__ import annotations

import datetime as dt
import hashlib
import hmac
import json
import logging
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from api import db

logger = logging.getLogger(__name__)

WEBHOOK_EVENTS = [
    "signal.buy",
    "signal.sell",
    "signal.regime_change",
    "risk.sri_alert",
    "risk.sornette_warning",
    "pe.health_alert",
    "smb.cash_runway_warning",
    "ml.model_updated",
    "system.health_degraded",
]


@dataclass
class WebhookSubscription:
    subscription_id: str
    tenant_id: str
    event_type: str
    callback_url: str
    secret: str
    filter: Dict[str, Any] = field(default_factory=dict)
    is_active: bool = True
    retry_count: int = 0
    created_at: Optional[dt.datetime] = None


@dataclass
class WebhookDelivery:
    delivery_id: str
    subscription_id: str
    event_type: str
    payload: Dict[str, Any]
    delivered_at: Optional[dt.datetime]
    status: str
    http_status_code: Optional[int]
    retry_count: int = 0


def sign_webhook_payload(payload: Dict[str, Any], secret: str) -> str:
    """HMAC-SHA256 hex digest of the JSON-serialized payload."""
    body = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode()
    return hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()


def create_subscription(
    tenant_id: str,
    event_type: str,
    callback_url: str,
    secret: str,
    filter_config: Optional[Dict[str, Any]] = None,
) -> WebhookSubscription:
    if event_type not in WEBHOOK_EVENTS:
        raise ValueError(f"Unknown event_type: {event_type}. Valid: {WEBHOOK_EVENTS}")

    sub = WebhookSubscription(
        subscription_id=str(uuid.uuid4()),
        tenant_id=tenant_id,
        event_type=event_type,
        callback_url=callback_url,
        secret=secret,
        filter=filter_config or {},
        is_active=True,
        retry_count=0,
        created_at=dt.datetime.utcnow(),
    )

    if db.db_write_enabled():
        try:
            db.safe_execute(
                """
                INSERT INTO webhook_subscriptions
                    (subscription_id, tenant_id, event_type, callback_url, secret,
                     filter, is_active, retry_count, created_at)
                VALUES (%s, %s, %s, %s, %s, %s::jsonb, %s, %s, %s)
                """,
                (
                    sub.subscription_id, sub.tenant_id, sub.event_type,
                    sub.callback_url, sub.secret,
                    json.dumps(sub.filter), sub.is_active,
                    sub.retry_count, sub.created_at,
                ),
            )
        except Exception as exc:
            logger.warning("webhook.create_subscription_failed error=%s", exc)

    return sub


def list_subscriptions(tenant_id: str) -> List[WebhookSubscription]:
    if not db.db_read_enabled():
        return []
    try:
        rows = db.safe_fetchall(
            """
            SELECT subscription_id, tenant_id, event_type, callback_url, secret,
                   filter, is_active, retry_count, created_at
              FROM webhook_subscriptions
             WHERE tenant_id = %s
             ORDER BY created_at DESC
            """,
            (tenant_id,),
        )
        return [
            WebhookSubscription(
                subscription_id=r[0], tenant_id=r[1], event_type=r[2],
                callback_url=r[3], secret=r[4],
                filter=r[5] if isinstance(r[5], dict) else {},
                is_active=bool(r[6]), retry_count=int(r[7] or 0),
                created_at=r[8],
            )
            for r in (rows or [])
        ]
    except Exception as exc:
        logger.warning("webhook.list_failed error=%s", exc)
        return []


def delete_subscription(subscription_id: str, tenant_id: str) -> bool:
    if not db.db_write_enabled():
        return False
    try:
        db.safe_execute(
            "DELETE FROM webhook_subscriptions WHERE subscription_id = %s AND tenant_id = %s",
            (subscription_id, tenant_id),
        )
        return True
    except Exception as exc:
        logger.warning("webhook.delete_failed error=%s", exc)
        return False


def deliver_webhook(
    subscription: WebhookSubscription,
    event_type: str,
    payload: Dict[str, Any],
) -> WebhookDelivery:
    """Attempt HTTP delivery to the subscription's callback_url."""
    delivery_id = str(uuid.uuid4())
    signature = sign_webhook_payload(payload, subscription.secret)
    enriched = dict(payload)
    enriched["event_type"] = event_type
    enriched["subscription_id"] = subscription.subscription_id
    enriched["delivered_at"] = dt.datetime.utcnow().isoformat()

    status = "failed"
    http_status_code: Optional[int] = None

    try:
        import urllib.request
        import urllib.error
        body = json.dumps(enriched).encode()
        req = urllib.request.Request(
            subscription.callback_url,
            data=body,
            headers={
                "Content-Type": "application/json",
                "X-AXIOM-Signature": signature,
                "X-AXIOM-Event": event_type,
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=5) as resp:
            http_status_code = resp.status
            if 200 <= http_status_code < 300:
                status = "delivered"
    except Exception as exc:
        logger.debug("webhook.delivery_failed sub=%s err=%s", subscription.subscription_id, exc)

    delivery = WebhookDelivery(
        delivery_id=delivery_id,
        subscription_id=subscription.subscription_id,
        event_type=event_type,
        payload=enriched,
        delivered_at=dt.datetime.utcnow() if status == "delivered" else None,
        status=status,
        http_status_code=http_status_code,
        retry_count=0,
    )

    if db.db_write_enabled():
        try:
            db.safe_execute(
                """
                INSERT INTO webhook_deliveries
                    (delivery_id, subscription_id, event_type, payload,
                     delivered_at, status, http_status_code, retry_count)
                VALUES (%s, %s, %s, %s::jsonb, %s, %s, %s, %s)
                """,
                (
                    delivery.delivery_id, delivery.subscription_id, delivery.event_type,
                    json.dumps(delivery.payload), delivery.delivered_at,
                    delivery.status, delivery.http_status_code, delivery.retry_count,
                ),
            )
        except Exception as exc:
            logger.warning("webhook.store_delivery_failed error=%s", exc)

    return delivery


def check_and_fire_webhooks(event_type: str, event_payload: Dict[str, Any]) -> List[WebhookDelivery]:
    """Find all active subscriptions for event_type and attempt delivery."""
    if not db.db_read_enabled():
        return []
    try:
        rows = db.safe_fetchall(
            """
            SELECT subscription_id, tenant_id, event_type, callback_url, secret,
                   filter, is_active, retry_count, created_at
              FROM webhook_subscriptions
             WHERE event_type = %s AND is_active = TRUE
            """,
            (event_type,),
        )
    except Exception as exc:
        logger.warning("webhook.check_failed error=%s", exc)
        return []

    deliveries: List[WebhookDelivery] = []
    for r in (rows or []):
        sub = WebhookSubscription(
            subscription_id=r[0], tenant_id=r[1], event_type=r[2],
            callback_url=r[3], secret=r[4],
            filter=r[5] if isinstance(r[5], dict) else {},
            is_active=bool(r[6]), retry_count=int(r[7] or 0),
            created_at=r[8],
        )
        delivery = deliver_webhook(sub, event_type, event_payload)
        deliveries.append(delivery)

    return deliveries
