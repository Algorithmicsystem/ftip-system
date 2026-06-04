"""Phase 20.1: Tamper-evident audit trail with hash chaining."""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import logging
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from api import db

logger = logging.getLogger(__name__)

AUDIT_EVENT_TYPES = [
    # Data events
    "data.bar_ingested",
    "data.feature_computed",
    "data.axiom_score_generated",
    # Signal events
    "signal.generated",
    "signal.delivered",
    "signal.acted_upon",
    # Analysis events
    "analysis.report_generated",
    "analysis.query_answered",
    "analysis.explanation_produced",
    # Portfolio events
    "portfolio.allocation_computed",
    "portfolio.var_computed",
    "portfolio.stress_tested",
    # Access events
    "access.api_key_used",
    "access.data_exported",
    "access.report_accessed",
    # Administrative events
    "admin.tenant_created",
    "admin.tier_changed",
    "admin.webhook_created",
    "admin.partner_registered",
]


@dataclass
class AuditRecord:
    event_id: str
    event_type: str
    tenant_id: Optional[str]
    actor_type: str                  # "user" | "system" | "scheduler" | "api"
    resource_type: str
    resource_id: str
    symbol: Optional[str]
    as_of_date: Optional[dt.date]
    output_hash: str
    output_summary: str
    previous_event_hash: str
    event_hash: str
    ip_address: Optional[str]
    api_version: str
    created_at: dt.datetime


def _compute_output_hash(output: Dict[str, Any]) -> str:
    body = json.dumps(output, separators=(",", ":"), sort_keys=True).encode()
    return hashlib.sha256(body).hexdigest()


def compute_event_hash(record: AuditRecord, previous_hash: str = "") -> str:
    content = (
        f"{record.event_id}|{record.event_type}|{record.tenant_id}|"
        f"{record.resource_id}|{record.output_hash}|"
        f"{record.created_at.isoformat()}|{previous_hash}"
    )
    return hashlib.sha256(content.encode()).hexdigest()


def _fetch_last_event_hash() -> str:
    """Get the hash of the most recently written audit record."""
    if not db.db_read_enabled():
        return ""
    try:
        row = db.safe_fetchone(
            "SELECT event_hash FROM audit_trail ORDER BY created_at DESC LIMIT 1"
        )
        if row and row[0]:
            return str(row[0])
    except Exception:
        pass
    return ""


def write_audit_record(
    event_type: str,
    resource_type: str,
    resource_id: str,
    output: Dict[str, Any],
    tenant_id: Optional[str] = None,
    symbol: Optional[str] = None,
    actor_type: str = "system",
    output_summary: str = "",
    ip_address: Optional[str] = None,
    as_of_date: Optional[dt.date] = None,
) -> AuditRecord:
    event_id = str(uuid.uuid4())
    created_at = dt.datetime.utcnow()
    output_hash = _compute_output_hash(output)
    previous_hash = _fetch_last_event_hash()

    record = AuditRecord(
        event_id=event_id,
        event_type=event_type,
        tenant_id=tenant_id,
        actor_type=actor_type,
        resource_type=resource_type,
        resource_id=resource_id,
        symbol=symbol,
        as_of_date=as_of_date,
        output_hash=output_hash,
        output_summary=output_summary or f"{event_type} for {resource_id}",
        previous_event_hash=previous_hash,
        event_hash="",
        ip_address=ip_address,
        api_version="v1",
        created_at=created_at,
    )
    record.event_hash = compute_event_hash(record, previous_hash)

    if db.db_write_enabled():
        try:
            db.safe_execute(
                """
                INSERT INTO audit_trail
                    (event_id, event_type, tenant_id, actor_type, resource_type,
                     resource_id, symbol, as_of_date, output_hash, output_summary,
                     previous_event_hash, event_hash, ip_address, api_version, created_at)
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                """,
                (
                    record.event_id, record.event_type, record.tenant_id,
                    record.actor_type, record.resource_type, record.resource_id,
                    record.symbol, record.as_of_date, record.output_hash,
                    record.output_summary, record.previous_event_hash,
                    record.event_hash, record.ip_address, record.api_version,
                    record.created_at,
                ),
            )
        except Exception as exc:
            logger.warning("audit_trail.write_failed event_id=%s err=%s", event_id, exc)

    return record


def verify_audit_chain(
    start_event_id: Optional[str] = None,
    limit: int = 100,
) -> Dict[str, Any]:
    """Re-compute each event_hash and compare to stored value."""
    if not db.db_read_enabled():
        return {"verified": True, "records_checked": 0, "first_broken_event_id": None,
                "chain_intact": True}

    try:
        if start_event_id:
            rows = db.safe_fetchall(
                """
                SELECT event_id, event_type, tenant_id, resource_id, output_hash,
                       previous_event_hash, event_hash, created_at
                  FROM audit_trail
                 WHERE created_at >= (SELECT created_at FROM audit_trail WHERE event_id = %s)
                 ORDER BY created_at ASC LIMIT %s
                """,
                (start_event_id, limit),
            )
        else:
            rows = db.safe_fetchall(
                """
                SELECT event_id, event_type, tenant_id, resource_id, output_hash,
                       previous_event_hash, event_hash, created_at
                  FROM audit_trail ORDER BY created_at ASC LIMIT %s
                """,
                (limit,),
            )
    except Exception as exc:
        err_str = str(exc).lower()
        if "disabled" in err_str or "not enabled" in err_str:
            # DB disabled — no records available; treat as empty chain (intact)
            return {"verified": True, "records_checked": 0, "first_broken_event_id": None,
                    "chain_intact": True}
        logger.warning("audit_trail.verify_failed err=%s", exc)
        return {"verified": False, "records_checked": 0, "first_broken_event_id": None,
                "chain_intact": False}

    if not rows:
        return {"verified": True, "records_checked": 0, "first_broken_event_id": None,
                "chain_intact": True}

    broken_event_id: Optional[str] = None
    for row in rows:
        (event_id, event_type, tenant_id, resource_id, output_hash,
         previous_hash, stored_hash, created_at) = row

        stub = AuditRecord(
            event_id=str(event_id),
            event_type=str(event_type),
            tenant_id=tenant_id,
            actor_type="",
            resource_type="",
            resource_id=str(resource_id),
            symbol=None,
            as_of_date=None,
            output_hash=str(output_hash),
            output_summary="",
            previous_event_hash=str(previous_hash or ""),
            event_hash="",
            ip_address=None,
            api_version="v1",
            created_at=created_at if isinstance(created_at, dt.datetime)
                       else dt.datetime.fromisoformat(str(created_at)),
        )
        recomputed = compute_event_hash(stub, str(previous_hash or ""))
        if recomputed != str(stored_hash):
            broken_event_id = str(event_id)
            break

    chain_intact = broken_event_id is None
    return {
        "verified": chain_intact,
        "records_checked": len(rows),
        "first_broken_event_id": broken_event_id,
        "chain_intact": chain_intact,
    }


def get_audit_trail(
    tenant_id: Optional[str] = None,
    event_types: Optional[List[str]] = None,
    symbol: Optional[str] = None,
    start_date: Optional[dt.date] = None,
    end_date: Optional[dt.date] = None,
    limit: int = 100,
) -> List[AuditRecord]:
    if not db.db_read_enabled():
        return []

    conditions = ["1=1"]
    params: List[Any] = []

    if tenant_id:
        conditions.append("tenant_id = %s")
        params.append(tenant_id)
    if event_types:
        conditions.append(f"event_type = ANY(%s)")
        params.append(event_types)
    if symbol:
        conditions.append("symbol = %s")
        params.append(symbol)
    if start_date:
        conditions.append("created_at >= %s")
        params.append(dt.datetime.combine(start_date, dt.time.min))
    if end_date:
        conditions.append("created_at <= %s")
        params.append(dt.datetime.combine(end_date, dt.time.max))

    params.append(limit)
    where = " AND ".join(conditions)

    try:
        rows = db.safe_fetchall(
            f"""
            SELECT event_id, event_type, tenant_id, actor_type, resource_type,
                   resource_id, symbol, as_of_date, output_hash, output_summary,
                   previous_event_hash, event_hash, ip_address, api_version, created_at
              FROM audit_trail WHERE {where}
             ORDER BY created_at DESC LIMIT %s
            """,
            params,
        )
    except Exception as exc:
        logger.warning("audit_trail.fetch_failed err=%s", exc)
        return []

    results: List[AuditRecord] = []
    for r in (rows or []):
        results.append(AuditRecord(
            event_id=str(r[0]), event_type=str(r[1]),
            tenant_id=r[2], actor_type=str(r[3] or "system"),
            resource_type=str(r[4] or ""), resource_id=str(r[5] or ""),
            symbol=r[6],
            as_of_date=r[7] if isinstance(r[7], dt.date) else None,
            output_hash=str(r[8] or ""), output_summary=str(r[9] or ""),
            previous_event_hash=str(r[10] or ""), event_hash=str(r[11] or ""),
            ip_address=r[12], api_version=str(r[13] or "v1"),
            created_at=r[14] if isinstance(r[14], dt.datetime)
                        else dt.datetime.fromisoformat(str(r[14])),
        ))
    return results


def get_audit_event(event_id: str) -> Optional[AuditRecord]:
    if not db.db_read_enabled():
        return None
    try:
        row = db.safe_fetchone(
            """
            SELECT event_id, event_type, tenant_id, actor_type, resource_type,
                   resource_id, symbol, as_of_date, output_hash, output_summary,
                   previous_event_hash, event_hash, ip_address, api_version, created_at
              FROM audit_trail WHERE event_id = %s
            """,
            (event_id,),
        )
        if not row:
            return None
        return AuditRecord(
            event_id=str(row[0]), event_type=str(row[1]),
            tenant_id=row[2], actor_type=str(row[3] or "system"),
            resource_type=str(row[4] or ""), resource_id=str(row[5] or ""),
            symbol=row[6],
            as_of_date=row[7] if isinstance(row[7], dt.date) else None,
            output_hash=str(row[8] or ""), output_summary=str(row[9] or ""),
            previous_event_hash=str(row[10] or ""), event_hash=str(row[11] or ""),
            ip_address=row[12], api_version=str(row[13] or "v1"),
            created_at=row[14] if isinstance(row[14], dt.datetime)
                        else dt.datetime.fromisoformat(str(row[14])),
        )
    except Exception as exc:
        logger.warning("audit_trail.get_event_failed err=%s", exc)
        return None
