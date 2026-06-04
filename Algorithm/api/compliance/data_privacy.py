"""Phase 20.3: Data privacy, GDPR, and retention infrastructure."""
from __future__ import annotations

import datetime as dt
import json
import logging
import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from api import db

logger = logging.getLogger(__name__)


@dataclass
class DataRetentionPolicy:
    tenant_id: str
    retain_trading_signals_days: int = 2555   # 7 years
    retain_audit_records_days: int = 3650     # 10 years
    retain_api_logs_days: int = 365           # 1 year
    retain_research_reports_days: int = 1825  # 5 years
    data_residency: str = "us"
    gdpr_applicable: bool = False
    right_to_erasure_requested: bool = False


def execute_right_to_erasure(
    tenant_id: str,
    erasure_scope: str = "personal_data",
) -> Dict[str, Any]:
    """GDPR Article 17: Right to Erasure.

    Deletes personal operational data. Never deletes audit_trail (regulatory).
    """
    tables_cleared: List[str] = []
    records_deleted = 0

    if not db.db_write_enabled():
        from api.compliance.audit_trail import write_audit_record
        record = write_audit_record(
            "access.data_exported",
            "tenant", tenant_id,
            {"erasure_scope": erasure_scope, "db_disabled": True},
            tenant_id=tenant_id,
            actor_type="user",
            output_summary=f"Right to erasure requested for tenant {tenant_id} (DB disabled)",
        )
        return {
            "tenant_id": tenant_id,
            "tables_cleared": [],
            "records_deleted": 0,
            "audit_trail_preserved": True,
            "erasure_record_id": record.event_id,
        }

    _ERASURE_TABLES = [
        ("api_usage_log", "tenant_id"),
        ("webhook_subscriptions", "tenant_id"),
        ("webhook_deliveries", "subscription_id"),
    ]

    # Also erase RIA / family office if tables exist
    _OPTIONAL_TABLES = [
        ("ria_client_profiles", "tenant_id"),
        ("family_office_portfolios", "tenant_id"),
    ]

    for table, col in _ERASURE_TABLES:
        try:
            if table == "webhook_deliveries":
                # Delete deliveries for subscriptions owned by this tenant
                row = db.safe_fetchone(
                    "SELECT COUNT(*) FROM webhook_deliveries wd "
                    "JOIN webhook_subscriptions ws ON ws.subscription_id = wd.subscription_id "
                    "WHERE ws.tenant_id = %s",
                    (tenant_id,),
                )
                if row and row[0]:
                    db.safe_execute(
                        "DELETE FROM webhook_deliveries WHERE subscription_id IN "
                        "(SELECT subscription_id FROM webhook_subscriptions WHERE tenant_id = %s)",
                        (tenant_id,),
                    )
                    records_deleted += int(row[0])
            else:
                row = db.safe_fetchone(
                    f"SELECT COUNT(*) FROM {table} WHERE {col} = %s", (tenant_id,)
                )
                count = int(row[0]) if row and row[0] else 0
                if count:
                    db.safe_execute(
                        f"DELETE FROM {table} WHERE {col} = %s", (tenant_id,)
                    )
                    records_deleted += count
            tables_cleared.append(table)
        except Exception as exc:
            logger.warning("erasure.table_failed table=%s err=%s", table, exc)

    for table, col in _OPTIONAL_TABLES:
        try:
            row = db.safe_fetchone(
                f"SELECT COUNT(*) FROM {table} WHERE {col} = %s", (tenant_id,)
            )
            count = int(row[0]) if row and row[0] else 0
            if count:
                db.safe_execute(f"DELETE FROM {table} WHERE {col} = %s", (tenant_id,))
                records_deleted += count
                tables_cleared.append(table)
        except Exception:
            pass

    if erasure_scope == "all_data":
        try:
            row = db.safe_fetchone(
                "SELECT COUNT(*) FROM api_tenants WHERE tenant_id = %s", (tenant_id,)
            )
            count = int(row[0]) if row and row[0] else 0
            if count:
                db.safe_execute("DELETE FROM api_tenants WHERE tenant_id = %s", (tenant_id,))
                records_deleted += count
                tables_cleared.append("api_tenants")
        except Exception as exc:
            logger.warning("erasure.api_tenants_failed err=%s", exc)

    # Write audit record of the erasure (audit trail is preserved)
    from api.compliance.audit_trail import write_audit_record
    record = write_audit_record(
        "access.data_exported",
        "tenant", tenant_id,
        {"erasure_scope": erasure_scope, "tables_cleared": tables_cleared,
         "records_deleted": records_deleted},
        tenant_id=tenant_id,
        actor_type="user",
        output_summary=f"GDPR Article 17 erasure executed for tenant {tenant_id}",
    )

    return {
        "tenant_id": tenant_id,
        "tables_cleared": tables_cleared,
        "records_deleted": records_deleted,
        "audit_trail_preserved": True,
        "erasure_record_id": record.event_id,
    }


def enforce_data_retention(as_of_date: Optional[dt.date] = None) -> Dict[str, Any]:
    """Delete records older than per-tenant retention policies."""
    aod = as_of_date or dt.date.today()
    if not db.db_write_enabled():
        return {"as_of_date": str(aod), "tenants_processed": 0, "total_deleted": 0}

    total_deleted = 0
    tenants_processed = 0

    try:
        rows = db.safe_fetchall(
            "SELECT tenant_id, retain_api_logs_days, retain_research_reports_days "
            "FROM data_retention_policies"
        )
        for row in (rows or []):
            tenant_id, api_days, report_days = row[0], int(row[1] or 365), int(row[2] or 1825)
            tenants_processed += 1

            api_cutoff = aod - dt.timedelta(days=api_days)
            try:
                r = db.safe_fetchone(
                    "SELECT COUNT(*) FROM api_usage_log WHERE tenant_id = %s AND created_at < %s",
                    (tenant_id, api_cutoff),
                )
                count = int(r[0]) if r and r[0] else 0
                if count:
                    db.safe_execute(
                        "DELETE FROM api_usage_log WHERE tenant_id = %s AND created_at < %s",
                        (tenant_id, api_cutoff),
                    )
                    total_deleted += count
            except Exception:
                pass

            report_cutoff = aod - dt.timedelta(days=report_days)
            try:
                r = db.safe_fetchone(
                    "SELECT COUNT(*) FROM research_reports WHERE tenant_id = %s AND created_at < %s",
                    (tenant_id, report_cutoff),
                )
                count = int(r[0]) if r and r[0] else 0
                if count:
                    db.safe_execute(
                        "DELETE FROM research_reports WHERE tenant_id = %s AND created_at < %s",
                        (tenant_id, report_cutoff),
                    )
                    total_deleted += count
            except Exception:
                pass

    except Exception as exc:
        logger.warning("retention.enforce_failed err=%s", exc)

    return {
        "as_of_date": str(aod),
        "tenants_processed": tenants_processed,
        "total_deleted": total_deleted,
    }


def generate_privacy_report(tenant_id: str) -> Dict[str, Any]:
    """GDPR Article 15: Right of Access — summary of all data held."""
    data_categories: List[str] = []
    record_counts: Dict[str, int] = {}
    oldest_record: Optional[dt.date] = None

    _TABLES_TO_CHECK = [
        ("api_usage_log", "tenant_id", "created_at"),
        ("webhook_subscriptions", "tenant_id", "created_at"),
        ("audit_trail", "tenant_id", "created_at"),
        ("api_tenants", "tenant_id", "created_at"),
    ]

    if db.db_read_enabled():
        for table, col, date_col in _TABLES_TO_CHECK:
            try:
                row = db.safe_fetchone(
                    f"SELECT COUNT(*), MIN({date_col}) FROM {table} WHERE {col} = %s",
                    (tenant_id,),
                )
                count = int(row[0]) if row and row[0] else 0
                if count:
                    data_categories.append(table)
                    record_counts[table] = count
                    if row[1]:
                        oldest = row[1].date() if isinstance(row[1], dt.datetime) else row[1]
                        if oldest_record is None or oldest < oldest_record:
                            oldest_record = oldest
            except Exception:
                pass

    # Load retention policy
    policy = load_retention_policy(tenant_id)
    data_residency = policy.data_residency if policy else "us"
    gdpr = policy.gdpr_applicable if policy else False

    return {
        "tenant_id": tenant_id,
        "data_categories": data_categories,
        "record_counts": record_counts,
        "oldest_record": str(oldest_record) if oldest_record else None,
        "data_residency": data_residency,
        "gdpr_applicable": gdpr,
        "processing_purposes": [
            "API access authentication and authorization",
            "Usage analytics and billing",
            "Regulatory compliance and audit trail",
            "System monitoring and security",
        ],
    }


def save_retention_policy(policy: DataRetentionPolicy) -> bool:
    if not db.db_write_enabled():
        return False
    try:
        db.safe_execute(
            """
            INSERT INTO data_retention_policies
                (tenant_id, retain_trading_signals_days, retain_audit_records_days,
                 retain_api_logs_days, retain_research_reports_days,
                 data_residency, gdpr_applicable, created_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s, now())
            ON CONFLICT (tenant_id) DO UPDATE SET
                retain_trading_signals_days = EXCLUDED.retain_trading_signals_days,
                retain_audit_records_days   = EXCLUDED.retain_audit_records_days,
                retain_api_logs_days        = EXCLUDED.retain_api_logs_days,
                retain_research_reports_days = EXCLUDED.retain_research_reports_days,
                data_residency              = EXCLUDED.data_residency,
                gdpr_applicable             = EXCLUDED.gdpr_applicable
            """,
            (
                policy.tenant_id, policy.retain_trading_signals_days,
                policy.retain_audit_records_days, policy.retain_api_logs_days,
                policy.retain_research_reports_days, policy.data_residency,
                policy.gdpr_applicable,
            ),
        )
        return True
    except Exception as exc:
        logger.warning("retention.save_failed tenant=%s err=%s", policy.tenant_id, exc)
        return False


def load_retention_policy(tenant_id: str) -> Optional[DataRetentionPolicy]:
    if not db.db_read_enabled():
        return None
    try:
        row = db.safe_fetchone(
            """
            SELECT tenant_id, retain_trading_signals_days, retain_audit_records_days,
                   retain_api_logs_days, retain_research_reports_days,
                   data_residency, gdpr_applicable
              FROM data_retention_policies WHERE tenant_id = %s
            """,
            (tenant_id,),
        )
        if not row:
            return None
        return DataRetentionPolicy(
            tenant_id=str(row[0]),
            retain_trading_signals_days=int(row[1] or 2555),
            retain_audit_records_days=int(row[2] or 3650),
            retain_api_logs_days=int(row[3] or 365),
            retain_research_reports_days=int(row[4] or 1825),
            data_residency=str(row[5] or "us"),
            gdpr_applicable=bool(row[6]),
        )
    except Exception as exc:
        logger.warning("retention.load_failed tenant=%s err=%s", tenant_id, exc)
        return None
