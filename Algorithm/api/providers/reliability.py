"""Provider reliability time-series — persists daily health snapshots.

Consumers:
    from api.providers.reliability import (
        ProviderReliabilityRecord,
        snapshot_provider_reliability,
        get_provider_reliability_window,
        get_provider_reliability_summary,
    )
"""
from __future__ import annotations

import datetime as dt
from typing import Any, Dict, List, Optional

from pydantic import BaseModel

from api import db


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class ProviderReliabilityRecord(BaseModel):
    as_of_date: dt.date
    provider: str
    is_enabled: bool
    status: str          # ok | degraded | down
    message: str = ""
    meta: Dict[str, Any] = {}


# ---------------------------------------------------------------------------
# Status helpers
# ---------------------------------------------------------------------------

_UPTIME_WEIGHTS = {"ok": 1.0, "degraded": 0.5, "down": 0.0}


def _status_label(pct: float) -> str:
    if pct >= 0.95:
        return "reliable"
    if pct >= 0.75:
        return "intermittent"
    return "unreliable"


# ---------------------------------------------------------------------------
# Write
# ---------------------------------------------------------------------------

def snapshot_provider_reliability(
    health_response: Any,
    *,
    as_of_date: Optional[dt.date] = None,
) -> int:
    """Persist today's provider health snapshot to provider_reliability_daily.

    Accepts a ProvidersHealthResponse (or anything with a .providers list of
    objects having .name, .enabled, .status, .message attributes).
    Returns the number of rows upserted.
    """
    if not db.db_enabled():
        return 0
    date = as_of_date or dt.date.today()
    providers = getattr(health_response, "providers", None) or []
    count = 0
    for p in providers:
        try:
            db.safe_execute(
                """
                INSERT INTO provider_reliability_daily
                    (as_of_date, provider, is_enabled, status, message)
                VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT (as_of_date, provider)
                DO UPDATE SET
                    is_enabled  = EXCLUDED.is_enabled,
                    status      = EXCLUDED.status,
                    message     = EXCLUDED.message,
                    recorded_at = now()
                """,
                (
                    date,
                    str(p.name),
                    bool(getattr(p, "enabled", True)),
                    str(p.status),
                    str(getattr(p, "message", "") or ""),
                ),
            )
            count += 1
        except Exception:  # pragma: no cover — DB unavailable in tests
            pass
    return count


# ---------------------------------------------------------------------------
# Read
# ---------------------------------------------------------------------------

def get_provider_reliability_window(
    days: int = 30,
    *,
    provider: Optional[str] = None,
) -> List[ProviderReliabilityRecord]:
    """Return raw reliability records for the last N days, newest first."""
    if not db.db_read_enabled():
        return []
    cutoff = dt.date.today() - dt.timedelta(days=days)
    params: list = [cutoff]
    provider_clause = ""
    if provider:
        provider_clause = "AND provider = %s"
        params.append(provider)
    rows = db.safe_fetchall(
        f"""
        SELECT as_of_date, provider, is_enabled, status, message
        FROM provider_reliability_daily
        WHERE as_of_date >= %s {provider_clause}
        ORDER BY as_of_date DESC, provider
        """,
        tuple(params),
    )
    return [
        ProviderReliabilityRecord(
            as_of_date=row[0],
            provider=row[1],
            is_enabled=bool(row[2]),
            status=str(row[3]),
            message=str(row[4] or ""),
        )
        for row in (rows or [])
    ]


def get_provider_reliability_summary(days: int = 30) -> Dict[str, Any]:
    """Return per-provider uptime summary over the last N days.

    Returns a dict keyed by provider name, each with:
        total_days, ok_days, degraded_days, down_days, uptime_pct, status_label
    """
    records = get_provider_reliability_window(days=days)
    summary: Dict[str, Dict[str, Any]] = {}
    for rec in records:
        if rec.provider not in summary:
            summary[rec.provider] = {"total_days": 0, "ok_days": 0, "degraded_days": 0, "down_days": 0}
        entry = summary[rec.provider]
        entry["total_days"] += 1
        entry[f"{rec.status}_days"] = entry.get(f"{rec.status}_days", 0) + 1

    result: Dict[str, Any] = {}
    for prov, entry in summary.items():
        total = entry["total_days"]
        weighted = (
            entry.get("ok_days", 0) * 1.0
            + entry.get("degraded_days", 0) * 0.5
            + entry.get("down_days", 0) * 0.0
        )
        uptime_pct = round(weighted / total, 4) if total > 0 else None
        result[prov] = {
            **entry,
            "uptime_pct": uptime_pct,
            "status_label": _status_label(uptime_pct) if uptime_pct is not None else "unknown",
            "window_days": days,
        }
    return result
