from __future__ import annotations

import datetime as dt
import math
from typing import Any, Dict, Iterable, List, Optional, Sequence


OPERATIONAL_GUARDRAILS_ARTIFACT_KIND = "assistant_operational_guardrails_artifact"
HEALTH_SNAPSHOT_ARTIFACT_KIND = "assistant_health_snapshot"
SHADOW_DECISION_RECORD_KIND = "assistant_shadow_decision_record"
OPERATIONAL_INCIDENT_ARTIFACT_KIND = "assistant_operational_incident"
OPERATIONAL_GUARDRAILS_VERSION = "phase12_operational_guardrails_v1"

STATUS_HEALTHY = "healthy"
STATUS_WATCH = "watch"
STATUS_DEGRADED = "degraded"
STATUS_CRITICAL = "critical"

SEVERITY_INFO = "info"
SEVERITY_CAUTION = "caution"
SEVERITY_ELEVATED = "elevated_risk"
SEVERITY_SERIOUS = "serious_degradation"
SEVERITY_CRITICAL = "critical_pause"


def safe_float(value: Any) -> Optional[float]:
    if value in (None, ""):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, float(value)))


def mean(values: Sequence[Any]) -> Optional[float]:
    clean = [float(item) for item in values if safe_float(item) is not None]
    if not clean:
        return None
    return float(sum(clean) / len(clean))


def now_utc() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def compact_list(values: Iterable[Any], *, limit: int = 8) -> List[str]:
    items: List[str] = []
    for value in values:
        if value in (None, "", [], {}):
            continue
        if isinstance(value, dict):
            label = (
                value.get("summary")
                or value.get("title")
                or value.get("name")
                or value.get("label")
                or value.get("domain")
                or value.get("affected_component")
                or str(value)
            )
            items.append(str(label))
        else:
            items.append(str(value))
        if len(items) >= limit:
            break
    return items


def severity_rank(severity: Any) -> int:
    normalized = str(severity or "").strip().lower()
    mapping = {
        SEVERITY_INFO: 0,
        SEVERITY_CAUTION: 1,
        SEVERITY_ELEVATED: 2,
        SEVERITY_SERIOUS: 3,
        SEVERITY_CRITICAL: 4,
    }
    return mapping.get(normalized, 0)


def health_rank(status: Any) -> int:
    normalized = str(status or "").strip().lower()
    mapping = {
        STATUS_HEALTHY: 0,
        STATUS_WATCH: 1,
        STATUS_DEGRADED: 2,
        STATUS_CRITICAL: 3,
    }
    return mapping.get(normalized, 1)


def status_from_score(
    score: Any,
    *,
    healthy_floor: float = 75.0,
    watch_floor: float = 58.0,
    degraded_floor: float = 38.0,
) -> str:
    number = safe_float(score)
    if number is None:
        return STATUS_WATCH
    if number >= healthy_floor:
        return STATUS_HEALTHY
    if number >= watch_floor:
        return STATUS_WATCH
    if number >= degraded_floor:
        return STATUS_DEGRADED
    return STATUS_CRITICAL


def dedupe_dicts_by_key(
    rows: Sequence[Dict[str, Any]],
    *,
    key: str,
    limit: int = 8,
) -> List[Dict[str, Any]]:
    seen: set[str] = set()
    output: List[Dict[str, Any]] = []
    for row in rows:
        marker = str(row.get(key) or "")
        if not marker or marker in seen:
            continue
        seen.add(marker)
        output.append(row)
        if len(output) >= limit:
            break
    return output
