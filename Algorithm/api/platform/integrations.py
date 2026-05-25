from __future__ import annotations

import datetime as dt
import uuid
from typing import Any, Dict, List, Optional

from api.assistant.reports import sanitize_payload
from api.platform.contracts import ConnectorHealthState, IntegrationBinding
from api.platform.integration_registry import get_integration_definition, list_integration_definitions


def now_utc() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def build_integration_binding(
    *,
    integration_type: str,
    organization_id: Optional[str],
    workspace_id: Optional[str],
    status: str,
    config: Optional[Dict[str, Any]] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> IntegrationBinding:
    definition = get_integration_definition(integration_type)
    warnings: List[str] = []
    if not workspace_id and definition.scope == "workspace":
        warnings.append("Workspace-scoped integration has no workspace binding.")
    health = ConnectorHealthState(
        status="healthy" if status == "active" else "configured",
        checked_at=now_utc(),
        warnings=warnings,
        details={"capability_count": len(definition.capabilities)},
    )
    return IntegrationBinding(
        binding_id=str(uuid.uuid4()),
        integration_type=definition.integration_type,
        organization_id=organization_id,
        workspace_id=workspace_id,
        status=status,
        config=sanitize_payload(config or {}),
        metadata=sanitize_payload(metadata or {}),
        health=health,
        created_at=now_utc(),
        updated_at=now_utc(),
    )


def integration_health_summary(
    bindings: List[Dict[str, Any]],
    *,
    premium_connector_overview: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    counts: Dict[str, int] = {}
    warnings: List[str] = []
    for binding in bindings:
        status = str(((binding.get("health") or {}).get("status")) or "unknown")
        counts[status] = counts.get(status, 0) + 1
        warnings.extend(((binding.get("health") or {}).get("warnings")) or [])
    premium_status = str((premium_connector_overview or {}).get("status") or "unknown")
    if premium_connector_overview and premium_status != "ready":
        warnings.append(str((premium_connector_overview or {}).get("summary") or "Premium connector readiness is not fully ready."))
    overall_status = "healthy"
    if counts.get("degraded") or counts.get("failed"):
        overall_status = "degraded"
    elif counts.get("configured") or counts.get("unknown"):
        overall_status = "configured"
    if premium_status in {"limited", "partial"} and overall_status == "healthy":
        overall_status = "configured"
    elif premium_status == "unknown" and not bindings:
        overall_status = "configured"
    return sanitize_payload(
        {
            "binding_count": len(bindings),
            "status_counts": counts,
            "overall_status": overall_status,
            "warnings": warnings[:10],
            "premium_connector_overview": premium_connector_overview or {},
            "definitions": [item.model_dump(mode="python") for item in list_integration_definitions()],
        }
    )
