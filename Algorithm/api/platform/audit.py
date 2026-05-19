from __future__ import annotations

import datetime as dt
import uuid
from typing import Any, Dict, Optional

from api.assistant.reports import sanitize_payload
from api.platform.contracts import AuditEvent, ResourceRef, WorkflowTimelineEvent
from api.platform.security import actor_payload


def now_utc() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def build_audit_event(
    *,
    event_type: str,
    resource: ResourceRef,
    user_context: Any,
    payload: Optional[Dict[str, Any]] = None,
    rationale: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> AuditEvent:
    return AuditEvent(
        event_id=str(uuid.uuid4()),
        event_type=event_type,
        resource_type=resource.resource_type,
        resource_id=resource.resource_id,
        organization_id=resource.organization_id,
        workspace_id=resource.workspace_id,
        actor=actor_payload(user_context),
        timestamp=now_utc(),
        payload=sanitize_payload(payload or {}),
        rationale=rationale,
        metadata=sanitize_payload(metadata or {}),
    )


def audit_event_to_timeline(event: Dict[str, Any]) -> WorkflowTimelineEvent:
    payload = sanitize_payload(event.get("payload") or {})
    workflow_id = (
        payload.get("workflow_id")
        or event.get("metadata", {}).get("workflow_id")
        or event.get("resource_id")
        if event.get("resource_type") == "workflow"
        else None
    )
    dossier_id = (
        payload.get("dossier_id")
        or event.get("metadata", {}).get("dossier_id")
        or event.get("resource_id")
        if event.get("resource_type") == "dossier"
        else None
    )
    actor = event.get("actor") or {}
    title = str(event.get("event_type") or "platform_event").replace("_", " ").title()
    summary = (
        event.get("rationale")
        or payload.get("summary")
        or payload.get("status")
        or f"{title} recorded."
    )
    return WorkflowTimelineEvent(
        event_id=str(event.get("event_id")),
        workflow_id=str(workflow_id or ""),
        dossier_id=str(dossier_id) if dossier_id else None,
        event_type=str(event.get("event_type") or "platform_event"),
        title=title,
        summary=str(summary),
        stage=payload.get("stage"),
        status=payload.get("status"),
        actor_label=actor.get("user_name") or actor.get("role") or actor.get("user_id"),
        created_at=event.get("timestamp"),
        payload=payload,
    )
