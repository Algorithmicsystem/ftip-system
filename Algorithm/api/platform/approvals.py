from __future__ import annotations

import datetime as dt
import uuid
from typing import Any, Dict, Optional

from api.assistant.reports import sanitize_payload
from api.platform.contracts import ApprovalDecision, ApprovalRequest
from api.platform.security import actor_payload


def now_utc() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def build_approval_request(
    *,
    workflow_id: str,
    dossier_id: Optional[str],
    requested_role: str,
    user_context: Any,
    stage: Optional[str],
    rationale: Optional[str],
    required_permissions: Optional[list[str]] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> ApprovalRequest:
    return ApprovalRequest(
        approval_id=str(uuid.uuid4()),
        workflow_id=workflow_id,
        dossier_id=dossier_id,
        requested_role=requested_role,
        requested_by=actor_payload(user_context),
        status="pending",
        stage=stage,
        rationale=rationale,
        required_permissions=list(required_permissions or []),
        metadata=sanitize_payload(metadata or {}),
        created_at=now_utc(),
        updated_at=now_utc(),
    )


def build_approval_decision(
    *,
    approval_id: str,
    decision_type: str,
    user_context: Any,
    rationale: Optional[str],
    metadata: Optional[Dict[str, Any]] = None,
) -> ApprovalDecision:
    return ApprovalDecision(
        decision_id=str(uuid.uuid4()),
        approval_id=approval_id,
        decision_type=decision_type,
        decided_by=actor_payload(user_context),
        rationale=rationale,
        created_at=now_utc(),
        metadata=sanitize_payload(metadata or {}),
    )


def approval_status_after_decision(decision_type: str) -> str:
    normalized = str(decision_type or "").strip().lower()
    if normalized == "approve":
        return "approved"
    if normalized == "reject":
        return "rejected"
    if normalized == "request_changes":
        return "changes_requested"
    return "pending"
