from __future__ import annotations

import datetime as dt
import uuid
from typing import Any, Dict, Optional

from api.assistant.reports import sanitize_payload
from api.platform.contracts import ReviewComment
from api.platform.security import actor_payload


def now_utc() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def build_review_comment(
    *,
    organization_id: Optional[str],
    workspace_id: Optional[str],
    workflow_id: str,
    dossier_id: str,
    stage: Optional[str],
    user_context: Any,
    comment_type: str,
    body: str,
    severity: str,
    metadata: Optional[Dict[str, Any]] = None,
) -> ReviewComment:
    return ReviewComment(
        comment_id=str(uuid.uuid4()),
        organization_id=organization_id,
        workspace_id=workspace_id,
        workflow_id=workflow_id,
        dossier_id=dossier_id,
        stage=stage,
        author=actor_payload(user_context),
        comment_type=str(comment_type or "general"),
        body=str(body or "").strip(),
        severity=str(severity or "info"),
        created_at=now_utc(),
        status="open",
        metadata=sanitize_payload(metadata or {}),
    )


def resolve_review_comment(
    comment: Dict[str, Any],
    *,
    user_context: Any,
    rationale: Optional[str],
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    payload = sanitize_payload(comment)
    payload["status"] = "resolved"
    payload["resolved_at"] = now_utc()
    payload["metadata"] = sanitize_payload(
        {
            **dict(payload.get("metadata") or {}),
            **dict(metadata or {}),
            "resolution_rationale": rationale,
            "resolved_by": actor_payload(user_context),
        }
    )
    return payload


__all__ = ["build_review_comment", "resolve_review_comment"]
