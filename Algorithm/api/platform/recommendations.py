from __future__ import annotations

import datetime as dt
import uuid
from typing import Any, Dict, Optional

from api.assistant.reports import sanitize_payload
from api.platform.contracts import (
    EscalationRecord,
    RecommendationChangeRecord,
    RecommendationLockRecord,
    RecommendationState,
)
from api.platform.security import actor_payload


RECOMMENDATION_TRANSITIONS: Dict[str, set[str]] = {
    "draft": {
        "draft",
        "under_review",
        "approved_live",
        "approved_paper",
        "watch_only",
        "rejected",
        "archived",
    },
    "under_review": {
        "under_review",
        "approved_live",
        "approved_paper",
        "watch_only",
        "rejected",
        "archived",
    },
    "approved_live": {"approved_live", "approved_paper", "watch_only", "rejected", "archived"},
    "approved_paper": {"approved_paper", "approved_live", "watch_only", "rejected", "archived"},
    "watch_only": {"watch_only", "under_review", "approved_paper", "approved_live", "archived"},
    "rejected": {"rejected", "draft", "archived"},
    "archived": {"archived"},
}


def now_utc() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def default_recommendation_state() -> Dict[str, Any]:
    return RecommendationState(
        state="draft",
        locked=False,
        last_changed_at=now_utc(),
        metadata={},
    ).model_dump(mode="python")


def validate_recommendation_transition(
    current_state: str,
    next_state: str,
) -> None:
    allowed = RECOMMENDATION_TRANSITIONS.get(str(current_state or "draft"), {"draft"})
    if str(next_state or "draft") not in allowed:
        raise ValueError(
            f"Illegal recommendation state transition from {current_state or 'draft'} to {next_state or 'draft'}."
        )


def build_recommendation_state(
    current: Optional[Dict[str, Any]],
    *,
    next_state: str,
    user_context: Any,
    summary: Optional[str],
    rationale: Optional[str],
    lock_recommendation: Optional[bool],
    source_decision_id: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    payload = sanitize_payload(current or default_recommendation_state())
    current_state = str(payload.get("state") or "draft")
    validate_recommendation_transition(current_state, next_state)
    actor = actor_payload(user_context)
    locked = (
        bool(lock_recommendation)
        if lock_recommendation is not None
        else bool(payload.get("locked"))
    )
    if locked and not payload.get("locked"):
        payload["locked_at"] = now_utc()
        payload["locked_by"] = actor
    elif lock_recommendation is False:
        payload["locked_at"] = None
        payload["locked_by"] = {}
    payload.update(
        {
            "state": str(next_state or current_state),
            "locked": locked,
            "summary": summary or payload.get("summary"),
            "rationale": rationale,
            "last_changed_at": now_utc(),
            "last_changed_by": actor,
            "source_decision_id": source_decision_id or payload.get("source_decision_id"),
            "metadata": sanitize_payload(
                {
                    **dict(payload.get("metadata") or {}),
                    **dict(metadata or {}),
                }
            ),
        }
    )
    return RecommendationState.model_validate(payload).model_dump(mode="python")


def build_recommendation_change_record(
    *,
    organization_id: Optional[str],
    workspace_id: Optional[str],
    workflow_id: str,
    dossier_id: str,
    previous_state: Optional[str],
    new_state: str,
    action_type: str,
    recommendation_state: Dict[str, Any],
    rationale: Optional[str],
    user_context: Any,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    return RecommendationChangeRecord(
        change_id=str(uuid.uuid4()),
        organization_id=organization_id,
        workspace_id=workspace_id,
        workflow_id=workflow_id,
        dossier_id=dossier_id,
        previous_state=previous_state,
        new_state=new_state,
        action_type=action_type,
        locked=bool(recommendation_state.get("locked")),
        snapshot=sanitize_payload(recommendation_state),
        rationale=rationale,
        actor=actor_payload(user_context),
        created_at=now_utc(),
        metadata=sanitize_payload(metadata or {}),
    ).model_dump(mode="python")


def build_recommendation_lock_record(
    *,
    workflow_id: str,
    dossier_id: str,
    recommendation_state: Dict[str, Any],
    reason: Optional[str],
    user_context: Any,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    return RecommendationLockRecord(
        lock_id=str(uuid.uuid4()),
        workflow_id=workflow_id,
        dossier_id=dossier_id,
        recommendation_state=RecommendationState.model_validate(recommendation_state),
        reason=reason,
        actor=actor_payload(user_context),
        created_at=now_utc(),
        metadata=sanitize_payload(metadata or {}),
    ).model_dump(mode="python")


def build_escalation_record(
    *,
    organization_id: Optional[str],
    workspace_id: Optional[str],
    workflow_id: str,
    dossier_id: Optional[str],
    action_type: str,
    from_state: Optional[str],
    to_state: Optional[str],
    rationale: Optional[str],
    user_context: Any,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    return EscalationRecord(
        escalation_id=str(uuid.uuid4()),
        organization_id=organization_id,
        workspace_id=workspace_id,
        workflow_id=workflow_id,
        dossier_id=dossier_id,
        action_type=action_type,
        from_state=from_state,
        to_state=to_state,
        rationale=rationale,
        actor=actor_payload(user_context),
        created_at=now_utc(),
        metadata=sanitize_payload(metadata or {}),
    ).model_dump(mode="python")


__all__ = [
    "build_escalation_record",
    "build_recommendation_change_record",
    "build_recommendation_lock_record",
    "build_recommendation_state",
    "default_recommendation_state",
    "validate_recommendation_transition",
]
