from __future__ import annotations

import datetime as dt
import uuid
from typing import Any, Dict, Iterable, Optional, Union

from api.assistant.reports import sanitize_payload
from api.platform.contracts import (
    CommitteeDecisionSnapshot,
    DecisionCondition,
    DecisionOutcome,
)
from api.platform.security import actor_payload


def now_utc() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def build_decision_condition(value: Union[Dict[str, Any], str]) -> DecisionCondition:
    if isinstance(value, str):
        return DecisionCondition(
            condition_id=str(uuid.uuid4()),
            label=value,
        )
    return DecisionCondition(
        condition_id=str(value.get("condition_id") or uuid.uuid4()),
        label=str(value.get("label") or value.get("title") or "Condition"),
        status=str(value.get("status") or "required"),
        notes=list(value.get("notes") or []),
    )


def decision_outcome(decision_status: str, recommendation_state: Optional[str]) -> DecisionOutcome:
    normalized = str(decision_status or "").strip().lower()
    approved = normalized in {"approved", "approved_with_conditions"}
    return DecisionOutcome(
        outcome=normalized or "deferred",
        recommendation_state=recommendation_state,
        approved=approved,
    )


def build_committee_decision_snapshot(
    *,
    organization_id: Optional[str],
    workspace_id: Optional[str],
    workflow_id: str,
    dossier_id: str,
    stage: Optional[str],
    decision_status: str,
    recommendation_state: str,
    summary: str,
    conditions: Optional[Iterable[Union[Dict[str, Any], str]]],
    key_risks: Optional[Iterable[str]],
    key_evidence_strengths: Optional[Iterable[str]],
    key_evidence_gaps: Optional[Iterable[str]],
    user_context: Any,
    rationale: Optional[str],
    metadata: Optional[Dict[str, Any]] = None,
) -> CommitteeDecisionSnapshot:
    actor = actor_payload(user_context)
    return CommitteeDecisionSnapshot(
        decision_id=str(uuid.uuid4()),
        organization_id=organization_id,
        workspace_id=workspace_id,
        workflow_id=workflow_id,
        dossier_id=dossier_id,
        stage=stage,
        decision_status=str(decision_status or "deferred"),
        recommendation_state=str(recommendation_state or "under_review"),
        summary=str(summary or rationale or "Committee decision recorded."),
        conditions=[build_decision_condition(item) for item in list(conditions or [])],
        key_risks=[str(item) for item in list(key_risks or []) if item not in (None, "")],
        key_evidence_strengths=[
            str(item) for item in list(key_evidence_strengths or []) if item not in (None, "")
        ],
        key_evidence_gaps=[
            str(item) for item in list(key_evidence_gaps or []) if item not in (None, "")
        ],
        actor_context=actor,
        reviewer_context={
            "role": actor.get("role"),
            "actor_email": actor.get("actor_email"),
            "tenant_scope_summary": actor.get("tenant_scope_summary"),
        },
        created_at=now_utc(),
        metadata=sanitize_payload(
            {
                **dict(metadata or {}),
                "rationale": rationale,
                "decision_outcome": decision_outcome(
                    decision_status, recommendation_state
                ).model_dump(mode="python"),
            }
        ),
    )


__all__ = [
    "build_committee_decision_snapshot",
    "build_decision_condition",
    "decision_outcome",
]
