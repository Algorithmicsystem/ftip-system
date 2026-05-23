from __future__ import annotations

from typing import Any, Dict, List, Optional

from api.assistant.reports import sanitize_payload
from api.platform.contracts import (
    AssignmentRecord,
    ConcernFlag,
    DecisionRationale,
    ReviewSummary,
    ReviewThreadSummary,
    ReviewerSlot,
    RoleAssignmentSummary,
)


_CONCERN_COMMENT_TYPES = {
    "risk_concern",
    "evidence_gap",
    "valuation_concern",
    "liquidity_concern",
    "fragility_concern",
}


def build_review_thread_summary(
    *,
    workflow_id: str,
    dossier_id: str,
    comments: List[Dict[str, Any]],
) -> Dict[str, Any]:
    severity_counts: Dict[str, int] = {}
    type_counts: Dict[str, int] = {}
    unresolved = 0
    resolved = 0
    for comment in comments:
        severity = str(comment.get("severity") or "info")
        comment_type = str(comment.get("comment_type") or "general")
        severity_counts[severity] = severity_counts.get(severity, 0) + 1
        type_counts[comment_type] = type_counts.get(comment_type, 0) + 1
        if str(comment.get("status") or "open") == "resolved":
            resolved += 1
        else:
            unresolved += 1
    return ReviewThreadSummary(
        workflow_id=workflow_id,
        dossier_id=dossier_id,
        total_comments=len(comments),
        unresolved_comments=unresolved,
        resolved_comments=resolved,
        comments_by_severity=severity_counts,
        comments_by_type=type_counts,
        latest_comment=comments[0] if comments else None,
    ).model_dump(mode="python")


def build_concern_flags(comments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    flags: List[Dict[str, Any]] = []
    for comment in comments:
        if str(comment.get("status") or "open") == "resolved":
            continue
        comment_type = str(comment.get("comment_type") or "general")
        if comment_type not in _CONCERN_COMMENT_TYPES:
            continue
        flags.append(
            ConcernFlag(
                concern_id=str(comment.get("comment_id") or ""),
                workflow_id=comment.get("workflow_id"),
                dossier_id=comment.get("dossier_id"),
                concern_type=comment_type,
                severity=str(comment.get("severity") or "watch"),
                summary=str(comment.get("body") or ""),
                source_comment_id=str(comment.get("comment_id") or ""),
                status=str(comment.get("status") or "open"),
            ).model_dump(mode="python")
        )
    return flags


def build_review_summary(
    *,
    workflow_id: str,
    dossier_id: str,
    comments: List[Dict[str, Any]],
    committee_decision: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    thread = build_review_thread_summary(
        workflow_id=workflow_id,
        dossier_id=dossier_id,
        comments=comments,
    )
    flags = build_concern_flags(comments)
    rationale = None
    if committee_decision:
        rationale = DecisionRationale(
            summary=committee_decision.get("summary"),
            key_risks=list(committee_decision.get("key_risks") or []),
            key_evidence_strengths=list(
                committee_decision.get("key_evidence_strengths") or []
            ),
            key_evidence_gaps=list(committee_decision.get("key_evidence_gaps") or []),
            notes=list((committee_decision.get("metadata") or {}).get("notes") or []),
        )
    return ReviewSummary(
        workflow_id=workflow_id,
        dossier_id=dossier_id,
        thread_summary=ReviewThreadSummary.model_validate(thread),
        unresolved_concern_count=len(flags),
        concern_flags=[ConcernFlag.model_validate(item) for item in flags],
        latest_comments=comments[:5],
        decision_rationale=rationale,
    ).model_dump(mode="python")


def build_role_assignment_summary(
    *,
    workflow_id: str,
    dossier_id: Optional[str],
    assignments: List[Dict[str, Any]],
) -> Dict[str, Any]:
    latest_by_slot: Dict[str, Dict[str, Any]] = {}
    for item in assignments:
        slot = str(item.get("slot_type") or "owner")
        latest_by_slot.setdefault(slot, item)

    def _slot(slot_type: str) -> Optional[ReviewerSlot]:
        item = latest_by_slot.get(slot_type)
        if not item:
            return None
        return ReviewerSlot(
            slot_type=slot_type,
            assignee_placeholder=item.get("assignee_placeholder"),
            status=str(item.get("status") or "assigned"),
            notes=list(item.get("notes") or []),
            metadata=sanitize_payload(item.get("metadata") or {}),
        )

    observers = [
        ReviewerSlot(
            slot_type="observer",
            assignee_placeholder=item.get("assignee_placeholder"),
            status=str(item.get("status") or "assigned"),
            notes=list(item.get("notes") or []),
            metadata=sanitize_payload(item.get("metadata") or {}),
        )
        for item in assignments
        if str(item.get("slot_type") or "") == "observer"
    ]
    return RoleAssignmentSummary(
        workflow_id=workflow_id,
        dossier_id=dossier_id,
        owner=_slot("owner"),
        primary_reviewer=_slot("primary_reviewer"),
        risk_reviewer=_slot("risk_reviewer"),
        committee_reviewer=_slot("committee_reviewer"),
        observers=observers,
        assignments=[AssignmentRecord.model_validate(item) for item in assignments[:20]],
    ).model_dump(mode="python")


__all__ = [
    "build_concern_flags",
    "build_review_summary",
    "build_review_thread_summary",
    "build_role_assignment_summary",
]
