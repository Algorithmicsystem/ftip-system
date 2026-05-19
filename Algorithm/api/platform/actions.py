from __future__ import annotations

from typing import Any, Dict, List, Optional

from api.assistant.reports import sanitize_payload
from api.platform.contracts import StageTransition, WorkflowTemplate
from api.platform.permissions import get_role_definition


ACTION_PERMISSION_MAP: Dict[str, str] = {
    "advance_stage": "update_dossier",
    "send_for_review": "request_approval",
    "approve": "approve_stage",
    "reject": "approve_stage",
    "request_changes": "approve_stage",
    "attach_analysis": "attach_analysis",
    "refresh_summary": "update_dossier",
    "lock_recommendation": "approve_stage",
    "unlock_recommendation": "approve_stage",
    "export_pack": "export_report_pack",
}


APPROVAL_STAGE_SET = {
    "decision",
    "committee_view",
    "risk_committee",
    "client_memo",
    "watch_or_pursue",
}


def permission_for_action(action_type: str) -> str:
    return ACTION_PERMISSION_MAP.get(str(action_type or ""), "update_dossier")


def required_role_for_stage(template: WorkflowTemplate, stage: str) -> Optional[str]:
    if stage not in APPROVAL_STAGE_SET:
        return None
    if template.audience_type in {"hedge_fund", "family_office", "private_equity"}:
        return "committee"
    return "reviewer"


def stage_requires_approval(template: WorkflowTemplate, stage: str) -> bool:
    return required_role_for_stage(template, stage) is not None


def allowed_stage_transition(
    template: WorkflowTemplate,
    *,
    current_stage: str,
    requested_stage: Optional[str] = None,
) -> StageTransition:
    sequence = list(template.stage_sequence or ["intake"])
    stage = str(current_stage or sequence[0])
    target = str(requested_stage or "")
    if stage not in sequence:
        sequence = [stage, *sequence]
    try:
        current_index = sequence.index(stage)
    except ValueError:
        current_index = 0
    next_stage = sequence[current_index + 1] if current_index + 1 < len(sequence) else stage
    target_stage = target or next_stage
    allowed = bool(
        target_stage == stage
        or (
            target_stage in sequence
            and sequence.index(target_stage) <= current_index + 1
            and sequence.index(target_stage) >= current_index
        )
    )
    role = required_role_for_stage(template, target_stage)
    return StageTransition(
        template_id=template.template_id,
        from_stage=stage,
        to_stage=target_stage,
        allowed=allowed,
        requires_role=role,
        requires_approval=stage_requires_approval(template, target_stage),
        notes=[]
        if allowed
        else [f"Illegal transition from {stage} to {target_stage} for template {template.template_id}."],
    )


def list_allowed_actions(
    *,
    workflow: Dict[str, Any],
    dossier: Optional[Dict[str, Any]],
    template: WorkflowTemplate,
    effective_permissions: List[str],
    approvals: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    current_stage = str(workflow.get("stage") or "intake")
    stage_plan = allowed_stage_transition(template, current_stage=current_stage)
    locked = bool((dossier or {}).get("metadata", {}).get("recommendation_locked"))
    pending_approval = any(str(item.get("status")) == "pending" for item in approvals)
    actions: List[Dict[str, Any]] = []
    for action_type, permission in ACTION_PERMISSION_MAP.items():
        if permission not in effective_permissions:
            continue
        allowed = True
        note = None
        if action_type == "advance_stage" and not stage_plan.allowed:
            allowed = False
            note = "; ".join(stage_plan.notes)
        elif action_type == "approve" and not pending_approval:
            allowed = False
            note = "No pending approval request is available."
        elif action_type == "reject" and not pending_approval:
            allowed = False
            note = "No pending approval request is available."
        elif action_type == "request_changes" and not pending_approval:
            allowed = False
            note = "No pending approval request is available."
        elif action_type == "lock_recommendation" and locked:
            allowed = False
            note = "Recommendation is already locked."
        elif action_type == "unlock_recommendation" and not locked:
            allowed = False
            note = "Recommendation is not currently locked."
        actions.append(
            sanitize_payload(
                {
                    "action_type": action_type,
                    "permission": permission,
                    "allowed": allowed,
                    "next_stage": stage_plan.to_stage if action_type == "advance_stage" else None,
                    "requires_approval": stage_plan.requires_approval if action_type == "advance_stage" else False,
                    "requires_role": stage_plan.requires_role if action_type == "advance_stage" else None,
                    "note": note,
                }
            )
        )
    return actions


def apply_workflow_action(
    *,
    workflow: Dict[str, Any],
    dossier: Optional[Dict[str, Any]],
    template: WorkflowTemplate,
    action_type: str,
    requested_stage: Optional[str] = None,
    requested_status: Optional[str] = None,
    rationale: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    current_stage = str(workflow.get("stage") or "intake")
    stage_state = sanitize_payload(workflow.get("stage_state") or {})
    dossier_payload = sanitize_payload(dossier or {})
    workflow_updates: Dict[str, Any] = {}
    dossier_updates: Dict[str, Any] = {}
    summary = rationale or str(action_type or "").replace("_", " ").title()
    transition = allowed_stage_transition(
        template,
        current_stage=current_stage,
        requested_stage=requested_stage,
    )

    if action_type == "advance_stage":
        if not transition.allowed:
            raise ValueError("; ".join(transition.notes) or "illegal stage transition")
        completed = list(stage_state.get("completed_stages") or [])
        if current_stage and current_stage not in completed and transition.to_stage != current_stage:
            completed.append(current_stage)
        next_index = (template.stage_sequence or [transition.to_stage]).index(transition.to_stage)
        next_stage = None
        if next_index + 1 < len(template.stage_sequence or []):
            next_stage = (template.stage_sequence or [])[next_index + 1]
        stage_state.update(
            {
                "stage": transition.to_stage,
                "completed_stages": completed,
                "next_stage": next_stage,
                "status": requested_status or workflow.get("status") or "active",
                "notes": [*list(stage_state.get("notes") or []), summary],
            }
        )
        workflow_updates = {
            "stage": transition.to_stage,
            "status": requested_status or workflow.get("status") or "active",
            "stage_state": stage_state,
        }
        dossier_updates["workflow_stage_state"] = stage_state
    elif action_type == "send_for_review":
        stage_state["status"] = "in_review"
        workflow_updates = {
            "status": "in_review",
            "stage_state": stage_state,
        }
        dossier_updates["workflow_stage_state"] = stage_state
    elif action_type == "approve":
        stage_state["status"] = "approved"
        workflow_updates = {
            "status": "approved",
            "stage_state": stage_state,
        }
        dossier_updates["workflow_stage_state"] = stage_state
    elif action_type == "reject":
        stage_state["status"] = "rejected"
        workflow_updates = {
            "status": "rejected",
            "stage_state": stage_state,
        }
        dossier_updates["workflow_stage_state"] = stage_state
    elif action_type == "request_changes":
        stage_state["status"] = "draft"
        workflow_updates = {
            "status": "draft",
            "stage_state": stage_state,
        }
        dossier_updates["workflow_stage_state"] = stage_state
    elif action_type == "lock_recommendation":
        dossier_updates["metadata"] = {
            **(dossier_payload.get("metadata") or {}),
            "recommendation_locked": True,
            "locked_reason": rationale,
        }
    elif action_type == "unlock_recommendation":
        dossier_updates["metadata"] = {
            **(dossier_payload.get("metadata") or {}),
            "recommendation_locked": False,
            "unlock_reason": rationale,
        }
    elif action_type == "refresh_summary":
        dossier_updates["metadata"] = {
            **(dossier_payload.get("metadata") or {}),
            "last_refresh_reason": rationale,
        }
    elif action_type == "export_pack":
        dossier_updates["metadata"] = {
            **(dossier_payload.get("metadata") or {}),
            "last_export_request": sanitize_payload(metadata or {}),
        }
    return {
        "workflow_updates": sanitize_payload(workflow_updates),
        "dossier_updates": sanitize_payload(dossier_updates),
        "transition": transition.model_dump(mode="python"),
        "summary": summary,
    }
