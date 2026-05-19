from __future__ import annotations

from typing import Any, Dict, List, Optional

from api.assistant.reports import sanitize_payload


def approval_guardrails(
    *,
    workflow: Dict[str, Any],
    dossier: Optional[Dict[str, Any]],
) -> List[str]:
    warnings: List[str] = []
    if not workflow:
        warnings.append("Workflow context is missing.")
    if dossier is None:
        warnings.append("Dossier context is missing.")
    if dossier and not dossier.get("latest_axiom_analysis_id"):
        warnings.append("Dossier has no AXIOM analysis attached.")
    return warnings


def export_guardrails(
    *,
    dossier: Dict[str, Any],
    approval_status: Optional[str],
) -> List[str]:
    warnings: List[str] = []
    if not dossier.get("latest_axiom_analysis_id"):
        warnings.append("Export pack is being generated without a linked AXIOM analysis.")
    if approval_status not in {None, "", "approved"}:
        warnings.append(f"Export is being generated while approval status is {approval_status}.")
    return warnings


def workflow_integrity_checks(
    workflow: Dict[str, Any],
    dossier: Optional[Dict[str, Any]],
) -> List[str]:
    warnings: List[str] = []
    stage = str(workflow.get("stage") or "")
    stage_state = workflow.get("stage_state") or {}
    if stage and stage_state and stage != stage_state.get("stage"):
        warnings.append("Workflow stage and stage_state.stage are inconsistent.")
    if dossier and dossier.get("workflow_id") and str(dossier.get("workflow_id")) != str(workflow.get("workflow_id")):
        warnings.append("Dossier workflow linkage is inconsistent.")
    return warnings


def dossier_integrity_checks(dossier: Dict[str, Any]) -> List[str]:
    warnings: List[str] = []
    if not dossier.get("sections"):
        warnings.append("Dossier has no structured sections.")
    if not dossier.get("current_summary"):
        warnings.append("Dossier current summary is empty.")
    return warnings


def summarize_guardrails(*, workflow: Dict[str, Any], dossier: Optional[Dict[str, Any]], approval_status: Optional[str]) -> Dict[str, Any]:
    return sanitize_payload(
        {
            "workflow_integrity_checks": workflow_integrity_checks(workflow, dossier),
            "dossier_integrity_checks": dossier_integrity_checks(dossier or {}),
            "approval_guardrails": approval_guardrails(workflow=workflow, dossier=dossier),
            "export_guardrails": export_guardrails(dossier=dossier or {}, approval_status=approval_status),
        }
    )
