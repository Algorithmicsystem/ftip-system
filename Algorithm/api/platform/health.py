from __future__ import annotations

from typing import Any, Dict, List, Optional

from api.assistant.reports import sanitize_payload
from api.platform.contracts import PlatformHealthSummary
from api.platform.guardrails import dossier_integrity_checks, workflow_integrity_checks
from api.platform.integrations import integration_health_summary


def build_platform_health_summary(
    *,
    platform_version: str,
    workspace_id: Optional[str],
    organization_id: Optional[str],
    access_summary: Dict[str, Any],
    workflows: List[Dict[str, Any]],
    dossiers: List[Dict[str, Any]],
    approvals: List[Dict[str, Any]],
    exports: List[Dict[str, Any]],
    stored_exports: Optional[List[Dict[str, Any]]] = None,
    audit_events: List[Dict[str, Any]],
    integration_bindings: List[Dict[str, Any]],
    premium_connector_overview: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    workflow_checks: List[str] = []
    dossier_checks: List[str] = []
    for workflow in workflows[:20]:
        linked = next(
            (item for item in dossiers if str(item.get("workflow_id")) == str(workflow.get("workflow_id"))),
            None,
        )
        workflow_checks.extend(workflow_integrity_checks(workflow, linked))
    for dossier in dossiers[:20]:
        dossier_checks.extend(dossier_integrity_checks(dossier))
    export_checks = []
    for export in exports[:20]:
        if not export.get("content_hash"):
            export_checks.append(f"Export {export.get('export_id')} has no content hash.")
        if not export.get("ordered_sections"):
            export_checks.append(f"Export {export.get('export_id')} has no ordered sections.")
    for stored_export in (stored_exports or [])[:20]:
        if not stored_export.get("storage_key"):
            export_checks.append(
                f"Stored export {stored_export.get('stored_export_id')} has no storage key."
            )
        if not stored_export.get("checksum"):
            export_checks.append(
                f"Stored export {stored_export.get('stored_export_id')} has no checksum."
            )
    warnings = [
        *workflow_checks,
        *dossier_checks,
        *export_checks,
    ][:20]
    if premium_connector_overview and str(premium_connector_overview.get("status") or "unknown") != "ready":
        warnings = [
            *warnings,
            str(
                premium_connector_overview.get("summary")
                or "Premium connector readiness is not fully ready."
            ),
        ][:20]
    return PlatformHealthSummary(
        platform_version=platform_version,
        workspace_id=workspace_id,
        organization_id=organization_id,
        access_summary=sanitize_payload(access_summary),
        pending_approval_count=sum(1 for item in approvals if str(item.get("status")) == "pending"),
        export_count=len(exports),
        audit_event_count=len(audit_events),
        integration_health_summary=integration_health_summary(
            integration_bindings,
            premium_connector_overview=premium_connector_overview,
        ),
        workflow_integrity_checks=workflow_checks[:20],
        dossier_integrity_checks=dossier_checks[:20],
        export_integrity_checks=export_checks[:20],
        warnings=warnings,
    ).model_dump(mode="python")
