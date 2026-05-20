from __future__ import annotations

from typing import Any, Dict, List, Optional

from api.platform.contracts import DemoWorkspaceSnapshot, PlatformReadinessSnapshot


def build_demo_workspace_snapshot(
    *,
    platform_version: str,
    workspace: Optional[Dict[str, Any]],
    workspace_analytics: Dict[str, Any],
    approvals: List[Dict[str, Any]],
    exports: List[Dict[str, Any]],
    integration_summary: Dict[str, Any],
    health_summary: Dict[str, Any],
) -> Dict[str, Any]:
    warnings = list(health_summary.get("warnings") or [])[:10]
    pending = [item for item in approvals if str(item.get("status")) == "pending"]
    pilot_ready = bool(
        workspace_analytics.get("dossier_count", 0) > 0
        and len(pending) == 0
        and not warnings
        and int(integration_summary.get("binding_count") or 0) >= 0
    )
    payload = DemoWorkspaceSnapshot(
        platform_version=platform_version,
        workspace_id=(workspace or {}).get("workspace_id"),
        workspace_name=(workspace or {}).get("name"),
        top_dossiers=(workspace_analytics.get("high_dau_dossiers") or [])[:8],
        pending_approvals=pending[:8],
        recent_exports=exports[:8],
        integration_summary=integration_summary,
        health_summary=health_summary,
        pilot_ready=pilot_ready,
        warnings=warnings,
    )
    return payload.model_dump(mode="python")


def build_readiness_snapshot(
    *,
    platform_version: str,
    workspace: Optional[Dict[str, Any]],
    workspace_analytics: Dict[str, Any],
    health_summary: Dict[str, Any],
    integration_summary: Dict[str, Any],
) -> Dict[str, Any]:
    warnings = list(health_summary.get("warnings") or [])
    missing_items: List[str] = []
    analysis_readiness = (
        "ready"
        if workspace_analytics.get("dossier_count", 0) > 0
        else "partial"
    )
    workflow_readiness = (
        "ready"
        if workspace_analytics.get("workflow_count", 0) > 0
        and workspace_analytics.get("pending_approval_count", 0) == 0
        else "partial"
    )
    export_readiness = (
        "ready" if workspace_analytics.get("export_count", 0) > 0 else "partial"
    )
    integration_readiness = (
        "ready" if integration_summary.get("binding_count", 0) > 0 else "partial"
    )
    if export_readiness != "ready":
        missing_items.append("No rendered export workflow has been exercised yet.")
    if integration_readiness != "ready":
        missing_items.append("No live integration binding has been executed yet.")
    if warnings:
        missing_items.append("Platform health warnings remain open.")
    pilot_ready = (
        analysis_readiness == "ready"
        and workflow_readiness == "ready"
        and export_readiness == "ready"
        and not warnings
    )
    rationale = (
        "Pilot-ready for demo use."
        if pilot_ready
        else "Further export, integration, or workflow hardening is still required."
    )
    payload = PlatformReadinessSnapshot(
        platform_version=platform_version,
        workspace_id=(workspace or {}).get("workspace_id"),
        analysis_readiness=analysis_readiness,
        workflow_readiness=workflow_readiness,
        export_readiness=export_readiness,
        integration_readiness=integration_readiness,
        health_warnings=warnings[:10],
        missing_enterprise_items=missing_items[:10],
        pilot_ready=pilot_ready,
        rationale=rationale,
    )
    return payload.model_dump(mode="python")
