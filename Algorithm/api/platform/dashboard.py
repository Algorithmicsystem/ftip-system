from __future__ import annotations

from typing import Any, Dict, List, Optional

from api.assistant.reports import sanitize_payload


def build_dashboard_payload(
    *,
    workspace: Optional[Dict[str, Any]],
    summary_view: Dict[str, Any],
    workspace_analytics: Dict[str, Any],
    cross_workspace_analytics: Dict[str, Any],
    approvals: List[Dict[str, Any]],
    timeline: List[Dict[str, Any]],
    exports: List[Dict[str, Any]],
    integration_summary: Dict[str, Any],
    health_summary: Dict[str, Any],
) -> Dict[str, Any]:
    pending_approvals = [item for item in approvals if str(item.get("status")) == "pending"]
    recent_decisions = [
        item
        for item in approvals
        if str(item.get("status")) in {"approved", "rejected", "changes_requested"}
    ][:6]
    return sanitize_payload(
        {
            "workspace_context": workspace or {},
            "executive_metrics": {
                "workspace_name": (workspace or {}).get("name"),
                "dossier_count": workspace_analytics.get("dossier_count", 0),
                "workflow_count": workspace_analytics.get("workflow_count", 0),
                "pending_approval_count": len(pending_approvals),
                "export_count": workspace_analytics.get("export_count", 0),
                "integration_binding_count": workspace_analytics.get(
                    "integration_binding_count", 0
                ),
            },
            "summary_view": summary_view,
            "workspace_analytics": workspace_analytics,
            "cross_workspace_analytics": cross_workspace_analytics,
            "pending_approvals": pending_approvals[:8],
            "recent_decisions": recent_decisions,
            "recent_timeline": timeline[:10],
            "recent_exports": exports[:8],
            "high_dau_dossiers": workspace_analytics.get("high_dau_dossiers") or [],
            "integration_summary": integration_summary,
            "health_summary": health_summary,
        }
    )
