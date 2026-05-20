from __future__ import annotations

from typing import Any, Dict, List, Optional

from api.assistant.reports import sanitize_payload
from api.platform.contracts import PlatformAnalyticsView, WorkspaceAnalyticsView
from api.platform.dossiers import dossier_preview


def _dossier_snapshot_metrics(dossier: Dict[str, Any]) -> Dict[str, Any]:
    summary = dossier.get("current_summary") or {}
    report_snapshot = (dossier.get("metadata") or {}).get("latest_report_snapshot") or {}
    summary_card = summary.get("axiom_summary_card") or report_snapshot.get(
        "axiom_summary_card"
    ) or {}
    dau = report_snapshot.get("axiom_deployable_alpha_utility") or summary_card.get(
        "deployable_alpha_utility"
    )
    return {
        "dau": float(dau) if dau not in (None, "") else None,
        "size_band": dossier.get("latest_size_band") or summary.get("size_band"),
        "evidence_status": dossier.get("evidence_status") or summary.get("evidence_status") or "partial",
        "trade_family": dossier.get("latest_trade_family") or summary.get("trade_family") or "unknown",
        "live_candidate": (dossier.get("latest_deployability_tier") or "") == "live_candidate",
    }


def _increment(bucket: Dict[str, int], key: Any) -> None:
    name = str(key or "unknown")
    bucket[name] = bucket.get(name, 0) + 1


def build_workspace_analytics_view(
    *,
    workspace: Dict[str, Any],
    workflows: List[Dict[str, Any]],
    dossiers: List[Dict[str, Any]],
    approvals: List[Dict[str, Any]],
    exports: List[Dict[str, Any]],
    integration_bindings: List[Dict[str, Any]],
) -> WorkspaceAnalyticsView:
    workflow_by_id = {
        str(item.get("workflow_id")): item for item in workflows
    }
    by_tier: Dict[str, int] = {}
    by_regime: Dict[str, int] = {}
    by_stage: Dict[str, int] = {}
    evidence_distribution: Dict[str, int] = {}
    template_distribution: Dict[str, int] = {}
    size_distribution: Dict[str, int] = {}
    dau_values: List[float] = []
    live_count = 0
    dossier_records: List[Dict[str, Any]] = []
    for workflow in workflows:
        _increment(template_distribution, workflow.get("workflow_template_id"))
    for dossier in dossiers:
        preview = dossier_preview(dossier)
        workflow = workflow_by_id.get(str(dossier.get("workflow_id"))) or {}
        metrics = _dossier_snapshot_metrics(dossier)
        preview["deployable_alpha_utility"] = metrics["dau"]
        preview["workflow_template_id"] = workflow.get("workflow_template_id")
        preview["workspace_name"] = workspace.get("name")
        dossier_records.append(preview)
        _increment(by_tier, dossier.get("latest_deployability_tier"))
        _increment(by_regime, dossier.get("latest_regime_label"))
        _increment(by_stage, (dossier.get("workflow_stage_state") or {}).get("stage"))
        _increment(evidence_distribution, metrics["evidence_status"])
        _increment(size_distribution, metrics["size_band"])
        if metrics["dau"] is not None:
            dau_values.append(float(metrics["dau"]))
        if metrics["live_candidate"]:
            live_count += 1
    dossier_records.sort(
        key=lambda item: (
            -1 if item.get("deployable_alpha_utility") is None else item.get("deployable_alpha_utility"),
            item.get("title") or "",
        ),
        reverse=True,
    )
    return WorkspaceAnalyticsView(
        workspace_id=str(workspace.get("workspace_id")),
        organization_id=workspace.get("organization_id"),
        workspace_name=workspace.get("name"),
        audience_type=workspace.get("audience_type"),
        workflow_count=len(workflows),
        dossier_count=len(dossiers),
        pending_approval_count=sum(
            1 for item in approvals if str(item.get("status")) == "pending"
        ),
        export_count=len(exports),
        integration_binding_count=len(integration_bindings),
        dossiers_by_deployability_tier=by_tier,
        dossiers_by_regime=by_regime,
        dossiers_by_stage=by_stage,
        evidence_status_distribution=evidence_distribution,
        workflow_template_distribution=template_distribution,
        average_dau=(sum(dau_values) / len(dau_values)) if dau_values else None,
        live_candidate_ratio=(live_count / len(dossiers)) if dossiers else None,
        size_band_distribution=size_distribution,
        high_dau_dossiers=dossier_records[:8],
        recent_exports=exports[:8],
        recent_approvals=approvals[:8],
        dossier_records=dossier_records[:40],
    )


def build_cross_workspace_analytics(
    *,
    platform_version: str,
    workspaces: List[Dict[str, Any]],
    workflows: List[Dict[str, Any]],
    dossiers: List[Dict[str, Any]],
    approvals: List[Dict[str, Any]],
    exports: List[Dict[str, Any]],
    integration_bindings: List[Dict[str, Any]],
) -> Dict[str, Any]:
    workspace_map = {str(item.get("workspace_id")): item for item in workspaces}
    workflow_map = {str(item.get("workflow_id")): item for item in workflows}
    counts_by_audience: Dict[str, int] = {}
    counts_by_template: Dict[str, int] = {}
    deployability_distribution: Dict[str, int] = {}
    regime_distribution: Dict[str, int] = {}
    trade_family_distribution: Dict[str, int] = {}
    evidence_distribution: Dict[str, int] = {}
    size_distribution: Dict[str, int] = {}
    dau_values: List[float] = []
    live_count = 0
    supportive_count = 0
    recent_high_dau: List[Dict[str, Any]] = []
    workspace_views: List[WorkspaceAnalyticsView] = []

    for workspace in workspaces:
        workspace_id = workspace.get("workspace_id")
        workspace_workflows = [
            item for item in workflows if str(item.get("workspace_id")) == str(workspace_id)
        ]
        workspace_dossiers = [
            item
            for item in dossiers
            if str((workflow_map.get(str(item.get("workflow_id"))) or {}).get("workspace_id"))
            == str(workspace_id)
        ]
        workspace_approvals = [
            item
            for item in approvals
            if str((workflow_map.get(str(item.get("workflow_id"))) or {}).get("workspace_id"))
            == str(workspace_id)
        ]
        workspace_exports = [
            item for item in exports if str(item.get("workspace_id")) == str(workspace_id)
        ]
        workspace_integrations = [
            item for item in integration_bindings if str(item.get("workspace_id")) == str(workspace_id)
        ]
        workspace_views.append(
            build_workspace_analytics_view(
                workspace=workspace,
                workflows=workspace_workflows,
                dossiers=workspace_dossiers,
                approvals=workspace_approvals,
                exports=workspace_exports,
                integration_bindings=workspace_integrations,
            )
        )
        _increment(counts_by_audience, workspace.get("audience_type"))

    for workflow in workflows:
        _increment(counts_by_template, workflow.get("workflow_template_id"))
    for dossier in dossiers:
        metrics = _dossier_snapshot_metrics(dossier)
        _increment(deployability_distribution, dossier.get("latest_deployability_tier"))
        _increment(regime_distribution, dossier.get("latest_regime_label"))
        _increment(trade_family_distribution, dossier.get("latest_trade_family"))
        _increment(evidence_distribution, metrics["evidence_status"])
        _increment(size_distribution, metrics["size_band"])
        if metrics["dau"] is not None:
            dau_values.append(float(metrics["dau"]))
        if metrics["live_candidate"]:
            live_count += 1
        if metrics["evidence_status"] not in {"partial", "weak", "unknown"}:
            supportive_count += 1
        preview = dossier_preview(dossier)
        preview["deployable_alpha_utility"] = metrics["dau"]
        preview["workflow_template_id"] = (
            workflow_map.get(str(dossier.get("workflow_id"))) or {}
        ).get("workflow_template_id")
        workspace = workspace_map.get(
            str((workflow_map.get(str(dossier.get("workflow_id"))) or {}).get("workspace_id"))
        ) or {}
        preview["workspace_name"] = workspace.get("name")
        recent_high_dau.append(preview)
    recent_high_dau.sort(
        key=lambda item: (
            -1 if item.get("deployable_alpha_utility") is None else item.get("deployable_alpha_utility"),
            item.get("title") or "",
        ),
        reverse=True,
    )
    approval_throughput = {
        "pending": sum(1 for item in approvals if str(item.get("status")) == "pending"),
        "approved": sum(1 for item in approvals if str(item.get("status")) == "approved"),
        "rejected": sum(1 for item in approvals if str(item.get("status")) == "rejected"),
        "changes_requested": sum(1 for item in approvals if str(item.get("status")) == "changes_requested"),
    }
    export_throughput = {
        "generated": sum(1 for item in exports if str(item.get("status")) == "generated"),
        "workspace_count": len({str(item.get("workspace_id")) for item in exports if item.get("workspace_id")}),
    }
    payload = PlatformAnalyticsView(
        platform_version=platform_version,
        workspace_analytics=workspace_views,
        counts_by_audience_type=counts_by_audience,
        counts_by_workflow_template=counts_by_template,
        deployability_distribution=deployability_distribution,
        regime_distribution=regime_distribution,
        trade_family_distribution=trade_family_distribution,
        evidence_status_distribution=evidence_distribution,
        approval_throughput=approval_throughput,
        export_throughput=export_throughput,
        live_candidate_ratio=(live_count / len(dossiers)) if dossiers else None,
        supportive_evidence_ratio=(supportive_count / len(dossiers)) if dossiers else None,
        average_dau_across_workspaces=(sum(dau_values) / len(dau_values)) if dau_values else None,
        size_band_distribution=size_distribution,
        recent_high_dau_dossiers=recent_high_dau[:12],
    )
    return payload.model_dump(mode="python")
