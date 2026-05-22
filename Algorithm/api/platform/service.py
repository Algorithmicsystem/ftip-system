from __future__ import annotations

import datetime as dt
import uuid
from typing import Any, Dict, List, Optional

from fastapi import HTTPException

from api.assistant.reports import sanitize_payload
from api.platform.access import (
    build_access_summary,
    require_access,
    scope_resource_query,
)
from api.platform.actions import (
    apply_workflow_action,
    list_allowed_actions,
    permission_for_action,
    stage_requires_approval,
)
from api.platform.analytics import (
    build_cross_workspace_analytics,
    build_workspace_analytics_view,
)
from api.platform.approvals import (
    approval_status_after_decision,
    build_approval_decision,
    build_approval_request,
)
from api.platform.audit import audit_event_to_timeline, build_audit_event
from api.platform.contracts import DossierRecord, PlatformSummaryView, ResourceRef
from api.platform.dashboard import build_dashboard_payload
from api.platform.demo import build_demo_workspace_snapshot, build_readiness_snapshot
from api.platform.dossiers import (
    build_analysis_link,
    dossier_preview,
    refresh_dossier_record,
)
from api.platform.entities import build_coverage_entity
from api.platform.execution import (
    execute_internal_sink,
    execute_local_archive,
    execute_webhook_outbox,
)
from api.platform.export_renderers import render_export_manifest
from api.platform.exports import build_export_manifest, supported_pack_types
from api.platform.guardrails import summarize_guardrails
from api.platform.health import build_platform_health_summary
from api.platform.integration_registry import get_integration_definition, list_integration_definitions
from api.platform.integrations import build_integration_binding, integration_health_summary
from api.platform.persistence import PlatformStore, platform_store
from api.platform.profiles import (
    get_platform_profile,
    get_platform_profile_for_audience,
    list_platform_profiles,
)
from api.platform.security import normalize_user_context
from api.platform.tenant import build_tenancy_summary
from api.platform.templates import get_workflow_template, list_workflow_templates
from api.platform.workflows import build_workflow_instance


PLATFORM_FOUNDATION_VERSION = "platform_phase6_enterprise_controls_v1"


def _now_utc() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _normalize_symbol(symbol: str | None) -> str:
    return str(symbol or "").strip().upper()


def _default_org_name(profile_id: str) -> str:
    return f"FTIP {profile_id.replace('_', ' ').title()} Organization"


def _default_workspace_name(profile_id: str) -> str:
    return f"{profile_id.replace('_', ' ').title()} Workspace"


def _workspace_resource(workspace: Optional[Dict[str, Any]]) -> ResourceRef:
    return ResourceRef(
        resource_type="workspace",
        resource_id=(workspace or {}).get("workspace_id"),
        organization_id=(workspace or {}).get("organization_id"),
        workspace_id=(workspace or {}).get("workspace_id"),
    )


def _workflow_resource(
    workflow: Optional[Dict[str, Any]],
    workspace: Optional[Dict[str, Any]],
) -> ResourceRef:
    return ResourceRef(
        resource_type="workflow",
        resource_id=(workflow or {}).get("workflow_id"),
        organization_id=(workspace or {}).get("organization_id"),
        workspace_id=(workspace or {}).get("workspace_id"),
        metadata={"workflow_template_id": (workflow or {}).get("workflow_template_id")},
    )


def _dossier_resource(
    dossier: Optional[Dict[str, Any]],
    workspace: Optional[Dict[str, Any]],
) -> ResourceRef:
    return ResourceRef(
        resource_type="dossier",
        resource_id=(dossier or {}).get("dossier_id"),
        organization_id=(workspace or {}).get("organization_id"),
        workspace_id=(workspace or {}).get("workspace_id"),
        metadata={"workflow_id": (dossier or {}).get("workflow_id")},
    )


def _report_snapshot(report: Dict[str, Any]) -> Dict[str, Any]:
    return sanitize_payload(
        {
            "symbol": report.get("symbol"),
            "as_of_date": report.get("as_of_date"),
            "overall_analysis": report.get("overall_analysis"),
            "strategy_view": report.get("strategy_view"),
            "risk_quality_analysis": report.get("risk_quality_analysis"),
            "execution_quality_analysis": report.get("execution_quality_analysis"),
            "deployment_permission_analysis": report.get("deployment_permission_analysis"),
            "monitoring_triggers": report.get("monitoring_triggers") or [],
            "axiom_summary": report.get("axiom_summary"),
            "axiom_summary_card": report.get("axiom_summary_card") or {},
            "axiom_summary_card_text": report.get("axiom_summary_card_text"),
            "axiom_regime_label": report.get("axiom_regime_label"),
            "axiom_trade_family": report.get("axiom_trade_family"),
            "axiom_deployability_tier": report.get("axiom_deployability_tier"),
            "axiom_validated_edge": report.get("axiom_validated_edge"),
            "axiom_deployable_alpha_utility": report.get("axiom_deployable_alpha_utility"),
            "axiom_evidence_backed_deployability_tier": report.get(
                "axiom_evidence_backed_deployability_tier"
            ),
            "axiom_size_band_recommendation": report.get(
                "axiom_size_band_recommendation"
            ),
            "axiom_final_size_band": report.get("axiom_final_size_band"),
            "axiom_portfolio_fit_label": report.get("axiom_portfolio_fit_label"),
            "axiom_portfolio_governance": report.get("axiom_portfolio_governance")
            or {},
            "axiom_portfolio_governance_summary": report.get(
                "axiom_portfolio_governance_summary"
            ),
            "axiom_historical_evidence_report": report.get(
                "axiom_historical_evidence_report"
            )
            or {},
            "axiom_historical_evidence_summary_text": report.get(
                "axiom_historical_evidence_summary_text"
            ),
            "axiom_calibration_summary": report.get("axiom_calibration_summary") or {},
            "axiom_calibration_summary_text": report.get("axiom_calibration_summary_text"),
            "axiom_risk_deployability_memo": report.get(
                "axiom_risk_deployability_memo"
            )
            or {},
            "axiom_risk_deployability_memo_summary": report.get(
                "axiom_risk_deployability_memo_summary"
            ),
            "axiom_ic_memo": report.get("axiom_ic_memo") or {},
            "axiom_lineage": report.get("axiom_lineage") or {},
            "axiom_lineage_summary": report.get("axiom_lineage_summary"),
            "axiom_framework_version": report.get("axiom_framework_version"),
            "report_version": report.get("report_version"),
        }
    )


def _latest_report_from_dossier(dossier: Dict[str, Any]) -> Dict[str, Any]:
    return sanitize_payload(
        ((dossier.get("metadata") or {}).get("latest_report_snapshot") or {})
    )


def _latest_approval_status(
    workflow_id: str,
    *,
    store: PlatformStore,
) -> Optional[str]:
    approvals = store.list_approval_requests(workflow_id=workflow_id)
    return approvals[0].get("status") if approvals else None


def _workspace_scope_bundle(
    *,
    workspace_id: Optional[str],
    organization_ids: Optional[List[str]] = None,
    workspace_ids: Optional[List[str]] = None,
    store: PlatformStore,
) -> Dict[str, Any]:
    if organization_ids == [] and workspace_ids == []:
        return {
            "workspaces": [],
            "workflows": [],
            "dossiers": [],
            "approvals": [],
            "exports": [],
            "rendered_exports": [],
            "integration_bindings": [],
        }
    workspaces = (
        [
            store.get_workspace(
                workspace_id,
                organization_ids=organization_ids,
                workspace_ids=workspace_ids,
            )
        ]
        if workspace_id
        and store.get_workspace(
            workspace_id,
            organization_ids=organization_ids,
            workspace_ids=workspace_ids,
        )
        else store.list_workspaces(
            organization_ids=organization_ids,
            workspace_ids=workspace_ids,
        )
    )
    workflows = store.list_workflows(
        workspace_id=workspace_id,
        organization_ids=organization_ids,
        workspace_ids=workspace_ids,
    )
    dossiers = store.list_dossiers(
        workspace_id=workspace_id,
        organization_ids=organization_ids,
        workspace_ids=workspace_ids,
    )
    approvals: List[Dict[str, Any]] = []
    exports: List[Dict[str, Any]] = []
    rendered_exports: List[Dict[str, Any]] = []
    for workflow in workflows:
        approvals.extend(store.list_approval_requests(workflow_id=workflow.get("workflow_id")))
    for dossier in dossiers:
        dossier_exports = store.list_export_manifests(
            dossier.get("dossier_id"),
            organization_ids=organization_ids,
            workspace_ids=workspace_ids,
        )
        exports.extend(dossier_exports)
        for export in dossier_exports:
            rendered_exports.extend(
                store.list_rendered_exports(
                    export_id=export.get("export_id"),
                    organization_ids=organization_ids,
                    workspace_ids=workspace_ids,
                )
            )
    integrations = (
        store.list_integration_bindings(
            workspace_id=workspace_id,
            organization_ids=organization_ids,
            workspace_ids=workspace_ids,
        )
        if workspace_id
        else store.list_integration_bindings(
            organization_ids=organization_ids,
            workspace_ids=workspace_ids,
        )
    )
    return {
        "workspaces": [item for item in workspaces if item],
        "workflows": workflows,
        "dossiers": dossiers,
        "approvals": approvals,
        "exports": exports,
        "rendered_exports": rendered_exports,
        "integration_bindings": integrations,
    }


def _resolve_scope(
    *,
    user_context: Optional[Dict[str, Any]] = None,
    workspace_id: Optional[str] = None,
    organization_id: Optional[str] = None,
    store: PlatformStore,
) -> Dict[str, Any]:
    return scope_resource_query(
        user_context=user_context,
        requested_workspace_id=workspace_id,
        requested_organization_id=organization_id,
        store=store,
    )


def _scope_allows_listing(scope: Dict[str, Any]) -> bool:
    return bool(
        scope.get("has_access", True)
        or getattr(scope.get("user_context"), "is_system", False)
    )


def _create_audit_event(
    *,
    event_type: str,
    resource: ResourceRef,
    user_context: Any,
    payload: Optional[Dict[str, Any]],
    rationale: Optional[str],
    metadata: Optional[Dict[str, Any]],
    store: PlatformStore,
) -> Dict[str, Any]:
    event = build_audit_event(
        event_type=event_type,
        resource=resource,
        user_context=user_context,
        payload=payload,
        rationale=rationale,
        metadata=metadata,
    )
    return store.create_audit_event(event.model_dump(mode="python"))


def ensure_default_foundation(
    *,
    audience_type: str,
    report_profile: str,
    platform_profile: Optional[str],
    store: PlatformStore = platform_store,
) -> Dict[str, Dict[str, Any]]:
    profile = (
        get_platform_profile(platform_profile)
        if platform_profile
        else get_platform_profile_for_audience(audience_type)
    )
    organization_id = f"default-{profile.profile_id}-org"
    workspace_id = f"default-{profile.profile_id}-workspace"
    organization = store.get_organization(organization_id)
    if organization is None:
        organization = store.create_organization(
            {
                "organization_id": organization_id,
                "name": _default_org_name(profile.profile_id),
                "organization_type": profile.audience_type,
                "settings": {"platform_profile": profile.profile_id},
            }
        )
    workspace = store.get_workspace(workspace_id)
    if workspace is None:
        workspace = store.create_workspace(
            {
                "workspace_id": workspace_id,
                "organization_id": organization["organization_id"],
                "name": _default_workspace_name(profile.profile_id),
                "audience_type": audience_type or profile.audience_type,
                "report_profile": report_profile or profile.default_report_profile,
                "default_workflow_template": profile.default_workflow_template,
                "platform_profile": profile.profile_id,
                "settings": {
                    "default_memo_emphasis": profile.default_memo_emphasis,
                    "preferred_axiom_sections": profile.preferred_axiom_sections,
                    "preferred_dossier_sections": profile.preferred_dossier_sections,
                },
            }
        )
    return {
        "organization": organization,
        "workspace": workspace,
        "platform_profile": profile.model_dump(mode="python"),
    }


def create_workspace_service(
    payload: Dict[str, Any],
    *,
    user_context: Optional[Dict[str, Any]] = None,
    store: PlatformStore = platform_store,
) -> Dict[str, Any]:
    profile = get_platform_profile(payload.get("platform_profile"))
    organization_id = payload.get("organization_id")
    existing_org = store.get_organization(organization_id) if organization_id else None
    normalized = normalize_user_context(
        user_context,
        organization_id=organization_id,
    )
    if existing_org is not None:
        require_access(
            permission="edit_workspace",
            user_context=normalized,
            resource=ResourceRef(
                resource_type="organization",
                resource_id=existing_org["organization_id"],
                organization_id=existing_org["organization_id"],
            ),
            store=store,
        )
    if not organization_id:
        org = store.create_organization(
            {
                "name": payload.get("name") or _default_org_name(profile.profile_id),
                "organization_type": profile.audience_type,
                "settings": {"platform_profile": profile.profile_id},
            }
        )
        organization_id = org["organization_id"]
        normalized = normalize_user_context(
            user_context,
            organization_id=organization_id,
        )
    workspace = store.create_workspace(
        {
            "organization_id": organization_id,
            "name": payload.get("name"),
            "audience_type": payload.get("audience_type") or profile.audience_type,
            "report_profile": payload.get("report_profile")
            or profile.default_report_profile,
            "default_workflow_template": payload.get("default_workflow_template")
            or profile.default_workflow_template,
            "platform_profile": profile.profile_id,
            "settings": payload.get("settings") or {},
        }
    )
    normalized = normalize_user_context(
        normalized,
        organization_id=organization_id,
        workspace_id=workspace.get("workspace_id"),
    )
    _create_audit_event(
        event_type="workspace_created",
        resource=_workspace_resource(workspace),
        user_context=normalized,
        payload={"workspace": workspace},
        rationale="Workspace created.",
        metadata={"platform_profile": profile.profile_id},
        store=store,
    )
    return {
        "platform_version": PLATFORM_FOUNDATION_VERSION,
        "workspace": workspace,
        "platform_profile": profile.model_dump(mode="python"),
    }


def build_auth_session_service(
    *,
    workspace_id: Optional[str] = None,
    organization_id: Optional[str] = None,
    user_context: Optional[Dict[str, Any]] = None,
    store: PlatformStore = platform_store,
) -> Dict[str, Any]:
    scope = _resolve_scope(
        user_context=user_context,
        workspace_id=workspace_id,
        organization_id=organization_id,
        store=store,
    )
    normalized = scope["user_context"]
    accessible_workspaces = (
        []
        if not _scope_allows_listing(scope)
        else store.list_workspaces(
            organization_ids=scope["organization_ids"],
            workspace_ids=scope["workspace_ids"],
        )
    )
    return {
        "platform_version": PLATFORM_FOUNDATION_VERSION,
        "session": sanitize_payload(
            {
                "user_id": normalized.user_id,
                "user_name": normalized.user_name,
                "email": normalized.email,
                "username": normalized.username,
                "role": normalized.role,
                "auth_mode": normalized.auth_mode,
                "is_system": normalized.is_system,
                "session_id": normalized.session_id,
                "organization_id": normalized.organization_id,
                "workspace_id": normalized.workspace_id,
                "organization_ids": normalized.organization_ids,
                "workspace_ids": normalized.workspace_ids,
                "tenant_scope_summary": (normalized.metadata or {}).get(
                    "tenant_scope_summary"
                ),
                "fallback_used": bool((normalized.metadata or {}).get("fallback_used")),
            }
        ),
        "tenancy_summary": build_tenancy_summary(
            user_context=normalized,
            accessible_workspaces=accessible_workspaces,
        ),
    }


def list_workspaces_service(
    *,
    organization_id: Optional[str] = None,
    user_context: Optional[Dict[str, Any]] = None,
    store: PlatformStore = platform_store,
) -> Dict[str, Any]:
    scope = _resolve_scope(
        user_context=user_context,
        organization_id=organization_id,
        store=store,
    )
    workspaces = (
        []
        if not _scope_allows_listing(scope)
        else store.list_workspaces(
            organization_id=organization_id,
            organization_ids=scope["organization_ids"],
            workspace_ids=scope["workspace_ids"],
        )
    )
    return {
        "platform_version": PLATFORM_FOUNDATION_VERSION,
        "workspaces": workspaces,
        "tenancy_summary": build_tenancy_summary(
            user_context=scope["user_context"],
            accessible_workspaces=workspaces,
        ),
    }


def list_workflows_service(
    *,
    workspace_id: Optional[str] = None,
    user_context: Optional[Dict[str, Any]] = None,
    store: PlatformStore = platform_store,
) -> Dict[str, Any]:
    scope = _resolve_scope(
        user_context=user_context,
        workspace_id=workspace_id,
        store=store,
    )
    workflows = (
        []
        if not _scope_allows_listing(scope)
        else store.list_workflows(
            workspace_id=workspace_id,
            organization_ids=scope["organization_ids"],
            workspace_ids=scope["workspace_ids"],
        )
    )
    return {
        "platform_version": PLATFORM_FOUNDATION_VERSION,
        "workflows": workflows,
        "tenancy_summary": build_tenancy_summary(user_context=scope["user_context"]),
    }


def list_dossiers_service(
    *,
    workspace_id: Optional[str] = None,
    workflow_id: Optional[str] = None,
    user_context: Optional[Dict[str, Any]] = None,
    store: PlatformStore = platform_store,
) -> Dict[str, Any]:
    scope = _resolve_scope(
        user_context=user_context,
        workspace_id=workspace_id,
        store=store,
    )
    dossiers = (
        []
        if not _scope_allows_listing(scope)
        else store.list_dossiers(
            workspace_id=workspace_id,
            workflow_id=workflow_id,
            organization_ids=scope["organization_ids"],
            workspace_ids=scope["workspace_ids"],
        )
    )
    return {
        "platform_version": PLATFORM_FOUNDATION_VERSION,
        "dossiers": dossiers,
        "tenancy_summary": build_tenancy_summary(user_context=scope["user_context"]),
    }


def create_workflow_service(
    payload: Dict[str, Any],
    *,
    user_context: Optional[Dict[str, Any]] = None,
    store: PlatformStore = platform_store,
) -> Dict[str, Any]:
    workspace = store.get_workspace(payload.get("workspace_id") or "")
    if workspace is None:
        raise HTTPException(status_code=404, detail="workspace not found")
    normalized = normalize_user_context(
        user_context,
        organization_id=workspace.get("organization_id"),
        workspace_id=workspace.get("workspace_id"),
    )
    require_access(
        permission="view_workspace",
        user_context=normalized,
        resource=_workspace_resource(workspace),
        store=store,
    )
    template = get_workflow_template(
        payload.get("workflow_template_id") or workspace.get("default_workflow_template")
    )
    workflow = build_workflow_instance(
        workflow_id=str(uuid.uuid4()),
        workspace_id=workspace["workspace_id"],
        template=template,
        title=str(payload.get("title") or f"{template.title} Workflow"),
        status=str(payload.get("status") or "active"),
        stage=payload.get("stage"),
        priority=str(payload.get("priority") or "normal"),
        owner_placeholder=payload.get("owner_placeholder"),
        metadata=payload.get("metadata") or {},
    ).model_dump(mode="python")
    persisted = store.create_workflow(workflow)
    _create_audit_event(
        event_type="workflow_created",
        resource=_workflow_resource(persisted, workspace),
        user_context=normalized,
        payload={"workflow": persisted, "template": template.template_id},
        rationale="Workflow created.",
        metadata={"workspace_id": workspace["workspace_id"]},
        store=store,
    )
    return {
        "platform_version": PLATFORM_FOUNDATION_VERSION,
        "workflow": persisted,
        "template": template.model_dump(mode="python"),
        "workspace": workspace,
    }


def _resolve_entity(
    *,
    symbol: Optional[str],
    report: Optional[Dict[str, Any]],
    entity_id: Optional[str],
    display_name: Optional[str],
    entity_type: str,
    sector: Optional[str],
    strategy: Optional[str],
    theme: Optional[str],
    store: PlatformStore,
) -> Dict[str, Any]:
    if entity_id:
        for existing in getattr(store, "_entities", {}).values():
            if str(existing.get("entity_id")) == str(entity_id):
                return existing
    resolved_symbol = _normalize_symbol(symbol or (report or {}).get("symbol"))
    existing = (
        store.find_coverage_entity_by_symbol(resolved_symbol) if resolved_symbol else None
    )
    if existing is not None:
        return existing
    entity = build_coverage_entity(
        symbol=resolved_symbol,
        display_name=display_name or resolved_symbol,
        entity_type=entity_type,
        sector=sector
        or ((report or {}).get("data_bundle") or {}).get("symbol_meta", {}).get("sector"),
        strategy=strategy,
        theme=theme or (report or {}).get("axiom_trade_family"),
        metadata={"source": "platform_phase6"},
        entity_id=entity_id,
    )
    return store.upsert_coverage_entity(entity.model_dump(mode="python"))


def create_dossier_service(
    payload: Dict[str, Any],
    *,
    user_context: Optional[Dict[str, Any]] = None,
    store: PlatformStore = platform_store,
) -> Dict[str, Any]:
    workflow = store.get_workflow(payload.get("workflow_id") or "")
    if workflow is None:
        raise HTTPException(status_code=404, detail="workflow not found")
    workspace = store.get_workspace(workflow["workspace_id"])
    normalized = normalize_user_context(
        user_context,
        organization_id=(workspace or {}).get("organization_id"),
        workspace_id=(workspace or {}).get("workspace_id"),
    )
    require_access(
        permission="create_dossier",
        user_context=normalized,
        resource=_workflow_resource(workflow, workspace),
        store=store,
    )
    entity = _resolve_entity(
        symbol=payload.get("symbol"),
        report=None,
        entity_id=payload.get("entity_id"),
        display_name=payload.get("display_name"),
        entity_type=str(payload.get("entity_type") or "public_equity"),
        sector=payload.get("sector"),
        strategy=payload.get("strategy"),
        theme=payload.get("theme"),
        store=store,
    )
    dossier = store.create_dossier(
        DossierRecord(
            dossier_id=str(uuid.uuid4()),
            workflow_id=workflow["workflow_id"],
            entity_id=entity["entity_id"],
            dossier_type=str(payload.get("dossier_type") or "coverage"),
            title=str(
                payload.get("title")
                or f"{entity.get('display_name') or entity.get('symbol')} Dossier"
            ),
            current_summary={
                "symbol": entity.get("symbol"),
                "display_name": entity.get("display_name"),
                "status": "created",
            },
            evidence_status="partial",
            workflow_stage_state=workflow.get("stage_state"),
            metadata=payload.get("metadata") or {},
        ).model_dump(mode="python")
    )
    _create_audit_event(
        event_type="dossier_created",
        resource=_dossier_resource(dossier, workspace),
        user_context=normalized,
        payload={"dossier": dossier, "entity": entity},
        rationale="Dossier created.",
        metadata={"workflow_id": workflow["workflow_id"]},
        store=store,
    )
    return {
        "platform_version": PLATFORM_FOUNDATION_VERSION,
        "dossier": dossier,
        "workflow": workflow,
        "workspace": workspace,
        "entity": entity,
    }


def get_dossier_view(
    dossier_id: str,
    *,
    user_context: Optional[Dict[str, Any]] = None,
    store: PlatformStore = platform_store,
) -> Dict[str, Any]:
    dossier = store.get_dossier(dossier_id)
    if dossier is None:
        raise HTTPException(status_code=404, detail="dossier not found")
    workflow = store.get_workflow(dossier["workflow_id"])
    workspace = store.get_workspace(workflow["workspace_id"]) if workflow else None
    normalized = normalize_user_context(
        user_context,
        organization_id=(workspace or {}).get("organization_id"),
        workspace_id=(workspace or {}).get("workspace_id"),
    )
    require_access(
        permission="view_workspace",
        user_context=normalized,
        resource=_dossier_resource(dossier, workspace),
        store=store,
    )
    links = store.list_dossier_analysis_links(dossier_id)
    approvals = store.list_approval_requests(
        workflow_id=(workflow or {}).get("workflow_id"),
        dossier_id=dossier_id,
    )
    exports = store.list_export_manifests(dossier_id)
    timeline = list_workflow_timeline_service(
        (workflow or {}).get("workflow_id"),
        user_context=normalized,
        store=store,
    ) if workflow else {"timeline": []}
    return {
        "platform_version": PLATFORM_FOUNDATION_VERSION,
        "dossier": dossier,
        "workflow": workflow,
        "workspace": workspace,
        "analysis_links": links,
        "approvals": approvals,
        "exports": exports,
        "timeline": timeline.get("timeline") or [],
    }


def attach_analysis_to_dossier_service(
    dossier_id: str,
    payload: Dict[str, Any],
    *,
    user_context: Optional[Dict[str, Any]] = None,
    store: PlatformStore = platform_store,
) -> Dict[str, Any]:
    dossier = store.get_dossier(dossier_id)
    if dossier is None:
        raise HTTPException(status_code=404, detail="dossier not found")
    workflow = store.get_workflow(dossier["workflow_id"])
    workspace = store.get_workspace(workflow["workspace_id"]) if workflow else None
    normalized = normalize_user_context(
        user_context,
        organization_id=(workspace or {}).get("organization_id"),
        workspace_id=(workspace or {}).get("workspace_id"),
    )
    require_access(
        permission="attach_analysis",
        user_context=normalized,
        resource=_dossier_resource(dossier, workspace),
        store=store,
    )
    report = sanitize_payload(payload.get("report") or {})
    analysis_link = build_analysis_link(
        report=report,
        report_id=payload.get("report_id"),
        session_id=payload.get("session_id"),
        axiom_artifact_id=payload.get("axiom_artifact_id"),
        axiom_report_pack_artifact_id=payload.get("axiom_report_pack_artifact_id"),
        axiom_lineage_artifact_id=payload.get("axiom_lineage_artifact_id"),
        axiom_history_artifact_id=payload.get("axiom_history_artifact_id"),
        axiom_calibration_artifact_id=payload.get("axiom_calibration_artifact_id"),
    )
    store.add_dossier_analysis_link(dossier_id, analysis_link.model_dump(mode="python"))
    refreshed = refresh_dossier_record(dossier, report=report, analysis_link=analysis_link)
    refreshed["metadata"] = {
        **dict(refreshed.get("metadata") or {}),
        "latest_report_snapshot": _report_snapshot(report),
        "latest_report_id": payload.get("report_id"),
        "latest_session_id": payload.get("session_id"),
        "last_attached_at": _now_utc(),
    }
    persisted = store.update_dossier(dossier_id, refreshed)
    workflow = store.get_workflow(persisted["workflow_id"])
    workspace = store.get_workspace(workflow["workspace_id"]) if workflow else None
    _create_audit_event(
        event_type="analysis_attached",
        resource=_dossier_resource(persisted, workspace),
        user_context=normalized,
        payload={
            "dossier_id": dossier_id,
            "workflow_id": (workflow or {}).get("workflow_id"),
            "report_id": payload.get("report_id"),
            "axiom_artifact_id": payload.get("axiom_artifact_id"),
        },
        rationale="Analysis attached to dossier.",
        metadata={"workflow_id": (workflow or {}).get("workflow_id")},
        store=store,
    )
    return {
        "platform_version": PLATFORM_FOUNDATION_VERSION,
        "dossier": persisted,
        "workflow": workflow,
        "workspace": workspace,
        "analysis_link": analysis_link.model_dump(mode="python"),
        "dossier_preview": dossier_preview(persisted),
    }


def create_membership_service(
    payload: Dict[str, Any],
    *,
    store: PlatformStore = platform_store,
) -> Dict[str, Any]:
    membership = store.create_membership(payload)
    return {
        "platform_version": PLATFORM_FOUNDATION_VERSION,
        "membership": membership,
    }


def build_access_summary_service(
    *,
    workspace_id: Optional[str] = None,
    user_context: Optional[Dict[str, Any]] = None,
    store: PlatformStore = platform_store,
) -> Dict[str, Any]:
    scope = _resolve_scope(
        user_context=user_context,
        workspace_id=workspace_id,
        store=store,
    )
    workspace = (
        store.get_workspace(
            workspace_id,
            organization_ids=scope["organization_ids"],
            workspace_ids=scope["workspace_ids"],
        )
        if workspace_id
        else None
    )
    summary = build_access_summary(
        user_context=scope["user_context"],
        resource=_workspace_resource(workspace),
        store=store,
    )
    return {
        "platform_version": PLATFORM_FOUNDATION_VERSION,
        "access_summary": summary,
    }


def build_effective_access_service(
    *,
    workspace_id: Optional[str] = None,
    organization_id: Optional[str] = None,
    user_context: Optional[Dict[str, Any]] = None,
    store: PlatformStore = platform_store,
) -> Dict[str, Any]:
    session = build_auth_session_service(
        workspace_id=workspace_id,
        organization_id=organization_id,
        user_context=user_context,
        store=store,
    )
    access = build_access_summary_service(
        workspace_id=workspace_id,
        user_context=user_context,
        store=store,
    )
    return {
        "platform_version": PLATFORM_FOUNDATION_VERSION,
        "session": session["session"],
        "tenancy_summary": session["tenancy_summary"],
        "access_summary": access["access_summary"],
    }


def build_tenancy_summary_service(
    *,
    workspace_id: Optional[str] = None,
    organization_id: Optional[str] = None,
    user_context: Optional[Dict[str, Any]] = None,
    store: PlatformStore = platform_store,
) -> Dict[str, Any]:
    session = build_auth_session_service(
        workspace_id=workspace_id,
        organization_id=organization_id,
        user_context=user_context,
        store=store,
    )
    return {
        "platform_version": PLATFORM_FOUNDATION_VERSION,
        "tenancy_summary": session["tenancy_summary"],
        "session": session["session"],
    }


def execute_workflow_action_service(
    workflow_id: str,
    payload: Dict[str, Any],
    *,
    user_context: Optional[Dict[str, Any]] = None,
    store: PlatformStore = platform_store,
) -> Dict[str, Any]:
    workflow = store.get_workflow(workflow_id)
    if workflow is None:
        raise HTTPException(status_code=404, detail="workflow not found")
    workspace = store.get_workspace(workflow["workspace_id"])
    dossier = (
        store.get_dossier(str(payload.get("dossier_id")))
        if payload.get("dossier_id")
        else next(
            iter(store.list_dossiers(workflow_id=workflow_id, limit=1)),
            None,
        )
    )
    normalized = normalize_user_context(
        user_context,
        organization_id=(workspace or {}).get("organization_id"),
        workspace_id=(workspace or {}).get("workspace_id"),
    )
    permission = permission_for_action(str(payload.get("action_type") or ""))
    require_access(
        permission=permission,
        user_context=normalized,
        resource=_workflow_resource(workflow, workspace),
        store=store,
    )
    template = get_workflow_template(workflow.get("workflow_template_id"))
    approvals = store.list_approval_requests(
        workflow_id=workflow_id,
        dossier_id=(dossier or {}).get("dossier_id"),
    )
    try:
        applied = apply_workflow_action(
            workflow=workflow,
            dossier=dossier,
            template=template,
            action_type=str(payload.get("action_type") or ""),
            requested_stage=payload.get("requested_stage"),
            requested_status=payload.get("requested_status"),
            rationale=payload.get("rationale"),
            metadata=payload.get("metadata") or {},
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    transition = applied.get("transition") or {}
    if (
        str(payload.get("action_type") or "") == "advance_stage"
        and transition.get("requires_approval")
    ):
        approved = any(
            str(item.get("status")) == "approved"
            and str(item.get("stage") or "") == str(transition.get("to_stage") or "")
            for item in approvals
        )
        if not approved:
            raise HTTPException(
                status_code=409,
                detail=f"Stage {transition.get('to_stage')} requires approval before advancement.",
            )
    updated_workflow = workflow
    if applied.get("workflow_updates"):
        updated_workflow = store.create_workflow(
            {
                **workflow,
                **sanitize_payload(applied["workflow_updates"]),
                "updated_at": _now_utc(),
            }
        )
    updated_dossier = dossier
    if dossier is not None and applied.get("dossier_updates"):
        updated_dossier = store.update_dossier(
            dossier["dossier_id"],
            {
                **sanitize_payload(applied["dossier_updates"]),
                "updated_at": _now_utc(),
            },
        )
    audit = _create_audit_event(
        event_type=f"workflow_{payload.get('action_type')}",
        resource=_workflow_resource(updated_workflow, workspace),
        user_context=normalized,
        payload={
            "workflow_id": workflow_id,
            "dossier_id": (updated_dossier or {}).get("dossier_id"),
            "stage": updated_workflow.get("stage"),
            "status": updated_workflow.get("status"),
            "summary": applied.get("summary"),
        },
        rationale=payload.get("rationale"),
        metadata={
            "workflow_id": workflow_id,
            "dossier_id": (updated_dossier or {}).get("dossier_id"),
        },
        store=store,
    )
    guardrails = summarize_guardrails(
        workflow=updated_workflow,
        dossier=updated_dossier,
        approval_status=_latest_approval_status(workflow_id, store=store),
    )
    access_summary = build_access_summary(
        user_context=normalized,
        resource=_workflow_resource(updated_workflow, workspace),
        store=store,
    )
    allowed_actions = list_allowed_actions(
        workflow=updated_workflow,
        dossier=updated_dossier,
        template=template,
        effective_permissions=access_summary.get("effective_permissions") or [],
        approvals=store.list_approval_requests(
            workflow_id=workflow_id,
            dossier_id=(updated_dossier or {}).get("dossier_id"),
        ),
    )
    return {
        "platform_version": PLATFORM_FOUNDATION_VERSION,
        "workflow": updated_workflow,
        "dossier": updated_dossier,
        "action": sanitize_payload(payload),
        "transition": transition,
        "allowed_actions": allowed_actions,
        "guardrails": guardrails,
        "audit_event": audit,
    }


def handle_workflow_approval_service(
    workflow_id: str,
    payload: Dict[str, Any],
    *,
    user_context: Optional[Dict[str, Any]] = None,
    store: PlatformStore = platform_store,
) -> Dict[str, Any]:
    workflow = store.get_workflow(workflow_id)
    if workflow is None:
        raise HTTPException(status_code=404, detail="workflow not found")
    workspace = store.get_workspace(workflow["workspace_id"])
    dossier = (
        store.get_dossier(str(payload.get("dossier_id")))
        if payload.get("dossier_id")
        else next(iter(store.list_dossiers(workflow_id=workflow_id, limit=1)), None)
    )
    normalized = normalize_user_context(
        user_context,
        organization_id=(workspace or {}).get("organization_id"),
        workspace_id=(workspace or {}).get("workspace_id"),
    )
    mode = str(payload.get("mode") or "request")
    if mode == "request":
        require_access(
            permission="request_approval",
            user_context=normalized,
            resource=_workflow_resource(workflow, workspace),
            store=store,
        )
        template = get_workflow_template(workflow.get("workflow_template_id"))
        stage = (workflow.get("stage_state") or {}).get("next_stage") or workflow.get("stage")
        approval = build_approval_request(
            workflow_id=workflow_id,
            dossier_id=(dossier or {}).get("dossier_id"),
            requested_role=str(payload.get("requested_role") or "committee"),
            user_context=normalized,
            stage=str(stage),
            rationale=payload.get("rationale"),
            required_permissions=["approve_stage"]
            if stage_requires_approval(template, str(stage))
            else [],
            metadata=payload.get("metadata") or {},
        )
        persisted = store.create_approval_request(approval.model_dump(mode="python"))
        audit = _create_audit_event(
            event_type="approval_requested",
            resource=_workflow_resource(workflow, workspace),
            user_context=normalized,
            payload={
                "workflow_id": workflow_id,
                "dossier_id": (dossier or {}).get("dossier_id"),
                "approval_id": persisted["approval_id"],
                "stage": persisted.get("stage"),
                "status": persisted.get("status"),
            },
            rationale=payload.get("rationale"),
            metadata={"workflow_id": workflow_id, "dossier_id": (dossier or {}).get("dossier_id")},
            store=store,
        )
        return {
            "platform_version": PLATFORM_FOUNDATION_VERSION,
            "approval": persisted,
            "workflow": workflow,
            "dossier": dossier,
            "audit_event": audit,
        }

    require_access(
        permission="approve_stage",
        user_context=normalized,
        resource=_workflow_resource(workflow, workspace),
        store=store,
    )
    approval_id = payload.get("approval_id")
    approval = store.get_approval_request(str(approval_id or ""))
    if approval is None:
        raise HTTPException(status_code=404, detail="approval request not found")
    decision = build_approval_decision(
        approval_id=approval["approval_id"],
        decision_type=str(payload.get("decision_type") or "request_changes"),
        user_context=normalized,
        rationale=payload.get("rationale"),
        metadata=payload.get("metadata") or {},
    )
    decisions = list(approval.get("decisions") or [])
    decisions.append(decision.model_dump(mode="python"))
    updated_approval = store.update_approval_request(
        approval["approval_id"],
        {
            "status": approval_status_after_decision(decision.decision_type),
            "decisions": decisions,
            "updated_at": _now_utc(),
        },
    )
    action_map = {
        "approve": "approve",
        "reject": "reject",
        "request_changes": "request_changes",
    }
    workflow_result = execute_workflow_action_service(
        workflow_id,
        {
            "dossier_id": (dossier or {}).get("dossier_id"),
            "action_type": action_map.get(decision.decision_type, "request_changes"),
            "rationale": payload.get("rationale"),
        },
        user_context=normalized.model_dump(mode="python"),
        store=store,
    )
    audit = _create_audit_event(
        event_type=f"approval_{decision.decision_type}",
        resource=_workflow_resource(workflow_result["workflow"], workspace),
        user_context=normalized,
        payload={
            "workflow_id": workflow_id,
            "approval_id": updated_approval["approval_id"],
            "dossier_id": (dossier or {}).get("dossier_id"),
            "status": updated_approval["status"],
        },
        rationale=payload.get("rationale"),
        metadata={"workflow_id": workflow_id, "dossier_id": (dossier or {}).get("dossier_id")},
        store=store,
    )
    return {
        "platform_version": PLATFORM_FOUNDATION_VERSION,
        "approval": updated_approval,
        "decision": decision.model_dump(mode="python"),
        "workflow": workflow_result["workflow"],
        "dossier": workflow_result["dossier"],
        "audit_event": audit,
    }


def list_workflow_timeline_service(
    workflow_id: str,
    *,
    user_context: Optional[Dict[str, Any]] = None,
    store: PlatformStore = platform_store,
) -> Dict[str, Any]:
    workflow = store.get_workflow(str(workflow_id))
    if workflow is None:
        raise HTTPException(status_code=404, detail="workflow not found")
    workspace = store.get_workspace(workflow["workspace_id"])
    normalized = normalize_user_context(
        user_context,
        organization_id=(workspace or {}).get("organization_id"),
        workspace_id=(workspace or {}).get("workspace_id"),
    )
    require_access(
        permission="view_workspace",
        user_context=normalized,
        resource=_workflow_resource(workflow, workspace),
        store=store,
    )
    events = [
        item
        for item in store.list_audit_events(
            workspace_id=(workspace or {}).get("workspace_id"),
            limit=400,
        )
        if str((item.get("payload") or {}).get("workflow_id") or (item.get("metadata") or {}).get("workflow_id") or "") == str(workflow_id)
    ]
    timeline = [
        audit_event_to_timeline(item).model_dump(mode="python")
        for item in events
    ]
    return {
        "platform_version": PLATFORM_FOUNDATION_VERSION,
        "workflow": workflow,
        "timeline": timeline,
    }


def create_dossier_export_service(
    dossier_id: str,
    payload: Dict[str, Any],
    *,
    user_context: Optional[Dict[str, Any]] = None,
    store: PlatformStore = platform_store,
) -> Dict[str, Any]:
    dossier = store.get_dossier(dossier_id)
    if dossier is None:
        raise HTTPException(status_code=404, detail="dossier not found")
    workflow = store.get_workflow(dossier["workflow_id"])
    workspace = store.get_workspace(workflow["workspace_id"]) if workflow else None
    organization = (
        store.get_organization(workspace["organization_id"]) if workspace else None
    )
    normalized = normalize_user_context(
        user_context,
        organization_id=(workspace or {}).get("organization_id"),
        workspace_id=(workspace or {}).get("workspace_id"),
    )
    require_access(
        permission="export_report_pack",
        user_context=normalized,
        resource=_dossier_resource(dossier, workspace),
        store=store,
    )
    report = _latest_report_from_dossier(dossier)
    approval_status = _latest_approval_status(dossier["workflow_id"], store=store)
    manifest = build_export_manifest(
        pack_type=str(payload.get("pack_type") or "dossier_pack"),
        dossier=dossier,
        report=report,
        workflow=workflow,
        workspace=workspace,
        organization=organization,
        approval_status=approval_status,
        metadata=payload.get("metadata") or {},
    )
    persisted = store.create_export_manifest(manifest.model_dump(mode="python"))
    audit = _create_audit_event(
        event_type="export_generated",
        resource=_dossier_resource(dossier, workspace),
        user_context=normalized,
        payload={
            "workflow_id": dossier.get("workflow_id"),
            "dossier_id": dossier_id,
            "export_id": persisted["export_id"],
            "pack_type": persisted["pack_type"],
            "status": persisted["status"],
        },
        rationale="Export pack generated.",
        metadata={"workflow_id": dossier.get("workflow_id"), "dossier_id": dossier_id},
        store=store,
    )
    return {
        "platform_version": PLATFORM_FOUNDATION_VERSION,
        "export": persisted,
        "audit_event": audit,
    }


def list_dossier_exports_service(
    dossier_id: str,
    *,
    user_context: Optional[Dict[str, Any]] = None,
    store: PlatformStore = platform_store,
) -> Dict[str, Any]:
    dossier = store.get_dossier(dossier_id)
    if dossier is None:
        raise HTTPException(status_code=404, detail="dossier not found")
    workflow = store.get_workflow(dossier["workflow_id"])
    workspace = store.get_workspace(workflow["workspace_id"]) if workflow else None
    normalized = normalize_user_context(
        user_context,
        organization_id=(workspace or {}).get("organization_id"),
        workspace_id=(workspace or {}).get("workspace_id"),
    )
    require_access(
        permission="view_workspace",
        user_context=normalized,
        resource=_dossier_resource(dossier, workspace),
        store=store,
    )
    scope = _resolve_scope(
        user_context=normalized.model_dump(mode="python"),
        workspace_id=(workspace or {}).get("workspace_id"),
        store=store,
    )
    return {
        "platform_version": PLATFORM_FOUNDATION_VERSION,
        "supported_pack_types": supported_pack_types(),
        "exports": store.list_export_manifests(
            dossier_id,
            organization_ids=scope["organization_ids"],
            workspace_ids=scope["workspace_ids"],
        ),
        "rendered_exports": [
            rendered
            for export in store.list_export_manifests(
                dossier_id,
                organization_ids=scope["organization_ids"],
                workspace_ids=scope["workspace_ids"],
            )
            for rendered in store.list_rendered_exports(
                export_id=export.get("export_id"),
                organization_ids=scope["organization_ids"],
                workspace_ids=scope["workspace_ids"],
            )
        ],
    }


def _resolve_dossier_export_context(
    dossier_id: str,
    *,
    user_context: Optional[Dict[str, Any]],
    store: PlatformStore,
) -> Dict[str, Any]:
    dossier = store.get_dossier(dossier_id)
    if dossier is None:
        raise HTTPException(status_code=404, detail="dossier not found")
    workflow = store.get_workflow(dossier["workflow_id"])
    workspace = store.get_workspace(workflow["workspace_id"]) if workflow else None
    organization = (
        store.get_organization(workspace["organization_id"]) if workspace else None
    )
    normalized = normalize_user_context(
        user_context,
        organization_id=(workspace or {}).get("organization_id"),
        workspace_id=(workspace or {}).get("workspace_id"),
    )
    require_access(
        permission="export_report_pack",
        user_context=normalized,
        resource=_dossier_resource(dossier, workspace),
        store=store,
    )
    report = _latest_report_from_dossier(dossier)
    approval_status = _latest_approval_status(dossier["workflow_id"], store=store)
    return {
        "dossier": dossier,
        "workflow": workflow,
        "workspace": workspace,
        "organization": organization,
        "normalized": normalized,
        "report": report,
        "approval_status": approval_status,
    }


def preview_dossier_export_service(
    dossier_id: str,
    *,
    pack_type: str = "dossier_pack",
    export_format: str = "html",
    user_context: Optional[Dict[str, Any]] = None,
    store: PlatformStore = platform_store,
) -> Dict[str, Any]:
    context = _resolve_dossier_export_context(
        dossier_id,
        user_context=user_context,
        store=store,
    )
    manifest = build_export_manifest(
        pack_type=pack_type,
        dossier=context["dossier"],
        report=context["report"],
        workflow=context["workflow"],
        workspace=context["workspace"],
        organization=context["organization"],
        approval_status=context["approval_status"],
        metadata={"preview_only": True},
    )
    rendered = render_export_manifest(
        manifest,
        export_format=export_format,
        metadata={"preview_only": True},
    )
    audit = _create_audit_event(
        event_type="export_previewed",
        resource=_dossier_resource(context["dossier"], context["workspace"]),
        user_context=context["normalized"],
        payload={
            "workflow_id": context["dossier"].get("workflow_id"),
            "dossier_id": dossier_id,
            "pack_type": pack_type,
            "export_format": export_format,
        },
        rationale="Export preview generated.",
        metadata={"preview_only": True},
        store=store,
    )
    return {
        "platform_version": PLATFORM_FOUNDATION_VERSION,
        "preview": rendered.model_dump(mode="python"),
        "manifest": manifest.model_dump(mode="python"),
        "audit_event": audit,
    }


def render_dossier_export_service(
    dossier_id: str,
    payload: Dict[str, Any],
    *,
    user_context: Optional[Dict[str, Any]] = None,
    store: PlatformStore = platform_store,
) -> Dict[str, Any]:
    context = _resolve_dossier_export_context(
        dossier_id,
        user_context=user_context,
        store=store,
    )
    export_response = create_dossier_export_service(
        dossier_id,
        {
            "pack_type": str(payload.get("pack_type") or "dossier_pack"),
            "metadata": payload.get("metadata") or {},
        },
        user_context=context["normalized"].model_dump(mode="python"),
        store=store,
    )
    manifest = export_response["export"]
    rendered = render_export_manifest(
        manifest,
        export_format=str(payload.get("export_format") or "html"),
        metadata=payload.get("metadata") or {},
    )
    persisted_render = store.create_rendered_export(rendered.model_dump(mode="python"))
    audit = _create_audit_event(
        event_type="export_rendered",
        resource=_dossier_resource(context["dossier"], context["workspace"]),
        user_context=context["normalized"],
        payload={
            "workflow_id": context["dossier"].get("workflow_id"),
            "dossier_id": dossier_id,
            "export_id": manifest.get("export_id"),
            "render_id": persisted_render.get("render_id"),
            "pack_type": manifest.get("pack_type"),
            "export_format": persisted_render.get("export_format"),
        },
        rationale="Rendered export generated.",
        metadata={"checksum": persisted_render.get("checksum")},
        store=store,
    )
    return {
        "platform_version": PLATFORM_FOUNDATION_VERSION,
        "export": manifest,
        "rendered_export": persisted_render,
        "audit_event": audit,
    }


def get_rendered_export_service(
    render_id: str,
    *,
    user_context: Optional[Dict[str, Any]] = None,
    store: PlatformStore = platform_store,
) -> Dict[str, Any]:
    normalized = normalize_user_context(user_context)
    scope = _resolve_scope(
        user_context=normalized.model_dump(mode="python"),
        workspace_id=normalized.workspace_id,
        organization_id=normalized.organization_id,
        store=store,
    )
    rendered = store.get_rendered_export(
        render_id,
        organization_ids=scope["organization_ids"],
        workspace_ids=scope["workspace_ids"],
    )
    if rendered is None:
        raise HTTPException(status_code=404, detail="rendered export not found")
    manifest = store.get_export_manifest(
        str(rendered.get("export_id") or ""),
        organization_ids=scope["organization_ids"],
        workspace_ids=scope["workspace_ids"],
    )
    workspace = (
        store.get_workspace(
            str(manifest.get("workspace_id")),
            organization_ids=scope["organization_ids"],
            workspace_ids=scope["workspace_ids"],
        )
        if manifest and manifest.get("workspace_id")
        else None
    )
    normalized = normalize_user_context(
        normalized,
        organization_id=(workspace or {}).get("organization_id"),
        workspace_id=(workspace or {}).get("workspace_id"),
    )
    require_access(
        permission="view_workspace",
        user_context=normalized,
        resource=_workspace_resource(workspace),
        store=store,
    )
    return {
        "platform_version": PLATFORM_FOUNDATION_VERSION,
        "rendered_export": rendered,
        "export": manifest,
    }


def create_integration_binding_service(
    payload: Dict[str, Any],
    *,
    user_context: Optional[Dict[str, Any]] = None,
    store: PlatformStore = platform_store,
) -> Dict[str, Any]:
    workspace = (
        store.get_workspace(str(payload.get("workspace_id"))) if payload.get("workspace_id") else None
    )
    organization_id = payload.get("organization_id") or (
        workspace.get("organization_id") if workspace else None
    )
    normalized = normalize_user_context(
        user_context,
        organization_id=organization_id,
        workspace_id=(workspace or {}).get("workspace_id"),
    )
    require_access(
        permission="manage_integrations",
        user_context=normalized,
        resource=_workspace_resource(workspace),
        store=store,
    )
    binding = build_integration_binding(
        integration_type=str(payload.get("integration_type") or "custom"),
        organization_id=organization_id,
        workspace_id=(workspace or {}).get("workspace_id"),
        status=str(payload.get("status") or "configured"),
        config=payload.get("config") or {},
        metadata=payload.get("metadata") or {},
    )
    persisted = store.create_integration_binding(binding.model_dump(mode="python"))
    _create_audit_event(
        event_type="integration_binding_created",
        resource=_workspace_resource(workspace),
        user_context=normalized,
        payload={"binding": persisted},
        rationale="Integration binding created.",
        metadata={"workspace_id": (workspace or {}).get("workspace_id")},
        store=store,
    )
    return {
        "platform_version": PLATFORM_FOUNDATION_VERSION,
        "binding": persisted,
        "definition": get_integration_definition(binding.integration_type).model_dump(
            mode="python"
        ),
    }


def list_integrations_service(
    *,
    workspace_id: Optional[str] = None,
    user_context: Optional[Dict[str, Any]] = None,
    store: PlatformStore = platform_store,
) -> Dict[str, Any]:
    scope = _resolve_scope(
        user_context=user_context,
        workspace_id=workspace_id,
        store=store,
    )
    workspace = (
        store.get_workspace(
            workspace_id,
            organization_ids=scope["organization_ids"],
            workspace_ids=scope["workspace_ids"],
        )
        if workspace_id
        else None
    )
    normalized = scope["user_context"]
    require_access(
        permission="view_workspace",
        user_context=normalized,
        resource=_workspace_resource(workspace),
        store=store,
    )
    bindings = (
        store.list_integration_bindings(
            organization_id=(workspace or {}).get("organization_id"),
            workspace_id=workspace_id,
            organization_ids=scope["organization_ids"],
            workspace_ids=scope["workspace_ids"],
        )
        if _scope_allows_listing(scope)
        else []
    )
    enriched_bindings: List[Dict[str, Any]] = []
    for binding in bindings:
        history = store.list_integration_executions(
            str(binding.get("binding_id")),
            organization_ids=scope["organization_ids"],
            workspace_ids=scope["workspace_ids"],
        )
        enriched = sanitize_payload(
            {
                **binding,
                "definition": get_integration_definition(
                    str(binding.get("integration_type") or "custom")
                ).model_dump(mode="python"),
                "execution_count": len(history),
                "last_execution": history[0] if history else None,
            }
        )
        enriched_bindings.append(enriched)
    return {
        "platform_version": PLATFORM_FOUNDATION_VERSION,
        "integration_definitions": [
            item.model_dump(mode="python") for item in list_integration_definitions()
        ],
        "bindings": enriched_bindings,
        "health_summary": integration_health_summary(enriched_bindings),
    }


def list_integration_history_service(
    binding_id: str,
    *,
    user_context: Optional[Dict[str, Any]] = None,
    store: PlatformStore = platform_store,
) -> Dict[str, Any]:
    scope = _resolve_scope(
        user_context=user_context,
        store=store,
    )
    binding = store.get_integration_binding(
        binding_id,
        organization_ids=scope["organization_ids"],
        workspace_ids=scope["workspace_ids"],
    )
    if binding is None:
        raise HTTPException(status_code=404, detail="integration binding not found")
    workspace = (
        store.get_workspace(str(binding.get("workspace_id")))
        if binding.get("workspace_id")
        else None
    )
    normalized = normalize_user_context(
        scope["user_context"],
        organization_id=binding.get("organization_id"),
        workspace_id=(workspace or {}).get("workspace_id"),
    )
    require_access(
        permission="view_workspace",
        user_context=normalized,
        resource=_workspace_resource(workspace),
        store=store,
    )
    history = store.list_integration_executions(
        binding_id,
        organization_ids=scope["organization_ids"],
        workspace_ids=scope["workspace_ids"],
    )
    return {
        "platform_version": PLATFORM_FOUNDATION_VERSION,
        "binding": binding,
        "history": history,
        "definition": get_integration_definition(
            str(binding.get("integration_type") or "custom")
        ).model_dump(mode="python"),
    }


def list_audit_service(
    *,
    workspace_id: Optional[str] = None,
    resource_type: Optional[str] = None,
    resource_id: Optional[str] = None,
    user_context: Optional[Dict[str, Any]] = None,
    store: PlatformStore = platform_store,
) -> Dict[str, Any]:
    scope = _resolve_scope(
        user_context=user_context,
        workspace_id=workspace_id,
        store=store,
    )
    workspace = (
        store.get_workspace(
            workspace_id,
            organization_ids=scope["organization_ids"],
            workspace_ids=scope["workspace_ids"],
        )
        if workspace_id
        else None
    )
    require_access(
        permission="view_audit_log",
        user_context=scope["user_context"],
        resource=_workspace_resource(workspace),
        store=store,
    )
    return {
        "platform_version": PLATFORM_FOUNDATION_VERSION,
        "access_summary": build_access_summary_service(
            workspace_id=workspace_id,
            user_context=scope["user_context"].model_dump(mode="python"),
            store=store,
        )["access_summary"],
        "audit_events": (
            store.list_audit_events(
                workspace_id=workspace_id,
                organization_ids=scope["organization_ids"],
                workspace_ids=scope["workspace_ids"],
                resource_type=resource_type,
                resource_id=resource_id,
            )
            if _scope_allows_listing(scope)
            else []
        ),
    }


def execute_integration_service(
    binding_id: str,
    payload: Dict[str, Any],
    *,
    user_context: Optional[Dict[str, Any]] = None,
    store: PlatformStore = platform_store,
) -> Dict[str, Any]:
    scope = _resolve_scope(
        user_context=user_context,
        store=store,
    )
    binding = store.get_integration_binding(
        binding_id,
        organization_ids=scope["organization_ids"],
        workspace_ids=scope["workspace_ids"],
    )
    if binding is None:
        raise HTTPException(status_code=404, detail="integration binding not found")
    workspace = (
        store.get_workspace(str(binding.get("workspace_id")))
        if binding.get("workspace_id")
        else None
    )
    normalized = normalize_user_context(
        scope["user_context"],
        organization_id=binding.get("organization_id"),
        workspace_id=(workspace or {}).get("workspace_id"),
    )
    require_access(
        permission="manage_integrations",
        user_context=normalized,
        resource=_workspace_resource(workspace),
        store=store,
    )

    dossier_id = payload.get("dossier_id")
    export_id = payload.get("export_id")
    render_id = payload.get("render_id")
    rendered_export = (
        store.get_rendered_export(str(render_id))
        if render_id
        else None
    )
    if rendered_export is None and export_id:
        rendered_exports = store.list_rendered_exports(export_id=str(export_id))
        rendered_export = rendered_exports[0] if rendered_exports else None
    if rendered_export is None and dossier_id and payload.get("action_type") == "sync_export":
        rendered_response = render_dossier_export_service(
            str(dossier_id),
            {
                "pack_type": payload.get("pack_type") or "dossier_pack",
                "export_format": payload.get("export_format") or "html",
                "metadata": {
                    **dict(payload.get("metadata") or {}),
                    "integration_binding_id": binding_id,
                },
            },
            user_context=normalized.model_dump(mode="python"),
            store=store,
        )
        rendered_export = rendered_response.get("rendered_export")
        export_id = (rendered_response.get("export") or {}).get("export_id")
        render_id = (rendered_export or {}).get("render_id")

    try:
        integration_type = str(binding.get("integration_type") or "custom")
        if integration_type == "local_archive":
            if rendered_export is None:
                raise HTTPException(
                    status_code=400,
                    detail="local_archive execution requires a rendered export or dossier context",
                )
            execution = execute_local_archive(
                binding=binding,
                rendered_export=rendered_export,
                dossier_id=dossier_id,
                export_id=export_id,
                metadata=payload.get("metadata") or {},
            )
        elif integration_type == "webhook":
            execution = execute_webhook_outbox(
                binding=binding,
                rendered_export=rendered_export,
                dossier_id=dossier_id,
                export_id=export_id,
                event_type=payload.get("event_type") or "platform_export_event",
                metadata=payload.get("metadata") or {},
            )
        elif integration_type == "internal_sink":
            execution = execute_internal_sink(
                binding=binding,
                dossier_id=dossier_id,
                export_id=export_id,
                render_id=(rendered_export or {}).get("render_id") or render_id,
                event_type=payload.get("event_type") or "platform_sink_event",
                metadata=payload.get("metadata") or {},
            )
        else:
            raise HTTPException(
                status_code=400,
                detail=f"integration type {integration_type} is not executable in Phase 7",
            )
        persisted = store.create_integration_execution(execution)
        binding = store.update_integration_binding(
            binding_id,
            {
                "health": {
                    **dict(binding.get("health") or {}),
                    "status": "healthy" if persisted.get("status") in {"completed", "queued"} else "degraded",
                    "checked_at": _now_utc(),
                    "warnings": []
                    if persisted.get("status") in {"completed", "queued"}
                    else [persisted.get("error_summary") or "integration execution failed"],
                    "details": {
                        **dict((binding.get("health") or {}).get("details") or {}),
                        "last_execution_status": persisted.get("status"),
                        "last_execution_at": persisted.get("completed_at"),
                        "last_execution_id": persisted.get("execution_id"),
                    },
                },
                "updated_at": _now_utc(),
            },
        )
        audit = _create_audit_event(
            event_type="integration_executed",
            resource=_workspace_resource(workspace),
            user_context=normalized,
            payload={
                "binding_id": binding_id,
                "dossier_id": dossier_id,
                "export_id": export_id,
                "render_id": render_id or (rendered_export or {}).get("render_id"),
                "status": persisted.get("status"),
            },
            rationale="Integration execution completed.",
            metadata={"execution_id": persisted.get("execution_id")},
            store=store,
        )
        return {
            "platform_version": PLATFORM_FOUNDATION_VERSION,
            "binding": binding,
            "execution": persisted,
            "rendered_export": rendered_export,
            "audit_event": audit,
        }
    except HTTPException:
        raise
    except Exception as exc:
        failed_execution = store.create_integration_execution(
            {
                "execution_id": str(uuid.uuid4()),
                "binding_id": binding_id,
                "integration_type": binding.get("integration_type"),
                "action_type": payload.get("action_type") or "sync_export",
                "status": "failed",
                "workspace_id": binding.get("workspace_id"),
                "organization_id": binding.get("organization_id"),
                "dossier_id": dossier_id,
                "export_id": export_id,
                "render_id": render_id or (rendered_export or {}).get("render_id"),
                "started_at": _now_utc(),
                "completed_at": _now_utc(),
                "payload_summary": sanitize_payload(payload),
                "output_summary": {},
                "error_summary": str(exc),
                "metadata": payload.get("metadata") or {},
            }
        )
        store.update_integration_binding(
            binding_id,
            {
                "health": {
                    **dict(binding.get("health") or {}),
                    "status": "degraded",
                    "checked_at": _now_utc(),
                    "warnings": [str(exc)],
                    "details": {
                        **dict((binding.get("health") or {}).get("details") or {}),
                        "last_execution_status": "failed",
                        "last_execution_at": failed_execution.get("completed_at"),
                        "last_execution_id": failed_execution.get("execution_id"),
                    },
                },
                "updated_at": _now_utc(),
            },
        )
        audit = _create_audit_event(
            event_type="integration_failed",
            resource=_workspace_resource(workspace),
            user_context=normalized,
            payload={
                "binding_id": binding_id,
                "dossier_id": dossier_id,
                "export_id": export_id,
                "error_summary": str(exc),
            },
            rationale="Integration execution failed.",
            metadata={"execution_id": failed_execution.get("execution_id")},
            store=store,
        )
        raise HTTPException(
            status_code=500,
            detail=f"integration execution failed: {exc}",
        ) from exc


def build_platform_health_service(
    *,
    workspace_id: Optional[str] = None,
    user_context: Optional[Dict[str, Any]] = None,
    store: PlatformStore = platform_store,
) -> Dict[str, Any]:
    scope = _resolve_scope(
        user_context=user_context,
        workspace_id=workspace_id,
        store=store,
    )
    workspace = (
        store.get_workspace(
            workspace_id,
            organization_ids=scope["organization_ids"],
            workspace_ids=scope["workspace_ids"],
        )
        if workspace_id
        else None
    )
    normalized = scope["user_context"]
    access_summary = build_access_summary(
        user_context=normalized,
        resource=_workspace_resource(workspace),
        store=store,
    )
    scope_bundle = _workspace_scope_bundle(
        workspace_id=workspace_id,
        organization_ids=scope["organization_ids"],
        workspace_ids=scope["workspace_ids"],
        store=store,
    )
    workflows = scope_bundle["workflows"]
    dossiers = scope_bundle["dossiers"]
    approvals = scope_bundle["approvals"]
    exports = scope_bundle["exports"]
    audit_events = (
        store.list_audit_events(
            organization_ids=scope["organization_ids"],
            workspace_id=(workspace or {}).get("workspace_id"),
            workspace_ids=scope["workspace_ids"],
            limit=400,
        )
        if _scope_allows_listing(scope)
        else []
    )
    integration_bindings = scope_bundle["integration_bindings"]
    rendered_exports = scope_bundle["rendered_exports"]
    integration_execution_count = sum(
        len(
            store.list_integration_executions(
                str(item.get("binding_id")),
                organization_ids=scope["organization_ids"],
                workspace_ids=scope["workspace_ids"],
            )
        )
        for item in integration_bindings
    )
    health = build_platform_health_summary(
        platform_version=PLATFORM_FOUNDATION_VERSION,
        workspace_id=(workspace or {}).get("workspace_id"),
        organization_id=(workspace or {}).get("organization_id"),
        access_summary=access_summary,
        workflows=workflows,
        dossiers=dossiers,
        approvals=approvals,
        exports=exports,
        audit_events=audit_events,
        integration_bindings=integration_bindings,
    )
    health["rendered_export_count"] = len(rendered_exports)
    health["integration_execution_count"] = integration_execution_count
    return {
        "platform_version": PLATFORM_FOUNDATION_VERSION,
        "health": health,
    }


def build_workspace_analytics_service(
    workspace_id: str,
    *,
    user_context: Optional[Dict[str, Any]] = None,
    store: PlatformStore = platform_store,
) -> Dict[str, Any]:
    scope = _resolve_scope(
        user_context=user_context,
        workspace_id=workspace_id,
        store=store,
    )
    workspace = store.get_workspace(
        workspace_id,
        organization_ids=scope["organization_ids"],
        workspace_ids=scope["workspace_ids"],
    )
    if workspace is None:
        raise HTTPException(status_code=404, detail="workspace not found")
    normalized = scope["user_context"]
    require_access(
        permission="view_workspace",
        user_context=normalized,
        resource=_workspace_resource(workspace),
        store=store,
    )
    scope_bundle = _workspace_scope_bundle(
        workspace_id=workspace_id,
        organization_ids=scope["organization_ids"],
        workspace_ids=scope["workspace_ids"],
        store=store,
    )
    analytics = build_workspace_analytics_view(
        workspace=workspace,
        workflows=scope_bundle["workflows"],
        dossiers=scope_bundle["dossiers"],
        approvals=scope_bundle["approvals"],
        exports=scope_bundle["exports"],
        integration_bindings=scope_bundle["integration_bindings"],
    ).model_dump(mode="python")
    return {
        "platform_version": PLATFORM_FOUNDATION_VERSION,
        "workspace": workspace,
        "analytics": analytics,
    }


def build_platform_analytics_service(
    *,
    workspace_id: Optional[str] = None,
    user_context: Optional[Dict[str, Any]] = None,
    store: PlatformStore = platform_store,
) -> Dict[str, Any]:
    scope = _resolve_scope(
        user_context=user_context,
        workspace_id=workspace_id,
        store=store,
    )
    workspace = (
        store.get_workspace(
            workspace_id,
            organization_ids=scope["organization_ids"],
            workspace_ids=scope["workspace_ids"],
        )
        if workspace_id
        else None
    )
    normalized = scope["user_context"]
    require_access(
        permission="view_workspace",
        user_context=normalized,
        resource=_workspace_resource(workspace),
        store=store,
    )
    scope_bundle = _workspace_scope_bundle(
        workspace_id=workspace_id,
        organization_ids=scope["organization_ids"],
        workspace_ids=scope["workspace_ids"],
        store=store,
    )
    workspaces = (
        scope_bundle["workspaces"]
        if workspace_id
        else (
            store.list_workspaces(
                organization_ids=scope["organization_ids"],
                workspace_ids=scope["workspace_ids"],
            )
            if _scope_allows_listing(scope)
            else []
        )
    )
    analytics = build_cross_workspace_analytics(
        platform_version=PLATFORM_FOUNDATION_VERSION,
        workspaces=workspaces,
        workflows=scope_bundle["workflows"],
        dossiers=scope_bundle["dossiers"],
        approvals=scope_bundle["approvals"],
        exports=scope_bundle["exports"],
        integration_bindings=scope_bundle["integration_bindings"],
    )
    return {
        "platform_version": PLATFORM_FOUNDATION_VERSION,
        "analytics": analytics,
    }


def build_platform_dashboard_service(
    *,
    workspace_id: Optional[str] = None,
    user_context: Optional[Dict[str, Any]] = None,
    store: PlatformStore = platform_store,
    emit_audit: bool = False,
) -> Dict[str, Any]:
    scope = _resolve_scope(
        user_context=user_context,
        workspace_id=workspace_id,
        store=store,
    )
    workspace = (
        store.get_workspace(
            workspace_id,
            organization_ids=scope["organization_ids"],
            workspace_ids=scope["workspace_ids"],
        )
        if workspace_id
        else None
    )
    normalized = scope["user_context"]
    require_access(
        permission="view_workspace",
        user_context=normalized,
        resource=_workspace_resource(workspace),
        store=store,
    )
    summary = build_platform_summary_service(
        workspace_id=workspace_id,
        user_context=normalized.model_dump(mode="python"),
        store=store,
    )
    workspace_analytics = (
        build_workspace_analytics_service(
            workspace["workspace_id"],
            user_context=normalized.model_dump(mode="python"),
            store=store,
        )["analytics"]
        if workspace
        else {
            "workspace_name": None,
            "workflow_count": 0,
            "dossier_count": 0,
            "pending_approval_count": 0,
            "export_count": 0,
            "integration_binding_count": 0,
            "high_dau_dossiers": [],
            "recent_exports": [],
            "recent_approvals": [],
            "dossier_records": [],
        }
    )
    cross_workspace = build_platform_analytics_service(
        workspace_id=workspace_id,
        user_context=normalized.model_dump(mode="python"),
        store=store,
    )["analytics"]
    approvals: List[Dict[str, Any]] = []
    if workspace:
        for wf in store.list_workflows(
            workspace_id=workspace["workspace_id"],
            organization_ids=scope["organization_ids"],
            workspace_ids=scope["workspace_ids"],
        ):
            approvals.extend(
                store.list_approval_requests(workflow_id=wf.get("workflow_id"))
            )
    timeline = (
        store.list_audit_events(
            organization_ids=scope["organization_ids"],
            workspace_id=(workspace or {}).get("workspace_id"),
            workspace_ids=scope["workspace_ids"],
            limit=40,
        )
        if _scope_allows_listing(scope)
        else []
    )
    exports: List[Dict[str, Any]] = []
    if workspace:
        for dossier in store.list_dossiers(
            workspace_id=workspace["workspace_id"],
            organization_ids=scope["organization_ids"],
            workspace_ids=scope["workspace_ids"],
        ):
            exports.extend(
                store.list_export_manifests(
                    dossier.get("dossier_id"),
                    organization_ids=scope["organization_ids"],
                    workspace_ids=scope["workspace_ids"],
                )
            )
    integrations = list_integrations_service(
        workspace_id=(workspace or {}).get("workspace_id"),
        user_context=normalized.model_dump(mode="python"),
        store=store,
    )
    health = build_platform_health_service(
        workspace_id=(workspace or {}).get("workspace_id"),
        user_context=normalized.model_dump(mode="python"),
        store=store,
    )["health"]
    dashboard = build_dashboard_payload(
        workspace=workspace,
        summary_view=summary,
        workspace_analytics=workspace_analytics,
        cross_workspace_analytics=cross_workspace,
        approvals=approvals,
        timeline=timeline,
        exports=exports,
        integration_summary=integrations["health_summary"],
        health_summary=health,
    )
    if emit_audit:
        _create_audit_event(
            event_type="dashboard_view_generated",
            resource=_workspace_resource(workspace),
            user_context=normalized,
            payload={"workspace_id": (workspace or {}).get("workspace_id")},
            rationale="Platform dashboard generated.",
            metadata={"dashboard_scope": "workspace" if workspace else "global"},
            store=store,
        )
    return {
        "platform_version": PLATFORM_FOUNDATION_VERSION,
        "dashboard": dashboard,
    }


def build_demo_snapshot_service(
    *,
    workspace_id: Optional[str] = None,
    user_context: Optional[Dict[str, Any]] = None,
    store: PlatformStore = platform_store,
    emit_audit: bool = False,
) -> Dict[str, Any]:
    scope = _resolve_scope(
        user_context=user_context,
        workspace_id=workspace_id,
        store=store,
    )
    workspace = (
        store.get_workspace(
            workspace_id,
            organization_ids=scope["organization_ids"],
            workspace_ids=scope["workspace_ids"],
        )
        if workspace_id
        else None
    )
    normalized = scope["user_context"]
    require_access(
        permission="view_workspace",
        user_context=normalized,
        resource=_workspace_resource(workspace),
        store=store,
    )
    workspace_analytics = (
        build_workspace_analytics_service(
            workspace["workspace_id"],
            user_context=normalized.model_dump(mode="python"),
            store=store,
        )["analytics"]
        if workspace
        else {}
    )
    approvals: List[Dict[str, Any]] = []
    exports: List[Dict[str, Any]] = []
    if workspace:
        for wf in store.list_workflows(
            workspace_id=workspace["workspace_id"],
            organization_ids=scope["organization_ids"],
            workspace_ids=scope["workspace_ids"],
        ):
            approvals.extend(
                store.list_approval_requests(workflow_id=wf.get("workflow_id"))
            )
        for dossier in store.list_dossiers(
            workspace_id=workspace["workspace_id"],
            organization_ids=scope["organization_ids"],
            workspace_ids=scope["workspace_ids"],
        ):
            exports.extend(
                store.list_export_manifests(
                    dossier.get("dossier_id"),
                    organization_ids=scope["organization_ids"],
                    workspace_ids=scope["workspace_ids"],
                )
            )
    integrations = list_integrations_service(
        workspace_id=(workspace or {}).get("workspace_id"),
        user_context=normalized.model_dump(mode="python"),
        store=store,
    )
    health = build_platform_health_service(
        workspace_id=(workspace or {}).get("workspace_id"),
        user_context=normalized.model_dump(mode="python"),
        store=store,
    )["health"]
    snapshot = build_demo_workspace_snapshot(
        platform_version=PLATFORM_FOUNDATION_VERSION,
        workspace=workspace,
        workspace_analytics=workspace_analytics,
        approvals=approvals,
        exports=exports,
        integration_summary=integrations["health_summary"],
        health_summary=health,
    )
    if emit_audit:
        _create_audit_event(
            event_type="demo_snapshot_generated",
            resource=_workspace_resource(workspace),
            user_context=normalized,
            payload={"workspace_id": (workspace or {}).get("workspace_id")},
            rationale="Demo workspace snapshot generated.",
            metadata={"snapshot_type": "workspace"},
            store=store,
        )
    return {
        "platform_version": PLATFORM_FOUNDATION_VERSION,
        "snapshot": snapshot,
    }


def build_demo_readiness_service(
    *,
    workspace_id: Optional[str] = None,
    user_context: Optional[Dict[str, Any]] = None,
    store: PlatformStore = platform_store,
) -> Dict[str, Any]:
    scope = _resolve_scope(
        user_context=user_context,
        workspace_id=workspace_id,
        store=store,
    )
    workspace = (
        store.get_workspace(
            workspace_id,
            organization_ids=scope["organization_ids"],
            workspace_ids=scope["workspace_ids"],
        )
        if workspace_id
        else None
    )
    normalized = scope["user_context"]
    require_access(
        permission="view_workspace",
        user_context=normalized,
        resource=_workspace_resource(workspace),
        store=store,
    )
    workspace_analytics = (
        build_workspace_analytics_service(
            workspace["workspace_id"],
            user_context=normalized.model_dump(mode="python"),
            store=store,
        )["analytics"]
        if workspace
        else {}
    )
    health = build_platform_health_service(
        workspace_id=(workspace or {}).get("workspace_id"),
        user_context=normalized.model_dump(mode="python"),
        store=store,
    )["health"]
    integrations = list_integrations_service(
        workspace_id=(workspace or {}).get("workspace_id"),
        user_context=normalized.model_dump(mode="python"),
        store=store,
    )
    readiness = build_readiness_snapshot(
        platform_version=PLATFORM_FOUNDATION_VERSION,
        workspace=workspace,
        workspace_analytics=workspace_analytics,
        health_summary=health,
        integration_summary=integrations["health_summary"],
    )
    return {
        "platform_version": PLATFORM_FOUNDATION_VERSION,
        "readiness": readiness,
    }


def sync_analysis_into_platform(
    *,
    request: Dict[str, Any],
    report: Dict[str, Any],
    report_id: Optional[str],
    session_id: Optional[str],
    axiom_artifact_id: Optional[str],
    axiom_report_pack_artifact_id: Optional[str],
    axiom_lineage_artifact_id: Optional[str],
    axiom_history_artifact_id: Optional[str],
    axiom_calibration_artifact_id: Optional[str],
    store: PlatformStore = platform_store,
) -> Optional[Dict[str, Any]]:
    wants_platform = bool(
        request.get("create_dossier")
        or request.get("dossier_id")
        or request.get("workspace_id")
        or request.get("workflow_id")
        or request.get("workflow_template_id")
        or request.get("platform_profile")
    )
    if not wants_platform:
        return None

    audience_type = str(request.get("audience_type") or "general")
    report_profile = str(request.get("report_profile") or "trading_focused")
    platform_profile = request.get("platform_profile")
    platform_user_context = request.get("platform_user_context")

    workspace = None
    organization = None
    if request.get("workspace_id"):
        workspace = store.get_workspace(str(request.get("workspace_id")))
        if workspace is None:
            raise HTTPException(status_code=404, detail="workspace not found")
        organization = store.get_organization(workspace["organization_id"])
    else:
        foundation = ensure_default_foundation(
            audience_type=audience_type,
            report_profile=report_profile,
            platform_profile=platform_profile,
            store=store,
        )
        workspace = foundation["workspace"]
        organization = foundation["organization"]
        platform_profile = foundation["platform_profile"]["profile_id"]

    workflow = None
    if request.get("workflow_id"):
        workflow = store.get_workflow(str(request.get("workflow_id")))
        if workflow is None:
            raise HTTPException(status_code=404, detail="workflow not found")
    elif request.get("create_dossier") or request.get("workflow_template_id"):
        workflow_response = create_workflow_service(
            {
                "workspace_id": workspace["workspace_id"],
                "workflow_template_id": request.get("workflow_template_id")
                or workspace.get("default_workflow_template"),
                "title": f"{report.get('symbol')} {request.get('workflow_template_id') or workspace.get('default_workflow_template')}",
                "priority": "high"
                if report.get("axiom_evidence_backed_deployability_tier")
                == "live_candidate"
                else "normal",
            },
            user_context=platform_user_context,
            store=store,
        )
        workflow = workflow_response["workflow"]

    dossier_view = None
    dossier = None
    if request.get("dossier_id"):
        dossier_view = attach_analysis_to_dossier_service(
            str(request.get("dossier_id")),
            {
                "report": report,
                "report_id": report_id,
                "session_id": session_id,
                "axiom_artifact_id": axiom_artifact_id,
                "axiom_report_pack_artifact_id": axiom_report_pack_artifact_id,
                "axiom_lineage_artifact_id": axiom_lineage_artifact_id,
                "axiom_history_artifact_id": axiom_history_artifact_id,
                "axiom_calibration_artifact_id": axiom_calibration_artifact_id,
            },
            user_context=platform_user_context,
            store=store,
        )
        dossier = dossier_view["dossier"]
        workflow = dossier_view["workflow"] or workflow
        workspace = dossier_view["workspace"] or workspace
    elif request.get("create_dossier"):
        if workflow is None:
            workflow_response = create_workflow_service(
                {
                    "workspace_id": workspace["workspace_id"],
                    "workflow_template_id": workspace.get("default_workflow_template"),
                    "title": f"{report.get('symbol')} {workspace.get('default_workflow_template')}",
                },
                user_context=platform_user_context,
                store=store,
            )
            workflow = workflow_response["workflow"]
        create_response = create_dossier_service(
            {
                "workflow_id": workflow["workflow_id"],
                "symbol": report.get("symbol"),
                "display_name": report.get("symbol"),
                "entity_type": "public_equity",
                "sector": (((report.get("data_bundle") or {}).get("symbol_meta") or {}).get("sector")),
                "theme": report.get("axiom_trade_family"),
                "dossier_type": request.get("dossier_type") or "coverage",
                "title": f"{report.get('symbol')} Institutional Dossier",
                "metadata": {"created_from_analysis": True},
            },
            user_context=platform_user_context,
            store=store,
        )
        dossier = create_response["dossier"]
        dossier_view = attach_analysis_to_dossier_service(
            dossier["dossier_id"],
            {
                "report": report,
                "report_id": report_id,
                "session_id": session_id,
                "axiom_artifact_id": axiom_artifact_id,
                "axiom_report_pack_artifact_id": axiom_report_pack_artifact_id,
                "axiom_lineage_artifact_id": axiom_lineage_artifact_id,
                "axiom_history_artifact_id": axiom_history_artifact_id,
                "axiom_calibration_artifact_id": axiom_calibration_artifact_id,
            },
            user_context=platform_user_context,
            store=store,
        )
        dossier = dossier_view["dossier"]

    summary = build_platform_summary_service(
        workspace_id=workspace["workspace_id"] if workspace else None,
        user_context=platform_user_context,
        store=store,
        current_workspace=workspace,
        current_workflow=workflow,
        current_dossier=dossier,
    )
    template = (
        get_workflow_template(str(workflow.get("workflow_template_id"))) if workflow else None
    )
    profile = get_platform_profile(
        platform_profile or workspace.get("platform_profile") if workspace else None
    )
    access_summary = build_access_summary(
        user_context=platform_user_context,
        resource=_workspace_resource(workspace),
        store=store,
    )
    approvals = (
        store.list_approval_requests(
            workflow_id=(workflow or {}).get("workflow_id"),
            dossier_id=(dossier or {}).get("dossier_id"),
        )
        if workflow
        else []
    )
    allowed_actions = (
        list_allowed_actions(
            workflow=workflow or {},
            dossier=dossier,
            template=template or get_workflow_template("research_watchlist"),
            effective_permissions=access_summary.get("effective_permissions") or [],
            approvals=approvals,
        )
        if workflow
        else []
    )
    timeline = (
        list_workflow_timeline_service(
            workflow.get("workflow_id"),
            user_context=platform_user_context,
            store=store,
        ).get("timeline")
        if workflow
        else []
    )
    exports = (
        store.list_export_manifests((dossier or {}).get("dossier_id"))
        if dossier
        else []
    )
    rendered_exports = [
        rendered
        for export in exports
        for rendered in store.list_rendered_exports(export_id=export.get("export_id"))
    ]
    integration_payload = list_integrations_service(
        workspace_id=(workspace or {}).get("workspace_id"),
        user_context=platform_user_context,
        store=store,
    )
    integrations = integration_payload.get("bindings") or []
    health = build_platform_health_service(
        workspace_id=(workspace or {}).get("workspace_id"),
        user_context=platform_user_context,
        store=store,
    ).get("health")
    workspace_analytics = (
        build_workspace_analytics_service(
            workspace["workspace_id"],
            user_context=platform_user_context,
            store=store,
        ).get("analytics")
        if workspace
        else {}
    )
    platform_analytics = build_platform_analytics_service(
        workspace_id=None,
        user_context=platform_user_context,
        store=store,
    ).get("analytics")
    dashboard = build_platform_dashboard_service(
        workspace_id=(workspace or {}).get("workspace_id"),
        user_context=platform_user_context,
        store=store,
        emit_audit=False,
    ).get("dashboard")
    demo_snapshot = build_demo_snapshot_service(
        workspace_id=(workspace or {}).get("workspace_id"),
        user_context=platform_user_context,
        store=store,
        emit_audit=False,
    ).get("snapshot")
    readiness_snapshot = build_demo_readiness_service(
        workspace_id=(workspace or {}).get("workspace_id"),
        user_context=platform_user_context,
        store=store,
    ).get("readiness")
    return {
        "platform_foundation_version": PLATFORM_FOUNDATION_VERSION,
        "platform_profile": profile.model_dump(mode="python"),
        "organization": organization,
        "workspace": workspace,
        "workflow": workflow,
        "workflow_template": template.model_dump(mode="python") if template else None,
        "dossier": dossier,
        "analysis_link": dossier_view["analysis_link"] if dossier_view else None,
        "summary_view": summary,
        "access_summary": access_summary,
        "allowed_actions": allowed_actions,
        "approvals": approvals,
        "timeline": timeline,
        "exports": exports,
        "rendered_exports": rendered_exports,
        "integration_summary": integration_payload.get("health_summary") or integration_health_summary(integrations),
        "integration_bindings": integrations,
        "health_summary": health,
        "workspace_analytics": workspace_analytics,
        "platform_analytics": platform_analytics,
        "dashboard": dashboard,
        "demo_snapshot": demo_snapshot,
        "readiness_snapshot": readiness_snapshot,
        "supported_export_packs": supported_pack_types(),
    }


def build_platform_summary_service(
    *,
    workspace_id: Optional[str] = None,
    user_context: Optional[Dict[str, Any]] = None,
    store: PlatformStore = platform_store,
    current_workspace: Optional[Dict[str, Any]] = None,
    current_workflow: Optional[Dict[str, Any]] = None,
    current_dossier: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    scope = _resolve_scope(
        user_context=user_context,
        workspace_id=workspace_id or (current_workspace or {}).get("workspace_id"),
        organization_id=(current_workspace or {}).get("organization_id"),
        store=store,
    )
    workspace = current_workspace or (
        store.get_workspace(
            workspace_id,
            organization_ids=scope["organization_ids"],
            workspace_ids=scope["workspace_ids"],
        )
        if workspace_id
        else None
    )
    normalized = scope["user_context"]
    if _scope_allows_listing(scope):
        workspaces = store.list_workspaces(
            organization_ids=scope["organization_ids"],
            workspace_ids=scope["workspace_ids"],
        )
        workflows = store.list_workflows(
            workspace_id=workspace_id,
            organization_ids=scope["organization_ids"],
            workspace_ids=scope["workspace_ids"],
        )
        dossiers = store.list_dossiers(
            workspace_id=workspace_id,
            organization_ids=scope["organization_ids"],
            workspace_ids=scope["workspace_ids"],
        )
    else:
        workspaces = []
        workflows = []
        dossiers = []
    approvals: List[Dict[str, Any]] = []
    export_count = 0
    audit_count = (
        len(
            store.list_audit_events(
                workspace_id=(workspace or {}).get("workspace_id"),
                organization_ids=scope["organization_ids"],
                workspace_ids=scope["workspace_ids"],
                limit=400,
            )
        )
        if _scope_allows_listing(scope)
        else 0
    )
    by_tier: Dict[str, int] = {}
    by_regime: Dict[str, int] = {}
    by_stage: Dict[str, int] = {}
    latest: List[Dict[str, Any]] = []
    for workflow in workflows:
        approvals.extend(store.list_approval_requests(workflow_id=workflow["workflow_id"]))
    for dossier in dossiers[:12]:
        tier = str(dossier.get("latest_deployability_tier") or "unknown")
        regime = str(dossier.get("latest_regime_label") or "unknown")
        stage = str((dossier.get("workflow_stage_state") or {}).get("stage") or "unknown")
        by_tier[tier] = by_tier.get(tier, 0) + 1
        by_regime[regime] = by_regime.get(regime, 0) + 1
        by_stage[stage] = by_stage.get(stage, 0) + 1
        latest.append(dossier_preview(dossier))
        export_count += len(
            store.list_export_manifests(
                dossier["dossier_id"],
                organization_ids=scope["organization_ids"],
                workspace_ids=scope["workspace_ids"],
            )
        )
    summary = PlatformSummaryView(
        platform_version=PLATFORM_FOUNDATION_VERSION,
        workspace_count=len(workspaces),
        dossier_count=len(dossiers),
        workflow_count=len(workflows),
        latest_axiom_linked_dossiers=latest,
        dossiers_by_deployability_tier=by_tier,
        dossiers_by_regime=by_regime,
        dossiers_by_workflow_stage=by_stage,
        current_workspace=sanitize_payload(current_workspace or {}),
        current_workflow=sanitize_payload(current_workflow or {}),
        current_dossier=sanitize_payload(current_dossier or {}),
    ).model_dump(mode="python")
    summary["pending_approval_count"] = sum(
        1 for item in approvals if str(item.get("status")) == "pending"
    )
    summary["export_count"] = export_count
    summary["audit_event_count"] = audit_count
    summary["access_summary"] = build_access_summary(
        user_context=normalized,
        resource=_workspace_resource(workspace),
        store=store,
    )
    return summary


def list_templates_service() -> Dict[str, Any]:
    return {
        "platform_version": PLATFORM_FOUNDATION_VERSION,
        "workflow_templates": [
            template.model_dump(mode="python") for template in list_workflow_templates()
        ],
        "platform_profiles": [
            profile.model_dump(mode="python") for profile in list_platform_profiles()
        ],
        "integration_definitions": [
            definition.model_dump(mode="python")
            for definition in list_integration_definitions()
        ],
        "supported_export_packs": supported_pack_types(),
    }
