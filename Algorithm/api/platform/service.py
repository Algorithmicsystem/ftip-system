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
from api.platform.comments import build_review_comment, resolve_review_comment
from api.platform.committee import build_committee_decision_snapshot
from api.platform.contracts import (
    DossierRecord,
    ExportRetrievalResult,
    PlatformSummaryView,
    ResourceRef,
    StoredExportRecord,
)
from api.platform.dashboard import build_dashboard_payload
from api.platform.demo import build_demo_workspace_snapshot, build_readiness_snapshot
from api.platform.dossiers import (
    build_analysis_link,
    dossier_preview,
    refresh_dossier_record,
)
from api.platform.entities import build_coverage_entity
from api.platform.export_integrity import build_export_integrity_result
from api.platform.export_layouts import (
    build_export_layout_metadata,
    export_format_capabilities,
)
from api.platform.execution import (
    execute_internal_sink,
    execute_local_archive,
    execute_webhook_outbox,
)
from api.platform.export_renderers import render_export_manifest
from api.platform.export_storage import (
    default_storage_backend_name,
    retrieve_export_content,
    store_export_content,
)
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
from api.platform.recommendations import (
    build_escalation_record,
    build_recommendation_change_record,
    build_recommendation_lock_record,
    build_recommendation_state,
    default_recommendation_state,
)
from api.platform.reviews import build_review_summary, build_role_assignment_summary
from api.platform.security import normalize_user_context
from api.platform.tenant import build_tenancy_summary
from api.platform.templates import get_workflow_template, list_workflow_templates
from api.platform.workflows import build_workflow_instance


PLATFORM_FOUNDATION_VERSION = "platform_phase8c_collaboration_v1"


def _now_utc() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _normalize_symbol(symbol: Optional[str]) -> str:
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


def _default_recommendation_state_for_dossier(dossier: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    base = default_recommendation_state()
    if dossier:
        locked = bool((dossier.get("metadata") or {}).get("recommendation_locked"))
        if locked:
            base["locked"] = True
    return sanitize_payload(base)


def _current_recommendation_state(dossier: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    metadata = (dossier or {}).get("metadata") or {}
    recommendation_state = metadata.get("recommendation_state") or {}
    if recommendation_state:
        return sanitize_payload(recommendation_state)
    return _default_recommendation_state_for_dossier(dossier)


def _upsert_section(
    sections: List[Dict[str, Any]],
    *,
    section_key: str,
    title: str,
    summary: str,
    payload: Optional[Dict[str, Any]] = None,
    status: str = "available",
) -> List[Dict[str, Any]]:
    updated = [item for item in sections if str(item.get("section_key") or "") != section_key]
    updated.append(
        sanitize_payload(
            {
                "section_key": section_key,
                "title": title,
                "summary": summary or "No current summary available.",
                "payload": payload or {},
                "status": status,
            }
        )
    )
    return updated


def _recommendation_history_summary(history: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return sanitize_payload(
        [
            {
                "change_id": item.get("change_id"),
                "action_type": item.get("action_type"),
                "previous_state": item.get("previous_state"),
                "new_state": item.get("new_state"),
                "locked": item.get("locked"),
                "rationale": item.get("rationale"),
                "created_at": item.get("created_at"),
            }
            for item in history[:8]
        ]
    )


def _build_collaboration_bundle(
    *,
    workflow: Optional[Dict[str, Any]],
    dossier: Optional[Dict[str, Any]],
    store: PlatformStore,
    organization_ids: Optional[List[str]] = None,
    workspace_ids: Optional[List[str]] = None,
) -> Dict[str, Any]:
    if not workflow or not dossier:
        return {
            "comments": [],
            "review_summary": {},
            "assignments": [],
            "assignment_summary": {},
            "committee_decision": {},
            "recommendation_state": _default_recommendation_state_for_dossier(dossier),
            "recommendation_history": [],
        }
    comments = store.list_review_comments(
        workflow_id=workflow.get("workflow_id"),
        dossier_id=dossier.get("dossier_id"),
        organization_ids=organization_ids,
        workspace_ids=workspace_ids,
    )
    assignments = store.list_assignments(
        workflow_id=workflow.get("workflow_id"),
        dossier_id=dossier.get("dossier_id"),
        organization_ids=organization_ids,
        workspace_ids=workspace_ids,
    )
    committee_decision = (
        store.get_latest_committee_decision(
            workflow_id=workflow.get("workflow_id"),
            dossier_id=dossier.get("dossier_id"),
            organization_ids=organization_ids,
            workspace_ids=workspace_ids,
        )
        or {}
    )
    recommendation_history = store.list_recommendation_changes(
        workflow_id=workflow.get("workflow_id"),
        dossier_id=dossier.get("dossier_id"),
        organization_ids=organization_ids,
        workspace_ids=workspace_ids,
    )
    recommendation_state = (
        (recommendation_history[0] or {}).get("snapshot")
        if recommendation_history
        else _current_recommendation_state(dossier)
    ) or _default_recommendation_state_for_dossier(dossier)
    review_summary = build_review_summary(
        workflow_id=str(workflow.get("workflow_id") or ""),
        dossier_id=str(dossier.get("dossier_id") or ""),
        comments=comments,
        committee_decision=committee_decision or None,
    )
    assignment_summary = build_role_assignment_summary(
        workflow_id=str(workflow.get("workflow_id") or ""),
        dossier_id=str(dossier.get("dossier_id") or ""),
        assignments=assignments,
    )
    return {
        "comments": comments,
        "review_summary": review_summary,
        "assignments": assignments,
        "assignment_summary": assignment_summary,
        "committee_decision": committee_decision,
        "recommendation_state": sanitize_payload(recommendation_state),
        "recommendation_history": recommendation_history,
    }


def _refresh_dossier_collaboration_state(
    dossier: Optional[Dict[str, Any]],
    *,
    workflow: Optional[Dict[str, Any]],
    store: PlatformStore,
    organization_ids: Optional[List[str]] = None,
    workspace_ids: Optional[List[str]] = None,
) -> Optional[Dict[str, Any]]:
    if not dossier or not workflow:
        return dossier
    collaboration = _build_collaboration_bundle(
        workflow=workflow,
        dossier=dossier,
        store=store,
        organization_ids=organization_ids,
        workspace_ids=workspace_ids,
    )
    review_summary = collaboration["review_summary"]
    assignment_summary = collaboration["assignment_summary"]
    committee_decision = collaboration["committee_decision"]
    recommendation_state = collaboration["recommendation_state"]
    recommendation_history = collaboration["recommendation_history"]

    current_summary = sanitize_payload(dossier.get("current_summary") or {})
    current_summary.update(
        {
            "recommendation_state": recommendation_state.get("state"),
            "recommendation_locked": bool(recommendation_state.get("locked")),
            "unresolved_concern_count": int(
                review_summary.get("unresolved_concern_count") or 0
            ),
            "latest_committee_decision_status": committee_decision.get("decision_status"),
            "review_comment_count": int(
                ((review_summary.get("thread_summary") or {}).get("total_comments") or 0)
            ),
        }
    )
    metadata = sanitize_payload(dossier.get("metadata") or {})
    metadata.update(
        {
            "review_summary": review_summary,
            "review_comments": collaboration["comments"],
            "assignments": collaboration["assignments"],
            "assignment_summary": assignment_summary,
            "latest_committee_decision": committee_decision,
            "recommendation_state": recommendation_state,
            "recommendation_history_summary": _recommendation_history_summary(
                recommendation_history
            ),
            "unresolved_concern_count": int(
                review_summary.get("unresolved_concern_count") or 0
            ),
        }
    )
    sections = [sanitize_payload(item) for item in list(dossier.get("sections") or [])]
    sections = _upsert_section(
        sections,
        section_key="review_summary",
        title="Review Summary",
        summary=(
            f"{review_summary.get('unresolved_concern_count', 0)} unresolved concern(s) across "
            f"{((review_summary.get('thread_summary') or {}).get('total_comments') or 0)} review comment(s)."
        ),
        payload=review_summary,
    )
    sections = _upsert_section(
        sections,
        section_key="assignments",
        title="Assignments",
        summary=(
            f"Assignment coverage includes owner "
            f"{(((assignment_summary.get('owner') or {}).get('assignee_placeholder')) or 'unassigned')} "
            f"and committee reviewer "
            f"{(((assignment_summary.get('committee_reviewer') or {}).get('assignee_placeholder')) or 'unassigned')}."
        ),
        payload=assignment_summary,
        status="available" if assignment_summary else "partial",
    )
    sections = _upsert_section(
        sections,
        section_key="committee_decision",
        title="Committee Decision",
        summary=(
            committee_decision.get("summary")
            or "No committee decision snapshot is currently recorded."
        ),
        payload=committee_decision,
        status="available" if committee_decision else "partial",
    )
    sections = _upsert_section(
        sections,
        section_key="recommendation_state",
        title="Recommendation State",
        summary=(
            f"Recommendation is {recommendation_state.get('state') or 'draft'} "
            f"with lock state {bool(recommendation_state.get('locked'))}."
        ),
        payload={
            "recommendation_state": recommendation_state,
            "history": _recommendation_history_summary(recommendation_history),
        },
    )
    return store.update_dossier(
        str(dossier.get("dossier_id") or ""),
        {
            "current_summary": current_summary,
            "metadata": metadata,
            "sections": sections,
            "updated_at": _now_utc(),
        },
    )


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
            "stored_exports": [],
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
    stored_exports: List[Dict[str, Any]] = []
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
        stored_exports.extend(
            store.list_stored_exports(
                dossier_id=dossier.get("dossier_id"),
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
        "stored_exports": stored_exports,
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
    dossier = _refresh_dossier_collaboration_state(
        dossier,
        workflow=workflow,
        store=store,
    ) or dossier
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
    scoped_org_ids = None if normalized.is_system else normalized.organization_ids
    scoped_workspace_ids = None if normalized.is_system else normalized.workspace_ids
    dossier = (
        _refresh_dossier_collaboration_state(
            dossier,
            workflow=workflow,
            store=store,
            organization_ids=scoped_org_ids,
            workspace_ids=scoped_workspace_ids,
        )
        or dossier
    )
    approvals = store.list_approval_requests(
        workflow_id=(workflow or {}).get("workflow_id"),
        dossier_id=dossier_id,
    )
    exports = store.list_export_manifests(dossier_id)
    collaboration = _build_collaboration_bundle(
        workflow=workflow,
        dossier=dossier,
        store=store,
        organization_ids=scoped_org_ids,
        workspace_ids=scoped_workspace_ids,
    )
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
        "comments": collaboration["comments"],
        "review_summary": collaboration["review_summary"],
        "assignments": collaboration["assignments"],
        "assignment_summary": collaboration["assignment_summary"],
        "committee_decision": collaboration["committee_decision"],
        "recommendation_state": collaboration["recommendation_state"],
        "recommendation_history": collaboration["recommendation_history"],
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
    persisted = (
        _refresh_dossier_collaboration_state(
            persisted,
            workflow=workflow,
            store=store,
        )
        or persisted
    )
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


def list_dossier_comments_service(
    dossier_id: str,
    *,
    include_resolved: bool = True,
    user_context: Optional[Dict[str, Any]] = None,
    store: PlatformStore = platform_store,
) -> Dict[str, Any]:
    dossier_view = get_dossier_view(
        dossier_id,
        user_context=user_context,
        store=store,
    )
    dossier = dossier_view["dossier"]
    workflow = dossier_view["workflow"]
    workspace = dossier_view["workspace"]
    normalized = normalize_user_context(
        user_context,
        organization_id=(workspace or {}).get("organization_id"),
        workspace_id=(workspace or {}).get("workspace_id"),
    )
    comments = store.list_review_comments(
        workflow_id=(workflow or {}).get("workflow_id"),
        dossier_id=dossier_id,
        include_resolved=include_resolved,
        organization_ids=None if normalized.is_system else normalized.organization_ids,
        workspace_ids=None if normalized.is_system else normalized.workspace_ids,
    )
    review_summary = build_review_summary(
        workflow_id=str((workflow or {}).get("workflow_id") or ""),
        dossier_id=dossier_id,
        comments=comments,
        committee_decision=dossier_view.get("committee_decision") or None,
    )
    return {
        "platform_version": PLATFORM_FOUNDATION_VERSION,
        "dossier": dossier,
        "workflow": workflow,
        "comments": comments,
        "review_summary": review_summary,
    }


def create_dossier_comment_service(
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
        permission="update_dossier",
        user_context=normalized,
        resource=_dossier_resource(dossier, workspace),
        store=store,
    )
    comment = build_review_comment(
        organization_id=(workspace or {}).get("organization_id"),
        workspace_id=(workspace or {}).get("workspace_id"),
        workflow_id=str((workflow or {}).get("workflow_id") or ""),
        dossier_id=dossier_id,
        stage=payload.get("stage") or (workflow or {}).get("stage"),
        user_context=normalized,
        comment_type=str(payload.get("comment_type") or "general"),
        body=str(payload.get("body") or ""),
        severity=str(payload.get("severity") or "info"),
        metadata=payload.get("metadata") or {},
    )
    persisted_comment = store.create_review_comment(comment.model_dump(mode="python"))
    dossier = (
        _refresh_dossier_collaboration_state(
            dossier,
            workflow=workflow,
            store=store,
            organization_ids=None if normalized.is_system else normalized.organization_ids,
            workspace_ids=None if normalized.is_system else normalized.workspace_ids,
        )
        or dossier
    )
    audit = _create_audit_event(
        event_type="comment_added",
        resource=_dossier_resource(dossier, workspace),
        user_context=normalized,
        payload={
            "workflow_id": (workflow or {}).get("workflow_id"),
            "dossier_id": dossier_id,
            "comment_id": persisted_comment.get("comment_id"),
            "stage": persisted_comment.get("stage"),
            "status": persisted_comment.get("status"),
            "summary": persisted_comment.get("body"),
        },
        rationale="Structured review comment added.",
        metadata={"comment_type": persisted_comment.get("comment_type")},
        store=store,
    )
    return {
        "platform_version": PLATFORM_FOUNDATION_VERSION,
        "comment": persisted_comment,
        "dossier": dossier,
        "review_summary": (dossier.get("metadata") or {}).get("review_summary") or {},
        "audit_event": audit,
    }


def resolve_dossier_comment_service(
    dossier_id: str,
    comment_id: str,
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
        permission="update_dossier",
        user_context=normalized,
        resource=_dossier_resource(dossier, workspace),
        store=store,
    )
    comment = store.get_review_comment(
        comment_id,
        organization_ids=None if normalized.is_system else normalized.organization_ids,
        workspace_ids=None if normalized.is_system else normalized.workspace_ids,
    )
    if comment is None or str(comment.get("dossier_id") or "") != str(dossier_id):
        raise HTTPException(status_code=404, detail="review comment not found")
    resolved = resolve_review_comment(
        comment,
        user_context=normalized,
        rationale=payload.get("rationale"),
        metadata=payload.get("metadata") or {},
    )
    persisted_comment = store.update_review_comment(comment_id, resolved)
    dossier = (
        _refresh_dossier_collaboration_state(
            dossier,
            workflow=workflow,
            store=store,
            organization_ids=None if normalized.is_system else normalized.organization_ids,
            workspace_ids=None if normalized.is_system else normalized.workspace_ids,
        )
        or dossier
    )
    audit = _create_audit_event(
        event_type="comment_resolved",
        resource=_dossier_resource(dossier, workspace),
        user_context=normalized,
        payload={
            "workflow_id": (workflow or {}).get("workflow_id"),
            "dossier_id": dossier_id,
            "comment_id": persisted_comment.get("comment_id"),
            "stage": persisted_comment.get("stage"),
            "status": persisted_comment.get("status"),
            "summary": persisted_comment.get("body"),
        },
        rationale=payload.get("rationale") or "Review concern resolved.",
        metadata={"comment_type": persisted_comment.get("comment_type")},
        store=store,
    )
    return {
        "platform_version": PLATFORM_FOUNDATION_VERSION,
        "comment": persisted_comment,
        "dossier": dossier,
        "review_summary": (dossier.get("metadata") or {}).get("review_summary") or {},
        "audit_event": audit,
    }


def list_workflow_assignments_service(
    workflow_id: str,
    *,
    dossier_id: Optional[str] = None,
    user_context: Optional[Dict[str, Any]] = None,
    store: PlatformStore = platform_store,
) -> Dict[str, Any]:
    workflow = store.get_workflow(workflow_id)
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
    assignments = store.list_assignments(
        workflow_id=workflow_id,
        dossier_id=dossier_id,
        organization_ids=None if normalized.is_system else normalized.organization_ids,
        workspace_ids=None if normalized.is_system else normalized.workspace_ids,
    )
    summary = build_role_assignment_summary(
        workflow_id=workflow_id,
        dossier_id=dossier_id,
        assignments=assignments,
    )
    return {
        "platform_version": PLATFORM_FOUNDATION_VERSION,
        "workflow": workflow,
        "assignments": assignments,
        "assignment_summary": summary,
    }


def update_workflow_assignment_service(
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
    normalized = normalize_user_context(
        user_context,
        organization_id=(workspace or {}).get("organization_id"),
        workspace_id=(workspace or {}).get("workspace_id"),
    )
    require_access(
        permission="update_dossier",
        user_context=normalized,
        resource=_workflow_resource(workflow, workspace),
        store=store,
    )
    dossier_id = payload.get("dossier_id")
    if dossier_id:
        dossier = store.get_dossier(str(dossier_id))
        if dossier is None:
            raise HTTPException(status_code=404, detail="dossier not found")
    else:
        dossier = next(iter(store.list_dossiers(workflow_id=workflow_id, limit=1)), None)
    existing = store.find_assignment_by_slot(
        workflow_id=workflow_id,
        dossier_id=(dossier or {}).get("dossier_id"),
        slot_type=str(payload.get("slot_type") or "owner"),
        organization_ids=None if normalized.is_system else normalized.organization_ids,
        workspace_ids=None if normalized.is_system else normalized.workspace_ids,
    )
    assignment_payload = {
        "assignment_id": (existing or {}).get("assignment_id") or str(uuid.uuid4()),
        "organization_id": (workspace or {}).get("organization_id"),
        "workspace_id": (workspace or {}).get("workspace_id"),
        "workflow_id": workflow_id,
        "dossier_id": (dossier or {}).get("dossier_id"),
        "slot_type": str(payload.get("slot_type") or "owner"),
        "assignee_placeholder": payload.get("assignee_placeholder"),
        "assigned_by": {
            "user_id": normalized.user_id,
            "role": normalized.role,
            "auth_mode": normalized.auth_mode,
            "session_id": normalized.session_id,
            "actor_email": normalized.email,
        },
        "status": str(payload.get("status") or "assigned"),
        "notes": list(payload.get("notes") or []),
        "metadata": payload.get("metadata") or {},
        "created_at": (existing or {}).get("created_at"),
    }
    assignment = (
        store.update_assignment(existing["assignment_id"], assignment_payload)
        if existing
        else store.create_assignment(assignment_payload)
    )
    dossier = (
        _refresh_dossier_collaboration_state(
            dossier,
            workflow=workflow,
            store=store,
            organization_ids=None if normalized.is_system else normalized.organization_ids,
            workspace_ids=None if normalized.is_system else normalized.workspace_ids,
        )
        if dossier
        else dossier
    ) or dossier
    summary = build_role_assignment_summary(
        workflow_id=workflow_id,
        dossier_id=(dossier or {}).get("dossier_id"),
        assignments=store.list_assignments(
            workflow_id=workflow_id,
            dossier_id=(dossier or {}).get("dossier_id"),
            organization_ids=None if normalized.is_system else normalized.organization_ids,
            workspace_ids=None if normalized.is_system else normalized.workspace_ids,
        ),
    )
    audit = _create_audit_event(
        event_type="assignment_updated",
        resource=_workflow_resource(workflow, workspace),
        user_context=normalized,
        payload={
            "workflow_id": workflow_id,
            "dossier_id": (dossier or {}).get("dossier_id"),
            "assignment_id": assignment.get("assignment_id"),
            "slot_type": assignment.get("slot_type"),
            "status": assignment.get("status"),
            "summary": assignment.get("assignee_placeholder"),
        },
        rationale="Reviewer assignment updated.",
        metadata={"slot_type": assignment.get("slot_type")},
        store=store,
    )
    return {
        "platform_version": PLATFORM_FOUNDATION_VERSION,
        "assignment": assignment,
        "assignment_summary": summary,
        "dossier": dossier,
        "audit_event": audit,
    }


def get_workflow_review_summary_service(
    workflow_id: str,
    *,
    dossier_id: Optional[str] = None,
    user_context: Optional[Dict[str, Any]] = None,
    store: PlatformStore = platform_store,
) -> Dict[str, Any]:
    workflow = store.get_workflow(workflow_id)
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
    dossier = (
        store.get_dossier(str(dossier_id))
        if dossier_id
        else next(iter(store.list_dossiers(workflow_id=workflow_id, limit=1)), None)
    )
    collaboration = _build_collaboration_bundle(
        workflow=workflow,
        dossier=dossier,
        store=store,
        organization_ids=None if normalized.is_system else normalized.organization_ids,
        workspace_ids=None if normalized.is_system else normalized.workspace_ids,
    )
    return {
        "platform_version": PLATFORM_FOUNDATION_VERSION,
        "workflow": workflow,
        "dossier": dossier,
        "review_summary": collaboration["review_summary"],
        "comments": collaboration["comments"],
        "committee_decision": collaboration["committee_decision"],
    }


def _apply_recommendation_state_update(
    *,
    workflow: Dict[str, Any],
    dossier: Dict[str, Any],
    workspace: Optional[Dict[str, Any]],
    next_state: str,
    action_type: str,
    summary: Optional[str],
    rationale: Optional[str],
    lock_recommendation: Optional[bool],
    source_decision_id: Optional[str],
    user_context: Any,
    metadata: Optional[Dict[str, Any]],
    store: PlatformStore,
) -> Dict[str, Any]:
    current_state = _current_recommendation_state(dossier)
    next_payload = build_recommendation_state(
        current_state,
        next_state=next_state,
        user_context=user_context,
        summary=summary,
        rationale=rationale,
        lock_recommendation=lock_recommendation,
        source_decision_id=source_decision_id,
        metadata=metadata or {},
    )
    change_record = build_recommendation_change_record(
        organization_id=(workspace or {}).get("organization_id"),
        workspace_id=(workspace or {}).get("workspace_id"),
        workflow_id=str(workflow.get("workflow_id") or ""),
        dossier_id=str(dossier.get("dossier_id") or ""),
        previous_state=current_state.get("state"),
        new_state=str(next_payload.get("state") or ""),
        action_type=action_type,
        recommendation_state=next_payload,
        rationale=rationale,
        user_context=user_context,
        metadata=metadata or {},
    )
    persisted_change = store.create_recommendation_change(change_record)
    if lock_recommendation:
        lock_record = build_recommendation_lock_record(
            workflow_id=str(workflow.get("workflow_id") or ""),
            dossier_id=str(dossier.get("dossier_id") or ""),
            recommendation_state=next_payload,
            reason=rationale,
            user_context=user_context,
            metadata=metadata or {},
        )
    else:
        lock_record = None
    updated_dossier = store.update_dossier(
        str(dossier.get("dossier_id") or ""),
        {
            "metadata": {
                **dict(dossier.get("metadata") or {}),
                "recommendation_state": next_payload,
                "last_recommendation_change": persisted_change,
                "last_recommendation_lock": lock_record,
            },
            "updated_at": _now_utc(),
        },
    )
    updated_dossier = (
        _refresh_dossier_collaboration_state(
            updated_dossier,
            workflow=workflow,
            store=store,
            organization_ids=None if getattr(user_context, "is_system", False) else getattr(user_context, "organization_ids", None),
            workspace_ids=None if getattr(user_context, "is_system", False) else getattr(user_context, "workspace_ids", None),
        )
        or updated_dossier
    )
    return {
        "dossier": updated_dossier,
        "recommendation_state": next_payload,
        "recommendation_change": persisted_change,
        "recommendation_lock": lock_record,
    }


def record_committee_decision_service(
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
    normalized = normalize_user_context(
        user_context,
        organization_id=(workspace or {}).get("organization_id"),
        workspace_id=(workspace or {}).get("workspace_id"),
    )
    require_access(
        permission="approve_stage",
        user_context=normalized,
        resource=_workflow_resource(workflow, workspace),
        store=store,
    )
    dossier = (
        store.get_dossier(str(payload.get("dossier_id")))
        if payload.get("dossier_id")
        else next(iter(store.list_dossiers(workflow_id=workflow_id, limit=1)), None)
    )
    if dossier is None:
        raise HTTPException(status_code=404, detail="dossier not found")
    decision = build_committee_decision_snapshot(
        organization_id=(workspace or {}).get("organization_id"),
        workspace_id=(workspace or {}).get("workspace_id"),
        workflow_id=workflow_id,
        dossier_id=str(dossier.get("dossier_id") or ""),
        stage=workflow.get("stage"),
        decision_status=str(payload.get("decision_status") or "deferred"),
        recommendation_state=str(payload.get("recommendation_state") or "under_review"),
        summary=str(payload.get("summary") or ""),
        conditions=list(payload.get("conditions") or []),
        key_risks=list(payload.get("key_risks") or []),
        key_evidence_strengths=list(payload.get("key_evidence_strengths") or []),
        key_evidence_gaps=list(payload.get("key_evidence_gaps") or []),
        user_context=normalized,
        rationale=payload.get("rationale"),
        metadata=payload.get("metadata") or {},
    )
    persisted_decision = store.create_committee_decision(decision.model_dump(mode="python"))
    recommendation_update = _apply_recommendation_state_update(
        workflow=workflow,
        dossier=dossier,
        workspace=workspace,
        next_state=str(payload.get("recommendation_state") or "under_review"),
        action_type="committee_decision",
        summary=str(payload.get("summary") or ""),
        rationale=payload.get("rationale"),
        lock_recommendation=bool((payload.get("metadata") or {}).get("lock_recommendation")),
        source_decision_id=str(persisted_decision.get("decision_id") or ""),
        user_context=normalized,
        metadata=payload.get("metadata") or {},
        store=store,
    )
    updated_dossier = recommendation_update["dossier"]
    audit = _create_audit_event(
        event_type="committee_decision_recorded",
        resource=_workflow_resource(workflow, workspace),
        user_context=normalized,
        payload={
            "workflow_id": workflow_id,
            "dossier_id": dossier.get("dossier_id"),
            "decision_id": persisted_decision.get("decision_id"),
            "stage": persisted_decision.get("stage"),
            "status": persisted_decision.get("decision_status"),
            "summary": persisted_decision.get("summary"),
        },
        rationale=payload.get("rationale") or persisted_decision.get("summary"),
        metadata={"recommendation_state": persisted_decision.get("recommendation_state")},
        store=store,
    )
    return {
        "platform_version": PLATFORM_FOUNDATION_VERSION,
        "workflow": workflow,
        "dossier": updated_dossier,
        "committee_decision": persisted_decision,
        "recommendation_state": recommendation_update["recommendation_state"],
        "recommendation_change": recommendation_update["recommendation_change"],
        "audit_event": audit,
    }


def get_committee_decision_service(
    workflow_id: str,
    *,
    dossier_id: Optional[str] = None,
    user_context: Optional[Dict[str, Any]] = None,
    store: PlatformStore = platform_store,
) -> Dict[str, Any]:
    workflow = store.get_workflow(workflow_id)
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
    dossier = (
        store.get_dossier(str(dossier_id))
        if dossier_id
        else next(iter(store.list_dossiers(workflow_id=workflow_id, limit=1)), None)
    )
    decision = store.get_latest_committee_decision(
        workflow_id=workflow_id,
        dossier_id=(dossier or {}).get("dossier_id"),
        organization_ids=None if normalized.is_system else normalized.organization_ids,
        workspace_ids=None if normalized.is_system else normalized.workspace_ids,
    )
    return {
        "platform_version": PLATFORM_FOUNDATION_VERSION,
        "workflow": workflow,
        "dossier": dossier,
        "committee_decision": decision,
    }


def update_recommendation_state_service(
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
    normalized = normalize_user_context(
        user_context,
        organization_id=(workspace or {}).get("organization_id"),
        workspace_id=(workspace or {}).get("workspace_id"),
    )
    action_type = str(payload.get("action_type") or "revise_recommendation")
    require_access(
        permission=permission_for_action(action_type),
        user_context=normalized,
        resource=_workflow_resource(workflow, workspace),
        store=store,
    )
    dossier = (
        store.get_dossier(str(payload.get("dossier_id")))
        if payload.get("dossier_id")
        else next(iter(store.list_dossiers(workflow_id=workflow_id, limit=1)), None)
    )
    if dossier is None:
        raise HTTPException(status_code=404, detail="dossier not found")
    recommendation_update = _apply_recommendation_state_update(
        workflow=workflow,
        dossier=dossier,
        workspace=workspace,
        next_state=str(payload.get("recommendation_state") or "draft"),
        action_type=action_type,
        summary=payload.get("summary"),
        rationale=payload.get("rationale"),
        lock_recommendation=payload.get("lock_recommendation"),
        source_decision_id=None,
        user_context=normalized,
        metadata=payload.get("metadata") or {},
        store=store,
    )
    event_map = {
        "freeze_recommendation": "recommendation_frozen",
        "revise_recommendation": "recommendation_revised",
        "downgrade_to_watch": "downgraded_to_watch",
        "downgrade_to_paper": "downgraded_to_paper",
        "reopen_review": "review_reopened",
    }
    audit = _create_audit_event(
        event_type=event_map.get(action_type, "recommendation_revised"),
        resource=_workflow_resource(workflow, workspace),
        user_context=normalized,
        payload={
            "workflow_id": workflow_id,
            "dossier_id": dossier.get("dossier_id"),
            "stage": workflow.get("stage"),
            "status": workflow.get("status"),
            "summary": payload.get("summary"),
            "recommendation_state": recommendation_update["recommendation_state"],
        },
        rationale=payload.get("rationale"),
        metadata={"action_type": action_type},
        store=store,
    )
    return {
        "platform_version": PLATFORM_FOUNDATION_VERSION,
        "workflow": workflow,
        "dossier": recommendation_update["dossier"],
        "recommendation_state": recommendation_update["recommendation_state"],
        "recommendation_change": recommendation_update["recommendation_change"],
        "recommendation_lock": recommendation_update["recommendation_lock"],
        "audit_event": audit,
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
    action_type = str(payload.get("action_type") or "")
    if action_type == "escalate_to_committee" and updated_dossier is not None:
        existing_pending = next(
            (
                item
                for item in approvals
                if str(item.get("status") or "") == "pending"
                and str(item.get("requested_role") or "") == "committee"
            ),
            None,
        )
        escalation = build_escalation_record(
            organization_id=(workspace or {}).get("organization_id"),
            workspace_id=(workspace or {}).get("workspace_id"),
            workflow_id=workflow_id,
            dossier_id=updated_dossier.get("dossier_id"),
            action_type="escalate_to_committee",
            from_state=_current_recommendation_state(updated_dossier).get("state"),
            to_state="under_review",
            rationale=payload.get("rationale"),
            user_context=normalized,
            metadata=payload.get("metadata") or {},
        )
        if existing_pending is None:
            approval = build_approval_request(
                workflow_id=workflow_id,
                dossier_id=updated_dossier.get("dossier_id"),
                requested_role="committee",
                user_context=normalized,
                stage=str(updated_workflow.get("stage") or ""),
                rationale=payload.get("rationale"),
                required_permissions=["approve_stage"],
                metadata={
                    **dict(payload.get("metadata") or {}),
                    "escalation": escalation,
                },
            )
            store.create_approval_request(approval.model_dump(mode="python"))
    elif action_type == "resolve_concerns" and updated_dossier is not None:
        open_comments = store.list_review_comments(
            workflow_id=workflow_id,
            dossier_id=updated_dossier.get("dossier_id"),
            include_resolved=False,
            organization_ids=None if normalized.is_system else normalized.organization_ids,
            workspace_ids=None if normalized.is_system else normalized.workspace_ids,
        )
        for comment in open_comments:
            resolved = resolve_review_comment(
                comment,
                user_context=normalized,
                rationale=payload.get("rationale") or "Workflow concern resolution action applied.",
                metadata={"resolved_by_action": "resolve_concerns"},
            )
            store.update_review_comment(str(comment.get("comment_id") or ""), resolved)
    if action_type in {
        "downgrade_to_watch",
        "downgrade_to_paper",
        "freeze_recommendation",
        "revise_recommendation",
        "reopen_review",
    } and updated_dossier is not None:
        target_state_map = {
            "downgrade_to_watch": "watch_only",
            "downgrade_to_paper": "approved_paper",
            "freeze_recommendation": _current_recommendation_state(updated_dossier).get("state")
            or "draft",
            "revise_recommendation": str(
                (payload.get("metadata") or {}).get("recommendation_state")
                or _current_recommendation_state(updated_dossier).get("state")
                or "draft"
            ),
            "reopen_review": "under_review",
        }
        recommendation_result = _apply_recommendation_state_update(
            workflow=updated_workflow,
            dossier=updated_dossier,
            workspace=workspace,
            next_state=str(target_state_map.get(action_type) or "draft"),
            action_type=action_type,
            summary=(payload.get("metadata") or {}).get("summary")
            or payload.get("rationale"),
            rationale=payload.get("rationale"),
            lock_recommendation=True if action_type == "freeze_recommendation" else (
                False if action_type == "reopen_review" else None
            ),
            source_decision_id=None,
            user_context=normalized,
            metadata=payload.get("metadata") or {},
            store=store,
        )
        updated_dossier = recommendation_result["dossier"]
    elif updated_dossier is not None:
        updated_dossier = (
            _refresh_dossier_collaboration_state(
                updated_dossier,
                workflow=updated_workflow,
                store=store,
                organization_ids=None if normalized.is_system else normalized.organization_ids,
                workspace_ids=None if normalized.is_system else normalized.workspace_ids,
            )
            or updated_dossier
        )
    workflow_event_type = {
        "escalate_to_committee": "escalated_to_committee",
        "downgrade_to_watch": "downgraded_to_watch",
        "downgrade_to_paper": "downgraded_to_paper",
        "freeze_recommendation": "recommendation_frozen",
        "revise_recommendation": "recommendation_revised",
        "reopen_review": "review_reopened",
    }.get(action_type, f"workflow_{payload.get('action_type')}")
    audit = _create_audit_event(
        event_type=workflow_event_type,
        resource=_workflow_resource(updated_workflow, workspace),
        user_context=normalized,
        payload={
            "workflow_id": workflow_id,
            "dossier_id": (updated_dossier or {}).get("dossier_id"),
            "stage": updated_workflow.get("stage"),
            "status": updated_workflow.get("status"),
            "summary": applied.get("summary"),
            "recommendation_state": (
                ((updated_dossier or {}).get("metadata") or {}).get("recommendation_state")
            ),
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
        "stored_exports": store.list_stored_exports(
            dossier_id=dossier_id,
            organization_ids=scope["organization_ids"],
            workspace_ids=scope["workspace_ids"],
        ),
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


def _storage_version_group_key(
    *,
    dossier_id: str,
    pack_type: str,
    export_format: str,
) -> str:
    return f"{dossier_id}:{pack_type}:{export_format}"


def _storage_key_for_export(
    *,
    organization_id: Optional[str],
    workspace_id: Optional[str],
    dossier_id: Optional[str],
    pack_type: str,
    export_format: str,
    version_number: int,
    file_name_hint: str,
) -> str:
    org_part = str(organization_id or "org-unscoped")
    workspace_part = str(workspace_id or "workspace-unscoped")
    dossier_part = str(dossier_id or "dossier-unscoped")
    return (
        f"{org_part}/{workspace_part}/{dossier_part}/"
        f"{pack_type}/{export_format}/v{version_number:04d}/{file_name_hint}"
    )


def _export_evidence_status(
    *,
    dossier: Dict[str, Any],
    report: Dict[str, Any],
) -> str:
    return str(
        dossier.get("evidence_status")
        or (report.get("axiom_summary_card") or {}).get("evidence_status")
        or "partial"
    )


def _build_stored_export_record_payload(
    *,
    dossier: Dict[str, Any],
    workflow: Optional[Dict[str, Any]],
    workspace: Optional[Dict[str, Any]],
    report: Dict[str, Any],
    manifest: Dict[str, Any],
    rendered_export: Dict[str, Any],
    normalized_user_context: Any,
    store: PlatformStore,
    storage_backend: str,
    storage_key: str,
    storage_ref: Dict[str, Any],
    version_group_key: str,
    version_number: int,
) -> Dict[str, Any]:
    actor = build_access_summary(
        user_context=normalized_user_context,
        resource=_workspace_resource(workspace),
        store=store,
    )
    evidence_status = _export_evidence_status(dossier=dossier, report=report)
    pack_type = str(manifest.get("pack_type") or "dossier_pack")
    export_format = str(rendered_export.get("export_format") or "html")
    version_label = f"v{version_number}"
    format_capabilities = export_format_capabilities(export_format)
    layout_metadata = build_export_layout_metadata(
        manifest,
        export_format=export_format,
    )
    return StoredExportRecord(
        stored_export_id=str(uuid.uuid4()),
        export_id=str(manifest.get("export_id") or ""),
        render_id=str(rendered_export.get("render_id") or ""),
        organization_id=(workspace or {}).get("organization_id"),
        workspace_id=(workspace or {}).get("workspace_id"),
        dossier_id=dossier.get("dossier_id"),
        workflow_id=(workflow or {}).get("workflow_id"),
        pack_type=pack_type,
        export_format=export_format,
        framework_version=manifest.get("framework_version"),
        approval_status=manifest.get("approval_status"),
        evidence_status=evidence_status,
        checksum=rendered_export.get("checksum"),
        source_manifest_hash=manifest.get("content_hash"),
        content_hash=rendered_export.get("checksum"),
        manifest_hash=manifest.get("content_hash"),
        section_count=int(rendered_export.get("section_count") or 0),
        file_name_hint=str(rendered_export.get("file_name_hint") or ""),
        content_type=rendered_export.get("content_type"),
        storage_backend=storage_backend,
        storage_key=storage_key,
        storage_ref=storage_ref,
        version_group_key=version_group_key,
        version_number=version_number,
        version_label=version_label,
        status="stored",
        document_identity={
            "title": manifest.get("title"),
            "subtitle": manifest.get("subtitle"),
            "file_name_hint": rendered_export.get("file_name_hint"),
            "version_number": version_number,
            "version_label": version_label,
            "content_type": rendered_export.get("content_type"),
        },
        source_context={
            "organization": manifest.get("organization_context") or {},
            "workspace": manifest.get("workspace_context") or {},
            "entity": manifest.get("entity_context") or {},
            "dossier_id": dossier.get("dossier_id"),
            "workflow_id": (workflow or {}).get("workflow_id"),
        },
        approval_context={
            "approval_status": manifest.get("approval_status"),
            "workflow_stage": (workflow or {}).get("stage"),
            "workflow_status": (workflow or {}).get("status"),
        },
        axiom_context={
            "symbol": report.get("symbol"),
            "framework_version": manifest.get("framework_version"),
            "regime_label": dossier.get("latest_regime_label"),
            "trade_family": dossier.get("latest_trade_family"),
            "deployability_tier": dossier.get("latest_deployability_tier"),
            "size_band": dossier.get("latest_size_band"),
            "deployable_alpha_utility": report.get("axiom_deployable_alpha_utility"),
            "validated_edge": report.get("axiom_validated_edge"),
        },
        evidence_context={
            "evidence_status": evidence_status,
            "evidence_summary": manifest.get("evidence_summary"),
            "lineage_summary": report.get("axiom_lineage_summary"),
            "historical_evidence_summary": report.get(
                "axiom_historical_evidence_summary_text"
            ),
        },
        export_context={
            "pack_type": pack_type,
            "export_format": export_format,
            "storage_backend": storage_backend,
            "storage_key": storage_key,
            "format_capabilities": format_capabilities,
            "layout_metadata": layout_metadata,
            "print_ready": bool(layout_metadata.get("print_ready")),
        },
        lineage_summary=sanitize_payload(report.get("axiom_lineage") or {}),
        created_by={
            "user_id": normalized_user_context.user_id,
            "role": normalized_user_context.role,
            "auth_mode": normalized_user_context.auth_mode,
            "session_id": normalized_user_context.session_id,
            "actor_email": normalized_user_context.email,
        },
        metadata={
            "source_manifest_metadata": manifest.get("metadata") or {},
            "render_metadata": rendered_export.get("metadata") or {},
            "latest_access_role": actor.get("effective_role"),
            "content_preview": str(rendered_export.get("rendered_content") or "")[
                :240
            ],
        },
    ).model_dump(mode="python")


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


def _persist_latest_stored_export_reference(
    *,
    dossier: Dict[str, Any],
    stored_export: Dict[str, Any],
    store: PlatformStore,
) -> Dict[str, Any]:
    metadata = dict(dossier.get("metadata") or {})
    latest = dict(metadata.get("latest_stored_exports") or {})
    latest[
        f"{stored_export.get('pack_type')}:{stored_export.get('export_format')}"
    ] = {
        "stored_export_id": stored_export.get("stored_export_id"),
        "version_number": stored_export.get("version_number"),
        "version_label": stored_export.get("version_label"),
        "status": stored_export.get("status"),
        "generated_at": stored_export.get("created_at"),
    }
    metadata["latest_stored_exports"] = sanitize_payload(latest)
    return store.update_dossier(
        str(dossier.get("dossier_id") or ""),
        {"metadata": metadata},
    )


def store_dossier_export_service(
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
    organization_ids = (
        None
        if context["normalized"].is_system
        else context["normalized"].organization_ids
    )
    workspace_ids = (
        None if context["normalized"].is_system else context["normalized"].workspace_ids
    )
    rendered_export = None
    manifest = None
    requested_render_id = str(payload.get("render_id") or "").strip() or None
    if requested_render_id:
        rendered_export = store.get_rendered_export(
            requested_render_id,
            organization_ids=organization_ids,
            workspace_ids=workspace_ids,
        )
        if rendered_export is None:
            raise HTTPException(status_code=404, detail="rendered export not found")
        manifest = store.get_export_manifest(
            str(rendered_export.get("export_id") or ""),
            organization_ids=organization_ids,
            workspace_ids=workspace_ids,
        )
        if manifest is None or str(manifest.get("dossier_id") or "") != str(dossier_id):
            raise HTTPException(
                status_code=400,
                detail="rendered export does not belong to the requested dossier",
            )
    else:
        render_response = render_dossier_export_service(
            dossier_id,
            {
                "pack_type": str(payload.get("pack_type") or "dossier_pack"),
                "export_format": str(payload.get("export_format") or "html"),
                "metadata": payload.get("metadata") or {},
            },
            user_context=context["normalized"].model_dump(mode="python"),
            store=store,
        )
        manifest = render_response["export"]
        rendered_export = render_response["rendered_export"]
    pack_type = str((manifest or {}).get("pack_type") or payload.get("pack_type") or "dossier_pack")
    export_format = str(
        (rendered_export or {}).get("export_format") or payload.get("export_format") or "html"
    )
    version_group_key = _storage_version_group_key(
        dossier_id=dossier_id,
        pack_type=pack_type,
        export_format=export_format,
    )
    existing_versions = store.list_stored_exports(
        dossier_id=dossier_id,
        pack_type=pack_type,
        export_format=export_format,
        organization_ids=organization_ids,
        workspace_ids=workspace_ids,
    )
    version_number = max(
        [int(item.get("version_number") or 0) for item in existing_versions] or [0]
    ) + 1
    storage_backend = str(
        payload.get("storage_backend")
        or default_storage_backend_name(use_memory=store.use_memory)
    )
    storage_key = _storage_key_for_export(
        organization_id=(context["workspace"] or {}).get("organization_id"),
        workspace_id=(context["workspace"] or {}).get("workspace_id"),
        dossier_id=dossier_id,
        pack_type=pack_type,
        export_format=export_format,
        version_number=version_number,
        file_name_hint=str((rendered_export or {}).get("file_name_hint") or "export.txt"),
    )
    try:
        storage_ref = store_export_content(
            storage_backend=storage_backend,
            storage_key=storage_key,
            rendered_content=str((rendered_export or {}).get("rendered_content") or ""),
            content_type=str((rendered_export or {}).get("content_type") or "text/plain"),
            metadata={
                "export_id": (manifest or {}).get("export_id"),
                "render_id": (rendered_export or {}).get("render_id"),
                "pack_type": pack_type,
                "export_format": export_format,
            },
        ).model_dump(mode="python")
    except Exception as exc:
        audit = _create_audit_event(
            event_type="export_storage_failed",
            resource=_dossier_resource(context["dossier"], context["workspace"]),
            user_context=context["normalized"],
            payload={
                "workflow_id": context["dossier"].get("workflow_id"),
                "dossier_id": dossier_id,
                "pack_type": pack_type,
                "export_format": export_format,
                "storage_backend": storage_backend,
                "error": str(exc),
            },
            rationale="Durable export storage failed.",
            metadata={"storage_backend": storage_backend},
            store=store,
        )
        raise HTTPException(status_code=500, detail={"error": "export_storage_failed", "message": str(exc), "audit_event": audit})
    stored_payload = _build_stored_export_record_payload(
        dossier=context["dossier"],
        workflow=context["workflow"],
        workspace=context["workspace"],
        report=context["report"],
        manifest=manifest or {},
        rendered_export=rendered_export or {},
        normalized_user_context=context["normalized"],
        store=store,
        storage_backend=storage_backend,
        storage_key=storage_key,
        storage_ref=storage_ref,
        version_group_key=version_group_key,
        version_number=version_number,
    )
    stored_export = store.create_stored_export(stored_payload)
    superseded = store.supersede_stored_exports(
        version_group_key=version_group_key,
        except_stored_export_id=str(stored_export.get("stored_export_id") or ""),
    )
    dossier = _persist_latest_stored_export_reference(
        dossier=context["dossier"],
        stored_export=stored_export,
        store=store,
    )
    retrieval = retrieve_export_content(
        storage_backend=str(stored_export.get("storage_backend") or ""),
        storage_key=str(stored_export.get("storage_key") or ""),
    )
    integrity = build_export_integrity_result(
        stored_export=stored_export,
        manifest=manifest or {},
        rendered_export=rendered_export or {},
        rendered_content=str(retrieval.get("rendered_content") or ""),
    ).model_dump(mode="python")
    audit = _create_audit_event(
        event_type="export_stored",
        resource=_dossier_resource(dossier, context["workspace"]),
        user_context=context["normalized"],
        payload={
            "workflow_id": dossier.get("workflow_id"),
            "dossier_id": dossier_id,
            "stored_export_id": stored_export.get("stored_export_id"),
            "export_id": stored_export.get("export_id"),
            "render_id": stored_export.get("render_id"),
            "pack_type": stored_export.get("pack_type"),
            "export_format": stored_export.get("export_format"),
            "version_number": stored_export.get("version_number"),
            "status": stored_export.get("status"),
        },
        rationale="Durable stored export created.",
        metadata={
            "storage_backend": stored_export.get("storage_backend"),
            "storage_key": stored_export.get("storage_key"),
            "checksum": stored_export.get("checksum"),
        },
        store=store,
    )
    supersede_audits = [
        _create_audit_event(
            event_type="export_superseded",
            resource=_dossier_resource(dossier, context["workspace"]),
            user_context=context["normalized"],
            payload={
                "workflow_id": dossier.get("workflow_id"),
                "dossier_id": dossier_id,
                "stored_export_id": item.get("stored_export_id"),
                "status": item.get("status"),
                "version_number": item.get("version_number"),
            },
            rationale="Older stored export version superseded by a newer revision.",
            metadata={"version_group_key": version_group_key},
            store=store,
        )
        for item in superseded
    ]
    return {
        "platform_version": PLATFORM_FOUNDATION_VERSION,
        "dossier": dossier,
        "export": manifest,
        "rendered_export": rendered_export,
        "stored_export": stored_export,
        "integrity": integrity,
        "audit_event": audit,
        "superseded_events": supersede_audits,
    }


def list_dossier_export_history_service(
    dossier_id: str,
    *,
    user_context: Optional[Dict[str, Any]] = None,
    store: PlatformStore = platform_store,
) -> Dict[str, Any]:
    context = _resolve_dossier_export_context(
        dossier_id,
        user_context=user_context,
        store=store,
    )
    stored_exports = store.list_stored_exports(
        dossier_id=dossier_id,
        organization_ids=(
            None
            if context["normalized"].is_system
            else context["normalized"].organization_ids
        ),
        workspace_ids=(
            None
            if context["normalized"].is_system
            else context["normalized"].workspace_ids
        ),
    )
    return {
        "platform_version": PLATFORM_FOUNDATION_VERSION,
        "dossier": context["dossier"],
        "stored_exports": stored_exports,
    }


def list_workspace_stored_exports_service(
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
    require_access(
        permission="view_workspace",
        user_context=scope["user_context"],
        resource=_workspace_resource(workspace),
        store=store,
    )
    return {
        "platform_version": PLATFORM_FOUNDATION_VERSION,
        "workspace": workspace,
        "stored_exports": store.list_stored_exports(
            workspace_id=workspace_id,
            organization_ids=scope["organization_ids"],
            workspace_ids=scope["workspace_ids"],
        ),
    }


def _resolve_stored_export_context(
    stored_export_id: str,
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
    stored_export = store.get_stored_export(
        stored_export_id,
        organization_ids=scope["organization_ids"],
        workspace_ids=scope["workspace_ids"],
    )
    if stored_export is None:
        raise HTTPException(status_code=404, detail="stored export not found")
    workspace = (
        store.get_workspace(
            str(stored_export.get("workspace_id") or ""),
            organization_ids=scope["organization_ids"],
            workspace_ids=scope["workspace_ids"],
        )
        if stored_export.get("workspace_id")
        else None
    )
    normalized = normalize_user_context(
        scope["user_context"],
        organization_id=stored_export.get("organization_id"),
        workspace_id=(workspace or {}).get("workspace_id") or stored_export.get("workspace_id"),
    )
    require_access(
        permission="view_workspace",
        user_context=normalized,
        resource=_workspace_resource(workspace),
        store=store,
    )
    manifest = store.get_export_manifest(
        str(stored_export.get("export_id") or ""),
        organization_ids=scope["organization_ids"],
        workspace_ids=scope["workspace_ids"],
    )
    rendered_export = store.get_rendered_export(
        str(stored_export.get("render_id") or ""),
        organization_ids=scope["organization_ids"],
        workspace_ids=scope["workspace_ids"],
    )
    return {
        "normalized": normalized,
        "workspace": workspace,
        "stored_export": stored_export,
        "manifest": manifest,
        "rendered_export": rendered_export,
    }


def get_stored_export_metadata_service(
    stored_export_id: str,
    *,
    user_context: Optional[Dict[str, Any]] = None,
    store: PlatformStore = platform_store,
) -> Dict[str, Any]:
    context = _resolve_stored_export_context(
        stored_export_id,
        user_context=user_context,
        store=store,
    )
    return {
        "platform_version": PLATFORM_FOUNDATION_VERSION,
        "stored_export": context["stored_export"],
        "export": context["manifest"],
        "rendered_export": context["rendered_export"],
    }


def get_stored_export_content_service(
    stored_export_id: str,
    *,
    user_context: Optional[Dict[str, Any]] = None,
    store: PlatformStore = platform_store,
) -> Dict[str, Any]:
    context = _resolve_stored_export_context(
        stored_export_id,
        user_context=user_context,
        store=store,
    )
    stored_export = context["stored_export"]
    retrieved = retrieve_export_content(
        storage_backend=str(stored_export.get("storage_backend") or ""),
        storage_key=str(stored_export.get("storage_key") or ""),
    )
    payload = ExportRetrievalResult(
        stored_export_id=str(stored_export.get("stored_export_id") or ""),
        export_id=str(stored_export.get("export_id") or ""),
        render_id=str(stored_export.get("render_id") or ""),
        export_format=str(stored_export.get("export_format") or ""),
        content_type=str(stored_export.get("content_type") or retrieved.get("content_type") or "text/plain"),
        rendered_content=str(retrieved.get("rendered_content") or ""),
        file_name_hint=str(stored_export.get("file_name_hint") or ""),
        storage_ref=stored_export.get("storage_ref") or {
            "storage_backend": stored_export.get("storage_backend"),
            "storage_key": stored_export.get("storage_key"),
        },
        retrieved_at=_now_utc(),
        metadata={"size_bytes": retrieved.get("size_bytes")},
    ).model_dump(mode="python")
    audit = _create_audit_event(
        event_type="export_retrieved",
        resource=ResourceRef(
            resource_type="stored_export",
            resource_id=stored_export_id,
            organization_id=stored_export.get("organization_id"),
            workspace_id=stored_export.get("workspace_id"),
        ),
        user_context=context["normalized"],
        payload={
            "stored_export_id": stored_export_id,
            "export_id": stored_export.get("export_id"),
            "render_id": stored_export.get("render_id"),
            "pack_type": stored_export.get("pack_type"),
            "export_format": stored_export.get("export_format"),
            "version_number": stored_export.get("version_number"),
        },
        rationale="Stored export content retrieved.",
        metadata={"storage_backend": stored_export.get("storage_backend")},
        store=store,
    )
    return {
        "platform_version": PLATFORM_FOUNDATION_VERSION,
        "retrieval": payload,
        "audit_event": audit,
    }


def get_stored_export_integrity_service(
    stored_export_id: str,
    *,
    user_context: Optional[Dict[str, Any]] = None,
    store: PlatformStore = platform_store,
) -> Dict[str, Any]:
    context = _resolve_stored_export_context(
        stored_export_id,
        user_context=user_context,
        store=store,
    )
    stored_export = context["stored_export"]
    retrieved = retrieve_export_content(
        storage_backend=str(stored_export.get("storage_backend") or ""),
        storage_key=str(stored_export.get("storage_key") or ""),
    )
    integrity = build_export_integrity_result(
        stored_export=stored_export,
        manifest=context["manifest"],
        rendered_export=context["rendered_export"],
        rendered_content=str(retrieved.get("rendered_content") or ""),
    ).model_dump(mode="python")
    audit = _create_audit_event(
        event_type="export_integrity_checked",
        resource=ResourceRef(
            resource_type="stored_export",
            resource_id=stored_export_id,
            organization_id=stored_export.get("organization_id"),
            workspace_id=stored_export.get("workspace_id"),
        ),
        user_context=context["normalized"],
        payload={
            "stored_export_id": stored_export_id,
            "status": integrity.get("status"),
            "version_number": stored_export.get("version_number"),
        },
        rationale="Stored export integrity checked.",
        metadata={
            "checksum_expected": integrity.get("checksum_expected"),
            "checksum_actual": integrity.get("checksum_actual"),
        },
        store=store,
    )
    return {
        "platform_version": PLATFORM_FOUNDATION_VERSION,
        "integrity": integrity,
        "audit_event": audit,
    }


def list_stored_export_versions_service(
    stored_export_id: str,
    *,
    user_context: Optional[Dict[str, Any]] = None,
    store: PlatformStore = platform_store,
) -> Dict[str, Any]:
    context = _resolve_stored_export_context(
        stored_export_id,
        user_context=user_context,
        store=store,
    )
    versions = store.list_stored_export_versions(
        stored_export_id,
        organization_ids=context["normalized"].organization_ids,
        workspace_ids=context["normalized"].workspace_ids,
    )
    return {
        "platform_version": PLATFORM_FOUNDATION_VERSION,
        "stored_export": context["stored_export"],
        "versions": versions,
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
    stored_exports = scope_bundle["stored_exports"]
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
        stored_exports=stored_exports,
        audit_events=audit_events,
        integration_bindings=integration_bindings,
    )
    health["rendered_export_count"] = len(rendered_exports)
    health["stored_export_count"] = len(stored_exports)
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
    collaboration = _build_collaboration_bundle(
        workflow=workflow,
        dossier=dossier,
        store=store,
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
    stored_exports = (
        store.list_stored_exports(dossier_id=(dossier or {}).get("dossier_id"))
        if dossier
        else []
    )
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
        "review_comments": collaboration["comments"],
        "review_summary": collaboration["review_summary"],
        "assignments": collaboration["assignments"],
        "assignment_summary": collaboration["assignment_summary"],
        "committee_decision": collaboration["committee_decision"],
        "recommendation_state": collaboration["recommendation_state"],
        "recommendation_history": collaboration["recommendation_history"],
        "exports": exports,
        "rendered_exports": rendered_exports,
        "stored_exports": stored_exports,
        "export_capabilities": export_format_capabilities("html"),
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
    stored_export_count = 0
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
        stored_export_count += len(
            store.list_stored_exports(
                dossier_id=dossier["dossier_id"],
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
    summary["stored_export_count"] = stored_export_count
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
