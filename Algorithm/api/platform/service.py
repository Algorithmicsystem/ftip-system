from __future__ import annotations

import uuid
from typing import Any, Dict, List, Optional

from fastapi import HTTPException

from api.assistant.reports import sanitize_payload
from api.platform.contracts import (
    AnalysisLink,
    CoverageEntity,
    DossierRecord,
    PlatformSummaryView,
    WorkflowInstance,
    WorkspaceRecord,
)
from api.platform.dossiers import (
    build_analysis_link,
    build_dossier_record,
    dossier_preview,
    refresh_dossier_record,
)
from api.platform.entities import build_coverage_entity
from api.platform.persistence import PlatformStore, platform_store
from api.platform.profiles import get_platform_profile, get_platform_profile_for_audience, list_platform_profiles
from api.platform.templates import get_workflow_template, list_workflow_templates
from api.platform.workflows import build_workflow_instance


PLATFORM_FOUNDATION_VERSION = "platform_phase5_foundation_v1"


def _normalize_symbol(symbol: str | None) -> str:
    return str(symbol or "").strip().upper()


def _default_org_name(profile_id: str) -> str:
    return f"FTIP {profile_id.replace('_', ' ').title()} Organization"


def _default_workspace_name(profile_id: str) -> str:
    return f"{profile_id.replace('_', ' ').title()} Workspace"


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
    store: PlatformStore = platform_store,
) -> Dict[str, Any]:
    profile = get_platform_profile(payload.get("platform_profile"))
    organization_id = payload.get("organization_id")
    if not organization_id:
        org = store.create_organization(
            {
                "name": payload.get("name") or _default_org_name(profile.profile_id),
                "organization_type": profile.audience_type,
                "settings": {"platform_profile": profile.profile_id},
            }
        )
        organization_id = org["organization_id"]
    workspace = store.create_workspace(
        {
            "organization_id": organization_id,
            "name": payload.get("name"),
            "audience_type": payload.get("audience_type") or profile.audience_type,
            "report_profile": payload.get("report_profile") or profile.default_report_profile,
            "default_workflow_template": payload.get("default_workflow_template")
            or profile.default_workflow_template,
            "platform_profile": profile.profile_id,
            "settings": payload.get("settings") or {},
        }
    )
    return {
        "platform_version": PLATFORM_FOUNDATION_VERSION,
        "workspace": workspace,
        "platform_profile": profile.model_dump(mode="python"),
    }


def create_workflow_service(
    payload: Dict[str, Any],
    *,
    store: PlatformStore = platform_store,
) -> Dict[str, Any]:
    workspace = store.get_workspace(payload.get("workspace_id") or "")
    if workspace is None:
        raise HTTPException(status_code=404, detail="workspace not found")
    template = get_workflow_template(payload.get("workflow_template_id") or workspace.get("default_workflow_template"))
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
    existing = store.find_coverage_entity_by_symbol(resolved_symbol) if resolved_symbol else None
    if existing is not None:
        return existing
    entity = build_coverage_entity(
        symbol=resolved_symbol,
        display_name=display_name or resolved_symbol,
        entity_type=entity_type,
        sector=sector or ((report or {}).get("data_bundle") or {}).get("symbol_meta", {}).get("sector"),
        strategy=strategy,
        theme=theme or (report or {}).get("axiom_trade_family"),
        metadata={"source": "platform_phase5"},
        entity_id=entity_id,
    )
    return store.upsert_coverage_entity(entity.model_dump(mode="python"))


def create_dossier_service(
    payload: Dict[str, Any],
    *,
    store: PlatformStore = platform_store,
) -> Dict[str, Any]:
    workflow = store.get_workflow(payload.get("workflow_id") or "")
    if workflow is None:
        raise HTTPException(status_code=404, detail="workflow not found")
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
            title=str(payload.get("title") or f"{entity.get('display_name') or entity.get('symbol')} Dossier"),
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
    return {
        "platform_version": PLATFORM_FOUNDATION_VERSION,
        "dossier": dossier,
        "workflow": workflow,
        "entity": entity,
    }


def get_dossier_view(
    dossier_id: str,
    *,
    store: PlatformStore = platform_store,
) -> Dict[str, Any]:
    dossier = store.get_dossier(dossier_id)
    if dossier is None:
        raise HTTPException(status_code=404, detail="dossier not found")
    workflow = store.get_workflow(dossier["workflow_id"])
    workspace = store.get_workspace(workflow["workspace_id"]) if workflow else None
    links = store.list_dossier_analysis_links(dossier_id)
    return {
        "platform_version": PLATFORM_FOUNDATION_VERSION,
        "dossier": dossier,
        "workflow": workflow,
        "workspace": workspace,
        "analysis_links": links,
    }


def attach_analysis_to_dossier_service(
    dossier_id: str,
    payload: Dict[str, Any],
    *,
    store: PlatformStore = platform_store,
) -> Dict[str, Any]:
    dossier = store.get_dossier(dossier_id)
    if dossier is None:
        raise HTTPException(status_code=404, detail="dossier not found")
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
    persisted = store.update_dossier(dossier_id, refreshed)
    workflow = store.get_workflow(persisted["workflow_id"])
    workspace = store.get_workspace(workflow["workspace_id"]) if workflow else None
    return {
        "platform_version": PLATFORM_FOUNDATION_VERSION,
        "dossier": persisted,
        "workflow": workflow,
        "workspace": workspace,
        "analysis_link": analysis_link.model_dump(mode="python"),
        "dossier_preview": dossier_preview(persisted),
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
                "priority": "high" if report.get("axiom_evidence_backed_deployability_tier") == "live_candidate" else "normal",
            },
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
            store=store,
        )
        dossier = dossier_view["dossier"]

    summary = build_platform_summary_service(
        workspace_id=workspace["workspace_id"] if workspace else None,
        store=store,
        current_workspace=workspace,
        current_workflow=workflow,
        current_dossier=dossier,
    )
    template = (
        get_workflow_template(str(workflow.get("workflow_template_id"))) if workflow else None
    )
    profile = get_platform_profile(platform_profile or workspace.get("platform_profile") if workspace else None)
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
    }


def build_platform_summary_service(
    *,
    workspace_id: Optional[str] = None,
    store: PlatformStore = platform_store,
    current_workspace: Optional[Dict[str, Any]] = None,
    current_workflow: Optional[Dict[str, Any]] = None,
    current_dossier: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    workspaces = store.list_workspaces()
    workflows = store.list_workflows(workspace_id=workspace_id)
    dossiers = store.list_dossiers(workspace_id=workspace_id)
    by_tier: Dict[str, int] = {}
    by_regime: Dict[str, int] = {}
    by_stage: Dict[str, int] = {}
    latest: List[Dict[str, Any]] = []
    for dossier in dossiers[:12]:
        tier = str(dossier.get("latest_deployability_tier") or "unknown")
        regime = str(dossier.get("latest_regime_label") or "unknown")
        stage = str((dossier.get("workflow_stage_state") or {}).get("stage") or "unknown")
        by_tier[tier] = by_tier.get(tier, 0) + 1
        by_regime[regime] = by_regime.get(regime, 0) + 1
        by_stage[stage] = by_stage.get(stage, 0) + 1
        latest.append(dossier_preview(dossier))
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
    )
    return summary.model_dump(mode="python")


def list_templates_service() -> Dict[str, Any]:
    return {
        "platform_version": PLATFORM_FOUNDATION_VERSION,
        "workflow_templates": [template.model_dump(mode="python") for template in list_workflow_templates()],
        "platform_profiles": [profile.model_dump(mode="python") for profile in list_platform_profiles()],
    }

