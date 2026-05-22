from __future__ import annotations

from typing import Any, Dict, Optional

from fastapi import APIRouter, Query, Request

from api.platform import service
from api.platform.contracts import (
    AttachAnalysisRequest,
    CreateDossierRequest,
    CreateIntegrationBindingRequest,
    CreateWorkflowRequest,
    CreateWorkspaceRequest,
    DossierExportRequest,
    IntegrationExecutionRequest,
    RenderExportRequest,
    StoreExportRequest,
    WorkflowActionRequest,
    WorkflowApprovalRequestPayload,
)
from api.platform.persistence import platform_store
from api.platform.security import user_context_from_headers


router = APIRouter(prefix="/platform", tags=["platform"])


def _user_context(request: Request) -> Dict[str, Any]:
    return user_context_from_headers(
        request.headers,
        store=platform_store,
    ).model_dump(mode="python")


@router.get("/templates")
def list_templates() -> Dict[str, Any]:
    return service.list_templates_service()


@router.post("/workspaces")
def create_workspace(payload: CreateWorkspaceRequest, request: Request) -> Dict[str, Any]:
    return service.create_workspace_service(
        payload.model_dump(),
        user_context=_user_context(request),
        store=platform_store,
    )


@router.get("/workspaces")
def list_workspaces(
    request: Request,
    organization_id: Optional[str] = Query(default=None),
) -> Dict[str, Any]:
    return service.list_workspaces_service(
        organization_id=organization_id,
        user_context=_user_context(request),
        store=platform_store,
    )


@router.post("/workflows")
def create_workflow(payload: CreateWorkflowRequest, request: Request) -> Dict[str, Any]:
    return service.create_workflow_service(
        payload.model_dump(),
        user_context=_user_context(request),
        store=platform_store,
    )


@router.get("/workflows")
def list_workflows(
    request: Request,
    workspace_id: Optional[str] = Query(default=None),
) -> Dict[str, Any]:
    return service.list_workflows_service(
        workspace_id=workspace_id,
        user_context=_user_context(request),
        store=platform_store,
    )


@router.post("/workflows/{workflow_id}/actions")
def workflow_action(
    workflow_id: str,
    payload: WorkflowActionRequest,
    request: Request,
) -> Dict[str, Any]:
    return service.execute_workflow_action_service(
        workflow_id,
        payload.model_dump(mode="python"),
        user_context=_user_context(request),
        store=platform_store,
    )


@router.post("/workflows/{workflow_id}/approvals")
def workflow_approval(
    workflow_id: str,
    payload: WorkflowApprovalRequestPayload,
    request: Request,
) -> Dict[str, Any]:
    return service.handle_workflow_approval_service(
        workflow_id,
        payload.model_dump(mode="python"),
        user_context=_user_context(request),
        store=platform_store,
    )


@router.get("/workflows/{workflow_id}/timeline")
def workflow_timeline(workflow_id: str, request: Request) -> Dict[str, Any]:
    return service.list_workflow_timeline_service(
        workflow_id,
        user_context=_user_context(request),
        store=platform_store,
    )


@router.post("/dossiers")
def create_dossier(payload: CreateDossierRequest, request: Request) -> Dict[str, Any]:
    return service.create_dossier_service(
        payload.model_dump(),
        user_context=_user_context(request),
        store=platform_store,
    )


@router.get("/dossiers")
def list_dossiers(
    request: Request,
    workspace_id: Optional[str] = Query(default=None),
    workflow_id: Optional[str] = Query(default=None),
) -> Dict[str, Any]:
    return service.list_dossiers_service(
        workspace_id=workspace_id,
        workflow_id=workflow_id,
        user_context=_user_context(request),
        store=platform_store,
    )


@router.get("/dossiers/{dossier_id}")
def get_dossier(dossier_id: str, request: Request) -> Dict[str, Any]:
    return service.get_dossier_view(
        dossier_id,
        user_context=_user_context(request),
        store=platform_store,
    )


@router.post("/dossiers/{dossier_id}/attach-analysis")
def attach_analysis(
    dossier_id: str,
    payload: AttachAnalysisRequest,
    request: Request,
) -> Dict[str, Any]:
    return service.attach_analysis_to_dossier_service(
        dossier_id,
        payload.model_dump(mode="python"),
        user_context=_user_context(request),
        store=platform_store,
    )


@router.post("/dossiers/{dossier_id}/export")
def dossier_export(
    dossier_id: str,
    payload: DossierExportRequest,
    request: Request,
) -> Dict[str, Any]:
    return service.create_dossier_export_service(
        dossier_id,
        payload.model_dump(mode="python"),
        user_context=_user_context(request),
        store=platform_store,
    )


@router.get("/dossiers/{dossier_id}/exports")
def dossier_exports(dossier_id: str, request: Request) -> Dict[str, Any]:
    return service.list_dossier_exports_service(
        dossier_id,
        user_context=_user_context(request),
        store=platform_store,
    )


@router.get("/dossiers/{dossier_id}/preview-export")
def preview_dossier_export(
    dossier_id: str,
    request: Request,
    pack_type: str = Query(default="dossier_pack"),
    export_format: str = Query(default="html"),
) -> Dict[str, Any]:
    return service.preview_dossier_export_service(
        dossier_id,
        pack_type=pack_type,
        export_format=export_format,
        user_context=_user_context(request),
        store=platform_store,
    )


@router.post("/dossiers/{dossier_id}/render-export")
def render_dossier_export(
    dossier_id: str,
    payload: RenderExportRequest,
    request: Request,
) -> Dict[str, Any]:
    return service.render_dossier_export_service(
        dossier_id,
        payload.model_dump(mode="python"),
        user_context=_user_context(request),
        store=platform_store,
    )


@router.post("/dossiers/{dossier_id}/store-export")
def store_dossier_export(
    dossier_id: str,
    payload: StoreExportRequest,
    request: Request,
) -> Dict[str, Any]:
    return service.store_dossier_export_service(
        dossier_id,
        payload.model_dump(mode="python"),
        user_context=_user_context(request),
        store=platform_store,
    )


@router.get("/dossiers/{dossier_id}/exports/history")
def dossier_export_history(dossier_id: str, request: Request) -> Dict[str, Any]:
    return service.list_dossier_export_history_service(
        dossier_id,
        user_context=_user_context(request),
        store=platform_store,
    )


@router.get("/exports/{render_id}")
def get_rendered_export(render_id: str, request: Request) -> Dict[str, Any]:
    return service.get_rendered_export_service(
        render_id,
        user_context=_user_context(request),
        store=platform_store,
    )


@router.get("/exports/{stored_export_id}/metadata")
def get_stored_export_metadata(
    stored_export_id: str,
    request: Request,
) -> Dict[str, Any]:
    return service.get_stored_export_metadata_service(
        stored_export_id,
        user_context=_user_context(request),
        store=platform_store,
    )


@router.get("/exports/{stored_export_id}/content")
def get_stored_export_content(
    stored_export_id: str,
    request: Request,
) -> Dict[str, Any]:
    return service.get_stored_export_content_service(
        stored_export_id,
        user_context=_user_context(request),
        store=platform_store,
    )


@router.get("/exports/{stored_export_id}/integrity")
def get_stored_export_integrity(
    stored_export_id: str,
    request: Request,
) -> Dict[str, Any]:
    return service.get_stored_export_integrity_service(
        stored_export_id,
        user_context=_user_context(request),
        store=platform_store,
    )


@router.get("/exports/{stored_export_id}/versions")
def get_stored_export_versions(
    stored_export_id: str,
    request: Request,
) -> Dict[str, Any]:
    return service.list_stored_export_versions_service(
        stored_export_id,
        user_context=_user_context(request),
        store=platform_store,
    )


@router.get("/workspaces/{workspace_id}/exports")
def list_workspace_exports(workspace_id: str, request: Request) -> Dict[str, Any]:
    return service.list_workspace_stored_exports_service(
        workspace_id,
        user_context=_user_context(request),
        store=platform_store,
    )


@router.get("/summary")
def platform_summary(
    request: Request,
    workspace_id: Optional[str] = Query(default=None),
) -> Dict[str, Any]:
    return service.build_platform_summary_service(
        workspace_id=workspace_id,
        user_context=_user_context(request),
        store=platform_store,
    )


@router.get("/dashboard")
def platform_dashboard(
    request: Request,
    workspace_id: Optional[str] = Query(default=None),
) -> Dict[str, Any]:
    return service.build_platform_dashboard_service(
        workspace_id=workspace_id,
        user_context=_user_context(request),
        store=platform_store,
        emit_audit=True,
    )


@router.get("/analytics")
def platform_analytics(
    request: Request,
    workspace_id: Optional[str] = Query(default=None),
) -> Dict[str, Any]:
    return service.build_platform_analytics_service(
        workspace_id=workspace_id,
        user_context=_user_context(request),
        store=platform_store,
    )


@router.get("/workspaces/{workspace_id}/analytics")
def workspace_analytics(workspace_id: str, request: Request) -> Dict[str, Any]:
    return service.build_workspace_analytics_service(
        workspace_id,
        user_context=_user_context(request),
        store=platform_store,
    )


@router.get("/access/summary")
def access_summary(
    request: Request,
    workspace_id: Optional[str] = Query(default=None),
) -> Dict[str, Any]:
    return service.build_access_summary_service(
        workspace_id=workspace_id,
        user_context=_user_context(request),
        store=platform_store,
    )


@router.get("/auth/session")
def auth_session(
    request: Request,
    workspace_id: Optional[str] = Query(default=None),
    organization_id: Optional[str] = Query(default=None),
) -> Dict[str, Any]:
    return service.build_auth_session_service(
        workspace_id=workspace_id,
        organization_id=organization_id,
        user_context=_user_context(request),
        store=platform_store,
    )


@router.get("/access/effective")
def effective_access(
    request: Request,
    workspace_id: Optional[str] = Query(default=None),
    organization_id: Optional[str] = Query(default=None),
) -> Dict[str, Any]:
    return service.build_effective_access_service(
        workspace_id=workspace_id,
        organization_id=organization_id,
        user_context=_user_context(request),
        store=platform_store,
    )


@router.get("/tenancy/summary")
def tenancy_summary(
    request: Request,
    workspace_id: Optional[str] = Query(default=None),
    organization_id: Optional[str] = Query(default=None),
) -> Dict[str, Any]:
    return service.build_tenancy_summary_service(
        workspace_id=workspace_id,
        organization_id=organization_id,
        user_context=_user_context(request),
        store=platform_store,
    )


@router.get("/audit")
def list_audit(
    request: Request,
    workspace_id: Optional[str] = Query(default=None),
    resource_type: Optional[str] = Query(default=None),
    resource_id: Optional[str] = Query(default=None),
) -> Dict[str, Any]:
    return service.list_audit_service(
        workspace_id=workspace_id,
        resource_type=resource_type,
        resource_id=resource_id,
        user_context=_user_context(request),
        store=platform_store,
    )


@router.get("/integrations")
def list_integrations(
    request: Request,
    workspace_id: Optional[str] = Query(default=None),
) -> Dict[str, Any]:
    return service.list_integrations_service(
        workspace_id=workspace_id,
        user_context=_user_context(request),
        store=platform_store,
    )


@router.post("/integrations")
def create_integration(
    payload: CreateIntegrationBindingRequest,
    request: Request,
) -> Dict[str, Any]:
    return service.create_integration_binding_service(
        payload.model_dump(mode="python"),
        user_context=_user_context(request),
        store=platform_store,
    )


@router.post("/integrations/{binding_id}/execute")
def execute_integration(
    binding_id: str,
    payload: IntegrationExecutionRequest,
    request: Request,
) -> Dict[str, Any]:
    return service.execute_integration_service(
        binding_id,
        payload.model_dump(mode="python"),
        user_context=_user_context(request),
        store=platform_store,
    )


@router.get("/integrations/{binding_id}/history")
def integration_history(binding_id: str, request: Request) -> Dict[str, Any]:
    return service.list_integration_history_service(
        binding_id,
        user_context=_user_context(request),
        store=platform_store,
    )


@router.get("/health")
def platform_health(
    request: Request,
    workspace_id: Optional[str] = Query(default=None),
) -> Dict[str, Any]:
    return service.build_platform_health_service(
        workspace_id=workspace_id,
        user_context=_user_context(request),
        store=platform_store,
    )


@router.get("/demo/snapshot")
def platform_demo_snapshot(
    request: Request,
    workspace_id: Optional[str] = Query(default=None),
) -> Dict[str, Any]:
    return service.build_demo_snapshot_service(
        workspace_id=workspace_id,
        user_context=_user_context(request),
        store=platform_store,
        emit_audit=True,
    )


@router.get("/demo/readiness")
def platform_demo_readiness(
    request: Request,
    workspace_id: Optional[str] = Query(default=None),
) -> Dict[str, Any]:
    return service.build_demo_readiness_service(
        workspace_id=workspace_id,
        user_context=_user_context(request),
        store=platform_store,
    )
