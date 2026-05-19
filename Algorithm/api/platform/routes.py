from __future__ import annotations

from typing import Any, Dict, Optional

from fastapi import APIRouter, HTTPException, Query

from api.platform import service
from api.platform.contracts import (
    AttachAnalysisRequest,
    CreateDossierRequest,
    CreateWorkflowRequest,
    CreateWorkspaceRequest,
)
from api.platform.persistence import platform_store


router = APIRouter(prefix="/platform", tags=["platform"])


@router.get("/templates")
def list_templates() -> Dict[str, Any]:
    return service.list_templates_service()


@router.post("/workspaces")
def create_workspace(payload: CreateWorkspaceRequest) -> Dict[str, Any]:
    return service.create_workspace_service(payload.model_dump(), store=platform_store)


@router.get("/workspaces")
def list_workspaces() -> Dict[str, Any]:
    return {
        "platform_version": service.PLATFORM_FOUNDATION_VERSION,
        "workspaces": platform_store.list_workspaces(),
    }


@router.post("/workflows")
def create_workflow(payload: CreateWorkflowRequest) -> Dict[str, Any]:
    return service.create_workflow_service(payload.model_dump(), store=platform_store)


@router.get("/workflows")
def list_workflows(workspace_id: Optional[str] = Query(default=None)) -> Dict[str, Any]:
    return {
        "platform_version": service.PLATFORM_FOUNDATION_VERSION,
        "workflows": platform_store.list_workflows(workspace_id=workspace_id),
    }


@router.post("/dossiers")
def create_dossier(payload: CreateDossierRequest) -> Dict[str, Any]:
    return service.create_dossier_service(payload.model_dump(), store=platform_store)


@router.get("/dossiers")
def list_dossiers(
    workspace_id: Optional[str] = Query(default=None),
    workflow_id: Optional[str] = Query(default=None),
) -> Dict[str, Any]:
    return {
        "platform_version": service.PLATFORM_FOUNDATION_VERSION,
        "dossiers": platform_store.list_dossiers(
            workspace_id=workspace_id,
            workflow_id=workflow_id,
        ),
    }


@router.get("/dossiers/{dossier_id}")
def get_dossier(dossier_id: str) -> Dict[str, Any]:
    return service.get_dossier_view(dossier_id, store=platform_store)


@router.post("/dossiers/{dossier_id}/attach-analysis")
def attach_analysis(dossier_id: str, payload: AttachAnalysisRequest) -> Dict[str, Any]:
    return service.attach_analysis_to_dossier_service(
        dossier_id,
        payload.model_dump(mode="python"),
        store=platform_store,
    )


@router.get("/summary")
def platform_summary(workspace_id: Optional[str] = Query(default=None)) -> Dict[str, Any]:
    return service.build_platform_summary_service(
        workspace_id=workspace_id,
        store=platform_store,
    )

