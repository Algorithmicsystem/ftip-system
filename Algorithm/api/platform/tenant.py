from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional

from api.assistant.reports import sanitize_payload
from api.platform.auth import normalize_user_context
from api.platform.contracts import ResourceRef, UserContext


def allowed_organization_ids(
    user_context: Optional[Dict[str, Any] | UserContext],
) -> list[str]:
    normalized = normalize_user_context(user_context)
    return list(normalized.organization_ids)


def allowed_workspace_ids(
    user_context: Optional[Dict[str, Any] | UserContext],
) -> list[str]:
    normalized = normalize_user_context(user_context)
    return list(normalized.workspace_ids)


def tenant_scope_summary(
    user_context: Optional[Dict[str, Any] | UserContext],
) -> str:
    normalized = normalize_user_context(user_context)
    return str((normalized.metadata or {}).get("tenant_scope_summary") or "unscoped")


def resource_in_scope(
    *,
    resource: Optional[ResourceRef],
    user_context: Optional[Dict[str, Any] | UserContext],
) -> bool:
    if resource is None:
        return True
    normalized = normalize_user_context(
        user_context,
        organization_id=resource.organization_id,
        workspace_id=resource.workspace_id,
    )
    if normalized.is_system:
        return True
    workspace_ids = set(normalized.workspace_ids or [])
    organization_ids = set(normalized.organization_ids or [])
    if resource.workspace_id:
        if resource.workspace_id in workspace_ids:
            return True
        if resource.organization_id and resource.organization_id in organization_ids:
            return True
        return False
    if resource.organization_id:
        return resource.organization_id in organization_ids
    return True


def scope_records(
    records: Iterable[Dict[str, Any]],
    *,
    user_context: Optional[Dict[str, Any] | UserContext],
    organization_key: str = "organization_id",
    workspace_key: str = "workspace_id",
) -> List[Dict[str, Any]]:
    normalized = normalize_user_context(user_context)
    if normalized.is_system:
        return [sanitize_payload(item) for item in records]
    workspace_ids = set(normalized.workspace_ids or [])
    organization_ids = set(normalized.organization_ids or [])
    scoped: List[Dict[str, Any]] = []
    for item in records:
        org_id = item.get(organization_key)
        workspace_id = item.get(workspace_key)
        if workspace_id and workspace_ids and str(workspace_id) in workspace_ids:
            scoped.append(sanitize_payload(item))
            continue
        if org_id and organization_ids and str(org_id) in organization_ids:
            scoped.append(sanitize_payload(item))
            continue
    return scoped


def build_tenancy_summary(
    *,
    user_context: Optional[Dict[str, Any] | UserContext],
    accessible_workspaces: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    normalized = normalize_user_context(user_context)
    return sanitize_payload(
        {
            "development_mode": normalized.is_system,
            "auth_mode": normalized.auth_mode,
            "organization_id": normalized.organization_id,
            "workspace_id": normalized.workspace_id,
            "organization_ids": list(normalized.organization_ids),
            "workspace_ids": list(normalized.workspace_ids),
            "tenant_scope_summary": tenant_scope_summary(normalized),
            "accessible_workspace_count": len(accessible_workspaces or []),
            "accessible_workspace_ids": [
                item.get("workspace_id") for item in accessible_workspaces or []
            ],
        }
    )
