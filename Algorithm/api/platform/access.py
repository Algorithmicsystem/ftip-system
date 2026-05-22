from __future__ import annotations

from typing import Any, Dict, Optional

from fastapi import HTTPException

from api.assistant.reports import sanitize_payload
from api.platform.auth import normalize_user_context
from api.platform.contracts import AccessDecision, MembershipRecord, ResourceRef, UserContext
from api.platform.permissions import all_permissions, get_role_definition
from api.platform.tenant import resource_in_scope, tenant_scope_summary


def resolve_membership(
    *,
    user_context: UserContext,
    resource: Optional[ResourceRef],
    store: Any,
) -> Optional[MembershipRecord]:
    if user_context.is_system:
        return MembershipRecord(
            membership_id="system-membership",
            user_id=user_context.user_id,
            organization_id=resource.organization_id if resource else user_context.organization_id,
            workspace_id=resource.workspace_id if resource else user_context.workspace_id,
            role=user_context.role,
            permissions=list(user_context.permissions or all_permissions()),
            status="active",
            metadata={
                "system_context": True,
                "fallback_used": bool((user_context.metadata or {}).get("fallback_used")),
            },
        )
    if not hasattr(store, "find_membership"):
        return None
    membership = store.find_membership(
        user_id=user_context.user_id,
        organization_id=(resource.organization_id if resource else None)
        or user_context.organization_id,
        workspace_id=(resource.workspace_id if resource else None)
        or user_context.workspace_id,
        include_org_fallback=True,
    )
    if membership is None:
        return None
    return MembershipRecord.model_validate(membership)


def evaluate_access(
    *,
    permission: str,
    user_context: Optional[Dict[str, Any] | UserContext],
    resource: Optional[ResourceRef],
    store: Any,
) -> AccessDecision:
    normalized = normalize_user_context(
        user_context,
        organization_id=resource.organization_id if resource else None,
        workspace_id=resource.workspace_id if resource else None,
    )
    membership = resolve_membership(user_context=normalized, resource=resource, store=store)
    effective_permissions = set(normalized.permissions or [])
    if membership is not None:
        effective_permissions.update(membership.permissions)
        role = membership.role
        enforcement_mode = "membership"
    elif normalized.is_system:
        effective_permissions = set(all_permissions())
        role = normalized.role
        enforcement_mode = "system_default"
    else:
        role = normalized.role
        effective_permissions.update(get_role_definition(role).permissions.permissions)
        enforcement_mode = "role_fallback"

    allowed = permission in effective_permissions
    reason = (
        f"Permission {permission} granted via {role}."
        if allowed
        else f"Permission {permission} is not available for role {role}."
    )
    if not normalized.is_system and not resource_in_scope(
        resource=resource,
        user_context=normalized,
    ):
        allowed = False
        reason = "Tenant scope does not include the requested resource."
    elif not normalized.is_system and membership is None and resource and (
        resource.workspace_id or resource.organization_id
    ):
        allowed = False
        reason = "No active membership matches the requested organization or workspace."
    return AccessDecision(
        allowed=allowed,
        permission=permission,
        role=role,
        reason=reason,
        enforcement_mode=enforcement_mode,
        permissions=sorted(effective_permissions),
        missing_permissions=[] if allowed else [permission],
        membership=membership,
        resource=resource,
        organization_id=(resource.organization_id if resource else None)
        or normalized.organization_id,
        workspace_id=(resource.workspace_id if resource else None) or normalized.workspace_id,
        auth_mode=normalized.auth_mode,
        actor_user_id=normalized.user_id,
        tenant_scope_summary=tenant_scope_summary(normalized),
    )


def _deny(detail: AccessDecision) -> HTTPException:
    return HTTPException(
        status_code=403,
        detail=sanitize_payload(
            {
                "error": "access_denied",
                "reason": detail.reason,
                "permission": detail.permission,
                "role": detail.role,
                "resource": detail.resource.model_dump(mode="python")
                if detail.resource
                else None,
                "tenant_scope_summary": detail.tenant_scope_summary,
                "missing_permissions": detail.missing_permissions,
                "enforcement_mode": detail.enforcement_mode,
            }
        ),
    )


def require_access(
    *,
    permission: str,
    user_context: Optional[Dict[str, Any] | UserContext],
    resource: Optional[ResourceRef],
    store: Any,
) -> AccessDecision:
    decision = evaluate_access(
        permission=permission,
        user_context=user_context,
        resource=resource,
        store=store,
    )
    if not decision.allowed:
        raise _deny(decision)
    return decision


def require_permission(
    *,
    permission: str,
    user_context: Optional[Dict[str, Any] | UserContext],
    resource: Optional[ResourceRef],
    store: Any,
) -> AccessDecision:
    return require_access(
        permission=permission,
        user_context=user_context,
        resource=resource,
        store=store,
    )


def require_workspace_access(
    *,
    workspace_id: str,
    organization_id: Optional[str] = None,
    permission: str = "view_workspace",
    user_context: Optional[Dict[str, Any] | UserContext],
    store: Any,
) -> AccessDecision:
    return require_access(
        permission=permission,
        user_context=user_context,
        resource=ResourceRef(
            resource_type="workspace",
            workspace_id=workspace_id,
            organization_id=organization_id,
            resource_id=workspace_id,
        ),
        store=store,
    )


def require_org_access(
    *,
    organization_id: str,
    permission: str = "view_workspace",
    user_context: Optional[Dict[str, Any] | UserContext],
    store: Any,
) -> AccessDecision:
    return require_access(
        permission=permission,
        user_context=user_context,
        resource=ResourceRef(
            resource_type="organization",
            organization_id=organization_id,
            resource_id=organization_id,
        ),
        store=store,
    )


def scope_resource_query(
    *,
    user_context: Optional[Dict[str, Any] | UserContext],
    requested_workspace_id: Optional[str] = None,
    requested_organization_id: Optional[str] = None,
    store: Any,
) -> Dict[str, Any]:
    normalized = normalize_user_context(
        user_context,
    )
    if normalized.is_system:
        return {
            "user_context": normalized,
            "organization_ids": None,
            "workspace_ids": None,
            "requested_workspace_id": requested_workspace_id,
            "requested_organization_id": requested_organization_id,
            "has_access": True,
        }
    memberships = []
    if hasattr(store, "list_memberships"):
        memberships = store.list_memberships(user_id=normalized.user_id)
    organization_ids = sorted(
        {
            str(item.get("organization_id"))
            for item in memberships
            if item.get("organization_id")
        }
        | set(normalized.organization_ids or [])
    )
    workspace_ids = sorted(
        {
            str(item.get("workspace_id"))
            for item in memberships
            if item.get("workspace_id")
        }
        | set(normalized.workspace_ids or [])
    )
    if requested_workspace_id and (
        not workspace_ids or requested_workspace_id not in workspace_ids
    ):
        # Allow org-level access only when the requested workspace belongs to one of the user's orgs.
        workspace = store.get_workspace(requested_workspace_id) if hasattr(store, "get_workspace") else None
        workspace_org = (workspace or {}).get("organization_id")
        if not workspace_org or workspace_org not in organization_ids:
            raise _deny(
                AccessDecision(
                    allowed=False,
                    permission="view_workspace",
                    role=normalized.role,
                    reason="Requested workspace is outside the authenticated tenant scope.",
                    enforcement_mode="tenant_scope",
                    permissions=sorted(set(normalized.permissions or [])),
                    missing_permissions=["view_workspace"],
                    resource=ResourceRef(
                        resource_type="workspace",
                        resource_id=requested_workspace_id,
                        workspace_id=requested_workspace_id,
                        organization_id=workspace_org,
                    ),
                    organization_id=workspace_org,
                    workspace_id=requested_workspace_id,
                    auth_mode=normalized.auth_mode,
                    actor_user_id=normalized.user_id,
                    tenant_scope_summary=tenant_scope_summary(normalized),
                )
            )
    if requested_organization_id and (
        not organization_ids or requested_organization_id not in organization_ids
    ):
        raise _deny(
            AccessDecision(
                allowed=False,
                permission="view_workspace",
                role=normalized.role,
                reason="Requested organization is outside the authenticated tenant scope.",
                enforcement_mode="tenant_scope",
                permissions=sorted(set(normalized.permissions or [])),
                missing_permissions=["view_workspace"],
                resource=ResourceRef(
                    resource_type="organization",
                    resource_id=requested_organization_id,
                    organization_id=requested_organization_id,
                ),
                organization_id=requested_organization_id,
                auth_mode=normalized.auth_mode,
                actor_user_id=normalized.user_id,
                tenant_scope_summary=tenant_scope_summary(normalized),
            )
        )
    if requested_workspace_id:
        workspace_ids = [requested_workspace_id]
    if requested_organization_id and requested_organization_id not in organization_ids:
        organization_ids.append(requested_organization_id)
    normalized.organization_ids = list(organization_ids)
    normalized.workspace_ids = list(workspace_ids)
    if requested_organization_id:
        normalized.organization_id = requested_organization_id
    elif len(organization_ids) == 1:
        normalized.organization_id = organization_ids[0]
    if requested_workspace_id:
        normalized.workspace_id = requested_workspace_id
    elif len(workspace_ids) == 1:
        normalized.workspace_id = workspace_ids[0]
    if workspace_ids:
        normalized.metadata["tenant_scope_summary"] = (
            f"workspace_scoped:{len(workspace_ids)}"
        )
    elif organization_ids:
        normalized.metadata["tenant_scope_summary"] = (
            f"organization_scoped:{len(organization_ids)}"
        )
    else:
        normalized.metadata["tenant_scope_summary"] = "unscoped_authenticated"
    return {
        "user_context": normalized,
        "organization_ids": organization_ids,
        "workspace_ids": workspace_ids,
        "requested_workspace_id": requested_workspace_id,
        "requested_organization_id": requested_organization_id,
        "has_access": bool(organization_ids or workspace_ids),
    }


def build_access_summary(
    *,
    user_context: Optional[Dict[str, Any] | UserContext],
    resource: Optional[ResourceRef],
    store: Any,
) -> Dict[str, Any]:
    normalized = normalize_user_context(
        user_context,
        organization_id=resource.organization_id if resource else None,
        workspace_id=resource.workspace_id if resource else None,
    )
    membership = resolve_membership(user_context=normalized, resource=resource, store=store)
    permissions = sorted(
        set(normalized.permissions or [])
        | set((membership.permissions if membership else []) or [])
        | (
            set(all_permissions())
            if normalized.is_system
            else set(
                get_role_definition(
                    (membership.role if membership else normalized.role)
                ).permissions.permissions
            )
        )
    )
    return sanitize_payload(
        {
            "user_context": normalized.model_dump(mode="python"),
            "membership": membership.model_dump(mode="python") if membership else None,
            "effective_role": membership.role if membership else normalized.role,
            "effective_permissions": permissions,
            "resource": resource.model_dump(mode="python") if resource else None,
            "development_mode": normalized.is_system,
            "tenant_scope_summary": tenant_scope_summary(normalized),
            "organization_ids": list(normalized.organization_ids),
            "workspace_ids": list(normalized.workspace_ids),
            "auth_mode": normalized.auth_mode,
            "session_id": normalized.session_id,
        }
    )


def workspace_accessible(
    *,
    workspace_id: Optional[str],
    user_context: Optional[Dict[str, Any] | UserContext],
    store: Any,
) -> bool:
    normalized = normalize_user_context(user_context, workspace_id=workspace_id)
    if normalized.is_system:
        return True
    membership = resolve_membership(
        user_context=normalized,
        resource=ResourceRef(resource_type="workspace", workspace_id=workspace_id),
        store=store,
    )
    return membership is not None and membership.status == "active"
