from __future__ import annotations

from typing import Any, Dict, List, Optional

from fastapi import HTTPException

from api.assistant.reports import sanitize_payload
from api.platform.contracts import (
    AccessDecision,
    MembershipRecord,
    ResourceRef,
    UserContext,
)
from api.platform.permissions import all_permissions, get_role_definition


DEFAULT_SYSTEM_USER_ID = "platform-system"


def build_system_user_context(
    *,
    organization_id: Optional[str] = None,
    workspace_id: Optional[str] = None,
    role: str = "service_account",
) -> UserContext:
    definition = get_role_definition(role)
    return UserContext(
        user_id=DEFAULT_SYSTEM_USER_ID,
        user_name="FTIP Platform System",
        organization_id=organization_id,
        workspace_id=workspace_id,
        role=definition.role_id,
        permissions=list(definition.permissions.permissions),
        auth_mode="development",
        is_system=True,
    )


def normalize_user_context(
    user_context: Optional[Dict[str, Any] | UserContext],
    *,
    organization_id: Optional[str] = None,
    workspace_id: Optional[str] = None,
) -> UserContext:
    if isinstance(user_context, UserContext):
        base = user_context.model_copy(deep=True)
    elif isinstance(user_context, dict):
        payload = dict(user_context)
        role = str(payload.get("role") or "service_account")
        definition = get_role_definition(role)
        base = UserContext(
            user_id=str(payload.get("user_id") or DEFAULT_SYSTEM_USER_ID),
            user_name=payload.get("user_name"),
            organization_id=payload.get("organization_id"),
            workspace_id=payload.get("workspace_id"),
            role=definition.role_id,
            permissions=list(payload.get("permissions") or definition.permissions.permissions),
            auth_mode=str(payload.get("auth_mode") or "header"),
            is_system=bool(payload.get("is_system")) or str(payload.get("user_id") or "") == DEFAULT_SYSTEM_USER_ID,
            metadata=sanitize_payload(payload.get("metadata") or {}),
        )
    else:
        base = build_system_user_context(
            organization_id=organization_id,
            workspace_id=workspace_id,
        )
    if organization_id and not base.organization_id:
        base.organization_id = organization_id
    if workspace_id and not base.workspace_id:
        base.workspace_id = workspace_id
    if not base.permissions:
        base.permissions = list(get_role_definition(base.role).permissions.permissions)
    return base


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
            metadata={"system_context": True},
        )
    if not hasattr(store, "find_membership"):
        return None
    membership = store.find_membership(
        user_id=user_context.user_id,
        organization_id=(resource.organization_id if resource else None) or user_context.organization_id,
        workspace_id=(resource.workspace_id if resource else None) or user_context.workspace_id,
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
    if not normalized.is_system and membership is None and resource and resource.workspace_id:
        allowed = False
        reason = "No active workspace membership matches the requested resource."
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
        raise HTTPException(status_code=403, detail=decision.reason)
    return decision


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
            else set(get_role_definition((membership.role if membership else normalized.role)).permissions.permissions)
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

