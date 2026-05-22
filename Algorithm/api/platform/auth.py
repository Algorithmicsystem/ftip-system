from __future__ import annotations

from typing import Any, Mapping, Optional

from api.assistant.reports import sanitize_payload
from api.platform.contracts import (
    AuthResolution,
    AuthenticatedUser,
    MembershipRecord,
    SessionContext,
    TenantContext,
    UserContext,
)
from api.platform.permissions import get_role_definition


DEFAULT_SYSTEM_USER_ID = "platform-system"


def _split_permissions(raw: Optional[str]) -> list[str]:
    if not raw:
        return []
    return [item.strip() for item in str(raw).split(",") if item.strip()]


def _string_list(values: list[str] | None) -> list[str]:
    return sorted({str(value) for value in values or [] if str(value).strip()})


def _scope_summary(
    *,
    organization_ids: list[str],
    workspace_ids: list[str],
    fallback_used: bool,
) -> str:
    if fallback_used:
        return "development_system_fallback"
    if workspace_ids:
        return f"workspace_scoped:{len(workspace_ids)}"
    if organization_ids:
        return f"organization_scoped:{len(organization_ids)}"
    return "unscoped_authenticated"


def build_system_user_context(
    *,
    organization_id: Optional[str] = None,
    workspace_id: Optional[str] = None,
    role: str = "service_account",
    session_id: Optional[str] = None,
    request_metadata: Optional[dict[str, Any]] = None,
) -> UserContext:
    definition = get_role_definition(role)
    organization_ids = [organization_id] if organization_id else []
    workspace_ids = [workspace_id] if workspace_id else []
    return UserContext(
        user_id=DEFAULT_SYSTEM_USER_ID,
        user_name="FTIP Platform System",
        organization_id=organization_id,
        workspace_id=workspace_id,
        organization_ids=organization_ids,
        workspace_ids=workspace_ids,
        role=definition.role_id,
        permissions=list(definition.permissions.permissions),
        auth_mode="development",
        is_system=True,
        session_id=session_id,
        request_metadata=sanitize_payload(request_metadata or {}),
        metadata={
            "fallback_used": True,
            "tenant_scope_summary": _scope_summary(
                organization_ids=organization_ids,
                workspace_ids=workspace_ids,
                fallback_used=True,
            ),
        },
    )


def build_system_auth_resolution(
    *,
    organization_id: Optional[str] = None,
    workspace_id: Optional[str] = None,
    role: str = "service_account",
    session_id: Optional[str] = None,
    request_metadata: Optional[dict[str, Any]] = None,
) -> AuthResolution:
    user_context = build_system_user_context(
        organization_id=organization_id,
        workspace_id=workspace_id,
        role=role,
        session_id=session_id,
        request_metadata=request_metadata,
    )
    session = SessionContext(
        session_id=session_id,
        actor=AuthenticatedUser(
            user_id=user_context.user_id,
            user_name=user_context.user_name,
            auth_mode=user_context.auth_mode,
            is_system=True,
        ),
        tenant=TenantContext(
            organization_id=organization_id,
            workspace_id=workspace_id,
            organization_ids=list(user_context.organization_ids),
            workspace_ids=list(user_context.workspace_ids),
            role_bindings=[],
            is_scoped=bool(organization_id or workspace_id),
            fallback_used=True,
            scope_summary=str(
                (user_context.metadata or {}).get("tenant_scope_summary")
                or "development_system_fallback"
            ),
        ),
        role=user_context.role,
        permissions=list(user_context.permissions),
        auth_mode="development",
        is_system=True,
        request_metadata=sanitize_payload(request_metadata or {}),
    )
    return AuthResolution(
        session=session,
        user_context=user_context,
        auth_headers_present=False,
        fallback_used=True,
        enforcement_mode="development_fallback",
        effective_membership=None,
    )


def normalize_user_context(
    user_context: Optional[dict[str, Any] | UserContext | SessionContext | AuthResolution],
    *,
    organization_id: Optional[str] = None,
    workspace_id: Optional[str] = None,
) -> UserContext:
    if isinstance(user_context, AuthResolution):
        base = user_context.user_context.model_copy(deep=True)
    elif isinstance(user_context, SessionContext):
        definition = get_role_definition(user_context.role)
        base = UserContext(
            user_id=user_context.actor.user_id,
            user_name=user_context.actor.user_name,
            email=user_context.actor.email,
            username=user_context.actor.username,
            organization_id=user_context.tenant.organization_id,
            workspace_id=user_context.tenant.workspace_id,
            organization_ids=list(user_context.tenant.organization_ids),
            workspace_ids=list(user_context.tenant.workspace_ids),
            role=definition.role_id,
            permissions=list(user_context.permissions or definition.permissions.permissions),
            auth_mode=user_context.auth_mode,
            is_system=user_context.is_system,
            session_id=user_context.session_id,
            role_bindings=sanitize_payload(user_context.tenant.role_bindings),
            request_metadata=sanitize_payload(user_context.request_metadata),
            metadata={
                "fallback_used": bool(user_context.tenant.fallback_used),
                "tenant_scope_summary": user_context.tenant.scope_summary,
            },
        )
    elif isinstance(user_context, UserContext):
        base = user_context.model_copy(deep=True)
    elif isinstance(user_context, dict):
        payload = dict(user_context)
        role = str(payload.get("role") or "service_account")
        definition = get_role_definition(role)
        organization_ids = _string_list(
            list(payload.get("organization_ids") or [])
            + ([payload.get("organization_id")] if payload.get("organization_id") else [])
        )
        workspace_ids = _string_list(
            list(payload.get("workspace_ids") or [])
            + ([payload.get("workspace_id")] if payload.get("workspace_id") else [])
        )
        fallback_used = bool(payload.get("is_system")) or str(
            payload.get("user_id") or ""
        ) == DEFAULT_SYSTEM_USER_ID
        base = UserContext(
            user_id=str(payload.get("user_id") or DEFAULT_SYSTEM_USER_ID),
            user_name=payload.get("user_name"),
            email=payload.get("email"),
            username=payload.get("username"),
            organization_id=payload.get("organization_id"),
            workspace_id=payload.get("workspace_id"),
            organization_ids=organization_ids,
            workspace_ids=workspace_ids,
            role=definition.role_id,
            permissions=list(
                payload.get("permissions") or definition.permissions.permissions
            ),
            auth_mode=str(payload.get("auth_mode") or "header"),
            is_system=fallback_used,
            session_id=payload.get("session_id"),
            role_bindings=sanitize_payload(payload.get("role_bindings") or []),
            request_metadata=sanitize_payload(payload.get("request_metadata") or {}),
            metadata=sanitize_payload(
                {
                    **dict(payload.get("metadata") or {}),
                    "fallback_used": bool(
                        (payload.get("metadata") or {}).get("fallback_used")
                    )
                    or fallback_used,
                    "tenant_scope_summary": (
                        (payload.get("metadata") or {}).get("tenant_scope_summary")
                        or _scope_summary(
                            organization_ids=organization_ids,
                            workspace_ids=workspace_ids,
                            fallback_used=fallback_used,
                        )
                    ),
                }
            ),
        )
    else:
        return build_system_user_context(
            organization_id=organization_id,
            workspace_id=workspace_id,
        )
    if organization_id and not base.organization_id:
        base.organization_id = organization_id
    if workspace_id and not base.workspace_id:
        base.workspace_id = workspace_id
    if organization_id and organization_id not in base.organization_ids:
        base.organization_ids.append(organization_id)
    if workspace_id and workspace_id not in base.workspace_ids:
        base.workspace_ids.append(workspace_id)
    base.organization_ids = _string_list(base.organization_ids)
    base.workspace_ids = _string_list(base.workspace_ids)
    if not base.permissions:
        base.permissions = list(get_role_definition(base.role).permissions.permissions)
    base.metadata["tenant_scope_summary"] = _scope_summary(
        organization_ids=base.organization_ids,
        workspace_ids=base.workspace_ids,
        fallback_used=bool(base.is_system),
    )
    return base


def _payload_from_headers(
    headers: Mapping[str, Any],
    *,
    organization_id: Optional[str],
    workspace_id: Optional[str],
) -> tuple[dict[str, Any], bool]:
    payload = {
        "user_id": headers.get("x-platform-user-id") or headers.get("X-Platform-User-Id"),
        "user_name": headers.get("x-platform-user-name")
        or headers.get("X-Platform-User-Name"),
        "email": headers.get("x-platform-user-email")
        or headers.get("X-Platform-User-Email"),
        "username": headers.get("x-platform-username")
        or headers.get("X-Platform-Username"),
        "organization_id": headers.get("x-platform-organization-id")
        or headers.get("X-Platform-Organization-Id")
        or organization_id,
        "workspace_id": headers.get("x-platform-workspace-id")
        or headers.get("X-Platform-Workspace-Id")
        or workspace_id,
        "role": headers.get("x-platform-role") or headers.get("X-Platform-Role"),
        "permissions": _split_permissions(
            headers.get("x-platform-permissions")
            or headers.get("X-Platform-Permissions")
        ),
        "session_id": headers.get("x-platform-session-id")
        or headers.get("X-Platform-Session-Id"),
        "auth_mode": headers.get("x-platform-auth-mode")
        or headers.get("X-Platform-Auth-Mode")
        or "header",
    }
    headers_present = bool(
        payload.get("user_id")
        or payload.get("role")
        or payload.get("session_id")
        or payload.get("organization_id")
        or payload.get("workspace_id")
    )
    return payload, headers_present


def resolve_auth_resolution(
    *,
    headers: Optional[Mapping[str, Any]] = None,
    user_context: Optional[dict[str, Any] | UserContext | SessionContext | AuthResolution] = None,
    store: Any = None,
    organization_id: Optional[str] = None,
    workspace_id: Optional[str] = None,
    request_metadata: Optional[dict[str, Any]] = None,
) -> AuthResolution:
    if isinstance(user_context, AuthResolution):
        return user_context
    payload: dict[str, Any]
    headers_present = False
    if headers is not None:
        payload, headers_present = _payload_from_headers(
            headers,
            organization_id=organization_id,
            workspace_id=workspace_id,
        )
    elif isinstance(user_context, dict):
        payload = dict(user_context)
    elif isinstance(user_context, UserContext):
        payload = user_context.model_dump(mode="python")
    elif isinstance(user_context, SessionContext):
        payload = normalize_user_context(
            user_context,
            organization_id=organization_id,
            workspace_id=workspace_id,
        ).model_dump(mode="python")
    else:
        payload = {}

    auth_present = bool(payload.get("user_id") or payload.get("role"))
    if not auth_present:
        return build_system_auth_resolution(
            organization_id=organization_id or payload.get("organization_id"),
            workspace_id=workspace_id or payload.get("workspace_id"),
            session_id=payload.get("session_id"),
            request_metadata=request_metadata,
        )

    role = str(payload.get("role") or "read_only")
    definition = get_role_definition(role)
    actor = AuthenticatedUser(
        user_id=str(payload.get("user_id") or DEFAULT_SYSTEM_USER_ID),
        user_name=payload.get("user_name"),
        email=payload.get("email"),
        username=payload.get("username"),
        auth_mode=str(payload.get("auth_mode") or "header"),
        is_system=False,
    )
    memberships: list[MembershipRecord] = []
    if store is not None and hasattr(store, "list_memberships") and actor.user_id:
        membership_rows = store.list_memberships(user_id=actor.user_id)
        memberships = [
            MembershipRecord.model_validate(item)
            for item in membership_rows
            if str(item.get("status") or "active") == "active"
        ]
    organization_ids = _string_list(
        [membership.organization_id for membership in memberships if membership.organization_id]
        + list(payload.get("organization_ids") or [])
        + ([payload.get("organization_id")] if payload.get("organization_id") else [])
    )
    workspace_ids = _string_list(
        [membership.workspace_id for membership in memberships if membership.workspace_id]
        + list(payload.get("workspace_ids") or [])
        + ([payload.get("workspace_id")] if payload.get("workspace_id") else [])
    )
    requested_organization_id = organization_id or payload.get("organization_id")
    requested_workspace_id = workspace_id or payload.get("workspace_id")
    tenant = TenantContext(
        organization_id=requested_organization_id
        or (organization_ids[0] if len(organization_ids) == 1 else None),
        workspace_id=requested_workspace_id
        or (workspace_ids[0] if len(workspace_ids) == 1 else None),
        organization_ids=organization_ids,
        workspace_ids=workspace_ids,
        role_bindings=[item.model_dump(mode="python") for item in memberships],
        is_scoped=bool(organization_ids or workspace_ids),
        fallback_used=False,
        scope_summary=_scope_summary(
            organization_ids=organization_ids,
            workspace_ids=workspace_ids,
            fallback_used=False,
        ),
    )
    user_ctx = UserContext(
        user_id=actor.user_id,
        user_name=actor.user_name,
        email=actor.email,
        username=actor.username,
        organization_id=tenant.organization_id,
        workspace_id=tenant.workspace_id,
        organization_ids=organization_ids,
        workspace_ids=workspace_ids,
        role=definition.role_id,
        permissions=list(payload.get("permissions") or definition.permissions.permissions),
        auth_mode=actor.auth_mode,
        is_system=False,
        session_id=payload.get("session_id"),
        role_bindings=[item.model_dump(mode="python") for item in memberships],
        request_metadata=sanitize_payload(request_metadata or {}),
        metadata={
            "fallback_used": False,
            "tenant_scope_summary": tenant.scope_summary,
        },
    )
    effective_membership = None
    if requested_workspace_id or requested_organization_id:
        for membership in memberships:
            if requested_workspace_id and membership.workspace_id == requested_workspace_id:
                effective_membership = membership
                break
            if (
                requested_organization_id
                and membership.organization_id == requested_organization_id
                and membership.workspace_id is None
            ):
                effective_membership = membership
    session = SessionContext(
        session_id=user_ctx.session_id,
        actor=actor,
        tenant=tenant,
        role=effective_membership.role if effective_membership else definition.role_id,
        permissions=list(user_ctx.permissions),
        auth_mode=actor.auth_mode,
        is_system=False,
        request_metadata=sanitize_payload(request_metadata or {}),
    )
    return AuthResolution(
        session=session,
        user_context=user_ctx,
        auth_headers_present=headers_present,
        fallback_used=False,
        enforcement_mode="membership" if memberships else "header_unbound",
        effective_membership=effective_membership,
    )
