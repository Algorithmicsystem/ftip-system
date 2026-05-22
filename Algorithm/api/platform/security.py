from __future__ import annotations

from typing import Any, Mapping, Optional

from api.platform.auth import normalize_user_context, resolve_auth_resolution
from api.platform.contracts import AuthResolution, SessionContext, UserContext


def user_context_from_headers(
    headers: Mapping[str, Any],
    *,
    organization_id: Optional[str] = None,
    workspace_id: Optional[str] = None,
    store: Any = None,
) -> UserContext:
    return resolve_auth_resolution(
        headers=headers,
        organization_id=organization_id,
        workspace_id=workspace_id,
        store=store,
    ).user_context


def resolve_request_auth(
    headers: Mapping[str, Any],
    *,
    organization_id: Optional[str] = None,
    workspace_id: Optional[str] = None,
    store: Any = None,
    request_metadata: Optional[dict[str, Any]] = None,
) -> AuthResolution:
    return resolve_auth_resolution(
        headers=headers,
        organization_id=organization_id,
        workspace_id=workspace_id,
        store=store,
        request_metadata=request_metadata,
    )


def actor_payload(
    user_context: Optional[UserContext | SessionContext | AuthResolution | dict[str, Any]],
) -> dict[str, Any]:
    if user_context is None:
        return {}
    if isinstance(user_context, AuthResolution):
        session = user_context.session
        return {
            "user_id": session.actor.user_id,
            "user_name": session.actor.user_name,
            "actor_email": session.actor.email,
            "actor_username": session.actor.username,
            "role": session.role,
            "organization_id": session.tenant.organization_id,
            "workspace_id": session.tenant.workspace_id,
            "organization_ids": list(session.tenant.organization_ids),
            "workspace_ids": list(session.tenant.workspace_ids),
            "session_id": session.session_id,
            "auth_mode": session.auth_mode,
            "is_system": session.is_system,
            "tenant_scope_summary": session.tenant.scope_summary,
        }
    normalized = normalize_user_context(user_context)
    return {
        "user_id": normalized.user_id,
        "user_name": normalized.user_name,
        "actor_email": normalized.email,
        "actor_username": normalized.username,
        "role": normalized.role,
        "organization_id": normalized.organization_id,
        "workspace_id": normalized.workspace_id,
        "organization_ids": list(normalized.organization_ids),
        "workspace_ids": list(normalized.workspace_ids),
        "session_id": normalized.session_id,
        "auth_mode": normalized.auth_mode,
        "is_system": normalized.is_system,
        "tenant_scope_summary": (normalized.metadata or {}).get(
            "tenant_scope_summary"
        ),
    }
