from __future__ import annotations

from typing import Any, Mapping, Optional

from api.platform.access import normalize_user_context
from api.platform.contracts import UserContext


def _split_permissions(raw: Optional[str]) -> list[str]:
    if not raw:
        return []
    return [item.strip() for item in str(raw).split(",") if item.strip()]


def user_context_from_headers(
    headers: Mapping[str, Any],
    *,
    organization_id: Optional[str] = None,
    workspace_id: Optional[str] = None,
) -> UserContext:
    payload = {
        "user_id": headers.get("x-platform-user-id") or headers.get("X-Platform-User-Id"),
        "user_name": headers.get("x-platform-user-name")
        or headers.get("X-Platform-User-Name"),
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
        "auth_mode": "header",
    }
    return normalize_user_context(
        payload if payload.get("user_id") or payload.get("role") else None,
        organization_id=organization_id,
        workspace_id=workspace_id,
    )


def actor_payload(user_context: Optional[UserContext]) -> dict[str, Any]:
    if user_context is None:
        return {}
    return {
        "user_id": user_context.user_id,
        "user_name": user_context.user_name,
        "role": user_context.role,
        "organization_id": user_context.organization_id,
        "workspace_id": user_context.workspace_id,
        "auth_mode": user_context.auth_mode,
        "is_system": user_context.is_system,
    }
