from __future__ import annotations

from typing import Dict, List, Set

from api.platform.contracts import PermissionSet, RoleDefinition


PERMISSION_NAMES: List[str] = [
    "view_workspace",
    "edit_workspace",
    "create_dossier",
    "update_dossier",
    "attach_analysis",
    "request_approval",
    "approve_stage",
    "export_report_pack",
    "manage_integrations",
    "manage_profiles",
    "view_audit_log",
]


_ROLE_DEFINITIONS: Dict[str, RoleDefinition] = {
    "org_admin": RoleDefinition(
        role_id="org_admin",
        description="Organization-level administrator for platform configuration and governance.",
        scope="organization",
        permissions=PermissionSet(permissions=PERMISSION_NAMES),
    ),
    "workspace_admin": RoleDefinition(
        role_id="workspace_admin",
        description="Workspace-level administrator with broad editing and approval control.",
        scope="workspace",
        permissions=PermissionSet(permissions=PERMISSION_NAMES),
    ),
    "analyst": RoleDefinition(
        role_id="analyst",
        description="Research user who can create and update platform work objects.",
        scope="workspace",
        permissions=PermissionSet(
            permissions=[
                "view_workspace",
                "create_dossier",
                "update_dossier",
                "attach_analysis",
                "request_approval",
                "export_report_pack",
            ]
        ),
    ),
    "reviewer": RoleDefinition(
        role_id="reviewer",
        description="Review user who can assess workflow state and approve or request changes.",
        scope="workspace",
        permissions=PermissionSet(
            permissions=[
                "view_workspace",
                "update_dossier",
                "attach_analysis",
                "request_approval",
                "approve_stage",
                "export_report_pack",
                "view_audit_log",
            ]
        ),
    ),
    "committee": RoleDefinition(
        role_id="committee",
        description="Committee user with stage-approval and export authority.",
        scope="workspace",
        permissions=PermissionSet(
            permissions=[
                "view_workspace",
                "approve_stage",
                "export_report_pack",
                "view_audit_log",
            ]
        ),
    ),
    "read_only": RoleDefinition(
        role_id="read_only",
        description="Read-only user who can inspect workspaces and audit trails.",
        scope="workspace",
        permissions=PermissionSet(
            permissions=[
                "view_workspace",
                "view_audit_log",
            ]
        ),
    ),
    "service_account": RoleDefinition(
        role_id="service_account",
        description="Default system or automation context used in development and controlled services.",
        scope="workspace",
        permissions=PermissionSet(permissions=PERMISSION_NAMES),
    ),
}


def all_permissions() -> Set[str]:
    return set(PERMISSION_NAMES)


def list_role_definitions() -> List[RoleDefinition]:
    return [definition.model_copy(deep=True) for definition in _ROLE_DEFINITIONS.values()]


def get_role_definition(role_id: str | None) -> RoleDefinition:
    role = _ROLE_DEFINITIONS.get(str(role_id or ""))
    if role is not None:
        return role.model_copy(deep=True)
    return _ROLE_DEFINITIONS["read_only"].model_copy(deep=True)

