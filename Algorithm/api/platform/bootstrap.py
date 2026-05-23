from __future__ import annotations

import datetime as dt
from typing import Any, Dict, List, Optional

from api.assistant.reports import sanitize_payload
from api.platform.contracts import (
    DemoSeedBundle,
    PlatformProfile,
    WorkflowTemplate,
    WorkspaceBootstrapSummary,
)


def _now_utc() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def render_workspace_name(
    *,
    profile: PlatformProfile,
    organization_name: Optional[str],
    fallback_name: Optional[str] = None,
) -> str:
    organization_label = str(organization_name or "Institutional").strip() or "Institutional"
    pattern = str(
        profile.default_workspace_name_pattern
        or "{organization_name} Workspace"
    )
    try:
        rendered = pattern.format(organization_name=organization_label)
    except Exception:
        rendered = fallback_name or f"{organization_label} Workspace"
    return str(rendered or fallback_name or f"{organization_label} Workspace").strip()


def build_profile_bootstrap_defaults(
    *,
    profile: PlatformProfile,
    template: WorkflowTemplate,
    demo_bundle: Optional[DemoSeedBundle] = None,
) -> Dict[str, Any]:
    return sanitize_payload(
        {
            "profile_id": profile.profile_id,
            "audience_type": profile.audience_type,
            "default_workflow_template": profile.default_workflow_template,
            "default_report_profile": profile.default_report_profile,
            "default_workspace_name_pattern": profile.default_workspace_name_pattern,
            "default_dashboard_emphasis": profile.default_dashboard_emphasis,
            "default_export_pack_emphasis": profile.default_export_pack_emphasis,
            "preferred_axiom_sections": profile.preferred_axiom_sections,
            "preferred_dossier_sections": profile.preferred_dossier_sections,
            "pilot_bootstrap_defaults": {
                **dict(profile.pilot_bootstrap_defaults or {}),
                "workflow_template": template.template_id,
                "demo_bundle_id": (demo_bundle or {}).get("bundle_id")
                if isinstance(demo_bundle, dict)
                else (demo_bundle.bundle_id if demo_bundle else None),
            },
        }
    )


def build_workspace_bootstrap_summary(
    *,
    platform_profile: str,
    workflow_template_id: str,
    organization_created: bool,
    workspace_created: bool,
    demo_bundle_id: Optional[str],
    demo_seeded: bool,
    seeded_workflow_count: int,
    seeded_dossier_count: int,
    seeded_export_count: int,
    seeded_stored_export_count: int,
    seeded_integration_count: int,
    walkthrough_hints: List[str],
    warnings: Optional[List[str]] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    return WorkspaceBootstrapSummary(
        organization_created=organization_created,
        workspace_created=workspace_created,
        platform_profile=platform_profile,
        workflow_template_id=workflow_template_id,
        demo_bundle_id=demo_bundle_id,
        demo_seeded=demo_seeded,
        seeded_workflow_count=seeded_workflow_count,
        seeded_dossier_count=seeded_dossier_count,
        seeded_export_count=seeded_export_count,
        seeded_stored_export_count=seeded_stored_export_count,
        seeded_integration_count=seeded_integration_count,
        warnings=list(warnings or [])[:12],
        walkthrough_hints=list(walkthrough_hints or [])[:12],
        metadata=sanitize_payload(metadata or {}),
        created_at=_now_utc(),
    ).model_dump(mode="python")

