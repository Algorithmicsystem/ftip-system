from __future__ import annotations

import datetime as dt
from typing import Any, Dict, Iterable, List, Optional

from api.assistant.reports import sanitize_payload
from api.platform.contracts import (
    DeploymentReadinessCheck,
    DeploymentReadinessReport,
    PilotPackageSummary,
    ReadinessCategoryResult,
    ReadinessWarning,
)


def _now_utc() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _warning(
    category: str,
    message: str,
    *,
    severity: str = "warning",
    remediation: Optional[str] = None,
    blocking: bool = False,
    metadata: Optional[Dict[str, Any]] = None,
) -> ReadinessWarning:
    return ReadinessWarning(
        category=category,
        severity=severity,
        message=message,
        remediation=remediation,
        blocking=blocking,
        metadata=sanitize_payload(metadata or {}),
    )


def _check(
    check_key: str,
    label: str,
    status: str,
    summary: str,
    *,
    severity: str = "warning",
    remediation_notes: Optional[Iterable[str]] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> DeploymentReadinessCheck:
    return DeploymentReadinessCheck(
        check_key=check_key,
        label=label,
        status=status,
        summary=summary,
        severity=severity,
        remediation_notes=list(remediation_notes or []),
        metadata=sanitize_payload(metadata or {}),
    )


def _status_score(status: str) -> int:
    normalized = str(status or "partial")
    if normalized == "ready":
        return 100
    if normalized == "blocked":
        return 20
    return 60


def _category(
    category: str,
    summary: str,
    checks: List[DeploymentReadinessCheck],
    warnings: List[ReadinessWarning],
    remediation_notes: Optional[List[str]] = None,
) -> ReadinessCategoryResult:
    statuses = [str(item.status or "partial") for item in checks]
    if any(status == "blocked" for status in statuses) or any(
        warning.blocking for warning in warnings
    ):
        status = "blocked"
    elif statuses and all(status == "ready" for status in statuses) and not warnings:
        status = "ready"
    else:
        status = "partial"
    score = round(
        (
            sum(_status_score(item.status) for item in checks) / len(checks)
            if checks
            else _status_score(status)
        ),
        2,
    )
    notes: List[str] = []
    for item in remediation_notes or []:
        if item and item not in notes:
            notes.append(item)
    for warning in warnings:
        if warning.remediation and warning.remediation not in notes:
            notes.append(warning.remediation)
    return ReadinessCategoryResult(
        category=category,
        status=status,
        score=score,
        summary=summary,
        warnings=warnings[:10],
        remediation_notes=notes[:10],
        checks=checks[:10],
    )


def build_deployment_readiness_report(
    *,
    platform_version: str,
    organization: Optional[Dict[str, Any]],
    workspace: Optional[Dict[str, Any]],
    access_summary: Dict[str, Any],
    summary_view: Dict[str, Any],
    workspace_analytics: Dict[str, Any],
    health_summary: Dict[str, Any],
    integration_summary: Dict[str, Any],
    dashboard: Dict[str, Any],
    demo_snapshot: Dict[str, Any],
    dossiers: List[Dict[str, Any]],
    workflows: List[Dict[str, Any]],
    approvals: List[Dict[str, Any]],
    exports: List[Dict[str, Any]],
    stored_exports: List[Dict[str, Any]],
    audit_events: List[Dict[str, Any]],
) -> Dict[str, Any]:
    categories: List[ReadinessCategoryResult] = []

    dev_mode = bool(access_summary.get("development_mode"))
    auth_warnings: List[ReadinessWarning] = []
    if dev_mode:
        auth_warnings.append(
            _warning(
                "auth_tenant",
                "Workspace is running under development/system fallback rather than an authenticated tenant session.",
                remediation="Provision a real tenant-scoped session before external pilot use.",
                metadata={"auth_mode": access_summary.get("auth_mode")},
            )
        )
    auth_checks = [
        _check(
            "tenant_scope_present",
            "Tenant scope",
            "ready"
            if access_summary.get("tenant_scope_summary")
            and access_summary.get("tenant_scope_summary") != "unscoped_authenticated"
            else ("partial" if dev_mode else "blocked"),
            f"Tenant scope is {access_summary.get('tenant_scope_summary') or 'not established'}.",
            severity="critical" if not dev_mode and not access_summary.get("tenant_scope_summary") else "warning",
            remediation_notes=[
                "Create workspace memberships and use authenticated workspace-scoped requests."
            ],
        ),
        _check(
            "permissions_available",
            "Effective permissions",
            "ready" if (access_summary.get("effective_permissions") or []) else "blocked",
            f"{len(access_summary.get('effective_permissions') or [])} effective permissions are attached.",
            severity="critical",
            remediation_notes=["Ensure the acting user has workspace membership and assigned platform role."],
        ),
    ]
    categories.append(
        _category(
            "auth_tenant",
            "Tenant scope and effective permissions gate whether the pilot can be demonstrated safely.",
            auth_checks,
            auth_warnings,
        )
    )

    pending_approvals = sum(1 for item in approvals if str(item.get("status")) == "pending")
    workflow_warnings: List[ReadinessWarning] = []
    if pending_approvals:
        workflow_warnings.append(
            _warning(
                "workflow_state",
                f"{pending_approvals} approval request(s) remain pending.",
                remediation="Resolve or explicitly leave pending approvals in place before committee walkthroughs.",
            )
        )
    workflow_checks = [
        _check(
            "workflow_count",
            "Workflow coverage",
            "ready" if workflows else "blocked",
            f"{len(workflows)} workflow(s) exist in the current workspace scope.",
            severity="critical",
            remediation_notes=["Seed at least one workflow before the pilot walkthrough."],
        ),
        _check(
            "approval_state",
            "Approval state",
            "ready" if pending_approvals == 0 else "partial",
            f"{pending_approvals} pending approval(s) remain open.",
            remediation_notes=["Resolve pending approvals or explain why the dossier is intentionally awaiting review."],
        ),
    ]
    categories.append(
        _category(
            "workflow_state",
            "Workflow and approval state should be coherent enough for a clean institutional walkthrough.",
            workflow_checks,
            workflow_warnings,
        )
    )

    demo_seeded_count = sum(
        1 for item in dossiers if bool((item.get("metadata") or {}).get("demo_seeded"))
    )
    missing_analysis = sum(1 for item in dossiers if not item.get("latest_axiom_analysis_id"))
    dossier_warnings: List[ReadinessWarning] = []
    if missing_analysis:
        dossier_warnings.append(
            _warning(
                "dossier_state",
                f"{missing_analysis} dossier(s) have no attached AXIOM analysis.",
                remediation="Attach an AXIOM-linked report before using the dossier in a pilot.",
                blocking=True,
            )
        )
    dossier_checks = [
        _check(
            "dossier_count",
            "Dossier coverage",
            "ready" if dossiers else "blocked",
            f"{len(dossiers)} dossier(s) are available for the pilot walkthrough.",
            severity="critical",
            remediation_notes=["Seed or create at least one dossier before the pilot demo."],
        ),
        _check(
            "analysis_attachment",
            "AXIOM linkage",
            "ready" if missing_analysis == 0 and dossiers else ("blocked" if dossiers else "partial"),
            f"{len(dossiers) - missing_analysis if dossiers else 0} dossier(s) carry attached AXIOM analysis links.",
            severity="critical",
            remediation_notes=["Attach AXIOM analysis and refresh dossier summary fields."],
        ),
        _check(
            "evidence_distribution",
            "Evidence coverage",
            "ready"
            if any(str(item.get("evidence_status")) == "supportive" for item in dossiers)
            else "partial",
            f"Evidence distribution is {sanitize_payload(workspace_analytics.get('evidence_status_distribution') or {})}.",
            remediation_notes=["Improve evidence support or explain why the demo is intentionally limited."],
        ),
    ]
    categories.append(
        _category(
            "dossier_state",
            "Dossiers should carry attached AXIOM analysis, coherent evidence state, and clear recommendation context.",
            dossier_checks,
            dossier_warnings,
        )
    )

    export_warnings: List[ReadinessWarning] = []
    if not stored_exports:
        export_warnings.append(
            _warning(
                "export_state",
                "No durable stored export history is available.",
                remediation="Store at least one export pack so the pilot demonstrates durable institutional output.",
                blocking=True,
            )
        )
    export_checks = [
        _check(
            "manifest_generation",
            "Export manifests",
            "ready" if exports else "partial",
            f"{len(exports)} export manifest(s) exist in the current scope.",
            remediation_notes=["Generate at least one institutional export pack."],
        ),
        _check(
            "stored_versions",
            "Stored export versions",
            "ready" if stored_exports else "blocked",
            f"{len(stored_exports)} stored export version(s) are retrievable.",
            severity="critical",
            remediation_notes=["Render and store at least one export pack with versioned history."],
        ),
        _check(
            "integrity_warnings",
            "Export integrity warnings",
            "ready" if not (health_summary.get("export_integrity_checks") or []) else "partial",
            f"{len(health_summary.get('export_integrity_checks') or [])} export integrity warning(s) are open.",
            remediation_notes=["Resolve export integrity warnings before external pilot review."],
        ),
    ]
    categories.append(
        _category(
            "export_state",
            "Institutional pilot use requires retrievable and versioned document output.",
            export_checks,
            export_warnings,
        )
    )

    integration_bindings = int(integration_summary.get("binding_count") or 0)
    integration_warnings: List[ReadinessWarning] = []
    if integration_bindings == 0:
        integration_warnings.append(
            _warning(
                "integration_state",
                "No integration binding is configured for the workspace.",
                remediation="Seed at least an internal_sink or local_archive binding for the pilot walkthrough.",
            )
        )
    if integration_summary.get("warnings"):
        for warning in (integration_summary.get("warnings") or [])[:5]:
            integration_warnings.append(
                _warning(
                    "integration_state",
                    str(warning),
                    remediation="Resolve integration health warnings before pilot delivery.",
                )
            )
    integration_checks = [
        _check(
            "binding_count",
            "Integration bindings",
            "ready" if integration_bindings > 0 else "partial",
            f"{integration_bindings} integration binding(s) are configured.",
            remediation_notes=["Create at least one executable integration binding for demo use."],
        ),
        _check(
            "integration_health",
            "Integration health",
            "ready" if not integration_summary.get("warnings") else "partial",
            f"Integration warnings: {len(integration_summary.get('warnings') or [])}.",
            remediation_notes=["Execute a healthy integration path and clear integration warnings."],
        ),
    ]
    categories.append(
        _category(
            "integration_state",
            "Pilot integrations do not need to be broad, but at least one healthy execution path should be demonstrable.",
            integration_checks,
            integration_warnings,
        )
    )

    audit_warnings: List[ReadinessWarning] = []
    if len(audit_events) < 4:
        audit_warnings.append(
            _warning(
                "audit_state",
                "Audit coverage is thin for the current workspace scope.",
                remediation="Exercise workflow, export, and demo actions so pilot audit history is visible.",
            )
        )
    audit_checks = [
        _check(
            "audit_event_count",
            "Audit history",
            "ready" if len(audit_events) >= 4 else "partial",
            f"{len(audit_events)} audit event(s) are available for review.",
            remediation_notes=["Generate bootstrap, attachment, export, and readiness events before the demo."],
        ),
        _check(
            "health_warnings",
            "Platform warnings",
            "ready" if not (health_summary.get("warnings") or []) else "partial",
            f"{len(health_summary.get('warnings') or [])} platform warning(s) are open.",
            remediation_notes=["Resolve platform-health warnings where feasible before pilot delivery."],
        ),
    ]
    categories.append(
        _category(
            "audit_state",
            "Auditability and platform-health visibility should be sufficient for institutional review.",
            audit_checks,
            audit_warnings,
        )
    )

    ui_warnings: List[ReadinessWarning] = []
    if not dashboard:
        ui_warnings.append(
            _warning(
                "ui_console_state",
                "No dashboard payload is available.",
                remediation="Generate a dashboard payload so executive, dossier, and workflow panels render coherently.",
            )
        )
    ui_checks = [
        _check(
            "dashboard_payload",
            "Executive dashboard",
            "ready" if dashboard else "partial",
            f"Dashboard exposes {len(dashboard.get('high_dau_dossiers') or []) if dashboard else 0} high-DAU dossier(s).",
            remediation_notes=["Build a workspace dashboard snapshot before the pilot walkthrough."],
        ),
        _check(
            "demo_snapshot",
            "Demo snapshot",
            "ready" if demo_snapshot else "partial",
            f"Demo snapshot pilot_ready={demo_snapshot.get('pilot_ready') if demo_snapshot else False}.",
            remediation_notes=["Generate a demo snapshot so the workspace can be summarized quickly."],
        ),
    ]
    categories.append(
        _category(
            "ui_console_state",
            "The console should summarize dossier, export, approval, and analytics state clearly for the pilot.",
            ui_checks,
            ui_warnings,
        )
    )

    seed_warnings: List[ReadinessWarning] = []
    if demo_seeded_count == 0:
        seed_warnings.append(
            _warning(
                "pilot_seed_state",
                "No demo-seeded dossier is present in the current workspace.",
                remediation="Apply a demo bundle or create at least one dossier with attached AXIOM analysis before the pilot.",
            )
        )
    seed_checks = [
        _check(
            "demo_seed_presence",
            "Demo bundle coverage",
            "ready" if demo_seeded_count > 0 else "partial",
            f"{demo_seeded_count} dossier(s) are marked as demo_seeded.",
            remediation_notes=["Apply a deterministic demo bundle for guided pilot walkthroughs."],
        ),
        _check(
            "top_dossier_visibility",
            "Top dossier visibility",
            "ready" if (demo_snapshot.get("top_dossiers") or []) else "partial",
            f"{len(demo_snapshot.get('top_dossiers') or [])} top dossier(s) are visible in the demo snapshot.",
            remediation_notes=["Refresh the dashboard and demo snapshot after seeding the workspace."],
        ),
    ]
    categories.append(
        _category(
            "pilot_seed_state",
            "Seeded walkthrough objects should exist so the pilot can be demonstrated consistently end-to-end.",
            seed_checks,
            seed_warnings,
        )
    )

    all_warnings = [
        warning
        for category in categories
        for warning in category.warnings
    ]
    blocked = any(category.status == "blocked" for category in categories)
    ready_count = sum(1 for category in categories if category.status == "ready")
    overall_score = round(
        sum(float(category.score or _status_score(category.status)) for category in categories)
        / max(len(categories), 1),
        2,
    )
    pilot_ready = (
        not blocked
        and ready_count >= max(len(categories) - 2, 1)
        and any(category.category == "export_state" and category.status == "ready" for category in categories)
        and any(category.category == "dossier_state" and category.status == "ready" for category in categories)
    )
    overall_status = "ready" if pilot_ready else ("blocked" if blocked else "partial")

    remediation_notes: List[str] = []
    for category in categories:
        for note in category.remediation_notes:
            if note and note not in remediation_notes:
                remediation_notes.append(note)
    return DeploymentReadinessReport(
        platform_version=platform_version,
        organization_id=(organization or {}).get("organization_id"),
        workspace_id=(workspace or {}).get("workspace_id"),
        overall_status=overall_status,
        overall_score=overall_score,
        pilot_ready=pilot_ready,
        generated_at=_now_utc(),
        categories=categories,
        warnings=all_warnings[:20],
        remediation_notes=remediation_notes[:20],
        metadata={
            "workspace_name": (workspace or {}).get("name"),
            "dashboard_metric_count": len((dashboard.get("executive_metrics") or {})),
            "demo_seeded_dossier_count": demo_seeded_count,
            "pending_approval_count": pending_approvals,
            "stored_export_count": len(stored_exports),
            "audit_event_count": len(audit_events),
        },
    ).model_dump(mode="python")


def build_pilot_package_summary(
    *,
    platform_version: str,
    organization: Optional[Dict[str, Any]],
    workspace: Optional[Dict[str, Any]],
    platform_profile: Dict[str, Any],
    workflow_template: Dict[str, Any],
    summary_view: Dict[str, Any],
    workspace_analytics: Dict[str, Any],
    integration_summary: Dict[str, Any],
    readiness_report: Dict[str, Any],
    demo_snapshot: Dict[str, Any],
    dashboard: Dict[str, Any],
    dossiers: List[Dict[str, Any]],
    stored_exports: List[Dict[str, Any]],
) -> Dict[str, Any]:
    unresolved_concerns = sum(
        int((((item.get("metadata") or {}).get("review_summary") or {}).get("unresolved_concern_count")) or 0)
        for item in dossiers
    )
    locked_recommendations = sum(
        1
        for item in dossiers
        if bool((((item.get("metadata") or {}).get("recommendation_state") or {}).get("locked")))
    )
    committee_snapshot_count = sum(
        1
        for item in dossiers
        if bool((item.get("metadata") or {}).get("latest_committee_decision"))
    )
    walkthrough_hints = list(
        (((workspace or {}).get("settings") or {}).get("pilot_bootstrap_summary") or {}).get(
            "walkthrough_hints"
        )
        or []
    )
    return PilotPackageSummary(
        platform_version=platform_version,
        organization=sanitize_payload(organization or {}),
        workspace=sanitize_payload(workspace or {}),
        platform_profile=sanitize_payload(platform_profile or {}),
        workflow_template=sanitize_payload(workflow_template or {}),
        workspace_summary=sanitize_payload(
            {
                "summary_view": summary_view,
                "workspace_analytics": workspace_analytics,
                "dashboard_metrics": dashboard.get("executive_metrics") or {},
                "demo_snapshot": demo_snapshot,
            }
        ),
        top_dossiers=sanitize_payload(
            (workspace_analytics.get("high_dau_dossiers") or [])
            or (demo_snapshot.get("top_dossiers") or [])
            or (summary_view.get("latest_axiom_linked_dossiers") or [])
        )[:8],
        export_summary={
            "stored_export_count": len(stored_exports),
            "latest_stored_exports": sanitize_payload(stored_exports[:8]),
        },
        collaboration_summary={
            "unresolved_concern_count": unresolved_concerns,
            "locked_recommendation_count": locked_recommendations,
            "committee_snapshot_count": committee_snapshot_count,
        },
        integration_summary=sanitize_payload(integration_summary or {}),
        readiness_categories=[
            ReadinessCategoryResult.model_validate(item)
            for item in (readiness_report.get("categories") or [])
        ],
        top_warnings=[
            ReadinessWarning.model_validate(item)
            for item in (readiness_report.get("warnings") or [])[:8]
        ],
        walkthrough_hints=walkthrough_hints[:8],
        pilot_ready=bool(readiness_report.get("pilot_ready")),
        metadata={
            "overall_status": readiness_report.get("overall_status"),
            "overall_score": readiness_report.get("overall_score"),
            "workspace_name": (workspace or {}).get("name"),
        },
    ).model_dump(mode="python")

