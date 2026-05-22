from __future__ import annotations

from fastapi.testclient import TestClient

from api.main import app
from api.platform import service as platform_service
from api.platform.persistence import PlatformStore, platform_store
from tests.test_platform_phase5 import _reset_platform_store, _sample_platform_report
from tests.test_platform_phase6 import _bootstrap_workspace


def _auth_headers(
    user_id: str,
    role: str,
    *,
    workspace_id: str | None = None,
    organization_id: str | None = None,
    email: str | None = None,
    session_id: str | None = None,
) -> dict[str, str]:
    headers = {
        "X-Platform-User-Id": user_id,
        "X-Platform-Role": role,
    }
    if workspace_id:
        headers["X-Platform-Workspace-Id"] = workspace_id
    if organization_id:
        headers["X-Platform-Organization-Id"] = organization_id
    if email:
        headers["X-Platform-User-Email"] = email
    if session_id:
        headers["X-Platform-Session-Id"] = session_id
    return headers


def _create_second_workspace_same_org(store: PlatformStore, organization_id: str) -> dict:
    return platform_service.create_workspace_service(
        {
            "organization_id": organization_id,
            "name": "HF Secondary Workspace",
            "platform_profile": "hf_core",
            "audience_type": "hedge_fund",
            "report_profile": "ic_memo",
        },
        store=store,
    )["workspace"]


def _create_workflow_and_dossier(
    store: PlatformStore,
    *,
    workspace_id: str,
    symbol: str,
    title_prefix: str,
) -> tuple[dict, dict]:
    workflow = platform_service.create_workflow_service(
        {
            "workspace_id": workspace_id,
            "workflow_template_id": "hedge_fund_research",
            "title": f"{title_prefix} Workflow",
        },
        store=store,
    )["workflow"]
    dossier = platform_service.create_dossier_service(
        {
            "workflow_id": workflow["workflow_id"],
            "symbol": symbol,
            "display_name": symbol,
            "title": f"{title_prefix} Dossier",
        },
        store=store,
    )["dossier"]
    return workflow, dossier


def test_phase8a_builds_explicit_session_and_dev_fallback_foundation() -> None:
    store = PlatformStore(use_memory=True)
    workspace, _, _ = _bootstrap_workspace(store)

    fallback = platform_service.build_auth_session_service(
        workspace_id=workspace["workspace_id"],
        store=store,
    )
    assert fallback["session"]["is_system"] is True
    assert fallback["session"]["auth_mode"] == "development"
    assert fallback["tenancy_summary"]["development_mode"] is True

    platform_service.create_membership_service(
        {
            "user_id": "analyst-1",
            "workspace_id": workspace["workspace_id"],
            "organization_id": workspace["organization_id"],
            "role": "analyst",
        },
        store=store,
    )
    resolved = platform_service.build_auth_session_service(
        workspace_id=workspace["workspace_id"],
        user_context={
            "user_id": "analyst-1",
            "role": "analyst",
            "auth_mode": "header",
            "email": "analyst@example.com",
            "session_id": "session-1",
        },
        store=store,
    )
    assert resolved["session"]["is_system"] is False
    assert resolved["session"]["auth_mode"] == "header"
    assert resolved["session"]["workspace_ids"] == [workspace["workspace_id"]]
    assert resolved["session"]["organization_ids"] == [workspace["organization_id"]]
    assert resolved["tenancy_summary"]["accessible_workspace_count"] == 1


def test_phase8a_scopes_workspace_and_dossier_queries_by_tenant() -> None:
    store = PlatformStore(use_memory=True)
    workspace_one, _, _ = _bootstrap_workspace(store)
    workspace_two_response = platform_service.create_workspace_service(
        {
            "name": "PE Tenant Workspace",
            "platform_profile": "pe_core",
            "audience_type": "private_equity",
            "report_profile": "diligence_focused",
        },
        store=store,
    )
    workspace_two = workspace_two_response["workspace"]
    workflow_one, dossier_one = _create_workflow_and_dossier(
        store,
        workspace_id=workspace_one["workspace_id"],
        symbol="NVDA",
        title_prefix="Tenant One",
    )
    workflow_two, dossier_two = _create_workflow_and_dossier(
        store,
        workspace_id=workspace_two["workspace_id"],
        symbol="MSFT",
        title_prefix="Tenant Two",
    )
    platform_service.create_membership_service(
        {
            "user_id": "reviewer-1",
            "workspace_id": workspace_one["workspace_id"],
            "organization_id": workspace_one["organization_id"],
            "role": "reviewer",
        },
        store=store,
    )

    scoped_workspaces = platform_service.list_workspaces_service(
        user_context={"user_id": "reviewer-1", "role": "reviewer"},
        store=store,
    )["workspaces"]
    scoped_workflows = platform_service.list_workflows_service(
        user_context={"user_id": "reviewer-1", "role": "reviewer"},
        store=store,
    )["workflows"]
    scoped_dossiers = platform_service.list_dossiers_service(
        user_context={"user_id": "reviewer-1", "role": "reviewer"},
        store=store,
    )["dossiers"]

    assert {item["workspace_id"] for item in scoped_workspaces} == {
        workspace_one["workspace_id"]
    }
    assert workflow_one["workflow_id"] in {
        item["workflow_id"] for item in scoped_workflows
    }
    assert dossier_one["dossier_id"] in {
        item["dossier_id"] for item in scoped_dossiers
    }
    assert workflow_two["workflow_id"] not in {item["workflow_id"] for item in scoped_workflows}
    assert dossier_two["dossier_id"] not in {item["dossier_id"] for item in scoped_dossiers}


def test_phase8a_org_admin_can_span_workspaces_within_org_only() -> None:
    store = PlatformStore(use_memory=True)
    workspace_one, _, _ = _bootstrap_workspace(store)
    workspace_same_org = _create_second_workspace_same_org(
        store,
        workspace_one["organization_id"],
    )
    workspace_other_org = platform_service.create_workspace_service(
        {
            "name": "Other Org Workspace",
            "platform_profile": "research_core",
        },
        store=store,
    )["workspace"]
    platform_service.create_membership_service(
        {
            "user_id": "org-admin-1",
            "organization_id": workspace_one["organization_id"],
            "role": "org_admin",
        },
        store=store,
    )

    workspaces = platform_service.list_workspaces_service(
        user_context={"user_id": "org-admin-1", "role": "org_admin"},
        store=store,
    )["workspaces"]
    assert {item["workspace_id"] for item in workspaces} == {
        workspace_one["workspace_id"],
        workspace_same_org["workspace_id"],
    }
    assert workspace_other_org["workspace_id"] not in {
        item["workspace_id"] for item in workspaces
    }


def test_phase8a_enriches_audit_actor_and_tenant_metadata() -> None:
    store = PlatformStore(use_memory=True)
    workspace, workflow, dossier = _bootstrap_workspace(store)
    platform_service.create_membership_service(
        {
            "user_id": "analyst-1",
            "workspace_id": workspace["workspace_id"],
            "organization_id": workspace["organization_id"],
            "role": "analyst",
        },
        store=store,
    )

    platform_service.attach_analysis_to_dossier_service(
        dossier["dossier_id"],
        {
            "report": _sample_platform_report("NVDA"),
            "report_id": "report-1",
            "session_id": "session-attach-1",
            "axiom_artifact_id": "axiom-1",
        },
        user_context={
            "user_id": "analyst-1",
            "role": "analyst",
            "email": "analyst@example.com",
            "session_id": "session-attach-1",
        },
        store=store,
    )
    platform_service.execute_workflow_action_service(
        workflow["workflow_id"],
        {
            "dossier_id": dossier["dossier_id"],
            "action_type": "advance_stage",
            "requested_stage": "analysis",
        },
        user_context={
            "user_id": "analyst-1",
            "role": "analyst",
            "email": "analyst@example.com",
            "session_id": "session-attach-1",
        },
        store=store,
    )

    events = store.list_audit_events(workspace_id=workspace["workspace_id"])
    analysis_event = next(item for item in events if item["event_type"] == "analysis_attached")
    assert analysis_event["organization_id"] == workspace["organization_id"]
    assert analysis_event["workspace_id"] == workspace["workspace_id"]
    assert analysis_event["session_id"] == "session-attach-1"
    assert analysis_event["auth_mode"] == "header"
    assert analysis_event["actor"]["actor_email"] == "analyst@example.com"
    assert analysis_event["metadata"]["tenant_scope_summary"].startswith("workspace_scoped")


def test_phase8a_routes_enforce_workspace_scope_and_expose_session_endpoints() -> None:
    _reset_platform_store(platform_store)

    with TestClient(app) as client:
        workspace_one = client.post(
            "/platform/workspaces",
            json={
                "name": "Tenant Route Workspace",
                "platform_profile": "hf_core",
                "audience_type": "hedge_fund",
                "report_profile": "ic_memo",
            },
        ).json()["workspace"]
        workspace_two = client.post(
            "/platform/workspaces",
            json={
                "name": "Foreign Route Workspace",
                "platform_profile": "pe_core",
                "audience_type": "private_equity",
                "report_profile": "diligence_focused",
            },
        ).json()["workspace"]
        platform_store.create_membership(
            {
                "user_id": "analyst-1",
                "workspace_id": workspace_one["workspace_id"],
                "organization_id": workspace_one["organization_id"],
                "role": "analyst",
            }
        )
        workflow_resp = client.post(
            "/platform/workflows",
            json={
                "workspace_id": workspace_one["workspace_id"],
                "workflow_template_id": "hedge_fund_research",
                "title": "Tenant Workflow",
            },
            headers=_auth_headers(
                "analyst-1",
                "analyst",
                workspace_id=workspace_one["workspace_id"],
                email="analyst@example.com",
                session_id="route-session-1",
            ),
        )
        workflow = workflow_resp.json()["workflow"]
        dossier_resp = client.post(
            "/platform/dossiers",
            json={
                "workflow_id": workflow["workflow_id"],
                "symbol": "NVDA",
                "display_name": "NVIDIA",
            },
            headers=_auth_headers(
                "analyst-1",
                "analyst",
                workspace_id=workspace_one["workspace_id"],
                email="analyst@example.com",
                session_id="route-session-1",
            ),
        )
        dossier = dossier_resp.json()["dossier"]
        attach_resp = client.post(
            f"/platform/dossiers/{dossier['dossier_id']}/attach-analysis",
            json={
                "report": _sample_platform_report("NVDA"),
                "report_id": "report-route",
                "session_id": "route-session-1",
                "axiom_artifact_id": "axiom-route",
            },
            headers=_auth_headers(
                "analyst-1",
                "analyst",
                workspace_id=workspace_one["workspace_id"],
                email="analyst@example.com",
                session_id="route-session-1",
            ),
        )
        assert attach_resp.status_code == 200

        session_resp = client.get(
            "/platform/auth/session",
            params={"workspace_id": workspace_one["workspace_id"]},
            headers=_auth_headers(
                "analyst-1",
                "analyst",
                workspace_id=workspace_one["workspace_id"],
                email="analyst@example.com",
                session_id="route-session-1",
            ),
        )
        effective_resp = client.get(
            "/platform/access/effective",
            params={"workspace_id": workspace_one["workspace_id"]},
            headers=_auth_headers(
                "analyst-1",
                "analyst",
                workspace_id=workspace_one["workspace_id"],
                email="analyst@example.com",
                session_id="route-session-1",
            ),
        )
        tenancy_resp = client.get(
            "/platform/tenancy/summary",
            params={"workspace_id": workspace_one["workspace_id"]},
            headers=_auth_headers(
                "analyst-1",
                "analyst",
                workspace_id=workspace_one["workspace_id"],
                email="analyst@example.com",
                session_id="route-session-1",
            ),
        )
        workspaces_resp = client.get(
            "/platform/workspaces",
            headers=_auth_headers(
                "analyst-1",
                "analyst",
                workspace_id=workspace_one["workspace_id"],
            ),
        )
        dossiers_resp = client.get(
            "/platform/dossiers",
            params={"workspace_id": workspace_one["workspace_id"]},
            headers=_auth_headers(
                "analyst-1",
                "analyst",
                workspace_id=workspace_one["workspace_id"],
            ),
        )
        denied_dossiers_resp = client.get(
            "/platform/dossiers",
            params={"workspace_id": workspace_two["workspace_id"]},
            headers=_auth_headers(
                "analyst-1",
                "analyst",
                workspace_id=workspace_one["workspace_id"],
            ),
        )

    assert session_resp.status_code == 200
    assert session_resp.json()["session"]["session_id"] == "route-session-1"
    assert session_resp.json()["session"]["email"] == "analyst@example.com"
    assert effective_resp.status_code == 200
    assert effective_resp.json()["access_summary"]["tenant_scope_summary"].startswith(
        "workspace_scoped"
    )
    assert tenancy_resp.status_code == 200
    assert tenancy_resp.json()["tenancy_summary"]["accessible_workspace_count"] == 1
    assert workspaces_resp.status_code == 200
    assert len(workspaces_resp.json()["workspaces"]) == 1
    assert dossiers_resp.status_code == 200
    assert len(dossiers_resp.json()["dossiers"]) == 1
    assert denied_dossiers_resp.status_code == 403
    assert (
        denied_dossiers_resp.json()["error"]["detail"]["error"] == "access_denied"
    )
