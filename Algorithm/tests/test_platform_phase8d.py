from __future__ import annotations

from fastapi.testclient import TestClient

from api.main import app
from api.platform import service as platform_service
from api.platform.persistence import PlatformStore, platform_store
from tests.test_platform_phase5 import _reset_platform_store
from tests.test_platform_phase8a import _auth_headers


def test_phase8d_bootstrap_workspace_creates_seeded_pilot_package() -> None:
    store = PlatformStore(use_memory=True)

    bootstrap = platform_service.bootstrap_workspace_service(
        {
            "organization_name": "Pilot HF",
            "workspace_name": "Pilot HF Workspace",
            "platform_profile": "hf_core",
            "workflow_template_id": "hedge_fund_research",
            "demo_bundle_id": "hedge_fund_demo_bundle",
            "seed_demo_bundle": True,
            "include_exports": True,
            "include_integrations": True,
            "metadata": {"requested_by": "test"},
        },
        store=store,
    )

    assert bootstrap["workspace"]["name"] == "Pilot HF Workspace"
    assert bootstrap["bootstrap_summary"]["demo_seeded"] is True
    assert bootstrap["bootstrap_summary"]["seeded_dossier_count"] == 1
    assert bootstrap["bootstrap_summary"]["seeded_stored_export_count"] >= 1
    assert bootstrap["readiness_report"]["overall_status"] in {"ready", "partial"}
    assert any(
        item["category"] == "export_state" and item["status"] == "ready"
        for item in bootstrap["readiness_report"]["categories"]
    )
    assert bootstrap["pilot_package"]["top_dossiers"]
    assert (
        bootstrap["workspace"]["settings"]["pilot_bootstrap_summary"]["platform_profile"]
        == "hf_core"
    )

    events = store.list_audit_events(workspace_id=bootstrap["workspace"]["workspace_id"])
    event_types = {item["event_type"] for item in events}
    assert "workspace_bootstrapped" in event_types
    assert "demo_bundle_applied" in event_types


def test_phase8d_bootstrap_templates_expose_profile_defaults_and_demo_bundles() -> None:
    catalog = platform_service.build_bootstrap_templates_service()

    assert catalog["bootstrap_profiles"]
    hf_profile = next(
        item for item in catalog["bootstrap_profiles"] if item["profile_id"] == "hf_core"
    )
    assert hf_profile["pilot_bootstrap_defaults"]["demo_bundle_id"] == "hedge_fund_demo_bundle"
    assert hf_profile["default_export_pack_emphasis"]
    assert any(
        item["bundle_id"] == "research_team_demo_bundle"
        for item in catalog["demo_seed_bundles"]
    )


def test_phase8d_authenticated_bootstrap_and_readiness_routes_are_tenant_safe() -> None:
    _reset_platform_store(platform_store)

    with TestClient(app) as client:
        bootstrap_resp = client.post(
            "/platform/bootstrap/workspace",
            json={
                "organization_name": "Tenant Pilot Org",
                "workspace_name": "Tenant Pilot Workspace",
                "platform_profile": "research_core",
                "demo_bundle_id": "research_team_demo_bundle",
                "seed_demo_bundle": True,
                "include_exports": True,
            },
            headers=_auth_headers(
                "bootstrap-1",
                "workspace_admin",
                email="bootstrap@example.com",
                session_id="bootstrap-session-1",
            ),
        )
        assert bootstrap_resp.status_code == 200
        workspace = bootstrap_resp.json()["workspace"]

        second_workspace = client.post(
            "/platform/workspaces",
            json={
                "name": "Foreign Tenant Workspace",
                "platform_profile": "hf_core",
                "audience_type": "hedge_fund",
                "report_profile": "ic_memo",
            },
        ).json()["workspace"]

        readiness_resp = client.get(
            f"/platform/workspaces/{workspace['workspace_id']}/readiness",
            headers=_auth_headers(
                "bootstrap-1",
                "workspace_admin",
                workspace_id=workspace["workspace_id"],
                email="bootstrap@example.com",
                session_id="bootstrap-session-1",
            ),
        )
        pilot_package_resp = client.get(
            f"/platform/workspaces/{workspace['workspace_id']}/pilot-package",
            headers=_auth_headers(
                "bootstrap-1",
                "workspace_admin",
                workspace_id=workspace["workspace_id"],
                email="bootstrap@example.com",
                session_id="bootstrap-session-1",
            ),
        )
        denied_resp = client.get(
            f"/platform/workspaces/{second_workspace['workspace_id']}/readiness",
            headers=_auth_headers(
                "bootstrap-1",
                "workspace_admin",
                workspace_id=workspace["workspace_id"],
                email="bootstrap@example.com",
                session_id="bootstrap-session-1",
            ),
        )

    assert readiness_resp.status_code == 200
    readiness_payload = readiness_resp.json()
    assert readiness_payload["readiness_report"]["categories"]
    assert readiness_payload["readiness_report"]["metadata"]["stored_export_count"] >= 1
    assert pilot_package_resp.status_code == 200
    assert pilot_package_resp.json()["pilot_package"]["top_dossiers"]
    assert denied_resp.status_code == 403

    events = platform_store.list_audit_events(workspace_id=workspace["workspace_id"])
    event_types = {item["event_type"] for item in events}
    assert "readiness_report_generated" in event_types
    assert "pilot_package_generated" in event_types


def test_phase8d_demo_bundle_apply_route_and_readiness_warning_behavior() -> None:
    _reset_platform_store(platform_store)
    workspace = platform_service.create_workspace_service(
        {
            "name": "PE Bootstrap Workspace",
            "platform_profile": "pe_core",
            "audience_type": "private_equity",
            "report_profile": "diligence_focused",
        },
        store=platform_store,
    )["workspace"]
    platform_service.create_membership_service(
        {
            "user_id": "pe-admin",
            "workspace_id": workspace["workspace_id"],
            "organization_id": workspace["organization_id"],
            "role": "workspace_admin",
        },
        store=platform_store,
    )

    with TestClient(app) as client:
        bundle_list = client.get("/platform/demo/bundles")
        assert bundle_list.status_code == 200
        assert any(
            item["bundle_id"] == "private_equity_demo_bundle"
            for item in bundle_list.json()["demo_seed_bundles"]
        )

        apply_resp = client.post(
            "/platform/demo/bundles/private_equity_demo_bundle/apply",
            json={
                "workspace_id": workspace["workspace_id"],
                "include_exports": True,
                "include_integrations": False,
            },
            headers=_auth_headers(
                "pe-admin",
                "workspace_admin",
                workspace_id=workspace["workspace_id"],
                email="pe@example.com",
                session_id="pe-session-1",
            ),
        )
        readiness_resp = client.get(
            "/platform/readiness",
            params={"workspace_id": workspace["workspace_id"]},
            headers=_auth_headers(
                "pe-admin",
                "workspace_admin",
                workspace_id=workspace["workspace_id"],
                email="pe@example.com",
                session_id="pe-session-1",
            ),
        )

    assert apply_resp.status_code == 200
    assert apply_resp.json()["demo_bundle"]["bundle_id"] == "private_equity_demo_bundle"
    assert apply_resp.json()["seeded_stored_exports"]
    assert readiness_resp.status_code == 200
    warnings = readiness_resp.json()["readiness_report"]["warnings"]
    assert any(item["category"] == "integration_state" for item in warnings)
