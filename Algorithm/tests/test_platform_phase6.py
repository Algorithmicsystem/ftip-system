from __future__ import annotations

from fastapi.testclient import TestClient

from api.main import app
from api.platform import service as platform_service
from api.platform.persistence import PlatformStore, platform_store
from tests.test_platform_phase5 import (
    _patch_analyze,
    _reset_platform_store,
    _sample_platform_report,
)


def _headers(
    user_id: str,
    role: str,
    *,
    workspace_id: str | None = None,
) -> dict[str, str]:
    headers = {
        "X-Platform-User-Id": user_id,
        "X-Platform-Role": role,
    }
    if workspace_id:
        headers["X-Platform-Workspace-Id"] = workspace_id
    return headers


def _bootstrap_workspace(store: PlatformStore) -> tuple[dict, dict, dict]:
    workspace_response = platform_service.create_workspace_service(
        {
            "name": "HF Controls Workspace",
            "platform_profile": "hf_core",
            "audience_type": "hedge_fund",
            "report_profile": "ic_memo",
        },
        store=store,
    )
    workspace = workspace_response["workspace"]
    workflow_response = platform_service.create_workflow_service(
        {
            "workspace_id": workspace["workspace_id"],
            "workflow_template_id": "hedge_fund_research",
            "title": "NVDA Enterprise Workflow",
        },
        store=store,
    )
    workflow = workflow_response["workflow"]
    dossier_response = platform_service.create_dossier_service(
        {
            "workflow_id": workflow["workflow_id"],
            "symbol": "NVDA",
            "display_name": "NVIDIA",
            "dossier_type": "coverage",
            "title": "NVDA Enterprise Dossier",
        },
        store=store,
    )
    dossier = dossier_response["dossier"]
    return workspace, workflow, dossier


def test_platform_access_summary_supports_default_and_membership_modes() -> None:
    store = PlatformStore(use_memory=True)
    workspace, _, _ = _bootstrap_workspace(store)

    default_summary = platform_service.build_access_summary_service(
        workspace_id=workspace["workspace_id"],
        store=store,
    )["access_summary"]
    assert default_summary["development_mode"] is True
    assert default_summary["effective_role"] == "service_account"
    assert "approve_stage" in default_summary["effective_permissions"]

    platform_service.create_membership_service(
        {
            "user_id": "analyst-1",
            "workspace_id": workspace["workspace_id"],
            "organization_id": workspace["organization_id"],
            "role": "analyst",
        },
        store=store,
    )
    member_summary = platform_service.build_access_summary_service(
        workspace_id=workspace["workspace_id"],
        user_context={"user_id": "analyst-1", "role": "analyst"},
        store=store,
    )["access_summary"]
    assert member_summary["development_mode"] is False
    assert member_summary["effective_role"] == "analyst"
    assert "create_dossier" in member_summary["effective_permissions"]
    assert "approve_stage" not in member_summary["effective_permissions"]


def test_platform_workflow_approval_export_integration_and_health_flow() -> None:
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
    platform_service.create_membership_service(
        {
            "user_id": "committee-1",
            "workspace_id": workspace["workspace_id"],
            "organization_id": workspace["organization_id"],
            "role": "committee",
        },
        store=store,
    )
    platform_service.create_membership_service(
        {
            "user_id": "admin-1",
            "workspace_id": workspace["workspace_id"],
            "organization_id": workspace["organization_id"],
            "role": "workspace_admin",
        },
        store=store,
    )

    attach_response = platform_service.attach_analysis_to_dossier_service(
        dossier["dossier_id"],
        {
            "report": _sample_platform_report("NVDA"),
            "report_id": "report-1",
            "session_id": "session-1",
            "axiom_artifact_id": "axiom-1",
            "axiom_report_pack_artifact_id": "axiom-report-pack-1",
            "axiom_lineage_artifact_id": "axiom-lineage-1",
            "axiom_history_artifact_id": "axiom-history-1",
            "axiom_calibration_artifact_id": "axiom-calibration-1",
        },
        user_context={"user_id": "analyst-1", "role": "analyst"},
        store=store,
    )
    assert attach_response["dossier"]["latest_axiom_analysis_id"] == "axiom-1"

    action_response = platform_service.execute_workflow_action_service(
        workflow["workflow_id"],
        {
            "dossier_id": dossier["dossier_id"],
            "action_type": "advance_stage",
            "requested_stage": "analysis",
            "rationale": "Initial research has been completed.",
        },
        user_context={"user_id": "analyst-1", "role": "analyst"},
        store=store,
    )
    assert action_response["workflow"]["stage"] == "analysis"

    approval_request = platform_service.handle_workflow_approval_service(
        workflow["workflow_id"],
        {
            "dossier_id": dossier["dossier_id"],
            "mode": "request",
            "requested_role": "committee",
            "rationale": "Need committee review before progression.",
        },
        user_context={"user_id": "analyst-1", "role": "analyst"},
        store=store,
    )
    assert approval_request["approval"]["status"] == "pending"

    approval_decision = platform_service.handle_workflow_approval_service(
        workflow["workflow_id"],
        {
            "approval_id": approval_request["approval"]["approval_id"],
            "dossier_id": dossier["dossier_id"],
            "mode": "decision",
            "decision_type": "approve",
            "rationale": "Committee approval granted.",
        },
        user_context={"user_id": "committee-1", "role": "committee"},
        store=store,
    )
    assert approval_decision["approval"]["status"] == "approved"
    assert approval_decision["workflow"]["status"] == "approved"

    export_response = platform_service.create_dossier_export_service(
        dossier["dossier_id"],
        {
            "pack_type": "ic_memo_pack",
            "metadata": {"requested_by": "committee-1"},
        },
        user_context={"user_id": "committee-1", "role": "committee"},
        store=store,
    )
    assert export_response["export"]["pack_type"] == "ic_memo_pack"
    assert export_response["export"]["content_hash"]
    assert export_response["export"]["ordered_sections"]

    integration_response = platform_service.create_integration_binding_service(
        {
            "integration_type": "document_storage",
            "organization_id": workspace["organization_id"],
            "workspace_id": workspace["workspace_id"],
            "status": "active",
            "config": {"storage_label": "internal-archive"},
        },
        user_context={"user_id": "admin-1", "role": "workspace_admin"},
        store=store,
    )
    assert integration_response["binding"]["status"] == "active"

    integrations = platform_service.list_integrations_service(
        workspace_id=workspace["workspace_id"],
        user_context={"user_id": "admin-1", "role": "workspace_admin"},
        store=store,
    )
    assert integrations["health_summary"]["binding_count"] == 1

    timeline = platform_service.list_workflow_timeline_service(
        workflow["workflow_id"],
        user_context={"user_id": "committee-1", "role": "committee"},
        store=store,
    )
    assert any(item["event_type"] == "analysis_attached" for item in timeline["timeline"])
    assert any("approval" in item["event_type"] for item in timeline["timeline"])

    health = platform_service.build_platform_health_service(
        workspace_id=workspace["workspace_id"],
        user_context={"user_id": "admin-1", "role": "workspace_admin"},
        store=store,
    )["health"]
    assert health["pending_approval_count"] == 0
    assert health["export_count"] == 1
    assert health["audit_event_count"] >= 5


def test_platform_routes_support_enterprise_controls() -> None:
    _reset_platform_store(platform_store)

    with TestClient(app) as client:
        workspace_resp = client.post(
            "/platform/workspaces",
            json={
                "name": "Platform Controls Workspace",
                "platform_profile": "hf_core",
                "audience_type": "hedge_fund",
                "report_profile": "ic_memo",
            },
        )
        workspace = workspace_resp.json()["workspace"]

        platform_store.create_membership(
            {
                "user_id": "analyst-1",
                "workspace_id": workspace["workspace_id"],
                "organization_id": workspace["organization_id"],
                "role": "analyst",
            }
        )
        platform_store.create_membership(
            {
                "user_id": "committee-1",
                "workspace_id": workspace["workspace_id"],
                "organization_id": workspace["organization_id"],
                "role": "committee",
            }
        )
        platform_store.create_membership(
            {
                "user_id": "admin-1",
                "workspace_id": workspace["workspace_id"],
                "organization_id": workspace["organization_id"],
                "role": "workspace_admin",
            }
        )

        workflow_resp = client.post(
            "/platform/workflows",
            json={
                "workspace_id": workspace["workspace_id"],
                "workflow_template_id": "hedge_fund_research",
                "title": "MSFT Enterprise Workflow",
            },
            headers=_headers("analyst-1", "analyst", workspace_id=workspace["workspace_id"]),
        )
        workflow = workflow_resp.json()["workflow"]

        dossier_resp = client.post(
            "/platform/dossiers",
            json={
                "workflow_id": workflow["workflow_id"],
                "symbol": "MSFT",
                "display_name": "Microsoft",
                "title": "MSFT Enterprise Dossier",
            },
            headers=_headers("analyst-1", "analyst", workspace_id=workspace["workspace_id"]),
        )
        dossier = dossier_resp.json()["dossier"]

        attach_resp = client.post(
            f"/platform/dossiers/{dossier['dossier_id']}/attach-analysis",
            json={
                "report": _sample_platform_report("MSFT"),
                "report_id": "report-msft",
                "session_id": "session-msft",
                "axiom_artifact_id": "axiom-msft",
            },
            headers=_headers("analyst-1", "analyst", workspace_id=workspace["workspace_id"]),
        )
        assert attach_resp.status_code == 200

        access_resp = client.get(
            "/platform/access/summary",
            params={"workspace_id": workspace["workspace_id"]},
            headers=_headers("analyst-1", "analyst", workspace_id=workspace["workspace_id"]),
        )
        assert access_resp.json()["access_summary"]["effective_role"] == "analyst"

        action_resp = client.post(
            f"/platform/workflows/{workflow['workflow_id']}/actions",
            json={
                "dossier_id": dossier["dossier_id"],
                "action_type": "advance_stage",
                "requested_stage": "analysis",
            },
            headers=_headers("analyst-1", "analyst", workspace_id=workspace["workspace_id"]),
        )
        assert action_resp.status_code == 200
        assert action_resp.json()["workflow"]["stage"] == "analysis"

        approval_req = client.post(
            f"/platform/workflows/{workflow['workflow_id']}/approvals",
            json={
                "dossier_id": dossier["dossier_id"],
                "mode": "request",
                "requested_role": "committee",
            },
            headers=_headers("analyst-1", "analyst", workspace_id=workspace["workspace_id"]),
        )
        approval_id = approval_req.json()["approval"]["approval_id"]

        approval_decision = client.post(
            f"/platform/workflows/{workflow['workflow_id']}/approvals",
            json={
                "approval_id": approval_id,
                "dossier_id": dossier["dossier_id"],
                "mode": "decision",
                "decision_type": "approve",
            },
            headers=_headers("committee-1", "committee", workspace_id=workspace["workspace_id"]),
        )
        assert approval_decision.status_code == 200
        assert approval_decision.json()["approval"]["status"] == "approved"

        export_resp = client.post(
            f"/platform/dossiers/{dossier['dossier_id']}/export",
            json={"pack_type": "dossier_pack"},
            headers=_headers("committee-1", "committee", workspace_id=workspace["workspace_id"]),
        )
        assert export_resp.status_code == 200
        assert export_resp.json()["export"]["content_hash"]

        exports_resp = client.get(
            f"/platform/dossiers/{dossier['dossier_id']}/exports",
            headers=_headers("committee-1", "committee", workspace_id=workspace["workspace_id"]),
        )
        assert exports_resp.status_code == 200
        assert exports_resp.json()["exports"]

        integration_resp = client.post(
            "/platform/integrations",
            json={
                "integration_type": "document_storage",
                "organization_id": workspace["organization_id"],
                "workspace_id": workspace["workspace_id"],
                "status": "active",
                "config": {"storage_label": "archive"},
            },
            headers=_headers("admin-1", "workspace_admin", workspace_id=workspace["workspace_id"]),
        )
        assert integration_resp.status_code == 200

        integrations = client.get(
            "/platform/integrations",
            params={"workspace_id": workspace["workspace_id"]},
            headers=_headers("admin-1", "workspace_admin", workspace_id=workspace["workspace_id"]),
        )
        assert integrations.status_code == 200
        assert integrations.json()["health_summary"]["binding_count"] == 1

        timeline = client.get(
            f"/platform/workflows/{workflow['workflow_id']}/timeline",
            headers=_headers("committee-1", "committee", workspace_id=workspace["workspace_id"]),
        )
        assert timeline.status_code == 200
        assert timeline.json()["timeline"]

        audit = client.get(
            "/platform/audit",
            params={"workspace_id": workspace["workspace_id"]},
            headers=_headers("committee-1", "committee", workspace_id=workspace["workspace_id"]),
        )
        assert audit.status_code == 200
        assert audit.json()["audit_events"]

        health = client.get(
            "/platform/health",
            params={"workspace_id": workspace["workspace_id"]},
            headers=_headers("admin-1", "workspace_admin", workspace_id=workspace["workspace_id"]),
        )
        assert health.status_code == 200
        assert health.json()["health"]["export_count"] == 1


def test_assistant_analyze_includes_phase6_platform_control_context(monkeypatch) -> None:
    _reset_platform_store(platform_store)
    _patch_analyze(monkeypatch)

    with TestClient(app) as client:
        response = client.post(
            "/assistant/analyze",
            json={
                "symbol": "NVDA",
                "horizon": "swing",
                "risk_mode": "balanced",
                "platform_profile": "hf_core",
                "workflow_template_id": "hedge_fund_research",
                "create_dossier": True,
            },
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["platform_access_control_summary"]
    assert payload["platform_workflow_actions_summary"]
    assert payload["platform_audit_timeline_summary"]
    assert payload["platform_export_summary"]
    assert payload["platform_integration_health_summary"]
    assert payload["platform_health_summary"]["access_summary"]["development_mode"] is True
