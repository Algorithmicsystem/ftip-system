from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from api.main import app
from api.platform import service as platform_service
from api.platform.persistence import PlatformStore, platform_store
from tests.test_platform_phase5 import _reset_platform_store, _sample_platform_report
from tests.test_platform_phase6 import _bootstrap_workspace, _headers
from tests.test_platform_phase7 import _attach_sample_analysis


def test_phase8b_store_retrieve_history_and_integrity() -> None:
    store = PlatformStore(use_memory=True)
    workspace, workflow, dossier = _bootstrap_workspace(store)
    _attach_sample_analysis(
        store,
        dossier["dossier_id"],
        symbol="NVDA",
        report_id="report-1",
        session_id="session-1",
    )

    first = platform_service.store_dossier_export_service(
        dossier["dossier_id"],
        {
            "pack_type": "ic_memo_pack",
            "export_format": "html",
            "storage_backend": "in_memory_store",
        },
        store=store,
    )
    second = platform_service.store_dossier_export_service(
        dossier["dossier_id"],
        {
            "pack_type": "ic_memo_pack",
            "export_format": "html",
            "storage_backend": "in_memory_store",
        },
        store=store,
    )

    assert first["stored_export"]["version_number"] == 1
    assert second["stored_export"]["version_number"] == 2
    assert second["stored_export"]["status"] == "stored"
    assert second["integrity"]["status"] == "valid"

    content = platform_service.get_stored_export_content_service(
        second["stored_export"]["stored_export_id"],
        store=store,
    )
    metadata = platform_service.get_stored_export_metadata_service(
        second["stored_export"]["stored_export_id"],
        store=store,
    )
    history = platform_service.list_dossier_export_history_service(
        dossier["dossier_id"],
        store=store,
    )
    versions = platform_service.list_stored_export_versions_service(
        second["stored_export"]["stored_export_id"],
        store=store,
    )

    assert "<!doctype html>" in content["retrieval"]["rendered_content"].lower()
    assert metadata["stored_export"]["document_identity"]["version_label"] == "v2"
    assert len(history["stored_exports"]) == 2
    assert any(item["status"] == "superseded" for item in versions["versions"])
    assert any(item["status"] == "stored" for item in versions["versions"])


def test_phase8b_local_file_store_and_integrity_detects_tamper(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("FTIP_PLATFORM_EXPORT_STORAGE_ROOT", str(tmp_path / "exports"))
    store = PlatformStore(use_memory=True)
    _, _, dossier = _bootstrap_workspace(store)
    _attach_sample_analysis(
        store,
        dossier["dossier_id"],
        symbol="NVDA",
        report_id="report-1",
        session_id="session-1",
    )

    stored = platform_service.store_dossier_export_service(
        dossier["dossier_id"],
        {
            "pack_type": "risk_deployability_pack",
            "export_format": "markdown",
            "storage_backend": "local_file_store",
        },
        store=store,
    )["stored_export"]

    path = Path(stored["storage_ref"]["local_path"])
    assert path.exists()
    path.write_text("tampered export body", encoding="utf-8")

    integrity = platform_service.get_stored_export_integrity_service(
        stored["stored_export_id"],
        store=store,
    )["integrity"]
    assert integrity["status"] == "failed"
    assert integrity["checksum_expected"] != integrity["checksum_actual"]


def test_phase8b_preserves_approval_context_at_export_time() -> None:
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
    _attach_sample_analysis(
        store,
        dossier["dossier_id"],
        symbol="NVDA",
        report_id="report-1",
        session_id="session-1",
    )
    approval = platform_service.handle_workflow_approval_service(
        workflow["workflow_id"],
        {
            "dossier_id": dossier["dossier_id"],
            "mode": "request",
            "requested_role": "committee",
            "rationale": "Committee review requested.",
        },
        user_context={"user_id": "analyst-1", "role": "analyst"},
        store=store,
    )["approval"]
    platform_service.handle_workflow_approval_service(
        workflow["workflow_id"],
        {
            "approval_id": approval["approval_id"],
            "dossier_id": dossier["dossier_id"],
            "mode": "decision",
            "decision_type": "approve",
            "rationale": "Approved for export.",
        },
        user_context={"user_id": "committee-1", "role": "committee"},
        store=store,
    )

    stored = platform_service.store_dossier_export_service(
        dossier["dossier_id"],
        {
            "pack_type": "ic_memo_pack",
            "export_format": "json",
            "storage_backend": "in_memory_store",
        },
        user_context={"user_id": "committee-1", "role": "committee"},
        store=store,
    )["stored_export"]

    later_approval = platform_service.handle_workflow_approval_service(
        workflow["workflow_id"],
        {
            "dossier_id": dossier["dossier_id"],
            "mode": "request",
            "requested_role": "committee",
            "rationale": "New review cycle.",
        },
        user_context={"user_id": "analyst-1", "role": "analyst"},
        store=store,
    )["approval"]
    platform_service.handle_workflow_approval_service(
        workflow["workflow_id"],
        {
            "approval_id": later_approval["approval_id"],
            "dossier_id": dossier["dossier_id"],
            "mode": "decision",
            "decision_type": "reject",
            "rationale": "Recommendation changed later.",
        },
        user_context={"user_id": "committee-1", "role": "committee"},
        store=store,
    )

    metadata = platform_service.get_stored_export_metadata_service(
        stored["stored_export_id"],
        user_context={"user_id": "committee-1", "role": "committee"},
        store=store,
    )["stored_export"]
    assert metadata["approval_status"] == "approved"
    assert metadata["approval_context"]["approval_status"] == "approved"


def test_phase8b_routes_enforce_tenant_safe_stored_export_access() -> None:
    _reset_platform_store(platform_store)
    workspace_one, workflow_one, dossier_one = _bootstrap_workspace(platform_store)
    platform_service.create_membership_service(
        {
            "user_id": "analyst-1",
            "workspace_id": workspace_one["workspace_id"],
            "organization_id": workspace_one["organization_id"],
            "role": "analyst",
        },
        store=platform_store,
    )
    _attach_sample_analysis(
        platform_store,
        dossier_one["dossier_id"],
        symbol="NVDA",
        report_id="report-1",
        session_id="session-1",
    )
    own_export = platform_service.store_dossier_export_service(
        dossier_one["dossier_id"],
        {
            "pack_type": "dossier_pack",
            "export_format": "html",
            "storage_backend": "in_memory_store",
        },
        user_context={"user_id": "analyst-1", "role": "analyst"},
        store=platform_store,
    )["stored_export"]

    workspace_two = platform_service.create_workspace_service(
        {
            "name": "Other Workspace",
            "platform_profile": "research_core",
            "audience_type": "research_team",
            "report_profile": "trading_focused",
        },
        store=platform_store,
    )["workspace"]
    workflow_two = platform_service.create_workflow_service(
        {
            "workspace_id": workspace_two["workspace_id"],
            "workflow_template_id": "research_watchlist",
            "title": "Other Workflow",
        },
        store=platform_store,
    )["workflow"]
    dossier_two = platform_service.create_dossier_service(
        {
            "workflow_id": workflow_two["workflow_id"],
            "symbol": "MSFT",
            "display_name": "Microsoft",
            "title": "MSFT Dossier",
        },
        store=platform_store,
    )["dossier"]
    platform_service.attach_analysis_to_dossier_service(
        dossier_two["dossier_id"],
        {
            "report": _sample_platform_report("MSFT"),
            "report_id": "report-2",
            "session_id": "session-2",
            "axiom_artifact_id": "axiom-msft",
            "axiom_report_pack_artifact_id": "axiom-pack-msft",
            "axiom_lineage_artifact_id": "axiom-lineage-msft",
            "axiom_history_artifact_id": "axiom-history-msft",
            "axiom_calibration_artifact_id": "axiom-calibration-msft",
        },
        store=platform_store,
    )
    foreign_export = platform_service.store_dossier_export_service(
        dossier_two["dossier_id"],
        {
            "pack_type": "ic_memo_pack",
            "export_format": "json",
            "storage_backend": "in_memory_store",
        },
        store=platform_store,
    )["stored_export"]

    with TestClient(app) as client:
        own_meta = client.get(
            f"/platform/exports/{own_export['stored_export_id']}/metadata",
            headers=_headers("analyst-1", "analyst", workspace_id=workspace_one["workspace_id"]),
        )
        own_workspace_exports = client.get(
            f"/platform/workspaces/{workspace_one['workspace_id']}/exports",
            headers=_headers("analyst-1", "analyst", workspace_id=workspace_one["workspace_id"]),
        )
        foreign_meta = client.get(
            f"/platform/exports/{foreign_export['stored_export_id']}/metadata",
            headers=_headers("analyst-1", "analyst", workspace_id=workspace_one["workspace_id"]),
        )

    assert own_meta.status_code == 200
    assert own_workspace_exports.status_code == 200
    assert len(own_workspace_exports.json()["stored_exports"]) == 1
    assert foreign_meta.status_code == 404
