from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from api.main import app
from api.platform import service as platform_service
from api.platform.export_renderers import render_export_manifest
from api.platform.persistence import PlatformStore, platform_store
from tests.test_platform_phase5 import _reset_platform_store, _sample_platform_report
from tests.test_platform_phase6 import _bootstrap_workspace


def _attach_sample_analysis(
    store: PlatformStore,
    dossier_id: str,
    *,
    symbol: str,
    report_id: str,
    session_id: str,
) -> dict:
    report = _sample_platform_report(symbol)
    return platform_service.attach_analysis_to_dossier_service(
        dossier_id,
        {
            "report": report,
            "report_id": report_id,
            "session_id": session_id,
            "axiom_artifact_id": f"axiom-{symbol.lower()}",
            "axiom_report_pack_artifact_id": f"axiom-pack-{symbol.lower()}",
            "axiom_lineage_artifact_id": f"axiom-lineage-{symbol.lower()}",
            "axiom_history_artifact_id": f"axiom-history-{symbol.lower()}",
            "axiom_calibration_artifact_id": f"axiom-calibration-{symbol.lower()}",
        },
        store=store,
    )


def test_phase7_export_renderers_are_deterministic() -> None:
    store = PlatformStore(use_memory=True)
    workspace, workflow, dossier = _bootstrap_workspace(store)
    _attach_sample_analysis(
        store,
        dossier["dossier_id"],
        symbol="NVDA",
        report_id="report-1",
        session_id="session-1",
    )

    export = platform_service.create_dossier_export_service(
        dossier["dossier_id"],
        {"pack_type": "ic_memo_pack"},
        store=store,
    )["export"]

    html_a = render_export_manifest(export, export_format="html")
    html_b = render_export_manifest(export, export_format="html")
    markdown_a = render_export_manifest(export, export_format="markdown")
    markdown_b = render_export_manifest(export, export_format="markdown")
    json_a = render_export_manifest(export, export_format="json")
    json_b = render_export_manifest(export, export_format="json")

    assert html_a.rendered_content == html_b.rendered_content
    assert html_a.checksum == html_b.checksum
    assert "<!doctype html>" in html_a.rendered_content.lower()
    assert markdown_a.rendered_content == markdown_b.rendered_content
    assert markdown_a.checksum == markdown_b.checksum
    assert markdown_a.rendered_content.startswith("# IC Memo Pack")
    assert json_a.rendered_content == json_b.rendered_content
    assert json_a.checksum == json_b.checksum
    assert '"pack_type":"ic_memo_pack"' in json_a.rendered_content


def test_phase7_workspace_dashboard_analytics_and_demo_snapshots() -> None:
    store = PlatformStore(use_memory=True)
    workspace_one, workflow_one, dossier_one = _bootstrap_workspace(store)
    _attach_sample_analysis(
        store,
        dossier_one["dossier_id"],
        symbol="NVDA",
        report_id="report-1",
        session_id="session-1",
    )

    workspace_two = platform_service.create_workspace_service(
        {
            "name": "PE Diligence Workspace",
            "platform_profile": "pe_core",
            "audience_type": "private_equity",
            "report_profile": "diligence_focused",
        },
        store=store,
    )["workspace"]
    workflow_two = platform_service.create_workflow_service(
        {
            "workspace_id": workspace_two["workspace_id"],
            "workflow_template_id": "private_equity_diligence",
            "title": "MSFT Diligence Workflow",
        },
        store=store,
    )["workflow"]
    dossier_two = platform_service.create_dossier_service(
        {
            "workflow_id": workflow_two["workflow_id"],
            "symbol": "MSFT",
            "display_name": "Microsoft",
            "dossier_type": "coverage",
            "title": "MSFT Diligence Dossier",
        },
        store=store,
    )["dossier"]
    report_two = _sample_platform_report("MSFT")
    report_two["axiom_summary_card"]["deployability_tier"] = "live_candidate"
    report_two["axiom_summary_card"]["regime_label"] = "compensation_capture"
    report_two["axiom_summary_card"]["trade_family"] = "compensation"
    report_two["axiom_summary_card"]["size_band"] = "medium"
    report_two["axiom_summary_card"]["evidence_status"] = "supportive"
    report_two["axiom_summary_card"]["deployable_alpha_utility"] = 81.2
    report_two["axiom_summary_card"]["validated_edge"] = 63.5
    report_two["axiom_regime_label"] = "compensation_capture"
    report_two["axiom_trade_family"] = "compensation"
    report_two["axiom_deployability_tier"] = "live_candidate"
    report_two["axiom_evidence_backed_deployability_tier"] = "live_candidate"
    report_two["axiom_deployable_alpha_utility"] = 81.2
    report_two["axiom_validated_edge"] = 63.5
    report_two["axiom_final_size_band"] = "medium"
    report_two["axiom_size_band_recommendation"] = "medium"
    report_two["axiom_portfolio_governance"]["final_size_band"] = "medium"
    report_two["axiom_portfolio_governance"]["portfolio_fit_label"] = "core_candidate"
    report_two["axiom_portfolio_governance_summary"] = "Portfolio governance supports core candidate treatment."
    platform_service.attach_analysis_to_dossier_service(
        dossier_two["dossier_id"],
        {
            "report": report_two,
            "report_id": "report-2",
            "session_id": "session-2",
            "axiom_artifact_id": "axiom-msft",
            "axiom_report_pack_artifact_id": "axiom-pack-msft",
            "axiom_lineage_artifact_id": "axiom-lineage-msft",
            "axiom_history_artifact_id": "axiom-history-msft",
            "axiom_calibration_artifact_id": "axiom-calibration-msft",
        },
        store=store,
    )

    platform_service.create_dossier_export_service(
        dossier_two["dossier_id"],
        {"pack_type": "institutional_one_pager_pack"},
        store=store,
    )

    workspace_analytics = platform_service.build_workspace_analytics_service(
        workspace_two["workspace_id"],
        store=store,
    )["analytics"]
    platform_analytics = platform_service.build_platform_analytics_service(
        store=store,
    )["analytics"]
    dashboard = platform_service.build_platform_dashboard_service(
        workspace_id=workspace_two["workspace_id"],
        store=store,
    )["dashboard"]
    snapshot = platform_service.build_demo_snapshot_service(
        workspace_id=workspace_two["workspace_id"],
        store=store,
    )["snapshot"]
    readiness = platform_service.build_demo_readiness_service(
        workspace_id=workspace_two["workspace_id"],
        store=store,
    )["readiness"]

    assert workspace_analytics["dossier_count"] == 1
    assert workspace_analytics["average_dau"] == 81.2
    assert workspace_analytics["dossiers_by_deployability_tier"]["live_candidate"] == 1
    assert platform_analytics["counts_by_audience_type"]["hedge_fund"] == 1
    assert platform_analytics["counts_by_audience_type"]["private_equity"] == 1
    assert platform_analytics["deployability_distribution"]["paper_trade_only"] == 1
    assert platform_analytics["deployability_distribution"]["live_candidate"] == 1
    assert platform_analytics["regime_distribution"]["fundamental_convergence"] == 1
    assert platform_analytics["regime_distribution"]["compensation_capture"] == 1
    assert dashboard["executive_metrics"]["dossier_count"] == 1
    assert dashboard["high_dau_dossiers"][0]["symbol"] == "MSFT"
    assert snapshot["workspace_name"] == "PE Diligence Workspace"
    assert snapshot["top_dossiers"][0]["symbol"] == "MSFT"
    assert readiness["analysis_readiness"] == "ready"
    assert readiness["export_readiness"] == "ready"


def test_phase7_integration_execution_updates_history_and_health(tmp_path: Path) -> None:
    store = PlatformStore(use_memory=True)
    workspace, workflow, dossier = _bootstrap_workspace(store)
    _attach_sample_analysis(
        store,
        dossier["dossier_id"],
        symbol="NVDA",
        report_id="report-1",
        session_id="session-1",
    )

    archive_binding = platform_service.create_integration_binding_service(
        {
            "integration_type": "local_archive",
            "organization_id": workspace["organization_id"],
            "workspace_id": workspace["workspace_id"],
            "status": "active",
            "config": {"target_root": str(tmp_path / "archive")},
        },
        store=store,
    )["binding"]
    archive_execution = platform_service.execute_integration_service(
        archive_binding["binding_id"],
        {
            "action_type": "sync_export",
            "dossier_id": dossier["dossier_id"],
            "pack_type": "risk_deployability_pack",
            "export_format": "markdown",
        },
        store=store,
    )
    archive_path = Path(archive_execution["execution"]["output_summary"]["archive_path"])
    assert archive_execution["execution"]["status"] == "completed"
    assert archive_path.exists()

    webhook_binding = platform_service.create_integration_binding_service(
        {
            "integration_type": "webhook",
            "organization_id": workspace["organization_id"],
            "workspace_id": workspace["workspace_id"],
            "status": "active",
            "config": {"outbox_root": str(tmp_path / "webhook")},
        },
        store=store,
    )["binding"]
    webhook_execution = platform_service.execute_integration_service(
        webhook_binding["binding_id"],
        {
            "action_type": "sync_export",
            "dossier_id": dossier["dossier_id"],
            "pack_type": "axiom_summary_pack",
            "export_format": "json",
            "event_type": "export_ready",
        },
        store=store,
    )
    assert webhook_execution["execution"]["status"] == "queued"
    assert Path(webhook_execution["execution"]["output_summary"]["outbox_path"]).exists()

    sink_binding = platform_service.create_integration_binding_service(
        {
            "integration_type": "internal_sink",
            "organization_id": workspace["organization_id"],
            "workspace_id": workspace["workspace_id"],
            "status": "active",
            "config": {"sink_path": str(tmp_path / "sink" / "events.jsonl")},
        },
        store=store,
    )["binding"]
    sink_execution = platform_service.execute_integration_service(
        sink_binding["binding_id"],
        {
            "action_type": "sync_export",
            "dossier_id": dossier["dossier_id"],
            "pack_type": "dossier_pack",
            "event_type": "dossier_sync",
        },
        store=store,
    )
    sink_path = Path(sink_execution["execution"]["output_summary"]["sink_path"])
    assert sink_execution["execution"]["status"] == "completed"
    assert sink_path.exists()

    history = platform_service.list_integration_history_service(
        archive_binding["binding_id"],
        store=store,
    )["history"]
    listed = platform_service.list_integrations_service(
        workspace_id=workspace["workspace_id"],
        store=store,
    )
    archive_listed = next(
        item
        for item in listed["bindings"]
        if item["binding_id"] == archive_binding["binding_id"]
    )
    assert len(history) == 1
    assert archive_listed["last_execution"]["status"] == "completed"
    assert archive_listed["health"]["details"]["last_execution_status"] == "completed"
    assert listed["health_summary"]["binding_count"] == 3
    audit_events = store.list_audit_events(
        workspace_id=workspace["workspace_id"],
        resource_type="workspace",
    )
    assert any(item["event_type"] == "integration_executed" for item in audit_events)


def test_phase7_routes_cover_dashboard_rendering_and_demo_views(tmp_path: Path) -> None:
    _reset_platform_store(platform_store)

    with TestClient(app) as client:
        workspace_resp = client.post(
            "/platform/workspaces",
            json={
                "name": "HF Demo Workspace",
                "platform_profile": "hf_core",
                "audience_type": "hedge_fund",
                "report_profile": "ic_memo",
            },
        )
        workspace = workspace_resp.json()["workspace"]
        workflow_resp = client.post(
            "/platform/workflows",
            json={
                "workspace_id": workspace["workspace_id"],
                "workflow_template_id": "hedge_fund_research",
                "title": "NVDA Demo Workflow",
            },
        )
        workflow = workflow_resp.json()["workflow"]
        dossier_resp = client.post(
            "/platform/dossiers",
            json={
                "workflow_id": workflow["workflow_id"],
                "symbol": "NVDA",
                "display_name": "NVIDIA",
                "dossier_type": "coverage",
                "title": "NVDA Demo Dossier",
            },
        )
        dossier = dossier_resp.json()["dossier"]
        attach_resp = client.post(
            f"/platform/dossiers/{dossier['dossier_id']}/attach-analysis",
            json={
                "report": _sample_platform_report("NVDA"),
                "report_id": "report-route-1",
                "session_id": "session-route-1",
                "axiom_artifact_id": "axiom-route-1",
                "axiom_report_pack_artifact_id": "axiom-pack-route-1",
                "axiom_lineage_artifact_id": "axiom-lineage-route-1",
                "axiom_history_artifact_id": "axiom-history-route-1",
                "axiom_calibration_artifact_id": "axiom-calibration-route-1",
            },
        )
        preview_resp = client.get(
            f"/platform/dossiers/{dossier['dossier_id']}/preview-export",
            params={"pack_type": "ic_memo_pack", "export_format": "markdown"},
        )
        render_resp = client.post(
            f"/platform/dossiers/{dossier['dossier_id']}/render-export",
            json={"pack_type": "ic_memo_pack", "export_format": "html"},
        )
        render_id = render_resp.json()["rendered_export"]["render_id"]
        export_detail = client.get(f"/platform/exports/{render_id}")
        integration_resp = client.post(
            "/platform/integrations",
            json={
                "integration_type": "local_archive",
                "organization_id": workspace["organization_id"],
                "workspace_id": workspace["workspace_id"],
                "status": "active",
                "config": {"target_root": str(tmp_path / "route-archive")},
            },
        )
        binding = integration_resp.json()["binding"]
        execute_resp = client.post(
            f"/platform/integrations/{binding['binding_id']}/execute",
            json={
                "action_type": "sync_export",
                "dossier_id": dossier["dossier_id"],
                "pack_type": "axiom_summary_pack",
                "export_format": "json",
            },
        )
        history_resp = client.get(f"/platform/integrations/{binding['binding_id']}/history")
        dashboard_resp = client.get(
            "/platform/dashboard",
            params={"workspace_id": workspace["workspace_id"]},
        )
        analytics_resp = client.get(
            "/platform/analytics",
            params={"workspace_id": workspace["workspace_id"]},
        )
        workspace_analytics_resp = client.get(
            f"/platform/workspaces/{workspace['workspace_id']}/analytics"
        )
        snapshot_resp = client.get(
            "/platform/demo/snapshot",
            params={"workspace_id": workspace["workspace_id"]},
        )
        readiness_resp = client.get(
            "/platform/demo/readiness",
            params={"workspace_id": workspace["workspace_id"]},
        )

    assert workspace_resp.status_code == 200
    assert workflow_resp.status_code == 200
    assert dossier_resp.status_code == 200
    assert attach_resp.status_code == 200
    assert preview_resp.status_code == 200
    assert preview_resp.json()["preview"]["export_format"] == "markdown"
    assert render_resp.status_code == 200
    assert render_resp.json()["rendered_export"]["content_type"].startswith("text/html")
    assert export_detail.status_code == 200
    assert export_detail.json()["rendered_export"]["render_id"] == render_id
    assert execute_resp.status_code == 200
    assert Path(execute_resp.json()["execution"]["output_summary"]["archive_path"]).exists()
    assert history_resp.status_code == 200
    assert history_resp.json()["history"][0]["status"] == "completed"
    assert dashboard_resp.status_code == 200
    assert dashboard_resp.json()["dashboard"]["executive_metrics"]["dossier_count"] == 1
    assert analytics_resp.status_code == 200
    assert analytics_resp.json()["analytics"]["deployability_distribution"]["paper_trade_only"] == 1
    assert workspace_analytics_resp.status_code == 200
    assert workspace_analytics_resp.json()["analytics"]["dossier_count"] == 1
    assert snapshot_resp.status_code == 200
    assert snapshot_resp.json()["snapshot"]["top_dossiers"][0]["symbol"] == "NVDA"
    assert readiness_resp.status_code == 200
    assert readiness_resp.json()["readiness"]["export_readiness"] == "ready"
