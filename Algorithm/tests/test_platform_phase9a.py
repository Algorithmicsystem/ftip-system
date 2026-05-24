from __future__ import annotations

import datetime as dt
from typing import Any

from fastapi.testclient import TestClient

from api.main import app
from api.platform import service as platform_service
from api.platform.persistence import PlatformStore, platform_store
from tests.test_platform_phase5 import _reset_platform_store, _sample_platform_report
from tests.test_platform_phase6 import _bootstrap_workspace, _headers
from tests.test_platform_phase7 import _attach_sample_analysis


def _synthetic_bar_fetcher(symbol: str, as_of_date: dt.date, limit: int) -> list[dict[str, Any]]:
    pattern_by_symbol = {
        "NVDA": 0.012,
        "MSFT": 0.006,
        "AMD": 0.009,
        "SHOP": 0.004,
        "TSLA": -0.009,
        "META": -0.006,
        "NFLX": -0.004,
        "AAPL": -0.002,
    }
    slope = pattern_by_symbol.get(str(symbol or "").upper(), 0.001)
    rows: list[dict[str, Any]] = []
    price = 100.0
    for offset in range(limit):
        date_value = as_of_date + dt.timedelta(days=offset)
        if offset:
            price *= 1.0 + slope
        high = price * (1.0 + max(abs(slope), 0.004) * 0.7)
        low = price * (1.0 - max(abs(slope), 0.004) * 0.7)
        rows.append(
            {
                "as_of_date": date_value.isoformat(),
                "open": round(price * 0.998, 6),
                "high": round(high, 6),
                "low": round(low, 6),
                "close": round(price, 6),
                "volume": 1_000_000 + offset * 1_000,
            }
        )
    return rows


def _attach_custom_analysis(
    store: PlatformStore,
    dossier_id: str,
    *,
    symbol: str,
    report_id: str,
    session_id: str,
    deployability_tier: str,
    regime_label: str,
    trade_family: str,
    dau: float,
    validated_edge: float,
    size_band: str = "small",
) -> dict[str, Any]:
    report = _sample_platform_report(symbol)
    report["axiom_summary_card"]["deployability_tier"] = deployability_tier
    report["axiom_summary_card"]["regime_label"] = regime_label
    report["axiom_summary_card"]["trade_family"] = trade_family
    report["axiom_summary_card"]["size_band"] = size_band
    report["axiom_summary_card"]["deployable_alpha_utility"] = dau
    report["axiom_summary_card"]["validated_edge"] = validated_edge
    report["axiom_regime_label"] = regime_label
    report["axiom_trade_family"] = trade_family
    report["axiom_deployability_tier"] = deployability_tier
    report["axiom_evidence_backed_deployability_tier"] = deployability_tier
    report["axiom_deployable_alpha_utility"] = dau
    report["axiom_validated_edge"] = validated_edge
    report["axiom_final_size_band"] = size_band
    report["axiom_size_band_recommendation"] = size_band
    report["axiom_portfolio_governance"]["final_size_band"] = size_band
    report["axiom_portfolio_governance"]["portfolio_fit_label"] = (
        "core_candidate" if deployability_tier == "live_candidate" else "watchlist_only"
    )
    report["data_bundle"]["market_price_volume"]["latest_close"] = 100.0
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


def _seed_tracked_dossiers(
    store: PlatformStore,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    workspace, workflow, dossier = _bootstrap_workspace(store)
    configs = [
        ("NVDA", "live_candidate", "fundamental_convergence", "convergence", 82.0, 61.0, "2024-01-02"),
        ("MSFT", "live_candidate", "compensation_capture", "compensation", 79.0, 58.0, "2024-01-05"),
        ("AMD", "paper_trade_only", "behavioral_continuation", "transmission", 71.0, 52.0, "2024-01-08"),
        ("SHOP", "paper_trade_only", "recovery_reset", "recovery", 67.0, 49.0, "2024-01-12"),
        ("TSLA", "monitor_only", "euphoria_critical", "none", 44.0, 31.0, "2024-02-02"),
        ("META", "paper_trade_only", "liquidity_fracture", "convexity", 51.0, 37.0, "2024-02-05"),
        ("NFLX", "monitor_only", "behavioral_continuation", "transmission", 47.0, 34.0, "2024-02-08"),
        ("AAPL", "monitor_only", "indeterminate", "none", 41.0, 28.0, "2024-02-12"),
    ]
    created: list[dict[str, Any]] = []

    for index, (
        symbol,
        tier,
        regime,
        trade_family,
        dau,
        validated_edge,
        tracking_start,
    ) in enumerate(configs):
        if index == 0:
            current_workflow = workflow
            current_dossier = dossier
        else:
            current_workflow = platform_service.create_workflow_service(
                {
                    "workspace_id": workspace["workspace_id"],
                    "workflow_template_id": "hedge_fund_research",
                    "title": f"{symbol} Proof Workflow",
                },
                store=store,
            )["workflow"]
            current_dossier = platform_service.create_dossier_service(
                {
                    "workflow_id": current_workflow["workflow_id"],
                    "symbol": symbol,
                    "display_name": symbol,
                    "title": f"{symbol} Proof Dossier",
                },
                store=store,
            )["dossier"]

        _attach_custom_analysis(
            store,
            current_dossier["dossier_id"],
            symbol=symbol,
            report_id=f"report-{symbol.lower()}",
            session_id=f"session-{symbol.lower()}",
            deployability_tier=tier,
            regime_label=regime,
            trade_family=trade_family,
            dau=dau,
            validated_edge=validated_edge,
            size_band="medium" if tier == "live_candidate" else "small",
        )
        tracking = platform_service.start_dossier_tracking_service(
            current_dossier["dossier_id"],
            {
                "tracking_start_at": tracking_start,
                "tracked_horizons": ["5d", "21d", "63d"],
            },
            store=store,
        )
        outcomes = platform_service.get_dossier_outcomes_service(
            current_dossier["dossier_id"],
            store=store,
        )
        created.append(
            {
                "workflow": current_workflow,
                "dossier": current_dossier,
                "tracking": tracking,
                "outcomes": outcomes,
            }
        )
    return workspace, created


def test_phase9a_tracking_and_outcomes_create_proof_objects(monkeypatch) -> None:
    monkeypatch.setattr(
        "api.platform.outcomes.default_ohlc_bar_fetcher",
        _synthetic_bar_fetcher,
    )
    store = PlatformStore(use_memory=True)
    workspace, workflow, dossier = _bootstrap_workspace(store)
    _attach_sample_analysis(
        store,
        dossier["dossier_id"],
        symbol="NVDA",
        report_id="report-1",
        session_id="session-1",
    )

    tracking = platform_service.start_dossier_tracking_service(
        dossier["dossier_id"],
        {
            "tracking_start_at": "2024-01-02",
            "tracked_horizons": ["5d", "21d", "63d"],
        },
        store=store,
    )
    assert tracking["track"]["symbol"] == "NVDA"
    assert tracking["paper_trade"]["entry_reference_date"] == "2024-01-02"
    assert tracking["outcome_snapshot"]["tracking_status"]["status"] in {"active", "complete"}

    refreshed = platform_service.get_dossier_outcomes_service(
        dossier["dossier_id"],
        store=store,
    )
    assert refreshed["outcome_snapshot"]["assessment"]["assessment_status"] in {
        "supportive",
        "mixed",
        "weak",
    }
    assert "21d" in refreshed["outcome_snapshot"]["windows"]
    assert refreshed["outcome_attribution"]["source_label"] == "paper_tracked"

    dossier_view = platform_service.get_dossier_view(
        dossier["dossier_id"],
        store=store,
    )
    assert dossier_view["recommendation_evidence_summary"]["evidence_status"] in {
        "supportive",
        "mixed",
        "weak",
    }
    section_keys = [item["section_key"] for item in dossier_view["dossier"]["sections"]]
    assert "proof_summary" in section_keys
    assert "outcome_attribution" in section_keys

    export = platform_service.create_dossier_export_service(
        dossier["dossier_id"],
        {"pack_type": "historical_evidence_pack"},
        store=store,
    )["export"]
    export_keys = [item["section_key"] for item in export["ordered_sections"]]
    assert "proof_summary" in export_keys
    assert "outcome_attribution" in export_keys


def test_phase9a_workspace_proof_calibration_drift_and_benchmarks(monkeypatch) -> None:
    monkeypatch.setattr(
        "api.platform.outcomes.default_ohlc_bar_fetcher",
        _synthetic_bar_fetcher,
    )
    store = PlatformStore(use_memory=True)
    workspace, seeded = _seed_tracked_dossiers(store)
    assert len(seeded) == 8

    proof = platform_service.build_workspace_proof_service(
        workspace["workspace_id"],
        store=store,
    )
    assert proof["proof_summary"]["tracked_recommendation_count"] == 8
    assert proof["calibration_hardening"]["matured_count"] >= 8
    assert proof["calibration_hardening"]["status"] == "available"
    assert proof["benchmarks"]["status"] == "available"
    assert proof["model_credibility_snapshot"]["status"] in {
        "supportive",
        "partial",
        "limited",
    }

    drift = platform_service.build_workspace_drift_service(
        workspace["workspace_id"],
        store=store,
    )["drift_summary"]
    assert drift["status"] in {"degrading", "stable", "improving"}
    assert drift["matured_count"] >= 8

    benchmarks = platform_service.build_workspace_benchmarks_service(
        workspace["workspace_id"],
        store=store,
    )["benchmarks"]
    assert benchmarks["cohorts"]["deployability_tier"]
    assert benchmarks["cohorts"]["regime"]
    assert benchmarks["strongest_cohorts"]

    summary = platform_service.build_proof_summary_service(store=store)
    assert summary["proof_summary"]["tracked_recommendation_count"] == 8


def test_phase9a_routes_are_tenant_safe_for_tracking_and_proof(monkeypatch) -> None:
    monkeypatch.setattr(
        "api.platform.outcomes.default_ohlc_bar_fetcher",
        _synthetic_bar_fetcher,
    )
    _reset_platform_store(platform_store)
    workspace_one, workflow_one, dossier_one = _bootstrap_workspace(platform_store)
    _attach_sample_analysis(
        platform_store,
        dossier_one["dossier_id"],
        symbol="NVDA",
        report_id="report-1",
        session_id="session-1",
    )
    workspace_two = platform_service.create_workspace_service(
        {
            "name": "PE Proof Workspace",
            "platform_profile": "pe_core",
            "audience_type": "private_equity",
            "report_profile": "diligence_focused",
        },
        store=platform_store,
    )["workspace"]
    workflow_two = platform_service.create_workflow_service(
        {
            "workspace_id": workspace_two["workspace_id"],
            "workflow_template_id": "private_equity_diligence",
            "title": "MSFT Proof Workflow",
        },
        store=platform_store,
    )["workflow"]
    dossier_two = platform_service.create_dossier_service(
        {
            "workflow_id": workflow_two["workflow_id"],
            "symbol": "MSFT",
            "display_name": "Microsoft",
            "title": "MSFT Proof Dossier",
        },
        store=platform_store,
    )["dossier"]
    _attach_sample_analysis(
        platform_store,
        dossier_two["dossier_id"],
        symbol="MSFT",
        report_id="report-2",
        session_id="session-2",
    )
    platform_service.create_membership_service(
        {
            "user_id": "analyst-1",
            "workspace_id": workspace_one["workspace_id"],
            "organization_id": workspace_one["organization_id"],
            "role": "analyst",
        },
        store=platform_store,
    )
    platform_service.create_membership_service(
        {
            "user_id": "analyst-2",
            "workspace_id": workspace_two["workspace_id"],
            "organization_id": workspace_two["organization_id"],
            "role": "analyst",
        },
        store=platform_store,
    )

    with TestClient(app) as client:
        start = client.post(
            f"/platform/dossiers/{dossier_one['dossier_id']}/start-tracking",
            json={"tracking_start_at": "2024-01-02", "tracked_horizons": ["5d", "21d"]},
            headers=_headers("analyst-1", "analyst", workspace_id=workspace_one["workspace_id"]),
        )
        assert start.status_code == 200

        own_tracking = client.get(
            f"/platform/dossiers/{dossier_one['dossier_id']}/tracking",
            headers=_headers("analyst-1", "analyst", workspace_id=workspace_one["workspace_id"]),
        )
        assert own_tracking.status_code == 200

        own_outcomes = client.get(
            f"/platform/dossiers/{dossier_one['dossier_id']}/outcomes",
            headers=_headers("analyst-1", "analyst", workspace_id=workspace_one["workspace_id"]),
        )
        assert own_outcomes.status_code == 200

        own_proof = client.get(
            f"/platform/workspaces/{workspace_one['workspace_id']}/proof",
            headers=_headers("analyst-1", "analyst", workspace_id=workspace_one["workspace_id"]),
        )
        assert own_proof.status_code == 200

        foreign_tracking = client.get(
            f"/platform/dossiers/{dossier_one['dossier_id']}/tracking",
            headers=_headers("analyst-2", "analyst", workspace_id=workspace_two["workspace_id"]),
        )
        assert foreign_tracking.status_code == 403

        foreign_proof = client.get(
            f"/platform/workspaces/{workspace_one['workspace_id']}/proof",
            headers=_headers("analyst-2", "analyst", workspace_id=workspace_two["workspace_id"]),
        )
        assert foreign_proof.status_code == 403


def test_phase9a_audit_events_capture_tracking_and_proof_generation(monkeypatch) -> None:
    monkeypatch.setattr(
        "api.platform.outcomes.default_ohlc_bar_fetcher",
        _synthetic_bar_fetcher,
    )
    store = PlatformStore(use_memory=True)
    workspace, workflow, dossier = _bootstrap_workspace(store)
    _attach_sample_analysis(
        store,
        dossier["dossier_id"],
        symbol="NVDA",
        report_id="report-1",
        session_id="session-1",
    )
    platform_service.create_membership_service(
        {
            "user_id": "analyst-1",
            "workspace_id": workspace["workspace_id"],
            "organization_id": workspace["organization_id"],
            "role": "analyst",
        },
        store=store,
    )

    platform_service.start_dossier_tracking_service(
        dossier["dossier_id"],
        {"tracking_start_at": "2024-01-02"},
        user_context={
            "user_id": "analyst-1",
            "role": "analyst",
            "organization_id": workspace["organization_id"],
            "workspace_id": workspace["workspace_id"],
            "email": "analyst@example.com",
            "session_id": "session-proof-1",
        },
        store=store,
    )
    platform_service.get_dossier_outcomes_service(
        dossier["dossier_id"],
        user_context={
            "user_id": "analyst-1",
            "role": "analyst",
            "organization_id": workspace["organization_id"],
            "workspace_id": workspace["workspace_id"],
            "email": "analyst@example.com",
            "session_id": "session-proof-1",
        },
        store=store,
    )
    platform_service.build_workspace_proof_service(
        workspace["workspace_id"],
        user_context={
            "user_id": "analyst-1",
            "role": "analyst",
            "organization_id": workspace["organization_id"],
            "workspace_id": workspace["workspace_id"],
            "email": "analyst@example.com",
            "session_id": "session-proof-1",
        },
        store=store,
        emit_audit=True,
    )
    platform_service.build_workspace_drift_service(
        workspace["workspace_id"],
        user_context={
            "user_id": "analyst-1",
            "role": "analyst",
            "organization_id": workspace["organization_id"],
            "workspace_id": workspace["workspace_id"],
            "email": "analyst@example.com",
            "session_id": "session-proof-1",
        },
        store=store,
        emit_audit=True,
    )

    event_types = {item["event_type"] for item in store.list_audit_events(workspace_id=workspace["workspace_id"])}
    assert "tracking_started" in event_types
    assert "tracking_refreshed" in event_types
    assert "outcome_snapshot_generated" in event_types
    assert "proof_summary_generated" in event_types
