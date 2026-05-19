from __future__ import annotations

import datetime as dt
from typing import Any

from fastapi.testclient import TestClient

from api.main import app
from api.platform import service as platform_service
from api.platform.persistence import PlatformStore, platform_store


def _reset_platform_store(store: PlatformStore) -> None:
    store.use_memory = True
    store._organizations.clear()
    store._workspaces.clear()
    store._entities.clear()
    store._workflows.clear()
    store._dossiers.clear()
    store._dossier_links.clear()


def _sample_platform_report(symbol: str = "NVDA") -> dict[str, Any]:
    return {
        "symbol": symbol,
        "as_of_date": "2024-01-02",
        "overall_analysis": f"{symbol} remains constructive but still governed by disciplined deployability controls.",
        "signal_summary": f"{symbol} remains a constructive but still gated opportunity.",
        "strategy_view": "The workflow still prefers evidence review before wider deployment.",
        "risk_quality_analysis": "Critical fragility remains contained but still requires monitoring.",
        "execution_quality_analysis": "Liquidity and execution quality are adequate for controlled institutional review.",
        "portfolio_fit_analysis": "Portfolio fit is constructive but still sensitive to overlap and evidence status.",
        "deployment_permission_analysis": "Paper-trade-only remains appropriate until calibration and evidence improve.",
        "deployment_permission": "paper_shadow_only",
        "monitoring_triggers": [
            "Macro alignment deteriorates materially.",
            "Price confirms with stronger volume.",
        ],
        "data_bundle": {"symbol_meta": {"sector": "Technology"}},
        "axiom_summary": "AXIOM sees a constructive convergence setup with measured deployability.",
        "axiom_summary_card_text": "Constructive convergence setup with measured deployability and controlled size.",
        "axiom_summary_card": {
            "symbol": symbol,
            "regime_label": "fundamental_convergence",
            "trade_family": "convergence",
            "deployability_tier": "paper_trade_only",
            "size_band": "small",
            "evidence_status": "limited",
        },
        "axiom_regime_label": "fundamental_convergence",
        "axiom_trade_family": "convergence",
        "axiom_deployability_tier": "paper_trade_only",
        "axiom_evidence_backed_deployability_tier": "paper_trade_only",
        "axiom_size_band_recommendation": "small",
        "axiom_final_size_band": "small",
        "axiom_portfolio_fit_label": "watchlist_only",
        "axiom_historical_evidence_report": {
            "status": "available",
            "history_horizon_label": "21d",
            "recent_symbol_evidence": [f"{symbol} has limited but constructive history."],
        },
        "axiom_historical_evidence_summary_text": "Historical evidence is available but still sample-constrained.",
        "axiom_calibration_summary_text": "Calibration remains constructive but still limited in mature sample depth.",
        "axiom_portfolio_governance_summary": "Portfolio governance still caps the size band at small.",
        "axiom_portfolio_governance": {
            "symbol": symbol,
            "portfolio_rank_score": 58.0,
            "portfolio_fit_label": "watchlist_only",
            "final_size_band": "small",
            "rationale": "Fit remains constructive but still constrained by overlap and evidence depth.",
        },
        "axiom_risk_deployability_memo": {
            "fragility_engine": {"score": 31.0},
            "liquidity_convexity_engine": {"score": 69.0},
            "downgrade_triggers": [
                "Research integrity weakens materially.",
                "Execution quality deteriorates.",
            ],
        },
        "axiom_lineage_summary": "Lineage is primarily direct-source with a few partial historical replay estimates.",
        "axiom_lineage": {
            "lineage_version": "axiom50_phase4_lineage_v1",
            "engine_lineage": {
                "critical_fragility": {
                    "engine": "critical_fragility",
                    "confidence": 63.0,
                    "blocks": [
                        {
                            "component": "gap_jump_risk_component",
                            "derived_from": [
                                "market_price_volume.gap_frequency_21d",
                                "fragility_intelligence.instability_score",
                            ],
                            "evidence_type": "direct_source",
                            "coverage_status": "available",
                        }
                    ],
                }
            },
        },
        "axiom_artifact_id": "axiom-1",
        "axiom_history_artifact_id": "axiom-history-1",
        "axiom_calibration_artifact_id": "axiom-calibration-1",
        "axiom_report_pack_artifact_id": "axiom-report-pack-1",
        "axiom_lineage_artifact_id": "axiom-lineage-1",
    }


def _patch_analyze(monkeypatch) -> None:
    from api.assistant import orchestrator

    async def _fake_freshness(symbol: str, refresh: bool = True):
        return {
            "as_of_date": dt.date(2024, 1, 2),
            "bars_ok": True,
            "news_ok": True,
            "sentiment_ok": True,
            "bars_updated_at": "2024-01-02T00:00:00Z",
            "news_updated_at": "2024-01-02T00:00:00Z",
            "sentiment_updated_at": "2024-01-02T00:00:00Z",
            "warnings": [],
        }

    async def _noop(*_args, **_kwargs):
        return None

    monkeypatch.setattr(orchestrator, "ensure_freshness", _fake_freshness)
    monkeypatch.setattr(orchestrator, "run_features", _noop)
    monkeypatch.setattr(orchestrator, "run_signals", _noop)
    monkeypatch.setattr(
        orchestrator,
        "fetch_signal",
        lambda *_args, **_kwargs: {
            "action": "BUY",
            "score": 0.7,
            "confidence": 0.6,
            "entry_low": 100,
            "entry_high": 110,
            "stop_loss": 90,
            "take_profit_1": 130,
            "take_profit_2": 150,
            "horizon_days": 10,
            "reason_codes": ["MOMO_UP"],
            "reason_details": {"MOMO_UP": "Momentum rising"},
        },
    )
    monkeypatch.setattr(
        orchestrator,
        "fetch_key_features",
        lambda *_args, **_kwargs: {"ret_5d": 0.12, "vol_21d": 0.3, "regime_label": "trend"},
    )
    monkeypatch.setattr(
        orchestrator,
        "fetch_quality",
        lambda *_args, **_kwargs: {
            "bars_ok": True,
            "news_ok": True,
            "sentiment_ok": True,
            "warnings": [],
        },
    )


def test_platform_service_builds_workspace_workflow_and_dossier() -> None:
    store = PlatformStore(use_memory=True)
    workspace_response = platform_service.create_workspace_service(
        {
            "name": "HF Core Workspace",
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
            "title": "NVDA Hedge Fund Research",
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
            "title": "NVDA Institutional Dossier",
        },
        store=store,
    )
    dossier = dossier_response["dossier"]

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
        store=store,
    )

    refreshed = attach_response["dossier"]
    assert refreshed["latest_axiom_analysis_id"] == "axiom-1"
    assert refreshed["latest_deployability_tier"] == "paper_trade_only"
    assert refreshed["latest_regime_label"] == "fundamental_convergence"
    assert refreshed["latest_size_band"] == "small"
    assert refreshed["evidence_status"] == "limited"
    assert len(refreshed["sections"]) >= 10
    assert attach_response["analysis_link"]["report_id"] == "report-1"
    assert attach_response["analysis_link"]["session_id"] == "session-1"

    dossier_view = platform_service.get_dossier_view(refreshed["dossier_id"], store=store)
    assert dossier_view["analysis_links"][0]["axiom_artifact_id"] == "axiom-1"
    assert dossier_view["workspace"]["name"] == "HF Core Workspace"

    summary = platform_service.build_platform_summary_service(
        workspace_id=workspace["workspace_id"],
        store=store,
    )
    assert summary["workspace_count"] == 1
    assert summary["workflow_count"] == 1
    assert summary["dossier_count"] == 1
    assert summary["dossiers_by_deployability_tier"]["paper_trade_only"] == 1
    assert summary["dossiers_by_regime"]["fundamental_convergence"] == 1
    assert summary["dossiers_by_workflow_stage"]["intake"] == 1


def test_platform_routes_support_workspace_dossier_flow() -> None:
    _reset_platform_store(platform_store)

    with TestClient(app) as client:
        templates = client.get("/platform/templates")
        workspace_resp = client.post(
            "/platform/workspaces",
            json={
                "name": "HF Core Workspace",
                "platform_profile": "hf_core",
                "audience_type": "hedge_fund",
                "report_profile": "ic_memo",
            },
        )
        workspace_id = workspace_resp.json()["workspace"]["workspace_id"]
        workflow_resp = client.post(
            "/platform/workflows",
            json={
                "workspace_id": workspace_id,
                "workflow_template_id": "hedge_fund_research",
                "title": "MSFT Hedge Fund Research",
            },
        )
        workflow_id = workflow_resp.json()["workflow"]["workflow_id"]
        dossier_resp = client.post(
            "/platform/dossiers",
            json={
                "workflow_id": workflow_id,
                "symbol": "MSFT",
                "display_name": "Microsoft",
                "dossier_type": "coverage",
                "title": "MSFT Institutional Dossier",
            },
        )
        dossier_id = dossier_resp.json()["dossier"]["dossier_id"]
        attach_resp = client.post(
            f"/platform/dossiers/{dossier_id}/attach-analysis",
            json={
                "report": _sample_platform_report("MSFT"),
                "report_id": "report-route-1",
                "session_id": "session-route-1",
                "axiom_artifact_id": "axiom-route-1",
                "axiom_report_pack_artifact_id": "axiom-pack-route-1",
                "axiom_lineage_artifact_id": "axiom-lineage-route-1",
                "axiom_history_artifact_id": "axiom-history-route-1",
                "axiom_calibration_artifact_id": "axiom-cal-route-1",
            },
        )
        dossier_detail = client.get(f"/platform/dossiers/{dossier_id}")
        summary = client.get(f"/platform/summary?workspace_id={workspace_id}")

    assert templates.status_code == 200
    assert any(
        item["template_id"] == "hedge_fund_research"
        for item in templates.json()["workflow_templates"]
    )
    assert any(
        item["profile_id"] == "hf_core"
        for item in templates.json()["platform_profiles"]
    )
    assert workspace_resp.status_code == 200
    assert workflow_resp.status_code == 200
    assert dossier_resp.status_code == 200
    assert attach_resp.status_code == 200
    assert attach_resp.json()["dossier"]["latest_axiom_analysis_id"] == "axiom-route-1"
    assert attach_resp.json()["analysis_link"]["report_id"] == "report-route-1"
    assert dossier_detail.status_code == 200
    assert dossier_detail.json()["analysis_links"][0]["session_id"] == "session-route-1"
    assert summary.status_code == 200
    assert summary.json()["dossier_count"] == 1
    assert summary.json()["dossiers_by_deployability_tier"]["paper_trade_only"] == 1


def test_assistant_analyze_can_create_platform_dossier(monkeypatch) -> None:
    _reset_platform_store(platform_store)
    _patch_analyze(monkeypatch)

    with TestClient(app) as client:
        resp = client.post(
            "/assistant/analyze",
            json={
                "symbol": "NVDA",
                "horizon": "swing",
                "risk_mode": "balanced",
                "audience_type": "hedge_fund",
                "report_profile": "ic_memo",
                "platform_profile": "hf_core",
                "workflow_template_id": "hedge_fund_research",
                "create_dossier": True,
            },
        )

    assert resp.status_code == 200
    data = resp.json()
    assert data["platform_foundation_version"] == platform_service.PLATFORM_FOUNDATION_VERSION
    assert data["platform_profile"] == "hf_core"
    assert data["platform_profile_details"]["profile_id"] == "hf_core"
    assert data["platform_workspace"]["workspace_id"]
    assert data["platform_workflow"]["workflow_template_id"] == "hedge_fund_research"
    assert data["platform_dossier"]["dossier_id"]
    assert data["platform_dossier"]["latest_axiom_analysis_id"] == data["axiom_artifact_id"]
    assert data["platform_analysis_link"]["report_id"] == data["report_id"]
    assert data["platform_dossier_summary"]
    assert data["platform_monitoring_summary"]
    assert data["platform_summary_view"]["dossier_count"] >= 1
    assert data["active_analysis"]["platform_profile"] == "hf_core"
    assert (
        data["active_analysis"]["platform_dossier_id"]
        == data["platform_dossier"]["dossier_id"]
    )
    assert data["active_analysis"]["platform_workflow_stage"] == data["platform_workflow"]["stage"]
