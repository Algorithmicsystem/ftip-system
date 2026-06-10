from __future__ import annotations

from api import source_governance
from api.assistant import reports
from api.assistant.phase13 import (
    SOURCE_GOVERNANCE_VERSION,
    build_source_governance_artifact,
)


def _base_report() -> dict:
    return reports.build_analysis_report(
        symbol="NVDA",
        as_of_date="2024-01-02",
        horizon="swing",
        risk_mode="balanced",
        signal={
            "action": "BUY",
            "score": 0.81,
            "confidence": 0.68,
            "reason_codes": ["TREND_UP"],
            "reason_details": {"TREND_UP": "Trend remains constructive."},
        },
        key_features={"ret_21d": 0.08, "vol_21d": 0.25, "regime_label": "trend"},
        quality={"bars_ok": True, "news_ok": True, "sentiment_ok": True, "warnings": []},
        evidence={
            "reason_codes": ["TREND_UP"],
            "reason_details": {"TREND_UP": "Trend remains constructive."},
            "sources": ["market_bars_daily", "news_raw"],
        },
        data_bundle={
            "quality_provenance": {
                "source_map": {
                    "fundamental_filing": ["sec_edgar", "finnhub_profile"],
                    "sentiment_narrative_flow": ["gnews", "google_news_rss"],
                    "macro_cross_asset": ["fred", "world_bank"],
                },
                "provider_status_map": {
                    "fundamental_filing": {"sec_edgar": {"status": "ok"}},
                    "sentiment_narrative_flow": {
                        "gnews": {"status": "ok"},
                        "google_news_rss": {"status": "ok"},
                    },
                },
            },
            "canonical_alpha_core": {
                "feature_meta": {
                    "price_source": "alphavantage",
                    "event_source": "fundamentals+news_heuristic",
                    "breadth_source": "stooq",
                }
            },
        },
        strategy={
            "final_signal": "HOLD",
            "strategy_posture": "watchlist_positive",
            "confidence": 0.57,
            "confidence_score": 57.0,
            "conviction_tier": "moderate",
            "actionability_score": 44.0,
            "scenario_matrix": {"base": {"summary": "Constructive but not fully actionable."}},
        },
    )


def test_source_inventory_classifies_sources_and_profiles(monkeypatch) -> None:
    monkeypatch.setattr(source_governance.config, "source_profile", lambda: "buyer_demo")
    inventory = source_governance.build_source_inventory(profile="buyer_demo")
    indexed = {item["source_name"]: item for item in inventory}

    # yfinance is GOVERNANCE_REQUIRES_COMMERCIAL_REVIEW — allowed in buyer_demo, not restricted
    assert indexed["yfinance"]["source_restriction_flag"] is False
    assert indexed["google_news_rss"]["source_restriction_flag"] is True
    assert indexed["massive_polygon"]["buyer_demo_allowed"] is True
    assert indexed["finnhub"]["requires_legal_review"] is True
    assert indexed["alphavantage"]["production_status"] == "internal_research_only"


def test_source_governance_artifact_flags_cleanup_and_blockers(monkeypatch) -> None:
    monkeypatch.setattr(source_governance.config, "source_profile", lambda: "buyer_demo")
    artifact = build_source_governance_artifact(current_report=_base_report())
    readiness = artifact["commercialization_readiness"]

    assert artifact["source_governance_version"] == SOURCE_GOVERNANCE_VERSION
    assert artifact["source_profile"] == "buyer_demo"
    assert readiness["buyer_demo_suitability"] == "blocked_by_profile"
    assert "google_news_rss" in readiness["high_risk_sources"]
    assert any("google_news_rss" in blocker for blocker in readiness["commercial_blockers"])
    assert readiness["commercial_cleanup_queue"]
    assert artifact["commercialization_readiness_summary"]
    assert artifact["source_governance_summary"]
    assert artifact["buyer_diligence_summary"]


def test_attach_source_governance_context_updates_report_fields(monkeypatch) -> None:
    monkeypatch.setattr(source_governance.config, "source_profile", lambda: "commercial_candidate")
    report = _base_report()
    artifact = build_source_governance_artifact(current_report=report)
    enriched = reports.attach_source_governance_context(
        report,
        artifact,
        source_governance_artifact_id="source-governance-1",
    )

    assert enriched["source_governance_artifact_id"] == "source-governance-1"
    assert enriched["source_governance"]["source_governance_version"] == SOURCE_GOVERNANCE_VERSION
    assert enriched["source_profile"] == "commercial_candidate"
    assert enriched["commercialization_readiness_summary"]
    assert enriched["source_governance_summary"]
    assert enriched["buyer_diligence_summary"]
    assert "commercialization_readiness_summary" in (enriched["evidence_map"] or {})
