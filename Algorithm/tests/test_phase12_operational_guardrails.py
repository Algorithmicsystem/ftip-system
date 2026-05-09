from __future__ import annotations

from api.assistant import reports
from api.assistant.phase12 import (
    OPERATIONAL_GUARDRAILS_VERSION,
    build_operational_guardrails_artifact,
)
from api.assistant.phase12 import health as health_module
from api.assistant.phase5.context import build_narrator_context
from api.assistant.phase5.routing import route_question
from api.assistant.storage import AssistantStorage


def _base_report() -> dict:
    report = reports.build_analysis_report(
        symbol="NVDA",
        as_of_date="2024-01-02",
        horizon="swing",
        risk_mode="balanced",
        signal={
            "action": "BUY",
            "score": 0.73,
            "confidence": 0.64,
            "reason_codes": ["TREND_UP"],
            "reason_details": {"TREND_UP": "Trend remains constructive."},
        },
        key_features={"ret_21d": 0.08, "vol_21d": 0.25, "regime_label": "trend"},
        quality={"bars_ok": True, "news_ok": True, "sentiment_ok": True, "warnings": []},
        evidence={
            "reason_codes": ["TREND_UP"],
            "reason_details": {"TREND_UP": "Trend remains constructive."},
            "sources": ["market_bars_daily", "news_raw", "sentiment_daily"],
        },
        data_bundle={
            "quality_provenance": {
                "quality_score": 64,
                "freshness_summary": {
                    "market": {"status": "fresh", "updated_at": "2024-01-02T00:00:00Z"},
                    "macro": {
                        "status": "stale_but_usable",
                        "updated_at": "2023-12-20T00:00:00Z",
                    },
                    "cross_asset": {
                        "status": "mixed",
                        "updated_at": "2023-12-28T00:00:00Z",
                    },
                    "event": {
                        "status": "stale_but_usable",
                        "updated_at": "2023-12-22T00:00:00Z",
                    },
                },
                "domain_availability": {
                    "market": {
                        "coverage_status": "available",
                        "fallback_used": False,
                        "fallback_source": [],
                    },
                    "macro": {
                        "coverage_status": "partial",
                        "fallback_used": True,
                        "fallback_source": ["SPY"],
                    },
                    "cross_asset": {
                        "coverage_status": "partial",
                        "fallback_used": True,
                        "fallback_source": ["QQQ"],
                    },
                    "event": {
                        "coverage_status": "available",
                        "fallback_used": False,
                        "fallback_source": [],
                    },
                },
                "provider_notes": ["macro coverage is relying on fallback sources"],
            },
            "external_data_fabric": {"status": "mixed"},
            "canonical_alpha_core": {
                "feature_vector": {
                    "market_stress_score": 63.0,
                    "cross_asset_conflict_score": 58.0,
                    "breadth_confirmation_score": 42.0,
                }
            },
        },
        feature_factor_bundle={
            "regime_intelligence": {"regime_instability": 57.0},
        },
        strategy={
            "strategy_version": "phase4_institutional_v1",
            "final_signal": "HOLD",
            "strategy_posture": "watchlist_positive",
            "confidence": 0.57,
            "confidence_score": 57.0,
            "conviction_tier": "moderate",
            "actionability_score": 44.0,
            "scenario_matrix": {"base": {"summary": "Constructive but not fully actionable."}},
        },
    )
    report = reports.attach_canonical_validation_context(
        report,
        {
            "validation_version": "phase10_research_truth_v1",
            "status": "available",
            "prediction_linkage_summary": {"matured_count": 12},
            "walkforward_summary": {"window_count": 2},
            "net_return_summary": {"average_edge_return": 0.011, "hit_rate": 0.58},
            "friction_cost_summary": {"average_cost_drag": 0.003},
            "readiness_scorecard": {
                "paper_vs_live_candidate_quality_summary": 0.02
            },
            "suppression_effect_summary": {
                "suppression_effect_edge_spread": 0.01
            },
            "validation_summary": "Canonical validation is available.",
            "walkforward_validation_summary": "Walk-forward remains available.",
            "net_of_friction_summary": "Net edge remains positive after friction.",
            "suppression_readiness_validation_summary": "Suppression and readiness remain helpful.",
            "drawdown_invalidation_summary": "Drawdown remains controlled.",
        },
        canonical_validation_artifact_id="validation-ops",
    )
    report = reports.attach_deployment_context(
        report,
        {
            "deployment_readiness_version": "phase8_capital_readiness_v1",
            "deployment_mode": {
                "active_mode": "limited_live",
                "rollout_stage": "limited_live_monitoring",
            },
            "model_readiness": {
                "model_readiness_status": "constrained",
                "live_readiness_score": 56.0,
                "live_readiness_blockers": [
                    "confidence calibration quality is not strong enough for live escalation"
                ],
            },
            "signal_admission_control": {
                "admitted_for_strategy": True,
                "admitted_for_paper": True,
                "admitted_for_live": False,
            },
            "deployment_permission": {
                "deployment_permission": "paper_shadow_only",
                "deployment_blockers": ["fragility remains too high for live admission"],
                "deployment_rationale": "The setup is paper-worthy, but it does not clear the live gate.",
                "trust_tier": "paper_only",
                "minimum_required_review": "senior_analyst_and_risk_review",
                "human_review_required": True,
            },
            "risk_budgeting": {
                "risk_budget_tier": "shadow_only",
                "exposure_caution_level": "high",
                "fragility_adjusted_size_band": "0.10x-0.25x pilot unit",
                "confidence_adjusted_size_band": "0.10x-0.25x pilot unit",
                "maximum_risk_mode_allowed": "paper_shadow",
            },
            "rollout_workflow": {
                "rollout_stage": "limited_live_monitoring",
                "readiness_checkpoint": "watch",
                "promotion_criteria": ["confidence reliability remains above the stage threshold"],
                "demotion_criteria": ["fragility rises into the blocked zone"],
                "stage_transition_notes": ["Continue paper/shadow evidence collection before live escalation."],
            },
            "drift_monitor": {
                "pause_recommended": False,
                "degrade_to_paper_recommended": True,
                "drift_alerts": ["confidence reliability is below the live-support comfort zone"],
                "deployment_risk_alerts": ["live readiness has slipped below the controlled-live comfort zone"],
            },
            "audit_snapshot": {
                "rationale_summary": "Paper-shadow only until calibration and fragility improve.",
            },
        },
        readiness_artifact_id="readiness-ops",
        deployment_audit_artifact_id="audit-ops",
    )
    report["learning_drift_alerts"] = [
        {
            "affected_component": "confidence_calibration",
            "severity": "moderate",
            "evidence": "Trend-regime calibration is weaker than the cohort median.",
        }
    ]
    return report


def test_operational_guardrails_artifact_builds_health_shadow_and_controls(monkeypatch) -> None:
    store = AssistantStorage(use_memory=True)
    session_id = store.create_session()
    report = _base_report()
    store.save_artifact(session_id, "assistant_shadow_decision_record", {"symbol": "NVDA", "horizon": "swing", "risk_mode": "balanced", "deployment_permission": "paper_shadow_only"})

    monkeypatch.setattr(health_module.config, "llm_enabled", lambda: True)
    monkeypatch.setattr(health_module.db, "db_enabled", lambda: True)
    monkeypatch.setattr(health_module.db, "db_read_enabled", lambda: True)
    monkeypatch.setattr(health_module.db, "db_write_enabled", lambda: True)
    monkeypatch.setattr(
        health_module.metrics_tracker,
        "snapshot",
        lambda: {
            "request_counts": {"/assistant/analyze": 4},
            "status_4xx": 0,
            "status_5xx": 0,
            "rate_limit_hits": 0,
            "snapshot_runs": 2,
            "strategy_graph_runs": 0,
        },
    )

    artifact = build_operational_guardrails_artifact(
        current_report=report,
        current_report_id="report-ops",
        session_id=session_id,
        store=store,
    )

    assert artifact["operational_guardrails_version"] == OPERATIONAL_GUARDRAILS_VERSION
    assert artifact["system_health"]["system_health_status"] in {
        "healthy",
        "watch",
        "degraded",
        "critical",
    }
    assert artifact["shadow_mode"]["shadow_mode_status"] in {
        "active_shadow",
        "live_candidate_monitor",
        "watch_only",
        "paused",
    }
    assert artifact["drift_monitoring"]["model_drift_score"] is not None
    assert artifact["control_state"]["current_operating_mode"] in {
        "normal",
        "increased_review",
        "shadow_only",
        "paused",
    }
    assert artifact["operational_alerts"]
    assert artifact["system_health_summary"]
    assert artifact["shadow_mode_summary"]
    assert artifact["incident_history_summary"]


def test_operational_guardrails_attach_to_report_and_narrator_context(monkeypatch) -> None:
    store = AssistantStorage(use_memory=True)
    session_id = store.create_session()
    report = _base_report()

    monkeypatch.setattr(health_module.config, "llm_enabled", lambda: True)
    monkeypatch.setattr(health_module.db, "db_enabled", lambda: True)
    monkeypatch.setattr(health_module.db, "db_read_enabled", lambda: True)
    monkeypatch.setattr(health_module.db, "db_write_enabled", lambda: True)
    monkeypatch.setattr(
        health_module.metrics_tracker,
        "snapshot",
        lambda: {
            "request_counts": {"/assistant/analyze": 2},
            "status_4xx": 0,
            "status_5xx": 0,
            "rate_limit_hits": 0,
            "snapshot_runs": 1,
            "strategy_graph_runs": 0,
        },
    )

    artifact = build_operational_guardrails_artifact(
        current_report=report,
        current_report_id="report-ops",
        session_id=session_id,
        store=store,
    )
    report = reports.attach_operational_context(
        report,
        artifact,
        operational_guardrails_artifact_id="ops-1",
        health_snapshot_artifact_id="health-1",
        shadow_decision_artifact_id="shadow-1",
        operational_incident_artifact_ids=["incident-1"],
    )
    route = route_question(
        "Is the system healthy right now and why was it downgraded into shadow mode?"
    )
    narrator_context = build_narrator_context(
        report,
        active_analysis={"symbol": "NVDA", "system_health_status": report["system_health_status"]},
        route=route,
        user_message="Is the system healthy right now and why was it downgraded into shadow mode?",
    )

    assert report["system_health_summary"]
    assert report["shadow_mode_summary"]
    assert route["intent"] == "operational_health"
    assert narrator_context["operational_snapshot"]["current_operating_mode"]
    assert "system_health_summary" in narrator_context["section_summaries"]
