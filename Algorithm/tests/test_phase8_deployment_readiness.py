from api.assistant import reports
from api.assistant.phase8 import (
    DEPLOYMENT_AUDIT_RECORD_KIND,
    build_deployment_readiness_artifact,
)
from api.assistant.storage import AssistantStorage


def _sample_report(
    *,
    fragility_score: float = 34.0,
    confidence_score: float = 66.0,
    actionability_score: float = 63.0,
    conflict_score: float = 24.0,
    agreement_score: float = 76.0,
    quality_score: float = 86.0,
    freshness_status: str = "fresh",
    matured_count: int = 16,
    reliability_score: float = 72.0,
    actionable_spread: float = 0.05,
    missingness: float = 0.02,
    fallback_domains: tuple[str, ...] = (),
) -> dict:
    availability = {
        "market": {"coverage_status": "available", "fallback_used": "market" in fallback_domains},
        "fundamentals": {
            "coverage_status": "available",
            "fallback_used": "fundamentals" in fallback_domains,
        },
        "sentiment": {
            "coverage_status": "available",
            "fallback_used": "sentiment" in fallback_domains,
        },
        "macro": {"coverage_status": "available", "fallback_used": "macro" in fallback_domains},
    }
    report = reports.build_analysis_report(
        symbol="NVDA",
        as_of_date="2024-01-02",
        horizon="swing",
        risk_mode="balanced",
        signal={
            "action": "BUY",
            "score": 0.79,
            "confidence": 0.68,
            "entry_low": 100,
            "entry_high": 105,
            "stop_loss": 95,
            "take_profit_1": 112,
            "take_profit_2": 120,
            "horizon_days": 21,
            "reason_codes": ["TREND_UP"],
            "reason_details": {"TREND_UP": "Trend remains constructive."},
        },
        key_features={"ret_21d": 0.09, "vol_21d": 0.22, "regime_label": "trend"},
        quality={
            "bars_ok": True,
            "fundamentals_ok": True,
            "sentiment_ok": True,
            "news_ok": True,
            "warnings": [],
            "anomaly_flags": [],
            "quality_score": quality_score,
            "missingness": missingness,
        },
        evidence={
            "reason_codes": ["TREND_UP"],
            "reason_details": {"TREND_UP": "Trend remains constructive."},
            "sources": ["market_bars_daily", "news_raw", "sentiment_daily"],
        },
        job_context={
            "scenario": "base",
            "analysis_depth": "standard",
            "refresh_mode": "refresh_stale",
            "market_regime": "auto",
        },
        data_bundle={
            "quality_provenance": {
                "quality_score": quality_score,
                "freshness_summary": {
                    "bars": {"status": freshness_status, "updated_at": "2024-01-02T00:00:00Z"},
                    "news": {"status": freshness_status, "updated_at": "2024-01-02T00:00:00Z"},
                    "sentiment": {"status": freshness_status, "updated_at": "2024-01-02T00:00:00Z"},
                },
                "domain_availability": availability,
            },
            "domain_availability": availability,
            "relative_context": {
                "relative_move_summary": {
                    "market_relative_note": "Relative strength remains constructive versus the benchmark."
                }
            },
        },
        feature_factor_bundle={
            "proprietary_scores": {
                "Signal Fragility Index": {
                    "score": fragility_score,
                    "coverage_status": "available",
                },
                "Macro Alignment Score": {"score": 66.0, "coverage_status": "available"},
                "Narrative Crowding Index": {"score": 43.0, "coverage_status": "available"},
                "Opportunity Quality Score": {"score": 69.0, "coverage_status": "available"},
            },
            "composite_intelligence": {
                "Signal Fragility Index": fragility_score,
                "Macro Alignment Score": 66.0,
                "Narrative Crowding Index": 43.0,
                "Opportunity Quality Score": 69.0,
            },
            "domain_agreement": {
                "domain_agreement_score": agreement_score,
                "domain_conflict_score": conflict_score,
            },
            "regime_intelligence": {
                "regime_label": "trend",
                "regime_confidence": 62.0,
                "regime_instability": 29.0,
                "transition_risk": 31.0,
            },
        },
        strategy={
            "strategy_version": "phase4_institutional_v1",
            "final_signal": "BUY",
            "strategy_posture": "actionable_long",
            "confidence": confidence_score / 100.0,
            "confidence_score": confidence_score,
            "conviction_tier": "high",
            "actionability_score": actionability_score,
            "fragility_tier": "contained",
            "participant_fit": ["swing trader"],
            "primary_participant_fit": "swing trader",
            "strategy_summary": "The setup is constructive and broadly aligned.",
            "scenario_matrix": {
                "base": {"summary": "Base case remains constructive."},
                "bull": {"summary": "Bull case follows clean continuation."},
                "bear": {"summary": "Bear case follows relative weakness."},
                "stress": {"summary": "Stress case follows regime instability."},
            },
            "invalidators": {"top_invalidators": ["Macro alignment deteriorates materially."]},
            "confirmation_triggers": ["Price confirms with stronger volume."],
            "deterioration_triggers": ["Relative strength rolls over."],
            "execution_posture": {
                "preferred_posture": "wait_for_confirmation",
                "urgency_level": "measured",
                "patience_level": "high",
            },
        },
    )
    evaluation = {
        "status": "available",
        "evaluation_version": "phase6_proof_of_edge_v1",
        "prediction_linkage_summary": {
            "total_predictions": matured_count,
            "linked_outcome_status": {"matured": matured_count},
        },
        "signal_scorecard": {
            "final_signal_overall": {
                "matured_count": matured_count,
                "hit_rate": 0.75,
                "average_forward_return": 0.032,
            }
        },
        "strategy_scorecard": {
            "actionable_vs_watchlist_return_spread": actionable_spread,
        },
        "calibration_summary": {
            "confidence_reliability_score": reliability_score,
            "confidence_monotonicity": "higher_confidence_buckets_outperform",
            "calibration_drift_notes": [],
        },
        "weakest_conditions": [],
        "strongest_conditions": [
            {"dimension": "regime_label", "label": "trend", "average_forward_return": 0.05}
        ],
        "evaluation_summary": "Historically similar setups have been constructive on balance.",
        "confidence_reliability_summary": "Confidence reliability is improving and currently acceptable.",
        "regime_usefulness_summary": "Trending regimes have been the strongest cohort.",
    }
    return reports.attach_evaluation_context(
        report,
        evaluation,
        prediction_record_id="prediction-1",
        evaluation_artifact_id="evaluation-1",
    )


def test_phase8_builds_low_risk_live_permission_for_clean_setup(monkeypatch) -> None:
    monkeypatch.setenv("FTIP_DEPLOYMENT_MODE", "low_risk_live")
    store = AssistantStorage(use_memory=True)
    report = _sample_report()

    readiness = build_deployment_readiness_artifact(current_report=report, store=store)

    assert readiness["deployment_mode"]["active_mode"] == "low_risk_live"
    assert readiness["model_readiness"]["live_readiness_score"] >= 72.0
    assert readiness["signal_admission_control"]["admitted_for_live"] is True
    assert readiness["deployment_permission"]["deployment_permission"] == "low_risk_live_eligible"
    assert readiness["deployment_permission"]["trust_tier"] == "conditional_live"
    assert readiness["risk_budgeting"]["risk_budget_tier"] == "pilot_probe"
    assert readiness["rollout_workflow"]["readiness_checkpoint"] in {"pass", "watch"}


def test_phase8_paused_mode_blocks_live_use_and_surfaces_pause_controls(monkeypatch) -> None:
    monkeypatch.setenv("FTIP_DEPLOYMENT_MODE", "paused")
    store = AssistantStorage(use_memory=True)
    report = _sample_report()

    readiness = build_deployment_readiness_artifact(current_report=report, store=store)

    assert readiness["deployment_mode"]["active_mode"] == "paused"
    assert readiness["drift_monitor"]["pause_recommended"] is True
    assert readiness["deployment_permission"]["deployment_permission"] == "blocked_paused"
    assert readiness["deployment_permission"]["trust_tier"] == "paused"
    assert readiness["rollout_workflow"]["rollout_stage"] == "deployment_paused"


def test_phase8_degraded_setup_surfaces_blockers_and_audit_context(monkeypatch) -> None:
    monkeypatch.setenv("FTIP_DEPLOYMENT_MODE", "limited_live")
    store = AssistantStorage(use_memory=True)
    for index in range(2):
        store.save_artifact(
            "session-prior",
            DEPLOYMENT_AUDIT_RECORD_KIND,
            {
                "symbol": "NVDA",
                "horizon": "swing",
                "risk_mode": "balanced",
                "deployment_permission": "blocked_weak_evidence",
                "pause_recommended": True,
            },
        )

    report = _sample_report(
        fragility_score=66.0,
        confidence_score=52.0,
        actionability_score=44.0,
        conflict_score=66.0,
        agreement_score=48.0,
        quality_score=54.0,
        freshness_status="mixed_stale",
        matured_count=5,
        reliability_score=44.0,
        actionable_spread=0.01,
        missingness=0.12,
        fallback_domains=("market", "fundamentals", "sentiment"),
    )

    readiness = build_deployment_readiness_artifact(current_report=report, store=store)

    assert readiness["model_readiness"]["live_readiness_blockers"]
    assert readiness["deployment_permission"]["deployment_permission"] in {
        "blocked_weak_evidence",
        "paper_shadow_only",
    }
    assert readiness["drift_monitor"]["pause_recommended"] or readiness["drift_monitor"]["degrade_to_paper_recommended"]
    assert readiness["audit_snapshot"]["deployment_permission"] == readiness["deployment_permission"]["deployment_permission"]
    assert readiness["audit_snapshot"]["rationale_summary"]
