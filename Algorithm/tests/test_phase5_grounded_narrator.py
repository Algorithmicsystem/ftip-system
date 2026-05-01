from api.assistant import reports
from api.assistant.phase5.context import build_narrator_context
from api.assistant.phase5.grounding import resolve_active_report
from api.assistant.phase5.routing import route_question
from api.assistant.storage import AssistantStorage


def _sample_report(symbol: str) -> dict:
    proprietary_scores = {
        "Market Structure Integrity Score": {"score": 67.2, "coverage_status": "available"},
        "Regime Stability Score": {"score": 58.4, "coverage_status": "available"},
        "Signal Fragility Index": {"score": 34.1, "coverage_status": "available"},
        "Narrative Crowding Index": {"score": 61.7, "coverage_status": "available"},
        "Macro Alignment Score": {"score": 55.6, "coverage_status": "available"},
        "Cross-Domain Conviction Score": {"score": 64.8, "coverage_status": "available"},
        "Opportunity Quality Score": {"score": 57.3, "coverage_status": "available"},
    }
    report = reports.build_analysis_report(
        symbol=symbol,
        as_of_date="2024-01-02",
        horizon="swing",
        risk_mode="balanced",
        signal={
            "action": "BUY",
            "score": 0.72,
            "confidence": 0.61,
            "entry_low": 100,
            "entry_high": 104,
            "stop_loss": 95,
            "take_profit_1": 110,
            "take_profit_2": 118,
            "horizon_days": 21,
            "reason_codes": ["TREND_UP"],
            "reason_details": {"TREND_UP": "Trend remains constructive."},
        },
        key_features={
            "ret_1d": 0.01,
            "ret_5d": 0.03,
            "ret_21d": 0.08,
            "vol_21d": 0.24,
            "atr_pct": 0.04,
            "trend_slope_21d": 0.19,
            "regime_label": "trend",
        },
        quality={
            "bars_ok": True,
            "fundamentals_ok": True,
            "sentiment_ok": True,
            "news_ok": True,
            "warnings": [],
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
                "quality_score": 88,
                "freshness_summary": {
                    "market_price_volume": {"status": "fresh"},
                    "fundamental_filing": {"status": "mixed"},
                },
            },
            "market_price_volume": {"ret_21d": 0.08},
            "technical_market_structure": {"trend_quality": 0.62},
            "fundamental_filing": {
                "coverage_score": 0.82,
                "strength_summary": ["Margins remain constructive."],
                "weakness_summary": ["Balance-sheet detail is only partial."],
            },
            "sentiment_narrative_flow": {"aggregated_sentiment_bias": 0.14},
            "macro_cross_asset": {"macro_regime_context": {"regime": "mixed_supportive"}},
            "geopolitical_policy": {"event_intensity_score": 0.24},
            "relative_context": {"benchmark_relative_strength": 0.12},
        },
        feature_factor_bundle={
            "proprietary_scores": proprietary_scores,
            "composite_intelligence": {
                name: payload["score"] for name, payload in proprietary_scores.items()
            },
            "factor_groups": {
                "market_structure": {"trend_quality": 0.62},
            },
            "regime_intelligence": {
                "regime_label": "trend",
                "regime_confidence": 62.0,
                "regime_instability": 29.0,
                "transition_risk": 33.0,
                "breakout_readiness": 58.0,
            },
            "fragility_intelligence": {
                "instability_score": 34.0,
                "clean_setup_score": 63.0,
                "confidence_degradation_triggers": ["narrative crowding is elevated"],
            },
            "domain_agreement": {
                "domain_agreement_score": 74.0,
                "domain_conflict_score": 26.0,
                "strongest_confirming_domains": [
                    {"domain": "technical"},
                    {"domain": "macro"},
                ],
                "strongest_conflicting_domains": [{"domain": "sentiment"}],
                "agreement_flags": ["many domains agree"],
            },
            "conviction_components": {"technical": 0.68, "macro": 0.55},
            "opportunity_quality_components": {"structure": 0.64, "fragility": 0.49},
        },
        strategy={
            "strategy_version": "phase4_institutional_v1",
            "final_signal": "HOLD",
            "strategy_posture": "watchlist_positive",
            "confidence_score": 54.0,
            "confidence": 0.54,
            "conviction_tier": "moderate",
            "actionability_score": 47.0,
            "fragility_tier": "contained",
            "participant_fit": ["swing trader", "wait / observe"],
            "primary_participant_fit": "swing trader",
            "strategy_summary": "The strategy remains HOLD / watchlist positive with moderate confidence.",
            "scenario_matrix": {
                "base": {
                    "summary": "Base case keeps the setup constructive but not yet fully actionable.",
                    "supporting_conditions": ["trend quality holds", "macro stays mixed-supportive"],
                    "risk_conditions": ["crowding remains elevated"],
                    "expected_posture_shift": "stay HOLD",
                    "confidence_level": "moderate",
                    "fragility_notes": ["fragility is contained but not absent"],
                },
                "bull": {
                    "summary": "Bull case requires stronger confirmation and cleaner cross-domain agreement.",
                    "supporting_conditions": ["price confirms with volume"],
                    "risk_conditions": ["macro stalls"],
                    "expected_posture_shift": "upgrade toward BUY",
                    "confidence_level": "low_to_moderate",
                    "fragility_notes": ["needs cleaner follow-through"],
                },
                "bear": {
                    "summary": "Bear case is driven by relative weakness and crowding without price confirmation.",
                    "supporting_conditions": ["relative strength fades"],
                    "risk_conditions": ["crowding accelerates"],
                    "expected_posture_shift": "downgrade toward fragile_hold",
                    "confidence_level": "moderate",
                    "fragility_notes": ["sentiment can turn into a drag quickly"],
                },
                "stress": {
                    "summary": "Stress case assumes regime instability and macro slippage overwhelm the current setup.",
                    "supporting_conditions": ["regime breaks lower"],
                    "risk_conditions": ["macro alignment turns negative"],
                    "expected_posture_shift": "no_trade",
                    "confidence_level": "moderate",
                    "fragility_notes": ["setup quality would degrade materially"],
                },
            },
            "invalidators": {
                "top_invalidators": ["Macro alignment deteriorates materially."],
                "regime_invalidators": ["The regime shifts into unstable high-volatility chop."],
                "narrative_invalidators": ["Crowding rises further without price confirmation."],
                "macro_invalidators": ["Rates and growth alignment flip against the setup."],
                "quality_freshness_invalidators": ["Coverage deteriorates or core data turns stale."],
            },
            "confirmation_triggers": ["Price confirms with stronger volume participation."],
            "deterioration_triggers": ["Relative strength continues to fade."],
            "fragility_vetoes": [{"name": "narrative_crowding", "detail": "Narrative crowding is elevated."}],
            "uncertainty_notes": ["Fundamental detail is partly coverage-constrained."],
            "confidence_degraders": ["sentiment is not fully aligned with price"],
            "top_contributors": [
                {
                    "label": "trend_following",
                    "score": 0.32,
                    "detail": "trend quality remains constructive",
                }
            ],
            "top_detractors": [
                {
                    "label": "sentiment_aware",
                    "score": -0.14,
                    "detail": "narrative crowding is elevated",
                }
            ],
            "execution_posture": {
                "preferred_posture": "wait_for_confirmation",
                "urgency_level": "measured",
                "patience_level": "high",
                "signal_cleanliness": "mixed_clean",
                "entry_quality_proxy": 54.0,
                "risk_context_summary": "constructive but not clean enough for immediate action",
            },
        },
    )
    return reports.attach_deployment_context(
        report,
        {
            "deployment_readiness_version": "phase8_capital_readiness_v1",
            "deployment_mode": {
                "active_mode": "paper_shadow",
                "rollout_stage": "forward_shadow_validation",
            },
            "model_readiness": {
                "model_readiness_status": "constrained",
                "live_readiness_score": 58.0,
                "live_readiness_blockers": ["similar setups do not yet have enough matured observations for live escalation"],
                "recent_degradation_flags": ["confidence reliability is still building"],
            },
            "signal_admission_control": {
                "admitted_for_strategy": True,
                "admitted_for_paper": True,
                "admitted_for_live": False,
            },
            "deployment_permission": {
                "deployment_permission": "paper_shadow_only",
                "deployment_blockers": ["confidence calibration quality is not strong enough for live escalation"],
                "deployment_rationale": "The setup is analyzable, but it remains paper-only until calibration and matured sample support improve.",
                "trust_tier": "paper_only",
                "minimum_required_review": "analyst_review",
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
                "rollout_stage": "forward_shadow_validation",
                "readiness_checkpoint": "watch",
                "promotion_criteria": ["confidence reliability remains above the stage threshold"],
                "demotion_criteria": ["drift monitoring recommends paper or paused mode"],
                "stage_transition_notes": ["Continue forward shadow validation before any live escalation."],
            },
            "drift_monitor": {
                "pause_recommended": False,
                "degrade_to_paper_recommended": False,
                "drift_alerts": ["confidence reliability is below the live-support comfort zone"],
                "deployment_risk_alerts": ["live readiness has slipped below the controlled-live comfort zone"],
            },
            "audit_snapshot": {
                "rationale_summary": "Paper-shadow only while calibration and matured sample depth improve.",
            },
        },
        readiness_artifact_id="readiness-1",
        deployment_audit_artifact_id="audit-1",
    )


def test_phase5_routes_questions_into_grounded_answer_modes() -> None:
    strategy_route = route_question("Why is this HOLD instead of BUY?")
    assert strategy_route["intent"] == "strategy"
    assert strategy_route["answer_mode"] == "strategist"

    fundamental_route = route_question("What is weakest in the fundamentals?")
    assert fundamental_route["intent"] == "fundamental"
    assert fundamental_route["answer_mode"] == "analyst"

    risk_route = route_question("What makes it fragile?")
    assert risk_route["intent"] == "risk_invalidation"
    assert risk_route["answer_mode"] == "risk"

    evidence_route = route_question("What evidence supports this view?")
    assert evidence_route["intent"] == "evidence_provenance"
    assert evidence_route["answer_mode"] == "evidence"

    deployment_route = route_question("Is this ready for live capital or only paper mode?")
    assert deployment_route["intent"] == "deployment_readiness"
    assert deployment_route["answer_mode"] == "deployment"


def test_phase5_builds_grounded_narrator_context_from_active_report() -> None:
    report = _sample_report("NVDA")
    active_analysis = reports.build_active_analysis_reference(
        report,
        session_id="session-1",
        report_id="report-1",
    )
    route = route_question("What is the bear case?")

    narrator_context = build_narrator_context(
        report,
        active_analysis=active_analysis,
        route=route,
        user_message="What is the bear case?",
        caller_context={"active_analysis": active_analysis},
    )

    assert narrator_context["active_context"]["symbol"] == "NVDA"
    assert narrator_context["active_context"]["conviction_tier"] == "moderate"
    assert narrator_context["signal_snapshot"]["strategy_posture"] == "watchlist_positive"
    assert narrator_context["active_context"]["deployment_permission"] == "paper_shadow_only"
    assert "strategy_view" in narrator_context["selected_sections"]
    assert "risks_weaknesses_invalidators" in narrator_context["selected_sections"]
    assert narrator_context["scenario_matrix"]["bear"]["summary"]
    assert narrator_context["invalidators"]["top_invalidators"]
    assert narrator_context["evidence_summary"]["strong_evidence"]
    assert narrator_context["followup_questions"]
    assert narrator_context["deployment_readiness_snapshot"]["deployment_permission"] == "paper_shadow_only"
    assert narrator_context["deployment_readiness_snapshot"]["blockers"]


def test_phase5_preserves_session_symbol_continuity_for_chat_followups() -> None:
    store = AssistantStorage(use_memory=True)
    session_id = store.create_session()
    nvda_report = _sample_report("NVDA")
    aapl_report = _sample_report("AAPL")
    nvda_report_id = store.save_artifact(session_id, reports.ANALYSIS_REPORT_KIND, nvda_report)
    store.save_artifact(session_id, reports.ANALYSIS_REPORT_KIND, aapl_report)
    store.upsert_session_metadata(
        session_id,
        {
            "active_analysis": reports.build_active_analysis_reference(
                nvda_report,
                session_id=session_id,
                report_id=nvda_report_id,
            )
        },
    )

    report, reference = resolve_active_report(
        session_id=session_id,
        message="What makes it fragile?",
        context=None,
        store=store,
    )
    assert report is not None
    assert report["symbol"] == "NVDA"
    assert reference["symbol"] == "NVDA"

    switched_report, switched_reference = resolve_active_report(
        session_id=session_id,
        message="What about AAPL risk?",
        context=None,
        store=store,
    )
    assert switched_report is not None
    assert switched_report["symbol"] == "AAPL"
    assert switched_reference["symbol"] == "AAPL"
