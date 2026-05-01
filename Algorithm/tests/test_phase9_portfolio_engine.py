from api.assistant import reports
from api.assistant.phase9 import (
    PORTFOLIO_CONSTRUCTION_VERSION,
    build_portfolio_construction_artifact,
)
from api.assistant.storage import AssistantStorage


def _build_report(
    symbol: str,
    *,
    sector: str = "Technology",
    benchmark: str = "QQQ",
    theme: str = "ai_infrastructure",
    final_signal: str = "BUY",
    strategy_posture: str = "actionable_long",
    conviction_tier: str = "high",
    actionability_score: float = 72.0,
    confidence_score: float = 71.0,
    deployment_permission: str = "low_risk_live_eligible",
    trust_tier: str = "conditional_live",
    live_readiness_score: float = 79.0,
    opportunity_quality: float = 81.0,
    cross_domain_conviction: float = 76.0,
    signal_fragility: float = 31.0,
    regime_stability: float = 69.0,
    fundamental_durability: float = 71.0,
    macro_alignment: float = 65.0,
    narrative_crowding: float = 34.0,
    evaluation_reliability: float = 74.0,
    evaluation_hit_rate: float = 0.62,
    ret_21d: float = 0.11,
    atr_pct: float = 0.034,
    vol_21d: float = 0.22,
    gap_pct: float = 0.008,
    gap_instability_10d: float = 0.18,
    volume_anomaly: float = 1.08,
    horizon_days: int = 21,
    preferred_posture: str = "wait_for_confirmation",
    urgency_level: str = "measured",
    patience_level: str = "high",
    signal_cleanliness: str = "mixed_clean",
) -> dict:
    proprietary_scores = {
        "Market Structure Integrity Score": {"score": max(0.0, opportunity_quality - 4.0)},
        "Regime Stability Score": {"score": regime_stability},
        "Signal Fragility Index": {"score": signal_fragility},
        "Narrative Crowding Index": {"score": narrative_crowding},
        "Fundamental Durability Score": {"score": fundamental_durability},
        "Macro Alignment Score": {"score": macro_alignment},
        "Cross-Domain Conviction Score": {"score": cross_domain_conviction},
        "Opportunity Quality Score": {"score": opportunity_quality},
    }
    report = reports.build_analysis_report(
        symbol=symbol,
        as_of_date="2024-01-02",
        horizon="swing",
        risk_mode="balanced",
        signal={
            "action": final_signal,
            "score": 0.72,
            "confidence": 0.64,
            "horizon_days": horizon_days,
            "reason_codes": ["TREND_UP"],
            "reason_details": {"TREND_UP": "Trend remains constructive."},
        },
        key_features={
            "ret_21d": ret_21d,
            "vol_21d": vol_21d,
            "atr_pct": atr_pct,
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
            "sources": ["market_bars_daily", "news_raw"],
        },
        data_bundle={
            "market_price_volume": {
                "ret_21d": ret_21d,
                "atr_pct": atr_pct,
                "realized_vol_21d": vol_21d,
                "gap_pct": gap_pct,
                "gap_instability_10d": gap_instability_10d,
                "volume_anomaly": volume_anomaly,
            },
            "relative_context": {
                "sector": sector,
                "benchmark_proxy": benchmark,
            },
            "macro_cross_asset": {
                "benchmark_proxy": benchmark,
            },
            "sentiment_narrative_flow": {
                "top_narratives": [{"topic": theme}],
            },
            "symbol_meta": {
                "sector": sector,
            },
        },
        feature_factor_bundle={
            "proprietary_scores": proprietary_scores,
            "composite_intelligence": {
                name: payload["score"] for name, payload in proprietary_scores.items()
            },
            "regime_intelligence": {"regime_label": "trend"},
        },
        strategy={
            "strategy_version": "phase4_institutional_v1",
            "final_signal": final_signal,
            "strategy_posture": strategy_posture,
            "confidence": confidence_score / 100.0,
            "confidence_score": confidence_score,
            "conviction_tier": conviction_tier,
            "actionability_score": actionability_score,
            "participant_fit": ["swing trader"],
            "primary_participant_fit": "swing trader",
            "scenario_matrix": {"base": {"summary": "Constructive base case."}},
            "execution_posture": {
                "preferred_posture": preferred_posture,
                "urgency_level": urgency_level,
                "patience_level": patience_level,
                "signal_cleanliness": signal_cleanliness,
                "entry_quality_proxy": 58.0,
                "risk_context_summary": "Execution remains measured.",
            },
        },
    )
    report = reports.attach_deployment_context(
        report,
        {
            "deployment_readiness_version": "phase8_capital_readiness_v1",
            "deployment_mode": {
                "active_mode": "low_risk_live",
                "rollout_stage": "low_risk_live_pilot",
            },
            "model_readiness": {
                "model_readiness_status": "ready",
                "live_readiness_score": live_readiness_score,
                "live_readiness_blockers": [],
                "recent_degradation_flags": [],
            },
            "signal_admission_control": {
                "admitted_for_strategy": True,
                "admitted_for_paper": True,
                "admitted_for_live": deployment_permission.endswith("eligible"),
            },
            "deployment_permission": {
                "deployment_permission": deployment_permission,
                "deployment_blockers": [],
                "deployment_rationale": "Deployment gate reflects the current quality and readiness state.",
                "trust_tier": trust_tier,
                "minimum_required_review": "analyst_review",
                "human_review_required": True,
            },
            "risk_budgeting": {
                "risk_budget_tier": "pilot_band",
                "exposure_caution_level": "measured",
                "fragility_adjusted_size_band": "0.50x pilot band",
                "confidence_adjusted_size_band": "0.50x pilot band",
                "maximum_risk_mode_allowed": "low_risk_live",
            },
            "rollout_workflow": {
                "rollout_stage": "low_risk_live_pilot",
                "readiness_checkpoint": "watch",
                "promotion_criteria": ["evaluation reliability remains healthy"],
                "demotion_criteria": ["fragility rises materially"],
                "stage_transition_notes": ["Remain reversible to paper mode if quality degrades."],
            },
            "drift_monitor": {
                "pause_recommended": False,
                "degrade_to_paper_recommended": False,
                "drift_alerts": [],
                "deployment_risk_alerts": [],
            },
            "audit_snapshot": {
                "rationale_summary": "Readiness state reflects the current deployment gate and evidence stack.",
            },
        },
        readiness_artifact_id=f"readiness-{symbol}",
        deployment_audit_artifact_id=f"audit-{symbol}",
    )
    report["evaluation"] = {
        "status": "available",
        "evaluation_version": "phase6_eval_v1",
        "calibration_summary": {
            "confidence_reliability_score": evaluation_reliability,
        },
        "signal_scorecard": {
            "final_signal_overall": {
                "hit_rate": evaluation_hit_rate,
            }
        },
    }
    report["evaluation_summary"] = "Evaluation context is available."
    return report


def test_portfolio_engine_ranks_and_classifies_candidates() -> None:
    store = AssistantStorage(use_memory=True)
    session_id = store.create_session()
    current_report = _build_report("NVDA")
    current_report_id = store.save_artifact(session_id, reports.ANALYSIS_REPORT_KIND, current_report)
    store.save_artifact(
        session_id,
        reports.ANALYSIS_REPORT_KIND,
        _build_report(
            "AAPL",
            opportunity_quality=69.0,
            cross_domain_conviction=68.0,
            actionability_score=61.0,
            confidence_score=62.0,
            live_readiness_score=73.0,
        ),
    )
    store.save_artifact(
        session_id,
        reports.ANALYSIS_REPORT_KIND,
        _build_report(
            "XOM",
            sector="Energy",
            benchmark="XLE",
            theme="energy_supply",
            opportunity_quality=63.0,
            cross_domain_conviction=58.0,
            fundamental_durability=67.0,
            macro_alignment=59.0,
        ),
    )

    artifact = build_portfolio_construction_artifact(
        current_report=current_report,
        current_report_id=current_report_id,
        session_id=session_id,
        store=store,
    )

    current = artifact["current_candidate"]
    assert artifact["portfolio_construction_version"] == PORTFOLIO_CONSTRUCTION_VERSION
    assert current["symbol"] == "NVDA"
    assert current["candidate_classification"] == "top_priority_candidate"
    assert current["portfolio_rank"] == 1
    assert current["portfolio_candidate_score"] >= 70.0
    assert "NVDA" in artifact["workflow"]["prioritized_watchlist"]
    assert "NVDA" in artifact["workflow"]["active_portfolio_candidates"]


def test_portfolio_engine_surfaces_overlap_and_exposure_redundancy() -> None:
    store = AssistantStorage(use_memory=True)
    session_id = store.create_session()
    current_report = _build_report(
        "NVDA",
        opportunity_quality=66.0,
        cross_domain_conviction=63.0,
        deployment_permission="paper_shadow_only",
        trust_tier="paper_only",
        live_readiness_score=58.0,
    )
    current_report_id = store.save_artifact(session_id, reports.ANALYSIS_REPORT_KIND, current_report)
    store.save_artifact(
        session_id,
        reports.ANALYSIS_REPORT_KIND,
        _build_report(
            "AMD",
            opportunity_quality=65.0,
            cross_domain_conviction=62.0,
            deployment_permission="paper_shadow_only",
            trust_tier="paper_only",
            live_readiness_score=57.0,
        ),
    )
    store.save_artifact(
        session_id,
        reports.ANALYSIS_REPORT_KIND,
        _build_report(
            "AAPL",
            opportunity_quality=64.0,
            cross_domain_conviction=61.0,
            deployment_permission="paper_shadow_only",
            trust_tier="paper_only",
            live_readiness_score=56.0,
        ),
    )

    artifact = build_portfolio_construction_artifact(
        current_report=current_report,
        current_report_id=current_report_id,
        session_id=session_id,
        store=store,
    )
    current = artifact["current_candidate"]

    assert current["overlap_score"] >= 70.0
    assert current["redundancy_score"] >= 70.0
    assert current["most_redundant_symbol"] in {"AMD", "AAPL"}
    assert current["concentration_warning"] is not None
    assert current["theme_exposure_warning"] is not None
    assert current["diversification_status"] in {"mixed", "concentrated"}


def test_portfolio_engine_degrades_size_band_and_execution_for_fragile_setups() -> None:
    store = AssistantStorage(use_memory=True)
    session_id = store.create_session()
    current_report = _build_report(
        "SHOP",
        sector="Consumer Discretionary",
        benchmark="XLY",
        theme="ecommerce",
        strategy_posture="watchlist_positive",
        conviction_tier="low",
        actionability_score=43.0,
        confidence_score=49.0,
        deployment_permission="paper_shadow_only",
        trust_tier="paper_only",
        live_readiness_score=52.0,
        opportunity_quality=58.0,
        cross_domain_conviction=55.0,
        signal_fragility=67.0,
        regime_stability=46.0,
        fundamental_durability=52.0,
        macro_alignment=48.0,
        narrative_crowding=63.0,
        ret_21d=0.07,
        atr_pct=0.082,
        vol_21d=0.44,
        gap_pct=0.033,
        gap_instability_10d=0.56,
        volume_anomaly=0.92,
        horizon_days=5,
        urgency_level="high",
        patience_level="low",
        signal_cleanliness="noisy",
    )
    current_report_id = store.save_artifact(session_id, reports.ANALYSIS_REPORT_KIND, current_report)

    artifact = build_portfolio_construction_artifact(
        current_report=current_report,
        current_report_id=current_report_id,
        session_id=session_id,
        store=store,
    )
    current = artifact["current_candidate"]

    assert current["candidate_classification"] == "too_fragile_candidate"
    assert current["size_band"] == "paper / shadow band"
    assert current["caution_level"] == "high"
    assert current["execution_quality_score"] < 40.0
    assert current["wait_for_better_entry_flag"] is True
    assert "fragility is too elevated" in current["candidate_blockers"]


def test_portfolio_engine_marks_rotation_pressure_when_a_cleaner_peer_exists() -> None:
    store = AssistantStorage(use_memory=True)
    session_id = store.create_session()
    current_report = _build_report(
        "NVDA",
        opportunity_quality=61.0,
        cross_domain_conviction=60.0,
        actionability_score=55.0,
        confidence_score=56.0,
        deployment_permission="paper_shadow_only",
        trust_tier="paper_only",
        live_readiness_score=56.0,
    )
    current_report_id = store.save_artifact(session_id, reports.ANALYSIS_REPORT_KIND, current_report)
    store.save_artifact(
        session_id,
        reports.ANALYSIS_REPORT_KIND,
        _build_report(
            "AAPL",
            opportunity_quality=78.0,
            cross_domain_conviction=74.0,
            actionability_score=70.0,
            confidence_score=72.0,
            deployment_permission="low_risk_live_eligible",
            trust_tier="conditional_live",
            live_readiness_score=80.0,
        ),
    )

    artifact = build_portfolio_construction_artifact(
        current_report=current_report,
        current_report_id=current_report_id,
        session_id=session_id,
        store=store,
    )
    workflow = artifact["workflow"]

    assert workflow["priority_shift_flag"] is True
    assert workflow["rebalance_attention_flag"] is True
    assert workflow["replacement_candidate_notes"]
    assert workflow["rotation_pressure_score"] > 0
