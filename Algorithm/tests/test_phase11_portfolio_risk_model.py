from __future__ import annotations

import datetime as dt
import math

from api.assistant import reports
from api.assistant.phase11 import (
    PORTFOLIO_RISK_MODEL_VERSION,
    build_portfolio_risk_model_artifact,
)
from api.assistant.phase5.context import build_narrator_context
from api.assistant.phase5.routing import route_question
from api.assistant.storage import AssistantStorage


def _history_from_returns(
    *,
    start: dt.date,
    start_close: float,
    returns: list[float],
) -> list[dict]:
    close = start_close
    rows = []
    for index, daily_return in enumerate(returns):
        close *= 1.0 + daily_return
        rows.append(
            {
                "as_of_date": (start + dt.timedelta(days=index)).isoformat(),
                "close": round(close, 6),
            }
        )
    return rows


def _tech_returns(length: int, *, phase: float = 0.0, drift: float = 0.0012) -> list[float]:
    series: list[float] = []
    for index in range(length):
        wave = math.sin((index + phase) / 5.0) * 0.0056
        pulse = 0.0032 if index % 17 == 0 else 0.0
        series.append(drift + wave + pulse)
    return series


def _energy_returns(length: int, *, phase: float = 0.0, drift: float = 0.0005) -> list[float]:
    series: list[float] = []
    for index in range(length):
        wave = math.cos((index + phase) / 6.0) * 0.0063
        pulse = -0.0025 if index % 13 == 0 else 0.0010
        series.append(drift + wave + pulse)
    return series


def _build_report(
    symbol: str,
    *,
    sector: str,
    benchmark: str,
    theme: str,
    history_rows: list[dict],
    opportunity_quality: float,
    cross_domain_conviction: float,
    signal_fragility: float,
    macro_alignment: float,
    narrative_crowding: float,
    confidence_score: float = 72.0,
    actionability_score: float = 70.0,
    final_signal: str = "BUY",
    strategy_posture: str = "actionable_long",
    deployment_permission: str = "limited_live_eligible",
    trust_tier: str = "conditional_live",
    live_readiness_score: float = 76.0,
    event_overhang_score: float = 28.0,
    implementation_fragility_score: float = 34.0,
    market_stress_score: float = 32.0,
    breadth_state: str = "broad_healthy_participation",
    cross_asset_conflict_score: float = 26.0,
    candidate_classification: str | None = None,
) -> dict:
    ret_21d = (history_rows[-1]["close"] / history_rows[-22]["close"]) - 1.0
    report = reports.build_analysis_report(
        symbol=symbol,
        as_of_date=history_rows[-1]["as_of_date"],
        horizon="swing",
        risk_mode="balanced",
        signal={
            "action": final_signal,
            "score": 0.61 if final_signal == "BUY" else -0.28,
            "confidence": confidence_score / 100.0,
            "horizon_days": 21,
            "reason_codes": ["TREND_UP"],
            "reason_details": {"TREND_UP": "Trend is constructive."},
        },
        key_features={
            "ret_21d": ret_21d,
            "vol_21d": 0.22,
            "atr_pct": 0.035,
            "regime_label": "trend",
        },
        quality={"bars_ok": True, "warnings": []},
        evidence={"reason_codes": ["TREND_UP"], "reason_details": {}},
        data_bundle={
            "portfolio_risk_inputs": {"return_history": history_rows},
            "market_price_volume": {
                "ret_21d": ret_21d,
                "atr_pct": 0.035,
                "realized_vol_21d": 0.22,
                "gap_pct": 0.007,
                "gap_instability_10d": 0.17,
                "volume_anomaly": 1.06,
            },
            "relative_context": {"sector": sector, "benchmark_proxy": benchmark},
            "macro_cross_asset": {"benchmark_proxy": benchmark},
            "symbol_meta": {"sector": sector},
            "sentiment_narrative_flow": {"top_narratives": [{"topic": theme}]},
            "event_catalyst_risk": {
                "event_risk_classification": "high_event_risk"
                if event_overhang_score >= 65.0
                else "low_event_risk",
                "event_overhang_score": event_overhang_score,
                "event_uncertainty_score": max(event_overhang_score - 8.0, 0.0),
                "days_to_next_event": 2 if event_overhang_score >= 65.0 else 18,
            },
            "liquidity_execution_fragility": {
                "implementation_fragility_score": implementation_fragility_score,
                "liquidity_quality_score": max(100.0 - implementation_fragility_score, 20.0),
                "tradability_state": "implementation_fragile"
                if implementation_fragility_score >= 60.0
                else "clean_liquid_setup",
            },
            "market_breadth_internals": {
                "breadth_state": breadth_state,
                "breadth_confirmation_score": 68.0 if breadth_state == "broad_healthy_participation" else 44.0,
            },
            "cross_asset_confirmation": {
                "cross_asset_conflict_score": cross_asset_conflict_score,
                "benchmark_confirmation_score": max(100.0 - cross_asset_conflict_score, 20.0),
                "sector_confirmation_score": max(92.0 - cross_asset_conflict_score, 18.0),
            },
            "stress_spillover_conditions": {
                "market_stress_score": market_stress_score,
                "spillover_risk_score": market_stress_score - 4.0,
            },
        },
        feature_factor_bundle={
            "proprietary_scores": {
                "Market Structure Integrity Score": {"score": max(opportunity_quality - 6.0, 0.0)},
                "Regime Stability Score": {"score": 70.0},
                "Signal Fragility Index": {"score": signal_fragility},
                "Narrative Crowding Index": {"score": narrative_crowding},
                "Fundamental Durability Score": {"score": 72.0},
                "Macro Alignment Score": {"score": macro_alignment},
                "Cross-Domain Conviction Score": {"score": cross_domain_conviction},
                "Opportunity Quality Score": {"score": opportunity_quality},
            },
            "regime_intelligence": {"regime_label": "trend"},
        },
        strategy={
            "strategy_version": "phase4_institutional_v1",
            "final_signal": final_signal,
            "strategy_posture": strategy_posture,
            "confidence": confidence_score / 100.0,
            "confidence_score": confidence_score,
            "conviction_tier": "high" if confidence_score >= 70.0 else "moderate",
            "actionability_score": actionability_score,
            "participant_fit": ["swing trader"],
            "primary_participant_fit": "swing trader",
            "execution_posture": {
                "preferred_posture": "wait_for_confirmation",
                "urgency_level": "measured",
                "patience_level": "high",
                "signal_cleanliness": "mixed_clean",
                "entry_quality_proxy": 58.0,
            },
        },
    )
    report["deployment_permission"] = deployment_permission
    report["trust_tier"] = trust_tier
    report["deployment_mode"] = "limited_live"
    report["live_readiness_score"] = live_readiness_score
    report["rollout_stage"] = "limited_live_monitor"
    report["evaluation"] = {
        "calibration_summary": {"confidence_reliability_score": 73.0},
        "signal_scorecard": {"final_signal_overall": {"hit_rate": 0.61}},
    }
    if candidate_classification:
        report["candidate_classification"] = candidate_classification
    return report


def test_portfolio_risk_model_builds_realized_overlap_and_exposures() -> None:
    store = AssistantStorage(use_memory=True)
    session_id = store.create_session()
    start = dt.date(2024, 1, 2)
    tech = _tech_returns(84, phase=0.0)
    tech_peer = [value * 0.94 + (0.0005 if index % 11 == 0 else 0.0) for index, value in enumerate(tech)]
    energy = _energy_returns(84, phase=2.0)

    current_report = _build_report(
        "NVDA",
        sector="Technology",
        benchmark="QQQ",
        theme="ai_infrastructure",
        history_rows=_history_from_returns(start=start, start_close=100.0, returns=tech),
        opportunity_quality=82.0,
        cross_domain_conviction=77.0,
        signal_fragility=28.0,
        macro_alignment=66.0,
        narrative_crowding=35.0,
    )
    current_id = store.save_artifact(session_id, reports.ANALYSIS_REPORT_KIND, current_report)
    current_report["report_id"] = current_id
    current_report["session_id"] = session_id
    store.save_artifact(
        session_id,
        reports.ANALYSIS_REPORT_KIND,
        _build_report(
            "AMD",
            sector="Technology",
            benchmark="QQQ",
            theme="ai_infrastructure",
            history_rows=_history_from_returns(start=start, start_close=90.0, returns=tech_peer),
            opportunity_quality=77.0,
            cross_domain_conviction=72.0,
            signal_fragility=31.0,
            macro_alignment=63.0,
            narrative_crowding=38.0,
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
            history_rows=_history_from_returns(start=start, start_close=80.0, returns=energy),
            opportunity_quality=62.0,
            cross_domain_conviction=58.0,
            signal_fragility=34.0,
            macro_alignment=57.0,
            narrative_crowding=27.0,
            market_stress_score=38.0,
            cross_asset_conflict_score=33.0,
        ),
    )

    artifact = build_portfolio_risk_model_artifact(
        current_report=current_report,
        session_id=session_id,
        store=store,
    )

    current = artifact["current_candidate"]
    assert artifact["portfolio_risk_model_version"] == PORTFOLIO_RISK_MODEL_VERSION
    assert current["factor_exposure_vector"]
    assert current["exposure_confidence"] >= 60.0
    assert artifact["top_pairwise_relationships"][0]["peer_symbol"] == "AMD"
    assert artifact["top_pairwise_relationships"][0]["pairwise_correlation"] is not None
    assert artifact["top_pairwise_relationships"][0]["hidden_overlap_score"] >= 60.0
    assert current["portfolio_stress_score"] is not None


def test_portfolio_risk_model_surfaces_redundancy_and_replacement_logic() -> None:
    store = AssistantStorage(use_memory=True)
    session_id = store.create_session()
    start = dt.date(2024, 1, 2)
    tech = _tech_returns(84, phase=0.0)
    stronger_tech = [value * 0.98 + 0.0006 for value in tech]
    diversifier = _energy_returns(84, phase=3.5, drift=0.0009)

    current_report = _build_report(
        "NVDA",
        sector="Technology",
        benchmark="QQQ",
        theme="ai_infrastructure",
        history_rows=_history_from_returns(start=start, start_close=100.0, returns=tech),
        opportunity_quality=68.0,
        cross_domain_conviction=63.0,
        signal_fragility=42.0,
        macro_alignment=60.0,
        narrative_crowding=48.0,
        live_readiness_score=68.0,
        candidate_classification="secondary_candidate",
    )
    current_id = store.save_artifact(session_id, reports.ANALYSIS_REPORT_KIND, current_report)
    current_report["report_id"] = current_id
    current_report["session_id"] = session_id
    store.save_artifact(
        session_id,
        reports.ANALYSIS_REPORT_KIND,
        _build_report(
            "AAPL",
            sector="Technology",
            benchmark="QQQ",
            theme="ai_infrastructure",
            history_rows=_history_from_returns(start=start, start_close=95.0, returns=stronger_tech),
            opportunity_quality=84.0,
            cross_domain_conviction=78.0,
            signal_fragility=26.0,
            macro_alignment=67.0,
            narrative_crowding=34.0,
            live_readiness_score=80.0,
            candidate_classification="top_priority_candidate",
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
            history_rows=_history_from_returns(start=start, start_close=82.0, returns=diversifier),
            opportunity_quality=60.0,
            cross_domain_conviction=57.0,
            signal_fragility=30.0,
            macro_alignment=58.0,
            narrative_crowding=22.0,
            market_stress_score=40.0,
            cross_asset_conflict_score=31.0,
            candidate_classification="watchlist_candidate",
        ),
    )

    artifact = build_portfolio_risk_model_artifact(
        current_report=current_report,
        session_id=session_id,
        store=store,
    )

    current = artifact["current_candidate"]
    overlay = artifact["portfolio_overlay"]
    assert current["better_alternative_flag"] is True
    assert current["replacement_candidate"] in {"AAPL", "XOM"}
    assert current["hidden_overlap_score"] >= 55.0
    assert current["portfolio_quality_upgrade_reason"]
    assert overlay["hidden_concentration_score"] >= 45.0
    assert overlay["cluster_risk_score"] >= 40.0
    assert overlay["portfolio_risk_warnings"]


def test_portfolio_risk_context_surfaces_in_report_and_narrator() -> None:
    store = AssistantStorage(use_memory=True)
    session_id = store.create_session()
    start = dt.date(2024, 1, 2)
    current_report = _build_report(
        "NVDA",
        sector="Technology",
        benchmark="QQQ",
        theme="ai_infrastructure",
        history_rows=_history_from_returns(start=start, start_close=100.0, returns=_tech_returns(84)),
        opportunity_quality=68.0,
        cross_domain_conviction=63.0,
        signal_fragility=42.0,
        macro_alignment=60.0,
        narrative_crowding=48.0,
    )
    current_id = store.save_artifact(session_id, reports.ANALYSIS_REPORT_KIND, current_report)
    current_report["report_id"] = current_id
    current_report["session_id"] = session_id
    store.save_artifact(
        session_id,
        reports.ANALYSIS_REPORT_KIND,
        _build_report(
            "AAPL",
            sector="Technology",
            benchmark="QQQ",
            theme="ai_infrastructure",
            history_rows=_history_from_returns(
                start=start,
                start_close=95.0,
                returns=[value * 0.98 + 0.0006 for value in _tech_returns(84)],
            ),
            opportunity_quality=84.0,
            cross_domain_conviction=78.0,
            signal_fragility=26.0,
            macro_alignment=67.0,
            narrative_crowding=34.0,
        ),
    )

    artifact = build_portfolio_risk_model_artifact(
        current_report=current_report,
        session_id=session_id,
        store=store,
    )
    enriched_report = reports.attach_portfolio_risk_context(
        current_report,
        artifact,
        portfolio_risk_model_artifact_id="portfolio-risk-1",
    )
    route = route_question(
        "Which candidate is strongest but most redundant for the portfolio, and what hidden risks are stacked?"
    )
    narrator_context = build_narrator_context(
        enriched_report,
        active_analysis=reports.build_active_analysis_reference(
            enriched_report, session_id=session_id, report_id=current_id
        ),
        route=route,
        user_message="Which candidate is strongest but most redundant for the portfolio, and what hidden risks are stacked?",
        caller_context=None,
    )

    assert enriched_report["portfolio_risk_model_summary"]
    assert enriched_report["hidden_overlap_redundancy_analysis"]
    assert enriched_report["factor_exposure_summary"]
    assert enriched_report["concentration_cluster_risk_analysis"]
    assert enriched_report["replacement_diversification_analysis"]
    assert enriched_report["portfolio_stress_fragility_summary"]
    assert route["intent"] == "portfolio_construction"
    assert "portfolio_risk_model_summary" in narrator_context["selected_sections"]
    assert "hidden_overlap_redundancy_analysis" in narrator_context["selected_sections"]
    assert "concentration_cluster_risk_analysis" in narrator_context["selected_sections"]
    assert narrator_context["portfolio_snapshot"]["replacement_candidate"] == "AAPL"
