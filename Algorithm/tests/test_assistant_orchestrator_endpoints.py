import datetime as dt
import math
from decimal import Decimal

from fastapi.testclient import TestClient

from api.main import app


def _sample_chat_report(symbol: str) -> dict:
    from api.assistant import reports

    report = reports.build_analysis_report(
        symbol=symbol,
        as_of_date="2024-01-02",
        horizon="swing",
        risk_mode="balanced",
        signal={
            "action": "BUY",
            "score": 0.7,
            "confidence": 0.6,
            "entry_low": 100,
            "entry_high": 104,
            "stop_loss": 95,
            "take_profit_1": 112,
            "take_profit_2": 118,
            "horizon_days": 21,
            "reason_codes": ["TREND_UP"],
            "reason_details": {"TREND_UP": "Trend remains constructive."},
        },
        key_features={"ret_5d": 0.08, "vol_21d": 0.24, "regime_label": "trend"},
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
        feature_factor_bundle={
            "proprietary_scores": {
                "Cross-Domain Conviction Score": {"score": 64.1, "coverage_status": "available"},
                "Signal Fragility Index": {"score": 33.4, "coverage_status": "available"},
            },
            "composite_intelligence": {
                "Cross-Domain Conviction Score": 64.1,
                "Signal Fragility Index": 33.4,
                "Market Structure Integrity Score": 62.8,
                "Regime Stability Score": 58.9,
            },
            "regime_intelligence": {
                "regime_label": "trend",
                "regime_confidence": 61.0,
                "regime_instability": 28.0,
                "transition_risk": 34.0,
            },
            "fragility_intelligence": {
                "instability_score": 33.0,
                "clean_setup_score": 61.0,
                "confidence_degradation_triggers": ["crowding remains elevated"],
            },
            "domain_agreement": {
                "domain_agreement_score": 73.0,
                "domain_conflict_score": 27.0,
                "strongest_confirming_domains": [
                    {"domain": "technical"},
                    {"domain": "macro"},
                ],
                "strongest_conflicting_domains": [{"domain": "sentiment"}],
                "agreement_flags": ["many domains agree"],
            },
        },
        strategy={
            "strategy_version": "phase4_institutional_v1",
            "final_signal": "HOLD",
            "strategy_posture": "watchlist_positive",
            "confidence_score": 53.0,
            "confidence": 0.53,
            "conviction_tier": "moderate",
            "actionability_score": 46.0,
            "participant_fit": ["swing trader", "wait / observe"],
            "primary_participant_fit": "swing trader",
            "strategy_summary": "The strategy remains HOLD / watchlist positive with moderate conviction.",
            "scenario_matrix": {
                "base": {"summary": "Base case remains constructive but not fully actionable."},
                "bull": {"summary": "Bull case needs cleaner confirmation."},
                "bear": {"summary": "Bear case follows relative weakness and crowding."},
                "stress": {"summary": "Stress case assumes regime instability dominates."},
            },
            "invalidators": {
                "top_invalidators": ["Macro alignment deteriorates materially."],
                "regime_invalidators": ["The regime breaks into unstable chop."],
            },
            "confirmation_triggers": ["Price confirms with stronger volume."],
            "deterioration_triggers": ["Relative strength rolls over."],
            "fragility_vetoes": [
                {"name": "narrative_crowding", "detail": "Narrative crowding is elevated."}
            ],
            "uncertainty_notes": ["Fundamental detail is partly coverage constrained."],
            "confidence_degraders": ["sentiment is not fully aligned with price"],
            "execution_posture": {
                "preferred_posture": "wait_for_confirmation",
                "urgency_level": "measured",
                "patience_level": "high",
                "signal_cleanliness": "mixed_clean",
                "entry_quality_proxy": 53.0,
                "risk_context_summary": "constructive but not yet clean enough for immediate action",
            },
        },
    )
    report = reports.attach_deployment_context(
        report,
        {
            "deployment_readiness_version": "phase8_capital_readiness_v1",
            "deployment_mode": {
                "active_mode": "paper_shadow",
                "rollout_stage": "forward_shadow_validation",
            },
            "model_readiness": {
                "model_readiness_status": "constrained",
                "live_readiness_score": 57.0,
                "live_readiness_blockers": ["confidence calibration quality is not strong enough for live escalation"],
                "recent_degradation_flags": ["confidence reliability is still building"],
            },
            "signal_admission_control": {
                "admitted_for_strategy": True,
                "admitted_for_paper": True,
                "admitted_for_live": False,
            },
            "deployment_permission": {
                "deployment_permission": "paper_shadow_only",
                "deployment_blockers": ["fragility remains too high for live admission"],
                "deployment_rationale": "The setup is analyzable, but it remains paper-only until calibration and fragility improve.",
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
    report = reports.attach_portfolio_context(
        report,
        {
            "portfolio_construction_version": "phase9_portfolio_construction_v1",
            "current_candidate": {
                "symbol": symbol,
                "candidate_classification": "watchlist_candidate",
                "ranked_opportunity_score": 62.0,
                "portfolio_candidate_score": 59.0,
                "watchlist_priority_score": 65.0,
                "deployability_rank": 53.0,
                "portfolio_rank": 2,
                "portfolio_fit_quality": 47.0,
                "overlap_score": 75.0,
                "redundancy_score": 78.0,
                "diversification_contribution_score": 32.0,
                "most_redundant_symbol": "AAPL",
                "size_band": "paper / shadow band",
                "weight_band": "0.00x live weight",
                "risk_budget_band": "shadow_risk_band",
                "execution_quality_score": 54.0,
                "friction_penalty": 35.0,
                "turnover_penalty": 44.0,
                "wait_for_better_entry_flag": True,
                "confirmation_preferred_flag": True,
                "candidate_blockers": ["the idea is redundant with existing tracked exposures"],
            },
            "cohort_ranking": [
                {
                    "portfolio_rank": 1,
                    "symbol": "AAPL",
                    "candidate_classification": "top_priority_candidate",
                    "portfolio_candidate_score": 76.0,
                    "portfolio_fit_quality": 72.0,
                    "size_band": "exploratory allocation band",
                    "deployment_permission": "low_risk_live_eligible",
                    "strategy_posture": "actionable_long",
                    "conviction_tier": "high",
                },
                {
                    "portfolio_rank": 2,
                    "symbol": symbol,
                    "candidate_classification": "watchlist_candidate",
                    "portfolio_candidate_score": 59.0,
                    "portfolio_fit_quality": 47.0,
                    "size_band": "paper / shadow band",
                    "deployment_permission": "paper_shadow_only",
                    "strategy_posture": "watchlist_positive",
                    "conviction_tier": "moderate",
                },
            ],
            "workflow": {
                "candidate_watchlist": [symbol],
                "prioritized_watchlist": ["AAPL", symbol],
                "active_portfolio_candidates": ["AAPL"],
                "blocked_candidates": [],
                "stale_review_needed": [],
                "priority_shift_flag": True,
                "rebalance_attention_flag": True,
                "candidate_upgrade_reason": None,
                "candidate_downgrade_reason": "portfolio redundancy is capping priority",
                "replacement_candidate_notes": "AAPL currently offers a cleaner portfolio-adjusted candidate score.",
                "rotation_pressure_score": 70.0,
            },
            "portfolio_context_summary": "Portfolio rank is 2 of 2 tracked candidates and the setup remains watchlist-only because fit is modest.",
            "portfolio_fit_analysis": "Overlap and redundancy are elevated versus AAPL, so diversification contribution is limited.",
            "execution_quality_analysis": "Execution quality is acceptable, but confirmation is preferred and live size remains blocked.",
            "portfolio_workflow_summary": "AAPL is the higher-priority candidate while the active name remains on the watchlist.",
        },
        portfolio_construction_artifact_id="portfolio-1",
    )
    return reports.attach_learning_context(
        report,
        {
            "continuous_learning_version": "phase10_alpha_acceleration_v1",
            "cohort_summary": {
                "tracked_reports": 8,
                "peer_reports": 7,
                "unique_symbols": 7,
                "horizon": "swing",
                "risk_mode": "balanced",
                "prior_learning_cycles": 2,
            },
            "active_setup_archetype": {
                "archetype_id": "watchlist_only_thesis",
                "archetype_name": "Watchlist Only Thesis",
                "summary": "The setup is constructive but still gated by crowding and confirmation needs.",
                "defining_characteristics": ["moderate conviction", "paper-shadow only"],
                "common_failure_modes": ["crowding rises without confirmation"],
                "best_regimes": ["trend"],
                "worst_regimes": ["transition", "high_vol"],
                "strategy_fit": "wait_for_confirmation",
                "deployment_caution_level": "elevated",
            },
            "motif_discovery": {
                "active_motifs": [
                    {
                        "motif_id": "crowding_divergence",
                        "motif_summary": "Narrative crowding remains elevated relative to confirmation quality.",
                    }
                ],
                "motif_library": [],
            },
            "signal_family_library": {
                "archetype_cohorts": [
                    {
                        "archetype_name": "Watchlist Only Thesis",
                        "sample_count": 6,
                        "average_reliability": 58.0,
                    }
                ]
            },
            "regime_conditioned_learnings": [
                {
                    "regime_label": "trend",
                    "sample_size": 8,
                    "average_reliability": 60.0,
                    "average_hit_rate": 0.58,
                    "decision_quality_summary": "Trend setups still work, but crowding suppresses cleaner deployment.",
                    "adaptation_suggestion": "Keep confirmation gates firm until crowding eases.",
                }
            ],
            "feature_interaction_candidates": [
                {
                    "interaction_candidate": "trend_plus_low_fragility_plus_macro_alignment",
                    "description": "This interaction remains the cleanest continuation pattern.",
                }
            ],
            "reweighting_candidates": [
                {
                    "target_family": "Narrative Crowding Index",
                    "suggested_weight_changes": [
                        {"direction": "increase_penalty", "target": "crowding_penalty"}
                    ],
                    "confidence_in_recommendation": 0.74,
                    "sample_size": 8,
                }
            ],
            "research_hypotheses": [
                {
                    "hypothesis_title": "Crowding penalty deserves more weight",
                    "observed_pattern": "Crowded trend setups underperform cleaner continuations.",
                }
            ],
            "drift_alerts": [
                {
                    "affected_component": "confidence_calibration",
                    "severity": "moderate",
                    "evidence": "The active regime still shows weaker reliability than the cohort median.",
                }
            ],
            "experiment_registry": {
                "open_experiments": [
                    {
                        "title": "Tighten crowding penalty in trend regimes",
                        "validation_status": "candidate",
                        "approval_status": "review",
                    }
                ],
                "approved_improvements": [],
                "rejected_improvements": [],
            },
            "improvement_queue": [
                {"title": "Increase crowding penalty in trend regimes", "priority": "high"}
            ],
            "learning_summary": "The active setup is a watchlist-only thesis and crowding discipline remains the top learning priority.",
            "regime_learning_summary": "Trend regimes still work, but reliability softens when crowding stays elevated.",
            "adaptation_queue_summary": "Raise crowding penalties and keep confirmation gates firm until reliability improves.",
            "experiment_registry_summary": "One governed crowding-penalty experiment is queued for review.",
            "archetype_motif_summary": "This setup clusters into the watchlist-only thesis family with a crowding-divergence motif.",
        },
        learning_artifact_id="learning-1",
    )


def test_assistant_analyze_returns_schema(monkeypatch):
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
        lambda *_args, **_kwargs: {"ret_5d": 0.12, "vol_21d": 0.3},
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

    with TestClient(app) as client:
        resp = client.post(
            "/assistant/analyze",
            json={"symbol": "NVDA", "horizon": "swing", "risk_mode": "balanced"},
        )
        assert resp.status_code == 200
        data = resp.json()

    assert {
        "symbol",
        "as_of_date",
        "horizon",
        "risk_mode",
        "analysis_job",
        "freshness_summary",
        "signal",
        "key_features",
        "quality",
        "evidence",
        "data_bundle",
        "domain_availability",
        "feature_factor_bundle",
        "proprietary_scores",
        "factor_groups",
        "regime_intelligence",
        "fragility_intelligence",
        "domain_agreement",
        "conviction_components",
        "opportunity_quality_components",
        "strategy_summary",
        "strategy_posture",
        "confidence_score",
        "actionability_score",
        "participant_fit",
        "scenario_matrix",
        "invalidators",
        "confirmation_triggers",
        "fragility_vetoes",
        "execution_posture",
        "uncertainty_notes",
        "evaluation",
        "evaluation_summary",
        "confidence_reliability_summary",
        "regime_usefulness_summary",
        "evaluation_research_analysis",
        "deployment_readiness",
        "deployment_mode",
        "rollout_stage",
        "deployment_permission",
        "deployment_blockers",
        "deployment_rationale",
        "trust_tier",
        "minimum_required_review",
        "human_review_required",
        "model_readiness_status",
        "live_readiness_score",
        "live_readiness_blockers",
        "risk_budget_tier",
        "deployment_readiness_summary",
        "deployment_permission_analysis",
        "risk_budget_exposure_analysis",
        "rollout_stage_summary",
        "strategy",
        "why_this_signal",
        "signal_summary",
        "technical_analysis",
        "fundamental_analysis",
        "statistical_analysis",
        "sentiment_analysis",
        "macro_geopolitical_analysis",
        "risk_quality_analysis",
        "overall_analysis",
        "strategy_view",
        "risks_weaknesses_invalidators",
        "evidence_provenance",
        "session_id",
        "report_id",
        "prediction_record_artifact_id",
        "evaluation_artifact_id",
        "deployment_readiness_artifact_id",
        "deployment_audit_artifact_id",
        "portfolio_construction",
        "portfolio_construction_artifact_id",
        "portfolio_context_summary",
        "portfolio_fit_analysis",
        "execution_quality_analysis",
        "portfolio_workflow_summary",
        "candidate_classification",
        "portfolio_candidate_score",
        "portfolio_fit_quality",
        "size_band",
        "execution_quality_score",
        "overlap_score",
        "redundancy_score",
        "continuous_learning",
        "learning_artifact_id",
        "research_version",
        "setup_archetype",
        "active_motifs",
        "regime_conditioned_learnings",
        "reweighting_candidates",
        "research_hypotheses",
        "interaction_candidates",
        "learning_drift_alerts",
        "experiment_registry",
        "signal_family_library",
        "motif_library",
        "improvement_queue",
        "learning_priority",
        "learning_summary",
        "regime_learning_summary",
        "adaptation_queue_summary",
        "experiment_registry_summary",
        "archetype_motif_summary",
        "canonical_validation",
        "canonical_validation_summary",
        "walkforward_validation_summary",
        "net_of_friction_validation_summary",
        "suppression_readiness_validation_summary",
        "drawdown_invalidation_validation_summary",
        "canonical_validation_artifact_id",
        "active_analysis",
    }.issubset(data.keys())
    assert data["signal"]["action"] == "BUY"
    assert data["active_analysis"]["symbol"] == "NVDA"
    assert data["strategy"]["final_signal"] in {"BUY", "HOLD", "SELL"}
    assert set(data["strategy"]["scenario_matrix"].keys()) == {"base", "bull", "bear", "stress"}
    assert data["strategy"]["execution_posture"]["preferred_posture"]
    assert data["deployment_readiness"]["deployment_readiness_version"]
    assert data["active_analysis"]["deployment_permission"]
    assert data["active_analysis"]["candidate_classification"]
    assert data["active_analysis"]["setup_archetype"]
    assert data["canonical_validation"]["validation_version"]
    assert data["overall_analysis"]


def test_assistant_analyze_surfaces_enriched_data_fabric(monkeypatch):
    from api.assistant import orchestrator
    from api.assistant import intelligence

    monkeypatch.setenv("FTIP_DATA_FABRIC_ENABLED", "1")

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
            "action": "HOLD",
            "score": 0.1,
            "confidence": 0.55,
            "entry_low": 100,
            "entry_high": 110,
            "stop_loss": 95,
            "take_profit_1": 115,
            "take_profit_2": 120,
            "horizon_days": 21,
            "reason_codes": ["MIXED"],
            "reason_details": {"MIXED": "Mixed setup"},
        },
    )
    monkeypatch.setattr(
        orchestrator,
        "fetch_key_features",
        lambda *_args, **_kwargs: {"ret_5d": 0.01, "vol_21d": 0.3},
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
    monkeypatch.setattr(
        intelligence.data_fabric,
        "enrich_data_bundle",
        lambda **_kwargs: {
            "enabled": True,
            "status": "ok",
            "domains": {
                "fundamental_filing": {
                    "filing_backbone": {
                        "latest_form": "10-Q",
                        "latest_filing_date": "2024-01-01",
                        "latest_10k": {"filing_date": "2023-03-01"},
                        "latest_10q": {"filing_date": "2024-01-01"},
                    },
                    "statement_snapshot": {
                        "latest_quarter": {"revenue": 1200, "report_date": "2024-01-01"},
                    },
                    "normalized_metrics": {
                        "revenue_growth_yoy": 0.18,
                        "operating_margin": 0.24,
                        "net_margin": 0.2,
                        "current_ratio": 2.1,
                        "debt_to_equity": 0.4,
                        "free_cash_flow_margin": 0.16,
                    },
                    "quality_proxies": {
                        "reporting_quality_proxy": 82.0,
                        "business_quality_durability": 79.0,
                    },
                    "coverage_score": 0.88,
                    "strength_summary": ["Quarterly revenue growth remains strong."],
                    "weakness_summary": ["Cash-flow detail is based on quarterly facts only."],
                    "coverage_caveats": ["Leverage detail is partly supported by Finnhub metrics."],
                    "filing_recency_days": 45,
                    "meta": {
                        "sources": ["sec_edgar", "finnhub_basic_financials"],
                        "status": "fresh",
                    },
                },
                "sentiment_narrative_flow": {
                    "source_breakdown": {"gnews": 3, "gdelt": 2},
                    "aggregated_sentiment_bias": 0.12,
                    "meta": {"sources": ["gnews", "gdelt"]},
                },
                "macro_cross_asset": {
                    "macro_regime_context": {"regime": "growth_supportive"},
                    "fred_series": {"rates": {"latest": 4.2}},
                    "meta": {"sources": ["fred", "world_bank"]},
                },
                "geopolitical_policy": {
                    "event_buckets": {"policy_regulation": 2},
                    "event_intensity_score": 0.4,
                    "meta": {"sources": ["gdelt"]},
                },
                "quality_provenance": {
                    "source_map": {
                        "fundamental_filing": ["sec_edgar", "finnhub_basic_financials"],
                        "sentiment_narrative_flow": ["gnews", "gdelt"],
                    },
                    "provider_notes": [],
                    "meta": {"status": "fresh"},
                },
            },
        },
    )

    with TestClient(app) as client:
        resp = client.post(
            "/assistant/analyze",
            json={"symbol": "NVDA", "horizon": "swing", "risk_mode": "balanced"},
        )

    assert resp.status_code == 200
    data = resp.json()
    assert data["data_bundle"]["fundamental_filing"]["filing_recency_days"] == 45
    assert data["data_bundle"]["sentiment_narrative_flow"]["source_breakdown"]["gnews"] == 3


def test_assistant_chat_endpoint_returns_grounded_active_report(monkeypatch):
    from api.assistant import reports, service

    monkeypatch.setenv("FTIP_LLM_ENABLED", "1")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    store = service.storage
    store.use_memory = True
    store._sessions.clear()
    store._messages.clear()
    store._artifacts.clear()

    session_id = store.create_session()
    report = _sample_chat_report("NVDA")
    report_id = store.save_artifact(session_id, reports.ANALYSIS_REPORT_KIND, report)
    store.upsert_session_metadata(
        session_id,
        {
            "active_analysis": reports.build_active_analysis_reference(
                report,
                session_id=session_id,
                report_id=report_id,
            )
        },
    )

    monkeypatch.setattr(
        service,
        "_safe_completion",
        lambda messages: (
            "Grounded strategist response for the active NVDA report.",
            "model",
            {"prompt_tokens": 20, "completion_tokens": 25},
        ),
    )

    with TestClient(app) as client:
        resp = client.post(
            "/assistant/chat",
            json={
                "session_id": session_id,
                "message": "Why is this HOLD instead of BUY?",
            },
        )

    assert resp.status_code == 200
    data = resp.json()
    assert data["report_found"] is True
    assert data["reply"] == "Grounded strategist response for the active NVDA report."
    assert data["active_analysis"]["symbol"] == "NVDA"
    assert data["active_analysis"]["conviction_tier"] == "moderate"
    assert data["active_analysis"]["strategy_posture"] == "watchlist_positive"
    assert data["active_analysis"]["candidate_classification"] == "watchlist_candidate"
    assert data["active_analysis"]["size_band"] == "paper / shadow band"
    assert data["active_analysis"]["setup_archetype"] == "Watchlist Only Thesis"
    assert "strategy_view" in data["citations"]


def test_assistant_top_picks_schema(monkeypatch):
    from api.assistant import orchestrator

    monkeypatch.setattr(
        orchestrator,
        "fetch_top_picks",
        lambda limit: (
            dt.date(2024, 1, 2),
            [
                {
                    "symbol": "NVDA",
                    "direction": "long",
                    "score": 0.7,
                    "confidence": 0.6,
                    "reason_codes": ["MOMO_UP"],
                }
            ],
        ),
    )
    monkeypatch.setattr(
        orchestrator, "universe_coverage", lambda *_args, **_kwargs: 0.95
    )

    with TestClient(app) as client:
        resp = client.post(
            "/assistant/top-picks",
            json={
                "universe": "sp500",
                "horizon": "swing",
                "risk_mode": "balanced",
                "limit": 1,
            },
        )
        assert resp.status_code == 200
        data = resp.json()

    assert "picks" in data
    assert data["picks"][0]["symbol"] == "NVDA"


def test_assistant_analyze_sanitizes_non_finite_numeric_values(monkeypatch):
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
            "score": Decimal("0.7"),
            "confidence": Decimal("Infinity"),
            "entry_low": 100.0,
            "entry_high": 110.0,
            "stop_loss": 90.0,
            "take_profit_1": 130.0,
            "take_profit_2": 150.0,
            "horizon_days": 10,
            "reason_codes": ["MOMO_UP"],
            "reason_details": {"MOMO_UP": "Momentum rising"},
        },
    )
    monkeypatch.setattr(
        orchestrator,
        "fetch_key_features",
        lambda *_args, **_kwargs: {
            "ret_5d": Decimal("-Infinity"),
            "finite_decimal": Decimal("1.25"),
            "vol_21d": Decimal("0.3"),
            "nested": {
                "feature": float("nan"),
                "decimal_feature": Decimal("NaN"),
                "items": [Decimal("1.5"), float("-inf"), {"z": Decimal("Infinity")}],
            },
        },
    )
    monkeypatch.setattr(
        orchestrator,
        "fetch_quality",
        lambda *_args, **_kwargs: {
            "bars_ok": True,
            "news_ok": True,
            "sentiment_ok": True,
            "risk": {"drawdown": float("inf"), "sharpe": Decimal("1.2")},
            "trace": [0.1, math.nan, {"z": Decimal("NaN"), "w": math.inf}],
            "warnings": [],
        },
    )

    with TestClient(app) as client:
        resp = client.post(
            "/assistant/analyze",
            json={"symbol": "NVDA", "horizon": "swing", "risk_mode": "balanced"},
        )

    assert resp.status_code == 200
    data = resp.json()
    assert data["signal"]["score"] == 0.7
    assert data["signal"]["confidence"] is None
    assert data["signal"]["entry_low"] == 100.0
    assert data["key_features"]["ret_5d"] is None
    assert data["key_features"]["vol_21d"] == 0.3
    assert data["key_features"]["finite_decimal"] == 1.25
    assert data["key_features"]["nested"]["feature"] is None
    assert data["key_features"]["nested"]["decimal_feature"] is None
    assert data["key_features"]["nested"]["items"] == [1.5, None, {"z": None}]
    assert data["quality"]["risk"]["drawdown"] is None
    assert data["quality"]["risk"]["sharpe"] == 1.2
    assert data["quality"]["trace"] == [0.1, None, {"z": None, "w": None}]
    assert data["signal_summary"]
    assert data["strategy_view"]
    assert data["freshness_summary"]["overall_status"]
    assert data["why_this_signal"]["top_positive_drivers"] is not None


def test_fetch_signal_falls_back_to_prosperity_row(monkeypatch):
    from api.assistant import orchestrator

    calls = {"count": 0}

    def _fake_fetchone(query, params):
        calls["count"] += 1
        if "FROM signals_daily" in query:
            return None
        if "FROM prosperity_signals_daily" in query:
            assert "as_of = %s" in query
            assert params == ("AAPL", dt.date(2024, 1, 2))
            return ("BUY", 0.81, 0.64)
        raise AssertionError(f"unexpected query: {query}")

    monkeypatch.setattr(orchestrator.db, "safe_fetchone", _fake_fetchone)

    signal = orchestrator.fetch_signal("AAPL", dt.date(2024, 1, 2))

    assert calls["count"] == 2
    assert signal == {
        "action": "BUY",
        "score": 0.81,
        "confidence": 0.64,
        "entry_low": None,
        "entry_high": None,
        "stop_loss": None,
        "take_profit_1": None,
        "take_profit_2": None,
        "horizon_days": None,
        "reason_codes": [],
        "reason_details": {},
    }


def test_fetch_signal_falls_back_to_prosperity_row_with_as_of_date(monkeypatch):
    from api.assistant import orchestrator

    orchestrator._PROSPERITY_SIGNAL_ASOF_COLUMN = None
    orchestrator._PROSPERITY_SIGNAL_ACTION_COLUMN = None

    def _fake_fetchone(query, params=None):
        if "FROM signals_daily" in query:
            return None
        if "FROM information_schema.columns" in query and "as_of_date" in query:
            return ("as_of_date",)
        if "FROM information_schema.columns" in query and "signal', 'action" in query:
            return ("signal",)
        if "FROM prosperity_signals_daily" in query:
            assert "as_of_date = %s" in query
            return ("SELL", -0.22, 0.51)
        raise AssertionError(f"unexpected query: {query}")

    monkeypatch.setattr(orchestrator.db, "safe_fetchone", _fake_fetchone)

    signal = orchestrator.fetch_signal("AAPL", dt.date(2024, 1, 2))

    assert signal is not None
    assert signal["action"] == "SELL"
    assert signal["score"] == -0.22
    assert signal["confidence"] == 0.51
