import asyncio
from typing import Any

from fastapi.testclient import TestClient

from api.assistant import intelligence, reports, service, strategy
from api.assistant.phase8 import (
    DEPLOYMENT_AUDIT_RECORD_KIND,
    DEPLOYMENT_READINESS_ARTIFACT_KIND,
)
from api.assistant.phase9 import PORTFOLIO_CONSTRUCTION_ARTIFACT_KIND
from api.assistant.phase10 import CONTINUOUS_LEARNING_ARTIFACT_KIND
from api.assistant.storage import AssistantStorage
from api.main import app


class DummySignal:
    def __init__(self, symbol: str = "AAPL"):
        self.symbol = symbol

    def model_dump(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "as_of": "2024-01-01",
            "lookback": 10,
            "signal": "BUY",
            "score": 1.0,
            "confidence": 0.8,
            "thresholds": {"buy": 0.5},
            "notes": ["test"],
        }


class DummyBacktest:
    def model_dump(self) -> dict[str, Any]:
        return {
            "total_return": 0.1,
            "sharpe": 1.0,
            "max_drawdown": -0.05,
            "volatility": 0.2,
            "lookback": 252,
        }


def _attach_learning_test_context(report: dict[str, Any]) -> dict[str, Any]:
    return reports.attach_learning_context(
        report,
        {
            "continuous_learning_version": "phase10_alpha_acceleration_v1",
            "cohort_summary": {
                "tracked_reports": 9,
                "peer_reports": 8,
                "unique_symbols": 8,
                "horizon": report.get("horizon"),
                "risk_mode": report.get("risk_mode"),
                "prior_learning_cycles": 3,
            },
            "active_setup_archetype": {
                "archetype_id": "watchlist_only_thesis",
                "archetype_name": "Watchlist Only Thesis",
                "summary": "The setup is analytically constructive but still constrained by crowding and confirmation needs.",
                "defining_characteristics": ["moderate conviction", "paper-shadow only"],
                "common_failure_modes": ["crowding rises without price confirmation"],
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
                        "sample_count": 7,
                        "average_reliability": 59.0,
                    }
                ]
            },
            "regime_conditioned_learnings": [
                {
                    "regime_label": "trend",
                    "sample_size": 9,
                    "average_reliability": 61.0,
                    "average_hit_rate": 0.58,
                    "decision_quality_summary": "Trend setups remain constructive, but crowding still depresses cleaner deployment.",
                    "adaptation_suggestion": "Keep confirmation and crowding penalties firm until reliability improves.",
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
                    "confidence_in_recommendation": 0.75,
                    "sample_size": 9,
                }
            ],
            "research_hypotheses": [
                {
                    "hypothesis_title": "Crowding penalty deserves more weight",
                    "observed_pattern": "Crowded trend setups underperform cleaner continuation setups.",
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
            "learning_summary": "The active setup sits in a watchlist-only archetype and crowding discipline remains the top learning priority.",
            "regime_learning_summary": "Trend regimes still work, but reliability weakens when crowding remains elevated.",
            "adaptation_queue_summary": "Raise crowding penalties and keep confirmation gates tight until reliability improves.",
            "experiment_registry_summary": "One governed crowding-penalty experiment is queued for review.",
            "archetype_motif_summary": "This setup clusters into the watchlist-only thesis family with a crowding-divergence motif.",
        },
        learning_artifact_id="learning-test",
    )


def test_chat_returns_503_when_disabled(monkeypatch):
    monkeypatch.setenv("FTIP_LLM_ENABLED", "0")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    client = TestClient(app)
    resp = client.post("/assistant/chat", json={"message": "hi"})
    assert resp.status_code == 503


def test_chat_missing_api_key(monkeypatch):
    monkeypatch.setenv("FTIP_LLM_ENABLED", "1")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    client = TestClient(app)
    resp = client.post("/assistant/chat", json={"message": "hi"})
    assert resp.status_code == 500
    assert "LLM API key not configured" in resp.json()["error"]["message"]


def test_storage_memory_roundtrip():
    store = AssistantStorage(use_memory=True)
    session_id = store.create_session(metadata={"foo": "bar"})
    store.add_message(session_id, "user", "hello")
    store.upsert_session_metadata(session_id, {"title": "Test"})
    artifact_id = store.save_artifact(
        session_id,
        reports.ANALYSIS_REPORT_KIND,
        {
          "symbol": "AAPL",
          "as_of_date": "2024-01-01",
          "horizon": "swing",
          "risk_mode": "balanced",
          "overall_analysis": "Stored report.",
        },
    )

    session = store.get_session(session_id)
    assert session is not None
    assert session["metadata"].get("foo") == "bar"
    assert session["metadata"].get("title") == "Test"

    messages = store.get_messages(session_id)
    assert len(messages) == 1
    assert messages[0]["content"] == "hello"
    report = store.get_latest_analysis_report(
        session_id=session_id,
        symbol="AAPL",
        horizon="swing",
        risk_mode="balanced",
    )
    assert report is not None
    assert report["report_id"] == artifact_id
    assert report["overall_analysis"] == "Stored report."


def test_generate_analysis_report_persists_artifact(monkeypatch):
    store = AssistantStorage(use_memory=True)

    async def _fake_freshness(symbol: str, refresh: bool = True):
        return {
            "as_of_date": "2024-01-02",
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

    monkeypatch.setattr(service.orchestrator, "ensure_freshness", _fake_freshness)
    monkeypatch.setattr(service.orchestrator, "run_features", _noop)
    monkeypatch.setattr(service.orchestrator, "run_signals", _noop)
    monkeypatch.setattr(
        service.orchestrator,
        "fetch_signal",
        lambda *_args, **_kwargs: {
            "action": "BUY",
            "score": 0.9,
            "confidence": 0.7,
            "entry_low": 100,
            "entry_high": 104,
            "stop_loss": 95,
            "take_profit_1": 112,
            "take_profit_2": 120,
            "horizon_days": 21,
            "reason_codes": ["TREND_UP"],
            "reason_details": {"TREND_UP": "Trend is rising"},
        },
    )
    monkeypatch.setattr(
        service.orchestrator,
        "fetch_key_features",
        lambda *_args, **_kwargs: {
            "ret_1d": 0.01,
            "ret_5d": 0.03,
            "ret_21d": 0.12,
            "vol_21d": 0.25,
            "vol_63d": 0.22,
            "atr_pct": 0.04,
            "trend_slope_21d": 0.2,
            "trend_slope_63d": 0.1,
            "mom_vol_adj_21d": 0.8,
            "sentiment_score": 0.4,
            "sentiment_surprise": 0.1,
            "regime_label": "trend",
            "regime_strength": 0.6,
        },
    )
    monkeypatch.setattr(
        service.orchestrator,
        "fetch_quality",
        lambda *_args, **_kwargs: {
            "bars_ok": True,
            "fundamentals_ok": False,
            "sentiment_ok": True,
            "intraday_ok": False,
            "missingness": 0.02,
            "anomaly_flags": [],
            "quality_score": 88,
            "news_ok": True,
            "bars_updated_at": "2024-01-02T00:00:00Z",
            "news_updated_at": "2024-01-02T00:00:00Z",
            "sentiment_updated_at": "2024-01-02T00:00:00Z",
            "warnings": [],
        },
    )

    result = asyncio.run(
        service.generate_analysis_report(
            {
                "symbol": "NVDA",
                "horizon": "swing",
                "risk_mode": "balanced",
            },
            store=store,
        )
    )

    assert result["report_id"]
    assert result["session_id"]
    assert result["overall_analysis"]
    assert result["strategy_view"]
    assert result["analysis_job"]["trace_id"]
    assert result["freshness_summary"]["overall_status"]
    assert result["data_bundle"]["quality_provenance"]
    assert result["domain_availability"]
    assert result["data_bundle"]["quality_provenance"]["domain_availability"]
    assert result["data_bundle"]["normalized_domains"]["news_sentiment_narrative"] == result["data_bundle"]["sentiment_narrative_flow"]
    assert result["feature_factor_bundle"]["composite_intelligence"]
    assert result["strategy"]["final_signal"]
    assert result["strategy"]["strategy_posture"]
    assert set(result["strategy"]["scenario_matrix"].keys()) == {"base", "bull", "bear", "stress"}
    assert result["strategy"]["execution_posture"]["preferred_posture"]
    assert result["prediction_record_artifact_id"]
    assert result["evaluation_artifact_id"]
    assert result["deployment_readiness_artifact_id"]
    assert result["deployment_audit_artifact_id"]
    assert result["portfolio_construction_artifact_id"]
    assert result["evaluation"]["evaluation_version"]
    assert result["evaluation_research_analysis"]
    assert result["deployment_readiness"]["deployment_readiness_version"]
    assert result["deployment_permission"]
    assert result["deployment_readiness_summary"]
    assert result["portfolio_construction"]["portfolio_construction_version"]
    assert result["portfolio_context_summary"]
    assert result["portfolio_fit_analysis"]
    assert result["execution_quality_analysis"]
    assert result["candidate_classification"]
    assert result["size_band"]
    assert result["learning_artifact_id"]
    assert result["continuous_learning"]["continuous_learning_version"]
    assert result["learning_summary"]
    assert result["regime_learning_summary"]
    assert result["adaptation_queue_summary"]
    assert result["experiment_registry_summary"]
    assert result["archetype_motif_summary"]
    assert result["setup_archetype"]["archetype_name"]
    assert result["learning_priority"]
    assert result["actionability_score"] is not None
    assert result["why_this_signal"]["top_positive_drivers"] is not None
    assert result["evidence_provenance"]

    session = store.get_session(result["session_id"])
    assert session is not None
    assert session["metadata"]["active_analysis"]["report_id"] == result["report_id"]
    assert session["metadata"]["active_analysis"]["signal"]
    assert session["metadata"]["active_analysis"]["deployment_permission"]
    assert session["metadata"]["active_analysis"]["candidate_classification"]
    assert session["metadata"]["active_analysis"]["setup_archetype"]
    assert session["metadata"]["deployment_readiness"]["artifact_id"] == result["deployment_readiness_artifact_id"]
    assert (
        session["metadata"]["portfolio_construction"]["artifact_id"]
        == result["portfolio_construction_artifact_id"]
    )
    assert (
        session["metadata"]["continuous_learning"]["artifact_id"]
        == result["learning_artifact_id"]
    )

    report = store.get_latest_analysis_report(session_id=result["session_id"], symbol="NVDA")
    assert report is not None
    assert report["report_id"] == result["report_id"]
    assert report["signal_summary"] == result["signal_summary"]
    assert report["overall_analysis"] == result["overall_analysis"]
    assert report["evaluation"]
    assert report["deployment_readiness"]
    assert report["portfolio_construction"]
    assert report["continuous_learning"]
    assert report["live_use_audit_snapshot"]
    assert (
        store.get_latest_artifact(
            kind=intelligence.ANALYSIS_JOB_KIND, session_id=result["session_id"]
        )
        is not None
    )
    assert (
        store.get_latest_artifact(
            kind=intelligence.DATA_BUNDLE_KIND, session_id=result["session_id"]
        )
        is not None
    )
    assert (
        store.get_latest_artifact(
            kind=intelligence.FEATURE_FACTOR_BUNDLE_KIND,
            session_id=result["session_id"],
        )
        is not None
    )
    assert (
        store.get_latest_artifact(
            kind=strategy.STRATEGY_ARTIFACT_KIND, session_id=result["session_id"]
        )
        is not None
    )
    assert (
        store.get_latest_artifact(
            kind="assistant_prediction_record", session_id=result["session_id"]
        )
        is not None
    )
    assert (
        store.get_latest_artifact(
            kind="assistant_evaluation_artifact", session_id=result["session_id"]
        )
        is not None
    )
    assert (
        store.get_latest_artifact(
            kind=DEPLOYMENT_READINESS_ARTIFACT_KIND, session_id=result["session_id"]
        )
        is not None
    )
    assert (
        store.get_latest_artifact(
            kind=DEPLOYMENT_AUDIT_RECORD_KIND, session_id=result["session_id"]
        )
        is not None
    )
    assert (
        store.get_latest_artifact(
            kind=PORTFOLIO_CONSTRUCTION_ARTIFACT_KIND, session_id=result["session_id"]
        )
        is not None
    )
    assert (
        store.get_latest_artifact(
            kind=CONTINUOUS_LEARNING_ARTIFACT_KIND, session_id=result["session_id"]
        )
        is not None
    )


def test_market_domain_computes_drawdown_windows_without_name_error(monkeypatch):
    as_of_date = "2024-01-02"
    daily_bars = []
    for index in range(140):
        close = 100.0 + index * 0.6
        if 40 <= index <= 55:
            close -= (index - 39) * 1.8
        if 56 <= index <= 70:
            close -= max(0, 27.0 - (index - 55) * 1.5)
        daily_bars.append(
            {
                "as_of_date": f"2023-08-{(index % 28) + 1:02d}",
                "open": close - 0.5,
                "high": close + 1.0,
                "low": close - 1.2,
                "close": close,
                "volume": 1_000_000 + index * 5_000,
                "source": "test",
                "ingested_at": f"{as_of_date}T00:00:00Z",
            }
        )

    monkeypatch.setattr(intelligence, "_load_daily_bars", lambda *_args, **_kwargs: daily_bars)
    monkeypatch.setattr(intelligence, "_load_intraday_bars", lambda *_args, **_kwargs: [])

    market_domain, _, _ = intelligence._market_domain(
        "NVDA",
        intelligence.dt.date(2024, 1, 2),
        {"bars_updated_at": f"{as_of_date}T00:00:00Z"},
        {"vol_21d": 0.25, "vol_63d": 0.22, "atr_pct": 0.03},
        {"bars_ok": True},
    )

    assert market_domain["maxdd_21d"] is not None
    assert market_domain["maxdd_63d"] is not None
    assert market_domain["maxdd_126d"] is not None
    assert market_domain["maxdd_21d"] <= 0
    assert market_domain["maxdd_63d"] <= 0
    assert market_domain["maxdd_126d"] <= 0


def test_chat_uses_persisted_analysis_report(monkeypatch):
    monkeypatch.setenv("FTIP_LLM_ENABLED", "1")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    store = AssistantStorage(use_memory=True)
    session_id = store.create_session()
    report = reports.build_analysis_report(
        symbol="NVDA",
        as_of_date="2024-01-02",
        horizon="swing",
        risk_mode="balanced",
        signal={
            "action": "BUY",
            "score": 0.8,
            "confidence": 0.7,
            "entry_low": 100,
            "entry_high": 104,
            "stop_loss": 95,
            "take_profit_1": 110,
            "take_profit_2": 118,
            "horizon_days": 21,
            "reason_codes": ["TREND_UP"],
            "reason_details": {"TREND_UP": "Trend is rising"},
        },
        key_features={
            "ret_1d": 0.01,
            "ret_5d": 0.03,
            "ret_21d": 0.08,
            "vol_21d": 0.25,
            "vol_63d": 0.22,
            "atr_pct": 0.04,
            "trend_slope_21d": 0.2,
            "trend_slope_63d": 0.1,
            "mom_vol_adj_21d": 0.8,
            "sentiment_score": 0.4,
            "sentiment_surprise": 0.1,
            "regime_label": "trend",
            "regime_strength": 0.6,
        },
        quality={
            "bars_ok": True,
            "fundamentals_ok": False,
            "sentiment_ok": True,
            "intraday_ok": False,
            "missingness": 0.02,
            "anomaly_flags": [],
            "quality_score": 88,
            "news_ok": True,
            "warnings": [],
        },
        evidence={
            "reason_codes": ["TREND_UP"],
            "reason_details": {"TREND_UP": "Trend is rising"},
            "sources": ["market_bars_daily", "news_raw", "sentiment_daily"],
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
    report_id = store.save_artifact(session_id, reports.ANALYSIS_REPORT_KIND, report)
    store.upsert_session_metadata(
        session_id,
        {
            "active_analysis": reports.build_active_analysis_reference(
                report, session_id=session_id, report_id=report_id
            )
        },
    )

    def _fake_completion(messages):
        combined = "\n".join(message["content"] for message in messages)
        assert "NVDA" in combined
        assert '"question_intent": "strategy"' in combined
        assert '"answer_mode": "strategist"' in combined
        assert report["overall_analysis"] in combined
        assert report["strategy_view"] in combined
        assert report["evidence_provenance"] in combined
        return (
            "Grounded reply about the stored NVDA analysis.",
            "model",
            {"prompt_tokens": 10, "completion_tokens": 12},
        )

    monkeypatch.setattr(service, "_safe_completion", _fake_completion)
    result = service.chat_with_assistant(
        {"session_id": session_id, "message": "Should I buy NVDA based on the analysis?"},
        store=store,
    )
    assert result["reply"] == "Grounded reply about the stored NVDA analysis."
    assert result["report_found"] is True
    assert result["active_analysis"]["symbol"] == "NVDA"
    grounding = store.get_latest_artifact(
        kind=strategy.CHAT_GROUNDING_CONTEXT_KIND, session_id=session_id
    )
    assert grounding is not None
    assert grounding["payload"]["report_found"] is True
    assert grounding["payload"]["route"]["intent"] == "strategy"
    assert grounding["payload"]["selected_sections"]["strategy_view"] == report["strategy_view"]


def test_chat_routes_deployment_readiness_questions_to_readiness_sections(monkeypatch):
    monkeypatch.setenv("FTIP_LLM_ENABLED", "1")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    store = AssistantStorage(use_memory=True)
    session_id = store.create_session()
    report = reports.attach_deployment_context(
        reports.build_analysis_report(
            symbol="NVDA",
            as_of_date="2024-01-02",
            horizon="swing",
            risk_mode="balanced",
            signal={
                "action": "BUY",
                "score": 0.8,
                "confidence": 0.7,
                "reason_codes": ["TREND_UP"],
                "reason_details": {"TREND_UP": "Trend is rising"},
            },
            key_features={"ret_21d": 0.08, "vol_21d": 0.25, "regime_label": "trend"},
            quality={"bars_ok": True, "news_ok": True, "sentiment_ok": True, "warnings": []},
            evidence={"reason_codes": ["TREND_UP"], "reason_details": {}, "sources": ["market_bars_daily"]},
            strategy={
                "final_signal": "HOLD",
                "strategy_posture": "watchlist_positive",
                "confidence": 0.57,
                "confidence_score": 57.0,
                "conviction_tier": "moderate",
                "actionability_score": 44.0,
                "scenario_matrix": {"base": {"summary": "Constructive but not fully actionable."}},
            },
        ),
        {
            "deployment_readiness_version": "phase8_capital_readiness_v1",
            "deployment_mode": {
                "active_mode": "low_risk_live",
                "rollout_stage": "low_risk_live_pilot",
            },
            "model_readiness": {
                "model_readiness_status": "constrained",
                "live_readiness_score": 54.0,
                "live_readiness_blockers": ["confidence calibration quality is not strong enough for live escalation"],
                "recent_degradation_flags": ["historical weakness is concentrated in the current regime (trend)"],
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
                "rollout_stage": "low_risk_live_pilot",
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
        readiness_artifact_id="readiness-2",
        deployment_audit_artifact_id="audit-2",
    )
    report_id = store.save_artifact(session_id, reports.ANALYSIS_REPORT_KIND, report)
    active_analysis = reports.build_active_analysis_reference(
        report, session_id=session_id, report_id=report_id
    )
    store.upsert_session_metadata(session_id, {"active_analysis": active_analysis})

    def _fake_completion(messages):
        combined = "\n".join(message["content"] for message in messages)
        assert '"question_intent": "deployment_readiness"' in combined
        assert '"answer_mode": "deployment"' in combined
        assert report["deployment_readiness_summary"] in combined
        assert report["deployment_permission_analysis"] in combined
        assert report["risk_budget_exposure_analysis"] in combined
        return (
            "The setup remains paper-shadow only because live-readiness and calibration are still constrained.",
            "model",
            {"prompt_tokens": 12, "completion_tokens": 16},
        )

    monkeypatch.setattr(service, "_safe_completion", _fake_completion)
    result = service.chat_with_assistant(
        {"session_id": session_id, "message": "Is this ready for live capital or only paper mode?"},
        store=store,
    )

    assert result["report_found"] is True
    assert result["active_analysis"]["deployment_permission"] == "paper_shadow_only"
    assert result["active_analysis"]["trust_tier"] == "paper_only"


def test_chat_routes_portfolio_questions_to_portfolio_sections(monkeypatch):
    monkeypatch.setenv("FTIP_LLM_ENABLED", "1")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    store = AssistantStorage(use_memory=True)
    session_id = store.create_session()
    report = reports.attach_portfolio_context(
        reports.attach_deployment_context(
            reports.build_analysis_report(
                symbol="NVDA",
                as_of_date="2024-01-02",
                horizon="swing",
                risk_mode="balanced",
                signal={
                    "action": "BUY",
                    "score": 0.8,
                    "confidence": 0.7,
                    "reason_codes": ["TREND_UP"],
                    "reason_details": {"TREND_UP": "Trend is rising"},
                    "horizon_days": 21,
                },
                key_features={"ret_21d": 0.08, "vol_21d": 0.25, "regime_label": "trend"},
                quality={"bars_ok": True, "news_ok": True, "sentiment_ok": True, "warnings": []},
                evidence={"reason_codes": ["TREND_UP"], "reason_details": {}, "sources": ["market_bars_daily"]},
                strategy={
                    "final_signal": "HOLD",
                    "strategy_posture": "watchlist_positive",
                    "confidence": 0.57,
                    "confidence_score": 57.0,
                    "conviction_tier": "moderate",
                    "actionability_score": 44.0,
                    "primary_participant_fit": "swing trader",
                    "participant_fit": ["swing trader", "wait / observe"],
                    "scenario_matrix": {"base": {"summary": "Constructive but not fully actionable."}},
                    "execution_posture": {
                        "preferred_posture": "wait_for_confirmation",
                        "urgency_level": "measured",
                        "patience_level": "high",
                        "signal_cleanliness": "mixed_clean",
                        "entry_quality_proxy": 51.0,
                        "risk_context_summary": "constructive but still crowded",
                    },
                },
            ),
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
                    "deployment_blockers": ["portfolio overlap is already elevated"],
                    "deployment_rationale": "The setup is analytically useful but still paper-only.",
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
                    "demotion_criteria": ["fragility rises into the blocked zone"],
                    "stage_transition_notes": ["Continue paper/shadow evidence collection before live escalation."],
                },
                "drift_monitor": {
                    "pause_recommended": False,
                    "degrade_to_paper_recommended": False,
                    "drift_alerts": ["confidence reliability is below the live-support comfort zone"],
                    "deployment_risk_alerts": [],
                },
                "audit_snapshot": {
                    "rationale_summary": "Paper-shadow only until calibration and overlap improve.",
                },
            },
            readiness_artifact_id="readiness-3",
            deployment_audit_artifact_id="audit-3",
        ),
        {
            "portfolio_construction_version": "phase9_portfolio_construction_v1",
            "current_candidate": {
                "symbol": "NVDA",
                "candidate_classification": "watchlist_candidate",
                "ranked_opportunity_score": 64.0,
                "portfolio_candidate_score": 61.5,
                "watchlist_priority_score": 68.0,
                "deployability_rank": 54.0,
                "portfolio_fit_quality": 49.0,
                "overlap_score": 78.0,
                "redundancy_score": 81.0,
                "diversification_contribution_score": 29.0,
                "most_redundant_symbol": "AAPL",
                "size_band": "paper / shadow band",
                "weight_band": "0.00x live weight",
                "risk_budget_band": "shadow_risk_band",
                "execution_quality_score": 52.0,
                "friction_penalty": 38.0,
                "turnover_penalty": 46.0,
                "wait_for_better_entry_flag": True,
                "confirmation_preferred_flag": True,
                "candidate_blockers": [
                    "the idea is redundant with existing tracked exposures",
                    "deployment gating blocks portfolio use",
                ],
                "concentration_warning": "Sector concentration is rising.",
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
                    "symbol": "NVDA",
                    "candidate_classification": "watchlist_candidate",
                    "portfolio_candidate_score": 61.5,
                    "portfolio_fit_quality": 49.0,
                    "size_band": "paper / shadow band",
                    "deployment_permission": "paper_shadow_only",
                    "strategy_posture": "watchlist_positive",
                    "conviction_tier": "moderate",
                },
            ],
            "workflow": {
                "candidate_watchlist": ["NVDA"],
                "prioritized_watchlist": ["AAPL", "NVDA"],
                "active_portfolio_candidates": ["AAPL"],
                "blocked_candidates": [
                    {
                        "symbol": "NVDA",
                        "classification": "watchlist_candidate",
                        "reasons": ["the idea is redundant with existing tracked exposures"],
                    }
                ],
                "stale_review_needed": [],
                "priority_shift_flag": True,
                "rebalance_attention_flag": True,
                "candidate_upgrade_reason": None,
                "candidate_downgrade_reason": "portfolio redundancy is capping priority",
                "replacement_candidate_notes": "AAPL currently offers a cleaner portfolio-adjusted candidate score.",
                "rotation_pressure_score": 72.0,
            },
            "portfolio_context_summary": "Portfolio rank is 2 of 2 tracked candidates and the setup remains watchlist-only because portfolio fit is modest.",
            "portfolio_fit_analysis": "Overlap and redundancy are elevated versus AAPL, so diversification contribution is limited.",
            "execution_quality_analysis": "Execution quality is acceptable, but confirmation is preferred and the live size band remains blocked.",
            "portfolio_workflow_summary": "AAPL is the higher-priority candidate while NVDA stays on the watchlist for diversification reasons.",
        },
        portfolio_construction_artifact_id="portfolio-1",
    )
    report_id = store.save_artifact(session_id, reports.ANALYSIS_REPORT_KIND, report)
    active_analysis = reports.build_active_analysis_reference(
        report, session_id=session_id, report_id=report_id
    )
    store.upsert_session_metadata(session_id, {"active_analysis": active_analysis})

    def _fake_completion(messages):
        combined = "\n".join(message["content"] for message in messages)
        assert '"question_intent": "portfolio_construction"' in combined
        assert '"answer_mode": "portfolio"' in combined
        assert report["portfolio_context_summary"] in combined
        assert report["portfolio_fit_analysis"] in combined
        assert report["execution_quality_analysis"] in combined
        return (
            "NVDA is analytically constructive, but AAPL currently fits the tracked portfolio better because NVDA is more redundant and remains paper-only.",
            "model",
            {"prompt_tokens": 12, "completion_tokens": 16},
        )

    monkeypatch.setattr(service, "_safe_completion", _fake_completion)
    result = service.chat_with_assistant(
        {"session_id": session_id, "message": "Why is NVDA a watchlist candidate instead of a deployable portfolio idea?"},
        store=store,
    )

    assert result["report_found"] is True
    assert result["active_analysis"]["candidate_classification"] == "watchlist_candidate"
    assert result["active_analysis"]["size_band"] == "paper / shadow band"


def test_chat_routes_learning_questions_to_learning_sections(monkeypatch):
    monkeypatch.setenv("FTIP_LLM_ENABLED", "1")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    store = AssistantStorage(use_memory=True)
    session_id = store.create_session()
    report = _attach_learning_test_context(
        reports.attach_portfolio_context(
            reports.attach_deployment_context(
                reports.build_analysis_report(
                    symbol="NVDA",
                    as_of_date="2024-01-02",
                    horizon="swing",
                    risk_mode="balanced",
                    signal={
                        "action": "BUY",
                        "score": 0.8,
                        "confidence": 0.7,
                        "reason_codes": ["TREND_UP"],
                        "reason_details": {"TREND_UP": "Trend is rising"},
                    },
                    key_features={"ret_21d": 0.08, "vol_21d": 0.25, "regime_label": "trend"},
                    quality={"bars_ok": True, "news_ok": True, "sentiment_ok": True, "warnings": []},
                    evidence={
                        "reason_codes": ["TREND_UP"],
                        "reason_details": {},
                        "sources": ["market_bars_daily"],
                    },
                    strategy={
                        "final_signal": "HOLD",
                        "strategy_posture": "watchlist_positive",
                        "confidence": 0.57,
                        "confidence_score": 57.0,
                        "conviction_tier": "moderate",
                        "actionability_score": 44.0,
                        "primary_participant_fit": "swing trader",
                        "participant_fit": ["swing trader", "wait / observe"],
                        "scenario_matrix": {"base": {"summary": "Constructive but not fully actionable."}},
                    },
                ),
                {
                    "deployment_readiness_version": "phase8_capital_readiness_v1",
                    "deployment_mode": {
                        "active_mode": "paper_shadow",
                        "rollout_stage": "forward_shadow_validation",
                    },
                    "model_readiness": {
                        "model_readiness_status": "constrained",
                        "live_readiness_score": 57.0,
                        "live_readiness_blockers": [
                            "confidence calibration quality is not strong enough for live escalation"
                        ],
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
                        "deployment_rationale": "The setup remains paper-only while reliability improves.",
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
                        "promotion_criteria": [
                            "confidence reliability remains above the stage threshold"
                        ],
                        "demotion_criteria": ["fragility rises into the blocked zone"],
                        "stage_transition_notes": [
                            "Continue paper/shadow evidence collection before live escalation."
                        ],
                    },
                    "drift_monitor": {
                        "pause_recommended": False,
                        "degrade_to_paper_recommended": False,
                        "drift_alerts": [
                            "confidence reliability is below the live-support comfort zone"
                        ],
                        "deployment_risk_alerts": [],
                    },
                    "audit_snapshot": {
                        "rationale_summary": "Paper-shadow only until calibration improves.",
                    },
                },
                readiness_artifact_id="readiness-learning",
                deployment_audit_artifact_id="audit-learning",
            ),
            {
                "portfolio_construction_version": "phase9_portfolio_construction_v1",
                "current_candidate": {
                    "symbol": "NVDA",
                    "candidate_classification": "watchlist_candidate",
                    "ranked_opportunity_score": 64.0,
                    "portfolio_candidate_score": 61.0,
                    "portfolio_fit_quality": 48.0,
                    "size_band": "paper / shadow band",
                    "execution_quality_score": 54.0,
                },
                "workflow": {"prioritized_watchlist": ["NVDA"]},
                "portfolio_context_summary": "NVDA stays on the watchlist while portfolio fit remains modest.",
                "portfolio_fit_analysis": "Overlap and crowding still limit deployability.",
                "execution_quality_analysis": "Confirmation is still preferred before escalation.",
            },
            portfolio_construction_artifact_id="portfolio-learning",
        )
    )
    report_id = store.save_artifact(session_id, reports.ANALYSIS_REPORT_KIND, report)
    active_analysis = reports.build_active_analysis_reference(
        report, session_id=session_id, report_id=report_id
    )
    store.upsert_session_metadata(session_id, {"active_analysis": active_analysis})

    def _fake_completion(messages):
        combined = "\n".join(message["content"] for message in messages)
        assert '"question_intent": "learning_research"' in combined
        assert '"answer_mode": "learning"' in combined
        assert report["learning_summary"] in combined
        assert report["regime_learning_summary"] in combined
        assert report["experiment_registry_summary"] in combined
        return (
            "The platform is currently learning that crowding discipline needs more weight in this watchlist-only setup family.",
            "model",
            {"prompt_tokens": 12, "completion_tokens": 16},
        )

    monkeypatch.setattr(service, "_safe_completion", _fake_completion)
    result = service.chat_with_assistant(
        {"session_id": session_id, "message": "What is the platform learning lately about NVDA?"},
        store=store,
    )

    assert result["report_found"] is True
    assert result["active_analysis"]["setup_archetype"] == "Watchlist Only Thesis"
    assert result["active_analysis"]["learning_priority"] == "high"


def test_chat_returns_no_analysis_message_when_report_absent(monkeypatch):
    monkeypatch.setenv("FTIP_LLM_ENABLED", "1")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    store = AssistantStorage(use_memory=True)

    monkeypatch.setattr(
        service,
        "_safe_completion",
        lambda _messages: (_ for _ in ()).throw(AssertionError("completion should not run")),
    )

    result = service.chat_with_assistant(
        {"message": "Explain NVDA based on the analysis."},
        store=store,
    )
    assert result["report_found"] is False
    assert "No stored analysis report exists for NVDA yet." in result["reply"]


def test_explain_signal_mocked(monkeypatch):
    monkeypatch.setenv("FTIP_LLM_ENABLED", "1")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    monkeypatch.setattr(
        service,
        "_safe_completion",
        lambda messages: (
            "mocked reply",
            "model",
            {"prompt_tokens": 1, "completion_tokens": 1},
        ),
    )
    result = service.explain_signal(
        {"symbol": "AAPL", "as_of": "2024-01-01", "lookback": 10},
        signal_fetcher=lambda symbol, as_of, lookback: DummySignal(symbol),
        store=AssistantStorage(use_memory=True),
    )
    assert result["reply"] == "mocked reply"


def test_explain_backtest_mocked(monkeypatch):
    monkeypatch.setenv("FTIP_LLM_ENABLED", "1")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    monkeypatch.setattr(
        service,
        "_safe_completion",
        lambda messages: (
            "backtest reply",
            "model",
            {"prompt_tokens": 1, "completion_tokens": 1},
        ),
    )
    result = service.explain_backtest(
        {
            "symbols": ["AAPL"],
            "from_date": "2023-01-01",
            "to_date": "2023-12-31",
            "lookback": 252,
            "rebalance_every": 21,
            "trading_cost_bps": 10.0,
            "slippage_bps": 5.0,
            "max_weight": None,
            "min_trade_delta": 0.0005,
            "max_turnover_per_rebalance": 0.25,
            "allow_shorts": False,
        },
        backtest_runner=lambda req: DummyBacktest(),
        store=AssistantStorage(use_memory=True),
    )
    assert result["reply"] == "backtest reply"


def test_feature_factor_and_strategy_layers_produce_structured_outputs():
    data_bundle = {
        "market_price_volume": {
            "day_return": 0.01,
            "ret_5d": 0.04,
            "ret_10d": 0.05,
            "ret_21d": 0.1,
            "ret_63d": 0.2,
            "ret_126d": 0.3,
            "ret_252d": 0.45,
            "realized_vol_5d": 0.22,
            "realized_vol_21d": 0.25,
            "realized_vol_63d": 0.24,
            "atr_pct": 0.04,
            "gap_pct": 0.01,
            "breakout_distance_63d": 0.03,
            "volume_anomaly": 1.4,
            "support_21d": 100.0,
            "resistance_21d": 118.0,
            "compression_ratio": 0.07,
        },
        "technical_market_structure": {
            "moving_averages": {"ma_10": 112, "ma_21": 109, "ma_63": 101, "ma_126": 92},
            "trend_slope_21d": 0.12,
            "trend_slope_63d": 0.08,
            "trend_curvature": 0.04,
            "mean_reversion_gap": 0.05,
            "breakout_state": "trend_extension",
            "volume_price_alignment": 0.09,
            "regime_label": "trend",
            "regime_strength": 0.6,
        },
        "fundamental_filing": {
            "latest_quarter": {"revenue": 1000, "op_margin": 0.24, "gross_margin": 0.61},
            "revenue_growth_yoy": 0.18,
            "margin_stability": 0.78,
            "positive_fcf_ratio": 0.75,
            "filing_recency_days": 54,
        },
        "sentiment_narrative_flow": {
            "sentiment_score": 0.32,
            "sentiment_surprise": 0.08,
            "sentiment_trend": 0.02,
            "headline_count": 14,
            "attention_crowding": 1.4,
            "novelty_ratio": 0.7,
            "narrative_concentration": 0.32,
            "disagreement_score": 0.18,
            "hype_price_divergence": 0.04,
        },
        "macro_cross_asset": {
            "benchmark_proxy": "XLK",
            "benchmark_ret_21d": 0.07,
            "benchmark_vol_21d": 0.19,
            "inferred_market_regime": "risk_on",
            "macro_alignment_score": 72.0,
            "risk_on_score": 0.06,
            "stress_overlay": -0.01,
            "meta": {"status": "fresh", "coverage_score": 0.85},
        },
        "geopolitical_policy": {
            "category_counts": {"rates_policy": 1, "regulation_policy": 0},
            "exogenous_event_score": 0.12,
        },
        "relative_context": {
            "sector": "Technology",
            "peer_count": 8,
            "relative_ret_21d": 0.05,
            "relative_momentum": 0.09,
            "relative_strength_percentile": 0.82,
            "peer_dispersion_score": 0.11,
            "meta": {"status": "fresh", "coverage_score": 0.9},
        },
        "quality_provenance": {
            "quality_score": 88,
            "missingness": 0.03,
            "warnings": [],
            "freshness_summary": {
                "bars": {"status": "fresh", "updated_at": "2026-04-05T00:00:00Z"},
                "news": {"status": "fresh", "updated_at": "2026-04-05T00:00:00Z"},
                "sentiment": {"status": "fresh", "updated_at": "2026-04-05T00:00:00Z"},
            },
        },
    }
    feature_bundle = intelligence.build_feature_factor_bundle(
        data_bundle=data_bundle,
        signal={"action": "BUY", "score": 0.8, "confidence": 0.74},
        key_features={
            "ret_1d": 0.01,
            "ret_5d": 0.04,
            "ret_21d": 0.1,
            "mom_vol_adj_21d": 0.75,
            "sentiment_score": 0.32,
            "regime_label": "trend",
        },
        quality={
            "missingness": 0.03,
            "fundamentals_ok": True,
        },
    )
    strategy_bundle = strategy.build_strategy_artifact(
        job_context={
            "horizon": "swing",
            "scenario": "base",
        },
        signal={"action": "BUY", "score": 0.8, "confidence": 0.74},
        data_bundle=data_bundle,
        feature_factor_bundle=feature_bundle,
    )

    assert "multi_horizon_price_momentum" in feature_bundle
    assert "composite_intelligence" in feature_bundle
    assert "Opportunity Quality Score" in feature_bundle["composite_intelligence"]
    assert strategy_bundle["final_signal"] in {"BUY", "HOLD", "SELL"}
    assert strategy_bundle["component_scores"]["trend_following"]["weight"] > 0
    assert strategy_bundle["top_contributors"]


def test_providers_health_is_registered_and_in_openapi() -> None:
    from fastapi.testclient import TestClient
    from api.main import app

    client = TestClient(app)
    r = client.get("/providers/health")
    assert r.status_code == 200
    data = r.json()
    assert "providers" in data
    for k in (
        "openai",
        "massive",
        "finnhub",
        "fred",
        "secedgar",
        "alphavantage",
        "gnews",
        "newsapi",
        "gdelt",
        "world_bank",
        "stooq",
    ):
        assert k in data["providers"]
    openapi = client.get("/openapi.json").json()
    assert "/providers/health" in openapi.get("paths", {})
