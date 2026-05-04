import datetime as dt

from api.assistant import reports as assistant_reports
from api.assistant.phase5.context import build_narrator_context
from api.assistant.phase5.routing import route_question
from api.backtest import service as backtest_service
from api.research.backtest import (
    BACKTEST_VALIDATION_ARTIFACT_KIND,
    build_validation_artifact,
    run_canonical_backtest,
)


def _validation_record(
    *,
    symbol: str,
    as_of_date: str,
    score: float,
    confidence_score: float,
    deployment_permission: str,
    candidate_classification: str,
    suppressed: bool,
    opportunity_quality: float,
    fragility: float,
    market_stress: float,
    net_edge: float,
) -> dict:
    action = "BUY" if score > 0.1 else "SELL" if score < -0.1 else "HOLD"
    outcome = {
        "outcome_status": "matured",
        "matured": True,
        "forward_return": net_edge if action == "BUY" else -net_edge if action == "SELL" else 0.0,
        "absolute_forward_return": abs(net_edge),
        "raw_signal_correct": net_edge > 0,
        "final_signal_correct": net_edge > 0,
        "raw_signal_edge_return": net_edge,
        "final_signal_edge_return": net_edge,
        "gross_edge_return": net_edge + 0.002,
        "net_edge_return": net_edge,
        "gross_trade_return": net_edge + 0.002,
        "net_trade_return": net_edge,
        "estimated_cost_bps": 12.0 if suppressed else 6.0,
        "friction_cost_summary": {
            "total_bps": 12.0 if suppressed else 6.0,
            "gap_risk_bps": 4.0 if suppressed else 1.5,
            "liquidity_bucket": "fragile" if suppressed else "clean",
        },
        "favorable_excursion": max(net_edge, 0.01),
        "adverse_excursion": -0.015 if suppressed else -0.006,
        "mae": 0.015 if suppressed else 0.006,
        "mfe": max(net_edge, 0.01),
        "invalidation_triggered": suppressed,
        "signal_half_life_days": 4 if suppressed else 8,
        "continuation_decay_score": 58.0 if suppressed else 26.0,
    }
    return {
        "symbol": symbol,
        "as_of_date": as_of_date,
        "horizon": "swing",
        "horizon_days": 5,
        "risk_mode": "balanced",
        "signal_action": action,
        "final_signal": action,
        "score": score,
        "confidence_score": confidence_score,
        "confidence": confidence_score / 100.0,
        "conviction_tier": "high" if confidence_score >= 70 else "moderate" if confidence_score >= 50 else "low",
        "strategy_posture": "actionable_long" if action == "BUY" and confidence_score >= 65 else "wait",
        "actionability_score": 72.0 if action == "BUY" else 34.0,
        "fragility_tier": "elevated" if fragility >= 60 else "contained",
        "deployment_permission": deployment_permission,
        "candidate_classification": candidate_classification,
        "proprietary_scores": {
            "Opportunity Quality Score": opportunity_quality,
            "Cross-Domain Conviction Score": confidence_score,
            "Signal Fragility Index": fragility,
            "Macro Alignment Score": 68.0 if not suppressed else 42.0,
        },
        "feature_vector": {
            "implementation_fragility_score": fragility,
            "market_stress_score": market_stress,
            "event_risk_classification": "high_event_risk" if suppressed else "low_event_risk",
            "tradability_state": "implementation_fragile" if suppressed else "clean_liquid_setup",
            "breadth_state": "narrow_leadership" if suppressed else "broad_healthy_participation",
            "cross_asset_conflict_score": 64.0 if suppressed else 28.0,
        },
        "signal_payload": {
            "suppression_flags": ["event_overhang", "market_stress"] if suppressed else [],
        },
        "outcome": outcome,
    }


def test_build_validation_artifact_generates_walkforward_and_research_scorecards():
    records = []
    start = dt.date(2024, 1, 2)
    for index in range(18):
        records.append(
            _validation_record(
                symbol="NVDA" if index % 2 == 0 else "AAPL",
                as_of_date=(start + dt.timedelta(days=index)).isoformat(),
                score=0.32 if index % 3 != 0 else -0.18,
                confidence_score=78.0 if index % 4 != 0 else 42.0,
                deployment_permission="limited_live_eligible" if index % 4 != 0 else "paper_shadow_only",
                candidate_classification="top_priority_candidate" if index % 4 != 0 else "watchlist_candidate",
                suppressed=index % 4 == 0,
                opportunity_quality=82.0 if index % 4 != 0 else 46.0,
                fragility=28.0 if index % 4 != 0 else 66.0,
                market_stress=34.0 if index % 4 != 0 else 72.0,
                net_edge=0.032 if index % 4 != 0 else -0.018,
            )
        )

    artifact = build_validation_artifact(
        records=records,
        cohort_symbol=None,
        cohort_horizon="swing",
        cohort_risk_mode="balanced",
        min_sample_size=4,
        train_window=6,
        validation_window=4,
        step_window=4,
    )

    assert artifact["status"] == "available"
    assert artifact["validation_version"]
    assert artifact["net_return_summary"]["average_edge_return"] is not None
    assert artifact["walkforward_summary"]["status"] == "available"
    assert artifact["walkforward_summary"]["window_count"] >= 1
    assert artifact["readiness_scorecard"]["permission_buckets"]["limited_live_eligible"]["matured_count"] >= 1
    assert artifact["suppression_effect_summary"]["suppressed_setups"]["matured_count"] >= 1
    assert artifact["mae_mfe_summary"]["invalidation_frequency"] is not None
    assert any(
        bucket["score_name"] == "Opportunity Quality Score"
        for bucket in artifact["ranking_scorecard"]["bucket_results"]
    )


def test_run_canonical_backtest_returns_validation_artifact():
    start = dt.date(2024, 1, 2)
    dates = [start + dt.timedelta(days=index) for index in range(14)]
    bars = {
        "AAA": {date: 100.0 + index * 1.5 for index, date in enumerate(dates)},
        "SPY": {date: 400.0 + index * 0.8 for index, date in enumerate(dates)},
    }
    market_states = {
        symbol: {
            date: {
                "open": close * 0.998,
                "high": close * 1.01,
                "low": close * 0.99,
                "close": close,
                "volume": 1_000_000 + index * 50_000,
            }
            for index, (date, close) in enumerate(series.items())
        }
        for symbol, series in bars.items()
        if symbol != "SPY"
    }

    def _resolver(symbol: str, as_of_date: dt.date) -> dict:
        index = (as_of_date - start).days
        action = "BUY" if index % 3 != 0 else "HOLD"
        return {
            "action": action,
            "score": 0.34 if action == "BUY" else 0.02,
            "confidence": 0.71 if action == "BUY" else 0.31,
            "payload": {"signal": action, "confidence": 0.71 if action == "BUY" else 0.31, "suppression_flags": []},
            "feature_payload": {"features": {"signal_regime_label": "TRENDING"}},
            "feature_vector": {
                "signal_regime_label": "TRENDING",
                "implementation_fragility_score": 24.0,
                "market_stress_score": 26.0,
                "event_risk_classification": "low_event_risk",
                "tradability_state": "clean_liquid_setup",
                "breadth_state": "broad_healthy_participation",
                "cross_asset_conflict_score": 22.0,
            },
            "snapshot_id": f"snap-{index}",
            "snapshot_version": "phase9_canonical_snapshot_v1",
            "feature_version": "phase9_canonical_features_v1",
            "signal_version": "phase9_canonical_signal_v1",
        }

    payload = run_canonical_backtest(
        symbols=["AAA"],
        bars=bars,
        market_states=market_states,
        start=dates[0],
        end=dates[-1],
        horizon="short",
        risk_mode="balanced",
        cost_model={"fee_bps": 1, "slippage_bps": 5, "spread_bps": 2, "overnight_gap_risk_bps": 1},
        signal_version_hash="phase9_canonical_signal_v1",
        quality_score_fetcher=lambda _symbol, _as_of: 55,
        signal_resolver=_resolver,
        friction_engine=object(),
    )

    assert payload["metrics"]["cagr"] is not None
    assert payload["validation_artifact"]["validation_version"]
    assert payload["validation_artifact"]["prediction_linkage_summary"]["total_predictions"] >= 1
    assert payload["validation_artifact"]["walkforward_summary"]["status"] in {"available", "insufficient_sample"}


def test_backtest_service_uses_canonical_signal_path_and_persists_validation(monkeypatch):
    start = dt.date(2024, 1, 2)
    dates = [start + dt.timedelta(days=index) for index in range(12)]
    bars = {
        "AAA": {date: 100.0 + index for index, date in enumerate(dates)},
        "SPY": {date: 400.0 + index for index, date in enumerate(dates)},
    }
    market_states = {
        "AAA": {
            date: {"open": close, "high": close * 1.01, "low": close * 0.99, "close": close, "volume": 1_000_000}
            for date, close in bars["AAA"].items()
        }
    }
    persisted = {}
    calls = {"count": 0}

    monkeypatch.setattr(backtest_service, "_require_db_enabled", lambda **_kwargs: None)
    monkeypatch.setattr(backtest_service, "_resolve_symbols", lambda symbol, universe: ["AAA"])
    monkeypatch.setattr(backtest_service, "_fetch_bars", lambda symbols, start, end: bars)
    monkeypatch.setattr(backtest_service, "_fetch_market_states", lambda symbols, start, end: market_states)
    monkeypatch.setattr(backtest_service, "_persist_backtest", lambda **kwargs: persisted.setdefault("run", kwargs))
    monkeypatch.setattr(backtest_service, "_persist_backtest_artifact", lambda **kwargs: persisted.setdefault("artifact", kwargs))

    def _resolver(symbol: str, as_of_date: dt.date, lookback: int = 252):
        calls["count"] += 1
        index = (as_of_date - start).days
        action = "BUY" if index % 2 == 0 else "HOLD"
        return {
            "action": action,
            "score": 0.25 if action == "BUY" else 0.03,
            "confidence": 0.66 if action == "BUY" else 0.28,
            "payload": {"signal": action, "confidence": 0.66 if action == "BUY" else 0.28, "suppression_flags": []},
            "feature_payload": {"features": {"signal_regime_label": "TRENDING"}},
            "feature_vector": {
                "signal_regime_label": "TRENDING",
                "implementation_fragility_score": 22.0,
                "market_stress_score": 24.0,
                "event_risk_classification": "low_event_risk",
                "tradability_state": "clean_liquid_setup",
                "breadth_state": "broad_healthy_participation",
                "cross_asset_conflict_score": 24.0,
            },
            "snapshot_id": f"svc-{index}",
            "snapshot_version": "phase9_canonical_snapshot_v1",
            "feature_version": "phase9_canonical_features_v1",
            "signal_version": "phase9_canonical_signal_v1",
        }

    monkeypatch.setattr(backtest_service, "_compute_canonical_signal_for_date", _resolver)

    result = backtest_service.run_backtest(
        symbol="AAA",
        universe="custom",
        date_start=dates[0].isoformat(),
        date_end=dates[-1].isoformat(),
        horizon="short",
        risk_mode="balanced",
        signal_version_hash="auto",
        cost_model={"fee_bps": 1, "slippage_bps": 5, "spread_bps": 2, "overnight_gap_risk_bps": 1},
    )

    assert result["status"] == "success"
    assert calls["count"] >= 1
    assert persisted["artifact"]["kind"] == BACKTEST_VALIDATION_ARTIFACT_KIND
    assert persisted["artifact"]["payload"]["validation_version"]


def test_report_and_narrator_surface_canonical_validation_sections():
    report = assistant_reports.attach_canonical_validation_context(
        {
            "symbol": "NVDA",
            "overall_analysis": "Core report.",
            "strategy_view": "Strategy section.",
            "risk_quality_analysis": "Risk section.",
            "evidence_provenance": "Evidence section.",
            "evaluation_research_analysis": "Evaluation section.",
            "evidence_map": {},
            "freshness_summary": {"overall_status": "fresh"},
            "signal": {"action": "BUY", "final_action": "BUY"},
            "strategy": {"final_signal": "BUY", "strategy_posture": "actionable_long", "conviction_tier": "high", "actionability_score": 72.0},
        },
        {
            "validation_version": "phase10_research_truth_v1",
            "status": "available",
            "prediction_linkage_summary": {"matured_count": 12},
            "walkforward_summary": {"window_count": 3},
            "net_return_summary": {"average_edge_return": 0.021},
            "friction_cost_summary": {"average_cost_drag": 0.004},
            "validation_summary": "Canonical validation tracks 12 matured decisions.",
            "walkforward_validation_summary": "Walk-forward remains constructive.",
            "net_of_friction_summary": "Net edge remains positive after friction.",
            "suppression_readiness_validation_summary": "Suppression filters improve net results.",
            "drawdown_invalidation_summary": "MAE stays contained and invalidations remain infrequent.",
        },
        canonical_validation_artifact_id="val-1",
    )
    route = route_question("How did walk-forward validation look net of friction and drawdown?")
    narrator_context = build_narrator_context(
        report,
        active_analysis={"symbol": "NVDA", "validation_version": "phase10_research_truth_v1"},
        route=route,
        user_message="How did walk-forward validation look net of friction and drawdown?",
    )

    assert report["canonical_validation_summary"]
    assert report["walkforward_validation_summary"]
    assert route["intent"] == "evaluation_performance"
    assert narrator_context["canonical_validation_snapshot"]["validation_version"] == "phase10_research_truth_v1"
    assert "walkforward_validation_summary" in narrator_context["section_summaries"]

