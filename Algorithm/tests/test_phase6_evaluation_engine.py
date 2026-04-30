import datetime as dt

from api.assistant.phase6 import build_evaluation_artifact, build_prediction_record
from api.assistant.phase6.linkage import link_realized_outcome


def _sample_report(
    *,
    symbol: str,
    as_of_date: str,
    final_signal: str,
    signal_action: str,
    confidence_score: float,
    conviction_tier: str,
    strategy_posture: str,
    actionability_score: float,
    opportunity_quality: float,
    conviction_score: float,
    fragility_score: float,
    macro_score: float,
    crowding_score: float,
    regime_label: str,
    fragility_tier: str,
) -> dict:
    return {
        "generated_at": f"{as_of_date}T00:00:00Z",
        "report_version": "2.1",
        "strategy_version": "phase4_institutional_v1",
        "symbol": symbol,
        "as_of_date": as_of_date,
        "horizon": "swing",
        "risk_mode": "balanced",
        "scenario": "base",
        "analysis_depth": "standard",
        "signal": {
            "action": signal_action,
            "final_action": final_signal,
            "score": 0.62,
            "confidence": 0.58,
            "horizon_days": 5,
        },
        "strategy": {
            "final_signal": final_signal,
            "confidence": confidence_score / 100.0,
            "confidence_score": confidence_score,
            "conviction_tier": conviction_tier,
            "confidence_quality": "adequate",
            "strategy_posture": strategy_posture,
            "actionability_score": actionability_score,
            "participant_fit": ["swing trader"],
            "primary_participant_fit": "swing trader",
            "fragility_tier": fragility_tier,
            "component_scores": {
                "trend_following": {
                    "score": 0.36 if final_signal == "BUY" else -0.24 if final_signal == "SELL" else 0.04,
                    "normalized_score": 62.0,
                    "weight": 0.2,
                },
                "macro_alignment": {
                    "score": 0.18 if macro_score >= 60 else -0.15,
                    "normalized_score": macro_score,
                    "weight": 0.12,
                },
                "fragility_risk_veto": {
                    "score": -0.22 if fragility_score >= 60 else 0.08,
                    "normalized_score": 100 - fragility_score,
                    "weight": 0.14,
                },
            },
        },
        "proprietary_scores": {
            "Opportunity Quality Score": {"score": opportunity_quality},
            "Cross-Domain Conviction Score": {"score": conviction_score},
            "Signal Fragility Index": {"score": fragility_score},
            "Macro Alignment Score": {"score": macro_score},
            "Narrative Crowding Index": {"score": crowding_score},
            "Regime Stability Score": {"score": 70.0 if regime_label == "trend" else 36.0},
        },
        "regime_intelligence": {"regime_label": regime_label},
        "domain_agreement": {"domain_agreement_score": conviction_score, "domain_conflict_score": 100 - conviction_score},
        "freshness_summary": {"overall_status": "fresh"},
        "quality": {"quality_score": 88.0, "missingness": 0.02},
        "domain_availability": {"fundamentals": {"coverage_status": "available"}},
        "invalidators": {"top_invalidators": ["macro flip", "regime instability"]},
        "confirmation_triggers": ["stronger price confirmation"],
        "deterioration_triggers": ["relative weakness"],
        "fragility_vetoes": [{"name": "narrative_crowding", "reason": "crowding elevated"}],
        "participant_fit": ["swing trader"],
    }


def _bar_fetcher_factory(returns_map: dict[str, tuple[float, int]]):
    def _fetch(symbol: str, as_of_date: dt.date, limit: int) -> list[dict]:
        forward_return, total_rows = returns_map[symbol]
        target_rows = min(limit, total_rows)
        if target_rows <= 0:
            return []
        prices = []
        for index in range(target_rows):
            progress = index / max(total_rows - 1, 1)
            close = 100.0 * (1.0 + forward_return * progress)
            prices.append(
                {
                    "as_of_date": (as_of_date + dt.timedelta(days=index)).isoformat(),
                    "close": close,
                }
            )
        return prices

    return _fetch


def test_phase6_prediction_record_links_to_realized_outcome_point_in_time() -> None:
    report = _sample_report(
        symbol="ALP",
        as_of_date="2024-01-02",
        final_signal="BUY",
        signal_action="BUY",
        confidence_score=78.0,
        conviction_tier="high",
        strategy_posture="actionable_long",
        actionability_score=72.0,
        opportunity_quality=82.0,
        conviction_score=76.0,
        fragility_score=28.0,
        macro_score=70.0,
        crowding_score=34.0,
        regime_label="trend",
        fragility_tier="contained",
    )
    prediction = build_prediction_record(report, report_id="report-1", session_id="session-1")
    outcome = link_realized_outcome(
        prediction,
        bar_fetcher=_bar_fetcher_factory({"ALP": (0.1, 6)}),
        evaluation_as_of_date=dt.date(2024, 1, 10),
    )

    assert prediction["report_id"] == "report-1"
    assert prediction["strategy_posture"] == "actionable_long"
    assert outcome["matured"] is True
    assert outcome["forward_return"] > 0
    assert outcome["final_signal_correct"] is True
    assert outcome["final_signal_edge_return"] > 0
    assert outcome["entry_date"] == "2024-01-02"
    assert outcome["exit_date"] == "2024-01-07"


def test_phase6_builds_scorecards_calibration_regime_and_attribution_views() -> None:
    reports = [
        _sample_report(
            symbol="ALP",
            as_of_date="2024-01-02",
            final_signal="BUY",
            signal_action="BUY",
            confidence_score=86.0,
            conviction_tier="very_high",
            strategy_posture="actionable_long",
            actionability_score=78.0,
            opportunity_quality=86.0,
            conviction_score=82.0,
            fragility_score=22.0,
            macro_score=74.0,
            crowding_score=28.0,
            regime_label="trend",
            fragility_tier="contained",
        ),
        _sample_report(
            symbol="BET",
            as_of_date="2024-01-02",
            final_signal="BUY",
            signal_action="BUY",
            confidence_score=80.0,
            conviction_tier="high",
            strategy_posture="actionable_long",
            actionability_score=70.0,
            opportunity_quality=80.0,
            conviction_score=76.0,
            fragility_score=26.0,
            macro_score=68.0,
            crowding_score=32.0,
            regime_label="trend",
            fragility_tier="contained",
        ),
        _sample_report(
            symbol="CRN",
            as_of_date="2024-01-02",
            final_signal="HOLD",
            signal_action="BUY",
            confidence_score=62.0,
            conviction_tier="moderate",
            strategy_posture="watchlist_positive",
            actionability_score=46.0,
            opportunity_quality=62.0,
            conviction_score=58.0,
            fragility_score=38.0,
            macro_score=58.0,
            crowding_score=45.0,
            regime_label="trend",
            fragility_tier="contained",
        ),
        _sample_report(
            symbol="DYN",
            as_of_date="2024-01-02",
            final_signal="HOLD",
            signal_action="HOLD",
            confidence_score=48.0,
            conviction_tier="low",
            strategy_posture="wait",
            actionability_score=32.0,
            opportunity_quality=48.0,
            conviction_score=46.0,
            fragility_score=48.0,
            macro_score=50.0,
            crowding_score=52.0,
            regime_label="choppy",
            fragility_tier="mixed",
        ),
        _sample_report(
            symbol="ELM",
            as_of_date="2024-01-02",
            final_signal="SELL",
            signal_action="SELL",
            confidence_score=72.0,
            conviction_tier="high",
            strategy_posture="actionable_short",
            actionability_score=68.0,
            opportunity_quality=74.0,
            conviction_score=70.0,
            fragility_score=42.0,
            macro_score=38.0,
            crowding_score=58.0,
            regime_label="transition",
            fragility_tier="elevated",
        ),
        _sample_report(
            symbol="FOX",
            as_of_date="2024-01-02",
            final_signal="SELL",
            signal_action="SELL",
            confidence_score=76.0,
            conviction_tier="high",
            strategy_posture="actionable_short",
            actionability_score=72.0,
            opportunity_quality=78.0,
            conviction_score=74.0,
            fragility_score=54.0,
            macro_score=34.0,
            crowding_score=62.0,
            regime_label="transition",
            fragility_tier="elevated",
        ),
        _sample_report(
            symbol="GLD",
            as_of_date="2024-01-02",
            final_signal="BUY",
            signal_action="BUY",
            confidence_score=34.0,
            conviction_tier="low",
            strategy_posture="fragile_hold",
            actionability_score=26.0,
            opportunity_quality=34.0,
            conviction_score=32.0,
            fragility_score=72.0,
            macro_score=36.0,
            crowding_score=74.0,
            regime_label="transition",
            fragility_tier="fragile",
        ),
        _sample_report(
            symbol="HZN",
            as_of_date="2024-01-02",
            final_signal="BUY",
            signal_action="BUY",
            confidence_score=28.0,
            conviction_tier="very_low",
            strategy_posture="no_trade",
            actionability_score=18.0,
            opportunity_quality=26.0,
            conviction_score=24.0,
            fragility_score=78.0,
            macro_score=28.0,
            crowding_score=82.0,
            regime_label="transition",
            fragility_tier="fragile",
        ),
        _sample_report(
            symbol="ION",
            as_of_date="2024-01-02",
            final_signal="BUY",
            signal_action="BUY",
            confidence_score=52.0,
            conviction_tier="moderate",
            strategy_posture="watchlist_positive",
            actionability_score=40.0,
            opportunity_quality=54.0,
            conviction_score=50.0,
            fragility_score=46.0,
            macro_score=52.0,
            crowding_score=48.0,
            regime_label="trend",
            fragility_tier="mixed",
        ),
    ]
    prediction_records = [
        build_prediction_record(report, report_id=f"report-{index}", session_id="session-1")
        for index, report in enumerate(reports, start=1)
    ]
    bar_fetcher = _bar_fetcher_factory(
        {
            "ALP": (0.12, 6),
            "BET": (0.08, 6),
            "CRN": (0.012, 6),
            "DYN": (-0.004, 6),
            "ELM": (-0.07, 6),
            "FOX": (-0.11, 6),
            "GLD": (-0.06, 6),
            "HZN": (0.03, 6),
            "ION": (0.018, 3),
        }
    )

    evaluation = build_evaluation_artifact(
        prediction_records=prediction_records,
        bar_fetcher=bar_fetcher,
        cohort_horizon="swing",
        cohort_risk_mode="balanced",
    )

    assert evaluation["status"] == "available"
    assert evaluation["prediction_linkage_summary"]["linked_outcome_status"]["matured"] == 8
    assert evaluation["prediction_linkage_summary"]["linked_outcome_status"]["pending"] == 1
    assert evaluation["signal_scorecard"]["final_signal_overall"]["matured_count"] == 8
    assert evaluation["strategy_scorecard"]["actionable_setups"]["matured_count"] >= 4
    assert evaluation["ranking_scorecard"]["available_bucket_count"] >= 1
    assert evaluation["calibration_summary"]["confidence_reliability_score"] is not None
    assert evaluation["regime_breakdown"]["regime_label"]
    assert evaluation["factor_attribution_summary"]["proprietary_score_attribution"]
    assert evaluation["evaluation_summary"]
    assert evaluation["confidence_reliability_summary"]
    assert evaluation["regime_usefulness_summary"]

