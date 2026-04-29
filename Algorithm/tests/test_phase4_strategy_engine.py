from api.assistant import intelligence, reports, strategy


def _base_data_bundle() -> dict:
    return {
        "symbol_meta": {"symbol": "NVDA", "sector": "Technology"},
        "market_price_volume": {
            "day_return": 0.015,
            "ret_3d": 0.03,
            "ret_5d": 0.06,
            "ret_10d": 0.08,
            "ret_21d": 0.14,
            "ret_63d": 0.32,
            "ret_126d": 0.48,
            "ret_252d": 0.72,
            "realized_vol_5d": 0.22,
            "realized_vol_10d": 0.20,
            "realized_vol_21d": 0.18,
            "realized_vol_63d": 0.16,
            "realized_vol_126d": 0.17,
            "realized_vol_252d": 0.19,
            "atr_pct": 0.03,
            "gap_pct": 0.004,
            "volume_anomaly": 1.35,
            "positive_day_ratio_10d": 0.70,
            "positive_day_ratio_21d": 0.67,
            "positive_day_ratio_63d": 0.63,
            "return_dispersion_10d": 0.015,
            "return_dispersion_21d": 0.017,
            "return_dispersion_63d": 0.018,
            "downside_asymmetry_21d": 1.15,
            "downside_asymmetry_63d": 1.10,
            "maxdd_21d": -0.04,
            "maxdd_63d": -0.08,
            "maxdd_126d": -0.12,
            "gap_instability_10d": 0.32,
            "up_down_volume_ratio_21d": 1.40,
            "vol_of_vol_proxy": 0.18,
            "support_21d": 100.0,
            "resistance_21d": 120.0,
            "compression_ratio": 0.05,
            "range_position_21d": 0.82,
            "range_expansion_ratio": 1.10,
            "breakout_distance_63d": 0.01,
            "meta": {"coverage_score": 0.95, "coverage_status": "available"},
        },
        "technical_market_structure": {
            "trend_slope_21d": 0.12,
            "trend_slope_63d": 0.07,
            "trend_curvature": 0.05,
            "ma_stack_alignment": 1.0,
            "mean_reversion_gap": 0.05,
            "breakout_state": "trend_extension",
            "volume_price_alignment": 0.09,
            "regime_label": "trend",
            "meta": {"coverage_score": 0.90, "coverage_status": "available"},
        },
        "fundamental_filing": {
            "normalized_metrics": {
                "revenue_growth_yoy": 0.22,
                "operating_margin": 0.28,
                "net_margin": 0.24,
                "return_on_equity": 0.31,
                "current_ratio": 2.5,
                "cash_ratio": 1.2,
                "debt_to_equity": 0.28,
                "liabilities_to_assets": 0.38,
                "positive_fcf_ratio": 0.80,
                "free_cash_flow_margin": 0.18,
                "free_cash_flow": 850.0,
            },
            "quality_proxies": {
                "balance_sheet_resilience": 78.0,
                "cash_flow_durability": 81.0,
                "filing_recency_score": 88.0,
                "reporting_quality_proxy": 84.0,
            },
            "durability_proxies": {"growth_quality": 82.0},
            "coverage_score": 0.90,
            "filing_recency_days": 42,
            "meta": {"coverage_score": 0.90, "coverage_status": "available"},
        },
        "sentiment_narrative_flow": {
            "sentiment_score": 0.22,
            "sentiment_trend": 0.02,
            "sentiment_confidence": 74.0,
            "novelty_ratio": 0.68,
            "novelty_score": 66.0,
            "repetition_score": 34.0,
            "persistence_score": 58.0,
            "narrative_concentration": 0.28,
            "attention_crowding": 1.3,
            "attention_score": 63.0,
            "disagreement_score": 0.18,
            "contradiction_score": 0.18,
            "hype_price_divergence": 0.04,
            "event_overlay": {"gdelt_article_count": 2},
            "headline_count": 14,
            "meta": {"coverage_score": 0.80, "coverage_status": "available"},
        },
        "macro_cross_asset": {
            "benchmark_proxy": "XLK",
            "benchmark_ret_21d": 0.07,
            "benchmark_vol_21d": 0.19,
            "inferred_market_regime": "risk_on",
            "macro_alignment_score": 72.0,
            "risk_on_score": 0.06,
            "stress_overlay": -0.01,
            "meta": {"coverage_score": 0.85, "coverage_status": "available"},
        },
        "geopolitical_policy": {
            "category_counts": {"rates_policy": 1, "regulation_policy": 0},
            "exogenous_event_score": 0.12,
            "meta": {"coverage_score": 0.70, "coverage_status": "partial"},
        },
        "relative_context": {
            "sector": "Technology",
            "peer_count": 8,
            "sector_median_ret_21d": 0.09,
            "relative_ret_21d": 0.05,
            "relative_momentum": 0.09,
            "relative_strength_percentile": 0.82,
            "benchmark_proxy": "XLK",
            "benchmark_relative_strength": 0.07,
            "peer_dispersion_score": 0.11,
            "relative_move_summary": {
                "vs_benchmark_ret_21d": 0.07,
                "vs_sector_ret_21d": 0.05,
                "market_relative_note": "The stock is outperforming XLK on a 21-day basis.",
                "sector_relative_note": "The stock is outperforming the local sector comparison set.",
            },
            "meta": {"coverage_score": 0.90, "coverage_status": "available"},
        },
        "quality_provenance": {
            "quality_score": 88.0,
            "missingness": 0.03,
            "anomaly_flags": [],
            "warnings": [],
            "freshness_summary": {
                "bars": {"status": "fresh"},
                "news": {"status": "fresh"},
                "sentiment": {"status": "fresh"},
            },
            "meta": {"coverage_score": 0.95, "coverage_status": "available"},
        },
        "domain_availability": {
            "market": {"coverage_status": "available"},
            "technical": {"coverage_status": "available"},
            "fundamentals": {"coverage_status": "available"},
            "sentiment": {"coverage_status": "available"},
            "macro": {"coverage_status": "available"},
            "geopolitical": {"coverage_status": "partial"},
            "cross_asset": {"coverage_status": "available"},
            "quality": {"coverage_status": "available"},
        },
    }


def _build_feature_bundle(data_bundle: dict, *, action: str = "BUY", score: float = 0.82, confidence: float = 0.76) -> dict:
    return intelligence.build_feature_factor_bundle(
        data_bundle=data_bundle,
        signal={"action": action, "score": score, "confidence": confidence},
        key_features={
            "ret_1d": data_bundle["market_price_volume"]["day_return"],
            "ret_5d": data_bundle["market_price_volume"]["ret_5d"],
            "ret_21d": data_bundle["market_price_volume"]["ret_21d"],
            "mom_vol_adj_21d": 0.78,
            "regime_label": data_bundle["technical_market_structure"].get("regime_label"),
        },
        quality={"missingness": data_bundle["quality_provenance"].get("missingness"), "fundamentals_ok": True},
    )


def test_phase4_strategy_artifact_exposes_components_scenarios_and_execution_fields():
    data_bundle = _base_data_bundle()
    feature_bundle = _build_feature_bundle(data_bundle)

    strategy_bundle = strategy.build_strategy_artifact(
        job_context={"horizon": "swing", "scenario": "base"},
        signal={"action": "BUY", "score": 0.82, "confidence": 0.76},
        data_bundle=data_bundle,
        feature_factor_bundle=feature_bundle,
    )

    assert strategy_bundle["final_signal"] in {"BUY", "HOLD", "SELL"}
    assert strategy_bundle["strategy_posture"]
    assert strategy_bundle["strategy_summary"]
    assert len(strategy_bundle["strategy_components"]) == 8
    assert set(strategy_bundle["scenario_matrix"].keys()) == {"base", "bull", "bear", "stress"}
    assert strategy_bundle["scenario_transitions"]["base_to_bull"]
    assert strategy_bundle["confidence_score"] is not None
    assert strategy_bundle["actionability_score"] is not None
    assert strategy_bundle["execution_posture"]["preferred_posture"] in {
        "immediate_action",
        "staged_watch",
        "wait_for_confirmation",
        "avoid_due_to_fragility",
    }
    assert strategy_bundle["invalidators"]["top_invalidators"]
    assert strategy_bundle["strategy_version"] == "phase4_institutional_v1"


def test_phase4_confidence_degrades_under_staleness_missingness_and_conflict():
    data_bundle = _base_data_bundle()
    data_bundle["quality_provenance"]["missingness"] = 0.18
    data_bundle["quality_provenance"]["freshness_summary"] = {
        "bars": {"status": "stale_but_usable"},
        "news": {"status": "stale"},
        "sentiment": {"status": "stale_but_usable"},
    }
    data_bundle["domain_availability"]["fundamentals"]["coverage_status"] = "partial"
    data_bundle["domain_availability"]["cross_asset"]["coverage_status"] = "limited"
    data_bundle["fundamental_filing"]["normalized_metrics"]["revenue_growth_yoy"] = -0.06
    data_bundle["fundamental_filing"]["normalized_metrics"]["operating_margin"] = 0.02
    data_bundle["fundamental_filing"]["quality_proxies"]["balance_sheet_resilience"] = 32.0
    data_bundle["fundamental_filing"]["quality_proxies"]["cash_flow_durability"] = 28.0

    feature_bundle = _build_feature_bundle(data_bundle)
    strategy_bundle = strategy.build_strategy_artifact(
        job_context={"horizon": "swing", "scenario": "base"},
        signal={"action": "BUY", "score": 0.66, "confidence": 0.72},
        data_bundle=data_bundle,
        feature_factor_bundle=feature_bundle,
    )

    assert strategy_bundle["confidence_score"] < 55
    assert strategy_bundle["confidence_quality"] in {"fragile", "low_evidence", "adequate"}
    assert strategy_bundle["calibration_status"] == "degraded_provisional"
    assert strategy_bundle["confidence_degraders"]
    assert strategy_bundle["uncertainty_notes"]


def test_phase4_fragility_veto_converts_constructive_setup_to_hold_when_instability_is_extreme():
    data_bundle = _base_data_bundle()
    data_bundle["market_price_volume"].update(
        {
            "ret_5d": 0.09,
            "ret_21d": 0.16,
            "realized_vol_5d": 0.68,
            "realized_vol_10d": 0.62,
            "realized_vol_21d": 0.58,
            "realized_vol_63d": 0.51,
            "gap_instability_10d": 0.96,
            "vol_of_vol_proxy": 0.95,
            "return_dispersion_10d": 0.07,
            "return_dispersion_21d": 0.06,
            "return_dispersion_63d": 0.05,
        }
    )
    data_bundle["technical_market_structure"].update(
        {
            "trend_slope_21d": 0.04,
            "trend_slope_63d": 0.01,
            "trend_curvature": 0.11,
            "regime_label": "transition",
            "volume_price_alignment": -0.08,
        }
    )
    data_bundle["quality_provenance"]["missingness"] = 0.16

    feature_bundle = _build_feature_bundle(data_bundle)
    strategy_bundle = strategy.build_strategy_artifact(
        job_context={"horizon": "swing", "scenario": "base"},
        signal={"action": "BUY", "score": 0.84, "confidence": 0.78},
        data_bundle=data_bundle,
        feature_factor_bundle=feature_bundle,
    )

    assert strategy_bundle["hard_veto"] is True
    assert strategy_bundle["final_signal"] == "HOLD"
    assert strategy_bundle["strategy_posture"] in {"fragile_hold", "no_trade"}
    assert strategy_bundle["actionability_score"] < 40
    assert strategy_bundle["fragility_vetoes"]


def test_phase4_participant_fit_can_shift_to_mean_reversion_in_chop():
    data_bundle = _base_data_bundle()
    data_bundle["market_price_volume"].update(
        {
            "ret_3d": -0.03,
            "ret_5d": -0.05,
            "ret_10d": -0.01,
            "ret_21d": 0.02,
            "ret_63d": 0.03,
            "compression_ratio": 0.03,
            "range_position_21d": 0.18,
            "range_expansion_ratio": 0.82,
        }
    )
    data_bundle["technical_market_structure"].update(
        {
            "trend_slope_21d": 0.01,
            "trend_slope_63d": 0.0,
            "trend_curvature": 0.02,
            "mean_reversion_gap": -0.04,
            "regime_label": "chop",
        }
    )

    feature_bundle = _build_feature_bundle(data_bundle, action="HOLD", score=0.12, confidence=0.58)
    strategy_bundle = strategy.build_strategy_artifact(
        job_context={"horizon": "swing", "scenario": "base"},
        signal={"action": "HOLD", "score": 0.12, "confidence": 0.58},
        data_bundle=data_bundle,
        feature_factor_bundle=feature_bundle,
    )

    assert "mean-reversion participant" in strategy_bundle["participant_fit"]
    assert strategy_bundle["time_horizon_fit"].startswith("swing_")


def test_phase4_report_surfaces_strategy_layer_language():
    data_bundle = _base_data_bundle()
    feature_bundle = _build_feature_bundle(data_bundle)
    strategy_bundle = strategy.build_strategy_artifact(
        job_context={"horizon": "swing", "scenario": "base", "analysis_depth": "deep"},
        signal={"action": "BUY", "score": 0.82, "confidence": 0.76},
        data_bundle=data_bundle,
        feature_factor_bundle=feature_bundle,
    )

    report = reports.build_analysis_report(
        symbol="NVDA",
        as_of_date="2026-04-29",
        horizon="swing",
        risk_mode="balanced",
        signal={"action": "BUY", "score": 0.82, "confidence": 0.76},
        key_features={"mom_vol_adj_21d": 0.78, "regime_label": "trend"},
        quality={"missingness": 0.03, "warnings": [], "anomaly_flags": []},
        evidence={"sources": ["signals_daily", "news_raw", "sec_edgar"]},
        job_context={"scenario": "base", "analysis_depth": "deep", "refresh_mode": "refresh_stale", "market_regime": "auto"},
        data_bundle=data_bundle,
        feature_factor_bundle=feature_bundle,
        strategy=strategy_bundle,
    )

    assert "actionability" in report["signal_summary"].lower()
    assert "bull case" in report["strategy_view"].lower()
    assert "stress case" in report["strategy_view"].lower()
    assert "execution posture" in report["strategy_view"].lower()
    assert "formal invalidators" in report["risks_weaknesses_invalidators"].lower()
    assert report["scenario_matrix"]["base"]["summary"]
    assert report["execution_posture"]["preferred_posture"]
