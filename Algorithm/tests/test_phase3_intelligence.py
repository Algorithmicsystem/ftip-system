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


def test_phase3_feature_factor_bundle_exposes_factor_families_and_scores():
    bundle = intelligence.build_feature_factor_bundle(
        data_bundle=_base_data_bundle(),
        signal={"action": "BUY", "score": 0.82, "confidence": 0.76},
        key_features={"ret_1d": 0.015, "ret_5d": 0.06, "ret_21d": 0.14, "mom_vol_adj_21d": 0.78},
        quality={"missingness": 0.03, "fundamentals_ok": True},
    )

    assert {
        "market_structure",
        "regime_intelligence",
        "fragility_intelligence",
        "sentiment_narrative_intelligence",
        "fundamental_durability",
        "macro_alignment",
        "cross_asset_relative_context",
    }.issubset(bundle["factor_groups"].keys())
    assert {
        "Market Structure Integrity Score",
        "Regime Stability Score",
        "Signal Fragility Index",
        "Narrative Crowding Index",
        "Fundamental Durability Score",
        "Macro Alignment Score",
        "Cross-Domain Conviction Score",
        "Opportunity Quality Score",
    }.issubset(bundle["proprietary_scores"].keys())
    assert bundle["market_structure"]["return_stack"]["252d"] == 0.72
    assert bundle["domain_agreement"]["domain_agreement_score"] is not None
    assert bundle["composite_intelligence"]["Opportunity Quality Score"] is not None


def test_phase3_fragility_and_regime_turn_defensive_when_instability_is_high():
    data_bundle = _base_data_bundle()
    data_bundle["market_price_volume"].update(
        {
            "ret_5d": -0.04,
            "ret_21d": -0.08,
            "ret_63d": -0.16,
            "realized_vol_5d": 0.62,
            "realized_vol_10d": 0.58,
            "realized_vol_21d": 0.54,
            "realized_vol_63d": 0.49,
            "gap_instability_10d": 0.95,
            "vol_of_vol_proxy": 0.92,
            "range_expansion_ratio": 1.8,
        }
    )
    data_bundle["technical_market_structure"].update(
        {
            "trend_slope_21d": -0.03,
            "trend_slope_63d": -0.01,
            "trend_curvature": -0.11,
            "ma_stack_alignment": 0.0,
            "breakout_state": "transition_lower",
            "volume_price_alignment": -0.12,
        }
    )

    bundle = intelligence.build_feature_factor_bundle(
        data_bundle=data_bundle,
        signal={"action": "SELL", "score": -0.74, "confidence": 0.81},
        key_features={"ret_1d": -0.02, "ret_5d": -0.04, "ret_21d": -0.08},
        quality={"missingness": 0.07, "fundamentals_ok": True},
    )

    assert bundle["regime_intelligence"]["regime_label"] in {"high_vol", "transition"}
    assert bundle["composite_intelligence"]["Signal Fragility Index"] >= 60
    assert bundle["fragility_intelligence"]["confidence_degradation_triggers"]


def test_phase3_domain_agreement_flags_cross_domain_conflict_patterns():
    data_bundle = _base_data_bundle()
    data_bundle["fundamental_filing"]["normalized_metrics"].update(
        {
            "revenue_growth_yoy": -0.08,
            "operating_margin": 0.01,
            "net_margin": -0.02,
            "return_on_equity": 0.02,
            "current_ratio": 0.9,
            "cash_ratio": 0.1,
            "debt_to_equity": 2.6,
            "liabilities_to_assets": 0.82,
            "positive_fcf_ratio": 0.2,
            "free_cash_flow_margin": -0.03,
        }
    )
    data_bundle["fundamental_filing"]["quality_proxies"].update(
        {
            "balance_sheet_resilience": 24.0,
            "cash_flow_durability": 22.0,
            "filing_recency_score": 60.0,
            "reporting_quality_proxy": 48.0,
        }
    )

    bundle = intelligence.build_feature_factor_bundle(
        data_bundle=data_bundle,
        signal={"action": "BUY", "score": 0.65, "confidence": 0.72},
        key_features={"ret_1d": 0.01, "ret_5d": 0.06, "ret_21d": 0.14},
        quality={"missingness": 0.05, "fundamentals_ok": True},
    )

    assert "technical strong / fundamentals weak" in bundle["domain_agreement"]["agreement_flags"]
    assert bundle["domain_agreement"]["strongest_conflicting_domains"]
    assert bundle["domain_agreement"]["confidence_penalty_from_conflict"] is not None


def test_phase3_report_surfaces_structural_fragility_and_agreement_layers():
    data_bundle = _base_data_bundle()
    feature_bundle = intelligence.build_feature_factor_bundle(
        data_bundle=data_bundle,
        signal={"action": "BUY", "score": 0.82, "confidence": 0.76},
        key_features={
            "ret_1d": 0.015,
            "ret_5d": 0.06,
            "ret_21d": 0.14,
            "mom_vol_adj_21d": 0.78,
            "regime_label": "trend",
        },
        quality={"missingness": 0.03, "fundamentals_ok": True},
    )
    strategy_bundle = strategy.build_strategy_artifact(
        job_context={"horizon": "swing", "scenario": "base"},
        signal={"action": "BUY", "score": 0.82, "confidence": 0.76},
        data_bundle=data_bundle,
        feature_factor_bundle=feature_bundle,
    )
    report = reports.build_analysis_report(
        symbol="NVDA",
        as_of_date="2026-04-11",
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

    assert "Structural quality is" in report["signal_summary"]
    assert "domain agreement" in report["statistical_analysis"].lower()
    assert "Narrative crowding" in report["sentiment_analysis"]
    assert "Macro fragility" in report["macro_geopolitical_analysis"]
    assert "Fragility markers show instability" in report["risk_quality_analysis"]
    assert report["proprietary_scores"]
    assert report["factor_groups"]
    assert report["domain_agreement"]
    assert report["conviction_components"]
    assert report["opportunity_quality_components"]

