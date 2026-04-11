import datetime as dt

from api.assistant import coverage, intelligence, reports


def _domain_availability(**overrides):
    base = {
        "market": {
            "coverage_status": "partial",
            "available_horizons": ["1d", "5d", "10d", "21d", "63d"],
            "missing_horizons": ["126d", "252d"],
            "missing_reason": "insufficient_history",
            "fallback_used": True,
            "fallback_source": ["features_daily"],
            "data_quality_note": "Return context is populated through 63d; longer-horizon metrics remain constrained by usable history.",
        },
        "technical": {
            "coverage_status": "partial",
            "available_horizons": ["10d", "21d", "63d"],
            "missing_horizons": ["126d"],
            "missing_reason": "insufficient_history",
            "fallback_used": False,
            "fallback_source": [],
            "data_quality_note": "Technical structure is populated through 63d; longer moving-average and trend layers remain history-constrained.",
        },
        "fundamentals": {
            "coverage_status": "partial",
            "available_horizons": [],
            "missing_horizons": [],
            "missing_reason": None,
            "fallback_used": True,
            "fallback_source": ["finnhub_basic_financials"],
            "data_quality_note": "Quarterly filing coverage is partial, so durability and cash-flow reads should be treated with reduced confidence.",
        },
        "sentiment": {
            "coverage_status": "partial",
            "available_horizons": [],
            "missing_horizons": [],
            "missing_reason": None,
            "fallback_used": False,
            "fallback_source": [],
            "data_quality_note": "Headline flow is available, but the density is still too thin for a stable sentiment-level inference.",
        },
        "macro": {
            "coverage_status": "partial",
            "available_horizons": [],
            "missing_horizons": [],
            "missing_reason": None,
            "fallback_used": True,
            "fallback_source": ["SPY"],
            "data_quality_note": "Sector-specific benchmark coverage is thin, so the broader-market fallback SPY is being used.",
        },
        "geopolitical": {
            "coverage_status": "not_relevant",
            "available_horizons": [],
            "missing_horizons": [],
            "missing_reason": "not_relevant",
            "fallback_used": False,
            "fallback_source": [],
            "data_quality_note": "Recent headline flow does not currently point to a material policy, conflict, or macro shock cluster.",
        },
        "cross_asset": {
            "coverage_status": "partial",
            "available_horizons": [],
            "missing_horizons": [],
            "missing_reason": None,
            "fallback_used": False,
            "fallback_source": [],
            "data_quality_note": "Peer-relative context is partial because the comparison set is still shallow.",
        },
        "quality": {
            "coverage_status": "available",
            "available_horizons": [],
            "missing_horizons": [],
            "missing_reason": None,
            "fallback_used": False,
            "fallback_source": [],
            "data_quality_note": "Coverage and freshness checks are current across the tracked operational domains.",
        },
    }
    base.update(overrides)
    return base


def _build_partial_report() -> dict:
    availability = _domain_availability()
    return reports.build_analysis_report(
        symbol="NVDA",
        as_of_date="2024-01-05",
        horizon="swing",
        risk_mode="balanced",
        signal={
            "action": "HOLD",
            "score": 0.2,
            "confidence": 0.58,
            "entry_low": 100,
            "entry_high": 104,
            "stop_loss": 95,
            "take_profit_1": 112,
            "take_profit_2": 118,
            "reason_codes": ["MIXED"],
        },
        key_features={
            "ret_1d": 0.01,
            "ret_5d": 0.03,
            "ret_21d": 0.08,
            "mom_vol_adj_21d": 0.22,
        },
        quality={
            "bars_ok": True,
            "fundamentals_ok": True,
            "sentiment_ok": True,
            "news_ok": True,
            "intraday_ok": False,
            "missingness": 0.11,
            "warnings": ["Cross-asset proxy coverage is partial."],
        },
        evidence={"sources": ["market_bars_daily", "news_raw"]},
        data_bundle={
            "market_price_volume": {
                "day_return": 0.01,
                "ret_5d": 0.03,
                "ret_21d": 0.08,
                "ret_63d": 0.14,
                "realized_vol_21d": 0.24,
                "atr_pct": 0.04,
                "gap_pct": 0.01,
                "support_21d": 100.0,
                "resistance_21d": 110.0,
                "support_window_days": 18,
                "volume_anomaly": 1.3,
                "compression_ratio": 0.06,
            },
            "technical_market_structure": {
                "moving_averages": {"ma_10": 105.0, "ma_21": 102.0, "ma_63": 98.0},
                "trend_slope_21d": 0.12,
                "trend_slope_63d": 0.07,
                "trend_curvature": 0.05,
            },
            "fundamental_filing": {
                "normalized_metrics": {
                    "revenue_growth_yoy": 0.14,
                    "operating_margin": 0.23,
                    "current_ratio": 2.1,
                },
                "quality_proxies": {
                    "reporting_quality_proxy": 81.0,
                    "business_quality_durability": 76.0,
                },
                "coverage_score": 0.68,
                "statement_snapshot": {
                    "latest_quarter": {"revenue": 1200.0},
                },
                "strength_summary": ["Revenue growth remains positive."],
                "weakness_summary": ["Cash-flow coverage is still partial."],
                "coverage_caveats": ["Balance-sheet detail is not fully populated."],
                "meta": {
                    "sources": ["sec_edgar", "finnhub_basic_financials"],
                    "status": "fresh",
                    "coverage_status": "partial",
                },
            },
            "sentiment_narrative_flow": {
                "headline_count": 4,
                "aggregated_sentiment_bias": 0.12,
                "top_narratives": [{"topic": "demand", "count": 2}],
                "source_breakdown": {"gnews": 2, "newsapi": 2},
                "meta": {"sources": ["gnews", "newsapi"]},
            },
            "macro_cross_asset": {
                "macro_alignment_score": 61.0,
                "meta": {"coverage_status": "partial"},
            },
            "geopolitical_policy": {
                "event_buckets": {},
                "meta": {"coverage_status": "not_relevant"},
            },
            "relative_context": {
                "benchmark_context": {
                    "benchmark_symbol": "SPY",
                    "benchmark_ret_21d": 0.04,
                    "benchmark_vol_21d": 0.18,
                },
                "sector": "Technology",
                "peer_count": 3,
                "meta": {"coverage_status": "partial"},
            },
            "quality_provenance": {
                "freshness_summary": {},
                "source_map": {"sentiment_narrative_flow": ["gnews", "newsapi"]},
                "domain_availability": availability,
            },
            "domain_availability": availability,
            "external_data_fabric": {"status": "ok"},
        },
        feature_factor_bundle={
            "composite_intelligence": {
                "Market Structure Integrity Score": 58.0,
                "Regime Stability Score": 62.0,
                "Cross-Domain Conviction Score": 57.0,
                "Signal Fragility Index": 48.0,
            }
        },
        strategy={
            "final_signal": "HOLD",
            "confidence": 0.56,
            "conviction_tier": "moderate",
            "fragility_tier": "medium",
            "top_contributors": [{"label": "trend", "detail": "short-horizon price remains constructive"}],
            "top_detractors": [{"label": "coverage", "detail": "longer-horizon structure is incomplete"}],
            "component_scores": {"trend_following": {"weight": 0.3}},
            "base_case": "The base case is to stay selective while the shorter-horizon trend remains constructive.",
            "invalidation_conditions": ["Break below the recent support zone."],
        },
    )


def test_classify_horizon_coverage_marks_partial_for_short_history():
    result = coverage.classify_horizon_coverage(70)

    assert result["coverage_status"] == "partial"
    assert result["available_horizons"] == ["1d", "5d", "10d", "21d", "63d"]
    assert result["missing_horizons"] == ["126d", "252d"]


def test_market_domain_tracks_fallback_sources_when_history_is_short(monkeypatch):
    as_of_date = dt.date(2024, 1, 5)
    bars = [
        {
            "as_of_date": (as_of_date - dt.timedelta(days=14 - idx)).isoformat(),
            "open": 100.0 + idx,
            "high": 101.0 + idx,
            "low": 99.0 + idx,
            "close": 100.0 + idx,
            "volume": 1_000_000 + idx * 10_000,
            "source": "market_bars_daily",
            "ingested_at": "2024-01-05T00:00:00Z",
        }
        for idx in range(15)
    ]
    monkeypatch.setattr(intelligence, "_load_daily_bars", lambda *_args, **_kwargs: bars)
    monkeypatch.setattr(intelligence, "_load_intraday_bars", lambda *_args, **_kwargs: [])

    domain, _, _ = intelligence._market_domain(
        "NVDA",
        as_of_date,
        freshness={"bars_updated_at": "2024-01-05T00:00:00Z"},
        key_features={"ret_21d": 0.07, "vol_21d": 0.25},
        quality={"bars_ok": True},
    )

    assert domain["ret_21d"] == 0.07
    assert domain["meta"]["coverage_status"] == "partial"
    assert domain["meta"]["fallback_used"] is True
    assert "features_daily" in domain["meta"]["fallback_source"]


def test_report_replaces_na_leakage_with_coverage_language():
    report = _build_partial_report()

    assert "n/a" not in report["technical_analysis"]
    assert "limited usable history" in report["technical_analysis"]
    assert "stable sentiment-level inference" in report["sentiment_analysis"]


def test_report_uses_benchmark_fallback_and_explains_non_relevant_geo():
    report = _build_partial_report()

    assert "SPY" in report["macro_geopolitical_analysis"]
    assert "not currently a primary driver" in report["macro_geopolitical_analysis"]
