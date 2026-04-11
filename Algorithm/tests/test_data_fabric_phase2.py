import datetime as dt

from api.assistant import data_fabric, reports
from api.data_providers.errors import ProviderUnavailable


def _bar_rows(symbol: str, end: dt.date, *, source: str = "stooq", drift: float = 1.0):
    rows = []
    for idx in range(90):
        rows.append(
            {
                "as_of_date": end - dt.timedelta(days=89 - idx),
                "close": 100.0 + drift * idx,
                "source": source,
            }
        )
    return rows


def test_build_news_overlay_expands_multi_source_narrative(monkeypatch):
    as_of_date = dt.date(2024, 1, 5)
    monkeypatch.setattr(
        data_fabric,
        "search_gnews",
        lambda *_args, **_kwargs: [
            {
                "published_at": dt.datetime(2024, 1, 5, 13, tzinfo=dt.timezone.utc),
                "source": "gnews",
                "title": "NVIDIA demand surges as AI orders expand",
                "url": "https://example.com/story-1",
                "content_snippet": "gnews",
            }
        ],
    )
    monkeypatch.setattr(
        data_fabric,
        "search_newsapi",
        lambda *_args, **_kwargs: [
            {
                "published_at": dt.datetime(2024, 1, 5, 14, tzinfo=dt.timezone.utc),
                "source": "newsapi",
                "title": "NVIDIA faces export regulation review",
                "url": "https://example.com/story-2",
                "content_snippet": "newsapi",
            }
        ],
    )
    monkeypatch.setattr(
        data_fabric,
        "fetch_company_news",
        lambda *_args, **_kwargs: [
            {
                "published_at": dt.datetime(2024, 1, 5, 13, tzinfo=dt.timezone.utc),
                "source": "finnhub_news",
                "title": "NVIDIA demand surges as AI orders expand",
                "url": "https://example.com/story-1",
                "content_snippet": "finnhub",
            }
        ],
    )
    monkeypatch.setattr(
        data_fabric,
        "search_gdelt_articles",
        lambda *_args, **_kwargs: [
            {
                "published_at": dt.datetime(2024, 1, 5, 15, tzinfo=dt.timezone.utc),
                "source": "gdelt",
                "title": "NVIDIA export policy pressures intensify",
                "url": "https://example.com/story-3",
                "content_snippet": "gdelt",
                "tone": -1.5,
            }
        ],
    )

    overlay = data_fabric._build_news_overlay(
        "NVDA",
        {"name": "NVIDIA", "sector": "Technology"},
        as_of_date,
    )

    assert overlay["headline_count"] == 3
    assert overlay["source_count"] == 4
    assert overlay["sentiment_summary"]["tone_label"] in {"constructive", "mixed", "negative"}
    assert overlay["sentiment_confidence"] is not None
    assert overlay["topic_clusters"]
    assert overlay["topic_buckets"]["policy_regulation"] >= 1
    assert overlay["event_overlay"]["gdelt_article_count"] == 1
    assert overlay["provenance"]["sources_used"] == ["finnhub_news", "gdelt", "gnews", "newsapi"]
    assert overlay["freshness"]["data_as_of"]


def test_build_macro_overlay_normalizes_contexts(monkeypatch):
    fred_map = {
        "DGS10": [{"date": "2024-01-01", "value": 4.3}, {"date": "2023-12-01", "value": 4.0}],
        "FEDFUNDS": [{"date": "2024-01-01", "value": 5.3}, {"date": "2023-12-01", "value": 5.2}],
        "CPIAUCSL": [{"date": "2024-01-01", "value": 311.0}, {"date": "2023-12-01", "value": 309.5}],
        "UNRATE": [{"date": "2024-01-01", "value": 3.9}, {"date": "2023-12-01", "value": 3.8}],
        "GDPC1": [{"date": "2024-01-01", "value": 2.8}, {"date": "2023-12-01", "value": 2.6}],
        "BAMLC0A0CM": [{"date": "2024-01-01", "value": 3.9}, {"date": "2023-12-01", "value": 3.7}],
    }
    monkeypatch.setattr(
        data_fabric,
        "fetch_fred_series",
        lambda series_id, limit=12: {"observations": fred_map[series_id]},
    )
    monkeypatch.setattr(
        data_fabric,
        "fetch_world_bank_indicator",
        lambda **_kwargs: {
            "observations": [
                {"date": "2023", "value": 2.7},
                {"date": "2022", "value": 2.2},
            ]
        },
    )

    overlay = data_fabric._build_macro_overlay(
        {"country": "US", "sector": "Technology"},
        {
            "benchmark_proxy": "QQQ",
            "rates_fx_commodities_context": {"rates_proxy": {"ret_21d": -0.04}},
        },
    )

    assert overlay["macro_regime_summary"]["regime"] in {
        "tightening_inflationary",
        "growth_supportive",
        "growth_softening",
        "neutral",
    }
    assert overlay["rates_context"]["direction"] == "rising"
    assert overlay["liquidity_context"]["regime"] in {"restrictive", "supportive", "neutral"}
    assert overlay["macro_alignment_notes"]
    assert overlay["meta"]["confidence"] is not None


def test_build_cross_asset_overlay_uses_sector_fallback_and_structures_context(monkeypatch):
    as_of_date = dt.date(2024, 1, 5)

    def _fetch(proxy, _start, end):
        if proxy == "XLK":
            raise ProviderUnavailable("PROVIDER_UNAVAILABLE", "sector ETF unavailable")
        return _bar_rows(proxy, end, source="stooq", drift=1.2 if proxy in {"SPY", "QQQ"} else 0.6)

    monkeypatch.setattr(data_fabric, "fetch_reference_bars", _fetch)

    overlay = data_fabric._build_cross_asset_overlay(
        {"sector": "Technology"},
        as_of_date,
        base_market={"ret_21d": 0.12, "ret_63d": 0.24},
    )

    assert overlay["benchmark_proxy"] == "SPY"
    assert overlay["sector_context"]["coverage_note"].startswith("Sector-specific proxy coverage")
    assert overlay["broad_market_context"]["major_benchmarks"]["QQQ"]["ret_21d"] is not None
    assert overlay["rates_fx_commodities_context"]["rates_proxy"]["symbol"] == "TLT"
    assert overlay["meta"]["fallback_used"] is True
    assert overlay["relative_move_summary"]["market_relative_note"]


def test_build_geopolitical_overlay_scores_event_relevance():
    overlay = data_fabric._build_geopolitical_overlay(
        {
            "aggregated_headlines": [
                {
                    "title": "NVIDIA faces export regulation review",
                    "published_at": "2024-01-05T14:00:00+00:00",
                    "source": "gdelt",
                },
                {
                    "title": "Tariff concerns return to semiconductor supply chain",
                    "published_at": "2024-01-05T13:00:00+00:00",
                    "source": "newsapi",
                },
            ],
            "event_overlay": {"gdelt_article_count": 1},
            "meta": {"provider_status": {"gdelt": {"status": "ok"}}},
        },
        {"sector": "Technology"},
    )

    assert overlay["event_buckets"]["policy_regulation"] >= 1
    assert overlay["event_buckets"]["trade_supply_chain"] >= 1
    assert overlay["relevance_label"] in {"background", "material"}
    assert overlay["provenance"]["confidence"] == overlay["confidence"]


def test_report_uses_enriched_narrative_macro_and_provenance_sections():
    data_bundle = {
        "sentiment_narrative_flow": {
            "headline_count": 7,
            "source_mix": [
                {"source": "gnews", "count": 2},
                {"source": "newsapi", "count": 2},
                {"source": "gdelt", "count": 2},
            ],
            "sentiment_summary": {"bias": 0.18, "tone_label": "constructive"},
            "sentiment_confidence": 74.0,
            "attention_score": 68.0,
            "novelty_score": 64.0,
            "persistence_score": 71.0,
            "contradiction_score": 0.12,
            "topic_clusters": [{"topic": "policy_regulation", "count": 3}],
            "topic_buckets": {"policy_regulation": 3, "product_ai_cycle": 2},
            "event_overlay": {"gdelt_article_count": 2, "gdelt_tone_average": -0.3},
        },
        "macro_cross_asset": {
            "benchmark_proxy": "SPY",
            "benchmark_ret_21d": 0.05,
            "benchmark_vol_21d": 0.17,
            "macro_alignment_score": 63.0,
            "macro_regime_summary": {
                "regime": "growth_supportive",
                "summary": "growth supportive regime: rates are rising, inflation is stable, growth is supportive, labor is firm, liquidity conditions are neutral.",
            },
            "rates_context": {"direction": "rising"},
            "inflation_context": {"direction": "stable"},
            "growth_context": {"interpretation": "supportive"},
            "liquidity_context": {"regime": "neutral"},
            "macro_alignment_notes": ["Technology setups are more sensitive to rates and liquidity conditions than slower-growth sectors."],
        },
        "geopolitical_policy": {
            "event_intensity_score": 0.35,
            "event_buckets": {"policy_regulation": 2},
            "relevance_label": "background",
        },
        "relative_context": {
            "benchmark_proxy": "SPY",
            "benchmark_ret_21d": 0.05,
            "benchmark_vol_21d": 0.17,
            "broad_market_context": {"risk_tone": "risk_on"},
            "sector_context": {"coverage_note": "Sector-specific proxy coverage is unavailable, so broader-market benchmarks are being used."},
            "relative_move_summary": {"market_relative_note": "The stock is outperforming SPY on a 21-day basis."},
            "meta": {"coverage_status": "partial"},
        },
        "quality_provenance": {
            "freshness_summary": {"news": {"status": "fresh"}},
            "source_map": {"sentiment_narrative_flow": ["gnews", "newsapi", "gdelt"]},
            "domain_confidence": {"sentiment": 72.0, "macro": 68.0},
            "provider_notes": ["Narrative coverage is broad enough to support a directional sentiment overlay."],
            "domain_availability": {
                "sentiment": {"coverage_status": "available"},
                "macro": {"coverage_status": "partial"},
                "geopolitical": {"coverage_status": "partial"},
                "cross_asset": {"coverage_status": "partial", "fallback_used": True},
            },
        },
        "domain_availability": {
            "sentiment": {"coverage_status": "available"},
            "macro": {"coverage_status": "partial"},
            "geopolitical": {"coverage_status": "partial"},
            "cross_asset": {"coverage_status": "partial", "fallback_used": True},
        },
        "external_data_fabric": {"status": "ok"},
    }

    report = reports.build_analysis_report(
        symbol="NVDA",
        as_of_date="2024-01-05",
        horizon="swing",
        risk_mode="balanced",
        signal={"action": "HOLD", "score": 0.2, "confidence": 0.58, "reason_codes": ["MIXED"]},
        key_features={"ret_5d": 0.03},
        quality={"warnings": []},
        evidence={"sources": ["market_bars_daily", "news_raw"]},
        data_bundle=data_bundle,
        feature_factor_bundle={"composite_intelligence": {"Opportunity Quality Score": 58.0}},
        strategy={
            "final_signal": "HOLD",
            "confidence": 0.56,
            "top_contributors": [],
            "top_detractors": [],
            "component_scores": {"trend_following": {"weight": 0.3}},
        },
    )

    assert "GDELT event overlay" in report["sentiment_analysis"]
    assert "growth supportive regime" in report["macro_geopolitical_analysis"]
    assert "outperforming SPY" in report["macro_geopolitical_analysis"]
    assert "Domain confidence currently reads" in report["evidence_provenance"]
