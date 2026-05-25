import datetime as dt

import pytest

from api.data_providers import bars, events, news, premium
from api.data_providers.errors import ProviderUnavailable


def test_fetch_daily_bars_with_meta_tracks_fallback(monkeypatch: pytest.MonkeyPatch):
    def fail_primary(symbol, start, end):
        raise ProviderUnavailable(
            "PROVIDER_UNAVAILABLE",
            "primary unavailable",
            provider_name="massive_polygon",
            source_type="market_data",
        )

    def succeed_secondary(symbol, start, end):
        return [
            {
                "symbol": symbol,
                "as_of_date": end,
                "open": 100.0,
                "high": 101.0,
                "low": 99.5,
                "close": 100.5,
                "volume": 1000,
                "source": "stooq",
            }
        ]

    monkeypatch.setattr(
        bars,
        "_daily_provider_attempts",
        lambda: [
            ("massive_polygon", fail_primary),
            ("stooq", succeed_secondary),
        ],
    )

    rows, metadata = bars.fetch_daily_bars_with_meta(
        "AAPL", dt.date(2024, 1, 1), dt.date(2024, 1, 5)
    )

    assert rows
    assert metadata["provider_name"] == "stooq"
    assert metadata["fallback_used"] is True
    assert metadata["attempt_count"] == 2
    assert metadata["attempts"][0]["status"] == "failed"
    assert metadata["attempts"][1]["status"] == "success"
    assert "fallback_chain_used" in metadata["source_warning_flags"]
    assert metadata["strength_label"] in {"weak", "mixed", "strong"}
    assert metadata["source_strength_summary"]
    assert metadata["connector_slot"]
    assert metadata["fallback_chain_used"] == ["massive_polygon", "stooq"]
    assert metadata["provider_domain"] == "market_data"


def test_fetch_daily_bars_with_meta_marks_run_suppressed_provider(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(
        bars,
        "_daily_provider_attempts",
        lambda: [
            ("massive_polygon", lambda *_args, **_kwargs: []),
            (
                "stooq",
                lambda symbol, _start, end: [
                    {
                        "symbol": symbol,
                        "as_of_date": end,
                        "open": 100.0,
                        "high": 101.0,
                        "low": 99.0,
                        "close": 100.0,
                        "volume": 10,
                        "source": "stooq",
                    }
                ],
            ),
        ],
    )

    rows, metadata = bars.fetch_daily_bars_with_meta(
        "AAPL",
        dt.date(2024, 1, 1),
        dt.date(2024, 1, 5),
        disabled_providers=["massive_polygon"],
    )

    assert rows
    assert metadata["provider_name"] == "stooq"
    assert metadata["fallback_used"] is True
    assert metadata["suppressed_providers"] == ["massive_polygon"]
    assert metadata["attempts"][0]["status"] == "suppressed"
    assert metadata["attempts"][1]["status"] == "success"
    assert "provider_run_suppressed" in metadata["source_warning_flags"]
    assert metadata["available_providers"][0] == "massive_polygon"


def test_scalar_helpers_handle_series_like_values() -> None:
    pd = pytest.importorskip("pandas")

    single_float = pd.Series([101.25])
    single_int = pd.Series([1200])

    assert bars._float_scalar_or_none(single_float) == 101.25
    assert bars._int_scalar_or_none(single_int) == 1200
    assert bars._float_scalar_or_none(None) is None


def test_fetch_news_items_with_meta_surfaces_partial_result_status(
    monkeypatch: pytest.MonkeyPatch,
):
    from_ts = dt.datetime(2024, 1, 1, tzinfo=dt.timezone.utc)
    to_ts = dt.datetime(2024, 1, 3, tzinfo=dt.timezone.utc)

    monkeypatch.setattr(news, "source_allowed", lambda provider_name: True)
    monkeypatch.setattr(
        news,
        "_fetch_google_rss",
        lambda symbol, _from_ts, _to_ts: [
            {
                "published_at": from_ts,
                "source": "google_news_rss",
                "title": f"{symbol} headline",
                "url": "https://example.com/story",
                "content_snippet": "demo snippet",
            }
        ],
    )
    monkeypatch.setattr(
        news,
        "search_gnews",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            news.NewsProviderError("PROVIDER_UNAVAILABLE", "gnews unavailable")
        ),
    )
    monkeypatch.setattr(news, "search_newsapi", lambda *args, **kwargs: [])
    monkeypatch.setattr(news, "fetch_company_news", lambda *args, **kwargs: [])
    monkeypatch.setattr(news, "search_gdelt_articles", lambda *args, **kwargs: [])

    items, metadata = news.fetch_news_items_with_meta("AAPL", from_ts, to_ts)

    assert len(items) == 1
    assert metadata["provider_name"] == "google_news_rss"
    assert metadata["partial_result"] is True
    assert metadata["attempt_count"] >= 2
    assert "partial_result" in metadata["source_warning_flags"]
    assert metadata["source_strength_summary"]
    assert metadata["connector_slot"]


def test_premium_connector_probe_gracefully_handles_missing_credentials(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(premium.config, "massive_api_key", lambda: None)
    monkeypatch.setattr(premium.config, "finnhub_api_key", lambda: None)
    monkeypatch.setattr(premium.config, "gnews_api_key", lambda: None)
    monkeypatch.setattr(premium.config, "news_api_key", lambda: None)
    monkeypatch.setattr(premium.config, "alphavantage_api_key", lambda: None)
    monkeypatch.setattr(premium.config, "sec_user_agent", lambda: "")

    market_probe = premium.probe_premium_connector(
        "premium_market_data",
        execute_live=True,
    )
    filings_probe = premium.probe_premium_connector(
        "filings_intel",
        execute_live=True,
    )
    overview = premium.build_premium_connector_overview(execute_live=True)

    assert market_probe["readiness_status"] == "missing_credentials"
    assert market_probe["live_probe_status"] == "blocked_missing_credentials"
    assert market_probe["failure_reason_classification"] == "missing_credentials"
    assert filings_probe["readiness_status"] == "misconfigured"
    assert filings_probe["live_probe_status"] == "blocked_missing_credentials"
    assert filings_probe["failure_reason_classification"] == "missing_credentials"
    assert overview["status"] == "limited"
    assert overview["connector_count"] == 4
    assert overview["warnings"]


def test_event_intelligence_overlay_classifies_filings_earnings_and_premium_source_support() -> None:
    as_of_date = dt.date(2026, 5, 24)
    fundamentals_overlay = {
        "filing_backbone": {
            "latest_filing_date": "2026-05-20",
            "filing_recency_days": 4,
            "status": "fresh",
            "recent_filings": [
                {"form": "10-Q", "filing_date": "2026-05-20"},
                {"form": "8-K", "filing_date": "2026-05-22"},
            ],
        },
        "normalized_metrics": {"revenue_growth_yoy": 0.18},
        "meta": {
            "sources": ["sec_edgar", "alphavantage"],
            "confidence": 82.0,
            "fallback_used": False,
        },
        "provider_snapshot": {
            "alphavantage_earnings_intel": {
                "latest_quarter": {
                    "reported_date": dt.date(2026, 5, 15),
                    "surprise_pct": 7.5,
                    "surprise_direction": "beat",
                },
                "recent_quarters": [
                    {"reported_date": dt.date(2026, 5, 15), "surprise_pct": 7.5}
                ],
                "estimate_revision_support": 72.0,
                "freshness_status": "fresh",
            }
        },
    }
    news_overlay = {
        "aggregated_headlines": [
            {
                "published_at": dt.datetime(2026, 5, 23, tzinfo=dt.timezone.utc),
                "source": "finnhub",
                "title": "NVDA raises guidance after strong earnings beat",
            }
        ],
        "meta": {
            "sources": ["finnhub"],
            "confidence": 74.0,
            "fallback_used": False,
        },
    }

    overlay = events.build_event_intelligence_overlay(
        symbol="NVDA",
        company_name="NVIDIA",
        as_of_date=as_of_date,
        fundamentals_overlay=fundamentals_overlay,
        news_overlay=news_overlay,
    )

    assert overlay["events"]
    assert overlay["event_type_counts"]["filing"] >= 1
    assert overlay["event_type_counts"]["earnings"] >= 1
    assert overlay["event_freshness"] in {"fresh", "imminent", "recent"}
    assert overlay["event_relevance"] in {"high", "medium"}
    assert overlay["catalyst_quality"] > 60.0
    assert overlay["filings_change_signal"] > 50.0
    assert overlay["estimate_revision_support"] == 72.0
    assert overlay["source_strength_support"] > overlay["source_strength_penalty"]
    assert overlay["premium_evidence_bonus"] > 0.0
    assert overlay["meta"]["source_strength_summary"]
