import datetime as dt

import pytest

from api.data_providers import bars, news
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
