import datetime as dt

from api.assistant import data_fabric
from api.data_providers import fundamentals, news


def test_fetch_news_items_deduplicates_multi_source_items(monkeypatch):
    from_ts = dt.datetime(2024, 1, 1, tzinfo=dt.timezone.utc)
    to_ts = dt.datetime(2024, 1, 5, tzinfo=dt.timezone.utc)

    monkeypatch.setattr(
        news,
        "_fetch_google_rss",
        lambda *_args, **_kwargs: [
            {
                "published_at": to_ts - dt.timedelta(hours=3),
                "source": "google_news_rss",
                "title": "NVIDIA beats expectations on AI demand",
                "url": "https://example.com/article-1",
                "content_snippet": "rss",
            }
        ],
    )
    monkeypatch.setattr(
        news,
        "search_gnews",
        lambda *_args, **_kwargs: [
            {
                "published_at": to_ts - dt.timedelta(hours=2),
                "source": "gnews",
                "title": "NVIDIA beats expectations on AI demand",
                "url": "https://example.com/article-1",
                "content_snippet": "gnews",
            }
        ],
    )
    monkeypatch.setattr(
        news,
        "search_newsapi",
        lambda *_args, **_kwargs: [
            {
                "published_at": to_ts - dt.timedelta(hours=1),
                "source": "newsapi",
                "title": "NVIDIA faces new export regulation questions",
                "url": "https://example.com/article-2",
                "content_snippet": "newsapi",
            }
        ],
    )
    monkeypatch.setattr(news, "fetch_company_news", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(news, "search_gdelt_articles", lambda *_args, **_kwargs: [])

    items = news.fetch_news_items("NVDA", from_ts, to_ts)

    assert len(items) == 2
    assert items[0]["source"] == "newsapi"
    deduped = next(item for item in items if item["url"] == "https://example.com/article-1")
    assert deduped["source"] == "google_news_rss|gnews"
    assert deduped["content_snippet"] == "rss"


def test_fetch_fundamentals_quarterly_prefers_alphavantage(monkeypatch):
    monkeypatch.setattr(
        fundamentals,
        "fetch_alphavantage_quarterly",
        lambda symbol: [
            {
                "symbol": symbol,
                "fiscal_period_end": dt.date(2024, 1, 31),
                "report_date": dt.date(2024, 1, 31),
                "revenue": 10.0,
                "eps": 1.0,
                "gross_margin": 0.5,
                "op_margin": 0.3,
                "fcf": 2.0,
                "source": "alphavantage",
            }
        ],
    )

    rows = fundamentals.fetch_fundamentals_quarterly("NVDA")

    assert rows[0]["source"] == "alphavantage"
    assert rows[0]["revenue"] == 10.0


def test_data_fabric_enriches_domains_with_provenance(monkeypatch):
    monkeypatch.setenv("FTIP_DATA_FABRIC_ENABLED", "1")

    as_of_date = dt.date(2024, 1, 5)

    def _bars(symbol, _start, end):
        rows = []
        for idx in range(30):
            rows.append(
                {
                    "as_of_date": end - dt.timedelta(days=29 - idx),
                    "close": 100.0 + idx,
                    "source": "stooq",
                }
            )
        return rows

    monkeypatch.setattr(data_fabric, "fetch_reference_bars", _bars)
    monkeypatch.setattr(
        data_fabric,
        "fetch_company_filing_profile",
        lambda _symbol, **_kwargs: {
            "mapping": {"match_type": "exact_ticker", "cik": "0001045810"},
            "latest_form": "10-Q",
            "filing_recency_days": 40,
            "coverage_flags": {"revenue": True, "net_income": True},
            "statement_snapshot": {
                "latest_quarter": {
                    "revenue": 1000.0,
                    "report_date": "2024-01-01",
                    "operating_income": 250.0,
                },
                "latest_balance_sheet": {
                    "assets": 5000.0,
                    "current_assets": 2200.0,
                    "current_liabilities": 1000.0,
                    "equity": 2500.0,
                },
            },
            "normalized_metrics": {
                "revenue_growth_yoy": 0.2,
                "operating_margin": 0.25,
                "current_ratio": 2.2,
                "debt_to_equity": 0.4,
                "free_cash_flow": 180.0,
                "free_cash_flow_margin": 0.18,
                "positive_fcf_ratio": 0.75,
            },
            "quality_proxies": {
                "filing_recency_score": 88.0,
                "reporting_completeness_score": 85.0,
                "reporting_quality_proxy": 82.0,
                "business_quality_durability": 79.0,
            },
            "coverage_score": 0.86,
            "strength_summary": ["Revenue growth remains strong."],
            "weakness_summary": [],
            "coverage_caveats": [],
        },
    )
    monkeypatch.setattr(
        data_fabric,
        "fetch_company_profile",
        lambda _symbol: {"name": "NVIDIA", "country": "US", "exchange": "NASDAQ"},
    )
    monkeypatch.setattr(
        data_fabric,
        "fetch_basic_financials",
        lambda _symbol: {
            "net_margin": 0.22,
            "operating_margin_ttm": 0.28,
            "current_ratio_quarterly": 2.1,
            "quick_ratio_quarterly": 1.8,
            "total_debt_to_equity_quarterly": 0.4,
            "revenue_growth_ttm_yoy": 0.18,
        },
    )
    monkeypatch.setattr(
        data_fabric,
        "fetch_company_overview",
        lambda _symbol: {
            "sector": "Technology",
            "industry": "Semiconductors",
            "profit_margin": 0.21,
            "operating_margin_ttm": 0.27,
            "quarterly_revenue_growth_yoy": 0.2,
        },
    )
    monkeypatch.setattr(
        data_fabric,
        "search_gnews",
        lambda *_args, **_kwargs: [
            {
                "published_at": dt.datetime(2024, 1, 5, 12, tzinfo=dt.timezone.utc),
                "source": "gnews",
                "title": "NVIDIA gains after policy concerns fade",
                "url": "https://example.com/gnews",
                "content_snippet": "gnews",
            }
        ],
    )
    monkeypatch.setattr(
        data_fabric,
        "search_newsapi",
        lambda *_args, **_kwargs: [
            {
                "published_at": dt.datetime(2024, 1, 5, 13, tzinfo=dt.timezone.utc),
                "source": "newsapi",
                "title": "NVIDIA faces export regulation review",
                "url": "https://example.com/newsapi",
                "content_snippet": "newsapi",
            }
        ],
    )
    monkeypatch.setattr(data_fabric, "fetch_company_news", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(
        data_fabric,
        "search_gdelt_articles",
        lambda *_args, **_kwargs: [
            {
                "published_at": dt.datetime(2024, 1, 5, 14, tzinfo=dt.timezone.utc),
                "source": "gdelt",
                "title": "NVIDIA sees stronger supply chain outlook",
                "url": "https://example.com/gdelt",
                "content_snippet": "gdelt",
            }
        ],
    )
    monkeypatch.setattr(
        data_fabric,
        "fetch_fred_series",
        lambda series_id, limit=12: {
            "series_id": series_id,
            "observations": [
                {"date": "2024-01-01", "value": 4.2},
                {"date": "2023-12-01", "value": 4.0},
            ],
        },
    )
    monkeypatch.setattr(
        data_fabric,
        "fetch_world_bank_indicator",
        lambda **_kwargs: {
            "observations": [
                {"date": "2023", "value": 2.7},
                {"date": "2022", "value": 2.1},
            ]
        },
    )

    base_bundle = {
        "market_price_volume": {"latest_close": 129.0},
        "quality_provenance": {"warnings": []},
    }
    overlay = data_fabric.enrich_data_bundle(
        job_context={"symbol": "NVDA", "as_of_date": as_of_date.isoformat()},
        symbol_meta={"name": "NVIDIA", "sector": "Technology", "country": "US"},
        data_bundle=base_bundle,
    )
    merged = data_fabric.merge_into_data_bundle(data_bundle=base_bundle, overlay=overlay)

    assert overlay["enabled"] is True
    assert merged["fundamental_filing"]["filing_recency_days"] == 40
    assert merged["fundamental_filing"]["normalized_metrics"]["operating_margin"] == 0.25
    assert merged["fundamental_filing"]["quality_proxies"]["reporting_quality_proxy"] == 82.0
    assert merged["sentiment_narrative_flow"]["source_breakdown"]["gnews"] == 1
    assert merged["sentiment_narrative_flow"]["sentiment_level_proxy"] is not None
    assert merged["macro_cross_asset"]["macro_regime_context"]["regime"]
    assert merged["geopolitical_policy"]["event_buckets"]["policy_regulation"] >= 1
    assert merged["quality_provenance"]["source_map"]["fundamental_filing"]
    assert merged["market_price_volume"]["external_verification"]["source"] == "stooq"
    assert merged["relative_context"]["benchmark_proxy"] == "XLK"
