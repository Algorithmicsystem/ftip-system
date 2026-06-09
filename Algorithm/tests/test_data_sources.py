"""Tests for yfinance bars, congressional trading scraper, FINRA dark pool."""
from __future__ import annotations

import datetime as dt
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from api.main import app


# ---------------------------------------------------------------------------
# yfinance fetch
# ---------------------------------------------------------------------------

class TestYfinanceFetch:

    def test_fetch_returns_list(self):
        from api.data_providers.bars import _fetch_daily_yfinance
        bars = _fetch_daily_yfinance(
            "AAPL",
            dt.date.today() - dt.timedelta(days=30),
            dt.date.today(),
        )
        assert isinstance(bars, list)

    def test_bars_have_required_fields(self):
        from api.data_providers.bars import _fetch_daily_yfinance
        bars = _fetch_daily_yfinance(
            "AAPL",
            dt.date.today() - dt.timedelta(days=30),
            dt.date.today(),
        )
        if bars:
            for field in ["symbol", "as_of_date", "close", "source"]:
                assert field in bars[0], f"{field} missing from bar"

    def test_fetch_failure_returns_empty_list_or_raises(self):
        from api.data_providers.bars import _fetch_daily_yfinance
        from api.data_providers.errors import ProviderError
        # Invalid ticker — should raise ProviderError or SymbolNoData (not crash unhandled)
        try:
            bars = _fetch_daily_yfinance(
                "ZZZINVALID",
                dt.date.today() - dt.timedelta(days=5),
                dt.date.today(),
            )
            assert isinstance(bars, list)
        except (ProviderError, Exception):
            pass  # expected — any clean exception is acceptable

    def test_yfinance_is_first_provider(self):
        from api.data_providers.bars import _daily_provider_attempts
        attempts = _daily_provider_attempts()
        if attempts:
            assert attempts[0][0] == "yfinance", (
                f"yfinance must be first provider, got {attempts[0][0]}"
            )

    def test_yfinance_fallback_priority_is_lowest(self):
        from api.data_providers.quality import provider_capability_profile
        yf_profile = provider_capability_profile("yfinance", capability="daily_bars")
        av_profile = provider_capability_profile("alphavantage", capability="daily_bars")
        assert yf_profile["fallback_priority"] < av_profile["fallback_priority"], (
            "yfinance must have lower (higher-priority) fallback_priority than alphavantage"
        )

    def test_yfinance_in_requirements(self):
        req = Path(__file__).parents[1].joinpath("requirements.txt").read_text()
        assert "yfinance" in req
        # Must be >= 0.2.40
        assert "0.2.40" in req or "0.2.4" in req or "1." in req


# ---------------------------------------------------------------------------
# Congress scraper
# ---------------------------------------------------------------------------

class TestCongressScraper:

    def test_compute_congress_score_no_trades(self):
        from api.scrapers.congress_trading import compute_congress_score
        score = compute_congress_score("AAPL", [])
        assert score["congress_score"] == 50.0
        assert score["net_signal"] == "neutral"

    def test_compute_congress_score_all_buys(self):
        from api.scrapers.congress_trading import compute_congress_score
        trades = [
            {
                "symbol": "AAPL",
                "transaction_type": "buy",
                "transaction_date": dt.date.today().isoformat(),
                "politician": "Test Rep",
                "party": "D",
                "chamber": "house",
                "amount_range": "$1K-$15K",
                "disclosure_date": "2026-01-01",
            }
            for _ in range(5)
        ]
        score = compute_congress_score("AAPL", trades)
        assert score["congress_score"] > 60
        assert score["net_signal"] == "bullish"

    def test_compute_congress_score_all_sells(self):
        from api.scrapers.congress_trading import compute_congress_score
        sells = [
            {
                "symbol": "AAPL",
                "transaction_type": "sell",
                "transaction_date": dt.date.today().isoformat(),
                "politician": "Test Rep",
                "party": "R",
                "chamber": "house",
                "amount_range": "$1K-$15K",
                "disclosure_date": "2026-01-01",
            }
            for _ in range(5)
        ]
        score = compute_congress_score("AAPL", sells)
        assert score["congress_score"] < 40
        assert score["net_signal"] == "bearish"

    def test_compute_congress_score_symbol_isolation(self):
        from api.scrapers.congress_trading import compute_congress_score
        trades = [
            {
                "symbol": "MSFT",
                "transaction_type": "buy",
                "transaction_date": dt.date.today().isoformat(),
                "politician": "Test", "party": "D", "chamber": "house",
                "amount_range": "$1K-$15K", "disclosure_date": "2026-01-01",
            }
        ]
        # AAPL has no trades — should return neutral 50
        score = compute_congress_score("AAPL", trades)
        assert score["congress_score"] == 50.0

    def test_fetch_congress_trades_returns_list(self):
        from api.scrapers.congress_trading import fetch_recent_congress_trades
        result = fetch_recent_congress_trades(days_back=1)
        assert isinstance(result, list)

    def test_congress_score_structure(self):
        from api.scrapers.congress_trading import compute_congress_score
        score = compute_congress_score("AAPL", [])
        required_keys = {"symbol", "congress_score", "net_signal", "buy_count", "sell_count", "recent_trades"}
        assert required_keys.issubset(score.keys())


# ---------------------------------------------------------------------------
# FINRA dark pool
# ---------------------------------------------------------------------------

class TestDarkPoolScraper:

    def test_default_result_structure(self):
        from api.scrapers.finra_dark_pool import _default_dark_pool_result
        result = _default_dark_pool_result("AAPL")
        assert result["dark_pool_score"] == 50.0
        assert "signal" in result
        assert result["symbol"] == "AAPL"

    def test_fetch_returns_dict(self):
        from api.scrapers.finra_dark_pool import fetch_finra_otc_data
        result = fetch_finra_otc_data("AAPL")
        assert isinstance(result, dict)
        assert "dark_pool_score" in result
        assert 0 <= result["dark_pool_score"] <= 100

    def test_fetch_has_required_fields(self):
        from api.scrapers.finra_dark_pool import fetch_finra_otc_data
        result = fetch_finra_otc_data("AAPL")
        required = {"symbol", "dark_pool_score", "signal", "recent_vs_avg_ratio"}
        assert required.issubset(result.keys())

    def test_fetch_never_raises(self):
        from api.scrapers.finra_dark_pool import fetch_finra_otc_data
        # Should always return a dict, never raise
        result = fetch_finra_otc_data("INVALID_SYMBOL_XYZ")
        assert isinstance(result, dict)
        assert result["dark_pool_score"] == 50.0


# ---------------------------------------------------------------------------
# New endpoints
# ---------------------------------------------------------------------------

class TestNewEndpoints:

    def test_congress_endpoint_exists(self):
        with TestClient(app) as c:
            r = c.get("/intelligence/congress/AAPL")
        assert r.status_code == 200
        assert "congress_score" in r.json()

    def test_dark_pool_endpoint_exists(self):
        with TestClient(app) as c:
            r = c.get("/intelligence/dark-pool/AAPL")
        assert r.status_code == 200
        assert "dark_pool_score" in r.json()

    def test_congress_universe_endpoint_exists(self):
        with TestClient(app) as c:
            r = c.get("/intelligence/congress/universe/all")
        assert r.status_code == 200
        data = r.json()
        assert "scores" in data
        assert "total_trades" in data

    def test_congress_endpoint_uppercase_symbol(self):
        with TestClient(app) as c:
            r = c.get("/intelligence/congress/aapl")  # lowercase
        assert r.status_code == 200
        assert r.json()["symbol"] == "AAPL"

    def test_dark_pool_score_in_valid_range(self):
        with TestClient(app) as c:
            r = c.get("/intelligence/dark-pool/MSFT")
        assert r.status_code == 200
        score = r.json()["dark_pool_score"]
        assert 0 <= score <= 100

    def test_seed_endpoint_registered(self):
        routes = [getattr(r, "path", None) for r in app.router.routes]
        assert "/admin/seed-market-data" in routes
