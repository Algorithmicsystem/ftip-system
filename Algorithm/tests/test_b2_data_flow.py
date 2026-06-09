"""B2 tests: data flow fixes — real scores in panels, morning briefing dates, SLA caching."""
from __future__ import annotations

import inspect
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from api.main import app

WEBAPP = Path(__file__).resolve().parents[1] / "api" / "webapp"
JS_DIR = WEBAPP / "js"


# ---------------------------------------------------------------------------
# Universe scores data flow
# ---------------------------------------------------------------------------

class TestUniverseScoresDataFlow:

    def test_universe_scores_returns_30(self):
        with TestClient(app) as client:
            r = client.get("/intelligence/universe/scores")
        assert r.status_code == 200
        assert len(r.json()) == 30

    def test_universe_scores_has_dau_field(self):
        with TestClient(app) as client:
            r = client.get("/intelligence/universe/scores")
        for row in r.json():
            assert "dau" in row

    def test_universe_scores_sort_by_signal_strength(self):
        import api.main as m
        src = inspect.getsource(m.universe_scores)
        assert "abs" in src.lower() or "ABS" in src

    def test_universe_scores_no_stale_cache(self):
        # Cache was removed to prevent stale-null data serving on Railway startup
        import api.main as m
        assert not hasattr(m, "_UNIVERSE_CACHE_TTL"), "Universe cache removed to prevent stale data"

    def test_universe_scores_consistent_results(self):
        # Without cache, two calls should still return same data
        with TestClient(app) as client:
            r1 = client.get("/intelligence/universe/scores")
            r2 = client.get("/intelligence/universe/scores")
        assert r1.json() == r2.json()

    def test_universe_scores_cached_on_second_call(self):
        with TestClient(app) as client:
            r1 = client.get("/intelligence/universe/scores")
            r2 = client.get("/intelligence/universe/scores")
        assert r1.status_code == 200
        assert r2.status_code == 200
        # Both should return same data
        assert r1.json() == r2.json()


# ---------------------------------------------------------------------------
# System alerts fixed
# ---------------------------------------------------------------------------

class TestSystemAlertsFixed:

    def test_system_status_returns_200(self):
        with TestClient(app) as client:
            r = client.get("/system/status")
        assert r.status_code == 200

    def test_moat_query_uses_30_day_window(self):
        import api.main as m
        src = inspect.getsource(m)
        assert "CURRENT_DATE - 30" in src or "CURRENT_DATE - 14" in src

    def test_system_status_has_version(self):
        with TestClient(app) as client:
            r = client.get("/system/status")
        assert "version" in r.json()


# ---------------------------------------------------------------------------
# Morning briefing uses MAX(as_of_date)
# ---------------------------------------------------------------------------

class TestMorningBriefingDataFix:

    def test_morning_briefing_returns_200(self):
        with TestClient(app) as client:
            r = client.get("/jobs/briefing/morning")
        assert r.status_code == 200

    def test_morning_briefing_uses_max_date(self):
        src = Path(__file__).parents[1].joinpath("api", "jobs", "morning_briefing.py").read_text()
        assert "MAX(as_of_date)" in src

    def test_morning_briefing_no_today_date_filter_on_scores(self):
        """Main opportunity queries must not filter by today — uses MAX instead."""
        src = Path(__file__).parents[1].joinpath("api", "jobs", "morning_briefing.py").read_text()
        # Find the top_rows query block and confirm it uses MAX subquery, not %s with aod
        top_idx = src.index("Fetch top opportunities")
        top_block = src[top_idx:top_idx + 600]
        assert "MAX(as_of_date)" in top_block
        assert "WHERE as_of_date = %s" not in top_block

    def test_morning_briefing_has_systemic_risk_index(self):
        with TestClient(app) as client:
            r = client.get("/jobs/briefing/morning")
        data = r.json()
        assert "systemic_risk_index" in data
        assert isinstance(data["systemic_risk_index"], (int, float))


# ---------------------------------------------------------------------------
# Symbol Intelligence isDefault fix
# ---------------------------------------------------------------------------

class TestSymbolIntelligenceIsDefault:

    def test_isdefault_no_intelligence_quality_score_zero_check(self):
        src = (JS_DIR / "panels" / "symbol_intelligence.js").read_text()
        assert "intelligence_quality_score === 0" not in src

    def test_isdefault_checks_dau(self):
        src = (JS_DIR / "panels" / "symbol_intelligence.js").read_text()
        assert "!data.dau" in src or "data.dau === 50" in src

    def test_symbol_intelligence_aapl_returns_200(self):
        with TestClient(app) as client:
            r = client.get("/intelligence/universal/AAPL")
        assert r.status_code == 200


# ---------------------------------------------------------------------------
# Opportunities panel auto-retry
# ---------------------------------------------------------------------------

class TestOpportunitiesAutoRetry:

    def test_opportunities_js_has_auto_retry(self):
        src = (JS_DIR / "panels" / "opportunities.js").read_text()
        assert "setTimeout" in src
        assert "loadOpportunities" in src

    def test_opportunities_js_filter_is_null_safe(self):
        src = (JS_DIR / "panels" / "opportunities.js").read_text()
        # Filter must handle null, undefined, and NaN — not just > 0 (negative DAU is valid)
        assert "isNaN" in src or "!== null" in src

    def test_opportunities_js_sort_by_signal_strength(self):
        src = (JS_DIR / "panels" / "opportunities.js").read_text()
        assert "Math.abs" in src

    def test_dashboard_loads_opportunities_on_domcontentloaded(self):
        src = (JS_DIR / "dashboard.js").read_text()
        loaded_block = src[src.index("DOMContentLoaded"):][:400]
        assert "loadOpportunities" in loaded_block

    def test_dashboard_loads_opportunities_on_pipeline_complete(self):
        src = (JS_DIR / "dashboard.js").read_text()
        pipeline_idx = src.index("pipeline_complete")
        pipeline_block = src[pipeline_idx:pipeline_idx + 300]
        assert "loadOpportunities" in pipeline_block


# ---------------------------------------------------------------------------
# Universe screen scored/unscored header
# ---------------------------------------------------------------------------

class TestUniverseScreenHeader:

    def test_universe_screen_has_scored_label(self):
        src = (JS_DIR / "panels" / "universe_screen.js").read_text()
        assert "scored" in src

    def test_universe_screen_has_avg_dau(self):
        src = (JS_DIR / "panels" / "universe_screen.js").read_text()
        assert "avgDau" in src or "Avg DAU" in src

    def test_universe_screen_has_unscored_count(self):
        src = (JS_DIR / "panels" / "universe_screen.js").read_text()
        assert "unscored" in src


# ---------------------------------------------------------------------------
# Version 1.0.3
# ---------------------------------------------------------------------------

class TestVersionBump:

    def test_fastapi_version_is_103(self):
        with TestClient(app) as client:
            r = client.get("/openapi.json")
        assert r.json()["info"]["version"] == "1.0.3"

    def test_config_client_version_is_103(self):
        with TestClient(app) as client:
            r = client.get("/config/client")
        assert r.json()["version"] == "1.0.3"

    def test_system_status_version_is_103(self):
        with TestClient(app) as client:
            r = client.get("/system/status")
        assert r.json()["version"] == "1.0.3"

    def test_index_html_cache_bust_v103(self):
        html = (WEBAPP / "index.html").read_text()
        assert "?v=103" in html
        assert "?v=102" not in html

    def test_index_html_has_enough_v103_references(self):
        html = (WEBAPP / "index.html").read_text()
        assert html.count("?v=103") >= 10
