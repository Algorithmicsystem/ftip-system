"""Prompt 1 regression tests: auth, briefing, navigation, schema, universe screen."""
from __future__ import annotations

import os
import re
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from api.main import app

WEBAPP = Path(__file__).resolve().parents[1] / "api" / "webapp"
MIGRATIONS = Path(__file__).resolve().parents[1] / "api" / "migrations"


# ---------------------------------------------------------------------------
# Auth fixes
# ---------------------------------------------------------------------------

class TestAuthFixes:

    def test_config_client_returns_200_without_auth(self):
        with TestClient(app) as client:
            r = client.get("/config/client")
        assert r.status_code == 200

    def test_config_client_returns_api_key_field(self):
        with TestClient(app) as client:
            r = client.get("/config/client")
        data = r.json()
        assert "api_key" in data
        assert "version" in data
        assert data["version"] in ("30.0.0", "31.0.0", "32.0.0", "33.0.0", "34.0.0", "1.0.0", "1.0.1")

    def test_api_client_sends_correct_header(self):
        src = (WEBAPP / "js" / "api_client.js").read_text()
        assert "X-FTIP-API-Key" in src, "api_client.js must send X-FTIP-API-Key header"
        assert "X-API-Key'" not in src, "old header name X-API-Key must not be present"

    def test_api_client_has_self_init_iife(self):
        src = (WEBAPP / "js" / "api_client.js").read_text()
        assert "axiomSelfInit" in src, "api_client.js must have axiomSelfInit IIFE"
        assert "/config/client" in src, "axiomSelfInit must fetch /config/client"

    def test_all_js_imports_have_cache_bust(self):
        html = (WEBAPP / "index.html").read_text()
        scripts = re.findall(r'<script src="(/app/static/[^"]+)"', html)
        assert len(scripts) >= 8, "Expected at least 8 local script tags"
        for src in scripts:
            assert "?v=" in src, f"Script {src} missing cache-busting ?v= parameter"

    def test_all_css_imports_have_cache_bust(self):
        html = (WEBAPP / "index.html").read_text()
        stylesheets = re.findall(r'<link[^>]+href="(/app/static/[^"]+)"', html)
        assert len(stylesheets) >= 2, "Expected at least 2 local CSS links"
        for href in stylesheets:
            assert "?v=" in href, f"Stylesheet {href} missing cache-busting ?v= parameter"


# ---------------------------------------------------------------------------
# Briefing
# ---------------------------------------------------------------------------

class TestMorningBriefing:

    def test_morning_briefing_returns_200_with_auth(self, monkeypatch):
        monkeypatch.setenv("FTIP_API_KEY", "test-key")
        with TestClient(app) as client:
            r = client.get(
                "/jobs/briefing/morning",
                headers={"X-FTIP-API-Key": "test-key"},
            )
        assert r.status_code == 200

    def test_morning_briefing_has_briefing_text(self, monkeypatch):
        monkeypatch.setenv("FTIP_API_KEY", "test-key")
        with TestClient(app) as client:
            r = client.get(
                "/jobs/briefing/morning",
                headers={"X-FTIP-API-Key": "test-key"},
            )
        assert r.status_code == 200
        data = r.json()
        assert "briefing_text" in data
        assert isinstance(data["briefing_text"], str)

    def test_morning_briefing_text_mentions_regime(self, monkeypatch):
        monkeypatch.setenv("FTIP_API_KEY", "test-key")
        with TestClient(app) as client:
            r = client.get(
                "/jobs/briefing/morning",
                headers={"X-FTIP-API-Key": "test-key"},
            )
        assert r.status_code == 200
        text = r.json().get("briefing_text", "")
        assert len(text) > 20, "briefing_text must be a real narrative, not empty"

    def test_briefing_js_shows_friendly_error(self):
        src = (WEBAPP / "js" / "panels" / "morning_briefing.js").read_text()
        assert "statusCode" in src or "status_code" in src.lower() or "err.statusCode" in src, \
            "morning_briefing.js must check error status code for friendly messages"
        assert "JSON.stringify" not in src, "morning_briefing.js must not show raw JSON to user"


# ---------------------------------------------------------------------------
# Universal intelligence
# ---------------------------------------------------------------------------

class TestUniversalIntelligence:

    def test_universal_aapl_returns_200(self):
        with TestClient(app) as client:
            r = client.get("/intelligence/universal/AAPL")
        assert r.status_code == 200

    def test_universal_primary_driver_field_present(self):
        with TestClient(app) as client:
            r = client.get("/intelligence/universal/AAPL")
        data = r.json()
        assert "primary_driver" in data

    def test_universal_top_supporting_evidence_field(self):
        with TestClient(app) as client:
            r = client.get("/intelligence/universal/AAPL")
        data = r.json()
        assert "top_supporting_evidence" in data
        assert isinstance(data["top_supporting_evidence"], list)

    def test_symbol_intelligence_js_uses_top_supporting_evidence(self):
        src = (WEBAPP / "js" / "panels" / "symbol_intelligence.js").read_text()
        assert "top_supporting_evidence" in src, \
            "symbol_intelligence.js must use top_supporting_evidence not key_reasons"
        assert "key_reasons" not in src or "top_supporting_evidence" in src


# ---------------------------------------------------------------------------
# Universe screen
# ---------------------------------------------------------------------------

class TestUniverseScreen:

    def test_universe_scores_endpoint_exists(self):
        with TestClient(app) as client:
            r = client.get("/intelligence/universe/scores")
        assert r.status_code == 200

    def test_universe_scores_returns_30_symbols(self):
        with TestClient(app) as client:
            r = client.get("/intelligence/universe/scores")
        assert r.status_code == 200
        data = r.json()
        assert len(data) == 30, f"Expected 30 symbols, got {len(data)}"

    def test_universe_scores_sorted_by_dau(self):
        with TestClient(app) as client:
            r = client.get("/intelligence/universe/scores")
        data = r.json()
        with_dau = [item for item in data if item["dau"] is not None]
        if len(with_dau) >= 2:
            for i in range(len(with_dau) - 1):
                assert with_dau[i]["dau"] >= with_dau[i + 1]["dau"], \
                    "Results must be sorted by dau DESC"

    def test_universe_scores_no_auth_required(self):
        with TestClient(app) as client:
            r = client.get("/intelligence/universe/scores")
        assert r.status_code == 200


# ---------------------------------------------------------------------------
# Schema audit
# ---------------------------------------------------------------------------

class TestSchemaAudit:

    def _migration_cols(self, filename: str) -> set:
        path = MIGRATIONS / filename
        if not path.exists():
            return set()
        sql = path.read_text()
        cols = set()
        for m in re.finditer(
            r"^\s{4}(\w+)\s+(?:TEXT|NUMERIC|DATE|INT|BOOLEAN|UUID|JSONB|TIMESTAMPTZ|DOUBLE\s+PRECISION)",
            sql, re.MULTILINE | re.IGNORECASE,
        ):
            cols.add(m.group(1).lower())
        return cols

    def test_morning_briefings_no_generated_at(self):
        """readiness_check.py used generated_at which doesn't exist — fixed to created_at."""
        cols = self._migration_cols("071_morning_briefings.sql")
        assert "created_at" in cols
        assert "generated_at" not in cols
        src = (Path(__file__).parents[1] / "api" / "cloud" / "readiness_check.py").read_text()
        assert "generated_at" not in src, "readiness_check.py must use created_at not generated_at"

    def test_market_breadth_daily_no_regime_label(self):
        """market_breadth_daily has no regime_label — queries must not reference it."""
        cols_034 = self._migration_cols("034_market_breadth_daily.sql")
        assert "regime_label" not in cols_034
        # conversational.py must not query regime_label from market_breadth_daily
        src = (Path(__file__).parents[1] / "api" / "explain" / "conversational.py").read_text()
        assert "regime_label FROM market_breadth_daily" not in src

    def test_portfolio_risk_daily_var_column(self):
        cols = self._migration_cols("073_portfolio_risk_daily.sql")
        assert "var_99_1d" in cols
        assert "var_1d_99" not in cols
        # intelligence_api.py must use var_99_1d in the query
        src = (Path(__file__).parents[1] / "api" / "universal" / "intelligence_api.py").read_text()
        assert "var_99_1d" in src

    def test_company_intelligence_archive_impact_score(self):
        cols = self._migration_cols("077_company_intelligence_archive.sql")
        assert "impact_score" in cols
        assert "intelligence_score" not in cols

    def test_signal_performance_archive_batting_average(self):
        cols = self._migration_cols("075_signal_performance_archive.sql")
        assert "batting_average" in cols


# ---------------------------------------------------------------------------
# Navigation & new panels
# ---------------------------------------------------------------------------

class TestNavigationAndPanels:

    def test_universe_screen_js_exists(self):
        assert (WEBAPP / "js" / "panels" / "universe_screen.js").exists()

    def test_system_status_js_exists(self):
        assert (WEBAPP / "js" / "panels" / "system_status.js").exists()

    def test_favicon_svg_exists(self):
        assert (WEBAPP / "favicon.svg").exists()
        content = (WEBAPP / "favicon.svg").read_text()
        assert "<svg" in content

    def test_favicon_linked_in_html(self):
        html = (WEBAPP / "index.html").read_text()
        assert "favicon.svg" in html

    def test_favicon_endpoint_200(self):
        with TestClient(app) as client:
            r = client.get("/favicon.ico")
        assert r.status_code == 200

    def test_dashboard_js_calls_load_universe_screen(self):
        src = (WEBAPP / "js" / "dashboard.js").read_text()
        assert "loadUniverseScreen" in src, \
            "dashboard.js PANELS.opportunities must call loadUniverseScreen"

    def test_dashboard_js_calls_load_system_status(self):
        src = (WEBAPP / "js" / "dashboard.js").read_text()
        assert "loadSystemStatus" in src, \
            "dashboard.js PANELS.pipeline must call loadSystemStatus"

    def test_new_panel_scripts_in_html(self):
        html = (WEBAPP / "index.html").read_text()
        assert "universe_screen.js" in html
        assert "system_status.js" in html
