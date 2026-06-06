"""Phase 26 tests: /config/client endpoint, panel containers, score column audit."""
from __future__ import annotations

import os
import re

import pytest
from fastapi.testclient import TestClient

from api.main import app


# ---------------------------------------------------------------------------
# A1: /config/client endpoint
# ---------------------------------------------------------------------------

class TestConfigClientEndpoint:

    def test_config_client_endpoint_returns_api_key(self, monkeypatch):
        monkeypatch.setenv("FTIP_API_KEY", "test-key-123")
        with TestClient(app) as client:
            r = client.get("/config/client")
        assert r.status_code == 200
        data = r.json()
        assert data["api_key"] == "test-key-123"
        assert "env" in data
        assert "version" in data

    def test_config_client_requires_no_auth(self):
        """Endpoint must return 200 with no API key header at all."""
        with TestClient(app) as client:
            r = client.get("/config/client")
        assert r.status_code == 200

    def test_config_client_empty_key_when_unset(self):
        """api_key must be empty string (not None) when FTIP_API_KEY is unset."""
        with TestClient(app) as client:
            r = client.get("/config/client")
        assert r.status_code == 200
        assert r.json()["api_key"] == ""

    def test_config_client_version_is_26(self):
        with TestClient(app) as client:
            r = client.get("/config/client")
        assert r.json()["version"] == "26.0.0"


# ---------------------------------------------------------------------------
# B/C: Panel container IDs present in index.html
# ---------------------------------------------------------------------------

class TestPanelContainersAllPresent:

    def _load_html(self) -> str:
        path = os.path.join(
            os.path.dirname(__file__), "..", "api", "webapp", "index.html"
        )
        with open(path) as f:
            return f.read()

    def test_panel_containers_all_present(self):
        html = self._load_html()
        expected_ids = [
            "panel-briefing",
            "panel-opportunities",
            "panel-symbol",
            "panel-risk",
            "panel-factors",
            "panel-pipeline",
            "view-dashboard",
            "view-pe",
            "view-smb",
        ]
        for eid in expected_ids:
            assert f'id="{eid}"' in html, f"Missing element id={eid} in index.html"

    def test_script_tags_have_cache_busting(self):
        html = self._load_html()
        scripts = re.findall(r'<script src="(/app/static/[^"]+)"', html)
        assert len(scripts) >= 8, "Expected at least 8 local script tags"
        for src in scripts:
            assert "?v=" in src, f"Script {src} missing cache-busting ?v= parameter"

    def test_dashboard_panels_key_is_opportunities_not_universe(self):
        """dashboard.js PANELS must use 'opportunities' key (matches data-panel in HTML)."""
        path = os.path.join(
            os.path.dirname(__file__), "..", "api", "webapp", "js", "dashboard.js"
        )
        with open(path) as f:
            src = f.read()
        assert "universe:" not in src, "PANELS key 'universe' must be renamed to 'opportunities'"
        assert "opportunities:" in src


# ---------------------------------------------------------------------------
# D: AXIOM scores columns match migration schema
# ---------------------------------------------------------------------------

class TestAxiomScoresColumnsMigration:

    def test_axiom_scores_columns_match_migration(self):
        """026_axiom_phase3_history.sql must define payload and deployable_alpha_utility columns."""
        migrations_dir = os.path.join(
            os.path.dirname(__file__), "..", "api", "migrations"
        )
        path = os.path.join(migrations_dir, "026_axiom_phase3_history.sql")
        if not os.path.isfile(path):
            pytest.skip("Migration 026 not present")
        with open(path) as f:
            sql = f.read()
        names: set = set()
        for m in re.finditer(
            r"^\s{4}(\w+)\s+(?:TEXT|NUMERIC|DATE|INT|BOOLEAN|UUID|JSONB|TIMESTAMPTZ|DOUBLE\s+PRECISION)",
            sql,
            re.MULTILINE | re.IGNORECASE,
        ):
            names.add(m.group(1).lower())
        assert "payload" in names, f"payload column missing from migration 026, got {names}"
        assert "deployable_alpha_utility" in names, (
            f"deployable_alpha_utility missing from migration 026, got {names}"
        )
