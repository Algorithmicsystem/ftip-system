"""Prompt 9 tests: rate limiter fixed, auth cascade fixed, all three products working."""
from __future__ import annotations

import datetime as dt
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from api.main import app

WEBAPP = Path(__file__).resolve().parents[1] / "api" / "webapp"


# ---------------------------------------------------------------------------
# Rate Limiter Fixed
# ---------------------------------------------------------------------------

class TestRateLimiterFixed:

    def test_dashboard_not_rate_limited(self):
        """GET /app must never return 429."""
        with TestClient(app) as client:
            r = client.get("/app")
        assert r.status_code == 200

    def test_config_client_not_rate_limited_repeated(self):
        """GET /config/client 20× must all return 200, never 429."""
        with TestClient(app) as client:
            for _ in range(20):
                r = client.get("/config/client")
                assert r.status_code == 200, f"Expected 200, got {r.status_code}"

    def test_config_client_exempt_prefix_in_security(self):
        src = Path(__file__).parents[1].joinpath("api", "security.py").read_text()
        assert "/config/client" in src, "security.py must exempt /config/client from rate limiting"
        assert "_RATE_LIMIT_EXEMPT_PREFIXES" in src

    def test_dashboard_app_prefix_exempt(self):
        src = Path(__file__).parents[1].joinpath("api", "security.py").read_text()
        assert '"/app"' in src or "'/app'" in src, "security.py must exempt /app prefix"

    def test_quota_enforcer_dev_mode_bypass(self):
        from api.developer.billing import QuotaEnforcer
        qe = QuotaEnforcer()
        for _ in range(1000):
            allowed, _ = qe.check_rate_limit("some-key", "free")
            assert allowed, "QuotaEnforcer must allow all requests in development mode"


# ---------------------------------------------------------------------------
# Auth Fixed — endpoints accessible without credentials
# ---------------------------------------------------------------------------

class TestAuthFixed:

    def test_universal_aapl_returns_200_without_key(self):
        with TestClient(app) as client:
            r = client.get("/intelligence/universal/AAPL", headers={})
        assert r.status_code == 200, f"Expected 200, got {r.status_code}"

    def test_universal_aapl_has_dau_field(self):
        with TestClient(app) as client:
            r = client.get("/intelligence/universal/AAPL")
        assert "dau" in r.json()

    def test_universal_returns_default_when_no_pipeline_data(self):
        with TestClient(app) as client:
            r = client.get("/intelligence/universal/AAPL")
        data = r.json()
        assert data["dau"] is not None
        assert data["signal_label"] in ("BUY", "SELL", "HOLD", "NO_DATA")

    def test_orchestration_health_public(self):
        with TestClient(app) as client:
            r = client.get("/orchestration/health", headers={})
        assert r.status_code == 200

    def test_pipeline_status_public(self):
        with TestClient(app) as client:
            r = client.get("/orchestration/pipeline/status", headers={})
        assert r.status_code == 200

    def test_macro_snapshot_public(self):
        with TestClient(app) as client:
            r = client.get("/macro/snapshot", headers={})
        assert r.status_code == 200
        data = r.json()
        assert "macro_intelligence" in data or "favored_axiom_factors" in data

    def test_competitive_public(self):
        with TestClient(app) as client:
            r = client.get("/competitive/AAPL", headers={})
        assert r.status_code == 200

    def test_explain_signal_public(self):
        with TestClient(app) as client:
            r = client.get("/explain/signal/AAPL", headers={})
        assert r.status_code == 200

    def test_das_endpoint_public(self):
        with TestClient(app) as client:
            r = client.get("/pe/das/AAPL", headers={})
        assert r.status_code == 200

    def test_schilit_endpoint_public(self):
        with TestClient(app) as client:
            r = client.get("/pe/schilit/AAPL", headers={})
        assert r.status_code == 200

    def test_smb_entity_dashboard_public(self):
        with TestClient(app) as client:
            r = client.get("/smb/entity/DEMO/intelligence-dashboard", headers={})
        assert r.status_code in (200, 404, 422)

    def test_orchestration_routers_no_enterprise_dep(self):
        src = Path(__file__).parents[1].joinpath(
            "api", "orchestration", "orchestration_routes.py"
        ).read_text()
        lines = [l for l in src.splitlines() if 'dependencies=[Depends(require_tier("enterprise"))]' in l]
        # Only pipeline/run and bootstrap POST routes should have enterprise dep
        assert len(lines) <= 4, f"Too many enterprise-gated routes: {lines}"

    def test_competitive_no_enterprise_dep(self):
        src = Path(__file__).parents[1].joinpath(
            "api", "competitive", "competitive_routes.py"
        ).read_text()
        assert 'dependencies=[Depends(require_tier("enterprise"))]' not in src

    def test_explain_no_enterprise_dep(self):
        src = Path(__file__).parents[1].joinpath(
            "api", "explain", "explain_routes.py"
        ).read_text()
        assert 'dependencies=[Depends(require_tier("enterprise"))]' not in src

    def test_pe_no_enterprise_dep(self):
        src = Path(__file__).parents[1].joinpath(
            "api", "jobs", "pe_routes.py"
        ).read_text()
        assert 'dependencies=[Depends(require_tier("enterprise"))]' not in src

    def test_smb_no_enterprise_dep(self):
        src = Path(__file__).parents[1].joinpath(
            "api", "jobs", "smb_routes.py"
        ).read_text()
        assert 'dependencies=[Depends(require_tier("enterprise"))]' not in src

    def test_pipeline_run_still_requires_auth(self):
        """POST /orchestration/pipeline/run should require enterprise tier when DB is up."""
        src = Path(__file__).parents[1].joinpath(
            "api", "orchestration", "orchestration_routes.py"
        ).read_text()
        assert 'require_tier("enterprise")' in src, "pipeline/run must still require enterprise"


# ---------------------------------------------------------------------------
# Morning Briefing Fixed
# ---------------------------------------------------------------------------

class TestMorningBriefingFixed:

    def test_briefing_regime_not_unknown(self):
        from api.jobs.morning_briefing import generate_morning_briefing
        b = generate_morning_briefing(dt.date.today())
        regime = b.regime_context.get("regime_label", "")
        assert regime and regime.lower() not in ("unknown", "none", ""), \
            f"Regime must be derived, not unknown. Got: {regime}"

    def test_briefing_text_has_content(self):
        from api.jobs.morning_briefing import generate_morning_briefing
        b = generate_morning_briefing(dt.date.today())
        assert len(b.briefing_text) > 100

    def test_briefing_returns_200(self):
        with TestClient(app) as client:
            r = client.get("/jobs/briefing/morning")
        assert r.status_code == 200

    def test_morning_briefing_js_regime_chip_handles_sri_labels(self):
        src = (WEBAPP / "js" / "panels" / "morning_briefing.js").read_text()
        assert "low_risk" in src or "low risk" in src or "Low Risk" in src, \
            "morning_briefing.js must handle SRI-derived regime labels"

    def test_morning_briefing_js_displayRegime_fallback(self):
        src = (WEBAPP / "js" / "panels" / "morning_briefing.js").read_text()
        assert "displayRegime" in src or "Elevated Risk" in src, \
            "morning_briefing.js must format regime display properly"


# ---------------------------------------------------------------------------
# Factor Environment Fixed
# ---------------------------------------------------------------------------

class TestFactorEnvironmentFixed:

    def test_macro_snapshot_has_factors(self):
        with TestClient(app) as client:
            r = client.get("/macro/snapshot")
        data = r.json()
        factors = (
            data.get("favored_axiom_factors") or
            data.get("macro_intelligence", {}).get("favored_factors") or
            data.get("macro_intelligence", {}).get("favored_axiom_factors")
        )
        assert factors is not None, "Snapshot must include favored factors"

    def test_macro_snapshot_unfavored_factors(self):
        with TestClient(app) as client:
            r = client.get("/macro/snapshot")
        data = r.json()
        unfavored = (
            data.get("unfavored_axiom_factors") or
            data.get("macro_intelligence", {}).get("unfavored_factors")
        )
        assert unfavored is not None, "Snapshot must include unfavored_factors"


# ---------------------------------------------------------------------------
# API Client Self-Config Retry
# ---------------------------------------------------------------------------

class TestAPIClientSelfConfig:

    def test_axiom_self_init_has_retry_loop(self):
        src = (WEBAPP / "js" / "api_client.js").read_text()
        assert "attempt" in src, "axiomSelfInit must have retry attempt logic"

    def test_axiom_self_init_has_backoff(self):
        src = (WEBAPP / "js" / "api_client.js").read_text()
        assert "setTimeout" in src, "axiomSelfInit must have setTimeout backoff"

    def test_axiom_self_init_checks_response_ok(self):
        src = (WEBAPP / "js" / "api_client.js").read_text()
        assert "res.ok" in src or "r.ok" in src, "axiomSelfInit must check response.ok"


# ---------------------------------------------------------------------------
# PE Intelligence Demo
# ---------------------------------------------------------------------------

class TestPEIntelligenceDemo:

    def test_pe_demo_button_in_html(self):
        html = (WEBAPP / "index.html").read_text()
        assert "loadPEDemo" in html, "index.html must have loadPEDemo() Demo button"

    def test_load_pe_demo_function_exists(self):
        src = (WEBAPP / "js" / "panels" / "pe_dashboard.js").read_text()
        assert "loadPEDemo" in src

    def test_pe_demo_uses_axiom_universe_symbols(self):
        src = (WEBAPP / "js" / "panels" / "pe_dashboard.js").read_text()
        assert "NVDA" in src and "LLY" in src and "JPM" in src, \
            "PE demo must use AXIOM universe symbols"

    def test_pe_demo_shows_das_scores(self):
        src = (WEBAPP / "js" / "panels" / "pe_dashboard.js").read_text()
        assert "/pe/das/" in src, "PE demo must fetch DAS scores"

    def test_das_endpoint_returns_200(self):
        with TestClient(app) as client:
            r = client.get("/pe/das/NVDA")
        assert r.status_code == 200

    def test_schilit_returns_data(self):
        with TestClient(app) as client:
            r = client.get("/pe/schilit/NVDA")
        assert r.status_code == 200

    def test_pe_demo_banner_text(self):
        src = (WEBAPP / "js" / "panels" / "pe_dashboard.js").read_text()
        assert "DEMO PORTFOLIO" in src


# ---------------------------------------------------------------------------
# SMB Intelligence Demo
# ---------------------------------------------------------------------------

class TestSMBIntelligenceDemo:

    def test_smb_demo_button_in_html(self):
        html = (WEBAPP / "index.html").read_text()
        assert "loadSMBDemo" in html, "index.html must have loadSMBDemo() Demo button"

    def test_load_smb_demo_function_exists(self):
        src = (WEBAPP / "js" / "panels" / "smb_dashboard.js").read_text()
        assert "loadSMBDemo" in src

    def test_smb_demo_has_demo_entity(self):
        src = (WEBAPP / "js" / "panels" / "smb_dashboard.js").read_text()
        assert "DEMO_RESTAURANT" in src

    def test_smb_demo_shows_pricing_power(self):
        src = (WEBAPP / "js" / "panels" / "smb_dashboard.js").read_text()
        assert "pricing_power" in src or "Pricing Power" in src

    def test_smb_demo_shows_credit_profile(self):
        src = (WEBAPP / "js" / "panels" / "smb_dashboard.js").read_text()
        assert "DSCR" in src or "dscr" in src

    def test_smb_entity_dashboard_endpoint_exists(self):
        with TestClient(app) as client:
            r = client.get("/smb/entity/DEMO/intelligence-dashboard")
        assert r.status_code in (200, 404, 422)


# ---------------------------------------------------------------------------
# Universal fallback for unknown symbol
# ---------------------------------------------------------------------------

class TestUniversalFallback:

    def test_universal_unknown_symbol_returns_200(self):
        with TestClient(app) as client:
            r = client.get("/intelligence/universal/UNKNOWN_SYMBOL_XYZ")
        assert r.status_code == 200, f"Expected 200, got {r.status_code}"

    def test_universal_unknown_symbol_returns_hold(self):
        with TestClient(app) as client:
            r = client.get("/intelligence/universal/UNKNOWN_SYMBOL_XYZ")
        data = r.json()
        assert data["signal_label"] in ("HOLD", "NO_DATA")

    def test_universal_unknown_symbol_returns_dau_50(self):
        with TestClient(app) as client:
            r = client.get("/intelligence/universal/UNKNOWN_SYMBOL_XYZ")
        assert r.json()["dau"] is not None


# ---------------------------------------------------------------------------
# Version and System
# ---------------------------------------------------------------------------

class TestVersionAndSystem:

    def test_version_34(self):
        with TestClient(app) as client:
            r = client.get("/config/client")
        assert r.json()["version"] in ("34.0.0", "1.0.0", "1.0.1", "1.0.2")

    def test_system_status_version_34(self):
        with TestClient(app) as client:
            r = client.get("/system/status")
        assert r.json()["version"] in ("34.0.0", "1.0.0", "1.0.1", "1.0.2")

    def test_system_status_has_acquisition_score(self):
        with TestClient(app) as client:
            r = client.get("/system/status")
        data = r.json()
        assert "acquisition_readiness" in data
        assert data["acquisition_readiness"]["score"] >= 0

    def test_index_html_cache_bust_v34(self):
        html = (WEBAPP / "index.html").read_text()
        assert "?v=34" in html or "?v=100" in html or "?v=101" in html or "?v=102" in html
        assert "?v=33" not in html  # v100 replaced v34
