"""Prompt A3 tests: chatbot endpoint, JS files, primary driver fix, version bump."""
from __future__ import annotations

import inspect
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from api.main import app

WEBAPP = Path(__file__).resolve().parents[1] / "api" / "webapp"
JS_DIR = WEBAPP / "js"


# ---------------------------------------------------------------------------
# LLM Chat endpoint
# ---------------------------------------------------------------------------

class TestLLMChatEndpoint:

    def test_llm_chat_returns_200_when_unavailable(self):
        with TestClient(app) as client:
            r = client.post("/llm/chat", json={"message": "What is the market regime?"})
        assert r.status_code == 200

    def test_llm_chat_has_reply_field(self):
        with TestClient(app) as client:
            r = client.post("/llm/chat", json={"message": "Explain EIS score."})
        assert "reply" in r.json()

    def test_llm_chat_reply_is_string(self):
        with TestClient(app) as client:
            r = client.post("/llm/chat", json={"message": "What is DAU?"})
        assert isinstance(r.json()["reply"], str)

    def test_llm_chat_rejects_empty_message(self):
        with TestClient(app) as client:
            r = client.post("/llm/chat", json={"message": ""})
        assert r.status_code == 422

    def test_llm_chat_rejects_oversized_message(self):
        with TestClient(app) as client:
            r = client.post("/llm/chat", json={"message": "x" * 1001})
        assert r.status_code == 422

    def test_llm_chat_endpoint_exists_in_main(self):
        import api.main as m
        src = inspect.getsource(m)
        assert "/llm/chat" in src

    def test_llm_exempt_from_rate_limit(self):
        from api.security import _RATE_LIMIT_EXEMPT_PREFIXES
        assert any(p.startswith("/llm") for p in _RATE_LIMIT_EXEMPT_PREFIXES)


# ---------------------------------------------------------------------------
# Chatbot JS file
# ---------------------------------------------------------------------------

class TestChatbotJS:

    def test_axiom_chatbot_js_exists(self):
        assert (JS_DIR / "axiom_chatbot.js").exists()

    def test_chatbot_has_floating_button(self):
        src = (JS_DIR / "axiom_chatbot.js").read_text()
        assert "axiom-chatbot-btn" in src

    def test_chatbot_posts_to_llm_chat(self):
        src = (JS_DIR / "axiom_chatbot.js").read_text()
        assert "/llm/chat" in src

    def test_chatbot_script_in_index_html(self):
        html = (WEBAPP / "index.html").read_text()
        assert "axiom_chatbot.js" in html

    def test_index_html_uses_v101(self):
        html = (WEBAPP / "index.html").read_text()
        assert "?v=101" in html
        assert "?v=100" not in html

    def test_chatbot_handles_errors_gracefully(self):
        src = (JS_DIR / "axiom_chatbot.js").read_text()
        assert "catch" in src
        assert "Network error" in src


# ---------------------------------------------------------------------------
# Primary driver fix
# ---------------------------------------------------------------------------

class TestPrimaryDriverFix:

    def test_symbol_intelligence_shows_factor_composite_fallback(self):
        src = (JS_DIR / "panels" / "symbol_intelligence.js").read_text()
        assert "Factor Composite" in src

    def test_symbol_intelligence_no_empty_evidence_message(self):
        src = (JS_DIR / "panels" / "symbol_intelligence.js").read_text()
        assert "No evidence items available" not in src

    def test_symbol_intelligence_pipeline_run_message(self):
        src = (JS_DIR / "panels" / "symbol_intelligence.js").read_text()
        assert "Evidence builds with each pipeline run" in src

    def test_explain_routes_derives_primary_driver_from_engines(self):
        src = Path(__file__).parents[1].joinpath("api", "explain", "explain_routes.py").read_text()
        assert "primary_driver" in src
        assert "engine_scores" in src


# ---------------------------------------------------------------------------
# Version bump
# ---------------------------------------------------------------------------

class TestVersionBump:

    def test_fastapi_version_is_101(self):
        with TestClient(app) as client:
            r = client.get("/openapi.json")
        assert r.json()["info"]["version"] == "1.0.1"

    def test_config_client_version_is_101(self):
        with TestClient(app) as client:
            r = client.get("/config/client")
        assert r.json()["version"] == "1.0.1"

    def test_pipeline_fix_friendly_error(self):
        src = (JS_DIR / "panels" / "system_status.js").read_text()
        assert "Pipeline trigger failed" in src
        assert "err.message" not in src


# ---------------------------------------------------------------------------
# Morning briefing config fix
# ---------------------------------------------------------------------------

class TestMorningBriefingConfigFix:

    def test_morning_briefing_imports_config(self):
        src = Path(__file__).parents[1].joinpath("api", "jobs", "morning_briefing.py").read_text()
        assert "from api import config" in src or "from api import" in src and "config" in src.split("from api import")[1].split("\n")[0]

    def test_morning_briefing_get_returns_200(self):
        with TestClient(app) as client:
            r = client.get("/jobs/briefing/morning")
        assert r.status_code == 200

    def test_morning_briefing_has_sri(self):
        with TestClient(app) as client:
            r = client.get("/jobs/briefing/morning")
        data = r.json()
        assert "systemic_risk_index" in data
        assert isinstance(data["systemic_risk_index"], (int, float))
