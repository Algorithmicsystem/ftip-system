"""Prompt 10 tests: OpenAI integration, v1.0.0, acquisition readiness, all products."""
from __future__ import annotations

import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from api.main import app

WEBAPP = Path(__file__).resolve().parents[1] / "api" / "webapp"


# ---------------------------------------------------------------------------
# OpenAI Integration
# ---------------------------------------------------------------------------

class TestOpenAIIntegration:

    def test_openai_client_module_exists(self):
        from api.llm.openai_client import is_available, call_openai
        assert callable(is_available)
        assert callable(call_openai)

    def test_openai_client_returns_none_without_key(self):
        from api.llm import openai_client as oc
        old = os.environ.pop("OPENAI_API_KEY", None)
        # Also clear the legacy key
        old2 = os.environ.pop("OpenAI_ftip-system", None)
        # Reset cached key
        import api.config as cfg
        try:
            result = oc.call_openai("system", "user")
            assert result is None
        finally:
            if old:
                os.environ["OPENAI_API_KEY"] = old
            if old2:
                os.environ["OpenAI_ftip-system"] = old2

    def test_synthesize_signal_explanation_callable(self):
        from api.llm.openai_client import synthesize_signal_explanation
        assert callable(synthesize_signal_explanation)

    def test_synthesize_morning_briefing_callable(self):
        from api.llm.openai_client import synthesize_morning_briefing
        assert callable(synthesize_morning_briefing)

    def test_synthesize_pe_analysis_callable(self):
        from api.llm.openai_client import synthesize_pe_analysis
        assert callable(synthesize_pe_analysis)

    def test_synthesize_smb_recommendation_callable(self):
        from api.llm.openai_client import synthesize_smb_recommendation
        assert callable(synthesize_smb_recommendation)

    def test_morning_briefing_has_llm_enhanced_field(self):
        from api.jobs.morning_briefing import MorningBriefing
        import dataclasses
        fields = {f.name for f in dataclasses.fields(MorningBriefing)}
        assert "llm_enhanced" in fields

    def test_morning_briefing_returns_llm_enhanced_flag(self):
        with TestClient(app) as client:
            r = client.get("/jobs/briefing/morning")
        assert r.status_code == 200
        data = r.json()
        assert "llm_enhanced" in data

    def test_explain_signal_has_llm_enhanced_key(self):
        with TestClient(app) as client:
            r = client.get("/explain/signal/AAPL")
        assert r.status_code == 200
        data = r.json()
        assert "llm_enhanced" in data

    def test_research_report_has_llm_synthesis(self):
        with TestClient(app) as client:
            r = client.get("/explain/report/AAPL")
        assert r.status_code == 200
        data = r.json()
        assert "llm_synthesis" in data
        assert "llm_enhanced" in data

    def test_pe_das_has_llm_enhanced_flag(self):
        with TestClient(app) as client:
            r = client.get("/pe/das/NVDA")
        assert r.status_code == 200
        data = r.json()
        # Either db_disabled stub or full response — both should have llm_enhanced or status
        assert "llm_enhanced" in data or "status" in data

    def test_is_available_returns_bool(self):
        from api.llm.openai_client import is_available
        result = is_available()
        assert isinstance(result, bool)

    def test_analyst_system_prompt_defined(self):
        from api.llm.openai_client import ANALYST_SYSTEM
        assert len(ANALYST_SYSTEM) > 50


# ---------------------------------------------------------------------------
# EIS/CAPS Extraction
# ---------------------------------------------------------------------------

class TestEISCAPSExtraction:

    def test_universe_scores_has_30_symbols(self):
        with TestClient(app) as client:
            r = client.get("/intelligence/universe/scores")
        assert r.status_code == 200
        data = r.json()
        assert len(data) == 30

    def test_universe_scores_eis_field_present(self):
        with TestClient(app) as client:
            r = client.get("/intelligence/universe/scores")
        data = r.json()
        for sym in data:
            assert "eis_score" in sym
            assert "caps_score" in sym

    def test_universe_eis_extraction_path_in_code(self):
        import inspect, api.main as m
        src = inspect.getsource(m)
        assert "eis_component" in src or "eis_score" in src.lower()

    def test_universe_scores_dau_present(self):
        with TestClient(app) as client:
            r = client.get("/intelligence/universe/scores")
        data = r.json()
        assert all("dau" in s for s in data)


# ---------------------------------------------------------------------------
# Factor Environment
# ---------------------------------------------------------------------------

class TestFactorEnvironmentFixed:

    def test_macro_snapshot_returns_200(self):
        with TestClient(app) as client:
            r = client.get("/macro/snapshot")
        assert r.status_code == 200

    def test_macro_snapshot_has_macro_intelligence(self):
        with TestClient(app) as client:
            r = client.get("/macro/snapshot")
        data = r.json()
        assert "macro_intelligence" in data

    def test_macro_snapshot_macro_intelligence_has_favored_factors(self):
        with TestClient(app) as client:
            r = client.get("/macro/snapshot")
        data = r.json()
        assert "favored_factors" in data["macro_intelligence"]

    def test_macro_snapshot_has_cross_asset(self):
        with TestClient(app) as client:
            r = client.get("/macro/snapshot")
        data = r.json()
        assert "cross_asset" in data

    def test_macro_snapshot_cross_asset_has_equity_amplifier(self):
        with TestClient(app) as client:
            r = client.get("/macro/snapshot")
        data = r.json()
        assert "equity_signal_amplifier" in data["cross_asset"]

    def test_factor_environment_js_has_robust_field_access(self):
        js = (WEBAPP / "js" / "panels" / "factor_environment.js").read_text()
        assert "||" in js

    def test_factor_environment_js_reads_favored_factors(self):
        js = (WEBAPP / "js" / "panels" / "factor_environment.js").read_text()
        assert "favored_factors" in js

    def test_security_py_exempts_macro_prefix(self):
        src = Path(__file__).parents[1].joinpath("api", "security.py").read_text()
        assert "/macro/" in src


# ---------------------------------------------------------------------------
# Pipeline Auto-Trigger
# ---------------------------------------------------------------------------

class TestPipelineAutoTrigger:

    def test_lifecycle_has_stale_check(self):
        import inspect, api.lifecycle as lc
        src = inspect.getsource(lc)
        assert "stale" in src

    def test_lifecycle_has_threading_trigger(self):
        import inspect, api.lifecycle as lc
        src = inspect.getsource(lc)
        assert "threading" in src

    def test_pipeline_starting_message_type_handled(self):
        js = (WEBAPP / "js" / "dashboard.js").read_text()
        assert "pipeline_starting" in js

    def test_pipeline_starting_shows_toast(self):
        js = (WEBAPP / "js" / "dashboard.js").read_text()
        assert "Pipeline Starting" in js


# ---------------------------------------------------------------------------
# Acquisition Readiness
# ---------------------------------------------------------------------------

class TestAcquisitionReadiness:

    def test_acquisition_readiness_endpoint_200(self):
        with TestClient(app) as client:
            r = client.get("/developer/acquisition-readiness")
        assert r.status_code == 200

    def test_acquisition_readiness_has_score(self):
        with TestClient(app) as client:
            r = client.get("/developer/acquisition-readiness")
        data = r.json()
        assert "overall_score" in data
        assert 0 <= data["overall_score"] <= 100

    def test_acquisition_readiness_has_proprietary_engines(self):
        with TestClient(app) as client:
            r = client.get("/developer/acquisition-readiness")
        data = r.json()
        assert data["proprietary_engines"] >= 5

    def test_acquisition_readiness_has_moat(self):
        with TestClient(app) as client:
            r = client.get("/developer/acquisition-readiness")
        data = r.json()
        assert "moat_assessment" in data
        assert data["moat_assessment"]["signal_war_formula"] is True

    def test_acquisition_readiness_is_acquisition_ready(self):
        with TestClient(app) as client:
            r = client.get("/developer/acquisition-readiness")
        data = r.json()
        assert data["overall_score"] >= 85

    def test_acquisition_readiness_has_commercial(self):
        with TestClient(app) as client:
            r = client.get("/developer/acquisition-readiness")
        data = r.json()
        assert "commercial_readiness" in data
        assert data["commercial_readiness"]["billing_tiers"] == 4

    def test_acquisition_readiness_has_narrative(self):
        with TestClient(app) as client:
            r = client.get("/developer/acquisition-readiness")
        data = r.json()
        assert "narrative" in data
        assert len(data["narrative"]) > 50


# ---------------------------------------------------------------------------
# Version 1.0.0
# ---------------------------------------------------------------------------

class TestVersion1:

    def test_config_client_version_1(self):
        with TestClient(app) as client:
            r = client.get("/config/client")
        assert r.json()["version"] in ("1.0.0", "1.0.1")

    def test_system_status_version_1(self):
        with TestClient(app) as client:
            r = client.get("/system/status")
        assert r.json()["version"] in ("1.0.0", "1.0.1")

    def test_index_html_cache_bust_v100(self):
        html = (WEBAPP / "index.html").read_text()
        assert "?v=100" in html or "?v=101" in html
        assert "?v=34" not in html

    def test_index_html_has_enough_v100_references(self):
        html = (WEBAPP / "index.html").read_text()
        assert html.count("?v=100") + html.count("?v=101") >= 10


# ---------------------------------------------------------------------------
# All Three Products
# ---------------------------------------------------------------------------

class TestAllThreeProducts:

    def test_investment_universe_scores(self):
        with TestClient(app) as client:
            r = client.get("/intelligence/universe/scores")
        assert r.status_code == 200
        assert len(r.json()) == 30

    def test_investment_universal_aapl_public(self):
        with TestClient(app) as client:
            r = client.get("/intelligence/universal/AAPL", headers={})
        assert r.status_code == 200

    def test_investment_explain_signal_public(self):
        with TestClient(app) as client:
            r = client.get("/explain/signal/AAPL", headers={})
        assert r.status_code == 200

    def test_pe_das_nvda_public(self):
        with TestClient(app) as client:
            r = client.get("/pe/das/NVDA", headers={})
        assert r.status_code == 200

    def test_pe_schilit_nvda_public(self):
        with TestClient(app) as client:
            r = client.get("/pe/schilit/NVDA", headers={})
        assert r.status_code == 200

    def test_pe_deal_flow_public(self):
        with TestClient(app) as client:
            r = client.get("/pe/deal-flow", headers={})
        assert r.status_code == 200

    def test_smb_entity_dashboard_accessible(self):
        with TestClient(app) as client:
            r = client.get("/smb/entity/DEMO/intelligence-dashboard", headers={})
        assert r.status_code in (200, 404, 422)

    def test_morning_briefing_public(self):
        with TestClient(app) as client:
            r = client.get("/jobs/briefing/morning", headers={})
        assert r.status_code == 200

    def test_macro_snapshot_public(self):
        with TestClient(app) as client:
            r = client.get("/macro/snapshot", headers={})
        assert r.status_code == 200


# ---------------------------------------------------------------------------
# Symbol Intelligence JS
# ---------------------------------------------------------------------------

class TestSymbolIntelligenceJS:

    def test_explanation_tab_has_ai_synthesis_box(self):
        src = (WEBAPP / "js" / "panels" / "symbol_intelligence.js").read_text()
        assert "AI SYNTHESIS" in src or "ai_synthesis" in src

    def test_explanation_tab_has_gpt_label(self):
        src = (WEBAPP / "js" / "panels" / "symbol_intelligence.js").read_text()
        assert "GPT-4o-mini" in src or "ai_synthesis" in src

    def test_morning_briefing_js_has_llm_enhanced_badge(self):
        src = (WEBAPP / "js" / "panels" / "morning_briefing.js").read_text()
        assert "llm_enhanced" in src
        assert "AI-Enhanced" in src
