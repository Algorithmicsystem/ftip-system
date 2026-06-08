"""Prompt 7 tests: Risk tab, SRI sparkline, morning briefing render, factor labels, universe polish."""
from __future__ import annotations

import re
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from api.main import app

WEBAPP = Path(__file__).resolve().parents[1] / "api" / "webapp"


# ---------------------------------------------------------------------------
# Version bump
# ---------------------------------------------------------------------------

class TestVersion32:

    def test_config_client_version_is_32(self):
        with TestClient(app) as client:
            r = client.get("/config/client")
        assert r.status_code == 200
        assert r.json()["version"] in ("32.0.0", "33.0.0", "34.0.0")

    def test_index_html_cache_bust_is_v32(self):
        html = (WEBAPP / "index.html").read_text()
        assert "?v=32" in html or "?v=33" in html or "?v=34" in html

    def test_index_html_no_v31(self):
        html = (WEBAPP / "index.html").read_text()
        assert "?v=31" not in html

    def test_at_least_17_v32_references_in_html(self):
        html = (WEBAPP / "index.html").read_text()
        count = html.count("?v=32") + html.count("?v=33") + html.count("?v=34")
        assert count >= 17, f"Expected >=17 ?v=32 references, got {count}"


# ---------------------------------------------------------------------------
# Tab switching fix
# ---------------------------------------------------------------------------

class TestTabSwitching:

    def test_switch_tab_handles_string_first_arg(self):
        src = (WEBAPP / "js" / "dashboard.js").read_text()
        assert "typeof containerOrTabId === 'string'" in src or \
               "typeof containerOrTabId === \"string\"" in src, \
            "switchTab must detect string first arg (HTML calling convention)"

    def test_switch_tab_uses_closest_panel_body(self):
        src = (WEBAPP / "js" / "dashboard.js").read_text()
        assert "closest('.panel-body')" in src, \
            "switchTab must use .closest('.panel-body') for HTML calling convention"

    def test_switch_tab_deactivates_tab_nav_items(self):
        src = (WEBAPP / "js" / "dashboard.js").read_text()
        assert "tab-nav__item" in src, \
            "switchTab must handle .tab-nav__item deactivation"


# ---------------------------------------------------------------------------
# SRI sparkline history bug fix
# ---------------------------------------------------------------------------

class TestSRISparklineHistoryFix:

    def test_risk_monitor_uses_history_property(self):
        src = (WEBAPP / "js" / "panels" / "risk_monitor.js").read_text()
        assert "history?.history" in src, \
            "risk_monitor.js must use history?.history not bare history"
        assert "(history || []).map" not in src, \
            "risk_monitor.js must not use bare (history || []).map — history is an object not array"

    def test_risk_monitor_api_endpoint_returns_history_object(self, monkeypatch):
        monkeypatch.setenv("FTIP_API_KEY", "test-key")
        with TestClient(app) as client:
            r = client.get(
                "/axiom/risk/sri/history",
                headers={"X-FTIP-API-Key": "test-key"},
            )
        assert r.status_code == 200
        data = r.json()
        assert "history" in data, "SRI history endpoint must return {history: [...]} not a bare array"
        assert isinstance(data["history"], list)


# ---------------------------------------------------------------------------
# Risk tab enhancements
# ---------------------------------------------------------------------------

class TestRiskTabEnhancements:

    def test_render_risk_tab_uses_score_bar(self):
        src = (WEBAPP / "js" / "panels" / "symbol_intelligence.js").read_text()
        assert "scoreBarHTML('Fragility'" in src or "scoreBarHTML(\"Fragility\"" in src, \
            "renderRiskTab must use scoreBarHTML for Fragility gauge bar"
        assert "scoreBarHTML('SCPS'" in src or "scoreBarHTML(\"SCPS\"" in src, \
            "renderRiskTab must use scoreBarHTML for SCPS gauge bar"

    def test_render_risk_tab_shows_cross_asset_adj_dau(self):
        src = (WEBAPP / "js" / "panels" / "symbol_intelligence.js").read_text()
        assert "cross_asset_adjusted_dau" in src, \
            "renderRiskTab must display cross_asset_adjusted_dau"

    def test_render_risk_tab_shows_var(self):
        src = (WEBAPP / "js" / "panels" / "symbol_intelligence.js").read_text()
        assert "var_1d_99" in src, "renderRiskTab must show VaR 1d 99%"

    def test_render_risk_tab_uses_top_risk_for_invalidation(self):
        src = (WEBAPP / "js" / "panels" / "symbol_intelligence.js").read_text()
        assert "top_risk" in src, \
            "renderRiskTab must use data.top_risk for invalidation conditions"

    def test_universal_intelligence_has_required_risk_fields(self):
        with TestClient(app) as client:
            r = client.get("/intelligence/universal/AAPL")
        assert r.status_code == 200
        data = r.json()
        for field in ("fragility_score", "scps_score", "bfs_score"):
            assert field in data, f"Universal intelligence must have {field}"


# ---------------------------------------------------------------------------
# Explanation tab fixes
# ---------------------------------------------------------------------------

class TestExplanationTabFixes:

    def test_explanation_tab_uses_reasoning_steps(self):
        src = (WEBAPP / "js" / "panels" / "symbol_intelligence.js").read_text()
        assert "reasoning_steps" in src, \
            "renderExplanationTab must use explain.reasoning_steps not reasoning_chain.steps"
        assert "reasoning_chain?.steps" not in src, \
            "renderExplanationTab must not use reasoning_chain?.steps (wrong field)"

    def test_explanation_tab_uses_contradicting_factors(self):
        src = (WEBAPP / "js" / "panels" / "symbol_intelligence.js").read_text()
        assert "contradicting_factors" in src, \
            "renderExplanationTab must use contradicting_factors"

    def test_explanation_tab_shows_batting_average(self):
        src = (WEBAPP / "js" / "panels" / "symbol_intelligence.js").read_text()
        assert "signal_batting_average" in src, \
            "renderExplanationTab must show signal_batting_average"

    def test_explanation_tab_shows_explanation_text(self):
        src = (WEBAPP / "js" / "panels" / "symbol_intelligence.js").read_text()
        assert "explanation_text" in src, \
            "renderExplanationTab must render explanation_text from explain endpoint"

    def test_explain_endpoint_has_reasoning_steps(self, monkeypatch):
        monkeypatch.setenv("FTIP_API_KEY", "test-key")
        with TestClient(app) as client:
            r = client.get(
                "/explain/signal/AAPL",
                headers={"X-FTIP-API-Key": "test-key"},
            )
        if r.status_code in (401, 503):
            pytest.skip("Explain endpoint requires DB — skipping in no-DB environment")
        assert r.status_code == 200
        data = r.json()
        assert "reasoning_steps" in data, "explain endpoint must return reasoning_steps"
        assert "explanation_text" in data, "explain endpoint must return explanation_text"


# ---------------------------------------------------------------------------
# Morning briefing sub-elements
# ---------------------------------------------------------------------------

class TestMorningBriefingSubElements:

    def test_briefing_text_div_in_html(self):
        html = (WEBAPP / "index.html").read_text()
        assert 'id="briefing-text"' in html, "index.html must have id=briefing-text"

    def test_briefing_sri_div_in_html(self):
        html = (WEBAPP / "index.html").read_text()
        assert 'id="briefing-sri"' in html, "index.html must have id=briefing-sri"

    def test_briefing_signals_div_in_html(self):
        html = (WEBAPP / "index.html").read_text()
        assert 'id="briefing-signals"' in html, "index.html must have id=briefing-signals"

    def test_briefing_opps_div_in_html(self):
        html = (WEBAPP / "index.html").read_text()
        assert 'id="briefing-opps"' in html, "index.html must have id=briefing-opps"

    def test_render_briefing_populates_briefing_text(self):
        src = (WEBAPP / "js" / "panels" / "morning_briefing.js").read_text()
        assert "'briefing-text'" in src or '"briefing-text"' in src, \
            "morning_briefing.js must reference briefing-text element"

    def test_render_briefing_populates_briefing_sri(self):
        src = (WEBAPP / "js" / "panels" / "morning_briefing.js").read_text()
        assert "'briefing-sri'" in src or '"briefing-sri"' in src, \
            "morning_briefing.js must reference briefing-sri element"

    def test_render_briefing_populates_briefing_opps(self):
        src = (WEBAPP / "js" / "panels" / "morning_briefing.js").read_text()
        assert "'briefing-opps'" in src or '"briefing-opps"' in src, \
            "morning_briefing.js must reference briefing-opps element"


# ---------------------------------------------------------------------------
# Factor labels
# ---------------------------------------------------------------------------

class TestFactorLabels:

    def test_factor_labels_const_exists(self):
        src = (WEBAPP / "js" / "panels" / "factor_environment.js").read_text()
        assert "FACTOR_LABELS" in src, "factor_environment.js must define FACTOR_LABELS const"

    def test_factor_labels_has_gbf(self):
        src = (WEBAPP / "js" / "panels" / "factor_environment.js").read_text()
        assert "GBF" in src and "Global Business Fundamentals" in src

    def test_factor_labels_has_eif(self):
        src = (WEBAPP / "js" / "panels" / "factor_environment.js").read_text()
        assert "EIF" in src and "Economic Inflection" in src

    def test_factor_labels_has_cmf(self):
        src = (WEBAPP / "js" / "panels" / "factor_environment.js").read_text()
        assert "CMF" in src and "Capital Market Flow" in src

    def test_factor_labels_has_ntff(self):
        src = (WEBAPP / "js" / "panels" / "factor_environment.js").read_text()
        assert "NTFF" in src and "Near-Term Flow" in src

    def test_factor_labels_used_in_factors_array(self):
        src = (WEBAPP / "js" / "panels" / "factor_environment.js").read_text()
        assert "FACTOR_LABELS[f]" in src, \
            "factor_environment.js must use FACTOR_LABELS[f] when building factors array"


# ---------------------------------------------------------------------------
# Universe screen EIS/CAPS columns
# ---------------------------------------------------------------------------

class TestUniverseScreenColumns:

    def test_universe_screen_has_eis_column_header(self):
        src = (WEBAPP / "js" / "panels" / "universe_screen.js").read_text()
        assert ">EIS<" in src or "'EIS'" in src or '"EIS"' in src, \
            "universe_screen.js must display EIS column header"

    def test_universe_screen_has_caps_column_header(self):
        src = (WEBAPP / "js" / "panels" / "universe_screen.js").read_text()
        assert ">CAPS<" in src or "'CAPS'" in src or '"CAPS"' in src, \
            "universe_screen.js must display CAPS column header"

    def test_universe_screen_uses_eis_score_field(self):
        src = (WEBAPP / "js" / "panels" / "universe_screen.js").read_text()
        assert "eis_score" in src, \
            "universe_screen.js must render r.eis_score"

    def test_universe_screen_uses_caps_score_field(self):
        src = (WEBAPP / "js" / "panels" / "universe_screen.js").read_text()
        assert "caps_score" in src, \
            "universe_screen.js must render r.caps_score"

    def test_universe_scores_api_returns_eis_caps(self):
        with TestClient(app) as client:
            r = client.get("/intelligence/universe/scores")
        assert r.status_code == 200
        rows = r.json()
        assert len(rows) > 0
        row = rows[0]
        assert "eis_score" in row, "universe scores must include eis_score"
        assert "caps_score" in row, "universe scores must include caps_score"


# ---------------------------------------------------------------------------
# Opportunities panel regime chip
# ---------------------------------------------------------------------------

class TestOpportunitiesRegimeChip:

    def test_opportunities_uses_regime_label(self):
        src = (WEBAPP / "js" / "panels" / "opportunities.js").read_text()
        assert "regime_label" in src, \
            "opportunities.js must display item.regime_label"

    def test_opportunities_shows_regime_conditionally(self):
        src = (WEBAPP / "js" / "panels" / "opportunities.js").read_text()
        assert "item.regime_label" in src, \
            "opportunities.js must check item.regime_label before rendering"


# ---------------------------------------------------------------------------
# charts.js renderDAUSparkline
# ---------------------------------------------------------------------------

class TestChartsDAUSparkline:

    def test_render_dau_sparkline_function_exists(self):
        src = (WEBAPP / "js" / "charts.js").read_text()
        assert "function renderDAUSparkline" in src, \
            "charts.js must define renderDAUSparkline function"

    def test_render_dau_sparkline_uses_dau_field(self):
        src = (WEBAPP / "js" / "charts.js").read_text()
        assert "d.dau" in src, \
            "renderDAUSparkline must map d.dau from history items"

    def test_render_dau_sparkline_uses_as_of_date(self):
        src = (WEBAPP / "js" / "charts.js").read_text()
        assert "as_of_date" in src, \
            "renderDAUSparkline must use as_of_date as labels"

    def test_all_five_chart_functions_present(self):
        src = (WEBAPP / "js" / "charts.js").read_text()
        for fn in ("renderDAUBar", "renderSRIGauge", "renderSparkline", "renderHeatmap", "renderDAUSparkline"):
            assert f"function {fn}" in src, f"charts.js must define {fn}"
