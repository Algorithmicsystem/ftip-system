"""Phase 21 tests: World-Class UX and Visualization — structural tests (no browser required)."""
from __future__ import annotations

import os
import re

import pytest

WEBAPP_DIR = os.path.join(os.path.dirname(__file__), "..", "api", "webapp")
JS_DIR     = os.path.join(WEBAPP_DIR, "js")
PANELS_DIR = os.path.join(JS_DIR, "panels")


# ── File existence ────────────────────────────────────────────────────────────

REQUIRED_FILES = [
    os.path.join(WEBAPP_DIR, "design_system.css"),
    os.path.join(WEBAPP_DIR, "index.html"),
    os.path.join(JS_DIR, "api_client.js"),
    os.path.join(JS_DIR, "charts.js"),
    os.path.join(JS_DIR, "dashboard.js"),
    os.path.join(PANELS_DIR, "morning_briefing.js"),
    os.path.join(PANELS_DIR, "opportunities.js"),
    os.path.join(PANELS_DIR, "symbol_intelligence.js"),
    os.path.join(PANELS_DIR, "risk_monitor.js"),
    os.path.join(PANELS_DIR, "factor_environment.js"),
    os.path.join(PANELS_DIR, "pipeline_status.js"),
    os.path.join(PANELS_DIR, "pe_dashboard.js"),
    os.path.join(PANELS_DIR, "smb_dashboard.js"),
]


@pytest.mark.parametrize("path", REQUIRED_FILES)
def test_required_file_exists(path):
    assert os.path.isfile(path), f"Required file missing: {path}"


@pytest.mark.parametrize("path", REQUIRED_FILES)
def test_required_file_nonempty(path):
    assert os.path.getsize(path) > 0, f"File is empty: {path}"


# ── design_system.css ─────────────────────────────────────────────────────────

def _read(path):
    with open(path) as f:
        return f.read()


CSS = lambda: _read(os.path.join(WEBAPP_DIR, "design_system.css"))


def test_css_has_root_variables():
    assert ":root" in CSS()


def test_css_has_bg_primary():
    assert "--bg-primary" in CSS()


def test_css_has_accent_primary():
    assert "--accent-primary" in CSS()


def test_css_has_signal_badges():
    css = CSS()
    assert ".signal-badge" in css


def test_css_has_buy_sell_hold_colors():
    css = CSS()
    assert "--signal-buy" in css
    assert "--signal-sell" in css
    assert "--signal-hold" in css


def test_css_has_metric_card():
    assert ".metric-card" in CSS()


def test_css_has_panel_class():
    assert ".panel" in CSS()


def test_css_has_loading_skeleton():
    assert ".loading-skeleton" in CSS()


def test_css_has_alert_banner():
    assert ".alert-banner" in CSS()


def test_css_has_pipeline_stage():
    assert ".pipeline-stage" in CSS()


def test_css_has_tab_nav():
    assert ".tab-nav" in CSS()


def test_css_has_score_bar():
    assert ".score-bar" in CSS()


def test_css_has_font_mono_variable():
    assert "--font-mono" in CSS()


def test_css_responsive_breakpoint():
    css = CSS()
    assert "@media" in css


# ── index.html ────────────────────────────────────────────────────────────────

HTML = lambda: _read(os.path.join(WEBAPP_DIR, "index.html"))


def test_html_has_doctype():
    assert "<!DOCTYPE html>" in HTML() or "<!doctype html>" in HTML().lower()


def test_html_has_design_system_css():
    assert "design_system.css" in HTML()


def test_html_loads_api_client():
    assert "api_client.js" in HTML()


def test_html_loads_charts():
    assert "charts.js" in HTML()


def test_html_loads_dashboard():
    assert "dashboard.js" in HTML()


def test_html_loads_all_panel_scripts():
    html = HTML()
    panels = [
        "morning_briefing.js", "opportunities.js", "symbol_intelligence.js",
        "risk_monitor.js", "factor_environment.js", "pipeline_status.js",
        "pe_dashboard.js", "smb_dashboard.js",
    ]
    for panel in panels:
        assert panel in html, f"index.html missing script tag for {panel}"


def test_html_has_pipeline_body():
    assert "pipeline-body" in HTML()


def test_html_has_symbol_search():
    assert "symbol-search" in HTML() or "global-symbol-input" in HTML()


def test_html_has_health_dot():
    assert "health-dot" in HTML()


def test_html_has_opportunities_body():
    assert "opportunities-body" in HTML()


def test_html_has_risk_body():
    assert "risk-body" in HTML()


def test_html_has_factors_body():
    assert "factors-body" in HTML()


def test_html_chart_js_loaded():
    assert "chart" in HTML().lower()


# ── api_client.js ─────────────────────────────────────────────────────────────

def test_api_client_has_get_method():
    assert "async get" in _read(os.path.join(JS_DIR, "api_client.js"))


def test_api_client_has_post_method():
    assert "async post" in _read(os.path.join(JS_DIR, "api_client.js"))


def test_api_client_has_api_object():
    assert "const API" in _read(os.path.join(JS_DIR, "api_client.js"))


def test_api_client_has_init_api_key():
    assert "initAPIKey" in _read(os.path.join(JS_DIR, "api_client.js"))


def test_api_client_has_api_error_class():
    assert "APIError" in _read(os.path.join(JS_DIR, "api_client.js"))


# ── charts.js ─────────────────────────────────────────────────────────────────

def test_charts_has_render_sparkline():
    assert "renderSparkline" in _read(os.path.join(JS_DIR, "charts.js"))


def test_charts_has_render_heatmap():
    assert "renderHeatmap" in _read(os.path.join(JS_DIR, "charts.js"))


def test_charts_has_score_bar_html():
    assert "scoreBarHTML" in _read(os.path.join(JS_DIR, "charts.js"))


def test_charts_has_render_dau_bar():
    assert "renderDAUBar" in _read(os.path.join(JS_DIR, "charts.js"))


# ── dashboard.js ──────────────────────────────────────────────────────────────

def test_dashboard_has_switch_panel():
    assert "switchPanel" in _read(os.path.join(JS_DIR, "dashboard.js"))


def test_dashboard_has_switch_tab():
    assert "switchTab" in _read(os.path.join(JS_DIR, "dashboard.js"))


def test_dashboard_has_auto_refresh():
    src = _read(os.path.join(JS_DIR, "dashboard.js"))
    assert "REFRESH_INTERVAL" in src or "setInterval" in src


def test_dashboard_has_handle_pe_load():
    assert "handlePELoad" in _read(os.path.join(JS_DIR, "dashboard.js"))


def test_dashboard_has_handle_smb_load():
    assert "handleSMBLoad" in _read(os.path.join(JS_DIR, "dashboard.js"))


def test_dashboard_has_domcontentloaded():
    assert "DOMContentLoaded" in _read(os.path.join(JS_DIR, "dashboard.js"))


def test_dashboard_refresh_interval_is_5_minutes():
    src = _read(os.path.join(JS_DIR, "dashboard.js"))
    # 5 minutes = 300000ms
    assert "300" in src or "5 * 60" in src or "5*60" in src


# ── panel: morning_briefing.js ────────────────────────────────────────────────

def test_morning_briefing_has_load_fn():
    assert "loadMorningBriefing" in _read(os.path.join(PANELS_DIR, "morning_briefing.js"))


def test_morning_briefing_has_render_fn():
    assert "renderBriefing" in _read(os.path.join(PANELS_DIR, "morning_briefing.js"))


def test_morning_briefing_calls_briefing_api():
    assert "/jobs/briefing/morning" in _read(os.path.join(PANELS_DIR, "morning_briefing.js"))


# ── panel: opportunities.js ───────────────────────────────────────────────────

def test_opportunities_has_load_fn():
    assert "loadOpportunities" in _read(os.path.join(PANELS_DIR, "opportunities.js"))


def test_opportunities_has_render_fn():
    assert "renderOpportunitiesList" in _read(os.path.join(PANELS_DIR, "opportunities.js"))


def test_opportunities_has_default_universe():
    assert "DEFAULT_UNIVERSE" in _read(os.path.join(PANELS_DIR, "opportunities.js"))


def test_opportunities_clickable_rows():
    assert "loadSymbolIntelligence" in _read(os.path.join(PANELS_DIR, "opportunities.js"))


# ── panel: symbol_intelligence.js ────────────────────────────────────────────

def test_symbol_intel_has_load_fn():
    assert "loadSymbolIntelligence" in _read(os.path.join(PANELS_DIR, "symbol_intelligence.js"))


def test_symbol_intel_has_three_tabs():
    src = _read(os.path.join(PANELS_DIR, "symbol_intelligence.js"))
    assert "renderIntelligenceTab" in src
    assert "renderRiskTab" in src
    assert "renderExplanationTab" in src


def test_symbol_intel_calls_universal_api():
    assert "/intelligence/universal/" in _read(os.path.join(PANELS_DIR, "symbol_intelligence.js"))


def test_symbol_intel_calls_explain_api():
    assert "/explain/" in _read(os.path.join(PANELS_DIR, "symbol_intelligence.js"))


def test_symbol_intel_has_global_search_handler():
    assert "handleGlobalSymbolSearch" in _read(os.path.join(PANELS_DIR, "symbol_intelligence.js"))


# ── panel: risk_monitor.js ────────────────────────────────────────────────────

def test_risk_monitor_has_load_fn():
    assert "loadRiskMonitor" in _read(os.path.join(PANELS_DIR, "risk_monitor.js"))


def test_risk_monitor_has_render_fn():
    assert "renderRiskPanel" in _read(os.path.join(PANELS_DIR, "risk_monitor.js"))


def test_risk_monitor_calls_sri_api():
    assert "/axiom/risk/sri" in _read(os.path.join(PANELS_DIR, "risk_monitor.js"))


def test_risk_monitor_has_sparkline():
    assert "renderSparkline" in _read(os.path.join(PANELS_DIR, "risk_monitor.js"))


# ── panel: factor_environment.js ─────────────────────────────────────────────

def test_factor_env_has_load_fn():
    assert "loadFactorEnvironment" in _read(os.path.join(PANELS_DIR, "factor_environment.js"))


def test_factor_env_calls_regime_api():
    assert "/macro/regime" in _read(os.path.join(PANELS_DIR, "factor_environment.js"))


def test_factor_env_calls_cross_asset_api():
    assert "/macro/cross-asset" in _read(os.path.join(PANELS_DIR, "factor_environment.js"))


def test_factor_env_has_heatmap():
    assert "renderHeatmap" in _read(os.path.join(PANELS_DIR, "factor_environment.js"))


# ── panel: pipeline_status.js ────────────────────────────────────────────────

def test_pipeline_has_load_fn():
    assert "loadPipelineStatus" in _read(os.path.join(PANELS_DIR, "pipeline_status.js"))


def test_pipeline_has_stage_names():
    src = _read(os.path.join(PANELS_DIR, "pipeline_status.js"))
    assert "PIPELINE_STAGE_NAMES" in src
    assert "bar_ingestion" in src
    assert "ml_inference" in src


def test_pipeline_has_trigger_fn():
    assert "triggerPipelineRun" in _read(os.path.join(PANELS_DIR, "pipeline_status.js"))


def test_pipeline_calls_orchestration_api():
    src = _read(os.path.join(PANELS_DIR, "pipeline_status.js"))
    assert "/orchestration/health" in src
    assert "/orchestration/pipeline/status" in src


def test_pipeline_updates_health_indicator():
    assert "updateHealthIndicator" in _read(os.path.join(PANELS_DIR, "pipeline_status.js"))


# ── panel: pe_dashboard.js ────────────────────────────────────────────────────

def test_pe_has_load_fn():
    assert "loadPEPortfolio" in _read(os.path.join(PANELS_DIR, "pe_dashboard.js"))


def test_pe_has_health_grid():
    assert "renderPortfolioHealthGrid" in _read(os.path.join(PANELS_DIR, "pe_dashboard.js"))


def test_pe_has_exit_pipeline():
    assert "renderExitPipeline" in _read(os.path.join(PANELS_DIR, "pe_dashboard.js"))


def test_pe_has_schillit_alerts():
    assert "renderSchilitAlerts" in _read(os.path.join(PANELS_DIR, "pe_dashboard.js"))


def test_pe_has_lp_report():
    assert "renderLPReportSummary" in _read(os.path.join(PANELS_DIR, "pe_dashboard.js"))


def test_pe_calls_portfolio_api():
    assert "/pe/portfolio/" in _read(os.path.join(PANELS_DIR, "pe_dashboard.js"))


def test_pe_shows_moic():
    assert "moic" in _read(os.path.join(PANELS_DIR, "pe_dashboard.js")).lower()


def test_pe_shows_distress_alerts():
    src = _read(os.path.join(PANELS_DIR, "pe_dashboard.js"))
    assert "distress" in src.lower() or "schillit" in src.lower()


# ── panel: smb_dashboard.js ───────────────────────────────────────────────────

def test_smb_has_load_fn():
    assert "loadSMBIntelligence" in _read(os.path.join(PANELS_DIR, "smb_dashboard.js"))


def test_smb_has_render_dashboard():
    assert "renderSMBDashboard" in _read(os.path.join(PANELS_DIR, "smb_dashboard.js"))


def test_smb_has_cashflow_forecast():
    assert "renderCashFlowForecast" in _read(os.path.join(PANELS_DIR, "smb_dashboard.js"))


def test_smb_has_supplier_risks():
    assert "renderSupplierRisks" in _read(os.path.join(PANELS_DIR, "smb_dashboard.js"))


def test_smb_has_pricing_intelligence():
    assert "renderPricingIntelligence" in _read(os.path.join(PANELS_DIR, "smb_dashboard.js"))


def test_smb_calls_intelligence_dashboard_api():
    assert "/smb/entity/" in _read(os.path.join(PANELS_DIR, "smb_dashboard.js"))
    assert "intelligence-dashboard" in _read(os.path.join(PANELS_DIR, "smb_dashboard.js"))


def test_smb_shows_cash_runway():
    assert "runway" in _read(os.path.join(PANELS_DIR, "smb_dashboard.js")).lower()


def test_smb_shows_pricing_power():
    assert "pricing_power" in _read(os.path.join(PANELS_DIR, "smb_dashboard.js"))


def test_smb_has_sparkline_for_cashflow():
    assert "renderSparkline" in _read(os.path.join(PANELS_DIR, "smb_dashboard.js"))
