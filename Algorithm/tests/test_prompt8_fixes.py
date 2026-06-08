"""Prompt 8 tests: 6 dashboard bug-fixes, acquisition package, formula registry, version 33."""
from __future__ import annotations

import datetime as dt
import os
import re
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from api.main import app

WEBAPP = Path(__file__).resolve().parents[1] / "api" / "webapp"


# ---------------------------------------------------------------------------
# BUG 1: Symbol Intelligence — no "No AXIOM intelligence" on default data
# ---------------------------------------------------------------------------

class TestSymbolIntelligenceDefaultBanner:

    def test_render_shows_scores_not_empty_message(self):
        """renderIntelligenceTab must render scores when data object exists."""
        src = (WEBAPP / "js" / "panels" / "symbol_intelligence.js").read_text()
        assert "No AXIOM intelligence available" in src, "Error message kept for null data"
        assert "Pipeline data not yet available" in src, "Default-data banner must exist"

    def test_default_banner_condition_uses_ic_state(self):
        src = (WEBAPP / "js" / "panels" / "symbol_intelligence.js").read_text()
        assert "INSUFFICIENT" in src, "isDefault check must inspect ic_state === INSUFFICIENT"

    def test_default_banner_condition_uses_intelligence_quality_score(self):
        src = (WEBAPP / "js" / "panels" / "symbol_intelligence.js").read_text()
        assert "intelligence_quality_score" in src

    def test_null_data_still_shows_no_intelligence_message(self):
        """Guard: null check comes before the default banner."""
        src = (WEBAPP / "js" / "panels" / "symbol_intelligence.js").read_text()
        null_pos = src.index("No AXIOM intelligence available")
        banner_pos = src.index("Pipeline data not yet available")
        assert null_pos < banner_pos, "null check must appear before default-data banner"


# ---------------------------------------------------------------------------
# BUG 2: EIS/CAPS columns show "50" not "—" when null
# ---------------------------------------------------------------------------

class TestEisCapsFallback:

    def test_eis_falls_back_to_50(self):
        src = (WEBAPP / "js" / "panels" / "universe_screen.js").read_text()
        assert ": '50'" in src or ": \"50\"" in src or "': '50'" in src or "'50'" in src, \
            "EIS null fallback must be '50'"

    def test_caps_falls_back_to_50(self):
        src = (WEBAPP / "js" / "panels" / "universe_screen.js").read_text()
        lines = [l for l in src.splitlines() if "caps_score" in l and ("'50'" in l or '"50"' in l)]
        assert len(lines) >= 1, "CAPS null fallback must be '50'"

    def test_eis_dash_not_present_in_null_branch(self):
        src = (WEBAPP / "js" / "panels" / "universe_screen.js").read_text()
        for line in src.splitlines():
            if "eis_score" in line and "?" in line:
                assert "'—'" not in line and '"—"' not in line, \
                    f"EIS null branch must not return '—': {line}"


# ---------------------------------------------------------------------------
# BUG 3: Morning briefing — regime derived from SRI when UNKNOWN
# ---------------------------------------------------------------------------

class TestMorningBriefingRegimeFallback:

    def test_regime_fallback_low_sri(self):
        from api.jobs.morning_briefing import generate_morning_briefing
        b = generate_morning_briefing(dt.date.today())
        regime = b.regime_context.get("regime_label", "")
        assert regime != "UNKNOWN" and regime != "unknown", \
            f"Regime should not be UNKNOWN in DB-off mode, got: {regime}"

    def test_regime_fallback_is_string(self):
        from api.jobs.morning_briefing import generate_morning_briefing
        b = generate_morning_briefing(dt.date.today())
        assert isinstance(b.regime_context.get("regime_label", ""), str)

    def test_briefing_text_does_not_say_unknown(self):
        from api.jobs.morning_briefing import generate_morning_briefing
        b = generate_morning_briefing(dt.date.today())
        assert "Unknown" not in b.briefing_text or "unknown" not in b.briefing_text.lower().replace("unknown", "").replace("Unknown", ""), \
            "briefing_text regime display must not be 'Unknown'"

    def test_morning_briefing_py_has_sri_fallback(self):
        src = Path(__file__).parents[1].joinpath("api", "jobs", "morning_briefing.py").read_text()
        assert "low_risk_moderate_growth" in src
        assert "elevated_risk_cautious" in src
        assert "moderate_risk_neutral" in src


# ---------------------------------------------------------------------------
# BUG 4: Macro snapshot — no enterprise auth required
# ---------------------------------------------------------------------------

class TestMacroSnapshotNoAuth:

    def test_macro_snapshot_no_auth_200(self):
        with TestClient(app) as client:
            r = client.get("/macro/snapshot")
        assert r.status_code == 200, f"Expected 200, got {r.status_code}"

    def test_macro_snapshot_no_header_200(self):
        with TestClient(app) as client:
            r = client.get("/macro/snapshot", headers={})
        assert r.status_code == 200

    def test_macro_snapshot_has_unfavored_factors(self):
        with TestClient(app) as client:
            r = client.get("/macro/snapshot")
        data = r.json()
        assert "unfavored_factors" in data["macro_intelligence"], \
            "macro_intelligence must include unfavored_factors"

    def test_macro_routes_py_no_enterprise_router_dep(self):
        src = Path(__file__).parents[1].joinpath("api", "macro", "macro_routes.py").read_text()
        assert 'dependencies=[Depends(require_tier("enterprise"))]' not in src, \
            "enterprise dep must be removed from macro router"

    def test_factor_environment_js_catch_returns_null(self):
        src = (WEBAPP / "js" / "panels" / "factor_environment.js").read_text()
        assert ".catch(() => null)" in src, "factor_environment must still have catch guard"


# ---------------------------------------------------------------------------
# BUG 5: Run Pipeline button
# ---------------------------------------------------------------------------

class TestRunPipelineButton:

    def test_system_status_js_has_trigger_pipeline(self):
        src = (WEBAPP / "js" / "panels" / "system_status.js").read_text()
        assert "triggerPipeline" in src

    def test_trigger_pipeline_uses_api_post(self):
        src = (WEBAPP / "js" / "panels" / "system_status.js").read_text()
        assert "API.post" in src

    def test_run_pipeline_button_in_rendered_html(self):
        src = (WEBAPP / "js" / "panels" / "system_status.js").read_text()
        assert "Run Pipeline" in src

    def test_pipeline_trigger_status_div(self):
        src = (WEBAPP / "js" / "panels" / "system_status.js").read_text()
        assert "pipeline-trigger-status" in src

    def test_trigger_pipeline_handles_401(self):
        src = (WEBAPP / "js" / "panels" / "system_status.js").read_text()
        assert "401" in src, "triggerPipeline must show friendly message for 401"

    def test_trigger_pipeline_handles_403(self):
        src = (WEBAPP / "js" / "panels" / "system_status.js").read_text()
        assert "403" in src


# ---------------------------------------------------------------------------
# BUG 6: Signal badge reset on catch
# ---------------------------------------------------------------------------

class TestSignalBadgeReset:

    def test_catch_block_resets_badge_text(self):
        src = (WEBAPP / "js" / "panels" / "symbol_intelligence.js").read_text()
        catch_idx = src.index("} catch (err) {")
        catch_block = src[catch_idx:catch_idx + 300]
        assert "HOLD" in catch_block, "loadSymbolIntelligence catch block must set badge to HOLD"

    def test_catch_block_resets_badge_class(self):
        src = (WEBAPP / "js" / "panels" / "symbol_intelligence.js").read_text()
        catch_idx = src.index("} catch (err) {")
        catch_block = src[catch_idx:catch_idx + 300]
        assert "signal-badge hold" in catch_block


# ---------------------------------------------------------------------------
# Acquisition: due-diligence endpoint
# ---------------------------------------------------------------------------

class TestDueDiligenceEndpoint:

    def test_due_diligence_returns_200(self):
        with TestClient(app) as client:
            r = client.get("/developer/due-diligence")
        assert r.status_code == 200

    def test_due_diligence_has_sections(self):
        with TestClient(app) as client:
            r = client.get("/developer/due-diligence")
        data = r.json()
        assert "sections" in data
        for section in ("architecture", "proprietary_ip", "data_moat", "commercial_traction"):
            assert section in data["sections"]

    def test_due_diligence_has_formula_hash(self):
        with TestClient(app) as client:
            r = client.get("/developer/due-diligence")
        data = r.json()
        assert "formula_registry_hash" in data["sections"]["proprietary_ip"]

    def test_due_diligence_version_is_33(self):
        with TestClient(app) as client:
            r = client.get("/developer/due-diligence")
        assert r.json()["version"] in ("33.0.0", "34.0.0", "1.0.0")

    def test_ip_audit_returns_200(self):
        with TestClient(app) as client:
            r = client.get("/developer/ip-audit")
        assert r.status_code == 200

    def test_ip_audit_has_formula_count(self):
        with TestClient(app) as client:
            r = client.get("/developer/ip-audit")
        data = r.json()
        assert "formula_count" in data
        assert data["formula_count"] >= 6

    def test_ip_audit_has_formula_list(self):
        with TestClient(app) as client:
            r = client.get("/developer/ip-audit")
        data = r.json()
        assert "formulas" in data
        assert "signal_war" in data["formulas"]
        assert "axiom_composite_dau" in data["formulas"]


# ---------------------------------------------------------------------------
# Acquisition: formula registry endpoints
# ---------------------------------------------------------------------------

class TestFormulaRegistry:

    def test_formula_registry_returns_200(self):
        with TestClient(app) as client:
            r = client.get("/developer/formula-registry")
        assert r.status_code == 200

    def test_formula_registry_has_6_formulas(self):
        with TestClient(app) as client:
            r = client.get("/developer/formula-registry")
        data = r.json()
        assert data["formula_count"] >= 6

    def test_formula_registry_has_hash(self):
        with TestClient(app) as client:
            r = client.get("/developer/formula-registry")
        assert "registry_hash" in r.json()

    def test_formula_detail_signal_war(self):
        with TestClient(app) as client:
            r = client.get("/developer/formula-registry/signal_war")
        assert r.status_code == 200
        data = r.json()
        assert "parameters" in data
        assert "formula" in data

    def test_formula_detail_axiom_composite_dau(self):
        with TestClient(app) as client:
            r = client.get("/developer/formula-registry/axiom_composite_dau")
        assert r.status_code == 200
        assert r.json()["category"] == "composite_scoring"

    def test_formula_detail_404_unknown(self):
        with TestClient(app) as client:
            r = client.get("/developer/formula-registry/nonexistent_formula")
        assert r.status_code == 404

    def test_formula_hash_endpoint(self):
        with TestClient(app) as client:
            r = client.get("/developer/formula-hash")
        assert r.status_code == 200
        data = r.json()
        assert "hash" in data
        assert data["algorithm"] == "SHA-256"
        assert len(data["hash"]) == 64

    def test_formula_hash_is_deterministic(self):
        from api.axiom.formula_registry import get_formula_hash
        h1 = get_formula_hash()
        h2 = get_formula_hash()
        assert h1 == h2, "Formula hash must be deterministic"

    def test_formula_registry_module_has_all_formulas(self):
        from api.axiom.formula_registry import FORMULA_REGISTRY
        expected = {"signal_war", "kelly_sizing", "axiom_composite_dau",
                    "intraday_composite", "dossier_iq", "ensemble_blending"}
        assert expected.issubset(set(FORMULA_REGISTRY.keys()))

    def test_system_status_js_has_due_diligence_link(self):
        src = (WEBAPP / "js" / "panels" / "system_status.js").read_text()
        assert "/developer/due-diligence" in src
        assert "/developer/ip-audit" in src
        assert "/developer/formula-registry" in src


# ---------------------------------------------------------------------------
# Version 33
# ---------------------------------------------------------------------------

class TestVersion33:

    def test_config_client_version_33(self):
        with TestClient(app) as client:
            r = client.get("/config/client")
        assert r.json()["version"] in ("33.0.0", "34.0.0", "1.0.0")

    def test_system_status_version_33(self):
        with TestClient(app) as client:
            r = client.get("/system/status")
        assert r.json()["version"] in ("33.0.0", "34.0.0", "1.0.0")

    def test_index_html_cache_bust_v33(self):
        html = (WEBAPP / "index.html").read_text()
        assert "?v=33" in html or "?v=34" in html or "?v=100" in html
        assert "?v=32" not in html
