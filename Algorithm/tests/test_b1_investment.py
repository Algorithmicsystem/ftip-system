"""B1 investment intelligence: universe scores, opportunities, ML threshold, chatbot context."""
from __future__ import annotations

import inspect
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from api.main import app

WEBAPP = Path(__file__).resolve().parents[1] / "api" / "webapp"


# ---------------------------------------------------------------------------
# Section 2 — Universe scores real data
# ---------------------------------------------------------------------------

class TestUniverseScoresRealData:

    def test_universe_scores_returns_30_symbols(self):
        with TestClient(app) as c:
            r = c.get("/intelligence/universe/scores")
        assert r.status_code == 200
        assert len(r.json()) == 30

    def test_universe_scores_has_dau_field(self):
        with TestClient(app) as c:
            r = c.get("/intelligence/universe/scores")
        for sym in r.json():
            assert "dau" in sym

    def test_universe_scores_sql_extracts_correct_field(self):
        import api.main as m
        src = inspect.getsource(m.universe_scores)
        assert "deployable_alpha_utility" in src or "dau" in src.lower()

    def test_universe_scores_eis_uses_earnings_quality(self):
        import api.main as m
        src = inspect.getsource(m.universe_scores)
        # Must NOT use the wrong eis_component key
        assert "eis_component" not in src
        assert "earnings_quality_component" in src

    def test_universal_aapl_returns_signal(self):
        with TestClient(app) as c:
            r = c.get("/intelligence/universal/AAPL")
        assert r.status_code == 200
        assert "signal_label" in r.json()
        assert r.json()["signal_label"] in ("BUY", "SELL", "HOLD", "STRONG_BUY", "STRONG_SELL")


# ---------------------------------------------------------------------------
# Section 5 — Opportunities panel enhanced
# ---------------------------------------------------------------------------

class TestOpportunitiesEnhanced:

    def setup_method(self):
        self.js = (WEBAPP / "js" / "panels" / "opportunities.js").read_text()

    def test_opportunities_js_has_summary_bar(self):
        assert "BUY" in self.js and "SELL" in self.js and "HOLD" in self.js

    def test_opportunities_js_has_click_handler(self):
        assert "loadSymbolIntelligence" in self.js or "switchPanel" in self.js

    def test_opportunities_js_has_color_coding(self):
        assert "signal-buy" in self.js or "var(--signal-buy)" in self.js

    def test_opportunities_js_shows_10_rows(self):
        assert "slice(0, 10)" in self.js or ".slice(0,10)" in self.js

    def test_opportunities_js_has_sort_order(self):
        assert "sigOrder" in self.js or "BUY: 0" in self.js


# ---------------------------------------------------------------------------
# Section 4 — ML training threshold
# ---------------------------------------------------------------------------

class TestMLThreshold:

    def test_ml_min_samples_lowered(self):
        from api.axiom.ml.ensemble import AxiomEnsemble
        e = AxiomEnsemble()
        threshold = getattr(e, "min_training_samples",
                    getattr(e, "MIN_TRAINING_SAMPLES",
                    getattr(e, "_min_samples", 50)))
        assert threshold <= 30, f"ML threshold too high: {threshold}, should be ≤30"

    def test_axiom_ensemble_class_exists(self):
        from api.axiom.ml.ensemble import AxiomEnsemble
        e = AxiomEnsemble()
        assert e is not None

    def test_initial_sample_threshold_at_most_25(self):
        from api.axiom.ml.training_data import MINIMUM_SAMPLES_INITIAL
        assert MINIMUM_SAMPLES_INITIAL <= 25


# ---------------------------------------------------------------------------
# Section 7 — Chatbot live context
# ---------------------------------------------------------------------------

class TestChatbotLiveContext:

    def test_llm_chat_accepts_context(self):
        with TestClient(app) as c:
            r = c.post("/llm/chat", json={"message": "test", "context": {"universe": "test"}})
        assert r.status_code == 200

    def test_llm_chat_context_field_optional(self):
        with TestClient(app) as c:
            r = c.post("/llm/chat", json={"message": "test"})
        assert r.status_code == 200

    def test_chatbot_js_builds_context(self):
        js = (WEBAPP / "js" / "axiom_chatbot.js").read_text()
        assert "buildContext" in js

    def test_chatbot_js_sends_context(self):
        js = (WEBAPP / "js" / "axiom_chatbot.js").read_text()
        assert "context" in js

    def test_llm_chat_model_has_context_field(self):
        import api.main as m
        import inspect
        src = inspect.getsource(m.LLMChatRequest)
        assert "context" in src


# ---------------------------------------------------------------------------
# Section 9 — PE/SMB auto-demo
# ---------------------------------------------------------------------------

class TestPEDemoAutoLoad:

    def test_pe_dashboard_js_has_auto_demo(self):
        js = (WEBAPP / "js" / "panels" / "pe_dashboard.js").read_text()
        assert "loadPEDemo" in js

    def test_pe_dashboard_js_auto_loads_on_no_org(self):
        js = (WEBAPP / "js" / "panels" / "pe_dashboard.js").read_text()
        # loadPEPortfolio with no orgId must auto-call loadPEDemo
        assert "loadPEDemo" in js
        # Must not just return immediately (the old `if (!orgId) return;`)
        idx_fn = js.find("async function loadPEPortfolio")
        idx_demo = js.find("await loadPEDemo", idx_fn)
        assert idx_demo > idx_fn, "loadPEPortfolio must call loadPEDemo when no orgId"

    def test_smb_dashboard_js_has_auto_demo(self):
        js = (WEBAPP / "js" / "panels" / "smb_dashboard.js").read_text()
        assert "loadSMBDemo" in js

    def test_smb_dashboard_js_auto_loads_on_no_entity(self):
        js = (WEBAPP / "js" / "panels" / "smb_dashboard.js").read_text()
        idx_fn = js.find("async function loadSMBIntelligence")
        idx_demo = js.find("await loadSMBDemo", idx_fn)
        assert idx_demo > idx_fn, "loadSMBIntelligence must call loadSMBDemo when no entityId"


# ---------------------------------------------------------------------------
# Section 10 — Version bump
# ---------------------------------------------------------------------------

class TestVersionBump:

    def test_version_1_0_2(self):
        with TestClient(app) as c:
            r = c.get("/config/client")
        assert r.status_code == 200
        assert r.json()["version"] == "1.0.2"
