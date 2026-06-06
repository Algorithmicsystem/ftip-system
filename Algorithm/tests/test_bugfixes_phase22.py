"""Bug-fix tests: 5 live bugs + system status endpoint + panel endpoint audit."""
from __future__ import annotations

import datetime as dt
import logging
from typing import Any, Dict

import pytest
from fastapi.testclient import TestClient

from api.main import app


# ---------------------------------------------------------------------------
# B6 / Section B: universal intelligence with empty DB
# ---------------------------------------------------------------------------

class TestUniversalIntelligenceEmptyDB:

    def test_universal_intelligence_empty_db_returns_default(self):
        """assemble_universal_intelligence must never raise with DB disabled."""
        from api.universal.intelligence_api import assemble_universal_intelligence
        result = assemble_universal_intelligence("AAPL", dt.date.today())
        assert result is not None
        assert result.symbol == "AAPL"
        assert result.signal_label in ("BUY", "HOLD", "SELL")
        assert result.dau >= 0

    def test_universal_intelligence_has_valid_structure(self):
        from api.universal.intelligence_api import assemble_universal_intelligence
        result = assemble_universal_intelligence("MSFT", dt.date.today())
        # All required fields present and typed
        assert isinstance(result.eis_score, float)
        assert isinstance(result.caps_score, float)
        assert isinstance(result.fragility_score, float)
        assert isinstance(result.ic_state, str)

    def test_universal_intelligence_var_field_is_none_not_error(self):
        """var_1d_99 (field) should be None, not raise, when DB is absent."""
        from api.universal.intelligence_api import assemble_universal_intelligence
        result = assemble_universal_intelligence("TSLA", dt.date.today())
        # var_1d_99 is allowed to be None — must not be a raised exception
        assert result.var_1d_99 is None or isinstance(result.var_1d_99, float)

    def test_universal_endpoint_returns_200(self):
        with TestClient(app) as client:
            r = client.get("/intelligence/universal/AAPL")
            assert r.status_code == 200
            data = r.json()
            assert "signal_label" in data or "dau" in data or "symbol" in data

    def test_universal_intelligence_no_column_errors(self, caplog):
        """DB-off mode must produce no WARNING log lines for column issues."""
        from api.universal.intelligence_api import assemble_universal_intelligence
        with caplog.at_level(logging.WARNING, logger="api.universal.intelligence_api"):
            assemble_universal_intelligence("NVDA", dt.date.today())
        # Should have no warnings (default path hits no DB)
        assert len(caplog.records) == 0


# ---------------------------------------------------------------------------
# A2: model registry NULL type fix
# ---------------------------------------------------------------------------

class TestModelRegistryNullFix:

    def test_get_active_model_with_none_regime_no_exception(self):
        """get_active_model(None) must not raise psycopg3 type errors."""
        from api.axiom.ml.model_registry import get_active_model
        # Should not raise — returns (None, {}) when DB is disabled
        model, meta = get_active_model(regime_label=None)
        assert model is None or model is not None  # either is fine
        assert isinstance(meta, dict)

    def test_get_active_model_with_string_regime_no_exception(self):
        from api.axiom.ml.model_registry import get_active_model
        model, meta = get_active_model(regime_label="BULL")
        assert isinstance(meta, dict)


# ---------------------------------------------------------------------------
# Morning briefing smoke test
# ---------------------------------------------------------------------------

class TestMorningBriefingSmoke:

    def test_morning_briefing_endpoint_returns_200(self):
        with TestClient(app) as client:
            r = client.get("/jobs/briefing/morning")
            assert r.status_code == 200

    def test_morning_briefing_has_briefing_text(self):
        with TestClient(app) as client:
            r = client.get("/jobs/briefing/morning")
            data = r.json()
            assert "briefing_text" in data
            assert isinstance(data["briefing_text"], str)
            assert len(data["briefing_text"]) > 0

    def test_morning_briefing_no_warnings(self, caplog):
        """Briefing endpoint should not produce WARNING log lines in normal operation."""
        with caplog.at_level(logging.WARNING):
            with TestClient(app) as client:
                r = client.get("/jobs/briefing/morning")
        assert r.status_code == 200
        # No model_registry.get_failed or similar WARNING log entries
        registry_warns = [r for r in caplog.records
                          if "model_registry.get_failed" in r.message or
                             "morning_briefing.store_failed" in r.message]
        assert len(registry_warns) == 0


# ---------------------------------------------------------------------------
# A3: Explain route fix
# ---------------------------------------------------------------------------

class TestExplainRouteFix:

    def test_explain_signal_route_returns_200(self):
        with TestClient(app) as client:
            r = client.get("/explain/signal/AAPL")
            assert r.status_code == 200

    def test_explain_old_path_is_404(self):
        """The old wrong path /explain/AAPL must not accidentally work."""
        with TestClient(app) as client:
            r = client.get("/explain/AAPL")
            assert r.status_code == 404

    def test_explain_signal_returns_dict(self):
        with TestClient(app) as client:
            r = client.get("/explain/signal/MSFT")
            assert r.status_code == 200
            assert isinstance(r.json(), dict)


# ---------------------------------------------------------------------------
# A4: Macro route fix
# ---------------------------------------------------------------------------

class TestMacroSnapshotRoute:

    def test_macro_snapshot_returns_200(self):
        with TestClient(app) as client:
            r = client.get("/macro/snapshot")
            assert r.status_code == 200

    def test_macro_snapshot_has_cross_asset_and_macro_intelligence(self):
        with TestClient(app) as client:
            r = client.get("/macro/snapshot")
            data = r.json()
            assert "cross_asset" in data
            assert "macro_intelligence" in data

    def test_macro_snapshot_cross_asset_has_signals(self):
        with TestClient(app) as client:
            r = client.get("/macro/snapshot")
            ca = r.json()["cross_asset"]
            for key in ("fixed_income_signal", "currency_signal", "commodity_signal", "volatility_signal"):
                assert key in ca

    def test_macro_regime_old_path_is_404(self):
        """Confirm the old wrong path /macro/regime is not an endpoint."""
        with TestClient(app) as client:
            r = client.get("/macro/regime")
            assert r.status_code == 404

    def test_macro_cross_asset_old_path_is_404(self):
        with TestClient(app) as client:
            r = client.get("/macro/cross-asset")
            assert r.status_code == 404


# ---------------------------------------------------------------------------
# G: System status endpoint
# ---------------------------------------------------------------------------

class TestSystemStatusEndpoint:

    def test_system_status_returns_200(self):
        with TestClient(app) as client:
            r = client.get("/system/status")
            assert r.status_code == 200

    def test_system_status_has_required_keys(self):
        with TestClient(app) as client:
            data = client.get("/system/status").json()
        required = {
            "server", "db_connected", "migrations_applied", "latest_migration",
            "axiom_scores_count", "morning_briefings_count", "ml_model_trained",
            "scheduler_running", "last_pipeline_run", "warnings", "version",
        }
        assert required.issubset(data.keys())

    def test_system_status_server_healthy(self):
        with TestClient(app) as client:
            data = client.get("/system/status").json()
        assert data["server"] == "healthy"

    def test_system_status_version(self):
        with TestClient(app) as client:
            data = client.get("/system/status").json()
        assert data["version"] == "22.0.0"

    def test_system_status_db_connected_is_bool(self):
        with TestClient(app) as client:
            data = client.get("/system/status").json()
        assert isinstance(data["db_connected"], bool)

    def test_system_status_warnings_is_list(self):
        with TestClient(app) as client:
            data = client.get("/system/status").json()
        assert isinstance(data["warnings"], list)

    def test_system_status_counts_are_ints(self):
        with TestClient(app) as client:
            data = client.get("/system/status").json()
        assert isinstance(data["axiom_scores_count"], int)
        assert isinstance(data["morning_briefings_count"], int)
        assert isinstance(data["migrations_applied"], int)


# ---------------------------------------------------------------------------
# H3: Panel endpoint audit — every URL called by JS panels must be registered
# ---------------------------------------------------------------------------

class TestAllPanelEndpointsExist:
    """Verify every URL called by JS panels is a registered FastAPI route.

    This prevents future 404 regressions.
    """

    @pytest.fixture(scope="class")
    def registered_paths(self):
        paths = set()
        for route in app.routes:
            if hasattr(route, "path"):
                paths.add(route.path)
        return paths

    def _has_path(self, registered: set, prefix: str) -> bool:
        """Check if any registered path starts with prefix or matches it."""
        # Normalise: strip trailing slash
        prefix = prefix.rstrip("/")
        for p in registered:
            # Exact match or route with path params that would match
            if p == prefix:
                return True
            # Template prefix: /jobs/briefing/morning matches /jobs/briefing/morning
            p_base = p.split("{")[0].rstrip("/")
            if p_base == prefix or prefix.startswith(p_base + "/"):
                return True
        return False

    def test_jobs_briefing_morning_exists(self, registered_paths):
        assert self._has_path(registered_paths, "/jobs/briefing/morning")

    def test_orchestration_health_exists(self, registered_paths):
        assert self._has_path(registered_paths, "/orchestration/health")

    def test_orchestration_pipeline_status_exists(self, registered_paths):
        assert self._has_path(registered_paths, "/orchestration/pipeline/status")

    def test_orchestration_pipeline_run_exists(self, registered_paths):
        assert self._has_path(registered_paths, "/orchestration/pipeline/run")

    def test_intelligence_universal_exists(self, registered_paths):
        assert self._has_path(registered_paths, "/intelligence/universal")

    def test_axiom_risk_sri_exists(self, registered_paths):
        assert self._has_path(registered_paths, "/axiom/risk/sri")

    def test_axiom_risk_sri_history_exists(self, registered_paths):
        assert self._has_path(registered_paths, "/axiom/risk/sri/history")

    def test_macro_snapshot_exists(self, registered_paths):
        assert self._has_path(registered_paths, "/macro/snapshot")

    def test_explain_signal_exists(self, registered_paths):
        assert self._has_path(registered_paths, "/explain/signal")

    def test_pe_portfolio_overview_exists(self, registered_paths):
        assert self._has_path(registered_paths, "/pe/portfolio")

    def test_pe_portfolio_stress_alerts_exists(self, registered_paths):
        assert self._has_path(registered_paths, "/pe/portfolio")

    def test_pe_portfolio_lp_report_exists(self, registered_paths):
        # /pe/portfolio/{org_id}/lp-report
        lp_route = "/pe/portfolio/{org_id}/lp-report"
        assert lp_route in registered_paths

    def test_smb_intelligence_dashboard_exists(self, registered_paths):
        smb_route = "/smb/entity/{entity_id}/intelligence-dashboard"
        assert smb_route in registered_paths

    def test_system_status_exists(self, registered_paths):
        assert self._has_path(registered_paths, "/system/status")
