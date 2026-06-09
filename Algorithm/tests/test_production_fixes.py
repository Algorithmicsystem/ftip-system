"""Production fix tests: FTIP_API_KEYS env var, scheduler, migrations, morning briefing."""
from __future__ import annotations

import importlib
import inspect
import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from api.main import app

WEBAPP = Path(__file__).resolve().parents[1] / "api" / "webapp"


# ---------------------------------------------------------------------------
# Fix 1 — API key env var fallback
# ---------------------------------------------------------------------------

class TestAPIKeyFallback:

    def test_config_returns_api_key_from_ftip_api_keys(self):
        """config.get_api_key() must read FTIP_API_KEYS (plural) when singular is absent."""
        os.environ.pop("FTIP_API_KEY", None)
        os.environ["FTIP_API_KEYS"] = "test-key-plural"
        try:
            import api.config as cfg
            key = cfg.get_api_key()
            assert key == "test-key-plural"
        finally:
            os.environ.pop("FTIP_API_KEYS", None)

    def test_config_prefers_ftip_api_key_singular(self):
        os.environ["FTIP_API_KEY"] = "singular-key"
        os.environ["FTIP_API_KEYS"] = "plural-key"
        try:
            import api.config as cfg
            key = cfg.get_api_key()
            assert key == "singular-key"
        finally:
            os.environ.pop("FTIP_API_KEY", None)
            os.environ.pop("FTIP_API_KEYS", None)

    def test_config_client_returns_api_key_field(self):
        with TestClient(app) as client:
            r = client.get("/config/client")
        assert r.status_code == 200
        assert "api_key" in r.json()

    def test_config_client_version_is_1(self):
        with TestClient(app) as client:
            r = client.get("/config/client")
        assert r.json()["version"] in ("1.0.0", "1.0.1", "1.0.2", "1.0.3", "33.0.0", "34.0.0")

    def test_get_api_keys_returns_set(self):
        os.environ["FTIP_API_KEYS"] = "key1,key2"
        try:
            import api.config as cfg
            keys = cfg.get_api_keys()
            assert isinstance(keys, set)
            assert "key1" in keys
            assert "key2" in keys
        finally:
            os.environ.pop("FTIP_API_KEYS", None)

    def test_get_api_keys_merges_singular_and_plural(self):
        os.environ["FTIP_API_KEY"] = "single-key"
        os.environ["FTIP_API_KEYS"] = "plural-key"
        try:
            import api.config as cfg
            keys = cfg.get_api_keys()
            assert "single-key" in keys
            assert "plural-key" in keys
        finally:
            os.environ.pop("FTIP_API_KEY", None)
            os.environ.pop("FTIP_API_KEYS", None)

    def test_config_main_reads_ftip_api_keys(self):
        """main.py /config/client must not read only os.environ['FTIP_API_KEY']."""
        import api.main as m
        src = inspect.getsource(m.client_config)
        assert "get_api_key" in src, "/config/client must call config.get_api_key()"

    def test_billing_uses_railway_environment(self):
        src = Path(__file__).parents[1].joinpath("api", "developer", "billing.py").read_text()
        assert "RAILWAY_ENVIRONMENT" in src, "billing.py must fall back to RAILWAY_ENVIRONMENT"


# ---------------------------------------------------------------------------
# Fix 2 — Scheduler stays running
# ---------------------------------------------------------------------------

class TestSchedulerStaysRunning:

    def test_lifespan_has_finally_block(self):
        import api.main as m
        src = inspect.getsource(m.lifespan)
        assert "finally" in src, "lifespan must have finally block for scheduler.stop()"

    def test_scheduler_not_gated_by_env_var(self):
        src = Path(__file__).parents[1].joinpath("api", "jobs", "scheduler.py").read_text()
        assert 'env_bool("FTIP_SCHEDULER_ENABLED", False)' not in src, \
            "start_scheduler must not default to disabled"

    def test_scheduler_suppressed_in_pytest(self):
        src = Path(__file__).parents[1].joinpath("api", "jobs", "scheduler.py").read_text()
        assert "pytest" in src, "start_scheduler must return early in pytest mode"

    def test_scheduler_has_heartbeat(self):
        src = Path(__file__).parents[1].joinpath("api", "jobs", "scheduler.py").read_text()
        assert "scheduler.alive" in src, "scheduler must log heartbeat"

    def test_intraday_update_uses_get_api_key(self):
        src = Path(__file__).parents[1].joinpath("api", "jobs", "scheduler.py").read_text()
        assert "get_api_key()" in src, "_job_intraday_update must use config.get_api_key()"
        assert 'env("FTIP_API_KEY")' not in src, "_job_intraday_update must not hardcode FTIP_API_KEY"


# ---------------------------------------------------------------------------
# Fix 3 — Migrations always run
# ---------------------------------------------------------------------------

class TestMigrationsAlwaysRun:

    def test_lifecycle_no_migrations_auto_skip(self):
        src = Path(__file__).parents[1].joinpath("api", "lifecycle.py").read_text()
        assert "migrations auto disabled; skipping" not in src, \
            "lifecycle.py must not skip migrations due to FTIP_MIGRATIONS_AUTO"

    def test_lifecycle_logs_schema_up_to_date(self):
        src = Path(__file__).parents[1].joinpath("api", "lifecycle.py").read_text()
        assert "schema up to date" in src

    def test_migration_error_does_not_crash_startup(self):
        src = Path(__file__).parents[1].joinpath("api", "lifecycle.py").read_text()
        assert "migration error" in src, "lifecycle must log migration errors without crashing"


# ---------------------------------------------------------------------------
# Fix 4 — Pipeline + morning briefing auto-trigger
# ---------------------------------------------------------------------------

class TestStartupTriggers:

    def test_stale_pipeline_trigger_exists(self):
        src = Path(__file__).parents[1].joinpath("api", "lifecycle.py").read_text()
        assert "_check_and_trigger_stale_pipeline" in src

    def test_morning_briefing_trigger_exists(self):
        src = Path(__file__).parents[1].joinpath("api", "lifecycle.py").read_text()
        assert "_trigger_morning_briefing_if_missing" in src

    def test_triggers_fire_without_migrations_auto(self):
        src = Path(__file__).parents[1].joinpath("api", "lifecycle.py").read_text()
        # Both Thread() calls must appear inside def startup()
        startup_src = src[src.index("def startup()"):]
        assert "target=_check_and_trigger_stale_pipeline" in startup_src
        assert "target=_trigger_morning_briefing_if_missing" in startup_src


# ---------------------------------------------------------------------------
# Fix 5 — API key topbar display
# ---------------------------------------------------------------------------

class TestTopbarDisplay:

    def test_api_key_status_span_in_html(self):
        html = (WEBAPP / "index.html").read_text()
        assert 'id="api-key-status"' in html

    def test_api_client_updates_status_span(self):
        src = (WEBAPP / "js" / "api_client.js").read_text()
        assert "api-key-status" in src

    def test_api_client_sets_green_color(self):
        src = (WEBAPP / "js" / "api_client.js").read_text()
        assert "#22c55e" in src

    def test_api_client_sets_border_color(self):
        src = (WEBAPP / "js" / "api_client.js").read_text()
        assert "borderColor" in src


# ---------------------------------------------------------------------------
# Fix 6 — Morning briefing public
# ---------------------------------------------------------------------------

class TestMorningBriefingPublic:

    def test_morning_briefing_no_auth_required(self):
        with TestClient(app) as client:
            r = client.get("/jobs/briefing/morning")
        assert r.status_code == 200, f"Expected 200, got {r.status_code}"

    def test_morning_briefing_no_api_key_gate_in_js(self):
        js = (WEBAPP / "js" / "panels" / "morning_briefing.js").read_text()
        assert "Configure your API key" not in js

    def test_morning_briefing_router_no_global_auth(self):
        src = Path(__file__).parents[1].joinpath("api", "jobs", "morning_briefing.py").read_text()
        router_def_end = src.index("router = APIRouter(")
        closing = src.index(")", router_def_end)
        router_block = src[router_def_end:closing]
        assert "require_prosperity_api_key" not in router_block, \
            "Router-level auth must be removed from morning briefing router"

    def test_morning_briefing_post_still_protected(self):
        src = Path(__file__).parents[1].joinpath("api", "jobs", "morning_briefing.py").read_text()
        post_idx = src.index("@router.post")
        post_block = src[post_idx:post_idx + 200]
        assert "require_prosperity_api_key" in post_block, \
            "POST /morning must still require auth"


# ---------------------------------------------------------------------------
# Fix 7 — Opportunities loading
# ---------------------------------------------------------------------------

class TestOpportunitiesLoading:

    def test_universe_scores_returns_200(self):
        with TestClient(app) as client:
            r = client.get("/intelligence/universe/scores")
        assert r.status_code == 200

    def test_universe_scores_has_symbols(self):
        with TestClient(app) as client:
            r = client.get("/intelligence/universe/scores")
        rows = r.json()
        assert len(rows) > 0
        assert "symbol" in rows[0]

    def test_opportunities_js_has_console_log(self):
        src = (WEBAPP / "js" / "panels" / "opportunities.js").read_text()
        assert "console.log" in src

    def test_opportunities_js_has_pipeline_running_message(self):
        src = (WEBAPP / "js" / "panels" / "opportunities.js").read_text()
        assert "Pipeline running" in src
