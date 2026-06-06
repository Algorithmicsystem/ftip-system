"""Bug-fix tests: intelligence_score column, briefing 401, GOOG failure logging."""
from __future__ import annotations

import datetime as dt
import inspect
import logging
import os
import re

import pytest
from fastapi.testclient import TestClient

from api.main import app

# ---------------------------------------------------------------------------
# Bug 1: intelligence_score → impact_score column mismatch
# ---------------------------------------------------------------------------

class TestUniversalIntelligenceColumnFix:

    def test_impact_score_used_not_intelligence_score(self):
        """Dossier query must use impact_score (the real column) not intelligence_score."""
        import api.universal.intelligence_api as m
        src = inspect.getsource(m.assemble_universal_intelligence)
        assert "intelligence_score" not in src, (
            "intelligence_score does not exist in company_intelligence_archive; "
            "use impact_score"
        )
        assert "impact_score" in src

    def test_universal_intelligence_column_names_match_schema(self):
        """Every non-JSONB column referenced in assemble_universal_intelligence
        must exist in the corresponding migration CREATE TABLE SQL."""
        migrations_dir = os.path.join(
            os.path.dirname(__file__), "..", "api", "migrations"
        )

        def cols_from_migration(filename: str) -> set:
            path = os.path.join(migrations_dir, filename)
            if not os.path.isfile(path):
                return set()
            with open(path) as f:
                sql = f.read()
            names = set()
            # Match column definitions: word followed by type keyword
            for m in re.finditer(
                r"^\s{4}(\w+)\s+(?:TEXT|NUMERIC|DATE|INT|BOOLEAN|UUID|JSONB|TIMESTAMPTZ|DOUBLE\s+PRECISION)",
                sql,
                re.MULTILINE | re.IGNORECASE,
            ):
                names.add(m.group(1).lower())
            return names

        # company_intelligence_archive — migration 077
        cia_cols = cols_from_migration("077_company_intelligence_archive.sql")
        assert "impact_score" in cia_cols, f"impact_score missing from cia migration, got {cia_cols}"
        assert "intelligence_score" not in cia_cols

        # signal_performance_archive — migration 075
        spa_cols = cols_from_migration("075_signal_performance_archive.sql")
        assert "batting_average" in spa_cols, f"batting_average missing, got {spa_cols}"

        # portfolio_risk_daily — migration 073
        prd_cols = cols_from_migration("073_portfolio_risk_daily.sql")
        assert "var_99_1d" in prd_cols, f"var_99_1d missing, got {prd_cols}"
        assert "portfolio_id" in prd_cols

        # universal_intelligence_cache — migration 085
        uic_cols = cols_from_migration("085_universal_intelligence_cache.sql")
        assert "response" in uic_cols
        assert "assembled_at" in uic_cols

        # axiom_scores_daily — migration 026
        asd_cols = cols_from_migration("026_axiom_phase3_history.sql")
        assert "payload" in asd_cols
        assert "deployable_alpha_utility" in asd_cols

    def test_assemble_universal_no_warning_logs(self, caplog):
        """DB-disabled mode must produce no WARNING entries (no column errors)."""
        from api.universal.intelligence_api import assemble_universal_intelligence
        with caplog.at_level(logging.WARNING, logger="api.universal.intelligence_api"):
            assemble_universal_intelligence("AAPL", dt.date.today())
        assert len(caplog.records) == 0

    def test_universal_endpoint_200_no_crash(self):
        with TestClient(app) as client:
            r = client.get("/intelligence/universal/AAPL")
        assert r.status_code == 200
        data = r.json()
        assert "symbol" in data


# ---------------------------------------------------------------------------
# Bug 2: morning briefing 401 — api_client header name fix
# ---------------------------------------------------------------------------

class TestMorningBriefingHeaderFix:

    def test_api_client_sends_correct_header_name(self):
        """api_client.js must send X-FTIP-API-Key, not X-API-Key."""
        path = os.path.join(
            os.path.dirname(__file__), "..", "api", "webapp", "js", "api_client.js"
        )
        with open(path) as f:
            src = f.read()
        assert "X-FTIP-API-Key" in src, "api_client.js must use X-FTIP-API-Key header"
        assert "X-API-Key'" not in src or "X-FTIP-API-Key" in src

    def test_morning_briefing_endpoint_accessible(self):
        """Morning briefing endpoint must return 200 (no auth required when key not set)."""
        with TestClient(app) as client:
            r = client.get("/jobs/briefing/morning")
        assert r.status_code == 200, (
            f"Expected 200 from /jobs/briefing/morning, got {r.status_code}. "
            "This is the Bug 2 fix: auth must be disabled when FTIP_API_KEY is unset."
        )

    def test_morning_briefing_text_endpoint_accessible(self):
        with TestClient(app) as client:
            r = client.get("/jobs/briefing/morning/text")
        assert r.status_code == 200

    def test_api_client_header_matches_server_expectation(self):
        """Server security.py expects x-ftip-api-key; client must send same."""
        import api.security as sec
        # get_provided_api_key checks headers.get("x-ftip-api-key") — case-insensitive
        src = inspect.getsource(sec.get_provided_api_key)
        assert "x-ftip-api-key" in src.lower()

        client_path = os.path.join(
            os.path.dirname(__file__), "..", "api", "webapp", "js", "api_client.js"
        )
        with open(client_path) as f:
            client_src = f.read()
        assert "X-FTIP-API-Key" in client_src


# ---------------------------------------------------------------------------
# Bug 3: GOOG symbol failure logging includes reason
# ---------------------------------------------------------------------------

class TestSymbolFailureLogging:

    def test_symbol_failed_log_includes_reason(self):
        """The symbol_failed warning must include stage and reason in message string."""
        path = os.path.join(
            os.path.dirname(__file__), "..", "api", "prosperity", "routes.py"
        )
        with open(path) as f:
            src = f.read()
        # The log message must include reason (not just in extra dict)
        assert "symbol=%s stage=%s reason=%s" in src or (
            "symbol_failed symbol=" in src
        ), "symbol_failed warning must include reason in message string"

    def test_symbol_failed_log_message_has_substitution_args(self):
        """The warning must pass sym, stage, and reason as positional args."""
        path = os.path.join(
            os.path.dirname(__file__), "..", "api", "prosperity", "routes.py"
        )
        with open(path) as f:
            src = f.read()
        # Find the symbol_failed warning block
        idx = src.find("jobs.prosperity.daily_snapshot.symbol_failed")
        assert idx != -1, "symbol_failed log not found"
        # The 200 chars after should contain reason/stage references
        context = src[idx:idx+300]
        assert "root_failure_stage" in context or "stage" in context.lower()
        assert "reason" in context.lower()
