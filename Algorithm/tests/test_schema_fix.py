"""Schema fix tests: axiom_scores_daily missing columns, error logging visibility."""
from __future__ import annotations

import inspect
from pathlib import Path

import pytest


MIGRATIONS_DIR = Path(__file__).resolve().parents[1] / "api" / "migrations"
MIGRATION_FILE = MIGRATIONS_DIR / "106_axiom_scores_columns.sql"


class TestMigrationFile:

    def test_migration_106_exists(self):
        assert MIGRATION_FILE.exists(), "Migration 106_axiom_scores_columns.sql must exist"

    def test_migration_has_deployable_alpha_utility(self):
        sql = MIGRATION_FILE.read_text()
        assert "deployable_alpha_utility" in sql

    def test_migration_has_payload(self):
        sql = MIGRATION_FILE.read_text()
        assert "payload" in sql

    def test_migration_has_regime_label(self):
        sql = MIGRATION_FILE.read_text()
        assert "regime_label" in sql

    def test_migration_has_moat_score(self):
        sql = MIGRATION_FILE.read_text()
        assert "moat_score" in sql

    def test_migration_uses_add_column_if_not_exists(self):
        sql = MIGRATION_FILE.read_text()
        assert "ADD COLUMN IF NOT EXISTS" in sql

    def test_migration_covers_all_insert_columns(self):
        """Every column in persistence.py INSERT that could be missing must appear in migration."""
        required = [
            "deployable_alpha_utility",
            "regime_label",
            "payload",
            "outcome_payload",
            "build_meta",
            "signal_version",
            "feature_version",
            "overall_coverage",
            "overall_confidence",
        ]
        sql = MIGRATION_FILE.read_text()
        for col in required:
            assert col in sql, f"Migration missing column: {col}"


class TestStartupSafetyNet:

    def test_lifecycle_has_add_column_if_not_exists(self):
        src = Path(__file__).parents[1].joinpath("api", "lifecycle.py").read_text()
        assert "ADD COLUMN IF NOT EXISTS" in src

    def test_lifecycle_covers_deployable_alpha_utility(self):
        src = Path(__file__).parents[1].joinpath("api", "lifecycle.py").read_text()
        assert "deployable_alpha_utility" in src

    def test_lifecycle_covers_moat_score(self):
        src = Path(__file__).parents[1].joinpath("api", "lifecycle.py").read_text()
        assert "moat_score" in src

    def test_lifecycle_covers_payload(self):
        src = Path(__file__).parents[1].joinpath("api", "lifecycle.py").read_text()
        assert "payload" in src

    def test_lifecycle_safety_net_after_ensure_schema(self):
        src = Path(__file__).parents[1].joinpath("api", "lifecycle.py").read_text()
        ensure_idx = src.index("ensure_schema()")
        add_col_idx = src.index("ADD COLUMN IF NOT EXISTS")
        assert add_col_idx > ensure_idx, "Safety net must come after ensure_schema()"

    def test_lifecycle_logs_columns_verified(self):
        src = Path(__file__).parents[1].joinpath("api", "lifecycle.py").read_text()
        assert "axiom_scores_daily columns verified" in src


class TestPipelineErrorVisibility:

    def test_pipeline_orchestrator_uses_logger_error_not_debug(self):
        src = Path(__file__).parents[1].joinpath(
            "api", "orchestration", "pipeline_orchestrator.py"
        ).read_text()
        # Must not swallow stage errors silently
        assert 'logger.debug("pipeline_stage_inner_error' not in src

    def test_pipeline_orchestrator_logs_stage_errors_as_error(self):
        src = Path(__file__).parents[1].joinpath(
            "api", "orchestration", "pipeline_orchestrator.py"
        ).read_text()
        assert 'logger.error("pipeline_stage_inner_error' in src
