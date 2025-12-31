import os

import psycopg
import pytest

from api import migrations


pytestmark = pytest.mark.skipif(
    os.getenv("DATABASE_URL") is None, reason="DATABASE_URL not set"
)


def _reset_schema() -> None:
    with psycopg.connect(os.environ["DATABASE_URL"], autocommit=True) as conn:
        with conn.cursor() as cur:
            cur.execute("DROP TABLE IF EXISTS schema_migrations CASCADE")
            cur.execute("DROP TABLE IF EXISTS ftip_job_runs CASCADE")


def test_migrations_commit_versions(monkeypatch):
    os.environ.setdefault("FTIP_DB_ENABLED", "1")
    _reset_schema()

    applied = migrations.ensure_schema()

    with psycopg.connect(os.environ["DATABASE_URL"]) as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT version FROM schema_migrations ORDER BY version")
            versions = [row[0] for row in cur.fetchall()]

    assert versions, "expected schema versions recorded"
    for version, _ in migrations.MIGRATIONS:
        assert version in versions, f"missing migration version {version}"
    assert set(applied).issubset(set(versions))


def test_ftip_job_runs_schema_contains_required_columns(monkeypatch):
    os.environ.setdefault("FTIP_DB_ENABLED", "1")
    _reset_schema()

    migrations.ensure_schema()

    with psycopg.connect(os.environ["DATABASE_URL"]) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT column_name
                FROM information_schema.columns
                WHERE table_name = 'ftip_job_runs'
                """
            )
            columns = {row[0] for row in cur.fetchall()}

    for col in [
        "as_of_date",
        "requested",
        "lock_owner",
        "started_at",
        "finished_at",
        "status",
    ]:
        assert col in columns, f"expected column {col} in ftip_job_runs"


def test_ftip_job_runs_migration_backfills_nullable_columns(monkeypatch):
    os.environ.setdefault("FTIP_DB_ENABLED", "1")
    _reset_schema()

    with psycopg.connect(os.environ["DATABASE_URL"], autocommit=True) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                CREATE TABLE ftip_job_runs (
                    run_id UUID PRIMARY KEY,
                    job_name TEXT,
                    created_at TIMESTAMPTZ
                )
                """
            )
            cur.execute(
                """
                INSERT INTO ftip_job_runs (run_id, job_name, created_at)
                VALUES (%s, %s, NULL)
                """,
                ("00000000-0000-0000-0000-000000000001", "legacy_job"),
            )

    migrations.ensure_schema()

    with psycopg.connect(os.environ["DATABASE_URL"]) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT column_name
                FROM information_schema.columns
                WHERE table_name = 'ftip_job_runs'
                """
            )
            columns = {row[0] for row in cur.fetchall()}

            cur.execute(
                """
                SELECT COUNT(*)
                FROM ftip_job_runs
                WHERE created_at IS NULL OR updated_at IS NULL
                """
            )
            null_timestamps = cur.fetchone()[0]

            cur.execute(
                """
                SELECT as_of_date, requested, lock_owner
                FROM ftip_job_runs
                WHERE run_id = %s
                """,
                ("00000000-0000-0000-0000-000000000001",),
            )
            as_of_date, requested, lock_owner = cur.fetchone()

    assert {"as_of_date", "lock_acquired_at", "lock_expires_at"}.issubset(columns)
    assert null_timestamps == 0
    assert requested == {}
    assert lock_owner == "unknown"
    assert as_of_date is not None
