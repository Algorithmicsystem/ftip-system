import datetime as dt
import os
import uuid

import psycopg
import pytest

from api import db, migrations
from api.jobs import prosperity
from api import main


def _reset_schema() -> None:
    with psycopg.connect(os.environ["DATABASE_URL"], autocommit=True) as conn:
        with conn.cursor() as cur:
            cur.execute("DROP TABLE IF EXISTS schema_migrations CASCADE")
            cur.execute("DROP TABLE IF EXISTS ftip_job_runs CASCADE")


@pytest.mark.skipif(os.getenv("DATABASE_URL") is None, reason="DATABASE_URL not set")
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


@pytest.mark.skipif(os.getenv("DATABASE_URL") is None, reason="DATABASE_URL not set")
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
        "lock_acquired_at",
        "lock_expires_at",
        "created_at",
        "updated_at",
        "started_at",
        "finished_at",
        "status",
    ]:
        assert col in columns, f"expected column {col} in ftip_job_runs"


@pytest.mark.skipif(os.getenv("DATABASE_URL") is None, reason="DATABASE_URL not set")
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
                WHERE created_at IS NULL
                  OR updated_at IS NULL
                  OR started_at IS NULL
                  OR requested IS NULL
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

            cur.execute(
                """
                SELECT indexdef
                FROM pg_indexes
                WHERE schemaname = ANY (current_schemas(false))
                  AND tablename = 'ftip_job_runs'
                  AND indexname = 'ftip_job_runs_active_job'
                """
            )
            index_def = cur.fetchone()[0]

            upsert_run_id = uuid.uuid4()
            cur.execute(
                """
                INSERT INTO ftip_job_runs (
                    run_id,
                    job_name,
                    as_of_date,
                    started_at,
                    requested,
                    lock_owner
                )
                VALUES (%s, %s, %s, now(), '{}'::jsonb, 'tester')
                ON CONFLICT (job_name) WHERE finished_at IS NULL
                DO NOTHING
                RETURNING run_id
                """,
                (upsert_run_id, "legacy_job", dt.date(2024, 1, 1)),
            )
            conflict_upserted = cur.fetchone()

    assert {"as_of_date", "lock_acquired_at", "lock_expires_at"}.issubset(columns)
    assert null_timestamps == 0
    assert requested == {}
    assert lock_owner is None
    assert as_of_date is None
    assert "finished_at IS NULL" in index_def
    assert conflict_upserted is None


@pytest.mark.skipif(os.getenv("DATABASE_URL") is None, reason="DATABASE_URL not set")
def test_acquire_job_lock_uses_active_index(monkeypatch):
    os.environ.setdefault("FTIP_DB_ENABLED", "1")
    os.environ.setdefault("FTIP_DB_WRITE_ENABLED", "1")
    os.environ.setdefault("FTIP_DB_READ_ENABLED", "1")
    _reset_schema()

    migrations.ensure_schema()

    run_id = str(uuid.uuid4())
    as_of = dt.date(2024, 1, 1)

    acquired, info = prosperity._acquire_job_lock(
        run_id,
        prosperity.JOB_NAME,
        as_of,
        {"symbols": ["AAPL"]},
        30,
        "tester",
    )

    assert acquired is True
    assert info["run_id"] == run_id

    second_run_id = str(uuid.uuid4())
    acquired_second, info_second = prosperity._acquire_job_lock(
        second_run_id,
        prosperity.JOB_NAME,
        as_of,
        {"symbols": ["MSFT"]},
        30,
        "tester2",
    )

    assert acquired_second is False
    assert info_second["run_id"] == run_id

    with psycopg.connect(os.environ["DATABASE_URL"]) as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT COUNT(*) FROM ftip_job_runs WHERE job_name = %s AND finished_at IS NULL",
                (prosperity.JOB_NAME,),
            )
            active = cur.fetchone()[0]

    assert active == 1


def test_startup_fails_fast_when_ensure_schema_errors(monkeypatch):
    os.environ.setdefault("FTIP_DB_ENABLED", "1")

    def boom():
        raise RuntimeError("boom")

    monkeypatch.setattr(db, "db_enabled", lambda: True)
    monkeypatch.setattr(migrations, "ensure_schema", boom)
    monkeypatch.setattr(db, "ensure_schema", lambda: None)

    with pytest.raises(RuntimeError):
        main._startup()
