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
