import os

import psycopg
import pytest

from api import migrations


@pytest.mark.skipif(os.getenv("DATABASE_URL") is None, reason="DATABASE_URL not set")
def test_prosperity_signals_primary_key_includes_score_mode():
    os.environ.setdefault("FTIP_DB_ENABLED", "1")
    migrations.ensure_schema()

    with psycopg.connect(os.environ["DATABASE_URL"]) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT kcu.column_name
                FROM information_schema.table_constraints tc
                JOIN information_schema.key_column_usage kcu
                  ON tc.constraint_name = kcu.constraint_name
                WHERE tc.table_name = 'prosperity_signals_daily'
                  AND tc.constraint_type = 'PRIMARY KEY'
                ORDER BY kcu.ordinal_position
                """
            )
            columns = [row[0] for row in cur.fetchall()]

    assert "score_mode" in columns
