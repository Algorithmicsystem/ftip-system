import os
from pathlib import Path

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


def test_prosperity_signals_pk_migration_avoids_ambiguous_constraint_name():
    sql_path = Path(migrations.__file__).with_name("019_fix_prosperity_signals_score_mode_pk.sql")
    sql = sql_path.read_text()

    assert "v_existing_pk" in sql
    assert "constraint_name text" not in sql
    assert "information_schema.table_constraints" not in sql
