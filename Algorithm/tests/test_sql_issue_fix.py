import os
from pathlib import Path

import psycopg
import pytest

from api import migrations


@pytest.mark.skipif(os.getenv("DATABASE_URL") is None, reason="DATABASE_URL not set")
def test_prosperity_signals_primary_key_is_symbol_asof_lookback():
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

    assert columns == ["symbol", "as_of", "lookback"]


def test_prosperity_signals_v1_uniqueness_migration_dedupes_and_updates_pk():
    sql_path = Path(migrations.__file__).with_name(
        "023_prosperity_signals_v1_uniqueness.sql"
    )
    sql = sql_path.read_text()

    assert "row_number() OVER" in sql
    assert "PARTITION BY symbol" in sql
    assert "PRIMARY KEY (symbol, %I, lookback)" in sql
