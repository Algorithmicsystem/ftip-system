from __future__ import annotations

import os
import sys

import psycopg


def main() -> int:
    db_url = os.getenv("DATABASE_URL")
    if not db_url:
        print("DATABASE_URL is required to reset the database.", file=sys.stderr)
        return 1

    os.environ.setdefault("FTIP_DB_ENABLED", "1")
    os.environ.setdefault("FTIP_MIGRATIONS_AUTO", "1")

    with psycopg.connect(db_url, autocommit=True) as conn:
        with conn.cursor() as cur:
            cur.execute("DROP SCHEMA IF EXISTS public CASCADE")
            cur.execute("CREATE SCHEMA public")
            cur.execute("GRANT ALL ON SCHEMA public TO public")

    from api import migrations

    applied = migrations.ensure_schema()
    print(f"Reset complete. Applied migrations: {applied}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
