from __future__ import annotations

import os
import sys


def main() -> int:
    db_url = os.getenv("DATABASE_URL")
    if not db_url:
        print("DATABASE_URL is required to run migrations.", file=sys.stderr)
        return 1

    os.environ.setdefault("FTIP_DB_ENABLED", "1")
    os.environ.setdefault("FTIP_MIGRATIONS_AUTO", "1")

    from api import migrations

    applied = migrations.ensure_schema()
    print(f"Migrations applied: {applied}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
