from __future__ import annotations

import logging
from pathlib import Path
from typing import List

from api import config, db

logger = logging.getLogger(__name__)

MIGRATIONS_DIR = Path(__file__).resolve().parent


def _load_migration_statements(path: Path) -> List[str]:
    sql = path.read_text()
    statements = []
    for stmt in sql.split(";"):
        cleaned = stmt.strip()
        if cleaned:
            statements.append(cleaned)
    return statements


def apply_migrations() -> None:
    if not config.db_enabled() or not config.migrations_auto():
        return

    pool = db.get_pool()
    paths = sorted(MIGRATIONS_DIR.glob("*.sql"))
    if not paths:
        logger.info("[migrations] no migration files found")
        return

    for path in paths:
        statements = _load_migration_statements(path)
        logger.info("[migrations] applying %s (%d statements)", path.name, len(statements))
        with pool.connection(timeout=10) as conn:
            with conn.cursor() as cur:
                for stmt in statements:
                    cur.execute(stmt)
            conn.commit()

    logger.info("[migrations] completed")
