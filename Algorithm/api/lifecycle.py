from __future__ import annotations

import logging
from typing import List

from api import config, db, migrations

logger = logging.getLogger("ftip.api")


def _missing_v1_tables() -> List[str]:
    required_tables = (
        "prosperity_universe",
        "prosperity_daily_bars",
        "prosperity_features_daily",
        "prosperity_signals_daily",
        "schema_migrations",
    )
    missing: List[str] = []
    for table in required_tables:
        row = db.safe_fetchone("SELECT to_regclass(%s)", (f"public.{table}",))
        if not row or row[0] is None:
            missing.append(table)
    return missing


def _enforce_db_runtime_contract() -> None:
    if not db.db_enabled():
        if config.db_required():
            raise RuntimeError(
                "FTIP_DB_REQUIRED=1 requires FTIP_DB_ENABLED=1 for the official v1 DB-backed path"
            )
        return

    if not config.env("DATABASE_URL"):
        raise RuntimeError("DATABASE_URL is required when FTIP_DB_ENABLED=1")

    if config.db_required():
        if not db.db_write_enabled() or not db.db_read_enabled():
            raise RuntimeError(
                "FTIP_DB_REQUIRED=1 requires FTIP_DB_WRITE_ENABLED=1 and FTIP_DB_READ_ENABLED=1"
            )
        row = db.safe_fetchone("SELECT 1")
        if not row or row[0] != 1:
            raise RuntimeError("database connectivity check failed (SELECT 1)")

        if not config.migrations_auto():
            missing_tables = _missing_v1_tables()
            if missing_tables:
                raise RuntimeError(
                    "official v1 tables are missing while FTIP_MIGRATIONS_AUTO=0; "
                    "run POST /prosperity/bootstrap before serving traffic. "
                    f"missing={','.join(missing_tables)}"
                )


def startup() -> List[str]:
    _enforce_db_runtime_contract()
    if not db.db_enabled():
        logger.info("[startup] database disabled; skipping migrations")
        return []
    if not config.migrations_auto():
        logger.info("[startup] migrations auto disabled; skipping")
        return []
    try:
        applied = migrations.ensure_schema()
        db.ensure_schema()
        if applied:
            logger.info("[startup] applied migrations", extra={"versions": applied})
        return applied
    except Exception as exc:
        logger.exception("[startup] ensure_schema failed", extra={"error": str(exc)})
        raise
