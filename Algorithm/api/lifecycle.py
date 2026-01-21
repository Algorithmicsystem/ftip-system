from __future__ import annotations

import logging
from typing import List

from api import config, db, migrations

logger = logging.getLogger("ftip.api")


def startup() -> List[str]:
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
