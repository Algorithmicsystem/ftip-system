from __future__ import annotations

import logging
import sys
from typing import List

from api import config, db, migrations

logger = logging.getLogger("ftip.api")

_STALE_PIPELINE_TRIGGERED = False


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


def _check_and_trigger_stale_pipeline() -> None:
    """If no data today, trigger pipeline automatically after startup.

    Guards:
      1. Only runs once per process lifetime.
      2. Only triggers between 06:00–23:00 UTC (avoids overnight deploys).
      3. Skips if a pipeline already ran within the last 4 hours
         (prevents multiple rapid deploys from queuing duplicate runs).
    """
    global _STALE_PIPELINE_TRIGGERED
    if "pytest" in sys.modules:
        return
    if _STALE_PIPELINE_TRIGGERED:
        return
    _STALE_PIPELINE_TRIGGERED = True
    import datetime as dt
    import time
    time.sleep(5)
    try:
        if not db.db_enabled():
            return

        now_utc = dt.datetime.utcnow()
        if not (6 <= now_utc.hour < 23):
            logger.info(
                "startup.pipeline_skipped reason=outside_window utc_hour=%d",
                now_utc.hour,
            )
            return

        row = db.safe_fetchone("SELECT MAX(as_of_date) FROM axiom_scores_daily")
        if not row or not row[0]:
            return
        last_date = row[0]
        today = dt.date.today()
        if last_date >= today:
            return  # already scored today

        # Guard: skip if a score was written for today within the last 4 hours
        # (catches the case where a prior deploy already triggered the pipeline)
        recent_row = db.safe_fetchone(
            """SELECT MAX(updated_at) FROM axiom_scores_daily
               WHERE updated_at >= NOW() - INTERVAL '4 hours'"""
        )
        if recent_row and recent_row[0]:
            logger.info(
                "startup.pipeline_skipped reason=recent_run last_update=%s",
                recent_row[0],
            )
            return

        logger.info("startup.data_stale last=%s triggering_pipeline", last_date)
        try:
            from api.realtime.websocket_manager import ws_manager
            ws_manager.broadcast_from_thread({
                "type": "pipeline_starting",
                "reason": "stale_data",
                "last_run": str(last_date),
                "message": f"Data from {last_date} — starting pipeline update...",
            })
        except Exception:
            pass
        from api.jobs.scheduler import _job_full_daily_pipeline
        import threading
        threading.Thread(target=_job_full_daily_pipeline, daemon=True).start()
    except Exception as exc:
        logger.debug("startup_pipeline_check_failed err=%s", exc)


def startup() -> List[str]:
    _enforce_db_runtime_contract()
    if not db.db_enabled():
        logger.info("[startup] database disabled; skipping migrations")
        return []
    applied: List[str] = []
    try:
        applied = migrations.ensure_schema()
        db.ensure_schema()
        if applied:
            logger.info("[startup] applied %d migrations", len(applied))
        else:
            logger.info("[startup] schema up to date")
    except Exception as exc:
        logger.error("[startup] migration error: %s", exc)
        # Don't crash — continue so the stale-data check still fires

    # Safety net: ensure axiom_scores_daily has all columns the INSERT needs.
    # Railway may have an older table definition missing columns added after initial deploy.
    _axiom_cols = [
        ("deployable_alpha_utility", "numeric"),
        ("regime_label", "text"),
        ("payload", "jsonb"),
        ("outcome_payload", "jsonb"),
        ("build_meta", "jsonb"),
        ("signal_version", "text"),
        ("feature_version", "text"),
        ("snapshot_version", "text"),
        ("snapshot_id", "text"),
        ("evidence_backed_deployability_tier", "text"),
        ("trade_family", "text"),
        ("deployability_tier", "text"),
        ("size_band", "text"),
        ("gross_opportunity", "numeric"),
        ("friction_burden", "numeric"),
        ("validated_edge", "numeric"),
        ("overall_coverage", "numeric"),
        ("overall_confidence", "numeric"),
        ("ml_adjusted_dau", "numeric"),
        ("moat_score", "numeric"),
        ("intelligence_quality_score", "numeric"),
    ]
    for _col, _typ in _axiom_cols:
        try:
            db.safe_execute(
                f"ALTER TABLE axiom_scores_daily ADD COLUMN IF NOT EXISTS {_col} {_typ}"
            )
        except Exception as _e:
            logger.warning("[startup] schema_fix_failed col=%s err=%s", _col, _e)
    logger.info("[startup] axiom_scores_daily columns verified")

    import threading
    threading.Thread(target=_check_and_trigger_stale_pipeline, daemon=True).start()
    threading.Thread(target=_trigger_morning_briefing_if_missing, daemon=True).start()
    return applied


def _trigger_morning_briefing_if_missing() -> None:
    """Generate today's morning briefing if it hasn't been generated yet."""
    import datetime as dt
    import time
    if "pytest" in sys.modules:
        return
    time.sleep(10)
    try:
        if not db.db_enabled():
            return
        today = dt.date.today()
        row = db.safe_fetchone(
            "SELECT 1 FROM morning_briefings WHERE briefing_date = %s LIMIT 1", (today,)
        )
        if not row:
            logger.info("startup.morning_briefing_missing triggering generation")
            from api.jobs.morning_briefing import generate_morning_briefing
            generate_morning_briefing(today)
    except Exception as exc:
        logger.debug("startup_briefing_check_failed err=%s", exc)
