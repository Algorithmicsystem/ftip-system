"""Phase 10.5: Production-Grade Scheduler.

Uses APScheduler BackgroundScheduler with 8 weekday jobs.
Starts automatically in production; can be suppressed with FTIP_SCHEDULER_ENABLED=0.
"""
from __future__ import annotations

import datetime as dt
import logging
import os
import socket
import sys
import threading
from typing import Any, Dict, Optional
from uuid import uuid4

from fastapi import APIRouter

router = APIRouter(prefix="/jobs/scheduler", tags=["scheduler"])
logger = logging.getLogger(__name__)

# Canonical job IDs (exactly 9)
_JOB_IDS = {
    "morning_briefing",
    "intraday_update_10",
    "intraday_ic_10",
    "intraday_update_12",
    "intraday_update_14",
    "intraday_update_16",
    "full_daily_pipeline",
    "ml_training_check",
    "outcome_fill",
    "memory_consolidation",
    "ws_heartbeat",
}


# ---------------------------------------------------------------------------
# Job implementations
# ---------------------------------------------------------------------------

def _job_morning_briefing() -> None:
    from api.jobs.morning_briefing import generate_morning_briefing
    try:
        generate_morning_briefing()
        logger.info("scheduler.morning_briefing done")
    except Exception as exc:
        logger.warning("scheduler.morning_briefing_failed error=%s", exc)


def _job_intraday_update(hour_label: str) -> None:
    from api import config
    try:
        import os
        import httpx
        base = f"http://localhost:{os.environ.get('PORT', '8000')}"
        httpx.post(
            f"{base}/axiom/intraday/run",
            json={"symbols": [], "avg_daily_volume": 1_000_000},
            headers={"X-FTIP-API-Key": config.get_api_key()},
            timeout=60,
        )
        logger.info("scheduler.intraday_update hour=%s", hour_label)
    except Exception as exc:
        logger.warning("scheduler.intraday_update_failed hour=%s error=%s", hour_label, exc)


def _job_intraday_ic(hour: int) -> None:
    from api.jobs.intraday_ic import compute_intraday_ic, store_intraday_ic
    try:
        result = compute_intraday_ic(dt.date.today(), hour)
        store_intraday_ic(result)
        logger.info("scheduler.intraday_ic hour=%d ic=%s", hour, result.get("ic_value"))
    except Exception as exc:
        logger.warning("scheduler.intraday_ic_failed error=%s", exc)


def _job_full_daily_pipeline() -> None:
    from api.orchestration.pipeline_orchestrator import run_full_pipeline
    try:
        result = run_full_pipeline()
        logger.info("scheduler.full_pipeline run_id=%s status=%s", result.run_id, result.overall_status)
    except Exception as exc:
        logger.warning("scheduler.full_daily_pipeline_failed error=%s", exc)
    finally:
        try:
            from api.cloud.performance import perf_tracker
            perf_tracker.clear()
        except Exception:
            pass


def _job_ml_training_check() -> None:
    from api.axiom.ml.training_data import MINIMUM_SAMPLES_INITIAL
    from api.axiom.ml.training_job import run_training_job
    try:
        result = run_training_job(min_samples=MINIMUM_SAMPLES_INITIAL)
        status = result.get("status")
        logger.info("scheduler.ml_training_check status=%s", status)
        if status == "trained":
            # Invalidate ensemble weight cache and broadcast
            try:
                from api.axiom.ml.ensemble import invalidate_ensemble_cache
                invalidate_ensemble_cache()
            except Exception:
                pass
            try:
                import datetime as _dt
                from api.realtime.websocket_manager import ws_manager
                ws_manager.broadcast_from_thread({
                    "type": "ml_model_updated",
                    "version": result.get("model_version", ""),
                    "quality": result.get("model_quality", "bootstrap"),
                    "cv_auc": round(result.get("test_metrics", {}).get("roc_auc", 0.0), 3),
                    "samples": result.get("sample_count", 0),
                    "timestamp": _dt.datetime.utcnow().isoformat(),
                })
            except Exception:
                pass
    except Exception as exc:
        logger.warning("scheduler.ml_training_check_failed error=%s", exc)


def _job_ws_heartbeat() -> None:
    try:
        import datetime as _dt
        from api.realtime.websocket_manager import ws_manager
        ws_manager.broadcast_from_thread({
            "type": "heartbeat",
            "timestamp": _dt.datetime.utcnow().isoformat(),
            "connections": ws_manager.connection_count(),
        })
    except Exception as exc:
        logger.debug("scheduler.ws_heartbeat_failed error=%s", exc)


def _job_outcome_fill() -> None:
    from api.jobs.outcome_fill import run_outcome_fill
    try:
        result = run_outcome_fill()
        logger.info("scheduler.outcome_fill filled=%s skipped=%s", result.get("filled"), result.get("skipped"))
    except Exception as exc:
        logger.warning("scheduler.outcome_fill_failed error=%s", exc)


def _job_memory_consolidation() -> None:
    aod = dt.date.today()
    try:
        from api.intelligence.signal_memory import update_signal_performance_archive
        r1 = update_signal_performance_archive(aod)
        logger.info("scheduler.memory_consolidation signals_updated=%s", r1.get("updated"))
    except Exception as exc:
        logger.warning("scheduler.memory_consolidation_signals_failed error=%s", exc)
    try:
        from api.intelligence.regime_playbook import update_regime_playbook
        r2 = update_regime_playbook(aod)
        logger.info("scheduler.memory_consolidation regimes_updated=%s", r2.get("regimes_updated"))
    except Exception as exc:
        logger.warning("scheduler.memory_consolidation_playbook_failed error=%s", exc)
    try:
        from api.intelligence.company_dossier import run_dossier_update_job
        r3 = run_dossier_update_job(aod)
        logger.info("scheduler.memory_consolidation dossier_events=%s", r3.get("events_created"))
    except Exception as exc:
        logger.warning("scheduler.memory_consolidation_dossier_failed error=%s", exc)
    try:
        from api.pe.deal_flow import run_daily_deal_flow_screen
        run_daily_deal_flow_screen()
        logger.info("scheduler.deal_flow_screen_complete")
    except Exception as exc:
        logger.warning("scheduler.deal_flow_screen_failed err=%s", exc)


# ---------------------------------------------------------------------------
# SchedulerManager
# ---------------------------------------------------------------------------

class SchedulerManager:
    def __init__(self) -> None:
        self._scheduler = None
        self._running = False
        self._last_run_results: Dict[str, Any] = {}

    def start(self) -> None:
        try:
            from apscheduler.schedulers.background import BackgroundScheduler
            from apscheduler.triggers.cron import CronTrigger
        except ImportError:
            logger.warning("scheduler.disabled apscheduler_not_installed")
            return

        self._scheduler = BackgroundScheduler(timezone="America/New_York")
        s = self._scheduler

        s.add_job(_job_morning_briefing, CronTrigger(day_of_week="mon-fri", hour=7, minute=30),
                  id="morning_briefing", replace_existing=True)
        s.add_job(lambda: _job_intraday_update("10"), CronTrigger(day_of_week="mon-fri", hour=10, minute=5),
                  id="intraday_update_10", replace_existing=True)
        s.add_job(lambda: _job_intraday_ic(10), CronTrigger(day_of_week="mon-fri", hour=10, minute=10),
                  id="intraday_ic_10", replace_existing=True)
        s.add_job(lambda: _job_intraday_update("12"), CronTrigger(day_of_week="mon-fri", hour=12, minute=5),
                  id="intraday_update_12", replace_existing=True)
        s.add_job(lambda: _job_intraday_update("14"), CronTrigger(day_of_week="mon-fri", hour=14, minute=5),
                  id="intraday_update_14", replace_existing=True)
        s.add_job(lambda: _job_intraday_update("16"), CronTrigger(day_of_week="mon-fri", hour=16, minute=5),
                  id="intraday_update_16", replace_existing=True)
        s.add_job(_job_full_daily_pipeline, CronTrigger(day_of_week="mon-fri", hour=18, minute=0),
                  id="full_daily_pipeline", replace_existing=True)
        s.add_job(_job_ml_training_check, CronTrigger(day_of_week="mon-fri", hour=18, minute=30),
                  id="ml_training_check", replace_existing=True)
        s.add_job(_job_outcome_fill, CronTrigger(day_of_week="mon-fri", hour=18, minute=45),
                  id="outcome_fill", replace_existing=True)
        s.add_job(_job_memory_consolidation, CronTrigger(day_of_week="mon-fri", hour=19, minute=0),
                  id="memory_consolidation", replace_existing=True)
        s.add_job(_job_ws_heartbeat, trigger="interval", seconds=30,
                  id="ws_heartbeat", replace_existing=True)

        s.start()
        self._running = True
        logger.info("scheduler.started jobs=%d", len(list(s.get_jobs())))

    @property
    def running(self) -> bool:
        if not self._running or self._scheduler is None:
            return False
        return bool(getattr(self._scheduler, "running", False))

    def stop(self) -> None:
        if not self._running:
            return  # never started — nothing to stop
        if self._scheduler:
            try:
                self._scheduler.shutdown(wait=False)
            except Exception:
                pass
        self._running = False
        logger.info("scheduler.stopped")

    def get_status(self) -> Dict[str, Any]:
        if not self.running or self._scheduler is None:
            return {"running": False, "next_run_times": {}, "last_run_results": self._last_run_results}

        next_runs: Dict[str, Any] = {}
        for job in self._scheduler.get_jobs():
            nrt = job.next_run_time
            next_runs[job.id] = nrt.isoformat() if nrt else None

        return {
            "running": True,
            "next_run_times": next_runs,
            "last_run_results": self._last_run_results,
        }

    def trigger_job(self, job_id: str) -> Dict[str, Any]:
        if job_id not in _JOB_IDS:
            return {"status": "error", "job_id": job_id, "error": "job_not_found"}

        if not self._running or self._scheduler is None:
            return {"status": "error", "job_id": job_id, "error": "scheduler_not_running"}

        try:
            job = self._scheduler.get_job(job_id)
            if job is None:
                return {"status": "error", "job_id": job_id, "error": "job_not_found"}
            job.modify(next_run_time=dt.datetime.now(dt.timezone.utc))
            return {"status": "triggered", "job_id": job_id}
        except Exception as exc:
            return {"status": "error", "job_id": job_id, "error": str(exc)}


# Module-level singleton
scheduler_manager = SchedulerManager()

# Stable worker identity for this process
_SCHEDULER_WORKER_ID: Optional[str] = None

# Process-level guard: only ONE thread in this process may call start_scheduler.
# Prevents the watchdog from racing against itself or multiple lifespan calls.
_SCHEDULER_THREAD_LOCK = threading.Lock()
_SCHEDULER_STARTED_IN_THIS_PROCESS = False


def _get_worker_id() -> str:
    global _SCHEDULER_WORKER_ID
    if _SCHEDULER_WORKER_ID is None:
        _SCHEDULER_WORKER_ID = f"{socket.gethostname()}:{os.getpid()}:{uuid4().hex[:8]}"
    return _SCHEDULER_WORKER_ID


_LOCK_TABLE_ENSURED = False


def _ensure_scheduler_lock_table() -> None:
    """Migrate to single-row lock schema (lock_id=1 pattern) if needed."""
    global _LOCK_TABLE_ENSURED
    if _LOCK_TABLE_ENSURED:
        return
    from api import db
    try:
        # Check for old schema (no lock_id column → worker_id TEXT PRIMARY KEY)
        row = db.safe_fetchone(
            "SELECT column_name FROM information_schema.columns "
            "WHERE table_name = 'scheduler_lock' AND column_name = 'lock_id'"
        )
        if row is None:
            # Old schema or missing table — drop and recreate (transient data, safe)
            db.safe_execute("DROP TABLE IF EXISTS scheduler_lock")
        db.safe_execute(
            """
            CREATE TABLE IF NOT EXISTS scheduler_lock (
                lock_id      INTEGER PRIMARY KEY DEFAULT 1,
                worker_id    TEXT NOT NULL,
                locked_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
                heartbeat_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                CHECK (lock_id = 1)
            )
            """
        )
        _LOCK_TABLE_ENSURED = True
    except Exception:
        pass


def _should_run_scheduler() -> bool:
    """Return True only if this worker holds the single-row scheduler lock.

    Uses a UPSERT on lock_id=1 that is safe under pgbouncer transaction pooling:
    only one row can ever exist, so the conflict is globally atomic — unlike the
    old worker_id-per-row pattern where each worker won its own INSERT.
    """
    from api import db
    if not db.db_enabled():
        return True

    _ensure_scheduler_lock_table()
    worker_id = _get_worker_id()

    try:
        # Claim the single lock row only if no other live worker holds it
        # (heartbeat_at < 90s means the current holder is still alive)
        row = db.safe_fetchone(
            """
            INSERT INTO scheduler_lock (lock_id, worker_id, locked_at, heartbeat_at)
            VALUES (1, %s, now(), now())
            ON CONFLICT (lock_id) DO UPDATE
                SET worker_id    = EXCLUDED.worker_id,
                    locked_at    = now(),
                    heartbeat_at = now()
                WHERE scheduler_lock.heartbeat_at < now() - interval '90 seconds'
            RETURNING worker_id
            """,
            (worker_id,),
        )
        if row is not None:
            return str(row[0]) == worker_id
        # UPSERT skipped (another live worker holds it) — check if it's us
        existing = db.safe_fetchone(
            "SELECT worker_id FROM scheduler_lock WHERE lock_id = 1"
        )
        return bool(existing and str(existing[0]) == worker_id)
    except Exception:
        return True


def _scheduler_heartbeat_loop() -> None:
    """Update table-based lock heartbeat every 30 s; log liveness every 60 s."""
    import time
    tick = 0
    while True:
        time.sleep(30)
        tick += 1
        # Keep our lock row alive
        try:
            from api import db
            if db.db_enabled():
                db.safe_execute(
                    "UPDATE scheduler_lock SET heartbeat_at = now() WHERE lock_id = 1 AND worker_id = %s",
                    (_get_worker_id(),),
                )
        except Exception:
            pass
        if tick % 2 == 0:
            try:
                status = scheduler_manager.get_status()
                running = status.get("running", False)
                n_jobs = len(status.get("next_run_times", {}))
                logger.info("scheduler.alive running=%s jobs=%d", running, n_jobs)
            except Exception:
                pass


def _scheduler_watchdog() -> None:
    """Restart the scheduler if it stops unexpectedly.

    Only restarts if THIS process originally won the scheduler lock (i.e.,
    _SCHEDULER_STARTED_IN_THIS_PROCESS is True). Workers that lost the
    initial lock race must never start the scheduler, even via watchdog.
    Checks the DB lock before restarting to guard against multi-worker races.
    """
    import time
    while True:
        time.sleep(30)
        if "pytest" in sys.modules:
            break
        if not _SCHEDULER_STARTED_IN_THIS_PROCESS:
            # This process lost the lock race — exit the watchdog thread entirely.
            return
        if not scheduler_manager.running:
            if not _should_run_scheduler():
                logger.info("scheduler.watchdog skipped — lost lock to another worker")
                continue
            logger.warning("scheduler.watchdog detected stopped scheduler — restarting")
            try:
                scheduler_manager.start()
            except Exception as exc:
                logger.error("scheduler.watchdog restart failed: %s", exc)


def start_scheduler() -> None:
    """Start the scheduler. Suppressed in pytest; can be disabled via FTIP_SCHEDULER_ENABLED=0.

    Uses a process-level lock so only ONE call per process can start the scheduler,
    regardless of how many threads or lifespan events trigger it concurrently.

    To trigger the pipeline manually from Railway console:
      python -c "from api.jobs.scheduler import _job_full_daily_pipeline; _job_full_daily_pipeline()"
    """
    global _SCHEDULER_STARTED_IN_THIS_PROCESS

    if "pytest" in sys.modules:
        return
    from api import config
    if config.env("FTIP_SCHEDULER_ENABLED") == "0":
        logger.info("scheduler.disabled FTIP_SCHEDULER_ENABLED=0")
        return

    with _SCHEDULER_THREAD_LOCK:
        if _SCHEDULER_STARTED_IN_THIS_PROCESS:
            logger.info("scheduler.already_started_in_this_process")
            return
        if not _should_run_scheduler():
            logger.info("scheduler.skipped — another worker holds the lock")
            return
        scheduler_manager.start()
        _SCHEDULER_STARTED_IN_THIS_PROCESS = True

    threading.Thread(target=_scheduler_heartbeat_loop, daemon=True).start()
    threading.Thread(target=_scheduler_watchdog, daemon=True).start()


def get_status() -> Dict[str, Any]:
    return scheduler_manager.get_status()


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@router.get("/status")
def scheduler_status() -> Dict[str, Any]:
    return get_status()


@router.post("/trigger/{job_id}")
def trigger_job(job_id: str) -> Dict[str, Any]:
    return scheduler_manager.trigger_job(job_id)
