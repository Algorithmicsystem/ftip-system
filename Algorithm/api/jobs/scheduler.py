"""Phase 10.5: Production-Grade Scheduler.

Uses APScheduler BackgroundScheduler with 8 weekday jobs.
Activated only when FTIP_SCHEDULER_ENABLED=1.
"""
from __future__ import annotations

import datetime as dt
import logging
from typing import Any, Dict, Optional

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
    "memory_consolidation",
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
        import httpx
        base = "http://localhost:8000"
        httpx.post(
            f"{base}/axiom/intraday/run",
            json={"symbols": [], "avg_daily_volume": 1_000_000},
            headers={"X-FTIP-API-Key": config.env("FTIP_API_KEY") or ""},
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
    from api import config
    try:
        import httpx
        base = "http://localhost:8000"
        today = dt.date.today().isoformat()
        key = config.env("FTIP_API_KEY") or ""
        headers = {"X-FTIP-API-Key": key}
        httpx.post(f"{base}/jobs/pnl/compute", json={"as_of_date": today, "horizons": [5, 21, 63], "store": True}, headers=headers, timeout=120)
        httpx.post(f"{base}/jobs/ic/daily-snapshot", json={"as_of_date": today}, headers=headers, timeout=60)
        httpx.post(f"{base}/jobs/breadth/daily-snapshot", json={"as_of_date": today}, headers=headers, timeout=60)
        logger.info("scheduler.full_daily_pipeline done date=%s", today)
    except Exception as exc:
        logger.warning("scheduler.full_daily_pipeline_failed error=%s", exc)


def _job_ml_training_check() -> None:
    from api.axiom.ml.training_job import run_training_job
    try:
        result = run_training_job(min_samples=50)
        logger.info("scheduler.ml_training_check status=%s", result.get("status"))
    except Exception as exc:
        logger.warning("scheduler.ml_training_check_failed error=%s", exc)


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
        s.add_job(_job_memory_consolidation, CronTrigger(day_of_week="mon-fri", hour=19, minute=0),
                  id="memory_consolidation", replace_existing=True)

        s.start()
        self._running = True
        logger.info("scheduler.started jobs=%d", len(list(s.get_jobs())))

    def stop(self) -> None:
        if self._scheduler and self._running:
            try:
                self._scheduler.shutdown(wait=False)
            except Exception:
                pass
        self._running = False
        logger.info("scheduler.stopped")

    def get_status(self) -> Dict[str, Any]:
        if not self._running or self._scheduler is None:
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


def start_scheduler() -> None:
    """Called from main.py lifespan if FTIP_SCHEDULER_ENABLED=1."""
    from api import config
    if config.env_bool("FTIP_SCHEDULER_ENABLED", False):
        scheduler_manager.start()


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
