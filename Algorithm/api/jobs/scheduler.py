"""Daily pipeline scheduler — runs when FTIP_SCHEDULER_ENABLED=1."""
from __future__ import annotations
import datetime as dt
import logging
import threading
from typing import Any, Dict, Optional

from fastapi import APIRouter

router = APIRouter(prefix="/jobs/scheduler", tags=["scheduler"])
logger = logging.getLogger(__name__)

_last_run: Dict[str, Any] = {}
_next_run: Optional[dt.datetime] = None
_scheduler_thread: Optional[threading.Thread] = None


def _run_daily_pipeline() -> None:
    """Execute the full daily pipeline in order."""
    from api import config
    global _last_run
    start = dt.datetime.now(dt.timezone.utc)
    stages: Dict[str, Any] = {}

    def _stage(name: str, fn):
        try:
            result = fn()
            stages[name] = {"status": "ok", "result": result}
        except Exception as exc:
            logger.warning("scheduler.stage_failed stage=%s error=%s", name, exc)
            stages[name] = {"status": "error", "error": str(exc)}

    import httpx
    base = "http://localhost:8000"
    today = dt.date.today().isoformat()

    _stage("pnl_compute", lambda: httpx.post(f"{base}/jobs/pnl/compute", json={"as_of_date": today, "horizons": [5, 10, 21, 33], "store": True}, timeout=120).json())
    _stage("calibration_daily", lambda: httpx.post(f"{base}/jobs/calibration/daily", json={}, timeout=60).json())
    _stage("ic_gate", lambda: httpx.post(f"{base}/jobs/ic/daily-snapshot", json={"as_of_date": today}, timeout=60).json())
    _stage("breadth_daily", lambda: httpx.post(f"{base}/jobs/breadth/daily-snapshot", json={"as_of_date": today}, headers={"X-FTIP-API-Key": config.env("FTIP_API_KEY") or ""}, timeout=60).json())

    _last_run = {
        "started_at": start.isoformat(),
        "finished_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "stages": stages,
    }
    logger.info("scheduler.pipeline_complete stages=%d", len(stages))


def start_scheduler() -> None:
    """Start the APScheduler background scheduler if FTIP_SCHEDULER_ENABLED=1."""
    from api import config
    if not config.env_bool("FTIP_SCHEDULER_ENABLED", False):
        return
    try:
        from apscheduler.schedulers.background import BackgroundScheduler
        from apscheduler.triggers.cron import CronTrigger
    except ImportError:
        logger.warning("scheduler.disabled apscheduler_not_installed")
        return

    scheduler = BackgroundScheduler(timezone="UTC")
    scheduler.add_job(
        _run_daily_pipeline,
        CronTrigger(day_of_week="mon-fri", hour=18, minute=0),
        id="daily_pipeline",
        replace_existing=True,
    )
    scheduler.start()
    logger.info("scheduler.started next_run=%s",
                scheduler.get_job("daily_pipeline").next_run_time)


def get_status() -> Dict[str, Any]:
    return {
        "enabled": False,  # simplified; actual APScheduler check would need the scheduler object
        "last_run": _last_run or None,
    }


@router.get("/status")
def scheduler_status() -> Dict[str, Any]:
    """Return scheduler status and last pipeline run info."""
    return get_status()
