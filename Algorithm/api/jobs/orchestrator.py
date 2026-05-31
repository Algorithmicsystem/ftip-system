"""Session 17: Daily pipeline orchestrator.

POST /jobs/daily-run  — chains all daily jobs in sequence:
  1. market breadth
  2. sector breadth
  3. IC snapshot
  4. alert scan
  5. universe screen

Each stage runs in a try/except block so a single failure does not abort
the rest. Returns a structured DailyDigest with per-stage results, a
human-readable headline, and the top opportunities from the screen stage.
"""
from __future__ import annotations

import datetime as dt
import logging
import time
from typing import Any, Callable, Dict, List, Optional

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from api import db, security

router = APIRouter(
    prefix="/jobs",
    tags=["jobs"],
    dependencies=[Depends(security.require_prosperity_api_key)],
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Stage runner helper
# ---------------------------------------------------------------------------

def _run_stage(
    name: str,
    fn: Callable[[], Dict[str, Any]],
    stages: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    t0 = time.monotonic()
    try:
        result = fn()
        stages[name] = {
            "status": "ok",
            "duration_ms": round((time.monotonic() - t0) * 1000),
            **result,
        }
        return result
    except Exception as exc:
        stages[name] = {
            "status": "error",
            "error": str(exc),
            "duration_ms": round((time.monotonic() - t0) * 1000),
        }
        logger.warning("orchestrator.stage_failed stage=%s error=%s", name, exc)
        return None


# ---------------------------------------------------------------------------
# Headline builder
# ---------------------------------------------------------------------------

def _build_headline(stages: Dict[str, Any]) -> str:
    breadth   = stages.get("market_breadth", {})
    screen    = stages.get("screen", {})
    alerts    = stages.get("alerts", {})
    ic_snap   = stages.get("ic_snapshot", {})
    pnl       = stages.get("signal_pnl", {})
    providers = stages.get("provider_reliability", {})
    ic_cal    = stages.get("ic_calibration", {})

    breadth_state = (
        breadth.get("breadth_state")
        or screen.get("breadth_state")
        or "UNKNOWN"
    )
    ic_state = screen.get("ic_state") or ic_cal.get("ic_state") or "UNKNOWN"
    fired = alerts.get("fired", 0)
    count = screen.get("count", 0)
    ic_rows = ic_snap.get("rows_written", 0)
    pnl_rows = pnl.get("rows_stored", 0)
    degraded = providers.get("degraded") or []
    hit_rate = ic_cal.get("hit_rate")

    parts = [f"{breadth_state} breadth, {ic_state} IC."]
    if ic_rows:
        parts.append(f"IC updated ({ic_rows} rows).")
    if hit_rate is not None:
        parts.append(f"Kelly hit-rate {hit_rate:.2f}.")
    if fired:
        parts.append(f"{fired} alert{'s' if fired != 1 else ''} fired.")
    if count:
        parts.append(f"{count} conviction candidate{'s' if count != 1 else ''} in screen.")
    if pnl_rows:
        parts.append(f"P&L updated ({pnl_rows} rows).")
    if degraded:
        parts.append(f"Provider degraded: {', '.join(degraded)}.")
    return " ".join(parts)


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def run_daily_pipeline(
    as_of_date: dt.date,
    *,
    lookback: int = 252,
    min_dau: float = 0.0,
    screen_limit: int = 20,
    skip_stages: Optional[set] = None,
) -> Dict[str, Any]:
    started_at = dt.datetime.utcnow().isoformat() + "Z"
    t_start = time.monotonic()
    stages: Dict[str, Any] = {}
    _skip = skip_stages or set()

    def _maybe_run(name: str, fn: Callable[[], Dict[str, Any]]) -> None:
        if name in _skip:
            stages[name] = {"status": "skipped"}
        else:
            _run_stage(name, fn, stages)

    write_ok = db.db_write_enabled()

    # ------------------------------------------------------------------
    # Stage 1: Market breadth
    # ------------------------------------------------------------------
    from api.jobs.breadth import compute_market_breadth, store_market_breadth

    def _breadth() -> Dict[str, Any]:
        payload = compute_market_breadth(as_of_date, lookback)
        stored = store_market_breadth(as_of_date, payload) if payload and write_ok else False
        return {
            "breadth_state": payload.get("breadth_state") if payload else None,
            "universe_size": payload.get("universe_size", 0) if payload else 0,
            "stored": stored,
        }

    _maybe_run("market_breadth", _breadth)

    # ------------------------------------------------------------------
    # Stage 2: Sector breadth
    # ------------------------------------------------------------------
    from api.jobs.sector_breadth import compute_sector_breadth, store_sector_breadth

    def _sector_breadth() -> Dict[str, Any]:
        sectors = compute_sector_breadth(as_of_date, lookback)
        stored = store_sector_breadth(as_of_date, sectors) if sectors and write_ok else 0
        expanding = sum(1 for s in sectors if s.get("breadth_state") == "EXPANDING")
        return {
            "sector_count": len(sectors),
            "expanding_count": expanding,
            "stored": stored,
        }

    _maybe_run("sector_breadth", _sector_breadth)

    # ------------------------------------------------------------------
    # Stage 3: IC snapshot
    # ------------------------------------------------------------------
    from api.jobs.ic import compute_ic_snapshot, store_ic_snapshot

    def _ic_snapshot() -> Dict[str, Any]:
        snapshot = compute_ic_snapshot(as_of_date)
        rows_written = store_ic_snapshot(as_of_date, snapshot) if snapshot and write_ok else 0
        return {
            "fields_computed": len(snapshot),
            "rows_written": rows_written,
        }

    _maybe_run("ic_snapshot", _ic_snapshot)

    # ------------------------------------------------------------------
    # Stage 4: Alert scan
    # ------------------------------------------------------------------
    from api.jobs.alerts import run_alert_scan

    def _alerts() -> Dict[str, Any]:
        summary = run_alert_scan(as_of_date)
        return {
            "rules_evaluated":   summary.rules_evaluated,
            "fired":             summary.fired,
            "suppressed":        summary.suppressed,
            "already_fired_today": summary.already_fired_today,
            "webhook_delivered": summary.webhook_delivered,
            "webhook_failed":    summary.webhook_failed,
        }

    _maybe_run("alerts", _alerts)

    # ------------------------------------------------------------------
    # Stage 5: Universe screen
    # ------------------------------------------------------------------
    from api.axiom.screener import screen_universe

    def _screen() -> Dict[str, Any]:
        result = screen_universe(
            as_of_date,
            min_dau=min_dau,
            limit=screen_limit,
        )
        top = result.get("results", [])
        return {
            "total_screened": result.get("total_screened", 0),
            "count": result.get("count", 0),
            "ic_state": result.get("ic_state"),
            "breadth_state": result.get("breadth_state"),
            "top_symbol": top[0]["symbol"] if top else None,
            "top_opportunities": top[:5],
        }

    _maybe_run("screen", _screen)

    # ------------------------------------------------------------------
    # Stage 6: Signal P&L update
    # ------------------------------------------------------------------
    from api.jobs.pnl import compute_signal_pnl, store_signal_pnl

    def _pnl() -> Dict[str, Any]:
        rows = compute_signal_pnl(as_of_date)
        stored = store_signal_pnl(rows) if rows and write_ok else 0
        return {"rows_computed": len(rows), "rows_stored": stored}

    _maybe_run("signal_pnl", _pnl)

    # ------------------------------------------------------------------
    # Stage 7: Provider reliability snapshot
    # ------------------------------------------------------------------
    from api.providers import get_providers_health
    from api.providers.reliability import snapshot_provider_reliability

    def _provider_reliability() -> Dict[str, Any]:
        health = get_providers_health()
        written = snapshot_provider_reliability(health, as_of_date=as_of_date) if write_ok else 0
        degraded = [p.name for p in health.providers if p.status != "ok" and p.enabled]
        return {
            "overall_status": health.status,
            "providers_checked": len(health.providers),
            "degraded": degraded,
            "rows_written": written,
        }

    _maybe_run("provider_reliability", _provider_reliability)

    # ------------------------------------------------------------------
    # Stage 8: Linkage graph refresh (sector peers)
    # ------------------------------------------------------------------
    from api.signals.linkage import graph as linkage_graph

    def _linkage() -> Dict[str, Any]:
        written = linkage_graph.build_from_sector() if write_ok else 0
        return {"links_written": written}

    _maybe_run("linkage_refresh", _linkage)

    # ------------------------------------------------------------------
    # Stage 9: IC calibration snapshot → Kelly hit-rate
    # ------------------------------------------------------------------
    from api.jobs.ic import snapshot_ic_as_calibration

    def _ic_calibration() -> Dict[str, Any]:
        return snapshot_ic_as_calibration(as_of_date, write_ok=write_ok)

    _maybe_run("ic_calibration", _ic_calibration)

    # ------------------------------------------------------------------
    # Assemble digest
    # ------------------------------------------------------------------
    any_error = any(v.get("status") == "error" for v in stages.values())
    all_ok = all(v.get("status") == "ok" for v in stages.values())
    overall_status = "ok" if all_ok else ("partial" if not all_ok else "error")

    finished_at = dt.datetime.utcnow().isoformat() + "Z"
    duration_ms = round((time.monotonic() - t_start) * 1000)
    headline = _build_headline(stages)

    top_opps = stages.get("screen", {}).get("top_opportunities", [])

    logger.info(
        "orchestrator.complete date=%s status=%s duration_ms=%d stages_ok=%d/%d",
        as_of_date,
        overall_status,
        duration_ms,
        sum(1 for v in stages.values() if v.get("status") == "ok"),
        len(stages),
    )

    return {
        "status": overall_status,
        "as_of_date": as_of_date.isoformat(),
        "started_at": started_at,
        "finished_at": finished_at,
        "duration_ms": duration_ms,
        "stages": stages,
        "headline": headline,
        "top_opportunities": top_opps,
    }


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------

class DailyRunRequest(BaseModel):
    as_of_date: Optional[str] = None
    lookback: int = Field(default=252, ge=5, le=2000)
    min_dau: float = Field(default=0.0, ge=0.0, le=100.0)
    screen_limit: int = Field(default=20, ge=1, le=100)
    skip_stages: List[str] = Field(
        default_factory=list,
        description="Stage names to skip, e.g. ['linkage_refresh', 'provider_reliability']",
    )


@router.post("/daily-run")
def daily_run(req: DailyRunRequest) -> Dict[str, Any]:
    try:
        as_of = (
            dt.date.fromisoformat(req.as_of_date)
            if req.as_of_date
            else dt.date.today() - dt.timedelta(days=1)
        )
    except ValueError:
        as_of = dt.date.today() - dt.timedelta(days=1)

    return run_daily_pipeline(
        as_of,
        lookback=req.lookback,
        min_dau=req.min_dau,
        screen_limit=req.screen_limit,
        skip_stages=set(req.skip_stages),
    )
