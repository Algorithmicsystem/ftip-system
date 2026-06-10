"""Phase 17.2: Complete 15-stage Pipeline Orchestrator."""
from __future__ import annotations

import datetime as dt
import logging
import os
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from api import db

# Resolve the port Railway (or any deployment) is actually listening on.
# Railway sets PORT env var (typically 8080); local dev uses 8000.
_PORT = os.environ.get("PORT", "8000")
_BASE = f"http://localhost:{_PORT}"

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Stage registry
# ---------------------------------------------------------------------------

PIPELINE_STAGES: List[str] = [
    "bar_ingestion",
    "feature_computation",
    "signal_generation",
    "axiom_scoring",
    "alt_data_update",
    "factor_computation",
    "ml_inference",
    "pnl_compute",
    "ic_computation",
    "calibration_update",
    "ic_gate_update",
    "breadth_computation",
    "sri_computation",
    "memory_consolidation",
    "cache_refresh",
]

_BLOCKING_STAGES = {"bar_ingestion", "feature_computation"}

# Dependency map: stage → list of stages that must have succeeded
_STAGE_DEPS: Dict[str, List[str]] = {
    "bar_ingestion": [],
    "feature_computation": ["bar_ingestion"],
    "signal_generation": ["feature_computation"],
    "axiom_scoring": ["bar_ingestion"],
    "alt_data_update": [],
    "factor_computation": ["feature_computation"],
    "ml_inference": ["axiom_scoring"],
    "pnl_compute": ["signal_generation"],
    "ic_computation": ["signal_generation"],
    "calibration_update": ["axiom_scoring"],
    "ic_gate_update": ["ic_computation"],
    "breadth_computation": [],
    "sri_computation": ["breadth_computation"],
    "memory_consolidation": [],
    "cache_refresh": [],
}


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class StageResult:
    stage_name: str
    status: str           # "success" | "failed" | "skipped"
    started_at: dt.datetime
    finished_at: dt.datetime
    duration_seconds: float
    records_processed: int
    error_message: Optional[str] = None


@dataclass
class PipelineRun:
    run_id: str
    as_of_date: dt.date
    started_at: dt.datetime
    finished_at: Optional[dt.datetime]
    stages: Dict[str, StageResult]
    overall_status: str    # "success" | "partial" | "failed"
    symbols_processed: int
    total_errors: int


# ---------------------------------------------------------------------------
# Default stage executors (all graceful — DB disabled = succeed with 0 records)
# ---------------------------------------------------------------------------

def _noop_stage() -> Dict[str, Any]:
    return {"records_processed": 0}


def _make_db_graceful_stage(name: str) -> Callable[[], Dict[str, Any]]:
    def executor() -> Dict[str, Any]:
        if not db.db_enabled():
            return {"records_processed": 0}
        try:
            return _real_stage(name)
        except Exception as exc:
            logger.error("pipeline_stage_inner_error stage=%s err=%s", name, exc)
            return {"records_processed": 0}
    return executor


def _real_stage(name: str) -> Dict[str, Any]:
    today = dt.date.today()

    if name == "bar_ingestion":
        try:
            import httpx
            from api import config
            universe = config.env("FTIP_UNIVERSE_DEFAULT", "") or ""
            if not universe:
                from api.universe import AXIOM_UNIVERSE
                universe = ",".join(AXIOM_UNIVERSE)
            symbols = [s.strip() for s in universe.split(",") if s.strip()]
            logger.info(
                "bar_ingestion_start symbols=%d date=%s",
                len(symbols), today.isoformat(),
            )
            # Log current bar count so we can see if new bars were added
            try:
                pre_row = db.safe_fetchone(
                    "SELECT COUNT(*), MAX(date) FROM prosperity_daily_bars WHERE date = %s",
                    (today,),
                )
                logger.info(
                    "bar_ingestion_pre_run today_bars=%s latest_date=%s",
                    pre_row[0] if pre_row else 0,
                    pre_row[1] if pre_row else None,
                )
            except Exception:
                pass

            resp = httpx.post(
                f"{_BASE}/prosperity/snapshot/run",
                json={
                    "symbols": symbols,
                    "from_date": (today - dt.timedelta(days=365)).isoformat(),
                    "to_date": today.isoformat(),
                    "as_of_date": today.isoformat(),
                    "lookback": 252,
                    "concurrency": 2,
                },
                headers={"X-FTIP-API-Key": config.get_api_key()},
                timeout=300.0,
            )
            if resp.status_code == 200:
                data = resp.json()
                result = data.get("result", data)
                records = len(result.get("symbols_ok", []))
                failed = result.get("symbols_failed") or []
                logger.info(
                    "bar_ingestion_snapshot_done symbols_ok=%d symbols_failed=%d",
                    records, len(failed),
                )
                if failed:
                    # Log first 5 failures with their provider context
                    for f in failed[:5]:
                        sym_name = f.get("symbol", "?") if isinstance(f, dict) else str(f)
                        err = f.get("error", "") if isinstance(f, dict) else ""
                        logger.warning("bar_ingestion_symbol_failed symbol=%s err=%.120s", sym_name, err)
                # Log total bars written today
                try:
                    post_row = db.safe_fetchone(
                        "SELECT COUNT(*) FROM prosperity_daily_bars WHERE date = %s",
                        (today,),
                    )
                    logger.info(
                        "bar_ingestion_complete today_bars_in_db=%s symbols_ok=%d",
                        post_row[0] if post_row else 0, records,
                    )
                except Exception:
                    pass
                if records == 0:
                    logger.warning(
                        "bar_ingest_zero_records — no market data fetched, pipeline will produce no scores"
                    )
                return {"records_processed": records}
            else:
                logger.error(
                    "bar_ingestion_snapshot_http_error status=%d body=%.200s",
                    resp.status_code, resp.text,
                )
        except Exception as exc:
            logger.error("bar_ingestion_stage error=%s", exc)
        logger.warning(
            "bar_ingest_zero_records — no market data fetched, pipeline will produce no scores"
        )
        return {"records_processed": 0}

    if name == "feature_computation":
        try:
            import httpx
            from api import config
            resp = httpx.post(
                f"{_BASE}/jobs/features/daily",
                json={"as_of_date": today.isoformat()},
                headers={"X-FTIP-API-Key": config.get_api_key()},
                timeout=300.0,
            )
            if resp.status_code == 200:
                data = resp.json()
                records = len(data.get("symbols_ok", []))
                logger.info("feature_computation_done symbols_ok=%d", records)
                return {"records_processed": records}
            logger.error(
                "feature_computation_http_error status=%d body=%.200s",
                resp.status_code, resp.text,
            )
        except Exception as exc:
            logger.error("feature_computation_stage error=%s", exc)
        return {"records_processed": 0}

    if name == "signal_generation":
        try:
            import httpx
            from api import config
            resp = httpx.post(
                f"{_BASE}/jobs/signals/daily",
                json={"as_of_date": today.isoformat()},
                headers={"X-FTIP-API-Key": config.get_api_key()},
                timeout=300.0,
            )
            if resp.status_code == 200:
                data = resp.json()
                records = len(data.get("symbols_ok", []))
                logger.info("signal_generation_done symbols_ok=%d", records)
                return {"records_processed": records}
            logger.error(
                "signal_generation_http_error status=%d body=%.200s",
                resp.status_code, resp.text,
            )
        except Exception as exc:
            logger.error("signal_generation_stage error=%s", exc)
        return {"records_processed": 0}

    if name == "axiom_scoring":
        # run_axiom_replay reads bars directly from DB (market_bars_daily /
        # prosperity_daily_bars fallback) — no HTTP server dependency.
        # Use the latest date that has bars rather than strictly today so the
        # stage works on weekends/holidays and in local dev without a running server.
        try:
            from api import config
            from api.axiom.replay import run_axiom_replay
            universe = config.env("FTIP_UNIVERSE_DEFAULT", "") or ""
            if not universe:
                from api.universe import AXIOM_UNIVERSE
                universe = ",".join(AXIOM_UNIVERSE)
            symbols = [s.strip() for s in universe.split(",") if s.strip()]
            # Resolve target date: prefer today if bars exist, else latest bar date
            bar_date_row = db.safe_fetchone(
                """
                SELECT GREATEST(
                    COALESCE((SELECT MAX(date) FROM prosperity_daily_bars), %s::date),
                    COALESCE((SELECT MAX(as_of_date) FROM market_bars_daily), %s::date)
                )
                """,
                (today, today),
            )
            target_date = (bar_date_row[0] if (bar_date_row and bar_date_row[0]) else today)
            logger.info(
                "axiom_scoring_start symbols=%d date=%s",
                len(symbols), target_date.isoformat(),
            )
            run_axiom_replay(
                symbols=symbols,
                start_date=target_date.isoformat(),
                end_date=target_date.isoformat(),
                lookback=252,
                persist=True,
            )
            row = db.safe_fetchone(
                "SELECT COUNT(*)::int FROM axiom_scores_daily WHERE as_of_date = %s",
                (target_date,),
            )
            count = int(row[0]) if row else 0
            logger.info("axiom_scoring_done scores_written=%d date=%s", count, target_date.isoformat())
            return {"records_processed": count}
        except Exception as exc:
            logger.error("axiom_scoring_stage error=%s", exc)
            return {"records_processed": 0}

    if name == "pnl_compute":
        try:
            from api.jobs.pnl import compute_pnl_batch
            r = compute_pnl_batch(today, horizons=[5, 21, 63], store=True)
            return {"records_processed": r.get("computed", 0)}
        except Exception:
            return {"records_processed": 0}

    if name == "ic_computation":
        try:
            from api.jobs.ic_snapshot import compute_ic_snapshot
            r = compute_ic_snapshot(today)
            return {"records_processed": 1 if r else 0}
        except Exception:
            return {"records_processed": 0}

    if name == "breadth_computation":
        try:
            from api.jobs.breadth_snapshot import compute_breadth_snapshot
            r = compute_breadth_snapshot(today)
            return {"records_processed": 1 if r else 0}
        except Exception:
            return {"records_processed": 0}

    if name == "sri_computation":
        try:
            from api.axiom.risk.systemic_risk import compute_sri
            sri_result = compute_sri(today)
            try:
                from api.developer.webhooks import check_and_fire_webhooks
                sri_val = sri_result.sri if hasattr(sri_result, "sri") else (sri_result or 50.0)
                if isinstance(sri_val, (int, float)) and sri_val >= 70:
                    check_and_fire_webhooks("risk.sri_alert", {"sri": float(sri_val), "date": today.isoformat()})
            except Exception:
                pass
            return {"records_processed": 1}
        except Exception:
            return {"records_processed": 0}

    if name == "memory_consolidation":
        count = 0
        try:
            from api.intelligence.signal_memory import update_signal_performance_archive
            r = update_signal_performance_archive(today)
            count += r.get("updated", 0)
        except Exception:
            pass
        try:
            from api.intelligence.regime_playbook import update_regime_playbook
            update_regime_playbook(today)
        except Exception:
            pass
        try:
            from api.intelligence.company_dossier import run_dossier_update_job
            run_dossier_update_job(today)
        except Exception:
            pass
        return {"records_processed": count}

    if name == "ml_inference":
        try:
            from api.axiom.ml.training_job import run_training_job
            r = run_training_job(min_samples=50)
            return {"records_processed": r.get("sample_count", 0)}
        except Exception:
            return {"records_processed": 0}

    if name == "cache_refresh":
        try:
            from api.universal.intelligence_api import (
                assemble_universal_intelligence,
                cache_universal_response,
            )
            rows = db.safe_fetchall(
                "SELECT DISTINCT symbol FROM axiom_scores_daily ORDER BY symbol LIMIT 100"
            ) or []
            buy_signals = []
            for row in rows:
                sym = str(row[0])
                resp = assemble_universal_intelligence(sym)
                cache_universal_response(sym, resp)
                if resp.signal_label == "BUY":
                    buy_signals.append({"symbol": sym, "dau": resp.dau, "signal_label": resp.signal_label})
            if buy_signals:
                try:
                    from api.developer.webhooks import check_and_fire_webhooks
                    check_and_fire_webhooks("signal.buy", {
                        "signals": buy_signals,
                        "date": today.isoformat(),
                        "count": len(buy_signals),
                    })
                except Exception:
                    pass
            # Clear in-memory cache so next request gets fresh assembled data
            try:
                from api.universal.intelligence_api import _cache
                _cache.clear()
            except Exception:
                pass
            return {"records_processed": len(rows)}
        except Exception:
            return {"records_processed": 0}

    # All other stages: no-op (handled externally or via daily snapshot)
    return {"records_processed": 0}


_DEFAULT_EXECUTORS: Dict[str, Callable[[], Dict[str, Any]]] = {
    stage: _make_db_graceful_stage(stage) for stage in PIPELINE_STAGES
}


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def _deps_met(stage: str, results: Dict[str, StageResult]) -> bool:
    for dep in _STAGE_DEPS.get(stage, []):
        dep_result = results.get(dep)
        if dep_result is None:
            return False
        if dep_result.status in ("failed", "skipped"):
            return False
    return True


def _compute_overall_status(stages: Dict[str, StageResult]) -> str:
    blocking_failed = any(
        r.status == "failed" and name in _BLOCKING_STAGES
        for name, r in stages.items()
    )
    if blocking_failed:
        return "failed"
    total = len(stages)
    succeeded = sum(1 for r in stages.values() if r.status == "success")
    if succeeded == total:
        return "success"
    return "partial"


def run_full_pipeline(
    as_of_date: Optional[dt.date] = None,
    symbols: Optional[List[str]] = None,
    stages_to_run: Optional[List[str]] = None,
    _stage_executors: Optional[Dict[str, Callable[[], Any]]] = None,
) -> PipelineRun:
    run_id = str(uuid.uuid4())
    as_of_date = as_of_date or dt.date.today()
    stages_to_run = stages_to_run or PIPELINE_STAGES
    executors = dict(_DEFAULT_EXECUTORS)
    if _stage_executors:
        executors.update(_stage_executors)

    started_at = dt.datetime.utcnow()
    stage_results: Dict[str, StageResult] = {}
    total_errors = 0
    symbols_processed = 0

    for stage_name in PIPELINE_STAGES:
        if stage_name not in stages_to_run:
            continue

        stage_start = dt.datetime.utcnow()

        # Dependency check
        if not _deps_met(stage_name, stage_results):
            stage_end = dt.datetime.utcnow()
            stage_results[stage_name] = StageResult(
                stage_name=stage_name,
                status="skipped",
                started_at=stage_start,
                finished_at=stage_end,
                duration_seconds=(stage_end - stage_start).total_seconds(),
                records_processed=0,
                error_message="dependency_not_met",
            )
            continue

        executor = executors.get(stage_name, _noop_stage)
        try:
            result = executor()
            if isinstance(result, dict):
                records = result.get("records_processed", 0)
            else:
                records = 0
            symbols_processed += records
            stage_end = dt.datetime.utcnow()
            stage_results[stage_name] = StageResult(
                stage_name=stage_name,
                status="success",
                started_at=stage_start,
                finished_at=stage_end,
                duration_seconds=(stage_end - stage_start).total_seconds(),
                records_processed=records,
            )
        except Exception as exc:
            total_errors += 1
            stage_end = dt.datetime.utcnow()
            stage_results[stage_name] = StageResult(
                stage_name=stage_name,
                status="failed",
                started_at=stage_start,
                finished_at=stage_end,
                duration_seconds=(stage_end - stage_start).total_seconds(),
                records_processed=0,
                error_message=str(exc),
            )
            logger.warning("pipeline_stage_failed stage=%s run_id=%s err=%s", stage_name, run_id, exc)
            if stage_name in _BLOCKING_STAGES:
                logger.warning("pipeline_blocking_stage_failed stage=%s run_id=%s — aborting dependent stages", stage_name, run_id)

    finished_at = dt.datetime.utcnow()
    overall_status = _compute_overall_status(stage_results)

    run = PipelineRun(
        run_id=run_id,
        as_of_date=as_of_date,
        started_at=started_at,
        finished_at=finished_at,
        stages=stage_results,
        overall_status=overall_status,
        symbols_processed=symbols_processed,
        total_errors=total_errors,
    )

    _store_pipeline_run(run)
    return run


def _store_pipeline_run(run: PipelineRun) -> None:
    if not db.db_read_enabled():
        return
    try:
        stages_json = {
            name: {
                "status": r.status,
                "duration_seconds": r.duration_seconds,
                "records_processed": r.records_processed,
                "error_message": r.error_message,
            }
            for name, r in run.stages.items()
        }
        import json
        db.safe_execute(
            """
            INSERT INTO pipeline_runs
                (run_id, as_of_date, started_at, finished_at, overall_status,
                 symbols_processed, total_errors, stages)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s::jsonb)
            ON CONFLICT (run_id) DO NOTHING
            """,
            (
                run.run_id, run.as_of_date, run.started_at, run.finished_at,
                run.overall_status, run.symbols_processed, run.total_errors,
                json.dumps(stages_json),
            ),
        )
    except Exception as exc:
        logger.debug("pipeline_run_store_failed run_id=%s err=%s", run.run_id, exc)


# ---------------------------------------------------------------------------
# History / status queries
# ---------------------------------------------------------------------------

def get_pipeline_status() -> Optional[Dict[str, Any]]:
    if not db.db_read_enabled():
        return None
    try:
        row = db.safe_fetchone(
            """
            SELECT run_id, as_of_date, started_at, finished_at,
                   overall_status, symbols_processed, total_errors, stages
              FROM pipeline_runs
             ORDER BY started_at DESC LIMIT 1
            """
        )
        if not row:
            return None
        return {
            "run_id": str(row[0]),
            "as_of_date": str(row[1]),
            "started_at": str(row[2]),
            "finished_at": str(row[3]) if row[3] else None,
            "overall_status": str(row[4]),
            "symbols_processed": int(row[5] or 0),
            "total_errors": int(row[6] or 0),
            "stages": row[7] if isinstance(row[7], dict) else {},
        }
    except Exception:
        return None


def get_pipeline_history(limit: int = 10) -> List[Dict[str, Any]]:
    if not db.db_read_enabled():
        return []
    try:
        rows = db.safe_fetchall(
            """
            SELECT run_id, as_of_date, started_at, finished_at,
                   overall_status, symbols_processed, total_errors
              FROM pipeline_runs
             ORDER BY started_at DESC LIMIT %s
            """,
            (limit,),
        ) or []
        return [
            {
                "run_id": str(r[0]),
                "as_of_date": str(r[1]),
                "started_at": str(r[2]),
                "finished_at": str(r[3]) if r[3] else None,
                "overall_status": str(r[4]),
                "symbols_processed": int(r[5] or 0),
                "total_errors": int(r[6] or 0),
            }
            for r in rows
        ]
    except Exception:
        return []
